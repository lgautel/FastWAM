"""FastWAM over OpenPI-compatible WebSocket protocol.

镜像 ``bt/fastwam_eval_policy.py`` 的模型加载 + 反归一化逻辑，
协议侧对齐 ``OldPi/pi/openpi/bt/evaluation/pi05_ws_server.py``。

客户端 obs (msgpack + numpy):
    {
        head_rgb: HxWx3 uint8 / float,
        left_wrist_rgb: HxWx3 uint8 / float,
        right_wrist_rgb: HxWx3 uint8 / float,
        state: (A_real,) float32,
        prompt: str,
    }

服务端响应:
    {
        actions: (T, A_real) float32  (denormalized),
        policy_timing: {infer_ms: float},
        server_timing: {infer_ms: float, prev_total_ms?: float},
    }
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import http
import inspect
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import websockets.asyncio.server as _server
import websockets.frames
from PIL import Image

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)

# msgpack: match the deployment client's custom ndarray wrapper.
try:
    from Websocket import msgpack as msgpack_numpy  # type: ignore
except Exception:
    try:
        from openpi_client import msgpack_numpy  # type: ignore
    except Exception:
        import msgpack
        import msgpack_numpy as _mpn

        class _Packer:
            def pack(self, obj):
                return msgpack.packb(obj, default=_mpn.encode, use_bin_type=True)

        def _unpack(buf):
            return msgpack.unpackb(buf, object_hook=_mpn.decode, raw=False)

        class _Module:
            Packer = _Packer
            unpackb = staticmethod(_unpack)

        msgpack_numpy = _Module()


logger = logging.getLogger(__name__)


# ─── helpers ─────────────────────────────────────────────────


def _configure_logging(log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _normalize_mp(mp: str) -> str:
    key = str(mp).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mp}")
    return key


def _mp_to_dtype(mp: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[_normalize_mp(mp)]


# Per-camera RGB color statistics (per-channel mean/std over 0-255), measured from the
# saved client frames in logs/ws_client_frame{,_trainsets}/ (see tmp/color_stats.py).
#   train_* : training-domain distribution (dataset frames replayed through this server) -- the target.
#   real_*  : confirmed real-robot frames -- used as the fixed source baseline by mode="fixed".
# The real camera system is systematically desaturated / cooler (less red+yellow) than training,
# so we Reinhard-match raw frames back to the training distribution before inference.
_COLOR_MATCH_STATS: dict[str, dict[str, tuple[float, float, float]]] = {
    "head_rgb": {
        "train_mean": (127.09, 72.47, 52.96),
        "train_std": (37.51, 36.24, 46.06),
        "real_mean": (129.15, 113.76, 103.17),
        "real_std": (35.51, 34.01, 41.81),
    },
    "left_wrist_rgb": {
        "train_mean": (106.93, 102.21, 100.36),
        "train_std": (52.48, 44.92, 47.98),
        "real_mean": (105.75, 105.79, 103.57),
        "real_std": (55.08, 55.27, 57.11),
    },
    "right_wrist_rgb": {
        "train_mean": (100.60, 92.02, 89.89),
        "train_std": (57.73, 53.05, 55.53),
        "real_mean": (106.73, 105.30, 103.92),
        "real_std": (61.40, 61.63, 64.40),
    },
}

_COLOR_MATCH_MODES = ("off", "fixed", "adaptive")


def _normalize_color_match(mode: str | None) -> str:
    key = str(mode if mode is not None else "off").strip().lower()
    if key not in _COLOR_MATCH_MODES:
        raise ValueError(
            f"Unsupported color_match mode: {mode!r}; want one of {_COLOR_MATCH_MODES}"
        )
    return key


def _normalize_color_match_strength(strength: float) -> float:
    value = float(strength)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"color_match_strength must be in [0, 1], got {strength}")
    return value


def _compose_cfg(task: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def _to_chw_uint8_device(img: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """HWC/CHW image -> CHW uint8 tensor on target device."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"image must be HxWx3, got {arr.shape}")
    if arr.shape[2] != 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        raise ValueError(f"image must have 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.to(device=device, non_blocking=True).permute(2, 0, 1)


def _chw_to_float01(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t.to(dtype=torch.float32) / 255.0
    return t.to(dtype=torch.float32)


_TV_INTERP_MODES = {
    "bilinear": TF.InterpolationMode.BILINEAR,
    "bicubic": TF.InterpolationMode.BICUBIC,
    "nearest": TF.InterpolationMode.NEAREST,
}


def _resize_chw(
    image: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    image = _chw_to_float01(image)
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    if mode not in _TV_INTERP_MODES:
        raise ValueError(f"unsupported resize mode={mode}")
    return TF.resize(
        image,
        size=list(size),
        interpolation=_TV_INTERP_MODES[mode],
        antialias=True,
    )


def _resize_smallest_side_chw(
    image: torch.Tensor,
    *,
    img_h: int,
    img_w: int,
    mode: str = "bicubic",
) -> torch.Tensor:
    orig_h, orig_w = image.shape[-2:]
    scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
    target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))
    return _resize_chw(image, target_size, mode=mode)


def _center_crop_chw(image: torch.Tensor, *, img_h: int, img_w: int) -> torch.Tensor:
    h, w = image.shape[-2:]
    if h < img_h or w < img_w:
        raise ValueError(f"cannot center crop image {(h, w)} to {(img_h, img_w)}")
    top = (h - img_h) // 2
    left = (w - img_w) // 2
    return image[..., top : top + img_h, left : left + img_w]


def _as_hwc_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"image must be HxWx3, got {arr.shape}")
    if arr.shape[2] != 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        raise ValueError(f"image must have 3 channels, got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _model_input_to_hwc_uint8(model_input: torch.Tensor) -> np.ndarray:
    t = model_input.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(f"model_input must be [1, 3, H, W] or [3, H, W], got {tuple(t.shape)}")
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


class _ClientFrameSaver:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir).expanduser()
        self._lock = threading.Lock()

    @staticmethod
    def _frame_stamp(ref_time: Any) -> str:
        if ref_time is None:
            return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if isinstance(ref_time, np.ndarray):
            ref_time = ref_time.reshape(-1)[0].item() if ref_time.size else None
        if ref_time is None:
            return dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Keep it filesystem-safe (strip path separators / spaces).
        return str(ref_time).replace("/", "-").replace(" ", "_").replace(":", "-")

    def save(
        self,
        obs: dict[str, Any],
        model_input: torch.Tensor,
        *,
        request_id: int | None = None,
    ) -> Path | None:
        with self._lock:
            stamp = self._frame_stamp(obs.get("ref_time"))
            frame_dir = self._output_dir / stamp
            frame_dir.mkdir(parents=True, exist_ok=True)
            for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                Image.fromarray(_as_hwc_uint8(obs[key])).save(frame_dir / f"{key}.png")
            Image.fromarray(_model_input_to_hwc_uint8(model_input)).save(
                frame_dir / "model_input.png"
            )
            return frame_dir


# Layout of the flat 23-dim state/action vector (see send_dataset_to_ws.py):
#   left_arm(7), right_arm(7), left_gripper(1), right_gripper(1), torso(4), chassis(3)
_STATE_ACTION_LAYOUT: tuple[tuple[str, int], ...] = (
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_gripper", 1),
    ("right_gripper", 1),
    ("torso", 4),
    ("chassis", 3),
)


def _flat_component_labels(prefix: str, expected_dim: int | None = None) -> list[str]:
    labels: list[str] = []
    for name, n in _STATE_ACTION_LAYOUT:
        if n == 1:
            labels.append(f"{prefix}_{name}")
        else:
            labels.extend(f"{prefix}_{name}_{i}" for i in range(n))
    if expected_dim is not None and len(labels) != expected_dim:
        # Layout doesn't match the configured dim; fall back to plain indices.
        return [f"{prefix}_{i}" for i in range(expected_dim)]
    return labels


def _write_frame_state_action_csv(
    csv_path: str | Path,
    *,
    state: np.ndarray,
    actions: np.ndarray,
) -> None:
    """Save this frame's obs state and the inferred action chunk next to its images."""
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    actions_np = np.asarray(actions, dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np[None, :]
    if actions_np.ndim != 2:
        raise ValueError(f"actions must be [T, A], got {actions_np.shape}")

    state_labels = _flat_component_labels("state", state_np.shape[0])
    action_labels = _flat_component_labels("action", actions_np.shape[1])
    header = ["chunk_offset", *state_labels, *action_labels]

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for chunk_offset, action in enumerate(actions_np):
            writer.writerow(
                [
                    int(chunk_offset),
                    *[float(v) for v in state_np],
                    *[float(v) for v in action],
                ]
            )


class _StateActionCsvLogger:
    def __init__(
        self,
        csv_path: str | Path,
        *,
        state_dim: int,
        action_dim: int,
        timestamp_fn=None,
    ) -> None:
        self._csv_path = Path(csv_path).expanduser()
        self._state_dim = int(state_dim)
        self._action_dim = int(action_dim)
        self._state_labels = _flat_component_labels("state", self._state_dim)
        self._action_labels = _flat_component_labels("action", self._action_dim)
        self._timestamp_fn = timestamp_fn or self._utc_timestamp
        self._lock = threading.Lock()
        # Start a fresh CSV on each server launch instead of appending across restarts.
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", encoding="utf-8", newline=""):
            pass

    @staticmethod
    def _utc_timestamp() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")

    def append(
        self,
        *,
        request_id: int,
        prompt: str | None,
        state: np.ndarray,
        actions: np.ndarray,
        infer_ms: float,
        total_ms: float,
    ) -> None:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim != 2:
            raise ValueError(f"actions must be [T, A], got {actions_np.shape}")
        if state_np.shape[0] != self._state_dim:
            raise ValueError(f"state dim {state_np.shape[0]} != expected {self._state_dim}")
        if actions_np.shape[1] != self._action_dim:
            raise ValueError(f"action dim {actions_np.shape[1]} != expected {self._action_dim}")

        timestamp = self._timestamp_fn()
        header = (
            ["timestamp", "request_id", "chunk_offset", "prompt", "infer_ms", "total_ms"]
            + self._state_labels
            + self._action_labels
        )
        with self._lock:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
            with open(self._csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                for chunk_offset, action in enumerate(actions_np):
                    writer.writerow(
                        [
                            timestamp,
                            int(request_id),
                            int(chunk_offset),
                            prompt or "",
                            float(infer_ms),
                            float(total_ms),
                            *[float(v) for v in state_np],
                            *[float(v) for v in action],
                        ]
                    )


def _describe_value(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    return type(v).__name__


def _describe_obs(obs: dict[str, Any]) -> str:
    parts = []
    for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb", "state", "prompt"):
        if key in obs:
            parts.append(f"{key}={_describe_value(obs[key])}")
    extras = sorted(set(obs) - {"head_rgb", "left_wrist_rgb", "right_wrist_rgb", "state", "prompt"})
    if extras:
        parts.append(f"extra_keys={extras}")
    return ", ".join(parts)


class _TwoClientObsMixer:
    """Cache full observations from two roles and splice image/state fields."""

    _IMAGE_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")

    def __init__(self, *, image_role: str = "image", state_role: str = "state") -> None:
        self._image_role = image_role
        self._state_role = state_role
        self._image_obs: dict[str, Any] | None = None
        self._state_obs: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def roles(self) -> tuple[str, str]:
        return self._image_role, self._state_role

    def update(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        role = str(obs.get("role", ""))
        if role not in self.roles:
            raise ValueError(
                f"obs role must be {self._image_role!r} or {self._state_role!r}, got {role!r}"
            )

        with self._lock:
            if role == self._image_role:
                self._image_obs = obs
            else:
                self._state_obs = obs

            if self._image_obs is None or self._state_obs is None:
                return None

            image_obs = self._image_obs
            state_obs = self._state_obs
            mixed = dict(image_obs)
            for key in self._IMAGE_KEYS:
                mixed[key] = image_obs[key]
            mixed["state"] = state_obs["state"]
            mixed["prompt"] = image_obs.get("prompt") or state_obs.get("prompt")
            if "ref_time" in image_obs:
                mixed["ref_time"] = image_obs["ref_time"]
            elif "ref_time" in state_obs:
                mixed["ref_time"] = state_obs["ref_time"]
            mixed["role"] = "mixed"
            mixed["mixed_sources"] = {
                "image_role": self._image_role,
                "state_role": self._state_role,
                "trigger_role": role,
            }
            return mixed


# ─── Adapter ─────────────────────────────────────────────────


class FastWAMAdapter:
    """Loads a FastWAM checkpoint + processor, exposes ``predict(obs) -> {actions,...}``."""

    def __init__(
        self,
        *,
        task: str,
        checkpoint: str | Path,
        dataset_stats: str | Path,
        action_horizon: int,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = 42,
        rand_device: str = "cpu",
        tiled: bool = False,
        device: str | None = None,
        mixed_precision: str | None = None,
        config_override: list[str] | None = None,
        default_prompt: str | None = None,
        load_text_encoder: bool = True,
        warmup: bool = True,
        save_client_frame_dir: str | Path | None = None,
        log_state_action_csv: str | Path | None = None,
        color_match: str = "off",
        color_match_strength: float = 1.0,
    ) -> None:
        self._task = task
        self._checkpoint = Path(checkpoint).expanduser().resolve()
        self._stats_path = Path(dataset_stats).expanduser().resolve()
        if not self._checkpoint.exists():
            raise FileNotFoundError(f"checkpoint missing: {self._checkpoint}")
        if not self._stats_path.exists():
            raise FileNotFoundError(f"dataset_stats missing: {self._stats_path}")

        cfg = _compose_cfg(task, list(config_override or []))
        self._cfg = cfg

        mp = mixed_precision or str(cfg.get("mixed_precision", "bf16"))
        self._dtype = _mp_to_dtype(mp)

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(dev).startswith("cuda") and not torch.cuda.is_available():
            logger.info("[fastwam_ws] CUDA unavailable, falling back to cpu")
            dev = "cpu"
        self._device = dev

        # ---- color match (raw real frames -> training distribution) ----
        self._color_match = _normalize_color_match(color_match)
        self._color_match_strength = _normalize_color_match_strength(color_match_strength)
        self._color_match_tensors: dict[str, tuple[torch.Tensor, ...]] = {}
        if self._color_match != "off":
            for cam_key, s in _COLOR_MATCH_STATS.items():
                def _v(name: str) -> torch.Tensor:
                    return torch.tensor(
                        s[name], dtype=torch.float32, device=self._device
                    ).view(3, 1, 1)

                self._color_match_tensors[cam_key] = (
                    _v("train_mean"),
                    _v("train_std"),
                    _v("real_mean"),
                    _v("real_std"),
                )
            logger.info(
                "[fastwam_ws] color match enabled: mode=%s strength=%.3f cams=%s",
                self._color_match,
                self._color_match_strength,
                list(self._color_match_tensors),
            )

        # ---- processor (normalizer + merger) ----
        proc_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
        processor = instantiate(proc_cfg)
        stats = load_dataset_stats_from_json(str(self._stats_path))
        processor.set_normalizer_from_stats(stats)
        processor.eval()
        self._processor = processor

        # ---- model ----
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = bool(load_text_encoder)
        logger.info(
            "[fastwam_ws] instantiating model (load_text_encoder=%s) dtype=%s device=%s",
            model_cfg.load_text_encoder,
            self._dtype,
            dev,
        )
        model = instantiate(model_cfg, model_dtype=self._dtype, device=str(dev))
        logger.info("[fastwam_ws] loading checkpoint: %s", self._checkpoint)
        model.load_checkpoint(str(self._checkpoint))
        model.eval()
        self._model = model

        # ---- geometry ----
        self._num_video_frames = int(cfg.data.train.num_frames)
        self._action_horizon = int(action_horizon)
        self._num_inference_steps = int(num_inference_steps)
        self._sigma_shift = sigma_shift
        self._seed = seed
        self._rand_device = rand_device
        self._tiled = tiled

        video_size = list(cfg.data.train.video_size)
        self._video_h, self._video_w = int(video_size[0]), int(video_size[1])
        self._concat_mode = str(cfg.data.train.get("concat_multi_camera", "robotwin"))

        self._image_metas = list(cfg.data.train.shape_meta.images)
        self._action_metas = list(cfg.data.train.shape_meta.action)
        self._state_metas = list(cfg.data.train.shape_meta.state)
        self._proprio_dim = int(cfg.data.train.processor.proprio_output_dim)
        self._action_dim = int(cfg.data.train.processor.action_output_dim)
        self._real_action_dim = int(sum(int(m["raw_shape"]) for m in self._action_metas))
        self._real_state_dim = int(sum(int(m["raw_shape"]) for m in self._state_metas))

        self._infer_supports_nvf = (
            "num_video_frames" in inspect.signature(model.infer_action).parameters
        )

        self._default_prompt = default_prompt
        self._prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._client_frame_saver = (
            _ClientFrameSaver(save_client_frame_dir)
            if save_client_frame_dir is not None
            else None
        )
        self._state_action_logger = (
            _StateActionCsvLogger(
                log_state_action_csv,
                state_dim=self._real_state_dim,
                action_dim=self._real_action_dim,
            )
            if log_state_action_csv is not None
            else None
        )

        if warmup:
            try:
                self._warmup()
            except Exception as e:  # noqa: BLE001
                logger.info("[fastwam_ws] warmup skipped: %s: %s", type(e).__name__, e)

    # ── preprocessing ─────────────────────────────────────────

    def _color_match_chw(self, t_uint8: torch.Tensor, cam_key: str) -> torch.Tensor:
        """Reinhard per-channel match of a CHW uint8 RGB frame to the training distribution.

        mode="fixed":    source stats are the constant real-robot baseline (temporally stable).
        mode="adaptive": source stats are computed from this frame itself (self-calibrating).
        Returns a CHW uint8 tensor; a no-op when color match is off or stats are missing.
        """
        stats = self._color_match_tensors.get(cam_key)
        if self._color_match == "off" or stats is None:
            return t_uint8
        tgt_mean, tgt_std, real_mean, real_std = stats
        x = t_uint8.to(dtype=torch.float32)
        if self._color_match == "adaptive":
            src_mean = x.mean(dim=(1, 2), keepdim=True)
            src_std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        else:  # "fixed"
            src_mean = real_mean
            src_std = real_std
        y = (x - src_mean) / src_std * tgt_std + tgt_mean
        if self._color_match_strength < 1.0:
            y = x.lerp(y, self._color_match_strength)
        return y.clamp_(0.0, 255.0).to(dtype=torch.uint8)

    def _prep_image(
        self,
        head: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
    ) -> torch.Tensor:
        """3 raw RGBs -> [1, 3, H, W] in [-1, 1] (matches RobotVideoDataset.robotwin layout)."""
        head_t = self._color_match_chw(_to_chw_uint8_device(head, self._device), "head_rgb")
        left_t = self._color_match_chw(_to_chw_uint8_device(left, self._device), "left_wrist_rgb")
        right_t = self._color_match_chw(_to_chw_uint8_device(right, self._device), "right_wrist_rgb")

        if self._concat_mode == "robotwin":
            cam_top = _resize_chw(head_t, (256, 320), mode="bilinear")
            cam_left = _resize_chw(left_t, (128, 160), mode="bilinear")
            cam_right = _resize_chw(right_t, (128, 160), mode="bilinear")
            bottom = torch.cat([cam_left, cam_right], dim=-1)        # [3, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)              # [3, 384, 320]
        elif self._concat_mode == "horizontal":
            video = torch.cat(
                [_chw_to_float01(head_t), _chw_to_float01(left_t), _chw_to_float01(right_t)],
                dim=-1,
            )
        elif self._concat_mode == "vertical":
            video = torch.cat(
                [_chw_to_float01(head_t), _chw_to_float01(left_t), _chw_to_float01(right_t)],
                dim=-2,
            )
        else:
            raise ValueError(f"unknown concat_multi_camera={self._concat_mode}")

        video = _resize_smallest_side_chw(
            video,
            img_h=self._video_h,
            img_w=self._video_w,
            mode="bicubic",
        )
        video = _center_crop_chw(video, img_h=self._video_h, img_w=self._video_w)
        video = video.clamp(0.0, 1.0)
        video = (video - 0.5) / 0.5  # -> [-1, 1]
        return video.unsqueeze(0).to(dtype=self._dtype)

    def _prep_proprio(self, state: np.ndarray) -> torch.Tensor:
        """raw state (A_real,) -> [1, proprio_output_dim] normalized + merged."""
        state_t = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        if state_t.shape[-1] != self._real_state_dim:
            raise ValueError(
                f"state dim {state_t.shape[-1]} != expected real_state_dim {self._real_state_dim}"
            )

        state_dict: dict[str, torch.Tensor] = {}
        offset = 0
        for meta in self._state_metas:
            raw = int(meta["raw_shape"])
            state_dict[meta["key"]] = state_t[:, offset : offset + raw].clone()  # [1, raw]
            offset += raw

        # 占位 action，确保 normalizer/merger 在 forward 时不会因为缺 key 报错。
        # 注意 ConcatLeftAlign.forward 要求 2D [T, raw]，所以这里用单步占位即可。
        action_dict = {
            meta["key"]: torch.zeros(1, int(meta["raw_shape"]), dtype=torch.float32)
            for meta in self._action_metas
        }

        batch: dict[str, Any] = {"state": state_dict, "action": action_dict}
        if self._processor.action_state_transforms is not None:
            for trans in self._processor.action_state_transforms:
                batch = trans.forward(batch)
        batch = self._processor.normalizer.forward(batch)
        batch = self._processor.action_state_merger.forward(batch)

        proprio = batch["state"]
        if proprio.ndim != 2 or proprio.shape[-1] != self._proprio_dim:
            raise ValueError(
                f"unexpected merged proprio shape {tuple(proprio.shape)}, want [1, {self._proprio_dim}]"
            )
        return proprio.to(device=self._device, dtype=self._dtype)

    def _denorm_action(
        self,
        action_norm: torch.Tensor,
        proprio_norm: torch.Tensor,
    ) -> np.ndarray:
        """[T, A_pad] (normalized) -> [T, A_real] denormalized numpy."""
        if action_norm.ndim == 2:
            action_norm = action_norm.unsqueeze(0)
        T = int(action_norm.shape[1])

        state_b = proprio_norm.detach().to(device="cpu", dtype=torch.float32)
        if state_b.ndim == 2:
            state_b = state_b.unsqueeze(0)
        if state_b.shape[1] < T:
            state_b = state_b.expand(-1, T, -1).contiguous()
        else:
            state_b = state_b[:, :T]

        batch = {
            "action": action_norm.detach().to(device="cpu", dtype=torch.float32),
            "state": state_b,
        }
        batch = self._processor.action_state_merger.backward(batch)
        batch = self._processor.normalizer.backward(batch)
        if self._processor.action_state_transforms is not None:
            for trans in reversed(self._processor.action_state_transforms):
                batch = trans.backward(batch)
        parts = [batch["action"][m["key"]] for m in self._action_metas]
        flat = torch.cat(parts, dim=-1)  # [1, T, A_real]
        return flat[0].detach().cpu().numpy().astype(np.float32)

    # ── inference ─────────────────────────────────────────────

    def _encode_prompt(self, prompt: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

        actual = prompt if prompt else self._default_prompt
        if not actual:
            raise ValueError("no `prompt` in obs and no --default-prompt configured")
        formatted = DEFAULT_PROMPT.format(task=actual)
        cached = self._prompt_cache.get(formatted)
        if cached is not None:
            return cached  # [1, L, D] / [1, L]
        ctx, mask = self._model.encode_prompt(formatted)
        self._prompt_cache[formatted] = (ctx, mask)
        return ctx, mask  # [1, L, D] / [1, L]

    def predict(
        self,
        obs: dict[str, Any],
        *,
        save_client_frame: bool = True,
        request_id: int | None = None,
    ) -> dict[str, Any]:
        for k in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb", "state"):
            if k not in obs:
                raise KeyError(f"obs missing required key: {k}")
        prompt = obs.get("prompt", None) or self._default_prompt
        logger.info("[fastwam_ws] infer begin: %s", _describe_obs(obs))

        t_total = time.monotonic()
        saved_dir: Path | None = None
        with torch.no_grad():
            input_image = self._prep_image(
                obs["head_rgb"], obs["left_wrist_rgb"], obs["right_wrist_rgb"]
            )
            if save_client_frame and self._client_frame_saver is not None:
                saved_dir = self._client_frame_saver.save(
                    obs, input_image, request_id=request_id
                )
                if saved_dir is not None:
                    logger.info("[fastwam_ws] saved client frame: %s", saved_dir)
            proprio_norm = self._prep_proprio(
                np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            )
            context, context_mask = self._encode_prompt(prompt)

            infer_kwargs: dict[str, Any] = {
                "prompt": None,
                "input_image": input_image,
                "action_horizon": self._action_horizon,
                "proprio": proprio_norm,
                "context": context,
                "context_mask": context_mask,
                "num_inference_steps": self._num_inference_steps,
                "sigma_shift": self._sigma_shift,
                "seed": self._seed,
                "rand_device": self._rand_device,
                "tiled": self._tiled,
            }
            if self._infer_supports_nvf:
                infer_kwargs["num_video_frames"] = self._num_video_frames

            t_inf = time.monotonic()
            out = self._model.infer_action(**infer_kwargs)
            infer_ms = (time.monotonic() - t_inf) * 1000.0

            actions_np = self._denorm_action(out["action"], proprio_norm)

        total_ms = (time.monotonic() - t_total) * 1000.0
        if saved_dir is not None:
            _write_frame_state_action_csv(
                saved_dir / "state_action.csv",
                state=np.asarray(obs["state"], dtype=np.float32).reshape(-1),
                actions=actions_np,
            )
        if self._state_action_logger is not None and request_id is not None:
            self._state_action_logger.append(
                request_id=request_id,
                prompt=prompt,
                state=np.asarray(obs["state"], dtype=np.float32).reshape(-1),
                actions=actions_np,
                infer_ms=infer_ms,
                total_ms=total_ms,
            )
        logger.info(
            "[fastwam_ws] infer done: actions_shape=%s model_infer_ms=%.1f total_ms=%.1f",
            actions_np.shape,
            infer_ms,
            total_ms,
        )
        return {
            "actions": actions_np,
            "policy_timing": {"infer_ms": infer_ms},
        }

    def _warmup(self) -> None:
        logger.info("[fastwam_ws] warmup ...")
        h, w = 480, 640
        dummy = {
            "head_rgb": np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            "left_wrist_rgb": np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            "right_wrist_rgb": np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            "state": np.zeros(self._real_state_dim, dtype=np.float32),
            "prompt": self._default_prompt or "do something",
        }
        t0 = time.monotonic()
        self.predict(dummy, save_client_frame=False)
        logger.info("[fastwam_ws] warmup done in %.1f ms", (time.monotonic() - t0) * 1000.0)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "fastwam",
            "version": 1,
            "task": self._task,
            "checkpoint": str(self._checkpoint),
            "action_horizon": self._action_horizon,
            "num_inference_steps": self._num_inference_steps,
            "real_action_dim": self._real_action_dim,
            "real_state_dim": self._real_state_dim,
            "video_size": [self._video_h, self._video_w],
            "concat_multi_camera": self._concat_mode,
        }


# ─── WebSocket Server ─────────────────────────────────────────


class FastWAMPolicyServer:
    def __init__(
        self,
        adapter: FastWAMAdapter,
        host: str,
        port: int,
        *,
        mix_two_client_obs: bool = False,
        image_role: str = "image",
        state_role: str = "state",
    ) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        self._obs_mixer = (
            _TwoClientObsMixer(image_role=image_role, state_role=state_role)
            if mix_two_client_obs
            else None
        )
        self._metadata = adapter.metadata
        if self._obs_mixer is not None:
            self._metadata = dict(self._metadata)
            self._metadata["mix_two_client_obs"] = {
                "image_role": image_role,
                "state_role": state_role,
            }

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            health_host = "127.0.0.1" if self._host in ("0.0.0.0", "::", "[::]") else self._host
            logger.info(
                "[fastwam_ws] listening ws://%s:%s  healthz: curl -s http://%s:%s/healthz",
                self._host,
                self._port,
                health_host,
                self._port,
            )
            await server.serve_forever()

    async def _handler(self, ws: _server.ServerConnection) -> None:
        logger.info("[fastwam_ws] client connected: %s", ws.remote_address)
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        logger.info("[fastwam_ws] metadata sent: %s", self._metadata)

        prev_total: float | None = None
        req_id = 0
        while True:
            try:
                t_start = time.monotonic()
                obs = msgpack_numpy.unpackb(await ws.recv())
                req_id += 1
                logger.info("[fastwam_ws] request #%s received", req_id)
                obs_for_predict = obs
                if self._obs_mixer is not None and "role" in obs:
                    obs_for_predict = self._obs_mixer.update(obs)
                    if obs_for_predict is None:
                        await ws.send(
                            packer.pack(
                                {
                                    "status": "waiting_for_pair",
                                    "server_timing": {"infer_ms": 0.0},
                                }
                            )
                        )
                        logger.info(
                            "[fastwam_ws] request #%s cached role=%s; waiting for both roles",
                            req_id,
                            obs.get("role"),
                        )
                        continue
                    logger.info(
                        "[fastwam_ws] request #%s using mixed obs from roles=%s",
                        req_id,
                        self._obs_mixer.roles,
                    )

                t_inf = time.monotonic()
                action = await asyncio.to_thread(self._adapter.predict, obs_for_predict, request_id=req_id)
                infer_ms = (time.monotonic() - t_inf) * 1000.0
                action.setdefault("server_timing", {})["infer_ms"] = infer_ms
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1000.0

                await ws.send(packer.pack(action))
                prev_total = time.monotonic() - t_start
                logger.info(
                    "[fastwam_ws] response #%s server_infer_ms=%.1f total_ms=%.1f",
                    req_id,
                    infer_ms,
                    prev_total * 1000.0,
                )
            except __import__("websockets").ConnectionClosed:
                logger.info("[fastwam_ws] client disconnected: %s", ws.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error("[fastwam_ws] error:\n%s", tb)
                try:
                    await ws.send(tb)
                except Exception:
                    pass
                await ws.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(conn: _server.ServerConnection, req: _server.Request):
    if req.path == "/healthz":
        return conn.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ─── entry ────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="FastWAM websocket policy server (OpenPI-compatible).")
    ap.add_argument("--task", default="r1_pro_chassis_uncond_3cam_384_1e-4",
                    help="Hydra task config name under configs/task.")
    ap.add_argument("--checkpoint", required=True, help="FastWAM checkpoint .pt path.")
    ap.add_argument("--dataset-stats", required=True, help="Path to training dataset_stats.json.")
    ap.add_argument("--config-override", action="append", default=[],
                    help="Extra Hydra override(s), e.g. data.train.text_embedding_cache_dir=/x.")
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--sigma-shift", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rand-device", type=str, default="cpu")
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    ap.add_argument("--default-prompt", type=str, default=None)
    ap.add_argument("--save-client-frame-dir", type=str, default=None,
                    help="If set, save every client frame's raw cameras and model input PNGs here, "
                         "one timestamped subfolder per request.")
    ap.add_argument("--log-state-action-csv", type=str, default=None,
                    help="If set, append each request's raw state and denormalized action chunk to this CSV.")
    ap.add_argument("--log-file", type=str, default="./logs/ws_infer.log",
                    help="Write logger.info output to this file as well as stdout. Use an empty string to disable.")
    ap.add_argument("--no-text-encoder", action="store_true",
                    help="Skip loading the T5 text encoder (only useful if you patch in cached embeddings).")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mix-two-client-obs", action="store_true",
                    help="Treat role-tagged full obs messages from two clients as image/state sources.")
    ap.add_argument("--image-role", type=str, default="image",
                    help="Role value whose obs supplies head/left/right images in --mix-two-client-obs mode.")
    ap.add_argument("--state-role", type=str, default="state",
                    help="Role value whose obs supplies proprio state in --mix-two-client-obs mode.")
    ap.add_argument("--color-match", type=str, default="off",
                    choices=list(_COLOR_MATCH_MODES),
                    help="Color-match raw camera frames to the training distribution before inference. "
                         "'fixed' applies a constant real->train transform (temporally stable, "
                         "recommended for real-robot deploy); 'adaptive' normalizes each frame's own "
                         "stats; 'off' (default) leaves frames untouched -- keep it off when replaying "
                         "dataset frames through this server.")
    ap.add_argument("--color-match-strength", type=float, default=1.0,
                    help="Blend strength for --color-match in [0, 1]. 0 leaves raw frames unchanged; "
                         "1 applies the full color match; values like 0.8 reduce over-saturation.")
    args = ap.parse_args()

    _configure_logging(args.log_file)
    logger.info(
        "[fastwam_ws] start task=%s ckpt=%s ws://%s:%s device=%s",
        args.task,
        args.checkpoint,
        args.host,
        args.port,
        args.device,
    )

    adapter = FastWAMAdapter(
        task=args.task,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        rand_device=args.rand_device,
        tiled=args.tiled,
        device=args.device,
        mixed_precision=args.mixed_precision,
        config_override=args.config_override,
        default_prompt=args.default_prompt,
        load_text_encoder=not args.no_text_encoder,
        warmup=not args.no_warmup,
        save_client_frame_dir=args.save_client_frame_dir,
        log_state_action_csv=args.log_state_action_csv,
        color_match=args.color_match,
        color_match_strength=args.color_match_strength,
    )

    FastWAMPolicyServer(
        adapter,
        host=args.host,
        port=args.port,
        mix_two_client_obs=args.mix_two_client_obs,
        image_role=args.image_role,
        state_role=args.state_role,
    ).serve_forever()


if __name__ == "__main__":
    main()