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
from fastwam.datasets.dataset_utils import (  # noqa: E402
    CenterCrop,
    ResizeSmallestSideAspectPreserving,
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


def _normalize_mp(mp: str) -> str:
    key = str(mp).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mp}")
    return key


def _mp_to_dtype(mp: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[_normalize_mp(mp)]


def _compose_cfg(task: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def _to_chw_float(img: np.ndarray) -> torch.Tensor:
    """HWC uint8/float -> CHW float in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"image must be HxWx3, got {arr.shape}")
    if arr.shape[2] != 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] != 3:
        raise ValueError(f"image must have 3 channels, got {arr.shape}")
    if arr.dtype == np.uint8:
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    else:
        t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)
        if float(t.max()) > 1.5:
            t = t / 255.0
    return t


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
        self._saved = False
        self._lock = threading.Lock()

    def save_once(self, obs: dict[str, Any], model_input: torch.Tensor) -> Path | None:
        with self._lock:
            if self._saved:
                return None
            self._output_dir.mkdir(parents=True, exist_ok=True)
            for key in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
                Image.fromarray(_as_hwc_uint8(obs[key])).save(self._output_dir / f"{key}.png")
            Image.fromarray(_model_input_to_hwc_uint8(model_input)).save(
                self._output_dir / "model_input.png"
            )
            self._saved = True
            return self._output_dir


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
        self._timestamp_fn = timestamp_fn or self._utc_timestamp
        self._lock = threading.Lock()

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
            + [f"state_{i}" for i in range(self._state_dim)]
            + [f"action_{i}" for i in range(self._action_dim)]
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
            print("[fastwam_ws] CUDA unavailable, falling back to cpu", flush=True)
            dev = "cpu"
        self._device = dev

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
        print(
            f"[fastwam_ws] instantiating model (load_text_encoder={model_cfg.load_text_encoder}) "
            f"dtype={self._dtype} device={dev}",
            flush=True,
        )
        model = instantiate(model_cfg, model_dtype=self._dtype, device=str(dev))
        print(f"[fastwam_ws] loading checkpoint: {self._checkpoint}", flush=True)
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
        self._resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self._video_w, "img_h": self._video_h},
        )
        self._crop_transform = CenterCrop(
            args={"img_w": self._video_w, "img_h": self._video_h},
        )

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
                print(f"[fastwam_ws] warmup skipped: {type(e).__name__}: {e}", flush=True)

    # ── preprocessing ─────────────────────────────────────────

    def _prep_image(
        self,
        head: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
    ) -> torch.Tensor:
        """3 raw RGBs -> [1, 3, H, W] in [-1, 1] (matches RobotVideoDataset.robotwin layout)."""
        video = torch.stack(
            [_to_chw_float(head), _to_chw_float(left), _to_chw_float(right)],
            dim=0,
        )  # [num_cameras, C, H, W]

        if self._concat_mode == "robotwin":
            cam_top = TF.resize(
                video[0],
                size=[256, 320],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = TF.resize(
                video[1],
                size=[128, 160],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = TF.resize(
                video[2],
                size=[128, 160],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)        # [3, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)              # [3, 384, 320]
        elif self._concat_mode == "horizontal":
            video = torch.cat([video[i] for i in range(video.shape[0])], dim=-1)
        elif self._concat_mode == "vertical":
            video = torch.cat([video[i] for i in range(video.shape[0])], dim=-2)
        else:
            raise ValueError(f"unknown concat_multi_camera={self._concat_mode}")

        video = self._resize_transform(video)
        video = self._crop_transform(video)
        video = (video - 0.5) / 0.5  # -> [-1, 1]
        return video.unsqueeze(0).to(device=self._device, dtype=self._dtype)

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
        ctx, mask = self._model.encode_prompt(formatted)
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
        print(f"[fastwam_ws] infer begin: {_describe_obs(obs)}", flush=True)

        t_total = time.monotonic()
        with torch.no_grad():
            input_image = self._prep_image(
                obs["head_rgb"], obs["left_wrist_rgb"], obs["right_wrist_rgb"]
            )
            if save_client_frame and self._client_frame_saver is not None:
                saved_dir = self._client_frame_saver.save_once(obs, input_image)
                if saved_dir is not None:
                    print(f"[fastwam_ws] saved client frame: {saved_dir}", flush=True)
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
        if self._state_action_logger is not None and request_id is not None:
            self._state_action_logger.append(
                request_id=request_id,
                prompt=prompt,
                state=np.asarray(obs["state"], dtype=np.float32).reshape(-1),
                actions=actions_np,
                infer_ms=infer_ms,
                total_ms=total_ms,
            )
        print(
            f"[fastwam_ws] infer done: actions_shape={actions_np.shape} "
            f"model_infer_ms={infer_ms:.1f} total_ms={total_ms:.1f}",
            flush=True,
        )
        return {
            "actions": actions_np,
            "policy_timing": {"infer_ms": infer_ms},
        }

    def _warmup(self) -> None:
        print("[fastwam_ws] warmup ...", flush=True)
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
        print(f"[fastwam_ws] warmup done in {(time.monotonic() - t0) * 1000.0:.1f} ms", flush=True)

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
    def __init__(self, adapter: FastWAMAdapter, host: str, port: int) -> None:
        self._adapter = adapter
        self._host = host
        self._port = port
        self._metadata = adapter.metadata

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
            print(
                f"[fastwam_ws] listening ws://{self._host}:{self._port}  "
                f"healthz: curl -s http://{health_host}:{self._port}/healthz",
                flush=True,
            )
            await server.serve_forever()

    async def _handler(self, ws: _server.ServerConnection) -> None:
        print(f"[fastwam_ws] client connected: {ws.remote_address}", flush=True)
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        print(f"[fastwam_ws] metadata sent: {self._metadata}", flush=True)

        prev_total: float | None = None
        req_id = 0
        while True:
            try:
                t_start = time.monotonic()
                obs = msgpack_numpy.unpackb(await ws.recv())
                req_id += 1
                print(f"[fastwam_ws] request #{req_id} received", flush=True)

                t_inf = time.monotonic()
                action = await asyncio.to_thread(self._adapter.predict, obs, request_id=req_id)
                infer_ms = (time.monotonic() - t_inf) * 1000.0
                action.setdefault("server_timing", {})["infer_ms"] = infer_ms
                if prev_total is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total * 1000.0

                await ws.send(packer.pack(action))
                prev_total = time.monotonic() - t_start
                print(
                    f"[fastwam_ws] response #{req_id} server_infer_ms={infer_ms:.1f} "
                    f"total_ms={prev_total * 1000.0:.1f}",
                    flush=True,
                )
            except __import__("websockets").ConnectionClosed:
                print(f"[fastwam_ws] client disconnected: {ws.remote_address}", flush=True)
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"[fastwam_ws] error:\n{tb}", flush=True)
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
                    help="If set, save the first real client frame's raw cameras and model input PNGs here.")
    ap.add_argument("--log-state-action-csv", type=str, default=None,
                    help="If set, append each request's raw state and denormalized action chunk to this CSV.")
    ap.add_argument("--no-text-encoder", action="store_true",
                    help="Skip loading the T5 text encoder (only useful if you patch in cached embeddings).")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(
        f"[fastwam_ws] start task={args.task} ckpt={args.checkpoint} "
        f"ws://{args.host}:{args.port} device={args.device}",
        flush=True,
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
    )

    FastWAMPolicyServer(adapter, host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()