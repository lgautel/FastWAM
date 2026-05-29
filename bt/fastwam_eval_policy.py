#!/usr/bin/env python3
"""FastWAM open-loop checkpoint evaluation on a LeRobot-style test set.

The script mirrors the chunk re-inference pattern used by
``bt/rldx1_eval_policy.py``: at every ``action_horizon`` episode steps, it
infers one action chunk from the current observation, denormalizes both
prediction and GT actions with the training dataset stats, then plots and
reports unnormalized error.

Example:

    python bt/fastwam_eval_policy.py \
        --checkpoint runs/r1/checkpoints/weights/step_002000.pt \
        --dataset-stats runs/r1/dataset_stats.json \
        --task r1_pro_chassis_uncond_3cam_384_1e-4 \
        --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_data_convert_chassis \
        --episode-index 0 \
        --steps 300 \
        --action-horizon 32 \
        --save-plot /tmp/fastwam_openloop.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
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


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _compose_cfg(task: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def _expand_checkpoints(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(os.path.expanduser(os.path.expandvars(value))))
        if matches:
            paths.extend(Path(m).resolve() for m in matches)
        else:
            paths.append(Path(value).expanduser().resolve())
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint path(s) not found: {missing}")
    return paths


def _resolve_dataset_stats(checkpoint: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(parent / "dataset_stats.json" for parent in list(checkpoint.parents)[:6])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass --dataset-stats explicitly "
        "or place dataset_stats.json in a checkpoint parent directory."
    )


def _make_eval_dataset(
    cfg: DictConfig,
    *,
    test_data: str | None,
    dataset_stats: Path,
    val_set_proportion: float | None,
):
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    if test_data:
        data_cfg.dataset_dirs = [str(Path(test_data).expanduser())]
    data_cfg.is_training_set = False
    data_cfg.pretrained_norm_stats = str(dataset_stats)
    if val_set_proportion is not None:
        data_cfg.val_set_proportion = float(val_set_proportion)
    elif float(data_cfg.get("val_set_proportion", 0.0)) <= 0.0:
        data_cfg.val_set_proportion = 0.0
    return instantiate(data_cfg)


def _episode_span(dataset, episode_index: int) -> tuple[int, int]:
    base = dataset.lerobot_dataset
    starts = base.episode_data_index["from"].detach().cpu().tolist()
    ends = base.episode_data_index["to"].detach().cpu().tolist()
    if episode_index < 0 or episode_index >= len(starts):
        raise IndexError(
            f"episode_index={episode_index} out of range (num episodes={len(starts)})."
        )
    return int(starts[episode_index]), int(ends[episode_index])


def _to_batched_action(action: torch.Tensor) -> torch.Tensor:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action [T,D] or [B,T,D], got {tuple(action.shape)}")
    return action.detach().to(device="cpu", dtype=torch.float32)


def _repeat_or_trim_state(proprio: torch.Tensor, horizon: int) -> torch.Tensor:
    state = proprio.detach().to(device="cpu", dtype=torch.float32)
    if state.ndim == 2:
        state = state.unsqueeze(0)
    if state.ndim != 3:
        raise ValueError(f"Expected proprio [T,D] or [B,T,D], got {tuple(state.shape)}")
    if state.shape[1] >= horizon:
        return state[:, :horizon]
    pad = state[:, -1:].repeat(1, horizon - state.shape[1], 1)
    return torch.cat([state, pad], dim=1)


def denormalize_merged_action(processor, action: torch.Tensor, proprio: torch.Tensor) -> np.ndarray:
    """Undo FastWAM processor normalization and return merged flat actions."""
    action_btd = _to_batched_action(action)
    state_btd = _repeat_or_trim_state(proprio, int(action_btd.shape[1]))

    batch = {
        "action": action_btd,
        "state": state_btd,
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    if processor.action_state_transforms is not None:
        for transform in reversed(processor.action_state_transforms):
            batch = transform.backward(batch)

    parts = [batch["action"][meta["key"]] for meta in processor.shape_meta["action"]]
    return torch.cat(parts, dim=-1).detach().cpu().numpy()


def denormalize_merged_state(processor, proprio: torch.Tensor, horizon: int) -> np.ndarray:
    state_btd = _repeat_or_trim_state(proprio, horizon)
    batch = {
        "action": torch.zeros(
            (state_btd.shape[0], horizon, int(processor.action_output_dim)),
            dtype=torch.float32,
        ),
        "state": state_btd,
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    if processor.action_state_transforms is not None:
        for transform in reversed(processor.action_state_transforms):
            batch = transform.backward(batch)
    parts = [batch["state"][meta["key"]] for meta in processor.shape_meta["state"]]
    return torch.cat(parts, dim=-1).detach().cpu().numpy()


def _dim_names(action_dim: int) -> list[str]:
    base = [
        "left_arm_0",
        "left_arm_1",
        "left_arm_2",
        "left_arm_3",
        "left_arm_4",
        "left_arm_5",
        "left_arm_6",
        "right_arm_0",
        "right_arm_1",
        "right_arm_2",
        "right_arm_3",
        "right_arm_4",
        "right_arm_5",
        "right_arm_6",
        "left_gripper",
        "right_gripper",
        "chassis_pose_0",
        "chassis_pose_1",
        "chassis_pose_2",
        "chassis_pose_3",
        "chassis_velocity_0",
        "chassis_velocity_1",
        "chassis_velocity_2",
    ]
    if action_dim <= len(base):
        return base[:action_dim]
    return base + [f"dim_{i}" for i in range(len(base), action_dim)]


def collect_fastwam_openloop(
    model,
    dataset,
    *,
    episode_index: int,
    steps: int,
    action_horizon: int,
    num_inference_steps: int,
    seed: int | None,
    rand_device: str,
    sigma_shift: float | None,
    tiled: bool,
    plot_state: bool,
    mse_drop_dims: list[int] | None,
) -> dict[str, Any]:
    start, end = _episode_span(dataset, episode_index)
    ep_len = end - start
    steps_use = min(int(steps), ep_len)
    processor = dataset.lerobot_dataset.processor

    gt_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []

    infer_signature = inspect.signature(model.infer_action).parameters
    for step_count in range(0, steps_use, int(action_horizon)):
        sample = dataset[start + step_count]
        video = sample["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError(f"Expected sample['video'] [C,T,H,W], got {type(video)} {getattr(video, 'shape', None)}")
        input_image = video[:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        proprio = sample["proprio"]
        proprio0 = proprio[0].to(device=model.device, dtype=model.torch_dtype)

        infer_kwargs = {
            "prompt": None,
            "input_image": input_image,
            "action_horizon": int(action_horizon),
            "proprio": proprio0,
            "context": sample["context"],
            "context_mask": sample["context_mask"],
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": rand_device,
            "tiled": tiled,
        }
        if "num_video_frames" in infer_signature:
            infer_kwargs["num_video_frames"] = int(video.shape[1])
        with torch.no_grad():
            pred = model.infer_action(**infer_kwargs)["action"]

        gt_norm = sample["action"]
        h = min(int(action_horizon), int(pred.shape[0]), int(gt_norm.shape[0]))
        if h <= 0:
            continue
        pred_denorm = denormalize_merged_action(processor, pred[:h], proprio[:h])[0]
        gt_denorm = denormalize_merged_action(processor, gt_norm[:h], proprio[:h])[0]

        pred_rows.extend(pred_denorm)
        gt_rows.extend(gt_denorm)
        if plot_state:
            state_denorm = denormalize_merged_state(processor, proprio[:h], h)[0]
            state_rows.extend(state_denorm)

    if not gt_rows:
        raise ValueError("No predictions collected; check episode length, steps, and action_horizon.")

    gt_a = np.stack(gt_rows, axis=0)[:steps_use]
    pred_a = np.stack(pred_rows, axis=0)[:steps_use]
    if gt_a.shape != pred_a.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={pred_a.shape}, gt={gt_a.shape}")
    if np.isnan(pred_a).any():
        raise ValueError("Predicted action contains NaN.")

    state_a = (
        np.stack(state_rows, axis=0)[:steps_use]
        if plot_state and state_rows
        else np.empty((0, 0), dtype=np.float32)
    )

    if mse_drop_dims:
        drop = set(int(i) for i in mse_drop_dims)
        keep = [i for i in range(gt_a.shape[1]) if i not in drop]
        mse = float(np.mean((gt_a[:, keep] - pred_a[:, keep]) ** 2))
    else:
        mse = float(np.mean((gt_a - pred_a) ** 2))

    return {
        "state_joints_across_time": state_a,
        "gt_action_across_time": gt_a,
        "pred_action_across_time": pred_a,
        "dim_names": _dim_names(int(gt_a.shape[1])),
        "traj_id": int(episode_index),
        "mse": mse,
        "mae": float(np.mean(np.abs(gt_a - pred_a))),
        "action_dim": int(gt_a.shape[1]),
        "action_horizon": int(action_horizon),
        "steps": int(steps),
        "steps_plotted": int(gt_a.shape[0]),
    }


def _unified_ylim(
    gt: np.ndarray,
    pred: np.ndarray,
    state: np.ndarray,
    *,
    margin_frac: float,
) -> tuple[float, float] | None:
    parts: list[np.ndarray] = [gt.reshape(-1), pred.reshape(-1)]
    if state.size and state.shape == gt.shape:
        parts.append(state.reshape(-1))
    stacked = np.concatenate(parts, axis=0)
    lo = float(np.nanmin(stacked))
    hi = float(np.nanmax(stacked))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    pad = (hi - lo) * float(margin_frac)
    return (lo - pad, hi + pad)


def plot_fastwam_openloop_trajectory(
    info: dict[str, Any],
    *,
    save_plot_path: str | Path | None,
    show: bool,
    title_suffix: str,
    unify_y_scale: bool,
    y_margin_frac: float,
    ylim_fixed: tuple[float, float] | None,
) -> None:
    if save_plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    action_dim = int(info["action_dim"])
    state = info["state_joints_across_time"]
    gt = info["gt_action_across_time"]
    pred = info["pred_action_across_time"]
    dim_names = info.get("dim_names") or [f"dim_{i}" for i in range(action_dim)]
    steps = int(info["steps_plotted"])
    action_horizon = int(info["action_horizon"])

    fig_h = max(4.0 * action_dim + 2.0, 6.0)
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)
    fig.suptitle(
        f"FastWAM open-loop - episode: {info['traj_id']}\n"
        f"Unnormalized MSE: {info['mse']:.6f}, MAE: {info['mae']:.6f}{title_suffix}",
        fontsize=14,
        fontweight="bold",
        color="#2E86AB",
        y=0.95,
    )

    for i, ax in enumerate(axes_flat):
        if state.size and state.shape == gt.shape:
            ax.plot(state[:, i], label="state (obs)", alpha=0.7)
        ax.plot(gt[:, i], label="gt action", linewidth=2)
        ax.plot(pred[:, i], label="pred action", linewidth=2)
        for j in range(0, steps, action_horizon):
            ax.plot(j, float(gt[j, i]), "ro", markersize=6 if j == 0 else 4)
        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        ax.set_title(f"Action - {name}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time step (padded chunk replay)", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if ylim_fixed is not None:
        y0, y1 = ylim_fixed
        for ax in axes_flat:
            ax.set_ylim(y0, y1)
            ax.set_ylabel("Value (manual y-range)", fontsize=10)
    elif unify_y_scale:
        ylim = _unified_ylim(gt, pred, state, margin_frac=y_margin_frac)
        if ylim is not None:
            for ax in axes_flat:
                ax.set_ylim(*ylim)
                ax.set_ylabel("Value (shared scale)", fontsize=10)

    if save_plot_path is not None:
        out = Path(save_plot_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out.resolve()}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


def _default_plot_path(save_plot: str | None, checkpoint: Path, multi_ckpt: bool) -> str | None:
    if save_plot is None:
        return None
    if not multi_ckpt:
        return save_plot
    out = Path(save_plot)
    suffix = out.suffix or ".png"
    stem = out.stem if out.suffix else out.name
    return str(out.with_name(f"{stem}_{checkpoint.stem}{suffix}"))


def _default_artifact_path(path: str | None, checkpoint: Path, multi_ckpt: bool, default_suffix: str) -> str | None:
    if path is None:
        return None
    if not multi_ckpt:
        return path
    out = Path(path)
    suffix = out.suffix or default_suffix
    stem = out.stem if out.suffix else out.name
    return str(out.with_name(f"{stem}_{checkpoint.stem}{suffix}"))


def save_fastwam_openloop_csv(
    info: dict[str, Any],
    *,
    csv_path: str | Path,
    checkpoint: Path,
    dataset_stats: Path,
) -> None:
    out = Path(csv_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    gt = np.asarray(info["gt_action_across_time"])
    pred = np.asarray(info["pred_action_across_time"])
    state = np.asarray(info["state_joints_across_time"])
    dim_names = info.get("dim_names") or [f"dim_{i}" for i in range(gt.shape[1])]

    if gt.shape != pred.shape:
        raise ValueError(f"Cannot save CSV with pred/GT shape mismatch: pred={pred.shape}, gt={gt.shape}")
    has_state = state.size and state.shape == gt.shape

    header = [
        "checkpoint",
        "dataset_stats",
        "episode_index",
        "time_step",
        "chunk_index",
        "chunk_offset",
    ]
    for name in dim_names:
        header.extend([f"gt_{name}", f"pred_{name}", f"abs_error_{name}", f"sq_error_{name}"])
        if has_state:
            header.append(f"state_{name}")

    action_horizon = int(info["action_horizon"])
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(gt.shape[0]):
            row: list[Any] = [
                str(checkpoint),
                str(dataset_stats),
                int(info["traj_id"]),
                int(t),
                int(t // action_horizon),
                int(t % action_horizon),
            ]
            for i in range(gt.shape[1]):
                diff = float(pred[t, i] - gt[t, i])
                row.extend(
                    [
                        float(gt[t, i]),
                        float(pred[t, i]),
                        abs(diff),
                        diff * diff,
                    ]
                )
                if has_state:
                    row.append(float(state[t, i]))
            writer.writerow(row)

    print(f"Saved CSV to {out.resolve()}")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.save_plot and not args.show and not args.save_csv:
        raise SystemExit("Pass --save-plot PATH, --save-csv PATH, and/or --show.")

    checkpoints = _expand_checkpoints(args.checkpoint)
    cfg = _compose_cfg(args.task, args.config_override)
    mixed_precision = args.mixed_precision or str(cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to cpu.")
        device = "cpu"

    results: list[dict[str, Any]] = []
    dataset_cache: dict[Path, Any] = {}

    for checkpoint in checkpoints:
        dataset_stats = _resolve_dataset_stats(checkpoint, args.dataset_stats)
        if dataset_stats not in dataset_cache:
            print(f"Loading eval dataset with stats: {dataset_stats}")
            dataset_cache[dataset_stats] = _make_eval_dataset(
                cfg,
                test_data=args.test_data,
                dataset_stats=dataset_stats,
                val_set_proportion=args.val_set_proportion,
            )
        dataset = dataset_cache[dataset_stats]

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = False
        model = instantiate(model_cfg, model_dtype=model_dtype, device=str(device))
        model.load_checkpoint(str(checkpoint))
        model.eval()
        print(f"Loaded checkpoint: {checkpoint}")

        action_horizon = args.action_horizon or int(cfg.data.train.num_frames) - 1
        info = collect_fastwam_openloop(
            model,
            dataset,
            episode_index=int(args.episode_index),
            steps=int(args.steps),
            action_horizon=int(action_horizon),
            num_inference_steps=int(args.num_inference_steps),
            seed=args.seed,
            rand_device=str(args.rand_device),
            sigma_shift=args.sigma_shift,
            tiled=bool(args.tiled),
            plot_state=bool(args.plot_state),
            mse_drop_dims=list(args.mse_drop_dims) if args.mse_drop_dims else None,
        )

        plot_path = _default_plot_path(args.save_plot, checkpoint, multi_ckpt=len(checkpoints) > 1)
        plot_fastwam_openloop_trajectory(
            info,
            save_plot_path=plot_path,
            show=bool(args.show),
            title_suffix=f"\ncheckpoint: {checkpoint.name}",
            unify_y_scale=not bool(args.no_unify_y_scale),
            y_margin_frac=float(args.y_margin_frac),
            ylim_fixed=args.ylim_fixed,
        )

        csv_path = _default_artifact_path(args.save_csv, checkpoint, multi_ckpt=len(checkpoints) > 1, default_suffix=".csv")
        if csv_path is not None:
            save_fastwam_openloop_csv(
                info,
                csv_path=csv_path,
                checkpoint=checkpoint,
                dataset_stats=dataset_stats,
            )

        summary = {
            "checkpoint": str(checkpoint),
            "dataset_stats": str(dataset_stats),
            "episode_index": int(args.episode_index),
            "steps_plotted": info["steps_plotted"],
            "action_horizon": info["action_horizon"],
            "action_dim": info["action_dim"],
            "mse": info["mse"],
            "mae": info["mae"],
            "plot": plot_path,
            "csv": csv_path,
        }
        print(json.dumps(summary, indent=2))
        results.append(summary)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_json:
        out = Path(args.save_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary to {out.resolve()}")

    return results


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM open-loop checkpoint evaluation on LeRobot-style test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Checkpoint .pt path(s) or glob(s).")
    parser.add_argument("--dataset-stats", type=str, default=None, help="Path to training dataset_stats.json.")
    parser.add_argument("--task", type=str, default="r1_pro_chassis_uncond_3cam_384_1e-4")
    parser.add_argument("--test-data", type=str, default=None, help="Override data.train.dataset_dirs with this test dataset dir.")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Extra Hydra override, e.g. data.train.text_embedding_cache_dir=/path.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", type=str, default="cpu")
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--val-set-proportion", type=float, default=None)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--save-csv", type=str, default=None, help="Write per-step GT/pred action values to CSV.")
    parser.add_argument("--plot-state", action="store_true")
    parser.add_argument("--no-unify-y-scale", action="store_true")
    parser.add_argument("--y-margin-frac", type=float, default=0.05)
    parser.add_argument("--ylim-min", type=float, default=None, dest="ylim_min")
    parser.add_argument("--ylim-max", type=float, default=None, dest="ylim_max")
    parser.add_argument(
        "--mse-drop-dims",
        type=int,
        nargs="*",
        default=None,
        help="Action dims to exclude from reported MSE only, e.g. 14 15.",
    )
    args = parser.parse_args()

    lo, hi = args.ylim_min, args.ylim_max
    if (lo is None) ^ (hi is None):
        raise SystemExit("Pass both --ylim-min and --ylim-max, or neither.")
    args.ylim_fixed = None if lo is None else (float(lo), float(hi))
    if args.ylim_fixed is not None and args.ylim_fixed[0] >= args.ylim_fixed[1]:
        raise SystemExit("--ylim-min must be strictly less than --ylim-max.")
    return args


def main() -> None:
    run(_parse_cli())


if __name__ == "__main__":
    main()
