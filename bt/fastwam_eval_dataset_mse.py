#!/usr/bin/env python3
"""Evaluate FastWAM MSE for every sample in a LeRobot-style dataset.

By default this script evaluates episode chunks in order: for each episode, it
runs action-only inference at offsets 0, action_horizon, 2*action_horizon, ...
and writes a JSON report sorted by MSE from largest to smallest. A consecutive
per-sample mode is also available via ``--eval-mode sample``.

Example:

    python bt/fastwam_eval_dataset_mse.py \
        --checkpoint runs/r1/checkpoints/weights/step_002000.pt \
        --dataset-stats runs/r1/dataset_stats.json \
        --task r1_pro_chassis_uncond_3cam_384_1e-4 \
        --test-data /mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_test_data \
        --action-horizon 16 \
        --save-json tmp/fastwam_dataset_mse.json
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BT_ROOT = PROJECT_ROOT / "bt"
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, BT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_eval_policy import (  # noqa: E402
    _compose_cfg,
    _expand_checkpoints,
    _make_eval_dataset,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats,
    collect_fastwam_openloop,
    denormalize_merged_action,
)


def _episode_lookup(dataset) -> tuple[list[int], list[int]]:
    base = dataset.lerobot_dataset
    starts = [int(x) for x in base.episode_data_index["from"].detach().cpu().tolist()]
    ends = [int(x) for x in base.episode_data_index["to"].detach().cpu().tolist()]
    return starts, ends


def _episode_for_index(starts: list[int], ends: list[int], dataset_index: int) -> tuple[int | None, int | None]:
    episode_index = bisect_right(starts, int(dataset_index)) - 1
    if episode_index < 0 or episode_index >= len(ends):
        return None, None
    if dataset_index >= ends[episode_index]:
        return None, None
    return episode_index, int(dataset_index - starts[episode_index])


def _build_eval_items(
    *,
    eval_mode: str,
    dataset_len: int,
    starts: list[int],
    ends: list[int],
    action_horizon: int,
    start_index: int,
    end_index: int,
    max_samples: int | None,
    episode_start: int,
    episode_end: int | None,
    max_episodes: int | None,
) -> list[dict[str, int | None]]:
    """Build evaluation units in deterministic order.

    ``sample`` mode evaluates consecutive dataset indices. ``chunk`` mode
    evaluates chunk starts inside each episode: 0, H, 2H, ... where H is the
    action horizon. ``episode`` mode evaluates one full episode per item.
    """
    if eval_mode == "sample":
        indices = list(range(start_index, end_index))
        if max_samples is not None:
            indices = indices[: int(max_samples)]
        return [
            {
                "dataset_index": int(idx),
                "episode_index": _episode_for_index(starts, ends, idx)[0],
                "episode_offset": _episode_for_index(starts, ends, idx)[1],
                "chunk_index": None,
            }
            for idx in indices
        ]

    if eval_mode not in {"chunk", "episode"}:
        raise ValueError(f"Unsupported eval_mode={eval_mode!r}. Expected 'chunk', 'episode', or 'sample'.")

    episode_count = len(starts)
    ep_start = max(0, int(episode_start))
    ep_end = episode_count if episode_end is None else min(episode_count, int(episode_end))
    if max_episodes is not None:
        ep_end = min(ep_end, ep_start + int(max_episodes))
    if ep_end <= ep_start:
        raise ValueError(f"Empty episode range: episode_start={ep_start}, episode_end={ep_end}")

    items: list[dict[str, int | None]] = []
    for episode_index in range(ep_start, ep_end):
        ep_from = int(starts[episode_index])
        ep_to = int(ends[episode_index])
        ep_len = ep_to - ep_from
        if eval_mode == "episode":
            items.append(
                {
                    "dataset_index": int(ep_from),
                    "episode_index": int(episode_index),
                    "episode_offset": 0,
                    "chunk_index": None,
                }
            )
            if max_samples is not None and len(items) >= int(max_samples):
                return items
            continue
        for chunk_index, episode_offset in enumerate(range(0, ep_len, int(action_horizon))):
            dataset_index = ep_from + episode_offset
            if dataset_index < 0 or dataset_index >= dataset_len:
                continue
            items.append(
                {
                    "dataset_index": int(dataset_index),
                    "episode_index": int(episode_index),
                    "episode_offset": int(episode_offset),
                    "chunk_index": int(chunk_index),
                }
            )
            if max_samples is not None and len(items) >= int(max_samples):
                return items
    return items


def _valid_horizon(sample: dict[str, Any], pred: torch.Tensor, action_horizon: int) -> int:
    gt_norm = sample["action"]
    h = min(int(action_horizon), int(pred.shape[0]), int(gt_norm.shape[0]))
    action_is_pad = sample.get("action_is_pad")
    if action_is_pad is not None:
        valid = (~action_is_pad[:h].bool()).detach().cpu()
        if bool(valid.any().item()):
            h = int(valid.nonzero()[-1].item()) + 1
        else:
            h = 0
    return h


def eval_one_sample(
    model,
    dataset,
    dataset_index: int,
    *,
    action_horizon: int,
    num_inference_steps: int,
    seed: int | None,
    rand_device: str,
    sigma_shift: float | None,
    tiled: bool,
    mse_drop_dims: list[int] | None,
    infer_signature,
) -> dict[str, Any]:
    processor = dataset.lerobot_dataset.processor
    sample = dataset[int(dataset_index)]
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

    h = _valid_horizon(sample, pred, int(action_horizon))
    if h <= 0:
        raise ValueError("No valid action steps for sample; all action steps appear padded.")

    pred_denorm = denormalize_merged_action(processor, pred[:h], proprio[:h])[0]
    gt_denorm = denormalize_merged_action(processor, sample["action"][:h], proprio[:h])[0]

    if pred_denorm.shape != gt_denorm.shape:
        raise ValueError(f"Prediction/GT shape mismatch: pred={pred_denorm.shape}, gt={gt_denorm.shape}")
    if np.isnan(pred_denorm).any():
        raise ValueError("Predicted action contains NaN.")

    if mse_drop_dims:
        drop = set(int(i) for i in mse_drop_dims)
        keep = [i for i in range(gt_denorm.shape[1]) if i not in drop]
        diff = pred_denorm[:, keep] - gt_denorm[:, keep]
        mse_dims = np.mean(diff**2, axis=0)
        mae_dims = np.mean(np.abs(diff), axis=0)
    else:
        diff = pred_denorm - gt_denorm
        mse_dims = np.mean(diff**2, axis=0)
        mae_dims = np.mean(np.abs(diff), axis=0)

    return {
        "dataset_index": int(dataset_index),
        "mse": float(np.mean(diff**2)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "valid_steps": int(h),
        "action_dim": int(gt_denorm.shape[1]),
        "mse_per_dim": [float(x) for x in mse_dims.tolist()],
        "mae_per_dim": [float(x) for x in mae_dims.tolist()],
    }


def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = _expand_checkpoints([args.checkpoint])
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one checkpoint, got {len(checkpoints)}")
    checkpoint = checkpoints[0]

    cfg = _compose_cfg(args.task, args.config_override)
    mixed_precision = args.mixed_precision or str(cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to cpu.")
        device = "cpu"

    dataset_stats = _resolve_dataset_stats(checkpoint, args.dataset_stats)
    print(f"Loading eval dataset with stats: {dataset_stats}", flush=True)
    dataset = _make_eval_dataset(
        cfg,
        test_data=args.test_data,
        dataset_stats=dataset_stats,
        val_set_proportion=args.val_set_proportion,
    )
    starts, ends = _episode_lookup(dataset)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model = instantiate(model_cfg, model_dtype=model_dtype, device=str(device))
    model.load_checkpoint(str(checkpoint))
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}", flush=True)

    dataset_len = len(dataset)
    start_index = max(0, int(args.start_index))
    end_index = dataset_len if args.end_index is None else min(dataset_len, int(args.end_index))
    if end_index <= start_index:
        raise ValueError(f"Empty evaluation range: start_index={start_index}, end_index={end_index}")

    action_horizon = args.action_horizon or int(cfg.data.train.num_frames) - 1
    eval_items = _build_eval_items(
        eval_mode=str(args.eval_mode),
        dataset_len=int(dataset_len),
        starts=starts,
        ends=ends,
        action_horizon=int(action_horizon),
        start_index=int(start_index),
        end_index=int(end_index),
        max_samples=args.max_samples,
        episode_start=int(args.episode_start),
        episode_end=args.episode_end,
        max_episodes=args.max_episodes,
    )
    if not eval_items:
        raise ValueError("No evaluation items selected.")

    infer_signature = inspect.signature(model.infer_action).parameters
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total = len(eval_items)
    t0 = time.monotonic()
    for offset, item in enumerate(eval_items, start=1):
        dataset_index = int(item["dataset_index"])
        if args.log_every > 0 and (offset == 1 or offset % int(args.log_every) == 0 or offset == total):
            elapsed = time.monotonic() - t0
            print(
                f"[{offset}/{total}] dataset_index={dataset_index} "
                f"episode={item['episode_index']} offset={item['episode_offset']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        ep_idx = item["episode_index"]
        ep_offset = item["episode_offset"]
        try:
            if str(args.eval_mode) == "episode":
                if ep_idx is None:
                    raise ValueError(f"dataset_index={dataset_index} does not map to a valid episode.")
                ep_len = int(ends[int(ep_idx)] - starts[int(ep_idx)])
                steps = ep_len if args.max_steps_per_episode is None else min(ep_len, int(args.max_steps_per_episode))
                info = collect_fastwam_openloop(
                    model,
                    dataset,
                    episode_index=int(ep_idx),
                    steps=int(steps),
                    action_horizon=int(action_horizon),
                    num_inference_steps=int(args.num_inference_steps),
                    seed=args.seed,
                    rand_device=str(args.rand_device),
                    sigma_shift=args.sigma_shift,
                    tiled=bool(args.tiled),
                    plot_state=False,
                    mse_drop_dims=list(args.mse_drop_dims) if args.mse_drop_dims else None,
                )
                result = {
                    "dataset_index": int(dataset_index),
                    "episode_index": int(ep_idx),
                    "episode_offset": 0,
                    "chunk_index": None,
                    "eval_mode": "episode",
                    "episode_len": int(ep_len),
                    "steps_evaluated": int(info["steps_plotted"]),
                    "num_chunks": int((int(info["steps_plotted"]) + int(action_horizon) - 1) // int(action_horizon)),
                    "mse": float(info["mse"]),
                    "mae": float(info["mae"]),
                    "action_dim": int(info["action_dim"]),
                    "action_horizon": int(info["action_horizon"]),
                }
            else:
                result = eval_one_sample(
                    model,
                    dataset,
                    dataset_index,
                    action_horizon=int(action_horizon),
                    num_inference_steps=int(args.num_inference_steps),
                    seed=args.seed,
                    rand_device=str(args.rand_device),
                    sigma_shift=args.sigma_shift,
                    tiled=bool(args.tiled),
                    mse_drop_dims=list(args.mse_drop_dims) if args.mse_drop_dims else None,
                    infer_signature=infer_signature,
                )
                result["episode_index"] = ep_idx
                result["episode_offset"] = ep_offset
                result["chunk_index"] = item["chunk_index"]
                result["eval_mode"] = str(args.eval_mode)
            results.append(result)
        except Exception as err:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            failures.append(
                {
                    "dataset_index": int(dataset_index),
                    "episode_index": ep_idx,
                    "episode_offset": ep_offset,
                    "chunk_index": item["chunk_index"],
                    "error": f"{type(err).__name__}: {err}",
                }
            )
            print(f"[warn] dataset_index={dataset_index} failed: {type(err).__name__}: {err}", flush=True)

    results_sorted = sorted(results, key=lambda item: item["mse"], reverse=True)
    for rank, item in enumerate(results_sorted, start=1):
        item["rank_by_mse_desc"] = rank

    mse_values = [item["mse"] for item in results_sorted]
    summary = {
        "checkpoint": str(checkpoint),
        "dataset_stats": str(dataset_stats),
        "task": str(args.task),
        "test_data": args.test_data,
        "eval_mode": str(args.eval_mode),
        "dataset_len": int(dataset_len),
        "num_episodes": int(len(starts)),
        "start_index": int(start_index),
        "end_index": int(end_index),
        "episode_start": int(args.episode_start),
        "episode_end": int(args.episode_end) if args.episode_end is not None else None,
        "max_episodes": int(args.max_episodes) if args.max_episodes is not None else None,
        "selected_items": int(total),
        "evaluated_items": int(len(results_sorted)),
        "failed_items": int(len(failures)),
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(args.num_inference_steps),
        "seed": args.seed,
        "mse_drop_dims": list(args.mse_drop_dims) if args.mse_drop_dims else [],
        "mse_mean": float(np.mean(mse_values)) if mse_values else None,
        "mse_median": float(np.median(mse_values)) if mse_values else None,
        "mse_max": float(np.max(mse_values)) if mse_values else None,
        "mse_min": float(np.min(mse_values)) if mse_values else None,
    }
    return {
        "summary": summary,
        "results": results_sorted,
        "failures": failures,
    }


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FastWAM denormalized MSE by episode chunks or samples and save sorted JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path or glob resolving to one file.")
    parser.add_argument("--dataset-stats", type=str, default=None, help="Path to training dataset_stats.json.")
    parser.add_argument("--task", type=str, default="r1_pro_chassis_uncond_3cam_384_1e-4")
    parser.add_argument("--test-data", type=str, default=None, help="Override data.train.dataset_dirs with this dataset dir.")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Extra Hydra override, e.g. data.train.text_embedding_cache_dir=/path.",
    )
    parser.add_argument("--save-json", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--eval-mode",
        choices=["episode", "chunk", "sample"],
        default="chunk",
        help=(
            "episode: one full trajectory per result; "
            "chunk: evaluate episode offsets 0,H,2H,...; "
            "sample: evaluate consecutive dataset indices."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive dataset index end, only used by sample mode.")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate at most this many items/chunks.")
    parser.add_argument("--episode-start", type=int, default=0, help="First episode index, only used by chunk mode.")
    parser.add_argument("--episode-end", type=int, default=None, help="Exclusive episode index end, only used by chunk mode.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Evaluate at most this many episodes in chunk mode.")
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional cap on evaluated steps per episode, only used by episode mode.",
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", type=str, default="cpu")
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--val-set-proportion", type=float, default=None)
    parser.add_argument("--mse-drop-dims", type=int, nargs="*", default=[])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli()
    report = evaluate_dataset(args)
    out = Path(args.save_json).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        "Saved sorted MSE report to "
        f"{out.resolve()} (items={report['summary']['evaluated_items']}, "
        f"failures={report['summary']['failed_items']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
