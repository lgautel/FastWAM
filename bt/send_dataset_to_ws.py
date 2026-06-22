#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import websockets

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

        class _MsgpackNumpy:
            Packer = _Packer
            unpackb = staticmethod(_unpack)

        msgpack_numpy = _MsgpackNumpy()

from bt.fastwam_eval_policy import (
    _compose_cfg,
    _episode_span,
    _make_eval_dataset,
    _normalized_image_to_hwc_uint8,
    denormalize_merged_state,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def _dataset_prompt_to_raw_task(prompt: str) -> str:
    prefix = DEFAULT_PROMPT.split("{task}")[0]
    if prompt.startswith(prefix):
        return prompt[len(prefix):]
    return prompt


def _resize_hwc_uint8(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got {image.dtype}")
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def _make_obs(
    dataset,
    sample_idx: int,
    *,
    prompt_override: str | None,
    client_image_width: int,
    client_image_height: int,
) -> dict:
    sample = dataset[sample_idx]

    pre = sample.get("pre_resize_video")
    if pre is None:
        raise RuntimeError(
            "sample has no pre_resize_video. Make sure dataset is eval mode "
            "(is_training_set=False), as done by _make_eval_dataset()."
        )
    if pre.ndim != 5 or pre.shape[0] < 3:
        raise RuntimeError(f"expected pre_resize_video [3,T,C,H,W], got {tuple(pre.shape)}")

    head = _normalized_image_to_hwc_uint8(pre[0, 0])
    left = _normalized_image_to_hwc_uint8(pre[1, 0])
    right = _normalized_image_to_hwc_uint8(pre[2, 0])
    head = _resize_hwc_uint8(head, width=client_image_width, height=client_image_height)
    left = _resize_hwc_uint8(left, width=client_image_width, height=client_image_height)
    right = _resize_hwc_uint8(right, width=client_image_width, height=client_image_height)

    processor = dataset.lerobot_dataset.processor
    proprio = sample["proprio"]
    state = denormalize_merged_state(processor, proprio[:1], horizon=1)[0, 0].astype(np.float32)

    if prompt_override is not None:
        prompt = prompt_override
    else:
        prompt = _dataset_prompt_to_raw_task(str(sample["prompt"]))

    return {
        "head_rgb": head,
        "left_wrist_rgb": left,
        "right_wrist_rgb": right,
        "state": state,
        "prompt": prompt,
        "role": "image",
    }


async def _send(args: argparse.Namespace) -> None:
    cfg = _compose_cfg(args.task, args.config_override or [])
    dataset = _make_eval_dataset(
        cfg,
        test_data=args.test_data,
        dataset_stats=Path(args.dataset_stats).expanduser(),
        val_set_proportion=args.val_set_proportion,
    )

    ep_start, ep_end = _episode_span(dataset, args.episode_index)
    packer = msgpack_numpy.Packer()

    csv_f = None
    writer = None
    if args.save_csv:
        out = Path(args.save_csv).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(out, "w", encoding="utf-8", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(
            ["request_idx", "sample_idx", "chunk_offset"]
            + [f"state_{i}" for i in range(23)]
            + [f"action_{i}" for i in range(23)]
        )

    try:
        async with websockets.connect(args.url, max_size=None, compression=None) as ws:
            metadata = msgpack_numpy.unpackb(await ws.recv())
            print("metadata:", metadata)

            for request_idx in range(args.num_requests):
                sample_idx = ep_start + args.start_step + request_idx * args.step_stride
                if sample_idx >= ep_end:
                    print(f"stop: sample_idx={sample_idx} >= episode_end={ep_end}")
                    break

                obs = _make_obs(
                    dataset,
                    sample_idx,
                    prompt_override=args.prompt,
                    client_image_width=args.client_image_width,
                    client_image_height=args.client_image_height,
                )
                await ws.send(packer.pack(obs))

                resp = msgpack_numpy.unpackb(await ws.recv())
                if isinstance(resp, str):
                    raise RuntimeError(resp)

                actions = np.asarray(resp["actions"], dtype=np.float32)
                state = np.asarray(obs["state"], dtype=np.float32)

                print(f"\nrequest={request_idx} sample_idx={sample_idx}")
                print("prompt:", obs["prompt"])
                print(
                    "image shapes:",
                    obs["head_rgb"].shape,
                    obs["left_wrist_rgb"].shape,
                    obs["right_wrist_rgb"].shape,
                )
                print("state[0:14]:", np.round(state[:14], 6).tolist())
                print("action[0,0:14]:", np.round(actions[0, :14], 6).tolist())
                print("action[-1,0:14]:", np.round(actions[-1, :14], 6).tolist())
                print("chunk delta first14:", np.round(actions[-1, :14] - actions[0, :14], 6).tolist())
                print("state[14:23]:", np.round(state[14:23], 6).tolist())
                print("action[0,14:23]:", np.round(actions[0, 14:23], 6).tolist())

                if writer is not None:
                    for chunk_offset, action in enumerate(actions):
                        writer.writerow(
                            [request_idx, sample_idx, chunk_offset]
                            + [float(x) for x in state]
                            + [float(x) for x in action]
                        )
    finally:
        if csv_f is not None:
            csv_f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--task", default="r1_pro_chassis_uncond_3cam_384_1e-4")
    ap.add_argument("--test-data", default="/mnt/r/share/zwy/datasets/r1_pro_data_v2/r1_pro_test_data")
    ap.add_argument("--dataset-stats", default="runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json")
    ap.add_argument("--episode-index", type=int, default=2)
    ap.add_argument("--start-step", type=int, default=0)
    ap.add_argument("--num-requests", type=int, default=2)
    ap.add_argument("--step-stride", type=int, default=32)
    ap.add_argument("--val-set-proportion", type=float, default=None)
    ap.add_argument("--config-override", action="append", default=[])
    ap.add_argument("--prompt", default=None, help="Override raw task prompt. Defaults to dataset task.")
    ap.add_argument("--save-csv", default="tmp/ws_dataset_probe.csv")
    ap.add_argument("--client-image-width", type=int, default=320)
    ap.add_argument("--client-image-height", type=int, default=240)
    args = ap.parse_args()
    asyncio.run(_send(args))


if __name__ == "__main__":
    main()