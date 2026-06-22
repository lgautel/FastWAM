#!/usr/bin/env python3
"""Compare real-robot and trainset VAE latents saved by the websocket server.

This script reads ``model_input.png`` frames from two log directories, encodes
them with FastWAM's VAE, and writes summary statistics plus diagnostic plots.

By default it compares:
  - logs/ws_client_frame
  - logs/ws_client_frame_trainsets

The comparison uses the already-saved ``model_input.png`` so it reflects the
exact preprocessing/color-match settings used by the server when the frame was
logged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bt.fastwam_ws_server_experiment import FastWAMAdapter  # noqa: E402


@dataclass
class DomainStats:
    name: str
    n: int
    dim: int
    scalar_mean: float
    scalar_std: float
    norm_mean: float
    norm_std: float
    norm_min: float
    norm_max: float


def _list_frame_dirs(root: Path, *, limit: int | None) -> list[Path]:
    dirs = sorted(
        [p for p in root.expanduser().iterdir() if (p / "model_input.png").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    if limit is not None and limit > 0:
        dirs = dirs[-limit:]
    return dirs


def _load_model_input(path: Path, *, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    # model_input.png is visualization of [-1, 1] tensor via (x + 1) * 127.5.
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor.to(dtype=dtype)


def _encode_domain(
    adapter: FastWAMAdapter,
    frame_dirs: list[Path],
    *,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, list[str], tuple[int, ...]]:
    flat_feats: list[np.ndarray] = []
    pooled_feats: list[np.ndarray] = []
    names: list[str] = []
    latent_shape: tuple[int, ...] | None = None

    for idx, frame_dir in enumerate(frame_dirs, start=1):
        inp = _load_model_input(frame_dir / "model_input.png", dtype=adapter._dtype)
        with torch.no_grad():
            z = adapter._model._encode_input_image_latents_tensor(inp, tiled=False)
        z = z.detach().float().cpu()
        latent_shape = tuple(z.shape)
        flat_feats.append(z.reshape(-1).numpy())
        pooled_feats.append(z.mean(dim=(-2, -1)).reshape(-1).numpy())
        names.append(frame_dir.name)
        print(
            f"[{domain}] encoded {idx:03d}/{len(frame_dirs):03d} "
            f"{frame_dir.name} latent_shape={latent_shape}",
            flush=True,
        )

    if not flat_feats:
        raise RuntimeError(f"No model_input.png frames found for domain={domain}")

    assert latent_shape is not None
    return np.stack(flat_feats), np.stack(pooled_feats), names, latent_shape


def _domain_stats(name: str, feats: np.ndarray) -> DomainStats:
    norms = np.linalg.norm(feats, axis=1)
    return DomainStats(
        name=name,
        n=int(feats.shape[0]),
        dim=int(feats.shape[1]),
        scalar_mean=float(feats.mean()),
        scalar_std=float(feats.std()),
        norm_mean=float(norms.mean()),
        norm_std=float(norms.std()),
        norm_min=float(norms.min()),
        norm_max=float(norms.max()),
    )


def _cosine_nn(
    query: np.ndarray,
    ref: np.ndarray,
    *,
    leave_one_out: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    sim = q @ r.T
    if leave_one_out:
        if query.shape != ref.shape or not np.allclose(query, ref):
            raise ValueError("leave_one_out=True expects query/ref to be the same array")
        np.fill_diagonal(sim, -np.inf)
    idx = sim.argmax(axis=1)
    dist = 1.0 - sim[np.arange(sim.shape[0]), idx]
    return dist, idx


def _pca_2d(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = feats - feats.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    return coords, explained[:2]


def _try_tsne(coords_or_feats: np.ndarray) -> tuple[np.ndarray | None, str]:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"skipped: scikit-learn unavailable ({type(exc).__name__}: {exc})"

    perplexity = max(5, min(30, (coords_or_feats.shape[0] - 1) // 3))
    if coords_or_feats.shape[0] <= perplexity + 1:
        return None, f"skipped: too few samples for perplexity={perplexity}"
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    return tsne.fit_transform(coords_or_feats), f"computed: perplexity={perplexity}"


def _plot_norm_hist(real_norm: np.ndarray, train_norm: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    bins = min(20, max(8, int(np.sqrt(real_norm.size + train_norm.size))))
    plt.hist(train_norm, bins=bins, alpha=0.65, label="train", color="#4C78A8")
    plt.hist(real_norm, bins=bins, alpha=0.65, label="real", color="#F58518")
    plt.xlabel("VAE latent L2 norm")
    plt.ylabel("count")
    plt.title("Latent Norm Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    out: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(
        coords[labels == "train", 0],
        coords[labels == "train", 1],
        s=28,
        alpha=0.8,
        label="train",
        color="#4C78A8",
    )
    plt.scatter(
        coords[labels == "real", 0],
        coords[labels == "real", 1],
        s=28,
        alpha=0.8,
        label="real",
        color="#F58518",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_nn_hist(train_nn: np.ndarray, real_to_train_nn: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    bins = min(20, max(8, int(np.sqrt(train_nn.size + real_to_train_nn.size))))
    plt.hist(train_nn, bins=bins, alpha=0.65, label="train leave-one-out NN", color="#4C78A8")
    plt.hist(real_to_train_nn, bins=bins, alpha=0.65, label="real -> train NN", color="#F58518")
    plt.xlabel("cosine nearest-neighbor distance (lower is closer)")
    plt.ylabel("count")
    plt.title("Nearest-Neighbor Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _plot_pooled_channel_delta(delta: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = np.arange(delta.size)
    colors = np.where(delta >= 0, "#F58518", "#4C78A8")
    plt.bar(x, delta, color=colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("pooled latent channel")
    plt.ylabel("real mean - train mean")
    plt.title("Pooled Latent Channel Mean Delta")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def _write_nn_csv(
    out: Path,
    real_names: list[str],
    train_names: list[str],
    dist: np.ndarray,
    idx: np.ndarray,
) -> None:
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["real_frame", "nearest_train_frame", "cosine_distance"])
        for real_name, nearest_idx, distance in zip(real_names, idx, dist):
            writer.writerow([real_name, train_names[int(nearest_idx)], float(distance)])


def _write_summary_txt(out: Path, summary: dict[str, Any]) -> None:
    real = summary["domain_stats"]["real"]
    train = summary["domain_stats"]["train"]
    nn = summary["nearest_neighbor"]
    pca = summary["pca"]

    lines = [
        "# VAE Latent Compare Summary",
        "",
        f"real frames: {real['n']}",
        f"train frames: {train['n']}",
        f"latent shape: {summary['latent_shape']}",
        "",
        "## Flat Latent Stats",
        f"real  scalar mean/std: {real['scalar_mean']:+.6f} / {real['scalar_std']:.6f}",
        f"train scalar mean/std: {train['scalar_mean']:+.6f} / {train['scalar_std']:.6f}",
        f"real  L2 norm mean/std: {real['norm_mean']:.4f} / {real['norm_std']:.4f}",
        f"train L2 norm mean/std: {train['norm_mean']:.4f} / {train['norm_std']:.4f}",
        "",
        "## Centroid Gap",
        f"centroid L2(real_mean, train_mean): {summary['centroid']['l2_real_train']:.4f}",
        f"train -> train center mean/std: {summary['centroid']['train_to_train_center_mean']:.4f} / "
        f"{summary['centroid']['train_to_train_center_std']:.4f}",
        f"real  -> train center mean/std: {summary['centroid']['real_to_train_center_mean']:.4f} / "
        f"{summary['centroid']['real_to_train_center_std']:.4f}",
        "",
        "## Nearest Neighbor",
        f"train leave-one-out NN mean/median/p90: {nn['train_leave_one_out_mean']:.6f} / "
        f"{nn['train_leave_one_out_median']:.6f} / {nn['train_leave_one_out_p90']:.6f}",
        f"real -> train NN mean/median/p90: {nn['real_to_train_mean']:.6f} / "
        f"{nn['real_to_train_median']:.6f} / {nn['real_to_train_p90']:.6f}",
        "",
        "## PCA",
        f"PC1/PC2 explained variance: {pca['explained_var_ratio'][0]:.4f} / "
        f"{pca['explained_var_ratio'][1]:.4f}",
        f"domain centroid separation: {pca['domain_centroid_separation']:.4f}",
        f"avg within-radius: {pca['avg_within_radius']:.4f}",
        f"separation / within: {pca['separation_over_within']:.4f}",
        "",
        f"t-SNE: {summary['tsne_status']}",
        "",
        "## Outputs",
    ]
    lines.extend(f"- {p}" for p in summary["outputs"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare real/train VAE latents from websocket logs.")
    parser.add_argument("--real-dir", type=Path, default=PROJECT_ROOT / "logs/ws_client_frame")
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "logs/ws_client_frame_trainsets")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--limit", type=int, default=32, help="Use latest N frames per domain. <=0 means all.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-26_02-36-01/checkpoints/weights/step_020000.pt",
    )
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=PROJECT_ROOT
        / "runs/r1_pro_chassis_uncond_3cam_384_1e-4/2026-05-23_02-39-01/dataset_stats.json",
    )
    parser.add_argument("--task", default="r1_pro_chassis_uncond_3cam_384_1e-4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mixed-precision", default=None, choices=["no", "fp16", "bf16", None])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = None if args.limit is not None and args.limit <= 0 else int(args.limit)

    real_frames = _list_frame_dirs(args.real_dir, limit=limit)
    train_frames = _list_frame_dirs(args.train_dir, limit=limit)
    print(f"real frames={len(real_frames)} train frames={len(train_frames)} out={out_dir}")
    if not real_frames or not train_frames:
        raise RuntimeError("Both domains need at least one model_input.png frame.")

    adapter = FastWAMAdapter(
        task=args.task,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        action_horizon=32,
        num_inference_steps=10,
        device=args.device,
        mixed_precision=args.mixed_precision,
        load_text_encoder=False,
        warmup=False,
    )

    real_feats, real_pooled, real_names, latent_shape = _encode_domain(
        adapter, real_frames, domain="real"
    )
    train_feats, train_pooled, train_names, train_latent_shape = _encode_domain(
        adapter, train_frames, domain="train"
    )
    if train_latent_shape != latent_shape:
        raise RuntimeError(f"latent shape mismatch: real={latent_shape}, train={train_latent_shape}")

    np.savez_compressed(
        out_dir / "vae_latents.npz",
        real_feats=real_feats,
        train_feats=train_feats,
        real_pooled=real_pooled,
        train_pooled=train_pooled,
        real_names=np.asarray(real_names),
        train_names=np.asarray(train_names),
        latent_shape=np.asarray(latent_shape),
    )

    real_stats = _domain_stats("real", real_feats)
    train_stats = _domain_stats("train", train_feats)

    real_norm = np.linalg.norm(real_feats, axis=1)
    train_norm = np.linalg.norm(train_feats, axis=1)
    train_center = train_feats.mean(axis=0)
    real_center = real_feats.mean(axis=0)
    train_to_train_center = np.linalg.norm(train_feats - train_center, axis=1)
    real_to_train_center = np.linalg.norm(real_feats - train_center, axis=1)

    real_to_train_nn, real_to_train_idx = _cosine_nn(real_feats, train_feats)
    train_loo_nn, _ = _cosine_nn(train_feats, train_feats, leave_one_out=True)

    combined = np.vstack([real_feats, train_feats])
    labels = np.asarray(["real"] * len(real_feats) + ["train"] * len(train_feats))
    pca_coords, pca_explained = _pca_2d(combined)
    real_pca = pca_coords[labels == "real"]
    train_pca = pca_coords[labels == "train"]
    pca_sep = float(np.linalg.norm(real_pca.mean(axis=0) - train_pca.mean(axis=0)))
    pca_within = float(
        (
            np.linalg.norm(real_pca - real_pca.mean(axis=0), axis=1).mean()
            + np.linalg.norm(train_pca - train_pca.mean(axis=0), axis=1).mean()
        )
        / 2.0
    )

    tsne_coords, tsne_status = _try_tsne(pca_coords)

    outputs: list[str] = []
    plot_paths = {
        "norm_hist": out_dir / "latent_norm_hist.png",
        "pca": out_dir / "latent_pca.png",
        "nn_hist": out_dir / "latent_nn_distance_hist.png",
        "channel_delta": out_dir / "latent_channel_delta.png",
    }
    _plot_norm_hist(real_norm, train_norm, plot_paths["norm_hist"])
    _plot_scatter(
        pca_coords,
        labels,
        plot_paths["pca"],
        title="VAE Latent PCA",
        xlabel=f"PC1 ({pca_explained[0] * 100:.1f}% var)",
        ylabel=f"PC2 ({pca_explained[1] * 100:.1f}% var)",
    )
    _plot_nn_hist(train_loo_nn, real_to_train_nn, plot_paths["nn_hist"])
    channel_delta = real_pooled.mean(axis=0) - train_pooled.mean(axis=0)
    _plot_pooled_channel_delta(channel_delta, plot_paths["channel_delta"])
    outputs.extend(str(p) for p in plot_paths.values())

    if tsne_coords is not None:
        tsne_path = out_dir / "latent_tsne.png"
        _plot_scatter(
            tsne_coords,
            labels,
            tsne_path,
            title="VAE Latent t-SNE",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        outputs.append(str(tsne_path))

    nn_csv = out_dir / "real_to_train_nearest_neighbors.csv"
    _write_nn_csv(nn_csv, real_names, train_names, real_to_train_nn, real_to_train_idx)
    outputs.append(str(nn_csv))

    summary_json = out_dir / "summary.json"
    summary_txt = out_dir / "summary.txt"
    latent_npz = out_dir / "vae_latents.npz"
    outputs.extend([str(summary_json), str(summary_txt), str(latent_npz)])

    summary = {
        "real_dir": str(args.real_dir),
        "train_dir": str(args.train_dir),
        "checkpoint": str(args.checkpoint),
        "latent_shape": list(latent_shape),
        "domain_stats": {
            "real": asdict(real_stats),
            "train": asdict(train_stats),
        },
        "centroid": {
            "l2_real_train": float(np.linalg.norm(real_center - train_center)),
            "train_to_train_center_mean": float(train_to_train_center.mean()),
            "train_to_train_center_std": float(train_to_train_center.std()),
            "real_to_train_center_mean": float(real_to_train_center.mean()),
            "real_to_train_center_std": float(real_to_train_center.std()),
        },
        "nearest_neighbor": {
            "train_leave_one_out_mean": float(train_loo_nn.mean()),
            "train_leave_one_out_median": float(np.median(train_loo_nn)),
            "train_leave_one_out_p90": float(np.percentile(train_loo_nn, 90)),
            "real_to_train_mean": float(real_to_train_nn.mean()),
            "real_to_train_median": float(np.median(real_to_train_nn)),
            "real_to_train_p90": float(np.percentile(real_to_train_nn, 90)),
        },
        "pca": {
            "explained_var_ratio": [float(pca_explained[0]), float(pca_explained[1])],
            "domain_centroid_separation": pca_sep,
            "avg_within_radius": pca_within,
            "separation_over_within": float(pca_sep / (pca_within + 1e-12)),
        },
        "pooled_channel": {
            "real_mean": real_pooled.mean(axis=0).astype(float).tolist(),
            "train_mean": train_pooled.mean(axis=0).astype(float).tolist(),
            "delta_real_minus_train": channel_delta.astype(float).tolist(),
            "real_std": real_pooled.std(axis=0).astype(float).tolist(),
            "train_std": train_pooled.std(axis=0).astype(float).tolist(),
        },
        "tsne_status": tsne_status,
        "outputs": outputs,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_txt(summary_txt, summary)

    print("\n=== summary ===")
    print(summary_txt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
