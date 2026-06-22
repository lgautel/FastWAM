#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DimensionPair:
    index: int
    state_col: str | None
    action_col: str | None


def _indexed_columns(df: pd.DataFrame, prefix: str) -> dict[int, str]:
    columns: dict[int, str] = {}
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        if suffix.isdigit():
            columns[int(suffix)] = col
    return columns


def collect_dimension_pairs(df: pd.DataFrame) -> list[DimensionPair]:
    state_cols = _indexed_columns(df, "state_")
    action_cols = _indexed_columns(df, "action_")
    indices = sorted(set(state_cols) | set(action_cols))
    if not indices:
        raise ValueError("CSV does not contain state_N or action_N columns")
    return [
        DimensionPair(
            index=index,
            state_col=state_cols.get(index),
            action_col=action_cols.get(index),
        )
        for index in indices
    ]


def choose_x_values(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "request_id" in df.columns and "chunk_offset" in df.columns:
        if df["request_id"].nunique(dropna=False) == 1:
            return df["chunk_offset"].to_numpy(), "chunk_offset"
    return np.arange(len(df)), "row"


def plot_state_action_by_dimension(
    df: pd.DataFrame,
    out_path: Path,
    *,
    cols: int = 3,
    title: str | None = None,
    show: bool = False,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = collect_dimension_pairs(df)
    x_values, x_label = choose_x_values(df)

    cols = max(1, cols)
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 2.6 * rows), sharex=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    for ax, pair in zip(axes_array, pairs):
        if pair.state_col is not None:
            ax.plot(x_values, df[pair.state_col], label=pair.state_col, linewidth=1.2)
        if pair.action_col is not None:
            ax.plot(
                x_values,
                df[pair.action_col],
                label=pair.action_col,
                linewidth=1.2,
                linestyle="--",
            )
        ax.set_title(f"dim {pair.index}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="x-small", loc="best")

    for ax in axes_array[len(pairs):]:
        ax.axis("off")

    for ax in axes_array[-cols:]:
        ax.set_xlabel(x_label)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ws_state_action.csv as one subplot per state/action dimension.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "logs" / "ws_state_action.csv",
        help="Input CSV path. Defaults to logs/ws_state_action.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "logs" / "ws_state_action_by_dim.png",
        help="Output PNG path. Defaults to logs/ws_state_action_by_dim.png.",
    )
    parser.add_argument(
        "--request-id",
        type=int,
        default=None,
        help="Only plot rows for this request_id.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of subplot columns in the output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.request_id is not None:
        if "request_id" not in df.columns:
            raise ValueError("--request-id requires a request_id column")
        df = df[df["request_id"] == args.request_id].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows found for request_id={args.request_id}")

    title = f"{args.csv.name}"
    if args.request_id is not None:
        title += f" request_id={args.request_id}"
    plot_state_action_by_dimension(
        df,
        args.out,
        cols=args.cols,
        title=title,
        show=args.show,
    )
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
