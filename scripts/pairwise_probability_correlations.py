#!/usr/bin/env python3
"""Build pairwise correlation panels across probability-table CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


DEFAULT_METRICS = ("plm_prob", "mut_prob")
JOIN_KEY_PREFERENCES = (
    ("lineage", "position", "ref_aa", "aa"),
    ("position", "ref_aa", "aa"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pairwise scatter/correlation panels from PLM-vs-mutation probability CSV exports."
    )
    parser.add_argument("csv_paths", nargs="+", type=Path, help="Input CSV files to compare.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary tables and figures.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated numeric columns to compare pairwise. Default: plm_prob,mut_prob",
    )
    parser.add_argument(
        "--join-keys",
        default=None,
        help="Optional comma-separated join keys. Defaults to inferred keys such as lineage,position,ref_aa,aa.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Maximum points to scatter per panel after merge.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used when subsampling points.",
    )
    return parser


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def infer_label(df: pd.DataFrame, path: Path) -> str:
    for col in ("model_display_label", "model", "epoch_label"):
        if col in df.columns:
            unique_vals = [str(val) for val in df[col].dropna().unique().tolist()]
            if len(unique_vals) == 1:
                return unique_vals[0]
    return path.stem


def choose_join_keys(dataframes: Sequence[pd.DataFrame], requested_keys: Sequence[str] | None) -> List[str]:
    if requested_keys:
        missing = [key for key in requested_keys if any(key not in df.columns for df in dataframes)]
        if missing:
            raise ValueError(f"Requested join keys are not present in all files: {missing}")
        return list(requested_keys)

    for candidate in JOIN_KEY_PREFERENCES:
        if all(all(key in df.columns for key in candidate) for df in dataframes):
            return list(candidate)

    common_cols = sorted(set.intersection(*(set(df.columns) for df in dataframes)))
    raise ValueError(
        "Could not infer join keys automatically. Common columns were: " + ", ".join(common_cols)
    )


def prepare_dataframe(df: pd.DataFrame, label: str, join_keys: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    missing_metrics = [metric for metric in metrics if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"File for {label!r} is missing metrics: {missing_metrics}")

    work = df[list(join_keys) + list(metrics)].copy()
    if work.duplicated(subset=list(join_keys)).any():
        work = work.groupby(list(join_keys), as_index=False)[list(metrics)].mean()
    return work


def hide_log_minor_ticks(ax) -> None:
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)


def safe_corr(method, x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    if len(x_vals) < 2 or len(y_vals) < 2:
        return np.nan
    if np.nanstd(x_vals) == 0 or np.nanstd(y_vals) == 0:
        return np.nan
    return float(method(x_vals, y_vals)[0])


def build_pairwise_summary(
    prepared: Dict[str, pd.DataFrame],
    labels: Sequence[str],
    join_keys: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        for left_label in labels:
            for right_label in labels:
                left_df = prepared[left_label][list(join_keys) + [metric]].rename(columns={metric: f"{metric}_left"})
                right_df = prepared[right_label][list(join_keys) + [metric]].rename(columns={metric: f"{metric}_right"})
                merged = left_df.merge(right_df, on=list(join_keys), how="inner")
                valid = merged[[f"{metric}_left", f"{metric}_right"]].replace([np.inf, -np.inf], np.nan).dropna()
                x_vals = valid[f"{metric}_left"].to_numpy(dtype=float)
                y_vals = valid[f"{metric}_right"].to_numpy(dtype=float)
                rows.append(
                    {
                        "metric": metric,
                        "left_label": left_label,
                        "right_label": right_label,
                        "n_rows": int(len(valid)),
                        "pearson_r": safe_corr(pearsonr, x_vals, y_vals),
                        "spearman_r": safe_corr(spearmanr, x_vals, y_vals),
                    }
                )
    return pd.DataFrame(rows)


def plot_metric_panel(
    metric: str,
    prepared: Dict[str, pd.DataFrame],
    labels: Sequence[str],
    join_keys: Sequence[str],
    output_dir: Path,
    max_points: int,
    random_seed: int,
) -> None:
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(4.2 * n, 4.0 * n), squeeze=False)

    for row_idx, y_label in enumerate(labels):
        for col_idx, x_label in enumerate(labels):
            ax = axes[row_idx, col_idx]
            left_df = prepared[x_label][list(join_keys) + [metric]].rename(columns={metric: "x_val"})
            right_df = prepared[y_label][list(join_keys) + [metric]].rename(columns={metric: "y_val"})
            merged = left_df.merge(right_df, on=list(join_keys), how="inner")
            merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_val", "y_val"])

            if row_idx == col_idx:
                vals = merged["x_val"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals) > 0:
                    bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 60)
                    ax.hist(vals, bins=bins, color="#4c72b0", alpha=0.85)
                    ax.set_xscale("log")
                    hide_log_minor_ticks(ax)
                ax.set_title(x_label)
                ax.set_ylabel("count")
                continue

            valid = merged[(merged["x_val"] > 0) & (merged["y_val"] > 0)].copy()
            if len(valid) > max_points:
                valid = valid.sample(max_points, random_state=random_seed)

            if not valid.empty:
                x_vals = valid["x_val"].to_numpy(dtype=float)
                y_vals = valid["y_val"].to_numpy(dtype=float)
                pearson_val = safe_corr(pearsonr, x_vals, y_vals)
                spearman_val = safe_corr(spearmanr, x_vals, y_vals)

                ax.scatter(x_vals, y_vals, s=6, alpha=0.18, edgecolors="none")
                ax.set_xscale("log")
                ax.set_yscale("log")
                hide_log_minor_ticks(ax)
                lim_min = min(np.min(x_vals), np.min(y_vals))
                lim_max = max(np.max(x_vals), np.max(y_vals))
                ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="grey", linewidth=1.0)
                ax.text(
                    0.04,
                    0.96,
                    f"n={len(valid)}\nPearson={pearson_val:.3f}\nSpearman={spearman_val:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )

            if row_idx == 0:
                ax.set_title(x_label)
            if col_idx == 0:
                ax.set_ylabel(y_label)
            if row_idx == n - 1:
                ax.set_xlabel(metric)

    fig.suptitle(f"Pairwise correlations for {metric}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / f"pairwise_{metric}_panel.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    metrics = parse_csv_list(args.metrics)
    requested_join_keys = parse_csv_list(args.join_keys) if args.join_keys else None

    loaded_frames: List[pd.DataFrame] = []
    labels: List[str] = []
    for path in args.csv_paths:
        df = pd.read_csv(path)
        loaded_frames.append(df)
        labels.append(infer_label(df, path))

    join_keys = choose_join_keys(loaded_frames, requested_join_keys)
    prepared = {
        label: prepare_dataframe(df, label, join_keys, metrics)
        for label, df in zip(labels, loaded_frames)
    }

    summary_df = build_pairwise_summary(prepared, labels, join_keys, metrics)
    summary_df.to_csv(output_dir / "pairwise_probability_correlations.tsv", sep="\t", index=False)

    for metric in metrics:
        plot_metric_panel(metric, prepared, labels, join_keys, output_dir, args.max_points, args.random_seed)

    with (output_dir / "pairwise_probability_correlations_manifest.txt").open("w", encoding="utf-8") as handle:
        handle.write("join_keys=" + ",".join(join_keys) + "\n")
        for label, path in zip(labels, args.csv_paths):
            handle.write(f"{label}\t{path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())