# %%
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_TAG = "OG_ESM2_t36_3B"
MODEL_TAG = "ESM2-HA80"
OUTPUT_SELECTOR = "full_H3N2_max10"
TARGET_LINEAGE = "K"
PANEL_MODEL_NAME = "full_plus_plm_mut_interaction"
USE_LOG_RESPONSE = False
PLOT_LOG_AXES = True
MIN_DEPTH = 1.0
PSEUDOCOUNT = 1e-6
HEXBIN_GRIDSIZE = 40
HEXBIN_BINS = "log"
HIST_BINS = 35

LINEAGE_PANEL_DIR = PROJECT_ROOT / "Results" / "test" / "lineage_panel_mutability_vs_plm" / "gisaidinc"
DMS_DIR = PROJECT_ROOT / "Results" / "DMS_investigation"
OUTDIR = PROJECT_ROOT / "Results" / "test" / "lineage_nested_glm"
os.makedirs(OUTDIR, exist_ok=True)


def infer_dms_file(model_tag: str) -> Path:
    if "HA80" in model_tag:
        filename = "ESM2-HA80_DMS_with_probabilities_grammar.csv"
    else:
        filename = "ESM2-H3_DMS_with_probabilities_grammar.csv"
    return DMS_DIR / filename


def load_lineage_combined(model_tag: str, output_selector: str) -> pd.DataFrame:
    path = (
        LINEAGE_PANEL_DIR
        / model_tag
        / f"lineage_combined_long_table_{output_selector}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Lineage combined table not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {
        "lineage", "position", "ref_aa", "aa", "plm_prob", "mut_prob", "obs_freq", "depth"
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_dms_features(dms_path: Path) -> pd.DataFrame:
    dms = pd.DataFrame(pd.read_csv(dms_path))
    required = {"canonical_mutation"}
    missing = required.difference(dms.columns)
    if missing:
        raise ValueError(f"Missing required DMS columns in {dms_path}: {sorted(missing)}")

    rename_map = {
        "MDCKSIAT1 cell entry": "dms_mdck_cell_entry",
        "sera escape": "dms_sera_escape",
        "pH stability": "dms_ph_stability",
        "semantic_score": "dms_semantic_score",
        "relative_grammaticality": "dms_relative_grammaticality",
    }
    present = [col for col in rename_map if col in dms.columns]
    dms = pd.DataFrame(dms.loc[:, ["canonical_mutation", *present]].copy())
    dms.columns = [rename_map.get(col, col) for col in dms.columns]
    return pd.DataFrame(dms)


def build_modeling_table(combined_df: pd.DataFrame, dms_df: pd.DataFrame) -> pd.DataFrame:
    df = combined_df.copy()

    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["position"]).copy()
    df["position"] = df["position"].astype(int)

    df["canonical_mutation"] = (
        df["ref_aa"].astype(str)
        + df["position"].astype(str)
        + df["aa"].astype(str)
    )

    df = df.merge(dms_df, on="canonical_mutation", how="left")

    df["plm_prob"] = pd.to_numeric(df["plm_prob"], errors="coerce")
    df["mut_prob"] = pd.to_numeric(df["mut_prob"], errors="coerce")
    df["obs_freq"] = pd.to_numeric(df["obs_freq"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")

    df = df.dropna(subset=["plm_prob", "mut_prob", "obs_freq", "depth"]).copy()
    df = df[df["depth"] >= MIN_DEPTH].copy()

    df["log_plm"] = np.log10(df["plm_prob"].clip(lower=PSEUDOCOUNT))
    df["log_mut"] = np.log10(df["mut_prob"].clip(lower=PSEUDOCOUNT))

    obs_count = np.rint(df["obs_freq"] * df["depth"]).astype(float)
    obs_count = np.clip(obs_count, 0.0, df["depth"])

    df["obs_count"] = obs_count
    df["obs_prop"] = np.where(df["depth"] > 0, df["obs_count"] / df["depth"], np.nan)
    df = df.dropna(subset=["obs_prop"]).copy()
    df["log_obs_prop"] = np.log10(df["obs_prop"].clip(lower=PSEUDOCOUNT))

    return df


def filter_to_target_lineage(combined_df: pd.DataFrame, target_lineage: str) -> pd.DataFrame:
    combined_df = pd.DataFrame(combined_df)
    available_lineages = sorted(combined_df["lineage"].dropna().astype(str).unique().tolist())
    if target_lineage not in available_lineages:
        raise ValueError(
            f"Target lineage '{target_lineage}' not found. Available lineages: {available_lineages}"
        )
    filtered = combined_df[combined_df["lineage"].astype(str).eq(target_lineage)].copy()
    return pd.DataFrame(filtered)


def fit_logscale_model(data: pd.DataFrame, model_name: str, rhs_terms: List[str]) -> Tuple[Dict, pd.DataFrame]:
    response_col = "log_obs_prop" if USE_LOG_RESPONSE else "obs_prop"
    response_scale = "log" if USE_LOG_RESPONSE else "raw"
    required_numeric = [response_col, "depth", "obs_prop"]
    raw_term_cols = [
        term for term in rhs_terms
        if ":" not in term
    ]
    fit_cols = list(dict.fromkeys([*required_numeric, *raw_term_cols]))

    fit_df = data[fit_cols].copy()
    fit_df[response_col] = pd.to_numeric(fit_df[response_col], errors="coerce")
    fit_df["depth"] = pd.to_numeric(fit_df["depth"], errors="coerce")
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if fit_df.empty:
        raise ValueError(f"No rows available for model {model_name} after filtering.")

    rhs = " + ".join(rhs_terms) if len(rhs_terms) > 0 else "1"
    formula = f"{response_col} ~ {rhs}"

    model = smf.wls(
        formula=formula,
        data=fit_df,
        weights=fit_df["depth"],
    )
    result = model.fit()

    pred_response = np.asarray(result.predict(fit_df), dtype=float)
    obs_response = fit_df[response_col].to_numpy(dtype=float)
    if USE_LOG_RESPONSE:
        pred = np.power(10.0, pred_response)
    else:
        pred = pred_response
    pred = np.clip(pred, PSEUDOCOUNT, 1.0)
    obs = fit_df["obs_prop"].to_numpy(dtype=float)
    w = fit_df["depth"].to_numpy(dtype=float)

    weighted_rmse = float(np.sqrt(np.average((obs - pred) ** 2, weights=w))) if np.sum(w) > 0 else np.nan
    weighted_mae = float(np.average(np.abs(obs - pred), weights=w)) if np.sum(w) > 0 else np.nan
    weighted_r2_response = _weighted_r2(obs_response, pred_response, w)

    metrics = {
        "model": model_name,
        "formula": formula,
        "response_col": response_col,
        "response_scale": response_scale,
        "n_rows": int(len(fit_df)),
        "lineage": TARGET_LINEAGE,
        "df_model": float(result.df_model),
        "aic": float(result.aic),
        "bic": float(result.bic) if np.isfinite(result.bic) else np.nan,
        "log_likelihood": float(result.llf),
        "weighted_r2_response": weighted_r2_response,
        "weighted_rmse": weighted_rmse,
        "weighted_mae": weighted_mae,
    }

    coef_df = pd.DataFrame(
        {
            "model": model_name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues,
            "p_value": result.pvalues,
        }
    )

    return metrics, coef_df


def _weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    w = np.asarray(weights, dtype=float)
    if np.sum(w) <= 0:
        return np.nan
    y_bar = np.average(y_true, weights=w)
    ss_res = np.sum(w * np.square(y_true - y_pred))
    ss_tot = np.sum(w * np.square(y_true - y_bar))
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - (ss_res / ss_tot))


def _main_axis_limits() -> Tuple[float, float]:
    if PLOT_LOG_AXES:
        return PSEUDOCOUNT * 0.3, 1.2
    return -0.02, 1.02


def _add_marginal_hists(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_zero_count: int,
    y_zero_count: int,
) -> None:
    divider = make_axes_locatable(ax)
    hist_x = divider.append_axes("bottom", size="23%", pad=0.08, sharex=ax)
    hist_y = divider.append_axes("right", size="23%", pad=0.08, sharey=ax)

    if PLOT_LOG_AXES:
        x_bins = np.logspace(np.log10(PSEUDOCOUNT), 0.0, HIST_BINS)
        y_bins = np.logspace(np.log10(PSEUDOCOUNT), 0.0, HIST_BINS)
        hist_x.set_xscale("log")
        hist_y.set_yscale("log")
    else:
        x_bins = np.linspace(0.0, 1.0, HIST_BINS)
        y_bins = np.linspace(0.0, 1.0, HIST_BINS)

    hist_x.hist(x_values, bins=x_bins, color="gray", alpha=0.7)
    hist_x.set_yscale("log")
    hist_x.set_ylabel("count")
    hist_x.set_title(f"x=0: {int(x_zero_count)}", fontsize=8, loc="right")
    hist_x.grid(alpha=0.2)

    hist_y.hist(y_values, bins=y_bins, orientation="horizontal", color="gray", alpha=0.7)
    hist_y.set_xscale("log")
    hist_y.set_xlabel("count")
    hist_y.set_title(f"y=0: {int(y_zero_count)}", fontsize=8)
    hist_y.grid(alpha=0.2)


def fit_and_predict_lineage(
    lineage_df: pd.DataFrame,
    rhs_terms: List[str],
    lineage_name: str,
) -> Tuple[pd.DataFrame, Dict]:
    response_col = "log_obs_prop" if USE_LOG_RESPONSE else "obs_prop"
    raw_term_cols = [term for term in rhs_terms if ":" not in term]
    fit_cols = list(dict.fromkeys(["obs_prop", "log_obs_prop", "depth", response_col, *raw_term_cols]))
    fit_df = lineage_df[fit_cols].copy().replace([np.inf, -np.inf], np.nan).dropna().copy()
    fit_df[response_col] = pd.to_numeric(fit_df[response_col], errors="coerce")
    fit_df["depth"] = pd.to_numeric(fit_df["depth"], errors="coerce")
    fit_df = fit_df[fit_df[response_col].notna() & fit_df["depth"].notna()].copy()

    if fit_df.empty:
        return pd.DataFrame(), {
            "lineage": lineage_name,
            "n_rows": 0,
            "r2_weighted_response": np.nan,
            "aic": np.nan,
            "response_scale": "log" if USE_LOG_RESPONSE else "raw",
            "formula": "",
        }

    rhs = " + ".join(rhs_terms) if len(rhs_terms) > 0 else "1"
    formula = f"{response_col} ~ {rhs}"
    model = smf.wls(
        formula=formula,
        data=fit_df,
        weights=fit_df["depth"],
    )
    result = model.fit()

    pred_response = np.asarray(result.predict(fit_df), dtype=float)
    if USE_LOG_RESPONSE:
        pred = np.power(10.0, pred_response)
    else:
        pred = pred_response
    pred = np.clip(pred, PSEUDOCOUNT, 1.0)
    obs = fit_df["obs_prop"].to_numpy(dtype=float)
    obs_response = fit_df[response_col].to_numpy(dtype=float)
    w = fit_df["depth"].to_numpy(dtype=float)
    r2_weighted = _weighted_r2(obs_response, pred_response, w)

    pred_df = pd.DataFrame(fit_df.copy())
    pred_df["pred_obs_prop"] = pred
    pred_df["pred_response"] = pred_response
    pred_df["lineage"] = lineage_name

    summary = {
        "lineage": lineage_name,
        "n_rows": int(len(fit_df)),
        "r2_weighted_response": r2_weighted,
        "aic": float(result.aic),
        "response_scale": "log" if USE_LOG_RESPONSE else "raw",
        "formula": formula,
    }
    return pd.DataFrame(pred_df), summary


def make_cross_lineage_allele_frequency_panel(
    modeling_df: pd.DataFrame,
    outdir: Path,
    tag: str,
) -> None:
    lineage_names = sorted(modeling_df["lineage"].dropna().astype(str).unique().tolist())
    if len(lineage_names) < 2:
        print("Not enough lineages for cross-lineage allele frequency panel.")
        return

    value_df = modeling_df[["lineage", "canonical_mutation", "obs_prop"]].copy()
    value_df["obs_prop"] = value_df["obs_prop"].clip(lower=PSEUDOCOUNT)
    pivot_df = value_df.pivot_table(index="canonical_mutation", columns="lineage", values="obs_prop", aggfunc="mean")

    pairs = []
    for i in range(len(lineage_names)):
        for j in range(i + 1, len(lineage_names)):
            pairs.append((lineage_names[i], lineage_names[j]))

    n = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    summary_rows = []
    for idx, (lin_a, lin_b) in enumerate(pairs):
        ax = axes_flat[idx]
        sub = pivot_df[[lin_a, lin_b]].dropna().copy()
        if sub.empty:
            ax.set_title(f"{lin_a} vs {lin_b} | no overlap")
            ax.axis("off")
            summary_rows.append(
                {
                    "lineage_a": lin_a,
                    "lineage_b": lin_b,
                    "n_overlap": 0,
                    "zero_zero_count": 0,
                    "x_zero_count": 0,
                    "y_zero_count": 0,
                    "pearson_r": np.nan,
                    "pearson_r2": np.nan,
                    "spearman_r": np.nan,
                    "spearman_r2": np.nan,
                }
            )
            continue

        x_raw = sub[lin_a].to_numpy(dtype=float)
        y_raw = sub[lin_b].to_numpy(dtype=float)
        x_plot = np.clip(x_raw, PSEUDOCOUNT, 1.0)
        y_plot = np.clip(y_raw, PSEUDOCOUNT, 1.0)

        if PLOT_LOG_AXES:
            x_corr = np.log10(x_plot)
            y_corr = np.log10(y_plot)
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            x_corr = x_plot
            y_corr = y_plot

        if len(x_corr) > 1 and np.std(x_corr) > 0 and np.std(y_corr) > 0:
            pearson_r_val = float(pearsonr(x_corr, y_corr)[0])
            spearman_r_val = float(spearmanr(x_corr, y_corr)[0])
        else:
            pearson_r_val = np.nan
            spearman_r_val = np.nan

        pearson_r2 = float(np.square(pearson_r_val)) if np.isfinite(pearson_r_val) else np.nan
        spearman_r2 = float(np.square(spearman_r_val)) if np.isfinite(spearman_r_val) else np.nan

        zero_zero_count = int(np.sum((x_raw <= PSEUDOCOUNT) & (y_raw <= PSEUDOCOUNT)))
        x_zero_count = int(np.sum(x_raw <= PSEUDOCOUNT))
        y_zero_count = int(np.sum(y_raw <= PSEUDOCOUNT))

        ax.scatter(x_plot, y_plot, s=10, alpha=0.3, color="tab:blue", edgecolors="none")
        lim_low = min(np.min(x_plot), np.min(y_plot))
        lim_high = max(np.max(x_plot), np.max(y_plot))
        ax.plot([lim_low, lim_high], [lim_low, lim_high], linestyle="--", linewidth=1)
        ax.set_title(
            f"{lin_a} vs {lin_b} | n={len(sub)}\n"
            f"Pearson R²={pearson_r2:.3f} | Spearman R²={spearman_r2:.3f} "
        )
        scale_label = "log" if PLOT_LOG_AXES else "linear"
        ax.set_xlabel(f"obs freq ({lin_a}, {scale_label})")
        ax.set_ylabel(f"obs freq ({lin_b}, {scale_label})")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"(0,0): {zero_zero_count}\nx=0: {x_zero_count}\ny=0: {y_zero_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        lower_lim, upper_lim = _main_axis_limits()
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        _add_marginal_hists(ax, x_plot, y_plot, x_zero_count, y_zero_count)

        summary_rows.append(
            {
                "lineage_a": lin_a,
                "lineage_b": lin_b,
                "n_overlap": int(len(sub)),
                "zero_zero_count": zero_zero_count,
                "x_zero_count": x_zero_count,
                "y_zero_count": y_zero_count,
                "pearson_r": pearson_r_val,
                "pearson_r2": pearson_r2,
                "spearman_r": spearman_r_val,
                "spearman_r2": spearman_r2,
            }
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Observed allele-frequency agreement across lineages ({'log' if PLOT_LOG_AXES else 'linear'} axis)\n{MODEL_TAG} | {OUTPUT_SELECTOR}",
        y=1.02,
    )
    plt.tight_layout()
    fig_path = outdir / f"cross_lineage_obs_freq_panel_{tag}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / f"cross_lineage_obs_freq_panel_summary_{tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    pivot_path = outdir / f"cross_lineage_obs_freq_matrix_{tag}.csv"
    pivot_df.to_csv(pivot_path)

    print(f"Saved cross-lineage panel figure: {fig_path}")
    print(f"Saved cross-lineage summary: {summary_path}")
    print(f"Saved cross-lineage matrix: {pivot_path}")


def make_lineage_prediction_panel(
    modeling_df: pd.DataFrame,
    model_specs: List[Dict[str, List[str]]],
    panel_model_name: str,
    outdir: Path,
    tag: str,
) -> None:
    spec_map = {spec["name"]: spec["rhs"] for spec in model_specs}
    if panel_model_name not in spec_map:
        raise ValueError(
            f"panel_model_name '{panel_model_name}' not in available model specs: {sorted(spec_map.keys())}"
        )

    rhs_terms = spec_map[panel_model_name]
    lineage_names = sorted(modeling_df["lineage"].dropna().astype(str).unique().tolist())

    all_pred_frames = []
    panel_rows = []
    for lineage_name in lineage_names:
        lineage_df = modeling_df[modeling_df["lineage"].astype(str).eq(lineage_name)].copy()
        pred_df, row = fit_and_predict_lineage(lineage_df, rhs_terms, lineage_name)
        panel_rows.append(row)
        if not pred_df.empty:
            all_pred_frames.append(pred_df)

    panel_summary_df = pd.DataFrame(panel_rows)
    panel_summary_path = outdir / f"lineage_prediction_panel_summary_{tag}.csv"
    panel_summary_df.to_csv(panel_summary_path, index=False)

    if len(all_pred_frames) == 0:
        print("No lineage prediction frames were generated for panel plot.")
        return

    pred_all_df = pd.concat(all_pred_frames, ignore_index=True)
    pred_all_path = outdir / f"lineage_prediction_panel_points_{tag}.csv"
    pred_all_df.to_csv(pred_all_path, index=False)

    plotted_lineages = panel_summary_df[panel_summary_df["n_rows"] > 0]["lineage"].tolist()
    n = len(plotted_lineages)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, lineage_name in enumerate(plotted_lineages):
        ax = axes_flat[idx]
        sub = pred_all_df[pred_all_df["lineage"] == lineage_name]
        x_raw = sub["obs_prop"].to_numpy(dtype=float)
        y_raw = sub["pred_obs_prop"].to_numpy(dtype=float)
        x_plot = np.clip(x_raw, PSEUDOCOUNT, 1.0)
        y_plot = np.clip(y_raw, PSEUDOCOUNT, 1.0)
        w = sub["depth"].to_numpy(dtype=float)
        point_size = np.clip(np.sqrt(np.clip(w, 1.0, None)), 6.0, 24.0) if len(w) > 0 else 8.0
        ax.scatter(x_plot, y_plot, s=point_size, alpha=0.3, color="tab:blue", edgecolors="none")
        ax.plot([PSEUDOCOUNT, 1.0], [PSEUDOCOUNT, 1.0], linestyle="--", linewidth=1)
        if PLOT_LOG_AXES:
            ax.set_xscale("log")
            ax.set_yscale("log")

        if PLOT_LOG_AXES:
            x_corr = np.log10(x_plot)
            y_corr = np.log10(y_plot)
        else:
            x_corr = x_plot
            y_corr = y_plot

        if len(x_corr) > 1 and np.std(x_corr) > 0 and np.std(y_corr) > 0:
            pearson_r2 = float(np.square(float(pearsonr(x_corr, y_corr)[0])))
            spearman_r2 = float(np.square(float(spearmanr(x_corr, y_corr)[0])))
        else:
            pearson_r2 = np.nan
            spearman_r2 = np.nan

        zero_zero_count = int(np.sum((x_raw <= PSEUDOCOUNT) & (y_raw <= PSEUDOCOUNT)))
        x_zero_count = int(np.sum(x_raw <= PSEUDOCOUNT))
        y_zero_count = int(np.sum(y_raw <= PSEUDOCOUNT))

        row = panel_summary_df[panel_summary_df["lineage"] == lineage_name].iloc[0]
        r2 = row["r2_weighted_response"]
        n_rows = int(row["n_rows"])
        ax.set_title(
            f"{lineage_name} | response-R²={r2:.3f} | n={n_rows} \n "
            f"Pearson R²={pearson_r2:.3f} | Spearman R²={spearman_r2:.3f} "
        )
        scale_label = "log" if PLOT_LOG_AXES else "linear"
        ax.set_xlabel(f"Observed allele frequency ({scale_label})")
        ax.set_ylabel(f"Predicted allele frequency ({scale_label})")
        lower_lim, upper_lim = _main_axis_limits()
        ax.set_xlim(lower_lim, upper_lim)
        ax.set_ylim(lower_lim, upper_lim)
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"(0,0): {zero_zero_count}\nx=0: {x_zero_count}\ny=0: {y_zero_count}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        _add_marginal_hists(ax, x_plot, y_plot, x_zero_count, y_zero_count)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Predicted vs observed allele frequency by lineage\n"
        f"Model: {panel_model_name} | response={'log' if USE_LOG_RESPONSE else 'raw'} | {MODEL_TAG} | {OUTPUT_SELECTOR}",
        y=1.02,
    )
    plt.tight_layout()
    fig_path = outdir / f"lineage_prediction_panel_{panel_model_name}_{tag}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved lineage panel figure: {fig_path}")
    print(f"Saved lineage panel summary: {panel_summary_path}")
    print(f"Saved lineage panel points: {pred_all_path}")


def main():
    response_scale_tag = "logresp" if USE_LOG_RESPONSE else "rawresp"
    axis_scale_tag = "logaxes" if PLOT_LOG_AXES else "linaxes"

    dms_path = infer_dms_file(MODEL_TAG)
    combined_all = load_lineage_combined(MODEL_TAG, OUTPUT_SELECTOR)
    combined = filter_to_target_lineage(combined_all, TARGET_LINEAGE)
    dms = load_dms_features(dms_path)
    modeling_df = build_modeling_table(combined, dms)
    modeling_df_all = build_modeling_table(combined_all, dms)

    dms_feature_terms = [
        "dms_mdck_cell_entry",
        "dms_sera_escape",
        "dms_ph_stability",
        "dms_semantic_score",
        "dms_relative_grammaticality",
    ]
    dms_feature_terms = [term for term in dms_feature_terms if term in modeling_df.columns]

    model_specs = [
        {"name": "baseline", "rhs": []},
        {"name": "dms_only", "rhs": [*dms_feature_terms]},
        {"name": "plm_only", "rhs": ["log_plm"]},
        {"name": "mut_only", "rhs": ["log_mut"]},
        {"name": "plm_plus_mut", "rhs": ["log_plm", "log_mut"]},
        {"name": "dms_plus_plm", "rhs": [*dms_feature_terms, "log_plm"]},
        {"name": "dms_plus_mut", "rhs": [*dms_feature_terms, "log_mut"]},
        {"name": "full_additive", "rhs": [*dms_feature_terms, "log_plm", "log_mut"]},
        {
            "name": "full_plus_plm_mut_interaction",
            "rhs": [*dms_feature_terms, "log_plm", "log_mut", "log_plm:log_mut"],
        },
    ]

    all_metrics = []
    all_coefs = []

    for spec in model_specs:
        metrics, coef_df = fit_logscale_model(modeling_df, spec["name"], spec["rhs"])
        all_metrics.append(metrics)
        all_coefs.append(coef_df)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("aic", ascending=True).reset_index(drop=True)

    best_aic = metrics_df["aic"].min()
    baseline_aic = metrics_df.loc[metrics_df["model"] == "baseline", "aic"]
    baseline_aic = float(baseline_aic.iloc[0]) if len(baseline_aic) > 0 else np.nan

    metrics_df["delta_aic_vs_best"] = metrics_df["aic"] - best_aic
    metrics_df["delta_aic_vs_baseline"] = metrics_df["aic"] - baseline_aic if np.isfinite(baseline_aic) else np.nan

    coef_df = pd.concat(all_coefs, ignore_index=True)

    tag = f"{MODEL_TAG}_{TARGET_LINEAGE}_{OUTPUT_SELECTOR}_{response_scale_tag}_{axis_scale_tag}"
    metrics_path = OUTDIR / f"nested_glm_model_comparison_{tag}.csv"
    coefs_path = OUTDIR / f"nested_glm_coefficients_{tag}.csv"
    data_path = OUTDIR / f"nested_glm_modeling_table_{tag}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    modeling_df.to_csv(data_path, index=False)

    print("Nested GLM run complete.")
    print(f"Target lineage: {TARGET_LINEAGE}")
    print(f"Response scale toggle USE_LOG_RESPONSE={USE_LOG_RESPONSE}")
    print(f"Model comparison saved: {metrics_path}")
    print(f"Coefficients saved: {coefs_path}")
    print(f"Modeling table saved: {data_path}")

    print("\nTop models by AIC:")
    print(metrics_df[["model", "aic", "delta_aic_vs_best", "weighted_r2_response", "weighted_rmse"]].head(20).to_string(index=False))

    make_lineage_prediction_panel(
        modeling_df=modeling_df_all,
        model_specs=model_specs,
        panel_model_name=PANEL_MODEL_NAME,
        outdir=OUTDIR,
        tag=f"{MODEL_TAG}_{OUTPUT_SELECTOR}_{response_scale_tag}_{axis_scale_tag}",
    )

    make_cross_lineage_allele_frequency_panel(
        modeling_df=modeling_df_all,
        outdir=OUTDIR,
        tag=f"{MODEL_TAG}_{OUTPUT_SELECTOR}_{response_scale_tag}_{axis_scale_tag}",
    )


if __name__ == "__main__":
    main()

# %%
print(OUTDIR)
print(LINEAGE_PANEL_DIR)
# %%
