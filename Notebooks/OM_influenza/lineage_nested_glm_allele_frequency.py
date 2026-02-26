# %%
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_TAG = "ESM2-HA80"
OUTPUT_SELECTOR = "full_H3N2_max10"
TARGET_LINEAGE = "J.2"
MIN_DEPTH = 1.0
PSEUDOCOUNT = 1e-12

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


def fit_binomial_glm(data: pd.DataFrame, model_name: str, rhs_terms: List[str]) -> Tuple[Dict, pd.DataFrame]:
    required_numeric = ["obs_prop", "depth"]
    raw_term_cols = [
        term for term in rhs_terms
        if ":" not in term
    ]
    fit_cols = [*required_numeric, *raw_term_cols]

    fit_df = data[fit_cols].copy()
    fit_df = fit_df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if fit_df.empty:
        raise ValueError(f"No rows available for model {model_name} after filtering.")

    rhs = " + ".join(rhs_terms) if len(rhs_terms) > 0 else "1"
    formula = f"obs_prop ~ {rhs}"

    model = smf.glm(
        formula=formula,
        data=fit_df,
        family=sm.families.Binomial(),
        freq_weights=fit_df["depth"],
    )
    result = model.fit()

    pred = np.asarray(result.predict(fit_df), dtype=float)
    pred = np.clip(pred, 0.0, 1.0)
    obs = fit_df["obs_prop"].to_numpy(dtype=float)
    w = fit_df["depth"].to_numpy(dtype=float)

    weighted_rmse = float(np.sqrt(np.average((obs - pred) ** 2, weights=w))) if np.sum(w) > 0 else np.nan
    weighted_mae = float(np.average(np.abs(obs - pred), weights=w)) if np.sum(w) > 0 else np.nan

    null_dev = float(result.null_deviance) if np.isfinite(result.null_deviance) else np.nan
    dev = float(result.deviance) if np.isfinite(result.deviance) else np.nan
    pseudo_r2 = float(1.0 - (dev / null_dev)) if np.isfinite(null_dev) and null_dev > 0 else np.nan

    metrics = {
        "model": model_name,
        "formula": formula,
        "n_rows": int(len(fit_df)),
        "lineage": TARGET_LINEAGE,
        "df_model": float(result.df_model),
        "aic": float(result.aic),
        "bic": float(result.bic) if np.isfinite(result.bic) else np.nan,
        "log_likelihood": float(result.llf),
        "deviance": dev,
        "null_deviance": null_dev,
        "pseudo_r2_mcFadden_like": pseudo_r2,
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


def main():
    dms_path = infer_dms_file(MODEL_TAG)
    combined = load_lineage_combined(MODEL_TAG, OUTPUT_SELECTOR)
    combined = filter_to_target_lineage(combined, TARGET_LINEAGE)
    dms = load_dms_features(dms_path)
    modeling_df = build_modeling_table(combined, dms)

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
        metrics, coef_df = fit_binomial_glm(modeling_df, spec["name"], spec["rhs"])
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

    tag = f"{MODEL_TAG}_{TARGET_LINEAGE}_{OUTPUT_SELECTOR}"
    metrics_path = OUTDIR / f"nested_glm_model_comparison_{tag}.csv"
    coefs_path = OUTDIR / f"nested_glm_coefficients_{tag}.csv"
    data_path = OUTDIR / f"nested_glm_modeling_table_{tag}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    coef_df.to_csv(coefs_path, index=False)
    modeling_df.to_csv(data_path, index=False)

    print("Nested GLM run complete.")
    print(f"Target lineage: {TARGET_LINEAGE}")
    print(f"Model comparison saved: {metrics_path}")
    print(f"Coefficients saved: {coefs_path}")
    print(f"Modeling table saved: {data_path}")

    print("\nTop models by AIC:")
    print(metrics_df[["model", "aic", "delta_aic_vs_best", "pseudo_r2_mcFadden_like", "weighted_rmse"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

# %%
print(OUTDIR)
