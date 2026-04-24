#!/usr/bin/env python3
"""Run codon-based mutational accessibility and PLM diversity scoring.

This script generalizes the notebook workflows for influenza and SARS-CoV-2 into
one CLI that supports either a single diversity FASTA or a guide file pointing
to many diversity FASTAs and reference sequences.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mutation-accessibility plus PLM scoring for viral diversity panels."
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("SINGLE_FASTA", "MONTHLY_GUIDE"),
        required=True,
        help="Use a single diversity FASTA or a guide file describing many targets.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        help="Focal/reference nucleotide CDS FASTA. Required for SINGLE_FASTA and default fallback for guide rows.",
    )
    parser.add_argument(
        "--diversity-fasta",
        type=Path,
        help="Diversity FASTA for SINGLE_FASTA mode.",
    )
    parser.add_argument(
        "--guide-path",
        type=Path,
        help="CSV/TSV guide with columns month|label, fasta|path, optional reference.",
    )
    parser.add_argument(
        "--label",
        default="population",
        help="Label to use for SINGLE_FASTA mode.",
    )
    parser.add_argument(
        "--mutation-model",
        choices=("SC2", "H1N1", "H3N2"),
        required=True,
        help="Nucleotide mutation model used to build codon accessibility probabilities.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all tables, plots, cached PLM matrices, and the run manifest.",
    )
    parser.add_argument(
        "--expect-protein-diversity",
        action="store_true",
        help="Treat diversity FASTA sequences as protein alignments unless they obviously look nucleotide-like.",
    )
    parser.add_argument(
        "--plm-max-aa-length",
        type=int,
        default=None,
        help="Optional amino-acid truncation length for PLM inference.",
    )
    parser.add_argument(
        "--plm-max-nt-length",
        type=int,
        default=None,
        help="Optional nucleotide truncation length for PLM inference. Codon-aware trimmed.",
    )
    parser.add_argument(
        "--filter-fixed-mutations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude mutations observed at frequency 1.0 in the diversity panel.",
    )
    parser.add_argument(
        "--filter-singleton-mutations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero out or skip mutations seen fewer than --min-obs-count times.",
    )
    parser.add_argument(
        "--skip-low-count-sites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When singleton filtering is enabled, skip those rows instead of zeroing the observed frequency.",
    )
    parser.add_argument(
        "--min-obs-count",
        type=int,
        default=2,
        help="Minimum count retained when --filter-singleton-mutations is enabled.",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=-1.0,
        help="Alpha grid start value.",
    )
    parser.add_argument(
        "--alpha-stop",
        type=float,
        default=1.0,
        help="Alpha grid stop value, inclusive.",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="Alpha grid step size.",
    )
    parser.add_argument(
        "--alpha-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate the alpha grid in parallel when large enough.",
    )
    parser.add_argument(
        "--alpha-sweep-min-grid",
        type=int,
        default=8,
        help="Minimum grid size before alpha sweep parallelism is enabled.",
    )
    parser.add_argument(
        "--alpha-sweep-max-workers",
        type=int,
        default=None,
        help="Optional worker cap for alpha sweep parallelism.",
    )
    parser.add_argument(
        "--scatter-alphas",
        default="-1,0,1",
        help="Comma-separated alpha values used for the per-group scatter grid.",
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=200000,
        help="Maximum points sampled per group for the scatter grid.",
    )
    parser.add_argument(
        "--test-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Limit guide targets and diversity records for smoke testing.",
    )
    parser.add_argument(
        "--test-max-targets",
        type=int,
        default=1,
        help="Maximum guide targets processed in test mode.",
    )
    parser.add_argument(
        "--test-max-records",
        type=int,
        default=5,
        help="Maximum diversity records processed per target in test mode.",
    )

    plm_group = parser.add_argument_group("PLM source")
    plm_source = plm_group.add_mutually_exclusive_group(required=True)
    plm_source.add_argument(
        "--precomputed-plm-path",
        type=Path,
        help="Reuse one existing PLM probability matrix for every target in this run.",
    )
    plm_source.add_argument(
        "--model-tag",
        help="Model label used for output naming when running PLM inference from a checkpoint or raw model.",
    )
    plm_group.add_argument(
        "--base-model",
        help="Base PLM name used with --model-tag, for example esm2_t33_650M_UR50D or esm-c600m.",
    )
    plm_group.add_argument(
        "--model-layer",
        type=int,
        help="Layer index to use for PLM inference.",
    )
    plm_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional fine-tuned checkpoint directory.",
    )
    plm_group.add_argument(
        "--checkpoint-glob",
        default=None,
        help="Optional glob pattern of checkpoint directories to score as separate epochs in one run.",
    )
    plm_group.add_argument(
        "--force-recompute-plm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate PLM matrices even if cached files already exist.",
    )
    plm_group.add_argument(
        "--gpu-required",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA when running checkpoint-backed PLM inference.",
    )
    parser.add_argument(
        "--mutation-baseline-x",
        type=float,
        default=-2.0,
        help="X-axis position used for mutation-probability baseline points on epoch summary plots.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.analysis_mode == "SINGLE_FASTA":
        if args.diversity_fasta is None:
            raise ValueError("--diversity-fasta is required for SINGLE_FASTA mode")
        if args.reference_fasta is None:
            raise ValueError("--reference-fasta is required for SINGLE_FASTA mode")
    if args.analysis_mode == "MONTHLY_GUIDE" and args.guide_path is None:
        raise ValueError("--guide-path is required for MONTHLY_GUIDE mode")

    if args.precomputed_plm_path is None:
        missing = []
        if not args.model_tag:
            missing.append("--model-tag")
        if not args.base_model:
            missing.append("--base-model")
        if args.model_layer is None:
            missing.append("--model-layer")
        if missing:
            raise ValueError(
                "Checkpoint-backed PLM runs require " + ", ".join(missing)
            )

    if args.checkpoint_glob and args.checkpoint_dir:
        raise ValueError("Provide either --checkpoint-dir or --checkpoint-glob, not both")

    if args.alpha_step <= 0:
        raise ValueError("--alpha-step must be > 0")


def parse_alpha_grid(args: argparse.Namespace) -> np.ndarray:
    values = np.arange(args.alpha_start, args.alpha_stop + (args.alpha_step * 0.5), args.alpha_step)
    return np.round(values, 6)


def parse_scatter_alphas(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def normalise_plm_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    first_row = raw_df.iloc[0, :]
    if raw_df.shape[0] > 1 and first_row.apply(lambda value: isinstance(value, str)).any():
        return raw_df.iloc[1:, :].apply(pd.to_numeric, errors="coerce")
    return raw_df.apply(pd.to_numeric, errors="coerce")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_diversity_records(
    fasta_path: Path,
    expect_protein_diversity: bool,
    test_mode: bool,
    test_max_records: int,
):
    from Functions_HuggingFace import _is_probably_nucleotide_sequence

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if test_mode:
        records = records[:test_max_records]
    if not records:
        return [], False

    any_nucleotide = False
    processed = []
    for record in records:
        seq_text = str(record.seq)
        is_nucleotide = _is_probably_nucleotide_sequence(seq_text)
        if is_nucleotide:
            any_nucleotide = True
        if not expect_protein_diversity and is_nucleotide:
            record.seq = record.seq.translate(to_stop=True)
        processed.append(record)
    return processed, any_nucleotide


def save_run_manifest(args: argparse.Namespace, output_dir: Path, target_specs: List[Dict[str, str]]) -> None:
    manifest = {
        "analysis_mode": args.analysis_mode,
        "mutation_model": args.mutation_model,
        "output_dir": str(output_dir),
        "plm_source": "precomputed" if args.precomputed_plm_path else "checkpoint",
        "precomputed_plm_path": str(args.precomputed_plm_path) if args.precomputed_plm_path else None,
        "model_tag": args.model_tag,
        "base_model": args.base_model,
        "model_layer": args.model_layer,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "checkpoint_glob": args.checkpoint_glob,
        "force_recompute_plm": bool(args.force_recompute_plm),
        "plm_max_aa_length": args.plm_max_aa_length,
        "plm_max_nt_length": args.plm_max_nt_length,
        "filter_fixed_migrations": bool(args.filter_fixed_mutations),
        "filter_singleton_mutations": bool(args.filter_singleton_mutations),
        "skip_low_count_sites": bool(args.skip_low_count_sites),
        "min_obs_count": args.min_obs_count,
        "alpha_grid": parse_alpha_grid(args).tolist(),
        "scatter_alphas": parse_scatter_alphas(args.scatter_alphas),
        "targets": target_specs,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def infer_epoch_value(epoch_label: str, fallback_index: int) -> float:
    matches = re.findall(r"(\d+(?:\.\d+)?)", str(epoch_label))
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return float(fallback_index)


def _discover_checkpoint_dirs(checkpoint_root: Path) -> List[Path]:
    if not checkpoint_root.is_dir():
        return []

    child_dirs = [
        child for child in checkpoint_root.iterdir()
        if child.is_dir() and (child / "model.safetensors").exists()
    ]
    if not child_dirs:
        return []

    return sorted(
        child_dirs,
        key=lambda path: (
            path.name == "final_checkpoint",
            infer_epoch_value(path.name, sys.maxsize),
            path.name,
        ),
    )


def build_model_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.precomputed_plm_path is not None:
        label = args.model_tag if args.model_tag else args.precomputed_plm_path.stem
        return [
            {
                "model_tag": label,
                "epoch_label": label,
                "epoch_value": infer_epoch_value(label, 0),
                "checkpoint_dir": None,
                "precomputed_plm_path": args.precomputed_plm_path,
            }
        ]

    if args.checkpoint_glob:
        matched_paths = [Path(path) for path in glob.glob(args.checkpoint_glob)]
        matched_paths = [path for path in matched_paths if path.exists()]
        if not matched_paths:
            raise ValueError(f"No checkpoint paths matched --checkpoint-glob={args.checkpoint_glob!r}")
        ordered = sorted(
            matched_paths,
            key=lambda path: (infer_epoch_value(path.name, 0), path.name),
        )
        specs = []
        for index, checkpoint_path in enumerate(ordered):
            epoch_label = checkpoint_path.name
            specs.append(
                {
                    "model_tag": f"{args.model_tag}_{epoch_label}",
                    "epoch_label": epoch_label,
                    "epoch_value": infer_epoch_value(epoch_label, index),
                    "checkpoint_dir": checkpoint_path,
                    "precomputed_plm_path": None,
                }
            )
        return specs

    if args.checkpoint_dir is not None:
        discovered_paths = _discover_checkpoint_dirs(args.checkpoint_dir)
        if discovered_paths:
            specs = []
            for index, checkpoint_path in enumerate(discovered_paths):
                epoch_label = checkpoint_path.name
                specs.append(
                    {
                        "model_tag": f"{args.model_tag}_{epoch_label}",
                        "epoch_label": epoch_label,
                        "epoch_value": infer_epoch_value(epoch_label, index),
                        "checkpoint_dir": checkpoint_path,
                        "precomputed_plm_path": None,
                    }
                )
            return specs

    epoch_label = args.checkpoint_dir.name if args.checkpoint_dir else args.model_tag
    return [
        {
            "model_tag": args.model_tag,
            "epoch_label": epoch_label,
            "epoch_value": infer_epoch_value(epoch_label, 0),
            "checkpoint_dir": args.checkpoint_dir,
            "precomputed_plm_path": None,
        }
    ]


def build_lineage_cache(args: argparse.Namespace, mutation_tables: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    from Functions_HuggingFace import (
        _load_single_focal_reference,
        _resolve_plm_max_nt_length,
        _safe_label,
        build_reference_to_alignment_column_map,
        compute_lineage_mutation_profile,
        compute_observed_diversity_profile_fast,
        load_analysis_targets,
        _translate_nt_to_protein,
    )

    target_specs = load_analysis_targets(
        analysis_mode=args.analysis_mode,
        guide_path=str(args.guide_path) if args.guide_path else None,
        diversity_fasta=str(args.diversity_fasta) if args.diversity_fasta else None,
        reference_fasta=str(args.reference_fasta) if args.reference_fasta else None,
        default_label=args.label,
        test_mode=args.test_mode,
        test_max_targets=args.test_max_targets,
    )

    lineage_cache: Dict[str, Dict[str, object]] = {}
    max_nt_length = _resolve_plm_max_nt_length(args.plm_max_aa_length, args.plm_max_nt_length)

    for target in target_specs:
        label = target["label"]
        diversity_path = Path(target["diversity_path"])
        reference_path = Path(target["reference_path"])

        if not diversity_path.exists():
            print(f"Skipping {label}: diversity FASTA not found at {diversity_path}")
            continue

        ref_payload = _load_single_focal_reference(str(reference_path), label)
        full_ref_nt = ref_payload["nucleotide"]
        full_ref_protein = ref_payload["protein"]
        if not full_ref_protein:
            print(f"Skipping {label}: reference translation is empty")
            continue

        if max_nt_length is not None and len(full_ref_nt) > max_nt_length:
            plm_ref_nt = full_ref_nt[:max_nt_length]
            plm_ref_protein = _translate_nt_to_protein(plm_ref_nt)
        else:
            plm_ref_protein = full_ref_protein

        coord_map = {i: i for i in range(len(plm_ref_protein))}
        records, any_nucleotide = load_diversity_records(
            diversity_path,
            expect_protein_diversity=args.expect_protein_diversity,
            test_mode=args.test_mode,
            test_max_records=args.test_max_records,
        )
        if not records:
            print(f"Skipping {label}: no diversity records found in {diversity_path}")
            continue

        ref_to_aln_col, aln_len, matched_pairs = build_reference_to_alignment_column_map(
            full_ref_protein,
            records,
            mutation_tables["aa_to_codons"],
            IGNORE_ALIGNMENT_CHARS,
        )
        obs_freq, obs_depth, diversity_stats = compute_observed_diversity_profile_fast(
            records,
            full_ref_protein,
            ref_to_aln_col,
            aln_len,
            mutation_tables["aa_to_codons"],
            IGNORE_ALIGNMENT_CHARS,
        )
        mut_profile = compute_lineage_mutation_profile(
            full_ref_nt,
            full_ref_protein,
            mutation_tables["aa_to_codons"],
            mutation_tables["codon_mutation_df"],
        )

        lineage_cache[label] = {
            "lineage_key": _safe_label(label),
            "records": records,
            "full_ref_protein": full_ref_protein,
            "plm_ref_protein": plm_ref_protein,
            "coord_map": coord_map,
            "mut_profile": mut_profile,
            "obs_freq": obs_freq,
            "obs_depth": obs_depth,
            "alignment_diff_stats": diversity_stats,
            "diversity_path": str(diversity_path),
            "reference_path": str(reference_path),
            "matched_pairs": matched_pairs,
            "any_nucleotide_diversity": any_nucleotide,
        }

    return lineage_cache


def ensure_plm_matrix(
    args: argparse.Namespace,
    model_spec: Dict[str, object],
    lineage_label: str,
    lineage_data: Dict[str, object],
    plm_dir: Path,
    runtime_cache: Dict[Tuple[str, str], Dict[str, object]],
):
    from Functions_HuggingFace import (
        _load_plm_runtime,
        get_mutation_prob_matrix,
        load_plm_probability_matrix,
    )
    import torch

    line_key = lineage_data["lineage_key"]
    model_tag = str(model_spec["model_tag"])
    plm_cache_path = plm_dir / f"{line_key}_{model_tag}_plm_probability_profile.csv"

    if model_spec["precomputed_plm_path"] is not None:
        raw_matrix = load_plm_probability_matrix(str(model_spec["precomputed_plm_path"]))
        return normalise_plm_matrix(raw_matrix), str(model_spec["precomputed_plm_path"])

    if plm_cache_path.exists() and not args.force_recompute_plm:
        raw_matrix = load_plm_probability_matrix(str(plm_cache_path))
        return normalise_plm_matrix(raw_matrix), str(plm_cache_path)

    cache_key = (model_tag, lineage_data["plm_ref_protein"])
    plm_payload = runtime_cache.get(cache_key)
    if plm_payload is None:
        runtime = runtime_cache.get(("__runtime__", model_tag))
        if runtime is None:
            if args.gpu_required and not torch.cuda.is_available():
                raise RuntimeError("--gpu-required was set but CUDA is unavailable for PLM inference")
            model, device, batch_converter, alphabet = _load_plm_runtime(
                args.base_model,
                checkpoint_dir=str(model_spec["checkpoint_dir"]) if model_spec["checkpoint_dir"] else None,
            )
            runtime = {
                "model": model,
                "device": device,
                "batch_converter": batch_converter,
                "alphabet": alphabet,
            }
            runtime_cache[("__runtime__", model_tag)] = runtime
        plm_payload = get_mutation_prob_matrix(
            reference_protein=lineage_data["plm_ref_protein"],
            model=runtime["model"],
            model_layers=args.model_layer,
            device=runtime["device"],
            batch_converter=runtime["batch_converter"],
            alphabet=runtime["alphabet"],
        )
        runtime_cache[cache_key] = plm_payload

    sequence_row = pd.DataFrame(
        [list(plm_payload["sequence"])],
        index=["sequence"],
        columns=plm_payload["positions"],
    )
    probability_rows = pd.DataFrame(
        plm_payload["mutation_matrix"],
        index=plm_payload["amino_acids"],
        columns=plm_payload["positions"],
    )
    pd.concat([sequence_row, probability_rows], axis=0).to_csv(plm_cache_path, header=False)
    return probability_rows, str(plm_cache_path)


def build_combined_rows(
    args: argparse.Namespace,
    model_spec: Dict[str, object],
    lineage_label: str,
    lineage_data: Dict[str, object],
    plm_matrix: pd.DataFrame,
) -> List[Dict[str, object]]:
    combined_rows: List[Dict[str, object]] = []
    coord_map = lineage_data["coord_map"]
    full_ref_protein = lineage_data["full_ref_protein"]

    for j, pos_label in enumerate(plm_matrix.columns):
        pos_plm_0 = j
        if pos_plm_0 not in coord_map:
            continue
        pos_full_0 = coord_map[pos_plm_0]
        pos_full_1 = pos_full_0 + 1
        if pos_full_1 not in lineage_data["mut_profile"].columns:
            continue

        ref_aa = full_ref_protein[pos_full_0]
        depth_here = int(lineage_data["obs_depth"].get(pos_full_1, 0))

        for aa in plm_matrix.index:
            if aa == ref_aa:
                continue
            if aa not in lineage_data["mut_profile"].index or aa not in lineage_data["obs_freq"].index:
                continue

            plm_prob = float(plm_matrix.iloc[plm_matrix.index.get_loc(aa), j])
            mut_prob = float(lineage_data["mut_profile"].loc[aa, pos_full_1])
            obs = float(lineage_data["obs_freq"].loc[aa, pos_full_1])
            obs_count_est = int(round(obs * depth_here)) if depth_here > 0 else 0

            if args.filter_fixed_mutations and obs >= 1.0:
                continue

            obs_final = obs
            obs_present_final = 1 if obs > 0 else 0
            if args.filter_singleton_mutations and obs_count_est < args.min_obs_count:
                if args.skip_low_count_sites:
                    continue
                obs_final = 0.0
                obs_present_final = 0

            combined_rows.append(
                {
                    "model": model_spec["model_tag"],
                    "epoch_label": model_spec["epoch_label"],
                    "epoch_value": float(model_spec["epoch_value"]),
                    "lineage": lineage_label,
                    "position": int(pos_full_1),
                    "ref_aa": ref_aa,
                    "aa": aa,
                    "plm_prob": plm_prob,
                    "mut_prob": mut_prob,
                    "obs_freq": obs_final,
                    "obs_present": obs_present_final,
                    "depth": float(depth_here),
                }
            )
    return combined_rows


def safe_spearman(x_vals: pd.Series, y_vals: pd.Series) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan
    result = spearmanr(x, y)
    try:
        return float(result.correlation)
    except AttributeError:
        return float(result[0])


def safe_pearson(x_vals: pd.Series, y_vals: pd.Series) -> float:
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def fit_logistic_site_correlation(score_values: pd.Series, binary_outcome: pd.Series) -> Tuple[float, float, float]:
    from scipy.optimize import minimize
    from scipy.special import expit

    x = np.asarray(score_values, dtype=float)
    y = np.asarray(binary_outcome, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan

    x = np.log10(np.clip(x[valid], 1e-32, None))
    y = y[valid]
    if np.unique(y).size < 2 or np.nanstd(x) <= 0:
        return np.nan, np.nan, np.nan

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std <= 0:
        return np.nan, np.nan, np.nan
    x_scaled = (x - x_mean) / x_std

    def neg_log_likelihood(params: np.ndarray) -> float:
        intercept, slope = params
        logits = intercept + slope * x_scaled
        probs = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    fit = minimize(neg_log_likelihood, x0=np.array([0.0, 0.0]), method="BFGS")
    if not fit.success:
        return np.nan, np.nan, np.nan

    intercept, slope = fit.x
    fitted_probs = expit(intercept + slope * x_scaled)
    if np.std(fitted_probs) <= 0 or np.std(y) <= 0:
        fitted_corr = np.nan
    else:
        fitted_corr = float(np.corrcoef(fitted_probs, y)[0, 1])
    return fitted_corr, float(intercept), float(slope)


def compute_epoch_lineage_metrics(combined_df: pd.DataFrame) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    grouped = combined_df.groupby(["model", "epoch_label", "epoch_value", "lineage"], sort=False)
    for (model_tag, epoch_label, epoch_value, lineage_name), lineage_df in grouped:
        site_df = (
            lineage_df.groupby(["position", "ref_aa"], as_index=False)
            .agg(
                site_plm_score=("plm_prob", "max"),
                site_mut_score=("mut_prob", "max"),
                site_obs_burden=("obs_freq", "sum"),
                site_mutated=("obs_present", "max"),
            )
        )

        logit_plm_corr, logit_plm_intercept, logit_plm_slope = fit_logistic_site_correlation(
            site_df["site_plm_score"],
            site_df["site_mutated"],
        )
        logit_mut_corr, logit_mut_intercept, logit_mut_slope = fit_logistic_site_correlation(
            site_df["site_mut_score"],
            site_df["site_mutated"],
        )

        rows.append(
            {
                "model": model_tag,
                "epoch_label": epoch_label,
                "epoch_value": float(epoch_value),
                "lineage": lineage_name,
                "n_mutation_rows": int(len(lineage_df)),
                "n_site_rows": int(len(site_df)),
                "logistic_site_mutated_vs_plm_corr": logit_plm_corr,
                "logistic_site_mutated_vs_plm_intercept": logit_plm_intercept,
                "logistic_site_mutated_vs_plm_slope": logit_plm_slope,
                "logistic_site_mutated_vs_mut_corr_baseline": logit_mut_corr,
                "logistic_site_mutated_vs_mut_intercept_baseline": logit_mut_intercept,
                "logistic_site_mutated_vs_mut_slope_baseline": logit_mut_slope,
                "spearman_obs_freq_vs_plm": safe_spearman(lineage_df["plm_prob"], lineage_df["obs_freq"]),
                "spearman_obs_freq_vs_mut_baseline": safe_spearman(lineage_df["mut_prob"], lineage_df["obs_freq"]),
                "pearson_obs_freq_vs_plm": safe_pearson(lineage_df["plm_prob"], lineage_df["obs_freq"]),
                "pearson_obs_freq_vs_mut_baseline": safe_pearson(lineage_df["mut_prob"], lineage_df["obs_freq"]),
                "spearman_plm_vs_mut": safe_spearman(lineage_df["plm_prob"], lineage_df["mut_prob"]),
                "pearson_plm_vs_mut": safe_pearson(lineage_df["plm_prob"], lineage_df["mut_prob"]),
                "spearman_mut_vs_mut_baseline": 1.0,
                "pearson_mut_vs_mut_baseline": 1.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_epoch_metrics(epoch_lineage_metrics_df: pd.DataFrame) -> pd.DataFrame:
    if epoch_lineage_metrics_df.empty:
        return pd.DataFrame()

    metric_cols = [
        "logistic_site_mutated_vs_plm_corr",
        "logistic_site_mutated_vs_mut_corr_baseline",
        "spearman_obs_freq_vs_plm",
        "spearman_obs_freq_vs_mut_baseline",
        "pearson_obs_freq_vs_plm",
        "pearson_obs_freq_vs_mut_baseline",
        "spearman_plm_vs_mut",
        "spearman_mut_vs_mut_baseline",
        "pearson_plm_vs_mut",
        "pearson_mut_vs_mut_baseline",
    ]
    summary = (
        epoch_lineage_metrics_df.groupby(["model", "epoch_label", "epoch_value"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["epoch_value", "epoch_label"])
    )
    return summary


def export_plots(
    output_dir: Path,
    combined_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    epoch_summary_df: pd.DataFrame,
    scatter_alphas: List[float],
    scatter_max_points: int,
    lineage_cache: Dict[str, Dict[str, object]],
    dynamic_pseudocount: float,
    mutation_baseline_x: float,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    if not combined_df.empty:
        comparison_df = combined_df[["position", "ref_aa", "aa", "plm_prob", "mut_prob"]].drop_duplicates()
        comparison_df.to_csv(output_dir / "plm_vs_mut_prob.csv", index=False)
        plot_mask = (comparison_df["plm_prob"] > 0) & (comparison_df["mut_prob"] > 0)
        plot_df = comparison_df.loc[plot_mask]
        if not plot_df.empty:
            rho, _ = spearmanr(plot_df["plm_prob"], plot_df["mut_prob"])
            plt.figure(figsize=(6, 5))
            plt.scatter(plot_df["plm_prob"], plot_df["mut_prob"], alpha=0.3, s=10, edgecolors="none")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("PLM Probability")
            plt.ylabel("Mutation Probability")
            plt.title(f"PLM vs mutation probability\nSpearman rho={rho:.3f}")
            plt.grid(True, which="both", ls="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(output_dir / "plm_vs_mut_prob_scatter.png", dpi=300)
            plt.close()

    if not alpha_df.empty:
        metric_cols = [
            "site_top10pct_mutated_enrichment",
            "site_top10pct_mutated_precision",
            "site_rank_spearman_r",
            "mut_flat_global_spearman_r",
            "mut_flat_global_pearson_r",
            "mut_flat_mean_site_nll",
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
        axes = axes.flatten()
        for i, metric_col in enumerate(metric_cols):
            axes[i].plot(alpha_df["alpha"], alpha_df[metric_col], marker="o")
            axes[i].set_title(metric_col)
            axes[i].set_xlabel("Alpha")
            axes[i].set_ylabel("Metric value")
            axes[i].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "alpha_sweep_metrics.png", dpi=300)
        plt.close()

    lineage_names = sorted(combined_df["lineage"].dropna().unique().tolist()) if not combined_df.empty else []
    if lineage_names and scatter_alphas:
        nrows = len(lineage_names)
        ncols = len(scatter_alphas)
        fig_sc, axes_sc = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharey="row")
        axes_sc = np.array(axes_sc)
        if axes_sc.ndim == 1:
            if nrows == 1:
                axes_sc = axes_sc.reshape(1, -1)
            else:
                axes_sc = axes_sc.reshape(-1, 1)

        for row_idx, lineage_name in enumerate(lineage_names):
            lineage_df = combined_df.loc[combined_df["lineage"] == lineage_name, ["obs_freq", "plm_prob", "mut_prob"]].copy()
            if len(lineage_df) > scatter_max_points:
                lineage_df = lineage_df.sample(scatter_max_points, random_state=0)
            n_seq = len(lineage_cache[lineage_name]["records"])
            lineage_pseudocount = float(10 ** -round(np.log10(10 * max(1, n_seq))))

            for col_idx, alpha_value in enumerate(scatter_alphas):
                ax = axes_sc[row_idx, col_idx]
                if lineage_df.empty:
                    ax.set_title(f"alpha={alpha_value:.2f}\nno data")
                    continue
                x_vals = np.log10(
                    lineage_df["plm_prob"].replace(0, 1e-32)
                    * np.power(lineage_df["mut_prob"].replace(0, 1e-32), alpha_value)
                )
                y_vals = np.log10(lineage_df["obs_freq"].clip(lower=lineage_pseudocount))
                sns.scatterplot(x=x_vals, y=y_vals, ax=ax, s=8, alpha=0.25, edgecolor=None)
                rho, _ = spearmanr(x_vals, y_vals)
                ax.set_title(f"alpha={alpha_value:.2f}\nρ={rho:.3f}, n_seq={n_seq}")
                ax.grid(alpha=0.25)
                if row_idx == nrows - 1:
                    ax.set_xlabel("log10(PLM × mut^alpha)")
                if col_idx == 0:
                    ax.set_ylabel(f"{lineage_name}\nlog10(observed freq)")
        fig_sc.suptitle(
            f"Observed mutation frequency vs PLM×mutation accessibility\npseudocount={dynamic_pseudocount:.1e}"
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(output_dir / "method2_obsfreq_vs_plm_mut_scatter_grid.png", dpi=300)
        plt.close()

    if not epoch_summary_df.empty:
        metric_specs = [
            (
                "logistic_site_mutated_vs_plm_corr",
                "logistic_site_mutated_vs_mut_corr_baseline",
                "Logistic: site mutated vs score",
            ),
            (
                "spearman_obs_freq_vs_plm",
                "spearman_obs_freq_vs_mut_baseline",
                "Spearman: observed frequency vs score",
            ),
            (
                "pearson_obs_freq_vs_plm",
                "pearson_obs_freq_vs_mut_baseline",
                "Pearson: observed frequency vs score",
            ),
            (
                "spearman_plm_vs_mut",
                "spearman_mut_vs_mut_baseline",
                "Spearman: PLM vs mutation probability",
            ),
            (
                "pearson_plm_vs_mut",
                "pearson_mut_vs_mut_baseline",
                "Pearson: PLM vs mutation probability",
            ),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        epoch_summary_df = epoch_summary_df.sort_values(["epoch_value", "epoch_label"])
        for ax, (metric_col, baseline_col, title) in zip(axes, metric_specs):
            epoch_x = epoch_summary_df["epoch_value"].to_numpy(dtype=float)
            epoch_y = epoch_summary_df[metric_col].to_numpy(dtype=float)
            baseline_y = float(epoch_summary_df[baseline_col].mean()) if baseline_col in epoch_summary_df else np.nan

            ax.plot(epoch_x, epoch_y, marker="o", linewidth=1.5, color="tab:blue")
            ax.scatter(epoch_x, epoch_y, color="tab:blue", s=35, zorder=3, label="PLM epoch mean")
            if np.isfinite(baseline_y):
                ax.scatter([mutation_baseline_x], [baseline_y], color="tab:red", s=55, zorder=4, label="Mutation baseline")
                ax.axvline(mutation_baseline_x, color="tab:red", linestyle="--", alpha=0.3)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Correlation coefficient")
            ax.grid(alpha=0.25)
            if len(epoch_summary_df) > 0:
                tick_positions = [mutation_baseline_x] + epoch_summary_df["epoch_value"].tolist()
                tick_labels = [str(mutation_baseline_x)] + [str(label) for label in epoch_summary_df["epoch_label"].tolist()]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)))
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        plt.savefig(output_dir / "epoch_metric_summary.png", dpi=300)
        plt.close()


def run_analysis(args: argparse.Namespace) -> int:
    from Functions_HuggingFace import build_codon_aa_mutation_tables, evaluate_alpha_sweep

    output_dir = ensure_dir(args.output_dir)
    group_dir = ensure_dir(output_dir / "groups")
    plm_dir = ensure_dir(output_dir / "plm_cache")
    tables_dir = ensure_dir(output_dir / "tables")
    plots_dir = ensure_dir(output_dir / "plots")
    model_tables_dir = ensure_dir(tables_dir / "per_model")

    mutation_tables = build_codon_aa_mutation_tables(args.mutation_model)
    lineage_cache = build_lineage_cache(args, mutation_tables)
    target_specs = [
        {
            "label": label,
            "diversity_path": data["diversity_path"],
            "reference_path": data["reference_path"],
        }
        for label, data in lineage_cache.items()
    ]
    save_run_manifest(args, output_dir, target_specs)

    if not lineage_cache:
        raise RuntimeError("No valid targets were resolved for this run")

    model_specs = build_model_specs(args)
    runtime_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
    metadata_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    all_combined_frames: List[pd.DataFrame] = []
    all_alpha_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []
    per_group_best_rows: List[Dict[str, object]] = []

    for model_spec in model_specs:
        model_label = str(model_spec["model_tag"])
        model_combined_rows: List[Dict[str, object]] = []

        for lineage_label, lineage_data in lineage_cache.items():
            print(
                f"Processing {model_label} / {lineage_label}: n_seq={len(lineage_data['records'])}, "
                f"plm_ref_len={len(lineage_data['plm_ref_protein'])}, full_ref_len={len(lineage_data['full_ref_protein'])}"
            )
            try:
                plm_matrix, plm_path = ensure_plm_matrix(args, model_spec, lineage_label, lineage_data, plm_dir, runtime_cache)
                rows = build_combined_rows(args, model_spec, lineage_label, lineage_data, plm_matrix)
                model_combined_rows.extend(rows)
                lineage_data["mut_profile"].to_csv(group_dir / f"{lineage_data['lineage_key']}_mutation_accessibility_profile.csv")
                lineage_data["obs_freq"].to_csv(group_dir / f"{lineage_data['lineage_key']}_observed_diversity_profile.csv")
                metadata_rows.append(
                    {
                        "model": model_label,
                        "epoch_label": model_spec["epoch_label"],
                        "epoch_value": float(model_spec["epoch_value"]),
                        "lineage": lineage_label,
                        "n_sequences": len(lineage_data["records"]),
                        "reference_length": len(lineage_data["full_ref_protein"]),
                        "mapped_ref_sites": int(lineage_data["alignment_diff_stats"]["mapped_sites"]),
                        "compared_sites_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["compared_sites"]),
                        "differing_sites_vs_reference_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["differing_sites"]),
                        "fixed_differing_sites_vs_reference_non_gap_non_stop": int(lineage_data["alignment_diff_stats"]["fixed_differing_sites"]),
                        "diversity_fasta": lineage_data["diversity_path"],
                        "reference_fasta": lineage_data["reference_path"],
                        "plm_profile": plm_path,
                        "diversity_sequences_detected_as_nucleotide": bool(lineage_data["any_nucleotide_diversity"]),
                    }
                )
            except Exception as exc:
                status_rows.append({"model": model_label, "lineage": lineage_label, "status": "failed", "reason": str(exc)})
                print(f"Failed on {model_label} / {lineage_label}: {exc}")

        model_combined_df = pd.DataFrame(model_combined_rows)
        if model_combined_df.empty:
            status_rows.append({"model": model_label, "lineage": "all", "status": "failed", "reason": "no combined rows produced"})
            continue

        all_combined_frames.append(model_combined_df)
        model_combined_df.to_csv(model_tables_dir / f"{model_label}_combined_long_table.csv", index=False)

        dynamic_pseudocount = float(10 ** -round(np.log10(10 * max(1, model_combined_df["depth"].max()))))
        alpha_grid = parse_alpha_grid(args)
        use_parallel = args.alpha_parallel and len(alpha_grid) >= args.alpha_sweep_min_grid
        alpha_df = evaluate_alpha_sweep(
            model_combined_df,
            alpha_grid,
            parallel=use_parallel,
            max_workers=args.alpha_sweep_max_workers,
            alpha_sweep_min_grid=args.alpha_sweep_min_grid,
            pseudocount=1e-16,
        )
        alpha_df["model"] = model_label
        alpha_df["epoch_label"] = model_spec["epoch_label"]
        alpha_df["epoch_value"] = float(model_spec["epoch_value"])
        alpha_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
        all_alpha_frames.append(alpha_df)

        idx_a = alpha_df["site_top10pct_mutated_enrichment"].idxmax()
        idx_b = alpha_df["mut_flat_global_spearman_r"].idxmax()
        best_rows.append(
            {
                "model": model_label,
                "epoch_label": model_spec["epoch_label"],
                "epoch_value": float(model_spec["epoch_value"]),
                "method": "Method A (Site-level)",
                "criterion": "max site_top10pct_mutated_enrichment",
                "best_alpha": float(alpha_df.loc[idx_a, "alpha"]),
                "best_value": float(alpha_df.loc[idx_a, "site_top10pct_mutated_enrichment"]),
            }
        )
        best_rows.append(
            {
                "model": model_label,
                "epoch_label": model_spec["epoch_label"],
                "epoch_value": float(model_spec["epoch_value"]),
                "method": "Method B (Mutation-level flattened)",
                "criterion": "max mut_flat_global_spearman_r",
                "best_alpha": float(alpha_df.loc[idx_b, "alpha"]),
                "best_value": float(alpha_df.loc[idx_b, "mut_flat_global_spearman_r"]),
            }
        )

        for lineage_name, lineage_df in model_combined_df.groupby("lineage"):
            lineage_alpha = evaluate_alpha_sweep(
                lineage_df,
                alpha_grid,
                parallel=use_parallel,
                max_workers=args.alpha_sweep_max_workers,
                alpha_sweep_min_grid=args.alpha_sweep_min_grid,
                pseudocount=1e-16,
            )
            if lineage_alpha.empty:
                continue
            idx_group_a = lineage_alpha["site_top10pct_mutated_enrichment"].idxmax()
            idx_group_b = lineage_alpha["mut_flat_global_spearman_r"].idxmax()
            per_group_best_rows.append(
                {
                    "model": model_label,
                    "epoch_label": model_spec["epoch_label"],
                    "epoch_value": float(model_spec["epoch_value"]),
                    "lineage": lineage_name,
                    "method": "Method A (Site-level)",
                    "criterion": "max site_top10pct_mutated_enrichment",
                    "best_alpha": float(lineage_alpha.loc[idx_group_a, "alpha"]),
                    "best_value": float(lineage_alpha.loc[idx_group_a, "site_top10pct_mutated_enrichment"]),
                }
            )
            per_group_best_rows.append(
                {
                    "model": model_label,
                    "epoch_label": model_spec["epoch_label"],
                    "epoch_value": float(model_spec["epoch_value"]),
                    "lineage": lineage_name,
                    "method": "Method B (Mutation-level flattened)",
                    "criterion": "max mut_flat_global_spearman_r",
                    "best_alpha": float(lineage_alpha.loc[idx_group_b, "alpha"]),
                    "best_value": float(lineage_alpha.loc[idx_group_b, "mut_flat_global_spearman_r"]),
                }
            )

        status_rows.append({"model": model_label, "lineage": "all", "status": "completed", "reason": "ok"})

    if not all_combined_frames:
        raise RuntimeError("No combined rows were produced for any checkpoint/model")

    combined_df = pd.concat(all_combined_frames, ignore_index=True)
    combined_df.to_csv(tables_dir / "combined_long_table.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(tables_dir / "panel_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(status_rows).to_csv(tables_dir / "model_run_status.tsv", sep="\t", index=False)

    alpha_df = pd.concat(all_alpha_frames, ignore_index=True) if all_alpha_frames else pd.DataFrame()
    if not alpha_df.empty:
        alpha_df.to_csv(tables_dir / "alpha_sweep_fit_metrics.tsv", sep="\t", index=False)

    if best_rows:
        pd.DataFrame(best_rows).to_csv(tables_dir / "best_alpha_two_methods.tsv", sep="\t", index=False)

    if per_group_best_rows:
        pd.DataFrame(per_group_best_rows).to_csv(tables_dir / "best_alpha_per_group_two_methods.tsv", sep="\t", index=False)

    epoch_lineage_metrics_df = compute_epoch_lineage_metrics(combined_df)
    epoch_summary_df = summarize_epoch_metrics(epoch_lineage_metrics_df)
    if not epoch_lineage_metrics_df.empty:
        epoch_lineage_metrics_df.to_csv(tables_dir / "epoch_lineage_metrics.tsv", sep="\t", index=False)
    if not epoch_summary_df.empty:
        epoch_summary_df.to_csv(tables_dir / "epoch_metric_summary.tsv", sep="\t", index=False)

    dynamic_pseudocount = float(10 ** -round(np.log10(10 * max(1, combined_df["depth"].max()))))
    export_plots(
        output_dir=plots_dir,
        combined_df=combined_df,
        alpha_df=alpha_df,
        epoch_summary_df=epoch_summary_df,
        scatter_alphas=parse_scatter_alphas(args.scatter_alphas),
        scatter_max_points=args.scatter_max_points,
        lineage_cache=lineage_cache,
        dynamic_pseudocount=dynamic_pseudocount,
        mutation_baseline_x=args.mutation_baseline_x,
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return run_analysis(args)
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())