#!/usr/bin/env python3
"""Run codon-based mutational accessibility and PLM diversity scoring.

This script generalizes the notebook workflows for influenza and SARS-CoV-2 into
one CLI that supports either a single diversity FASTA or a guide file pointing
to many diversity FASTAs and reference sequences.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Cap CPU thread usage to 12 threads for OpenBLAS, MKL, OMP, etc.
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["VECLIB_MAXIMUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"

try:
    import torch
    try:
        torch.set_num_threads(12)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(12)
    except RuntimeError:
        pass
except ImportError:
    pass


import numpy as np
import pandas as pd
from Bio import Align, SeqIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Functions_HuggingFace import _load_single_focal_reference

STANDARD_GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

def translate_codon(codon: str) -> str:
    return STANDARD_GENETIC_CODE.get(codon.upper().replace("U", "T"), "X")

IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
PANEL_CACHE_VERSION = 3
LOGISTIC_ALPHA_METRIC_COLUMNS = (
    "site_logistic_mutated_corr",
    "site_logistic_mutated_intercept",
    "site_logistic_mutated_slope",
    "site_logistic_tjur_r2",
    "site_logistic_mcfadden_r2",
    "site_logistic_brier_score",
    "site_logistic_auroc",
    "site_logistic_pr_auc",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mutation-accessibility plus PLM scoring for viral diversity panels."
    )
    parser.add_argument(
        "--analysis-mode",
        choices=("SINGLE_FASTA", "MONTHLY_GUIDE"),
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
        "--observed-mutation-fasta",
        type=Path,
        default=None,
        help="Optional comparison FASTA used for ranked-mutation diagnostics.",
    )
    parser.add_argument(
        "--observed-mutation-sequence-id",
        default=None,
        help="Optional sequence ID to select from --observed-mutation-fasta.",
    )
    parser.add_argument(
        "--observed-mutation-selection",
        default="last",
        help="Selection strategy used when --observed-mutation-sequence-id is omitted.",
    )
    parser.add_argument(
        "--label",
        default="population",
        help="Label to use for SINGLE_FASTA mode.",
    )
    parser.add_argument(
        "--mutation-model",
        choices=("SC2", "H1N1", "H3N2"),
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
        "--use-global-plm-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Align a reused PLM profile onto each lineage reference instead of assuming identical coordinates.",
    )
    parser.add_argument(
        "--diagnostic-exports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export notebook-style diagnostic tables and plots in addition to the core pooled-panel outputs.",
    )
    parser.add_argument(
        "--alignment-verify-max-cols",
        type=int,
        default=100,
        help="Maximum number of columns rendered in the alignment verification heatmap.",
    )
    parser.add_argument(
        "--rolling-identity-window",
        type=int,
        default=30,
        help="Window size used for the rolling sequence identity diagnostic plot.",
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
    parser.add_argument(
        "--regen-figures-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate figures from existing tables in --output-dir without rebuilding cached PLM or alpha outputs.",
    )

    plm_group = parser.add_argument_group("PLM source")
    plm_source = plm_group.add_mutually_exclusive_group(required=False)
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
        default=None,
        help="Layer index for PLM inference. Omit to use the model's final layer, "
             "resolved from the loaded model (33 for ESM-2 650M, 30/36/80 for ESM-C "
             "300M/600M/6B). A fixed value does not transfer between model sizes.",
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
    if getattr(args, "regen_figures_only", False):
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --regen-figures-only is used")
        return

    if args.analysis_mode is None:
        raise ValueError("--analysis-mode is required unless --regen-figures-only is used")
    if args.mutation_model is None:
        raise ValueError("--mutation-model is required unless --regen-figures-only is used")

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
    numeric_df: pd.DataFrame
    if raw_df.shape[0] > 1 and first_row.apply(lambda value: isinstance(value, str)).any():
        numeric_df = raw_df.iloc[1:, :].apply(pd.to_numeric, errors="coerce")
    else:
        numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")

    normalised_index = pd.Index([str(value).strip().upper() for value in numeric_df.index])
    numeric_df = numeric_df.copy()
    numeric_df.index = normalised_index
    numeric_df = numeric_df.loc[~numeric_df.index.duplicated(keep="first")]
    canonical_rows = [aa for aa in STANDARD_AMINO_ACIDS if aa in numeric_df.index]
    return numeric_df.loc[canonical_rows].copy()


def infer_plm_source_sequence(raw_df: pd.DataFrame) -> Optional[str]:
    if raw_df.empty:
        return None
    first_row = raw_df.iloc[0, :]
    if raw_df.shape[0] > 1 and first_row.apply(lambda value: isinstance(value, str)).any():
        return "".join(str(value) for value in first_row.tolist())
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_arg_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "use_global_plm_reference": False,
        "diagnostic_exports": False,
        "alignment_verify_max_cols": 100,
        "rolling_identity_window": 30,
        "observed_mutation_fasta": None,
        "observed_mutation_sequence_id": None,
        "observed_mutation_selection": "last",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Cap CPU thread counts to at most 12
    if hasattr(args, "alpha_sweep_max_workers"):
        if args.alpha_sweep_max_workers is None or args.alpha_sweep_max_workers > 12:
            args.alpha_sweep_max_workers = 12
    else:
        setattr(args, "alpha_sweep_max_workers", 12)

    return args


def export_publication_figure(output_path: Path, figure=None, png_dpi: int = 600) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = figure or plt.gcf()
    figure.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def generate_verified_coordinate_map(ref_seq: str, target_seq: str):
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    alignment = next(iter(aligner.align(ref_seq, target_seq)))
    coord_map: Dict[int, int] = {}
    ref_pos = 0
    target_pos = 0
    for ref_char, target_char in zip(alignment[0], alignment[1]):
        if ref_char != "-" and target_char != "-":
            coord_map[ref_pos] = target_pos
            ref_pos += 1
            target_pos += 1
        elif ref_char == "-":
            target_pos += 1
        elif target_char == "-":
            ref_pos += 1
    return coord_map, alignment


def export_rolling_identity_plot(alignment, window_size: int, outdir: Path, label: str) -> None:
    import matplotlib.pyplot as plt

    outdir = ensure_dir(Path(outdir))
    aln_ref, aln_tgt = alignment[0], alignment[1]
    identities: List[float] = []
    positions: List[int] = []
    ref_pos = 0

    for i in range(len(aln_ref) - window_size + 1):
        window_r = aln_ref[i : i + window_size]
        window_t = aln_tgt[i : i + window_size]
        valid_pairs = [(r, t) for r, t in zip(window_r, window_t) if not (r == "-" and t == "-")]
        if not valid_pairs:
            identities.append(0.0)
        else:
            matches = sum(1 for r, t in valid_pairs if r == t)
            identities.append((matches / len(valid_pairs)) * 100.0)
        if aln_ref[i] != "-":
            ref_pos += 1
        positions.append(ref_pos)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(positions, identities, color="k", linewidth=1.5)
    ax.set_title(f"Rolling Sequence Identity ({window_size}aa window) - {label}")
    ax.set_xlabel("Focal Sequence Position")
    ax.set_ylabel("% Identity")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    export_publication_figure(outdir / f"rolling_identity_{label}.png", figure=fig)
    plt.close(fig)


def export_alignment_verification_plot(
    plm_matrix: pd.DataFrame,
    ref_seq: str,
    target_seq: str,
    coord_map: Dict[int, int],
    month_label: str,
    outdir: Path,
    max_cols: int,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    outdir = ensure_dir(Path(outdir))
    plot_data: List[Dict[str, object]] = []
    limit = min(max_cols, len(ref_seq), plm_matrix.shape[1])
    for ref_pos in range(limit):
        target_pos = coord_map.get(ref_pos)
        top_aa = plm_matrix.iloc[:, ref_pos].idxmax()
        actual_ref_aa = ref_seq[ref_pos]
        actual_tgt_aa = target_seq[target_pos] if target_pos is not None and target_pos < len(target_seq) else "-"
        plot_data.append(
            {
                "Position": ref_pos + 1,
                "PLM_Top": top_aa,
                "Ref_AA": actual_ref_aa,
                "Target_AA": actual_tgt_aa,
                "Match": 1 if actual_tgt_aa == top_aa else 0,
            }
        )

    if not plot_data:
        return

    df = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap([df["Match"].tolist()], cmap=["#e74c3c", "#2ecc71"], cbar=False, ax=ax, linewidths=0.5)
    for i in range(len(df)):
        text = f"P:{df['PLM_Top'].iloc[i]}\nR:{df['Ref_AA'].iloc[i]}\nT:{df['Target_AA'].iloc[i]}"
        ax.text(i + 0.5, 0.5, text, ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks(np.arange(len(df)) + 0.5)
    ax.set_xticklabels(df["Position"], rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"PLM Prediction vs Sequences ({month_label})\nRed = Mismatch, Green = Match")
    plt.tight_layout()
    export_publication_figure(outdir / f"alignment_verification_{month_label}.png", figure=fig)
    plt.close(fig)


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
        "use_global_plm_reference": bool(getattr(args, "use_global_plm_reference", False)),
        "diagnostic_exports": bool(getattr(args, "diagnostic_exports", False)),
        "alignment_verify_max_cols": getattr(args, "alignment_verify_max_cols", 250),
        "rolling_identity_window": getattr(args, "rolling_identity_window", 100),
        "filter_fixed_mutations": bool(args.filter_fixed_mutations),
        "filter_singleton_mutations": bool(args.filter_singleton_mutations),
        "skip_low_count_sites": bool(args.skip_low_count_sites),
        "min_obs_count": args.min_obs_count,
        "observed_mutation_fasta": str(getattr(args, "observed_mutation_fasta", None)) if getattr(args, "observed_mutation_fasta", None) else None,
        "observed_mutation_sequence_id": getattr(args, "observed_mutation_sequence_id", None),
        "observed_mutation_selection": getattr(args, "observed_mutation_selection", "first"),
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


def _build_epoch_value(epoch_label: str, fallback_index: int) -> float:
    fallback_value = fallback_index if re.search(r"\d+(?:\.\d+)?", str(epoch_label)) else sys.maxsize
    return infer_epoch_value(epoch_label, fallback_value)


def _normalise_epoch_values(specs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    next_epoch_value = 1.0

    for spec in specs:
        epoch_label = str(spec["epoch_label"])
        if epoch_label == "raw_model":
            spec["epoch_value"] = 0.0
        else:
            spec["epoch_value"] = next_epoch_value
            next_epoch_value += 1.0

    return specs


def _format_epoch_tick_label(epoch_label: str, epoch_value: float) -> str:
    if epoch_label == "raw_model":
        return "raw"
    if epoch_label == "final_checkpoint":
        return "final"

    if float(epoch_value).is_integer():
        return str(int(epoch_value))
    return str(epoch_value)


def _load_cached_model_outputs(model_tables_dir: Path, model_spec: Dict[str, object]) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    model_label = str(model_spec["model_tag"])
    combined_path = model_tables_dir / f"{model_label}_combined_long_table.csv"
    alpha_path = model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv"
    alpha_by_lineage_path = model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv"
    if not (combined_path.exists() and alpha_path.exists() and alpha_by_lineage_path.exists()):
        return None

    combined_df = pd.read_csv(combined_path)
    alpha_df = pd.read_csv(alpha_path, sep="\t")
    combined_df["model"] = model_label
    combined_df["epoch_label"] = model_spec["epoch_label"]
    combined_df["epoch_value"] = float(model_spec["epoch_value"])
    alpha_df["model"] = model_label
    alpha_df["epoch_label"] = model_spec["epoch_label"]
    alpha_df["epoch_value"] = float(model_spec["epoch_value"])
    return combined_df, alpha_df


def _all_model_outputs_cached(model_tables_dir: Path, model_specs: List[Dict[str, object]]) -> bool:
    return all(_load_cached_model_outputs(model_tables_dir, model_spec) is not None for model_spec in model_specs)


def _build_lightweight_lineage_cache_from_metadata(panel_metadata_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    if panel_metadata_df.empty:
        return {}

    lineage_cache: Dict[str, Dict[str, object]] = {}
    grouped = panel_metadata_df.groupby("lineage", sort=False)
    for lineage_name, lineage_df in grouped:
        first_row = lineage_df.iloc[0]
        lineage_cache[str(lineage_name)] = {
            "n_sequences": int(lineage_df["n_sequences"].max()),
            "diversity_path": str(first_row.get("diversity_fasta", "")),
            "reference_path": str(first_row.get("reference_fasta", "")),
        }
    return lineage_cache


def _raw_model_spec(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "model_tag": f"{args.model_tag}_raw",
        "epoch_label": "raw_model",
        "epoch_value": 0.0,
        "base_model": args.base_model,
        "checkpoint_label": None,
        "model_display_label": f"{args.base_model} raw base model",
        "checkpoint_dir": None,
        "precomputed_plm_path": None,
    }


def _checkpoint_signature(checkpoint_dir: Optional[Path]) -> Optional[str]:
    if checkpoint_dir is None:
        return None
    weights_path = Path(checkpoint_dir) / "model.safetensors"
    if not weights_path.exists():
        return None

    digest = hashlib.sha256()
    with weights_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_model_specs(specs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    unique_specs: List[Dict[str, object]] = []
    seen_signatures = set()

    for spec in specs:
        epoch_label = str(spec["epoch_label"])
        if epoch_label == "raw_model":
            unique_specs.append(spec)
            continue

        signature = _checkpoint_signature(spec.get("checkpoint_dir"))
        if signature is not None:
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

        unique_specs.append(spec)

    return unique_specs


def build_model_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.precomputed_plm_path is not None:
        label = args.model_tag if args.model_tag else args.precomputed_plm_path.stem
        return _normalise_epoch_values([
            {
                "model_tag": label,
                "epoch_label": label,
                "epoch_value": _build_epoch_value(label, 0),
                "base_model": args.base_model,
                "checkpoint_label": None,
                "model_display_label": label,
                "checkpoint_dir": None,
                "precomputed_plm_path": args.precomputed_plm_path,
            }
        ])

    if args.checkpoint_glob:
        matched_paths = [Path(path) for path in glob.glob(args.checkpoint_glob)]
        matched_paths = [path for path in matched_paths if path.exists()]
        if not matched_paths:
            raise ValueError(f"No checkpoint paths matched --checkpoint-glob={args.checkpoint_glob!r}")
        ordered = sorted(
            matched_paths,
            key=lambda path: (infer_epoch_value(path.name, 0), path.name),
        )
        specs = [_raw_model_spec(args)]
        for index, checkpoint_path in enumerate(ordered):
            epoch_label = checkpoint_path.name
            specs.append(
                {
                    "model_tag": f"{args.model_tag}_{epoch_label}",
                    "epoch_label": epoch_label,
                    "epoch_value": _build_epoch_value(epoch_label, index),
                    "base_model": args.base_model,
                    "checkpoint_label": checkpoint_path.name,
                    "model_display_label": f"{args.base_model} + {checkpoint_path.name}",
                    "checkpoint_dir": checkpoint_path,
                    "precomputed_plm_path": None,
                }
            )
        return _normalise_epoch_values(_deduplicate_model_specs(specs))

    if args.checkpoint_dir is not None:
        discovered_paths = _discover_checkpoint_dirs(args.checkpoint_dir)
        if discovered_paths:
            specs = [_raw_model_spec(args)]
            for index, checkpoint_path in enumerate(discovered_paths):
                epoch_label = checkpoint_path.name
                specs.append(
                    {
                        "model_tag": f"{args.model_tag}_{epoch_label}",
                        "epoch_label": epoch_label,
                        "epoch_value": _build_epoch_value(epoch_label, index),
                        "base_model": args.base_model,
                        "checkpoint_label": checkpoint_path.name,
                        "model_display_label": f"{args.base_model} + {checkpoint_path.name}",
                        "checkpoint_dir": checkpoint_path,
                        "precomputed_plm_path": None,
                    }
                )
            return _normalise_epoch_values(_deduplicate_model_specs(specs))
        elif (args.checkpoint_dir / "model.safetensors").exists():
            print(
                f"Warning: no epoch subdirectories containing model.safetensors were found "
                f"inside {args.checkpoint_dir}. A model.safetensors was found directly in "
                f"that directory — treating it as a single checkpoint."
            )
            epoch_label = args.checkpoint_dir.name
            return _normalise_epoch_values(_deduplicate_model_specs([
                _raw_model_spec(args),
                {
                    "model_tag": args.model_tag,
                    "epoch_label": epoch_label,
                    "epoch_value": _build_epoch_value(epoch_label, 0),
                    "base_model": args.base_model,
                    "checkpoint_label": args.checkpoint_dir.name,
                    "model_display_label": f"{args.base_model} + {args.checkpoint_dir.name}",
                    "checkpoint_dir": args.checkpoint_dir,
                    "precomputed_plm_path": None,
                },
            ]))
        else:
            print(
                f"Warning: no model.safetensors found in {args.checkpoint_dir} or any "
                f"of its subdirectories. Proceeding, but PLM inference will likely fail."
            )

    epoch_label = args.checkpoint_dir.name if args.checkpoint_dir else args.model_tag
    return _normalise_epoch_values([
        {
            "model_tag": args.model_tag,
            "epoch_label": epoch_label,
            "epoch_value": _build_epoch_value(epoch_label, 0),
            "base_model": args.base_model,
            "checkpoint_label": args.checkpoint_dir.name if args.checkpoint_dir else None,
            "model_display_label": (
                f"{args.base_model} + {args.checkpoint_dir.name}"
                if args.checkpoint_dir else args.model_tag
            ),
            "checkpoint_dir": args.checkpoint_dir,
            "precomputed_plm_path": None,
        }
    ])


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
            "full_ref_nt": full_ref_nt,
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
        source_sequence = infer_plm_source_sequence(raw_matrix) or str(lineage_data["plm_ref_protein"])
        return normalise_plm_matrix(raw_matrix), str(model_spec["precomputed_plm_path"]), source_sequence

    if plm_cache_path.exists() and not args.force_recompute_plm:
        raw_matrix = load_plm_probability_matrix(str(plm_cache_path))
        source_sequence = infer_plm_source_sequence(raw_matrix) or str(lineage_data["plm_ref_protein"])
        return normalise_plm_matrix(raw_matrix), str(plm_cache_path), source_sequence

    cache_key = (model_tag, lineage_data["plm_ref_protein"])
    plm_payload = runtime_cache.get(cache_key)
    if plm_payload is None:
        runtime = runtime_cache.get(("__runtime__", model_tag))
        if runtime is None:
            if not args.gpu_required:
                print("Error: This application requires a GPU to run.", file=sys.stderr)
                sys.exit(1)
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
    return probability_rows, str(plm_cache_path), str(plm_payload["sequence"])


def resolve_plm_coordinate_maps(
    args: argparse.Namespace,
    source_plm_sequence: str,
    lineage_data: Dict[str, object],
):
    lineage_trim_map = dict(lineage_data["coord_map"])
    if not args.use_global_plm_reference:
        return lineage_trim_map, {idx: idx for idx in lineage_trim_map}, None

    global_to_lineage_trim, alignment = generate_verified_coordinate_map(source_plm_sequence, str(lineage_data["plm_ref_protein"]))
    plm_to_full: Dict[int, int] = {}
    for source_idx, target_trim_idx in global_to_lineage_trim.items():
        if target_trim_idx in lineage_trim_map:
            plm_to_full[source_idx] = int(lineage_trim_map[target_trim_idx])
    return plm_to_full, global_to_lineage_trim, alignment


def build_combined_rows(
    args: argparse.Namespace,
    model_spec: Dict[str, object],
    lineage_label: str,
    lineage_data: Dict[str, object],
    plm_matrix: pd.DataFrame,
    coord_map: Optional[Dict[int, int]] = None,
) -> List[Dict[str, object]]:
    combined_rows: List[Dict[str, object]] = []
    coord_map = lineage_data["coord_map"] if coord_map is None else coord_map
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
                    "model_display_label": model_spec.get("model_display_label", model_spec["model_tag"]),
                    "base_model": model_spec.get("base_model"),
                    "checkpoint_label": model_spec.get("checkpoint_label"),
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


def warn_on_excess_mutation_rows(
    combined_df: pd.DataFrame,
    *,
    context_label: str,
    max_rows_per_site: int = 20,
) -> None:
    required_cols = {"lineage", "position", "ref_aa"}
    if combined_df.empty or not required_cols.issubset(combined_df.columns):
        return

    grouped = combined_df.groupby("lineage", sort=False) if "lineage" in combined_df.columns else [("all", combined_df)]
    for lineage_name, lineage_df in grouped:
        site_count = int(lineage_df.loc[:, ["position", "ref_aa"]].drop_duplicates().shape[0])
        if site_count <= 0:
            continue
        row_count = int(len(lineage_df))
        max_expected_rows = int(max_rows_per_site * site_count)
        if row_count > max_expected_rows:
            aa_count = int(lineage_df["aa"].nunique()) if "aa" in lineage_df.columns else np.nan
            warnings.warn(
                f"{context_label}: lineage {lineage_name} has {row_count} mutation rows across {site_count} sites "
                f"({row_count / site_count:.2f} rows/site), exceeding the hard limit of {max_rows_per_site}x sites "
                f"({max_expected_rows} rows). Check PLM alphabet / duplicate candidate rows before interpreting statistics. "
                f"Unique non-reference amino-acid labels observed: {aa_count}.",
                UserWarning,
                stacklevel=2,
            )


def export_codon_model_diagnostics(output_dir: Path, mutation_tables: Dict[str, object]) -> None:
    from Functions_HuggingFace import _build_aa20_average_and_reconstruction, _flattened_fit_metrics

    output_dir = ensure_dir(output_dir)
    bases = ["A", "C", "G", "T"]
    active_transitions = np.array(mutation_tables["transitions"], dtype=float)
    active_transition_tag = str(mutation_tables["transition_meta"]["tag"])
    codon_mutation_df = mutation_tables["codon_mutation_df"]
    genetic_code = mutation_tables["genetic_code"]
    aa_to_codons_all = mutation_tables["aa_to_codons_all"]
    target_aas = mutation_tables["target_aas"]
    ordered_codons = mutation_tables["ordered_codons"]

    codon_to_aa_matrix = pd.DataFrame(0.0, index=ordered_codons, columns=target_aas)
    for codon_from in ordered_codons:
        for aa in target_aas:
            total_prob = 0.0
            for codon_to in aa_to_codons_all[aa]:
                total_prob += float(codon_mutation_df.loc[codon_from, codon_to])
            codon_to_aa_matrix.loc[codon_from, aa] = total_prob
        own_aa = genetic_code.get(codon_from)
        if own_aa in codon_to_aa_matrix.columns:
            codon_to_aa_matrix.loc[codon_from, own_aa] = np.nan

    codon_to_aa_matrix.to_csv(output_dir / f"codon_to_aa_matrix_{active_transition_tag}_with_stop.csv")
    aa20 = [aa for aa in target_aas if aa != "*"]
    codon_to_aa_20 = codon_to_aa_matrix.loc[ordered_codons, aa20].copy()
    codon_to_aa_20.to_csv(output_dir / f"codon_to_aa_matrix_{active_transition_tag}_aa20.csv")
    aa20_transition_avg, codon_to_aa_20_reconstructed, _ = _build_aa20_average_and_reconstruction(
        codon_to_aa_20,
        aa20,
        ordered_codons,
        genetic_code,
    )
    selected_self_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed)

    transition_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    row_total_mut_rate = float(np.mean(np.sum(active_transitions, axis=1)))
    kimura80_transitions = np.zeros((4, 4), dtype=float)
    for i, src_base in enumerate(bases):
        weights: Dict[int, float] = {}
        weight_sum = 0.0
        for j, dst_base in enumerate(bases):
            if i == j:
                continue
            weight = 2.0 if (src_base, dst_base) in transition_pairs else 1.0
            weights[j] = weight
            weight_sum += weight
        for j, weight in weights.items():
            kimura80_transitions[i, j] = row_total_mut_rate * (weight / weight_sum)

    kimura80_probs = kimura80_transitions.copy()
    for i in range(4):
        kimura80_probs[i, i] = 1.0 - np.sum(kimura80_transitions[i, :])

    codons = mutation_tables["codons"]
    codon_mutation_matrix_k80 = np.zeros((len(codons), len(codons)), dtype=float)
    for i, codon_from in enumerate(codons):
        for j, codon_to in enumerate(codons):
            prob = 1.0
            for k in range(3):
                prob *= kimura80_probs[bases.index(codon_from[k]), bases.index(codon_to[k])]
            codon_mutation_matrix_k80[i, j] = prob

    codon_mutation_df_k80 = pd.DataFrame(codon_mutation_matrix_k80, index=codons, columns=codons)
    codon_to_aa_matrix_k80 = pd.DataFrame(0.0, index=ordered_codons, columns=target_aas)
    for codon_from in ordered_codons:
        for aa in target_aas:
            total_prob = 0.0
            for codon_to in aa_to_codons_all[aa]:
                total_prob += float(codon_mutation_df_k80.loc[codon_from, codon_to])
            codon_to_aa_matrix_k80.loc[codon_from, aa] = total_prob
        own_aa = genetic_code.get(codon_from)
        if own_aa in codon_to_aa_matrix_k80.columns:
            codon_to_aa_matrix_k80.loc[codon_from, own_aa] = np.nan

    codon_to_aa_20_k80 = codon_to_aa_matrix_k80.loc[ordered_codons, aa20].copy()
    aa20_transition_avg_k80, codon_to_aa_20_reconstructed_k80, _ = _build_aa20_average_and_reconstruction(
        codon_to_aa_20_k80,
        aa20,
        ordered_codons,
        genetic_code,
    )
    generic_self_metrics = _flattened_fit_metrics(codon_to_aa_20_k80, codon_to_aa_20_reconstructed_k80)
    generic_to_selected_metrics = _flattened_fit_metrics(codon_to_aa_20, codon_to_aa_20_reconstructed_k80)
    aa20_selected_vs_k80_metrics = _flattened_fit_metrics(aa20_transition_avg, aa20_transition_avg_k80)
    if np.isfinite(generic_to_selected_metrics["residual_var"]) and generic_to_selected_metrics["residual_var"] > 0:
        error_reduction_pct = 100.0 * (
            (generic_to_selected_metrics["residual_var"] - selected_self_metrics["residual_var"])
            / generic_to_selected_metrics["residual_var"]
        )
    else:
        error_reduction_pct = np.nan

    compression_summary = pd.DataFrame(
        [
            {
                "comparison": f"{active_transition_tag}64_to_{active_transition_tag}20_reconstruction",
                "finite_entries_compared": selected_self_metrics["n_entries"],
                "total_variance": selected_self_metrics["total_var"],
                "residual_variance": selected_self_metrics["residual_var"],
                "retained_variation_percent": selected_self_metrics["retained_pct"],
                "flattened_correlation_r": selected_self_metrics["corr"],
                "rmse": selected_self_metrics["rmse"],
                "mae": selected_self_metrics["mae"],
            },
            {
                "comparison": "k80_64_to_k80_20_reconstruction",
                "finite_entries_compared": generic_self_metrics["n_entries"],
                "total_variance": generic_self_metrics["total_var"],
                "residual_variance": generic_self_metrics["residual_var"],
                "retained_variation_percent": generic_self_metrics["retained_pct"],
                "flattened_correlation_r": generic_self_metrics["corr"],
                "rmse": generic_self_metrics["rmse"],
                "mae": generic_self_metrics["mae"],
            },
            {
                "comparison": f"{active_transition_tag}64_to_k80_20_reconstruction",
                "finite_entries_compared": generic_to_selected_metrics["n_entries"],
                "total_variance": generic_to_selected_metrics["total_var"],
                "residual_variance": generic_to_selected_metrics["residual_var"],
                "retained_variation_percent": generic_to_selected_metrics["retained_pct"],
                "flattened_correlation_r": generic_to_selected_metrics["corr"],
                "rmse": generic_to_selected_metrics["rmse"],
                "mae": generic_to_selected_metrics["mae"],
            },
            {
                "comparison": f"{active_transition_tag}20_vs_k80_20",
                "finite_entries_compared": aa20_selected_vs_k80_metrics["n_entries"],
                "total_variance": aa20_selected_vs_k80_metrics["total_var"],
                "residual_variance": aa20_selected_vs_k80_metrics["residual_var"],
                "retained_variation_percent": aa20_selected_vs_k80_metrics["retained_pct"],
                "flattened_correlation_r": aa20_selected_vs_k80_metrics["corr"],
                "rmse": aa20_selected_vs_k80_metrics["rmse"],
                "mae": aa20_selected_vs_k80_metrics["mae"],
            },
        ]
    )
    gain_summary = pd.DataFrame(
        [
            {
                f"{active_transition_tag}_specific_error_reduction_vs_generic_percent": error_reduction_pct,
                "generic_model_assumption": "AT50_TiTv2to1",
                "generic_row_total_mutation_rate": row_total_mut_rate,
            }
        ]
    )
    compression_summary.to_csv(output_dir / "codon_to_aa_compression_summary.csv", index=False)
    gain_summary.to_csv(output_dir / f"{active_transition_tag}_vs_k80_gain_summary.csv", index=False)
    aa20_transition_avg.to_csv(output_dir / "aa20_transition_matrix_from_codon_averages.csv")
    aa20_transition_avg_k80.to_csv(output_dir / "aa20_transition_matrix_k80_generic.csv")


def _build_mapped_mutational_matrix(
    plm_matrix: pd.DataFrame,
    lineage_data: Dict[str, object],
    coord_map: Dict[int, int],
):
    mapped_cols: List[int] = []
    mapped_full_positions: List[int] = []
    ref_chars: List[str] = []
    full_ref_protein = str(lineage_data["full_ref_protein"])
    mut_profile = lineage_data["mut_profile"]

    for j in range(plm_matrix.shape[1]):
        if j not in coord_map:
            continue
        full_pos0 = int(coord_map[j])
        full_pos1 = full_pos0 + 1
        if full_pos1 not in mut_profile.columns:
            continue
        mapped_cols.append(j)
        mapped_full_positions.append(full_pos0)
        ref_chars.append(full_ref_protein[full_pos0])

    if not mapped_cols:
        return pd.DataFrame(index=plm_matrix.index), pd.DataFrame(index=plm_matrix.index), [], [], ""

    mapped_plm = plm_matrix.iloc[:, mapped_cols].copy()
    mapped_mut = pd.DataFrame(0.0, index=plm_matrix.index, columns=mapped_plm.columns)
    for subset_idx, full_pos0 in enumerate(mapped_full_positions):
        full_pos1 = full_pos0 + 1
        col_label = mapped_plm.columns[subset_idx]
        for aa in mapped_mut.index:
            if aa in mut_profile.index:
                mapped_mut.loc[aa, col_label] = float(mut_profile.loc[aa, full_pos1])
    return mapped_plm, mapped_mut, mapped_cols, mapped_full_positions, "".join(ref_chars)


def _load_observed_mutations_for_matrix(
    args: argparse.Namespace,
    lineage_data: Dict[str, object],
    mapped_full_positions: List[int],
) -> List[Tuple[int, str]]:
    from Functions_HuggingFace import _load_comparison_protein_sequence

    if args.observed_mutation_fasta is None:
        return []

    _, target_protein_seq = _load_comparison_protein_sequence(
        str(args.observed_mutation_fasta),
        sequence_id=args.observed_mutation_sequence_id,
        selection=args.observed_mutation_selection,
    )
    if target_protein_seq is None:
        return []

    full_ref_protein = str(lineage_data["full_ref_protein"])
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignment = next(iter(aligner.align(full_ref_protein, target_protein_seq)))

    full_to_subset = {full_pos: subset_idx for subset_idx, full_pos in enumerate(mapped_full_positions)}
    observed_mutations: List[Tuple[int, str]] = []
    ref_pos = 0
    for ref_char, target_char in zip(alignment[0], alignment[1]):
        if ref_char == "-":
            continue
        current_ref_pos = ref_pos
        ref_pos += 1
        if target_char == "-":
            continue
        if ref_char != target_char and current_ref_pos in full_to_subset:
            observed_mutations.append((full_to_subset[current_ref_pos], target_char))
    return observed_mutations


def export_lineage_diagnostics(
    args: argparse.Namespace,
    plot_dir: Path,
    table_dir: Path,
    model_label: str,
    lineage_label: str,
    lineage_data: Dict[str, object],
    plm_matrix: pd.DataFrame,
    coord_map: Dict[int, int],
    source_plm_sequence: str,
    mutation_tables: Dict[str, object],
    global_to_lineage_trim: Dict[int, int],
    remap_alignment,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr

    from Functions_HuggingFace import get_ranked_mutations, validate_mutational_matrix

    plot_dir = ensure_dir(plot_dir / model_label / str(lineage_data["lineage_key"]))
    table_dir = ensure_dir(table_dir / model_label / str(lineage_data["lineage_key"]))
    mapped_plm, mapped_mut, _, mapped_full_positions, mapped_ref_seq = _build_mapped_mutational_matrix(plm_matrix, lineage_data, coord_map)
    if mapped_plm.empty or mapped_mut.empty:
        return

    validate_mutational_matrix(mapped_mut)
    mapped_mut.to_csv(table_dir / "mutational_prob_matrix.csv")
    combined_prob = mapped_plm * mapped_mut
    combined_prob_sqrt = mapped_plm * np.sqrt(mapped_mut)
    combined_prob.to_csv(table_dir / "combined_prob_matrix.csv")
    combined_prob_sqrt.to_csv(table_dir / "combined_prob_sqrt_matrix.csv")

    hist_values = mapped_mut.to_numpy().flatten()
    hist_values = hist_values[np.isfinite(hist_values) & (hist_values > 0)]
    plt.figure(figsize=(10, 6))
    if hist_values.size > 0:
        bins = np.logspace(np.log10(hist_values.min()), np.log10(hist_values.max()), 100)
        ax = sns.histplot(hist_values, bins=bins, stat="count", element="bars", fill=True, color="#4c72b0", edgecolor="white", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    plt.title(f"Histogram of Mutational Probability Matrix Values ({model_label} / {lineage_label})")
    plt.xlabel("Mutational Probability")
    plt.ylabel("Count")
    export_publication_figure(plot_dir / "histogram_mutational_prob.png")
    plt.close()

    observed_mutations = _load_observed_mutations_for_matrix(args, lineage_data, mapped_full_positions)
    matrices_to_plot = {
        "PLM Probability": mapped_plm,
        "Mutational Probability": mapped_mut,
        "P_plm * P_mut": combined_prob,
        "P_plm * sqrt(P_mut)": combined_prob_sqrt,
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for i, (name, matrix) in enumerate(matrices_to_plot.items()):
        ax = axes_flat[i]
        ranked_df, obs_df = get_ranked_mutations(matrix, mapped_ref_seq, observed_mutations)
        if ranked_df.empty:
            continue
        ranked_df["log10Probability"] = np.log10(np.clip(ranked_df["Probability"], 1e-32, None))
        ax.plot(ranked_df["Rank"], ranked_df["log10Probability"], color="lightgray", linewidth=1)
        if not obs_df.empty:
            obs_df = obs_df.copy()
            obs_df["log10Probability"] = np.log10(np.clip(obs_df["Probability"], 1e-32, None))
            ax.scatter(obs_df["Rank"], obs_df["log10Probability"], color="red", zorder=5, s=20)
        ax.set_title(f"{name} ({model_label} / {lineage_label})")
        ax.set_xlabel("Rank (1 = Highest Prob)")
        ax.set_ylabel("log10(Probability)")
        ax.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    export_publication_figure(plot_dir / "ranked_mutations.png", figure=fig)
    plt.close(fig)

    plm_no_diag = mapped_plm.copy()
    mut_no_diag = mapped_mut.copy()
    for j in range(plm_no_diag.shape[1]):
        ref_aa = mapped_ref_seq[j]
        if ref_aa in plm_no_diag.index:
            row_idx = plm_no_diag.index.get_loc(ref_aa)
            plm_no_diag.iloc[row_idx, j] = np.nan
            mut_no_diag.iloc[row_idx, j] = np.nan
    plm_flat = plm_no_diag.to_numpy().flatten()
    mut_flat = mut_no_diag.to_numpy().flatten()
    valid_mask = np.isfinite(plm_flat) & np.isfinite(mut_flat)
    plm_flat = plm_flat[valid_mask]
    mut_flat = mut_flat[valid_mask]
    if plm_flat.size > 1 and mut_flat.size > 1:
        pearson_corr, p_p = pearsonr(plm_flat, mut_flat)
        spearman_corr, p_s = spearmanr(plm_flat, mut_flat)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=np.log10(np.clip(plm_flat, 1e-32, None)), y=np.log10(np.clip(mut_flat, 1e-32, None)), alpha=0.3)
        plt.title(
            f"PLM vs Mutational Probability ({model_label} / {lineage_label})\n"
            f"spearman: {spearman_corr:.3f} (p={p_s:.2e}), pearson: {pearson_corr:.3f} (p={p_p:.2e})"
        )
        plt.xlabel("log10(PLM Probability)")
        plt.ylabel("log10(Mutational Probability)")
        export_publication_figure(plot_dir / "plm_vs_mut_correlation.png")
        plt.close()

    active_transitions = np.array(mutation_tables["transitions"], dtype=float)
    max_raw_prob = float(np.max(active_transitions))
    high_prob_rows: List[Dict[str, object]] = []
    full_ref_nt = str(lineage_data["full_ref_nt"])
    for subset_idx, full_pos0 in enumerate(mapped_full_positions):
        ref_aa = mapped_ref_seq[subset_idx]
        current_codon = full_ref_nt[3 * full_pos0 : 3 * full_pos0 + 3]
        col_label = mapped_mut.columns[subset_idx]
        for target_aa in mapped_mut.index:
            if not isinstance(target_aa, str) or target_aa == ref_aa:
                continue
            prob = float(mapped_mut.loc[target_aa, col_label])
            if prob > max_raw_prob:
                high_prob_rows.append(
                    {
                        "Position": full_pos0 + 1,
                        "Ref_AA": ref_aa,
                        "Ref_Codon": current_codon,
                        "Target_AA": target_aa,
                        "Probability": prob,
                    }
                )
    if high_prob_rows:
        pd.DataFrame(high_prob_rows).sort_values("Probability", ascending=False).to_csv(table_dir / "high_prob_mutations.csv", index=False)

    if remap_alignment is not None:
        verify_dir = plot_dir / "alignment_verifications"
        export_alignment_verification_plot(
            plm_matrix=plm_matrix,
            ref_seq=source_plm_sequence,
            target_seq=str(lineage_data["plm_ref_protein"]),
            coord_map=global_to_lineage_trim,
            month_label=lineage_label,
            outdir=verify_dir,
            max_cols=args.alignment_verify_max_cols,
        )
        export_rolling_identity_plot(
            alignment=remap_alignment,
            window_size=args.rolling_identity_window,
            outdir=verify_dir,
            label=lineage_label,
        )


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


def safe_auroc(score_values: pd.Series, binary_outcome: pd.Series) -> float:
    from scipy.stats import rankdata

    scores = np.asarray(score_values, dtype=float)
    outcome = np.asarray(binary_outcome, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(outcome)
    if valid.sum() < 2:
        return np.nan
    scores = scores[valid]
    outcome = outcome[valid]
    pos_mask = outcome > 0.5
    neg_mask = ~pos_mask
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    pos_rank_sum = float(np.sum(ranks[pos_mask]))
    return float((pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def safe_pr_auc(score_values: pd.Series, binary_outcome: pd.Series) -> float:
    scores = np.asarray(score_values, dtype=float)
    outcome = np.asarray(binary_outcome, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(outcome)
    if valid.sum() < 2:
        return np.nan
    scores = scores[valid]
    outcome = outcome[valid]
    pos_mask = outcome > 0.5
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(~pos_mask))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = outcome[order] > 0.5
    tp = np.cumsum(y_sorted, dtype=float)
    fp = np.cumsum(~y_sorted, dtype=float)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(n_pos)
    precision_at_hits = precision[y_sorted]
    recall_at_hits = recall[y_sorted]
    recall_prev = np.concatenate(([0.0], recall_at_hits[:-1]))
    return float(np.sum((recall_at_hits - recall_prev) * precision_at_hits))


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

    y_mean = float(np.mean(y))
    y_mean = float(np.clip(y_mean, 1e-6, 1.0 - 1e-6))
    intercept_seed = float(np.log(y_mean / (1.0 - y_mean)))

    corr_seed = float(np.corrcoef(x_scaled, y)[0, 1]) if np.std(y) > 0 else 0.0
    if not np.isfinite(corr_seed):
        corr_seed = 0.0
    slope_seed = float(np.clip(4.0 * corr_seed, -8.0, 8.0))

    def neg_log_likelihood(params: np.ndarray) -> float:
        intercept, slope = params
        logits = intercept + slope * x_scaled
        probs = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    bounds = [(-20.0, 20.0), (-20.0, 20.0)]
    initial_guesses = [
        np.array([0.0, 0.0], dtype=float),
        np.array([intercept_seed, 0.0], dtype=float),
        np.array([intercept_seed, slope_seed], dtype=float),
        np.array([intercept_seed, -slope_seed], dtype=float),
        np.array([0.0, slope_seed], dtype=float),
        np.array([0.0, -slope_seed], dtype=float),
    ]
    methods = [
        ("L-BFGS-B", {"bounds": bounds}),
        ("Powell", {"bounds": bounds}),
        ("Nelder-Mead", {}),
        ("BFGS", {}),
    ]

    best_fit = None
    best_nll = np.inf
    for x0 in initial_guesses:
        for method_name, extra_kwargs in methods:
            try:
                fit = minimize(
                    neg_log_likelihood,
                    x0=x0,
                    method=method_name,
                    options={"maxiter": 2000},
                    **extra_kwargs,
                )
            except Exception:
                continue

            fit_nll = float(fit.fun) if np.isfinite(fit.fun) else np.inf
            if fit.success and np.isfinite(fit_nll) and fit_nll < best_nll and np.all(np.isfinite(fit.x)):
                best_fit = fit
                best_nll = fit_nll

    if best_fit is None:
        return np.nan, np.nan, np.nan

    intercept, slope = best_fit.x
    fitted_probs = expit(intercept + slope * x_scaled)
    if np.std(fitted_probs) <= 0 or np.std(y) <= 0:
        fitted_corr = np.nan
    else:
        fitted_corr = float(np.corrcoef(fitted_probs, y)[0, 1])
    return fitted_corr, float(intercept), float(slope)


def validate_site_logistic_fit_agreement(
    score_values: pd.Series,
    binary_outcome: pd.Series,
    *,
    context_label: str,
    atol: float = 1e-6,
) -> Tuple[float, float, float]:
    site_corr, site_intercept, site_slope = fit_logistic_site_correlation(score_values, binary_outcome)
    if not np.isfinite(site_corr):
        return site_corr, site_intercept, site_slope

    feature_result = fit_logistic_feature_model(
        pd.DataFrame({"log10_score": np.log10(np.clip(np.asarray(score_values, dtype=float), 1e-32, None))}),
        pd.Series(binary_outcome, copy=False),
    )
    feature_corr = float(feature_result.get("logistic_fitted_prob_corr", np.nan))
    if np.isfinite(feature_corr) and not np.isclose(site_corr, feature_corr, atol=atol, rtol=0.0):
        raise AssertionError(
            f"Site logistic validation failed for {context_label}: "
            f"site helper corr={site_corr:.12f}, generic fitter corr={feature_corr:.12f}"
        )
    return site_corr, site_intercept, site_slope


def compute_site_logistic_metrics(
    score_values: pd.Series,
    binary_outcome: pd.Series,
    *,
    context_label: str,
) -> Dict[str, float]:
    site_corr, site_intercept, site_slope = validate_site_logistic_fit_agreement(
        score_values,
        binary_outcome,
        context_label=context_label,
    )
    feature_result = fit_logistic_feature_model(
        pd.DataFrame({"log10_score": np.log10(np.clip(np.asarray(score_values, dtype=float), 1e-32, None))}),
        pd.Series(binary_outcome, copy=False),
    )
    return {
        "site_logistic_mutated_corr": float(site_corr),
        "site_logistic_mutated_intercept": float(site_intercept),
        "site_logistic_mutated_slope": float(site_slope),
        "site_logistic_tjur_r2": float(feature_result.get("logistic_tjur_r2", np.nan)),
        "site_logistic_mcfadden_r2": float(feature_result.get("logistic_mcfadden_r2", np.nan)),
        "site_logistic_brier_score": float(feature_result.get("logistic_brier_score", np.nan)),
        "site_logistic_auroc": float(feature_result.get("logistic_auroc", np.nan)),
        "site_logistic_pr_auc": float(feature_result.get("logistic_pr_auc", np.nan)),
    }


def compute_mutation_row_logistic_metrics(
    log_score_values: pd.Series,
    binary_outcome: pd.Series,
    *,
    feature_name: str,
) -> Dict[str, float]:
    feature_result = fit_logistic_feature_model(
        pd.DataFrame({feature_name: np.asarray(log_score_values, dtype=float)}),
        pd.Series(binary_outcome, copy=False),
        enforce_positive_coefficients=True,
    )
    coef_key = f"logistic_coef_{feature_name}"
    return {
        "site_logistic_mutated_corr": float(feature_result.get("logistic_fitted_prob_corr", np.nan)),
        "site_logistic_mutated_intercept": float(feature_result.get("logistic_intercept", np.nan)),
        "site_logistic_mutated_slope": float(feature_result.get(coef_key, np.nan)),
        "site_logistic_tjur_r2": float(feature_result.get("logistic_tjur_r2", np.nan)),
        "site_logistic_mcfadden_r2": float(feature_result.get("logistic_mcfadden_r2", np.nan)),
        "site_logistic_brier_score": float(feature_result.get("logistic_brier_score", np.nan)),
        "site_logistic_auroc": float(feature_result.get("logistic_auroc", np.nan)),
        "site_logistic_pr_auc": float(feature_result.get("logistic_pr_auc", np.nan)),
    }


def mean_finite(values: List[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return np.nan
    return float(np.mean(finite_values))


def compute_baseline_alpha_position(alpha_values: pd.Series | List[float] | np.ndarray, offset: float = 0.1) -> float:
    values = np.asarray(alpha_values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 1.1
    return float(np.round(max(1.1, float(np.max(finite_values) + offset)), 6))


def compute_mutation_only_alpha_baseline_row(combined_df: pd.DataFrame, pseudocount: float = 1e-16) -> Optional[Dict[str, object]]:
    from Functions_HuggingFace import evaluate_alpha_sweep

    required_cols = {"lineage", "position", "ref_aa", "aa", "mut_prob", "obs_freq", "obs_present", "depth"}
    if combined_df.empty or not required_cols.issubset(combined_df.columns):
        return None

    baseline_df = combined_df.loc[:, sorted(required_cols)].drop_duplicates().copy()
    if baseline_df.empty:
        return None

    baseline_df["plm_prob"] = 1.0
    baseline_metrics = evaluate_alpha_sweep(
        baseline_df,
        np.array([1.0]),
        parallel=False,
        pseudocount=pseudocount,
    )
    if baseline_metrics.empty:
        return None

    logistic_metrics = compute_mutation_row_logistic_metrics(
        np.log10(np.clip(baseline_df["mut_prob"], 1e-32, None)),
        baseline_df["obs_present"],
        feature_name="log_mut_prob",
    )

    row = baseline_metrics.iloc[0].to_dict()
    row.update(
        {
            "alpha": np.nan,
            "alpha_label": "mutation_only",
            "model_variant": "mutation_accessibility_only",
            "is_mutation_only_baseline": True,
            "input_score_formula": "mut_prob",
        }
    )
    row.update(logistic_metrics)
    return row


def summarize_lineage_metric_table(lineage_metric_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if lineage_metric_df.empty:
        return pd.DataFrame()

    existing_group_cols = [col for col in group_cols if col in lineage_metric_df.columns]
    excluded_cols = set(existing_group_cols) | {"lineage"}
    numeric_cols = [
        col for col in lineage_metric_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(lineage_metric_df[col])
    ]

    if existing_group_cols:
        summary_df = lineage_metric_df.groupby(existing_group_cols, as_index=False, dropna=False)[numeric_cols].mean()
        if "lineage" in lineage_metric_df.columns:
            count_df = lineage_metric_df.groupby(existing_group_cols, as_index=False, dropna=False).agg(
                n_lineages_averaged=("lineage", "nunique")
            )
            summary_df = summary_df.merge(count_df, on=existing_group_cols, how="left")
        sort_cols = [col for col in ["model", "epoch_value", "epoch_label", "alpha"] if col in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        return summary_df

    summary_row = {
        col: float(pd.to_numeric(lineage_metric_df[col], errors="coerce").mean())
        for col in numeric_cols
    }
    if "lineage" in lineage_metric_df.columns:
        summary_row["n_lineages_averaged"] = int(lineage_metric_df["lineage"].nunique())
    return pd.DataFrame([summary_row])


def evaluate_alpha_sweep_by_lineage(
    combined_df: pd.DataFrame,
    alpha_grid: np.ndarray,
    *,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    alpha_sweep_min_grid: int = 10,
    pseudocount: float = 1e-16,
) -> pd.DataFrame:
    from Functions_HuggingFace import evaluate_alpha_sweep

    if combined_df.empty or "lineage" not in combined_df.columns:
        return pd.DataFrame()

    lineage_frames: List[pd.DataFrame] = []
    for lineage_name, lineage_df in combined_df.groupby("lineage", sort=False):
        lineage_alpha_df = evaluate_alpha_sweep(
            lineage_df,
            alpha_grid,
            parallel=parallel,
            max_workers=max_workers,
            alpha_sweep_min_grid=alpha_sweep_min_grid,
            pseudocount=pseudocount,
        )
        if not lineage_alpha_df.empty:
            lineage_alpha_df = lineage_alpha_df.copy()
            lineage_alpha_df["lineage"] = lineage_name
            lineage_alpha_df["alpha_label"] = lineage_alpha_df["alpha"].map(lambda value: f"{float(value):.6g}")
            lineage_alpha_df["model_variant"] = "plm_alpha_sweep"
            lineage_alpha_df["is_mutation_only_baseline"] = False
            lineage_alpha_df["input_score_formula"] = "plm_prob * mut_prob^alpha"
            lineage_frames.append(lineage_alpha_df)

        baseline_row = compute_mutation_only_alpha_baseline_row(lineage_df, pseudocount=pseudocount)
        if baseline_row is not None:
            baseline_df = pd.DataFrame([baseline_row])
            baseline_df["lineage"] = lineage_name
            lineage_frames.append(baseline_df)

    if not lineage_frames:
        return pd.DataFrame()
    return pd.concat(lineage_frames, ignore_index=True, sort=False)


def evaluate_logistic_alpha_sweep_by_lineage(
    combined_df: pd.DataFrame,
    alpha_grid: np.ndarray,
) -> pd.DataFrame:
    required_cols = {"lineage", "plm_prob", "mut_prob", "obs_present"}
    if combined_df.empty or not required_cols.issubset(combined_df.columns):
        return pd.DataFrame()

    logistic_rows: List[Dict[str, object]] = []
    alpha_values = sorted(np.asarray(alpha_grid, dtype=float).tolist())
    for lineage_name, lineage_df in combined_df.groupby("lineage", sort=False):
        score_base_df = lineage_df.loc[:, ["plm_prob", "mut_prob", "obs_present"]].copy()
        score_base_df["plm_prob"] = score_base_df["plm_prob"].clip(lower=1e-32)
        score_base_df["mut_prob"] = score_base_df["mut_prob"].clip(lower=1e-32)
        for alpha_value in alpha_values:
            combined_score = np.log10(score_base_df["plm_prob"]) + (float(alpha_value) * np.log10(score_base_df["mut_prob"]))
            try:
                logistic_row = compute_mutation_row_logistic_metrics(
                    combined_score,
                    score_base_df["obs_present"],
                    feature_name="combined_score",
                )
            except Exception:
                logistic_row = {metric_col: np.nan for metric_col in LOGISTIC_ALPHA_METRIC_COLUMNS}
            logistic_row["lineage"] = lineage_name
            logistic_row["alpha"] = float(alpha_value)
            logistic_rows.append(logistic_row)

    return pd.DataFrame(logistic_rows)


def _coerce_numeric_series(values: pd.Series) -> pd.Series:
    cleaned = values.replace(r"^\s*$", np.nan, regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _merge_logistic_alpha_metrics(
    alpha_by_lineage_df: pd.DataFrame,
    logistic_alpha_by_lineage_df: pd.DataFrame,
) -> pd.DataFrame:
    if alpha_by_lineage_df.empty or logistic_alpha_by_lineage_df.empty:
        return alpha_by_lineage_df

    merge_keys = [
        key for key in ["model", "epoch_label", "epoch_value", "lineage", "alpha"]
        if key in alpha_by_lineage_df.columns and key in logistic_alpha_by_lineage_df.columns
    ]
    if "lineage" not in merge_keys or "alpha" not in merge_keys:
        return alpha_by_lineage_df

    merged = alpha_by_lineage_df.copy()
    target_mask = merged["model_variant"].astype(str).eq("plm_alpha_sweep") if "model_variant" in merged.columns else merged["alpha"].notna()
    if not target_mask.any():
        return merged

    target_rows = merged.loc[target_mask].copy()
    target_rows["__row_id__"] = target_rows.index
    logistic_payload = logistic_alpha_by_lineage_df.loc[:, merge_keys + list(LOGISTIC_ALPHA_METRIC_COLUMNS)].copy()
    target_rows = target_rows.merge(logistic_payload, on=merge_keys, how="left", suffixes=("", "_logistic"))

    for metric_col in LOGISTIC_ALPHA_METRIC_COLUMNS:
        existing_values = _coerce_numeric_series(target_rows[metric_col]) if metric_col in target_rows.columns else pd.Series(np.nan, index=target_rows.index, dtype=float)
        logistic_values = _coerce_numeric_series(target_rows[f"{metric_col}_logistic"]) if f"{metric_col}_logistic" in target_rows.columns else pd.Series(np.nan, index=target_rows.index, dtype=float)
        target_rows[metric_col] = existing_values.where(existing_values.notna(), logistic_values)

    merged.loc[target_rows["__row_id__"], list(LOGISTIC_ALPHA_METRIC_COLUMNS)] = target_rows.loc[:, list(LOGISTIC_ALPHA_METRIC_COLUMNS)].to_numpy()
    return merged


def _alpha_table_has_complete_logistic_metrics(alpha_df: pd.DataFrame) -> bool:
    if alpha_df.empty:
        return True
    if any(metric_col not in alpha_df.columns for metric_col in LOGISTIC_ALPHA_METRIC_COLUMNS):
        return False

    for metric_col in LOGISTIC_ALPHA_METRIC_COLUMNS:
        if _coerce_numeric_series(alpha_df[metric_col]).isna().any():
            return False
    return True


def _build_alpha_tables_from_combined(
    model_combined_df: pd.DataFrame,
    alpha_grid: np.ndarray,
    *,
    model_label: str,
    model_spec: Dict[str, object],
    parallel: bool = False,
    max_workers: Optional[int] = None,
    alpha_sweep_min_grid: int = 10,
    pseudocount: float = 1e-16,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    alpha_by_lineage_df = evaluate_alpha_sweep_by_lineage(
        model_combined_df,
        alpha_grid,
        parallel=parallel,
        max_workers=max_workers,
        alpha_sweep_min_grid=alpha_sweep_min_grid,
        pseudocount=pseudocount,
    )
    if alpha_by_lineage_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    alpha_by_lineage_df["model"] = model_label
    alpha_by_lineage_df["epoch_label"] = model_spec["epoch_label"]
    alpha_by_lineage_df["epoch_value"] = float(model_spec["epoch_value"])
    logistic_alpha_by_lineage_df = evaluate_logistic_alpha_sweep_by_lineage(
        model_combined_df,
        alpha_grid,
    )
    if not logistic_alpha_by_lineage_df.empty:
        logistic_alpha_by_lineage_df["model"] = model_label
        logistic_alpha_by_lineage_df["epoch_label"] = model_spec["epoch_label"]
        logistic_alpha_by_lineage_df["epoch_value"] = float(model_spec["epoch_value"])
        alpha_by_lineage_df = _merge_logistic_alpha_metrics(alpha_by_lineage_df, logistic_alpha_by_lineage_df)

    alpha_df = summarize_lineage_metric_table(
        alpha_by_lineage_df,
        group_cols=[
            "model",
            "epoch_label",
            "epoch_value",
            "alpha",
            "alpha_label",
            "model_variant",
            "is_mutation_only_baseline",
            "input_score_formula",
        ],
    )
    return alpha_df, alpha_by_lineage_df


def fit_logistic_feature_model(
    feature_frame: pd.DataFrame,
    binary_outcome: pd.Series,
    *,
    enforce_positive_coefficients: bool = False,
) -> Dict[str, object]:
    from scipy.optimize import minimize
    from scipy.special import expit

    feature_names = list(feature_frame.columns)
    result: Dict[str, object] = {
        "logistic_n_obs": 0,
        "logistic_positive_rate": np.nan,
        "logistic_intercept": np.nan,
        "logistic_tjur_r2": np.nan,
        "logistic_fitted_prob_corr": np.nan,
        "logistic_auroc": np.nan,
        "logistic_pr_auc": np.nan,
        "logistic_mcfadden_r2": np.nan,
        "logistic_brier_score": np.nan,
        "logistic_log_likelihood": np.nan,
        "logistic_null_log_likelihood": np.nan,
        "logistic_active_predictors": "",
        "logistic_response_definition": "binary_mutation_observed",
        "logistic_constraint": "non_negative_coefficients" if enforce_positive_coefficients else "unconstrained",
    }
    for feature_name in feature_names:
        result[f"logistic_coef_{feature_name}"] = np.nan

    x = np.asarray(feature_frame, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(binary_outcome, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if valid.sum() < 3:
        return result

    x = x[valid]
    y = y[valid]
    result["logistic_n_obs"] = int(len(y))
    result["logistic_positive_rate"] = float(np.mean(y)) if len(y) > 0 else np.nan
    if np.unique(y).size < 2:
        return result

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    active_mask = np.isfinite(x_std) & (x_std > 0)
    result["logistic_active_predictors"] = ",".join([name for name, is_active in zip(feature_names, active_mask) if is_active])
    if not np.any(active_mask):
        return result

    x_active = x[:, active_mask]
    mean_active = x_mean[active_mask]
    std_active = x_std[active_mask]
    x_scaled = (x_active - mean_active) / std_active

    y_mean = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    intercept_seed = float(np.log(y_mean / (1.0 - y_mean)))
    slope_seed = np.zeros(x_scaled.shape[1], dtype=float)
    if np.std(y) > 0:
        for feature_idx in range(x_scaled.shape[1]):
            corr_seed = float(np.corrcoef(x_scaled[:, feature_idx], y)[0, 1])
            if not np.isfinite(corr_seed):
                corr_seed = 0.0
            slope_seed[feature_idx] = float(np.clip(4.0 * corr_seed, -8.0, 8.0))

    def neg_log_likelihood(params: np.ndarray) -> float:
        intercept = params[0]
        slopes = params[1:]
        logits = intercept + np.dot(x_scaled, slopes)
        probs = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        return float(-np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))

    if enforce_positive_coefficients:
        bounds = [(-20.0, 20.0)] + [(0.0, 20.0)] * x_scaled.shape[1]
        slope_seed = np.abs(slope_seed)
    else:
        bounds = [(-20.0, 20.0)] * (x_scaled.shape[1] + 1)
    initial_guesses = [
        np.concatenate(([0.0], np.zeros(x_scaled.shape[1], dtype=float))),
        np.concatenate(([intercept_seed], np.zeros(x_scaled.shape[1], dtype=float))),
        np.concatenate(([intercept_seed], slope_seed)),
        np.concatenate(([0.0], slope_seed)),
    ]
    if not enforce_positive_coefficients:
        initial_guesses.extend([
            np.concatenate(([intercept_seed], -slope_seed)),
            np.concatenate(([0.0], -slope_seed)),
        ])
    methods = [
        ("L-BFGS-B", {"bounds": bounds}),
        ("Powell", {"bounds": bounds}),
    ]
    if not enforce_positive_coefficients:
        methods.extend([
            ("Nelder-Mead", {}),
            ("BFGS", {}),
        ])
    best_fit = None
    best_nll = np.inf
    for x0 in initial_guesses:
        for method_name, extra_kwargs in methods:
            try:
                fit = minimize(
                    neg_log_likelihood,
                    x0=x0,
                    method=method_name,
                    options={"maxiter": 2000},
                    **extra_kwargs,
                )
            except Exception:
                continue
            fit_nll = float(fit.fun) if np.isfinite(fit.fun) else np.inf
            if fit.success and fit_nll < best_nll and np.all(np.isfinite(fit.x)):
                best_fit = fit
                best_nll = fit_nll

    if best_fit is None:
        return result

    intercept_scaled = float(best_fit.x[0])
    slopes_scaled = np.asarray(best_fit.x[1:], dtype=float)
    raw_coefs_active = slopes_scaled / std_active
    raw_intercept = intercept_scaled - float(np.sum((slopes_scaled * mean_active) / std_active))
    raw_coefs = np.zeros(len(feature_names), dtype=float)
    raw_coefs[active_mask] = raw_coefs_active
    fitted_probs = expit(raw_intercept + np.dot(x, raw_coefs))
    base_prob = float(np.clip(np.mean(y), 1e-9, 1.0 - 1e-9))
    null_nll = float(-np.sum(y * np.log(base_prob) + (1.0 - y) * np.log(1.0 - base_prob)))
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    tjur_r2 = float(np.mean(fitted_probs[pos_mask]) - np.mean(fitted_probs[neg_mask])) if pos_mask.any() and neg_mask.any() else np.nan
    fitted_prob_corr = float(np.corrcoef(fitted_probs, y)[0, 1]) if np.std(fitted_probs) > 0 and np.std(y) > 0 else np.nan
    auroc = safe_auroc(pd.Series(fitted_probs), pd.Series(y))
    pr_auc = safe_pr_auc(pd.Series(fitted_probs), pd.Series(y))

    result.update(
        {
            "logistic_intercept": raw_intercept,
            "logistic_tjur_r2": tjur_r2,
            "logistic_fitted_prob_corr": fitted_prob_corr,
            "logistic_auroc": auroc,
            "logistic_pr_auc": pr_auc,
            "logistic_mcfadden_r2": float(1.0 - (best_nll / null_nll)) if np.isfinite(null_nll) and null_nll > 0 else np.nan,
            "logistic_brier_score": float(np.mean(np.square(y - fitted_probs))),
            "logistic_log_likelihood": float(-best_nll),
            "logistic_null_log_likelihood": float(-null_nll) if np.isfinite(null_nll) else np.nan,
        }
    )
    for feature_name, coef_value in zip(feature_names, raw_coefs):
        result[f"logistic_coef_{feature_name}"] = float(coef_value)
    return result


def fit_linear_frequency_model(feature_frame: pd.DataFrame, obs_freq: pd.Series) -> Dict[str, object]:
    feature_names = list(feature_frame.columns)
    result: Dict[str, object] = {
        "freq_regression_method": "ordinary_least_squares",
        "freq_n_obs": 0,
        "freq_intercept": np.nan,
        "freq_log10_r2": np.nan,
        "freq_log10_adjusted_r2": np.nan,
        "freq_log10_rmse": np.nan,
        "freq_log10_pearson_r": np.nan,
        "freq_log10_spearman_r": np.nan,
        "freq_raw_intercept": np.nan,
        "freq_raw_r2": np.nan,
        "freq_raw_adjusted_r2": np.nan,
        "freq_raw_rmse": np.nan,
        "freq_raw_pearson_r": np.nan,
        "freq_raw_spearman_r": np.nan,
        "freq_active_predictors": "",
        "freq_response_definition": "log10_nonzero_observed_frequency",
        "freq_raw_response_definition": "nonzero_observed_frequency",
        "freq_mean_observed_frequency": np.nan,
        "freq_mean_log10_observed_frequency": np.nan,
    }
    for feature_name in feature_names:
        result[f"freq_coef_{feature_name}"] = np.nan
        result[f"freq_raw_coef_{feature_name}"] = np.nan

    x = np.asarray(feature_frame, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y_raw = np.asarray(obs_freq, dtype=float)
    valid = np.isfinite(y_raw) & (y_raw > 0) & np.all(np.isfinite(x), axis=1)
    if valid.sum() < 2:
        return result

    x = x[valid]
    y_log = np.log10(np.clip(y_raw[valid], 1e-32, None))
    result["freq_n_obs"] = int(len(y_log))
    result["freq_mean_observed_frequency"] = float(np.mean(y_raw[valid]))
    result["freq_mean_log10_observed_frequency"] = float(np.mean(y_log))
    x_std = np.std(x, axis=0)
    active_mask = np.isfinite(x_std) & (x_std > 0)
    result["freq_active_predictors"] = ",".join([name for name, is_active in zip(feature_names, active_mask) if is_active])
    if not np.any(active_mask):
        return result

    x_active = x[:, active_mask]
    design = np.column_stack([np.ones(len(y_log), dtype=float), x_active])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(design, y_log, rcond=None)
    except np.linalg.LinAlgError:
        return result

    fitted = design @ coeffs
    residuals = y_log - fitted
    rss = float(np.sum(np.square(residuals)))
    tss = float(np.sum(np.square(y_log - np.mean(y_log))))
    r2 = float(1.0 - (rss / tss)) if tss > 0 else np.nan
    n_obs = len(y_log)
    n_params = int(1 + np.sum(active_mask))
    adjusted_r2 = float(1.0 - ((1.0 - r2) * (n_obs - 1) / (n_obs - n_params))) if np.isfinite(r2) and n_obs > n_params else np.nan
    raw_coefs = np.zeros(len(feature_names), dtype=float)
    raw_coefs[active_mask] = np.asarray(coeffs[1:], dtype=float)

    y_nonzero = y_raw[valid]
    try:
        coeffs_raw, _, _, _ = np.linalg.lstsq(design, y_nonzero, rcond=None)
    except np.linalg.LinAlgError:
        coeffs_raw = None

    raw_fit = np.full(len(y_nonzero), np.nan, dtype=float)
    raw_residuals = np.full(len(y_nonzero), np.nan, dtype=float)
    raw_r2 = np.nan
    raw_adjusted_r2 = np.nan
    raw_rmse = np.nan
    raw_pearson_r = np.nan
    raw_spearman_r = np.nan
    raw_scale_coefs = np.zeros(len(feature_names), dtype=float)
    raw_intercept = np.nan
    if coeffs_raw is not None:
        raw_fit = design @ coeffs_raw
        raw_residuals = y_nonzero - raw_fit
        raw_rss = float(np.sum(np.square(raw_residuals)))
        raw_tss = float(np.sum(np.square(y_nonzero - np.mean(y_nonzero))))
        raw_r2 = float(1.0 - (raw_rss / raw_tss)) if raw_tss > 0 else np.nan
        raw_adjusted_r2 = float(1.0 - ((1.0 - raw_r2) * (n_obs - 1) / (n_obs - n_params))) if np.isfinite(raw_r2) and n_obs > n_params else np.nan
        raw_rmse = float(np.sqrt(np.mean(np.square(raw_residuals))))
        raw_pearson_r = safe_pearson(pd.Series(raw_fit), pd.Series(y_nonzero))
        raw_spearman_r = safe_spearman(pd.Series(raw_fit), pd.Series(y_nonzero))
        raw_scale_coefs[active_mask] = np.asarray(coeffs_raw[1:], dtype=float)
        raw_intercept = float(coeffs_raw[0])

    result.update(
        {
            "freq_intercept": float(coeffs[0]),
            "freq_log10_r2": r2,
            "freq_log10_adjusted_r2": adjusted_r2,
            "freq_log10_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
            "freq_log10_pearson_r": safe_pearson(pd.Series(fitted), pd.Series(y_log)),
            "freq_log10_spearman_r": safe_spearman(pd.Series(fitted), pd.Series(y_log)),
            "freq_raw_intercept": raw_intercept,
            "freq_raw_r2": raw_r2,
            "freq_raw_adjusted_r2": raw_adjusted_r2,
            "freq_raw_rmse": raw_rmse,
            "freq_raw_pearson_r": raw_pearson_r,
            "freq_raw_spearman_r": raw_spearman_r,
        }
    )
    for feature_name, coef_value in zip(feature_names, raw_coefs):
        result[f"freq_coef_{feature_name}"] = float(coef_value)
    for feature_name, coef_value in zip(feature_names, raw_scale_coefs):
        result[f"freq_raw_coef_{feature_name}"] = float(coef_value)
    return result


def build_hurdle_base_frame(plot_frame: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"obs_present", "obs_freq", "plm_prob", "mut_prob"}
    if plot_frame.empty or not required_cols.issubset(plot_frame.columns):
        return pd.DataFrame()
    base_df = plot_frame.loc[:, sorted(required_cols)].copy()
    base_df["obs_present"] = pd.to_numeric(base_df["obs_present"], errors="coerce")
    base_df["obs_freq"] = pd.to_numeric(base_df["obs_freq"], errors="coerce")
    base_df["plm_prob"] = pd.to_numeric(base_df["plm_prob"], errors="coerce").clip(lower=1e-32)
    base_df["mut_prob"] = pd.to_numeric(base_df["mut_prob"], errors="coerce").clip(lower=1e-32)
    base_df["log_plm_prob"] = np.log10(base_df["plm_prob"])
    base_df["log_mut_prob"] = np.log10(base_df["mut_prob"])
    valid = np.isfinite(base_df["obs_present"]) & np.isfinite(base_df["obs_freq"]) & np.isfinite(base_df["log_plm_prob"]) & np.isfinite(base_df["log_mut_prob"])
    return base_df.loc[valid].reset_index(drop=True)


def evaluate_hurdle_alpha_sweep(plot_frame: pd.DataFrame, alpha_values: List[float]) -> pd.DataFrame:
    base_df = build_hurdle_base_frame(plot_frame)
    if base_df.empty or not alpha_values:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    sorted_alphas = sorted({float(alpha) for alpha in alpha_values})
    for alpha_presence in sorted_alphas:
        binary_score = base_df["log_plm_prob"] + (alpha_presence * base_df["log_mut_prob"])
        logistic_metrics = fit_logistic_feature_model(
            pd.DataFrame({"combined_score": binary_score}),
            base_df["obs_present"],
            enforce_positive_coefficients=True,
        )
        for alpha_frequency in sorted_alphas:
            positive_score = base_df["log_plm_prob"] + (alpha_frequency * base_df["log_mut_prob"])
            freq_metrics = fit_linear_frequency_model(pd.DataFrame({"combined_score": positive_score}), base_df["obs_freq"])
            row: Dict[str, object] = {
                "alpha_presence": float(alpha_presence),
                "alpha_frequency": float(alpha_frequency),
                "presence_score_formula": "log10(plm_prob) + alpha_presence * log10(mut_prob)",
                "frequency_score_formula": "log10(plm_prob) + alpha_frequency * log10(mut_prob)",
                "presence_stage_feature_set": "combined_score",
                "frequency_stage_feature_set": "combined_score",
                "hurdle_mean_r2": mean_finite([
                    float(logistic_metrics.get("logistic_tjur_r2", np.nan)),
                    float(freq_metrics.get("freq_log10_r2", np.nan)),
                ]),
                "hurdle_mean_r2_raw_frequency": mean_finite([
                    float(logistic_metrics.get("logistic_tjur_r2", np.nan)),
                    float(freq_metrics.get("freq_raw_r2", np.nan)),
                ]),
            }
            row.update(logistic_metrics)
            row.update(freq_metrics)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["alpha_presence", "alpha_frequency"]).reset_index(drop=True)


def summarize_hurdle_models(plot_frame: pd.DataFrame, hurdle_alpha_df: pd.DataFrame) -> pd.DataFrame:
    base_df = build_hurdle_base_frame(plot_frame)
    if base_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    baseline_presence_alpha = compute_baseline_alpha_position(hurdle_alpha_df["alpha_presence"]) if not hurdle_alpha_df.empty else 1.1
    baseline_frequency_alpha = compute_baseline_alpha_position(hurdle_alpha_df["alpha_frequency"]) if not hurdle_alpha_df.empty else 1.1
    mutation_only_logistic = fit_logistic_feature_model(
        base_df.loc[:, ["log_mut_prob"]],
        base_df["obs_present"],
        enforce_positive_coefficients=True,
    )
    mutation_only_freq = fit_linear_frequency_model(base_df.loc[:, ["log_mut_prob"]], base_df["obs_freq"])
    mutation_only_row: Dict[str, object] = {
        "model_variant": "mutation_only",
        "alpha_presence": baseline_presence_alpha,
        "alpha_frequency": baseline_frequency_alpha,
        "presence_stage_feature_set": "log_mut_prob",
        "frequency_stage_feature_set": "log_mut_prob",
        "hurdle_mean_r2": mean_finite([
            float(mutation_only_logistic.get("logistic_tjur_r2", np.nan)),
            float(mutation_only_freq.get("freq_log10_r2", np.nan)),
        ]),
        "hurdle_mean_r2_raw_frequency": mean_finite([
            float(mutation_only_logistic.get("logistic_tjur_r2", np.nan)),
            float(mutation_only_freq.get("freq_raw_r2", np.nan)),
        ]),
    }
    mutation_only_row.update(mutation_only_logistic)
    mutation_only_row.update(mutation_only_freq)
    rows.append(mutation_only_row)
    plm_only_rows = hurdle_alpha_df.loc[
        np.isclose(pd.to_numeric(hurdle_alpha_df["alpha_presence"], errors="coerce"), 0.0)
        & np.isclose(pd.to_numeric(hurdle_alpha_df["alpha_frequency"], errors="coerce"), 0.0)
    ] if not hurdle_alpha_df.empty else pd.DataFrame()
    if not plm_only_rows.empty:
        plm_only_row = plm_only_rows.iloc[0].to_dict()
    else:
        plm_only_logistic = fit_logistic_feature_model(
            base_df.loc[:, ["log_plm_prob"]],
            base_df["obs_present"],
            enforce_positive_coefficients=True,
        )
        plm_only_freq = fit_linear_frequency_model(base_df.loc[:, ["log_plm_prob"]], base_df["obs_freq"])
        plm_only_row = {
            "alpha_presence": 0.0,
            "alpha_frequency": 0.0,
            "presence_score_formula": "log10(plm_prob) + alpha_presence * log10(mut_prob)",
            "frequency_score_formula": "log10(plm_prob) + alpha_frequency * log10(mut_prob)",
            "presence_stage_feature_set": "combined_score",
            "frequency_stage_feature_set": "combined_score",
            "hurdle_mean_r2": mean_finite([
                float(plm_only_logistic.get("logistic_tjur_r2", np.nan)),
                float(plm_only_freq.get("freq_log10_r2", np.nan)),
            ]),
            "hurdle_mean_r2_raw_frequency": mean_finite([
                float(plm_only_logistic.get("logistic_tjur_r2", np.nan)),
                float(plm_only_freq.get("freq_raw_r2", np.nan)),
            ]),
        }
        plm_only_row.update(plm_only_logistic)
        plm_only_row.update(plm_only_freq)
    plm_only_row["model_variant"] = "plm_only_alpha0"
    rows.append(plm_only_row)
    if not hurdle_alpha_df.empty:
        best_idx = hurdle_alpha_df["hurdle_mean_r2"].astype(float).idxmax() if hurdle_alpha_df["hurdle_mean_r2"].notna().any() else hurdle_alpha_df.index[0]
        best_row = hurdle_alpha_df.loc[best_idx].to_dict()
        best_row["model_variant"] = "best_alpha_hurdle"
        rows.append(best_row)
    two_input_features = base_df.loc[:, ["log_mut_prob", "log_plm_prob"]]
    two_input_logistic = fit_logistic_feature_model(
        two_input_features,
        base_df["obs_present"],
        enforce_positive_coefficients=True,
    )
    two_input_freq = fit_linear_frequency_model(two_input_features, base_df["obs_freq"])
    two_input_row = {
        "model_variant": "two_input_hurdle",
        "alpha_presence": np.nan,
        "alpha_frequency": np.nan,
        "presence_stage_feature_set": "log_mut_prob,log_plm_prob",
        "frequency_stage_feature_set": "log_mut_prob,log_plm_prob",
        "hurdle_mean_r2": mean_finite([
            float(two_input_logistic.get("logistic_tjur_r2", np.nan)),
            float(two_input_freq.get("freq_log10_r2", np.nan)),
        ]),
        "hurdle_mean_r2_raw_frequency": mean_finite([
            float(two_input_logistic.get("logistic_tjur_r2", np.nan)),
            float(two_input_freq.get("freq_raw_r2", np.nan)),
        ]),
    }
    two_input_row.update(two_input_logistic)
    two_input_row.update(two_input_freq)
    rows.append(two_input_row)
    return pd.DataFrame(rows)


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

        logit_plm_corr, logit_plm_intercept, logit_plm_slope = validate_site_logistic_fit_agreement(
            site_df["site_plm_score"],
            site_df["site_mutated"],
            context_label=f"epoch lineage metrics PLM ({model_tag}, {lineage_name})",
        )
        logit_mut_corr, logit_mut_intercept, logit_mut_slope = validate_site_logistic_fit_agreement(
            site_df["site_mut_score"],
            site_df["site_mutated"],
            context_label=f"epoch lineage metrics mutational accessibility ({model_tag}, {lineage_name})",
        )

        rows.append(
            {
                "model": model_tag,
                "epoch_label": epoch_label,
                "epoch_value": float(epoch_value),
                "lineage": lineage_name,
                "n_mutation_rows": int(len(lineage_df)),
                "n_site_rows": int(len(site_df)),
                "n_nonzero_mutation_rows": int((pd.to_numeric(lineage_df["obs_freq"], errors="coerce") > 0).sum()),
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
                "spearman_obs_freq_mutated_vs_plm": safe_spearman(
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "plm_prob"],
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "obs_freq"],
                ),
                "spearman_obs_freq_mutated_vs_mut_baseline": safe_spearman(
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "mut_prob"],
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "obs_freq"],
                ),
                "pearson_obs_freq_mutated_vs_plm": safe_pearson(
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "plm_prob"],
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "obs_freq"],
                ),
                "pearson_obs_freq_mutated_vs_mut_baseline": safe_pearson(
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "mut_prob"],
                    lineage_df.loc[lineage_df["obs_freq"] > 0, "obs_freq"],
                ),
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
        "spearman_obs_freq_mutated_vs_plm",
        "spearman_obs_freq_mutated_vs_mut_baseline",
        "pearson_obs_freq_mutated_vs_plm",
        "pearson_obs_freq_mutated_vs_mut_baseline",
        "spearman_plm_vs_mut",
        "spearman_mut_vs_mut_baseline",
        "pearson_plm_vs_mut",
        "pearson_mut_vs_mut_baseline",
    ]
    metric_summary = (
        epoch_lineage_metrics_df.groupby(["model", "epoch_label", "epoch_value"], as_index=False)[metric_cols]
        .mean()
    )
    count_summary = (
        epoch_lineage_metrics_df.groupby(["model", "epoch_label", "epoch_value"], as_index=False)
        .agg(
            n_lineages=("lineage", "nunique"),
            mean_n_mutation_rows=("n_mutation_rows", "mean"),
            mean_n_site_rows=("n_site_rows", "mean"),
            mean_n_nonzero_mutation_rows=("n_nonzero_mutation_rows", "mean"),
        )
    )
    return metric_summary.merge(count_summary, on=["model", "epoch_label", "epoch_value"], how="left").sort_values(["epoch_value", "epoch_label"])


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
    metrics_output_dir: Optional[Path] = None,
    mutation_model: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
    from scipy.special import expit
    from scipy.stats import spearmanr
    from Functions_HuggingFace import evaluate_alpha_sweep, get_ranked_mutations
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    metrics_output_dir = ensure_dir(Path(metrics_output_dir) if metrics_output_dir is not None else output_dir)
    alpha_xlabel = "mutational accessibility weighting Alpha\n(mut_acc^alpha) 0= PLM only"
    title_fontsize = 14
    label_fontsize = 13
    tick_fontsize = 11
    legend_fontsize = 11
    plot_name_overrides = {
        "ESM2_650M_HA80": "HA80_fine-tune",
    }

    model_label_lookup = (
        combined_df.loc[:, [
            col for col in ["model", "model_display_label", "base_model", "checkpoint_label", "epoch_label"]
            if col in combined_df.columns
        ]]
        .drop_duplicates(subset=["model"], keep="first")
        .set_index("model")
        .to_dict("index")
        if "model" in combined_df.columns else {}
    )

    def _safe_output_label(text: object) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "model"

    def _epoch_colour_sequence(n_values: int) -> List[Tuple[float, float, float]]:
        return sns.color_palette("viridis", n_colors=max(1, n_values))

    def _plm_epoch_label(epoch_label: object, epoch_value: object, model_name: Optional[object] = None, model_display_label: Optional[object] = None) -> str:
        model_text = "" if model_name is None or pd.isna(model_name) else str(model_name).strip()
        if model_text in plot_name_overrides:
            return plot_name_overrides[model_text]

        display_text = "" if model_display_label is None or pd.isna(model_display_label) else str(model_display_label).strip()
        if display_text and display_text not in {"PLM raw"} and not display_text.startswith("PLM epoch "):
            return display_text

        metadata = model_label_lookup.get(model_text, {}) if model_name is not None else {}
        metadata_display = str(metadata.get("model_display_label", "")).strip() if metadata.get("model_display_label") is not None and not pd.isna(metadata.get("model_display_label")) else ""
        metadata_base_model = str(metadata.get("base_model", "")).strip() if metadata.get("base_model") is not None and not pd.isna(metadata.get("base_model")) else ""
        metadata_checkpoint = str(metadata.get("checkpoint_label", "")).strip() if metadata.get("checkpoint_label") is not None and not pd.isna(metadata.get("checkpoint_label")) else ""
        epoch_text = str(epoch_label)

        if epoch_text == "raw_model":
            if metadata_base_model:
                return metadata_base_model
            if metadata_display:
                return re.sub(r"\s+raw\s+base\s+model$", "", metadata_display, flags=re.IGNORECASE).strip() or metadata_display
            if model_text:
                return re.sub(r"(?:_raw| raw)$", "", model_text).strip("_ ") or model_text
            return "Base model"

        if metadata_display and metadata_display not in {"PLM raw"} and not metadata_display.startswith("PLM epoch "):
            return metadata_display
        if metadata_checkpoint and metadata_base_model:
            return f"{metadata_checkpoint} fine tuned ({metadata_base_model})"
        if metadata_checkpoint:
            return metadata_checkpoint
        if model_text:
            return model_text
        return str(epoch_label)

    def _format_count(value: object, decimals: int = 0) -> str:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        numeric_value = float(value)
        if decimals <= 0:
            return f"{int(round(numeric_value)):,}"
        return f"{numeric_value:,.{decimals}f}"

    def _append_sample_note(title: str, note: str) -> str:
        return title if not note else f"{title}\n{note}"

    def _valid_numeric_mask(frame: pd.DataFrame, required_cols: List[str]) -> pd.Series:
        if frame.empty:
            return pd.Series(False, index=frame.index, dtype=bool)
        mask = pd.Series(True, index=frame.index, dtype=bool)
        for col in required_cols:
            if col not in frame.columns:
                return pd.Series(False, index=frame.index, dtype=bool)
            mask &= pd.to_numeric(frame[col], errors="coerce").notna()
        return mask

    def _unique_row_count(frame: pd.DataFrame, unit_cols: List[str], valid_mask: pd.Series) -> Optional[int]:
        present_cols = [col for col in unit_cols if col in frame.columns]
        if not present_cols:
            return None
        return int(frame.loc[valid_mask, present_cols].drop_duplicates().shape[0])

    def _lineage_count_note(source_frame: pd.DataFrame, valid_mask: pd.Series, *, unit_label: str) -> str:
        if source_frame.empty or not valid_mask.any():
            return ""
        valid_frame = source_frame.loc[valid_mask].copy()
        if "lineage" in valid_frame.columns and {"position", "ref_aa"}.issubset(valid_frame.columns):
            site_counts = (
                valid_frame.loc[:, ["lineage", "position", "ref_aa"]]
                .drop_duplicates()
                .groupby("lineage", sort=False)
                .size()
            )
            if not site_counts.empty:
                n_lineages = _format_count(site_counts.shape[0])
                max_sites = _format_count(site_counts.max())
                if unit_label == "sites":
                    return f"{n_lineages} lineages, up to {max_sites} sites/lineage"
                if unit_label == "mutation_matrix":
                    return f"{n_lineages} lineages, up to {max_sites} sites/lineage (19xN matrix)"
                if unit_label == "nonzero_mutation_matrix":
                    return f"{n_lineages} lineages, non-zero mutations from up to {max_sites} sites/lineage"
        if "lineage" in valid_frame.columns:
            unit_counts = valid_frame.groupby("lineage", sort=False).size()
            if not unit_counts.empty:
                return f"{_format_count(unit_counts.shape[0])} lineages"
        fallback_cols = ["position", "ref_aa"] if unit_label == "sites" else ["aa", "position", "ref_aa"]
        unit_count = _unique_row_count(valid_frame, fallback_cols, pd.Series(True, index=valid_frame.index, dtype=bool))
        if unit_count is not None:
            fallback_label = "sites" if unit_label == "sites" else "mutation rows"
            return f"{_format_count(unit_count)} {fallback_label}"
        return ""

    def _lineage_mean_count_note(source_frame: pd.DataFrame, valid_mask: pd.Series, *, count_label: str) -> str:
        if source_frame.empty or not valid_mask.any() or "lineage" not in source_frame.columns:
            return ""
        valid_frame = source_frame.loc[valid_mask].copy()
        count_series = valid_frame.groupby("lineage", sort=False).size()
        if count_series.empty:
            return ""
        return f"{_format_count(count_series.shape[0])} lineages, mean {_format_count(count_series.mean(), 0)} {count_label}/lineage"

    def _sample_note_for_metric(metric_col: str, source_frame: pd.DataFrame, *, note_style: str = "matrix") -> str:
        if source_frame.empty:
            return ""

        if metric_col in {"site_top10pct_mutated_enrichment", "site_top10pct_mutated_precision", "site_rank_spearman_r", "mut_flat_mean_site_nll"}:
            valid_mask = _valid_numeric_mask(source_frame, ["plm_prob", "mut_prob", "obs_freq", "obs_present"])
            return _lineage_count_note(source_frame, valid_mask, unit_label="sites")

        if metric_col in {"mut_flat_global_spearman_r", "mut_flat_global_pearson_r", "mut_flat_logfreq_global_pearson_r"}:
            valid_mask = _valid_numeric_mask(source_frame, ["plm_prob", "obs_freq"])
            if note_style == "counts":
                return _lineage_mean_count_note(source_frame, valid_mask, count_label="candidate mutations")
            return _lineage_count_note(source_frame, valid_mask, unit_label="mutation_matrix")

        if metric_col in {"mut_flat_nonzero_spearman_r", "mut_flat_nonzero_pearson_r", "mut_flat_logfreq_nonzero_pearson_r"}:
            valid_mask = _valid_numeric_mask(source_frame, ["plm_prob", "obs_freq"])
            valid_mask &= pd.to_numeric(source_frame["obs_freq"], errors="coerce") > 0
            return _lineage_mean_count_note(source_frame, valid_mask, count_label="observed mutations")

        if metric_col in {
            "site_logistic_auroc",
            "site_logistic_pr_auc",
            "site_logistic_tjur_r2",
            "site_logistic_mcfadden_r2",
            "site_logistic_brier_score",
            "site_logistic_mutated_corr",
        }:
            valid_mask = _valid_numeric_mask(source_frame, ["plm_prob", "mut_prob", "obs_present"])
            if note_style == "counts":
                return _lineage_mean_count_note(source_frame, valid_mask, count_label="candidate mutations")
            return _lineage_count_note(source_frame, valid_mask, unit_label="mutation_matrix")

        return ""

    def _sample_note_for_epoch_metric(metric_col: str, epoch_summary_df_local: pd.DataFrame) -> str:
        if epoch_summary_df_local.empty:
            return ""
        mean_lineages = pd.to_numeric(epoch_summary_df_local.get("n_lineages"), errors="coerce") if "n_lineages" in epoch_summary_df_local.columns else pd.Series(dtype=float)
        mean_mut_rows = pd.to_numeric(epoch_summary_df_local.get("mean_n_mutation_rows"), errors="coerce") if "mean_n_mutation_rows" in epoch_summary_df_local.columns else pd.Series(dtype=float)
        mean_site_rows = pd.to_numeric(epoch_summary_df_local.get("mean_n_site_rows"), errors="coerce") if "mean_n_site_rows" in epoch_summary_df_local.columns else pd.Series(dtype=float)
        mean_nonzero_rows = pd.to_numeric(epoch_summary_df_local.get("mean_n_nonzero_mutation_rows"), errors="coerce") if "mean_n_nonzero_mutation_rows" in epoch_summary_df_local.columns else pd.Series(dtype=float)
        lineage_text = f"mean over {_format_count(mean_lineages.mean())} lineages/epoch" if not mean_lineages.empty and np.isfinite(mean_lineages.mean()) else ""

        if metric_col == "logistic_site_mutated_vs_plm_corr":
            row_text = f"mean n~{_format_count(mean_site_rows.mean(), 1)} lineage-site rows/lineage" if not mean_site_rows.empty and np.isfinite(mean_site_rows.mean()) else ""
        elif metric_col in {"spearman_obs_freq_mutated_vs_plm", "pearson_obs_freq_mutated_vs_plm"}:
            row_text = f"mean n~{_format_count(mean_nonzero_rows.mean(), 1)} non-zero mutation rows/lineage" if not mean_nonzero_rows.empty and np.isfinite(mean_nonzero_rows.mean()) else ""
        else:
            row_text = f"mean n~{_format_count(mean_mut_rows.mean(), 1)} mutation rows/lineage" if not mean_mut_rows.empty and np.isfinite(mean_mut_rows.mean()) else ""

        return "; ".join([text for text in [lineage_text, row_text] if text])

    def _hide_log_minor_ticks(ax) -> None:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)
        ax.xaxis.set_major_locator(LogLocator(base=100.0, subs=(1.0,), numticks=100))
        ax.yaxis.set_major_locator(LogLocator(base=100.0, subs=(1.0,), numticks=100))

    def _two_dp_formatter(value, _pos) -> str:
        return f"{value:.2f}"

    def _style_axis_text(ax, *, is_alpha_x: bool = False) -> None:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(FuncFormatter(_two_dp_formatter))
        if ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(FuncFormatter(_two_dp_formatter))
        if is_alpha_x:
            ax.set_xlabel(alpha_xlabel, fontsize=label_fontsize)
        else:
            ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)
        ax.title.set_fontsize(title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    def _dedupe_legend_entries(handles, labels):
        unique_handles = []
        unique_labels = []
        seen_labels = set()
        for handle, label in zip(handles, labels):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        return unique_handles, unique_labels

    def _collect_axes_legend_entries(axes) -> Tuple[List[object], List[str]]:
        handles_all: List[object] = []
        labels_all: List[str] = []
        for ax in np.array(axes, dtype=object).reshape(-1):
            handles, labels = ax.get_legend_handles_labels()
            handles_all.extend(handles)
            labels_all.extend(labels)
        return _dedupe_legend_entries(handles_all, labels_all)

    def _model_family_key(model_label: object, epoch_label: object) -> str:
        model_text = str(model_label)
        epoch_text = str(epoch_label)
        if epoch_text == "raw_model" and model_text.endswith("_raw"):
            return model_text[:-4]
        suffix = f"_{epoch_text}"
        if epoch_text and model_text.endswith(suffix):
            return model_text[: -len(suffix)]
        return model_text

    def _compute_mutation_only_alpha_baseline(
        plot_frame: pd.DataFrame,
        baseline_output_path: Optional[Path] = None,
    ) -> Optional[pd.Series]:
        required_cols = {"lineage", "position", "ref_aa", "aa", "mut_prob", "obs_freq", "obs_present", "depth"}
        if plot_frame.empty or not required_cols.issubset(plot_frame.columns):
            return None

        baseline_rows: List[Dict[str, object]] = []
        for lineage_name, lineage_df in plot_frame.groupby("lineage", sort=False):
            baseline_row = compute_mutation_only_alpha_baseline_row(lineage_df, pseudocount=1e-16)
            if baseline_row is None:
                continue
            baseline_row["lineage"] = lineage_name
            baseline_rows.append(baseline_row)
        if not baseline_rows:
            return None

        baseline_metrics = summarize_lineage_metric_table(
            pd.DataFrame(baseline_rows),
            group_cols=["alpha", "alpha_label", "model_variant", "is_mutation_only_baseline", "input_score_formula"],
        )
        if baseline_metrics.empty:
            return None
        if baseline_output_path is not None:
            baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_metrics.to_csv(baseline_output_path, sep="\t", index=False)
        return baseline_metrics.iloc[0]

    def _baseline_alpha_x(alpha_frame: pd.DataFrame) -> float:
        if "alpha" not in alpha_frame.columns:
            return 1.1
        return compute_baseline_alpha_position(alpha_frame["alpha"])

    def _compute_site_logistic_alpha_metrics(
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        required_cols = {"plm_prob", "mut_prob", "obs_present"}
        if plot_frame.empty or alpha_frame.empty or not required_cols.issubset(plot_frame.columns):
            return pd.DataFrame()

        group_cols = [
            col for col in ["model", "epoch_label", "epoch_value"]
            if col in plot_frame.columns and col in alpha_frame.columns
        ]
        logistic_rows: List[Dict[str, object]] = []

        if group_cols:
            group_keys = alpha_frame.loc[:, group_cols].drop_duplicates().to_dict("records")
        else:
            group_keys = [{}]

        for group_key in group_keys:
            if group_cols:
                group_mask = pd.Series(True, index=plot_frame.index)
                alpha_mask = pd.Series(True, index=alpha_frame.index)
                for col, value in group_key.items():
                    group_mask &= plot_frame[col] == value
                    alpha_mask &= alpha_frame[col] == value
                group_plot_df = plot_frame.loc[group_mask].copy()
                group_alpha_df = alpha_frame.loc[alpha_mask].copy()
            else:
                group_plot_df = plot_frame.copy()
                group_alpha_df = alpha_frame.copy()

            if group_plot_df.empty or group_alpha_df.empty:
                continue

            for lineage_name, lineage_df in group_plot_df.groupby("lineage", sort=False):
                score_base_df = lineage_df.loc[:, ["plm_prob", "mut_prob", "obs_present"]].copy()
                score_base_df["plm_prob"] = score_base_df["plm_prob"].clip(lower=1e-32)
                score_base_df["mut_prob"] = score_base_df["mut_prob"].clip(lower=1e-32)

                for alpha_value in sorted(group_alpha_df["alpha"].dropna().astype(float).unique().tolist()):
                    combined_score = np.log10(score_base_df["plm_prob"]) + (float(alpha_value) * np.log10(score_base_df["mut_prob"]))
                    try:
                        logistic_row = compute_mutation_row_logistic_metrics(
                            combined_score,
                            score_base_df["obs_present"],
                            feature_name="combined_score",
                        )
                    except Exception:
                        logistic_row = {
                            "site_logistic_mutated_corr": np.nan,
                            "site_logistic_mutated_intercept": np.nan,
                            "site_logistic_mutated_slope": np.nan,
                            "site_logistic_tjur_r2": np.nan,
                            "site_logistic_mcfadden_r2": np.nan,
                            "site_logistic_brier_score": np.nan,
                            "site_logistic_auroc": np.nan,
                            "site_logistic_pr_auc": np.nan,
                        }
                    logistic_row["alpha"] = float(alpha_value)
                    logistic_row["lineage"] = lineage_name
                    logistic_row.update(group_key)
                    logistic_rows.append(logistic_row)

        logistic_lineage_df = pd.DataFrame(logistic_rows)
        if logistic_lineage_df.empty:
            return logistic_lineage_df
        return summarize_lineage_metric_table(
            logistic_lineage_df,
            group_cols=[col for col in ["model", "epoch_label", "epoch_value", "alpha"] if col in logistic_lineage_df.columns],
        )

    def _compute_mutation_only_logistic_baseline(plot_frame: pd.DataFrame) -> Dict[str, float]:
        required_cols = {"mut_prob", "obs_present"}
        if plot_frame.empty or not required_cols.issubset(plot_frame.columns):
            return {
                "site_logistic_mutated_corr": np.nan,
                "site_logistic_mutated_intercept": np.nan,
                "site_logistic_mutated_slope": np.nan,
                "site_logistic_tjur_r2": np.nan,
                "site_logistic_mcfadden_r2": np.nan,
                "site_logistic_brier_score": np.nan,
                "site_logistic_auroc": np.nan,
                "site_logistic_pr_auc": np.nan,
            }

        lineage_rows: List[Dict[str, object]] = []
        for lineage_name, lineage_df in plot_frame.groupby("lineage", sort=False):
            lineage_row = compute_mutation_row_logistic_metrics(
                np.log10(np.clip(lineage_df["mut_prob"], 1e-32, None)),
                lineage_df["obs_present"],
                feature_name="log_mut_prob",
            )
            lineage_row["lineage"] = lineage_name
            lineage_rows.append(lineage_row)
        if not lineage_rows:
            return {
                "site_logistic_mutated_corr": np.nan,
                "site_logistic_mutated_intercept": np.nan,
                "site_logistic_mutated_slope": np.nan,
                "site_logistic_tjur_r2": np.nan,
                "site_logistic_mcfadden_r2": np.nan,
                "site_logistic_brier_score": np.nan,
                "site_logistic_auroc": np.nan,
                "site_logistic_pr_auc": np.nan,
            }
        return summarize_lineage_metric_table(pd.DataFrame(lineage_rows), group_cols=[]).iloc[0].to_dict()

    def _build_selected_alpha_metric_frame(
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_cols = [
            col for col in ["alpha", "model", "model_display_label", "epoch_label", "epoch_value", "mut_flat_global_spearman_r", "mut_flat_nonzero_pearson_r"]
            if col in alpha_frame.columns
        ]
        selected_alpha_df = alpha_frame.loc[:, selected_cols].copy()
        if "model_display_label" not in selected_alpha_df.columns:
            selected_alpha_df["model_display_label"] = np.nan
        if all(col in plot_frame.columns for col in ["model", "model_display_label"]):
            model_display_df = plot_frame.loc[:, ["model", "model_display_label"]].drop_duplicates(subset=["model"], keep="first")
            selected_alpha_df = selected_alpha_df.merge(model_display_df, on="model", how="left", suffixes=("", "_from_plot"))
            if "model_display_label_from_plot" in selected_alpha_df.columns:
                selected_alpha_df["model_display_label"] = selected_alpha_df["model_display_label"].where(selected_alpha_df["model_display_label"].notna(), selected_alpha_df["model_display_label_from_plot"])
                selected_alpha_df = selected_alpha_df.drop(columns=["model_display_label_from_plot"])
        if "model" in selected_alpha_df.columns:
            selected_alpha_df["model_display_label"] = [
                _plm_epoch_label(epoch_label, epoch_value, model_name=model_name, model_display_label=model_display_label)
                for model_name, epoch_label, epoch_value, model_display_label in zip(
                    selected_alpha_df["model"],
                    selected_alpha_df["epoch_label"] if "epoch_label" in selected_alpha_df.columns else pd.Series([""] * len(selected_alpha_df)),
                    selected_alpha_df["epoch_value"] if "epoch_value" in selected_alpha_df.columns else pd.Series([np.nan] * len(selected_alpha_df)),
                    selected_alpha_df["model_display_label"],
                )
            ]
        logistic_alpha_df = _compute_site_logistic_alpha_metrics(plot_frame, alpha_frame)
        if logistic_alpha_df.empty:
            for metric_col in [
                "site_logistic_mutated_corr",
                "site_logistic_tjur_r2",
                "site_logistic_mcfadden_r2",
                "site_logistic_brier_score",
                "site_logistic_auroc",
                "site_logistic_pr_auc",
            ]:
                selected_alpha_df[metric_col] = np.nan
            return selected_alpha_df

        merge_cols = [col for col in ["model", "epoch_label", "epoch_value", "alpha"] if col in selected_alpha_df.columns and col in logistic_alpha_df.columns]
        return selected_alpha_df.merge(logistic_alpha_df, on=merge_cols, how="left")

    def _plot_alpha_metric_grid(
        plot_frame: pd.DataFrame,
        count_source_frame: pd.DataFrame,
        metric_cols: List[str],
        title_map: Dict[str, str],
        output_name: str,
        nrows: int,
        ncols: int,
        figsize: Tuple[float, float],
        mutation_only_metrics: Optional[pd.Series],
    ) -> None:
        if plot_frame.empty:
            return

        epoch_groups = (
            plot_frame.groupby(["epoch_value", "epoch_label", "model"], sort=True)
            if all(c in plot_frame.columns for c in ["epoch_value", "epoch_label", "model"])
            else None
        )
        n_epochs = len(epoch_groups) if epoch_groups is not None else 1
        epoch_colours = _epoch_colour_sequence(n_epochs)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.array(axes, dtype=object).reshape(-1)
        for i, metric_col in enumerate(metric_cols):
            ax = axes[i]
            if epoch_groups is not None:
                for colour_idx, ((ep_val, ep_label, _model_name), grp) in enumerate(epoch_groups):
                    grp_sorted = grp.loc[np.isfinite(grp["alpha"]) & np.isfinite(grp[metric_col])].sort_values("alpha")
                    if grp_sorted.empty:
                        continue
                    tick_label = _plm_epoch_label(ep_label, ep_val)
                    ax.plot(
                        grp_sorted["alpha"],
                        grp_sorted[metric_col],
                        marker="o",
                        color=epoch_colours[colour_idx],
                        label=tick_label,
                        linewidth=1.5,
                        markersize=4,
                    )
            else:
                ax_sorted = plot_frame.loc[np.isfinite(plot_frame["alpha"]) & np.isfinite(plot_frame[metric_col])].sort_values("alpha")
                if ax_sorted.empty:
                    ax.set_title(_append_sample_note(title_map.get(metric_col, metric_col), _sample_note_for_metric(metric_col, count_source_frame)))
                    ax.set_xlabel(alpha_xlabel)
                    ax.set_ylabel("Metric value")
                    ax.grid(alpha=0.3)
                    _style_axis_text(ax, is_alpha_x=True)
                    continue
                ax.plot(ax_sorted["alpha"], ax_sorted[metric_col], marker="o", linewidth=1.5, color="#1f6f8b", label="PLM alpha sweep")

            if (
                mutation_only_metrics is not None
                and metric_col in mutation_only_metrics.index
                and np.isfinite(float(mutation_only_metrics[metric_col]))
            ):
                ax.scatter(
                    [_baseline_alpha_x(plot_frame)],
                    [float(mutation_only_metrics[metric_col])],
                    marker="s",
                    s=110,
                    color="#b58900",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label="Mutation accessibility only",
                )

            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(_append_sample_note(title_map.get(metric_col, metric_col), _sample_note_for_metric(metric_col, count_source_frame)))
            ax.set_xlabel(alpha_xlabel)
            ax.set_ylabel("Metric value")
            ax.grid(alpha=0.3)
            _style_axis_text(ax, is_alpha_x=True)

        for ax in axes[len(metric_cols):]:
            ax.axis("off")

        handles, labels = _collect_axes_legend_entries(axes)
        if handles:
            fig.legend(handles, labels, title="PLM sweep / baseline", loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=max(1, min(4, len(handles))), frameon=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
        plt.savefig(output_dir / output_name, dpi=300)
        plt.close()

    def _write_selected_alpha_sweep_plot(
        target_dir: Path,
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
        title_prefix: str,
        output_name: str = "alpha_sweep_metrics_selected.png",
        logistic_output_name: Optional[str] = None,
        note_style: str = "matrix",
        logistic_summary_metric_col: str = "site_logistic_auroc",
    ) -> None:
        if plot_frame.empty or alpha_frame.empty:
            return

        target_dir = ensure_dir(target_dir)
        focused_alpha_df = _build_selected_alpha_metric_frame(plot_frame, alpha_frame)
        mutation_only_metrics = _compute_mutation_only_alpha_baseline(
            plot_frame,
            baseline_output_path=target_dir / "alpha_sweep_fit_metrics_mutation_only.tsv",
        )
        mutation_only_logistic = _compute_mutation_only_logistic_baseline(plot_frame)
        if mutation_only_metrics is not None:
            mutation_only_metrics = mutation_only_metrics.copy()
            for metric_name, metric_value in mutation_only_logistic.items():
                mutation_only_metrics[metric_name] = metric_value

        plot_metric_cols = [
            metric_col for metric_col in [
                "mut_flat_global_spearman_r",
                logistic_summary_metric_col,
                "mut_flat_nonzero_pearson_r",
            ]
            if metric_col in focused_alpha_df.columns
        ]
        if not plot_metric_cols:
            return

        fig, axes = plt.subplots(1, len(plot_metric_cols), figsize=(6 * len(plot_metric_cols), 6.4), sharex=True)
        axes = np.array(axes, dtype=object).reshape(-1)
        selected_axis_label_fontsize = label_fontsize + 2
        title_map = {
            "mut_flat_global_spearman_r": "Spearman(PLM vs all possible AA freq)",
            "mut_flat_nonzero_pearson_r": "Pearson(PLM vs observed non-zero freq)",
            "site_logistic_auroc": "Logistic regression AUROC",
            "site_logistic_pr_auc": "Logistic regression PR-AUC",
        }
        ylabel_map = {
            "mut_flat_global_spearman_r": "Spearman r",
            "mut_flat_nonzero_pearson_r": "Pearson r",
            "site_logistic_auroc": "AUROC",
            "site_logistic_pr_auc": "PR-AUC",
        }

        epoch_groups = (
            focused_alpha_df.groupby(["epoch_value", "epoch_label", "model", "model_display_label"], sort=True, dropna=False)
            if all(c in focused_alpha_df.columns for c in ["epoch_value", "epoch_label", "model", "model_display_label"])
            else None
        )
        selected_tick_fontsize = tick_fontsize + 3
        selected_legend_fontsize = legend_fontsize + 3
        selected_markersize = 6
        n_groups = len(epoch_groups) if epoch_groups is not None else 1
        epoch_colours = _epoch_colour_sequence(n_groups)
        for ax, metric_col in zip(axes, plot_metric_cols):
            metric_y_values: List[float] = []
            if epoch_groups is not None:
                for colour_idx, ((ep_val, ep_label, model_name, model_display_label), grp) in enumerate(epoch_groups):
                    grp_sorted = grp.loc[np.isfinite(grp["alpha"]) & np.isfinite(grp[metric_col])].sort_values("alpha")
                    if grp_sorted.empty:
                        continue
                    tick_label = _plm_epoch_label(ep_label, ep_val, model_name=model_name, model_display_label=model_display_label)
                    ax.plot(
                        grp_sorted["alpha"],
                        grp_sorted[metric_col],
                        marker="o",
                        color=epoch_colours[colour_idx],
                        label=tick_label,
                        linewidth=1.5,
                        markersize=selected_markersize,
                    )
                    metric_y_values.extend(grp_sorted[metric_col].astype(float).tolist())
            else:
                grp_sorted = focused_alpha_df.loc[np.isfinite(focused_alpha_df["alpha"]) & np.isfinite(focused_alpha_df[metric_col])].sort_values("alpha")
                if grp_sorted.empty:
                    ax.set_title(title_map[metric_col])
                    ax.set_xlabel(alpha_xlabel)
                    ax.set_ylabel(ylabel_map.get(metric_col, metric_col))
                    ax.grid(alpha=0.3)
                    _style_axis_text(ax, is_alpha_x=True)
                    continue
                ax.plot(grp_sorted["alpha"], grp_sorted[metric_col], marker="o", linewidth=1.5, color="#1f6f8b", label="PLM alpha sweep", markersize=selected_markersize)
                metric_y_values.extend(grp_sorted[metric_col].astype(float).tolist())

            if (
                mutation_only_metrics is not None
                and metric_col in mutation_only_metrics.index
                and np.isfinite(float(mutation_only_metrics[metric_col]))
            ):
                ax.scatter(
                    [_baseline_alpha_x(focused_alpha_df)],
                    [float(mutation_only_metrics[metric_col])],
                    marker="s",
                    s=110,
                    color="#b58900",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label="Mutation accessibility only",
                )
                metric_y_values.append(float(mutation_only_metrics[metric_col]))

            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(_append_sample_note(title_map[metric_col], _sample_note_for_metric(metric_col, plot_frame, note_style=note_style)))
            ax.set_xlabel(alpha_xlabel)
            ax.set_ylabel(ylabel_map.get(metric_col, metric_col))

            finite_y = np.asarray([value for value in metric_y_values if np.isfinite(value)], dtype=float)
            if metric_col == "site_logistic_auroc":
                ax.set_ylim(0.48, 1.0)
            elif metric_col == "site_logistic_pr_auc":
                if finite_y.size:
                    y_top = float(np.max(finite_y))
                    pad = 0.05 if np.isclose(y_top, 0.0) else max(0.02, y_top * 0.08)
                    ax.set_ylim(0.0, y_top + pad)
                else:
                    ax.set_ylim(0.0, 0.1)
            elif finite_y.size:
                raw_y_min = float(np.min(finite_y))
                y_top = float(np.max(finite_y))
                if np.isclose(raw_y_min, y_top):
                    pad = 0.05 if np.isclose(y_top, 0.0) else max(0.05, abs(y_top) * 0.1)
                else:
                    pad = max(0.02, (y_top - raw_y_min) * 0.08)
                y_bottom = 0.0 if raw_y_min >= 0.0 else raw_y_min - pad
                ax.set_ylim(y_bottom, y_top + pad)
            ax.grid(alpha=0.3)
            _style_axis_text(ax, is_alpha_x=True)
            ax.xaxis.label.set_size(selected_axis_label_fontsize)
            ax.yaxis.label.set_size(selected_axis_label_fontsize)
            ax.tick_params(axis="both", which="major", labelsize=selected_tick_fontsize)

        logistic_metric_cols = [
            metric_col for metric_col in [
                "site_logistic_auroc",
                "site_logistic_pr_auc",
                "site_logistic_tjur_r2",
                "site_logistic_mcfadden_r2",
                "site_logistic_brier_score",
                "site_logistic_mutated_corr",
            ]
            if metric_col in focused_alpha_df.columns
        ]
        if logistic_metric_cols:
            logistic_title_map = {
                "site_logistic_auroc": "AUROC",
                "site_logistic_pr_auc": "PR-AUC",
                "site_logistic_tjur_r2": "Tjur R2",
                "site_logistic_mcfadden_r2": "McFadden R2",
                "site_logistic_brier_score": "Brier score",
                "site_logistic_mutated_corr": "Fitted-probability correlation",
            }
            logistic_ylabel_map = {
                "site_logistic_auroc": "AUROC",
                "site_logistic_pr_auc": "PR-AUC",
                "site_logistic_tjur_r2": "Tjur R2",
                "site_logistic_mcfadden_r2": "McFadden R2",
                "site_logistic_brier_score": "Brier score",
                "site_logistic_mutated_corr": "Fitted-probability correlation",
            }
            logistic_fig, logistic_axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
            logistic_axes = np.array(logistic_axes, dtype=object).reshape(-1)
            for ax, metric_col in zip(logistic_axes, logistic_metric_cols):
                metric_y_values: List[float] = []
                if epoch_groups is not None:
                    for colour_idx, ((ep_val, ep_label, model_name, model_display_label), grp) in enumerate(epoch_groups):
                        grp_sorted = grp.loc[np.isfinite(grp["alpha"]) & np.isfinite(grp[metric_col])].sort_values("alpha")
                        if grp_sorted.empty:
                            continue
                        tick_label = _plm_epoch_label(ep_label, ep_val, model_name=model_name, model_display_label=model_display_label)
                        ax.plot(grp_sorted["alpha"], grp_sorted[metric_col], marker="o", color=epoch_colours[colour_idx], label=tick_label, linewidth=1.5, markersize=selected_markersize)
                        metric_y_values.extend(grp_sorted[metric_col].astype(float).tolist())
                else:
                    grp_sorted = focused_alpha_df.loc[np.isfinite(focused_alpha_df["alpha"]) & np.isfinite(focused_alpha_df[metric_col])].sort_values("alpha")
                    if not grp_sorted.empty:
                        ax.plot(grp_sorted["alpha"], grp_sorted[metric_col], marker="o", linewidth=1.5, color="#1f6f8b", label="PLM alpha sweep", markersize=selected_markersize)
                        metric_y_values.extend(grp_sorted[metric_col].astype(float).tolist())
                if mutation_only_metrics is not None and metric_col in mutation_only_metrics.index and np.isfinite(float(mutation_only_metrics[metric_col])):
                    ax.scatter([_baseline_alpha_x(focused_alpha_df)], [float(mutation_only_metrics[metric_col])], marker="s", s=110, color="#b58900", edgecolors="black", linewidths=1.0, zorder=6, label="Mutation accessibility only")
                    metric_y_values.append(float(mutation_only_metrics[metric_col]))
                ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                if metric_col == "site_logistic_auroc":
                    ax.axhline(0.5, color="#9b2226", linestyle=":", linewidth=1.4, alpha=0.9)
                    ax.text(
                        0.98,
                        0.04,
                        "0.5 = random / no ranking signal",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=max(9, selected_tick_fontsize - 2),
                        color="#9b2226",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                    )
                elif metric_col == "site_logistic_pr_auc":
                    positive_rate = float(mutation_only_metrics.get("logistic_positive_rate", np.nan)) if mutation_only_metrics is not None else np.nan
                    if np.isfinite(positive_rate):
                        ax.axhline(positive_rate, color="#9b2226", linestyle=":", linewidth=1.4, alpha=0.9)
                        ax.text(
                            0.98,
                            0.04,
                            f"baseline = prevalence ({positive_rate:.3f})",
                            transform=ax.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=max(9, selected_tick_fontsize - 2),
                            color="#9b2226",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                        )
                ax.set_title(_append_sample_note(f"Logistic regression:\n{logistic_title_map[metric_col]}", _sample_note_for_metric(metric_col, plot_frame, note_style=note_style)))
                ax.set_xlabel(alpha_xlabel)
                ax.set_ylabel(logistic_ylabel_map[metric_col])
                finite_y = np.asarray([value for value in metric_y_values if np.isfinite(value)], dtype=float)
                if metric_col == "site_logistic_auroc":
                    ax.set_ylim(0.48, 1.0)
                elif metric_col == "site_logistic_pr_auc":
                    if finite_y.size:
                        y_top = float(np.max(finite_y))
                        pad = 0.05 if np.isclose(y_top, 0.0) else max(0.02, y_top * 0.08)
                        ax.set_ylim(0.0, y_top + pad)
                    else:
                        ax.set_ylim(0.0, 0.1)
                elif finite_y.size:
                    raw_y_min = float(np.min(finite_y))
                    y_top = float(np.max(finite_y))
                    if np.isclose(raw_y_min, y_top):
                        pad = 0.05 if np.isclose(y_top, 0.0) else max(0.05, abs(y_top) * 0.1)
                    else:
                        pad = max(0.02, (y_top - raw_y_min) * 0.08)
                    y_bottom = 0.0 if raw_y_min >= 0.0 else raw_y_min - pad
                    ax.set_ylim(y_bottom, y_top + pad)
                ax.grid(alpha=0.3)
                _style_axis_text(ax, is_alpha_x=True)
                ax.xaxis.label.set_size(selected_axis_label_fontsize)
                ax.yaxis.label.set_size(selected_axis_label_fontsize)
                ax.tick_params(axis="both", which="major", labelsize=selected_tick_fontsize)
            for ax in logistic_axes[len(logistic_metric_cols):]:
                ax.axis("off")
            logistic_handles, logistic_labels = _collect_axes_legend_entries(logistic_axes)
            if logistic_handles:
                logistic_fig.legend(logistic_handles, logistic_labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=max(1, min(4, len(logistic_handles))), frameon=False, fontsize=selected_legend_fontsize, markerscale=1.5)
            logistic_fig.suptitle(f"{title_prefix} | Mean across lineages | Logistic alpha-sweep metrics", y=0.985)
            logistic_fig._suptitle.set_fontsize(title_fontsize + 4)
            plt.tight_layout(rect=(0, 0, 1, 0.935))
            plt.savefig(target_dir / (logistic_output_name or "alpha_sweep_logistic_metrics_selected.png"), dpi=300)
            plt.close(logistic_fig)

        handles, labels = _collect_axes_legend_entries(axes)
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=max(1, min(4, len(handles))), frameon=False, fontsize=selected_legend_fontsize, markerscale=1.5)
        fig.suptitle(f"{title_prefix} | Mean across lineages", y=0.985)
        fig._suptitle.set_fontsize(title_fontsize + 4)
        plt.tight_layout(rect=(0, 0, 1, 0.935))
        output_path = target_dir / output_name
        plt.savefig(output_path, dpi=300)
        plt.savefig(output_path.with_suffix(".pdf"))
        plt.close(fig)

    def _write_hurdle_alpha_outputs(
        target_plot_dir: Path,
        target_metrics_dir: Path,
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
        title_prefix: str,
    ) -> None:
        if plot_frame.empty or alpha_frame.empty or "alpha" not in alpha_frame.columns:
            return

        target_plot_dir = ensure_dir(target_plot_dir)
        target_metrics_dir = ensure_dir(target_metrics_dir)

        if "model" in plot_frame.columns and plot_frame["model"].nunique() > 1:
            model_meta = plot_frame.loc[:, ["model", "epoch_label", "epoch_value"]].drop_duplicates().sort_values(["epoch_value", "epoch_label", "model"])
            model_groups = [
                (
                    str(row["model"]),
                    plot_frame.loc[plot_frame["model"] == row["model"]].copy(),
                    alpha_frame.loc[alpha_frame["model"] == row["model"]].copy() if "model" in alpha_frame.columns else alpha_frame.copy(),
                    str(row["epoch_label"]),
                    float(row["epoch_value"]),
                )
                for _, row in model_meta.iterrows()
            ]
        else:
            epoch_label = str(alpha_frame.iloc[0]["epoch_label"]) if "epoch_label" in alpha_frame.columns and not alpha_frame.empty else "single_model"
            epoch_value = float(alpha_frame.iloc[0]["epoch_value"]) if "epoch_value" in alpha_frame.columns and not alpha_frame.empty else 0.0
            model_groups = [(title_prefix, plot_frame.copy(), alpha_frame.copy(), epoch_label, epoch_value)]

        all_hurdle_frames: List[pd.DataFrame] = []
        all_summary_frames: List[pd.DataFrame] = []
        diagnostic_specs: List[Dict[str, object]] = []
        fig, axes = plt.subplots(len(model_groups), 3, figsize=(18, 5.5 * len(model_groups)), squeeze=False)

        for row_idx, (model_name, model_plot_frame, model_alpha_frame, epoch_label, epoch_value) in enumerate(model_groups):
            alpha_values = sorted(model_alpha_frame["alpha"].dropna().astype(float).unique().tolist())
            if not alpha_values:
                continue
            hurdle_alpha_df = evaluate_hurdle_alpha_sweep(model_plot_frame, alpha_values)
            if hurdle_alpha_df.empty:
                continue
            model_display_label = _plm_epoch_label(epoch_label, epoch_value, model_name=model_name)
            hurdle_alpha_df["model"] = model_name
            hurdle_alpha_df["epoch_label"] = epoch_label
            hurdle_alpha_df["epoch_value"] = float(epoch_value)
            hurdle_alpha_df["model_display_label"] = model_display_label
            all_hurdle_frames.append(hurdle_alpha_df)

            hurdle_summary_df = summarize_hurdle_models(model_plot_frame, hurdle_alpha_df)
            hurdle_summary_df["model"] = model_name
            hurdle_summary_df["epoch_label"] = epoch_label
            hurdle_summary_df["epoch_value"] = float(epoch_value)
            hurdle_summary_df["model_display_label"] = model_display_label
            all_summary_frames.append(hurdle_summary_df)

            base_df = build_hurdle_base_frame(model_plot_frame)
            for variant_name, variant_label in [("plm_only_alpha0", "PLM only (alpha = 0)"), ("best_alpha_hurdle", "Best hurdle fit")]:
                variant_rows = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == variant_name]
                if variant_rows.empty or base_df.empty:
                    continue
                diagnostic_specs.append(
                    {
                        "model_display_label": model_display_label,
                        "variant_label": variant_label,
                        "base_df": base_df.copy(),
                        "summary_row": variant_rows.iloc[0].to_dict(),
                    }
                )

            logistic_curve_df = hurdle_alpha_df.loc[:, ["alpha_presence", "logistic_tjur_r2"]].drop_duplicates().sort_values("alpha_presence")
            freq_curve_df = hurdle_alpha_df.loc[:, ["alpha_frequency", "freq_log10_r2", "freq_raw_r2"]].drop_duplicates().sort_values("alpha_frequency")
            baseline_row = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == "mutation_only"].iloc[0]
            two_input_row = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == "two_input_hurdle"].iloc[0]
            best_row = hurdle_alpha_df.loc[hurdle_alpha_df["hurdle_mean_r2"].astype(float).idxmax()] if hurdle_alpha_df["hurdle_mean_r2"].notna().any() else hurdle_alpha_df.iloc[0]

            row_axes = axes[row_idx, :]
            row_axes[0].plot(logistic_curve_df["alpha_presence"], logistic_curve_df["logistic_tjur_r2"], marker="o", linewidth=1.5, color="#1f6f8b", label="PLM-weighted hurdle sweep")
            if np.isfinite(float(baseline_row.get("logistic_tjur_r2", np.nan))):
                row_axes[0].scatter([float(baseline_row["alpha_presence"])], [float(baseline_row["logistic_tjur_r2"])], marker="s", s=110, color="#b58900", edgecolors="black", linewidths=1.0, zorder=5, label="Mutation accessibility only")
            row_axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            row_axes[0].set_title(
                _append_sample_note(
                    f"{model_display_label}\nBinary hurdle term",
                    f"n={_format_count(float(baseline_row.get('logistic_n_obs', np.nan)))} mutation rows",
                )
            )
            row_axes[0].set_xlabel("PLM alpha for zero/non-zero term")
            row_axes[0].set_ylabel("Tjur R2")
            row_axes[0].grid(alpha=0.3)
            _style_axis_text(row_axes[0])

            row_axes[1].plot(
                freq_curve_df["alpha_frequency"],
                freq_curve_df["freq_log10_r2"],
                marker="o",
                linewidth=1.5,
                color="#1f6f8b",
                label="PLM sweep, OLS on log10(non-zero freq)",
            )
            row_axes[1].plot(
                freq_curve_df["alpha_frequency"],
                freq_curve_df["freq_raw_r2"],
                marker="o",
                linewidth=1.2,
                linestyle="--",
                color="#2a9d8f",
                label="PLM sweep, OLS on raw non-zero freq",
            )
            if np.isfinite(float(baseline_row.get("freq_log10_r2", np.nan))):
                row_axes[1].scatter([float(baseline_row["alpha_frequency"])], [float(baseline_row["freq_log10_r2"])], marker="s", s=110, color="#b58900", edgecolors="black", linewidths=1.0, zorder=5, label="Mutation-only, OLS on log10(non-zero freq)")
            if np.isfinite(float(baseline_row.get("freq_raw_r2", np.nan))):
                row_axes[1].scatter([float(baseline_row["alpha_frequency"])], [float(baseline_row["freq_raw_r2"])], marker="D", s=90, color="#e9c46a", edgecolors="black", linewidths=1.0, zorder=5, label="Mutation-only, OLS on raw non-zero freq")
            row_axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            row_axes[1].set_title(
                _append_sample_note(
                    f"{model_display_label}\nPositive-frequency term",
                    f"n={_format_count(float(baseline_row.get('freq_n_obs', np.nan)))} non-zero mutation rows",
                )
            )
            row_axes[1].set_xlabel("PLM alpha for non-zero frequency term")
            row_axes[1].set_ylabel("R2 on non-zero observed frequency\nsolid = log10 scale, dashed = raw scale")
            row_axes[1].grid(alpha=0.3)
            _style_axis_text(row_axes[1])

            heatmap_df = hurdle_alpha_df.pivot(index="alpha_frequency", columns="alpha_presence", values="hurdle_mean_r2").sort_index().sort_index(axis=1)
            sns.heatmap(heatmap_df, ax=row_axes[2], cmap="viridis", cbar=(row_idx == 0), cbar_kws={"label": "Mean hurdle R2"})
            best_alpha_presence = float(best_row["alpha_presence"])
            best_alpha_frequency = float(best_row["alpha_frequency"])
            if best_alpha_presence in heatmap_df.columns and best_alpha_frequency in heatmap_df.index:
                row_axes[2].scatter([heatmap_df.columns.get_loc(best_alpha_presence) + 0.5], [heatmap_df.index.get_loc(best_alpha_frequency) + 0.5], marker="*", s=180, color="black", edgecolors="white", linewidths=0.8, zorder=7)
            row_axes[2].set_title(
                f"{model_display_label}\n"
                f"Combined hurdle alpha sweep\n"
                f"Two-input hurdle mean R2(log10)={float(two_input_row['hurdle_mean_r2']):.3f}; raw={float(two_input_row['hurdle_mean_r2_raw_frequency']):.3f}\n"
                f"binary n={_format_count(float(two_input_row.get('logistic_n_obs', np.nan)))}; freq n={_format_count(float(two_input_row.get('freq_n_obs', np.nan)))}"
            )
            row_axes[2].set_xlabel("PLM alpha for zero/non-zero term")
            row_axes[2].set_ylabel("PLM alpha for non-zero frequency term")
            row_axes[2].set_xticklabels([f"{float(value):.2f}" for value in heatmap_df.columns], rotation=0, fontsize=tick_fontsize)
            row_axes[2].set_yticklabels([f"{float(value):.2f}" for value in heatmap_df.index], rotation=0, fontsize=tick_fontsize)
            row_axes[2].xaxis.label.set_size(label_fontsize)
            row_axes[2].yaxis.label.set_size(label_fontsize)
            row_axes[2].title.set_fontsize(title_fontsize)
            row_axes[2].text(0.02, -0.24, "Star = best alpha pair; mutation-only baseline is shown only in the line panels", transform=row_axes[2].transAxes, fontsize=9, va="top")

        if all_hurdle_frames:
            pd.concat(all_hurdle_frames, ignore_index=True).to_csv(target_metrics_dir / "hurdle_alpha_sweep_metrics.csv", index=False)
        if all_summary_frames:
            pd.concat(all_summary_frames, ignore_index=True).to_csv(target_metrics_dir / "hurdle_model_summary.csv", index=False)

        if diagnostic_specs:
            diag_fig, diag_axes = plt.subplots(len(diagnostic_specs), 2, figsize=(14, 4.8 * len(diagnostic_specs)), squeeze=False)
            for row_idx, spec in enumerate(diagnostic_specs):
                row_axes = diag_axes[row_idx, :]
                base_df = spec["base_df"]
                summary_row = spec["summary_row"]
                presence_alpha = float(summary_row.get("alpha_presence", 0.0)) if np.isfinite(float(summary_row.get("alpha_presence", 0.0))) else 0.0
                frequency_alpha = float(summary_row.get("alpha_frequency", 0.0)) if np.isfinite(float(summary_row.get("alpha_frequency", 0.0))) else 0.0
                presence_score = base_df["log_plm_prob"] + (presence_alpha * base_df["log_mut_prob"])
                frequency_score = base_df["log_plm_prob"] + (frequency_alpha * base_df["log_mut_prob"])

                logistic_intercept = float(summary_row.get("logistic_intercept", np.nan))
                logistic_coef = float(summary_row.get("logistic_coef_combined_score", np.nan))
                row_axes[0].scatter(presence_score, base_df["obs_present"], s=10, alpha=0.10, color="#1f6f8b", label="Observed 0/1 mutation rows")
                if np.isfinite(logistic_intercept) and np.isfinite(logistic_coef):
                    sorted_idx = np.argsort(np.asarray(presence_score, dtype=float))
                    fitted_probs = expit(logistic_intercept + (logistic_coef * np.asarray(presence_score, dtype=float)))
                    row_axes[0].plot(np.asarray(presence_score, dtype=float)[sorted_idx], fitted_probs[sorted_idx], color="#d1495b", linewidth=2.0, label="Fitted logistic curve")
                try:
                    bin_df = pd.DataFrame({"score": presence_score, "obs_present": base_df["obs_present"]})
                    quantiles = min(12, max(4, int(bin_df["score"].nunique())))
                    bin_df["score_bin"] = pd.qcut(bin_df["score"], q=quantiles, duplicates="drop")
                    observed_bins = bin_df.groupby("score_bin", observed=False).agg(score=("score", "mean"), obs_present=("obs_present", "mean")).dropna()
                    if not observed_bins.empty:
                        row_axes[0].scatter(observed_bins["score"], observed_bins["obs_present"], s=32, color="#b58900", edgecolors="black", linewidths=0.5, zorder=5, label="Observed bin mean")
                except ValueError:
                    pass
                row_axes[0].set_title(
                    _append_sample_note(
                        f"{spec['model_display_label']}\n{spec['variant_label']}: binary hurdle regression",
                        f"n={_format_count(float(summary_row.get('logistic_n_obs', np.nan)))} mutation rows",
                    )
                )
                row_axes[0].set_xlabel("Combined hurdle score")
                row_axes[0].set_ylabel("Observed mutation present")
                row_axes[0].grid(alpha=0.3)
                _style_axis_text(row_axes[0])

                positive_mask = base_df["obs_freq"] > 0
                log10_obs_freq = np.log10(np.clip(base_df.loc[positive_mask, "obs_freq"], 1e-32, None))
                row_axes[1].scatter(frequency_score.loc[positive_mask], log10_obs_freq, s=12, alpha=0.14, color="#2a9d8f", label="Observed non-zero frequencies")
                freq_intercept = float(summary_row.get("freq_intercept", np.nan))
                freq_coef = float(summary_row.get("freq_coef_combined_score", np.nan))
                if positive_mask.any() and np.isfinite(freq_intercept) and np.isfinite(freq_coef):
                    x_vals = np.asarray(frequency_score.loc[positive_mask], dtype=float)
                    sorted_idx = np.argsort(x_vals)
                    fitted_log_freq = freq_intercept + (freq_coef * x_vals)
                    row_axes[1].plot(x_vals[sorted_idx], fitted_log_freq[sorted_idx], color="#d1495b", linewidth=2.0, label="Fitted OLS line")
                try:
                    freq_bin_df = pd.DataFrame({"score": frequency_score.loc[positive_mask], "log10_obs_freq": log10_obs_freq})
                    quantiles = min(12, max(4, int(freq_bin_df["score"].nunique())))
                    freq_bin_df["score_bin"] = pd.qcut(freq_bin_df["score"], q=quantiles, duplicates="drop")
                    observed_bins = freq_bin_df.groupby("score_bin", observed=False).agg(score=("score", "mean"), log10_obs_freq=("log10_obs_freq", "mean")).dropna()
                    if not observed_bins.empty:
                        row_axes[1].scatter(observed_bins["score"], observed_bins["log10_obs_freq"], s=32, color="#b58900", edgecolors="black", linewidths=0.5, zorder=5, label="Observed bin mean")
                except ValueError:
                    pass
                row_axes[1].set_title(
                    _append_sample_note(
                        f"{spec['model_display_label']}\n{spec['variant_label']}: positive-frequency regression",
                        f"n={_format_count(float(summary_row.get('freq_n_obs', np.nan)))} non-zero mutation rows",
                    )
                )
                row_axes[1].set_xlabel("Combined hurdle score")
                row_axes[1].set_ylabel("log10(non-zero observed frequency)")
                row_axes[1].grid(alpha=0.3)
                _style_axis_text(row_axes[1])

            diag_handles, diag_labels = _collect_axes_legend_entries(diag_axes.reshape(-1))
            if diag_handles:
                diag_fig.legend(diag_handles, diag_labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=max(1, min(3, len(diag_handles))), frameon=False, fontsize=legend_fontsize)
            diag_fig.suptitle(f"{title_prefix}\nRegression diagnostics")
            diag_fig._suptitle.set_fontsize(title_fontsize + 1)
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            plt.savefig(target_plot_dir / "hurdle_regression_diagnostics.png", dpi=300)
            plt.close(diag_fig)

        handles, labels = _collect_axes_legend_entries(axes[:, :2].reshape(-1))
        if handles:
            fig.legend(handles, labels, title="Hurdle legend", loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=max(1, min(3, len(handles))), frameon=False, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        fig.suptitle(title_prefix)
        fig._suptitle.set_fontsize(title_fontsize + 1)
        plt.tight_layout(rect=(0, 0, 1, 0.965))
        plt.savefig(target_plot_dir / "hurdle_alpha_sweep.png", dpi=300)
        plt.close(fig)

    def _write_logistic_comparison_report(
        target_metrics_dir: Path,
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
    ) -> None:
        if plot_frame.empty:
            return

        target_metrics_dir = ensure_dir(target_metrics_dir)
        if "model" in plot_frame.columns and plot_frame["model"].nunique() > 1:
            model_meta = (
                plot_frame.loc[:, [col for col in ["model", "epoch_label", "epoch_value", "model_display_label"] if col in plot_frame.columns]]
                .drop_duplicates()
                .sort_values([col for col in ["epoch_value", "epoch_label", "model"] if col in plot_frame.columns])
            )
            model_groups = [
                (
                    str(row["model"]),
                    plot_frame.loc[plot_frame["model"] == row["model"]].copy(),
                    alpha_frame.loc[alpha_frame["model"] == row["model"]].copy() if "model" in alpha_frame.columns else alpha_frame.copy(),
                    str(row["epoch_label"]) if "epoch_label" in row else "single_model",
                    float(row["epoch_value"]) if "epoch_value" in row else 0.0,
                    str(row["model_display_label"]) if "model_display_label" in row and pd.notna(row["model_display_label"]) else str(row["model"]),
                )
                for _, row in model_meta.iterrows()
            ]
        else:
            epoch_label = str(alpha_frame.iloc[0]["epoch_label"]) if "epoch_label" in alpha_frame.columns and not alpha_frame.empty else "single_model"
            epoch_value = float(alpha_frame.iloc[0]["epoch_value"]) if "epoch_value" in alpha_frame.columns and not alpha_frame.empty else 0.0
            model_name = str(plot_frame.iloc[0]["model"]) if "model" in plot_frame.columns and not plot_frame.empty else "single_model"
            model_display_label = str(plot_frame.iloc[0]["model_display_label"]) if "model_display_label" in plot_frame.columns and pd.notna(plot_frame.iloc[0]["model_display_label"]) else model_name
            model_groups = [(model_name, plot_frame.copy(), alpha_frame.copy(), epoch_label, epoch_value, model_display_label)]

        report_rows: List[Dict[str, object]] = []
        for model_name, model_plot_frame, model_alpha_frame, epoch_label, epoch_value, model_display_label in model_groups:
            base_df = build_hurdle_base_frame(model_plot_frame)
            standalone_metric_specs = [
                (
                    "standalone_binary_term",
                    "plm_prob",
                    base_df["log_plm_prob"],
                    "mutation_row",
                    "log_plm_prob",
                    "log_plm_prob",
                ),
                (
                    "standalone_binary_term",
                    "mutation_accessibility",
                    base_df["log_mut_prob"],
                    "mutation_row",
                    "log_mut_prob",
                    "log_mut_prob",
                ),
            ]
            for framework, predictor, score_series, observation_unit, score_definition, feature_name in standalone_metric_specs:
                row_metrics = compute_mutation_row_logistic_metrics(
                    score_series,
                    base_df["obs_present"],
                    feature_name=feature_name,
                )
                report_rows.append(
                    {
                        "model": model_name,
                        "model_display_label": model_display_label,
                        "epoch_label": epoch_label,
                        "epoch_value": float(epoch_value),
                        "framework": framework,
                        "predictor": predictor,
                        "model_variant": predictor,
                        "observation_unit": observation_unit,
                        "score_definition": score_definition,
                        "logistic_n_obs": int(len(base_df)),
                        "logistic_positive_rate": float(np.mean(base_df["obs_present"])) if len(base_df) > 0 else np.nan,
                        "logistic_auroc": float(row_metrics.get("site_logistic_auroc", np.nan)),
                        "logistic_tjur_r2": float(row_metrics.get("site_logistic_tjur_r2", np.nan)),
                        "logistic_mcfadden_r2": float(row_metrics.get("site_logistic_mcfadden_r2", np.nan)),
                        "logistic_brier_score": float(row_metrics.get("site_logistic_brier_score", np.nan)),
                        "logistic_fitted_prob_corr": float(row_metrics.get("site_logistic_mutated_corr", np.nan)),
                        "logistic_intercept": float(row_metrics.get("site_logistic_mutated_intercept", np.nan)),
                        "logistic_primary_coef": float(row_metrics.get("site_logistic_mutated_slope", np.nan)),
                    }
                )

            alpha_values = sorted(model_alpha_frame["alpha"].dropna().astype(float).unique().tolist()) if "alpha" in model_alpha_frame.columns else []
            hurdle_alpha_df = evaluate_hurdle_alpha_sweep(model_plot_frame, alpha_values) if alpha_values else pd.DataFrame()
            hurdle_summary_df = summarize_hurdle_models(model_plot_frame, hurdle_alpha_df)
            hurdle_rows = {
                "plm_only_alpha0": "plm_prob",
                "mutation_only": "mutation_accessibility",
            }
            for model_variant, predictor in hurdle_rows.items():
                variant_rows = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == model_variant]
                if variant_rows.empty:
                    continue
                variant_row = variant_rows.iloc[0]
                primary_coef_col = "logistic_coef_combined_score" if model_variant == "plm_only_alpha0" else "logistic_coef_log_mut_prob"
                report_rows.append(
                    {
                        "model": model_name,
                        "model_display_label": model_display_label,
                        "epoch_label": epoch_label,
                        "epoch_value": float(epoch_value),
                        "framework": "hurdle_binary_term",
                        "predictor": predictor,
                        "model_variant": model_variant,
                        "observation_unit": "mutation_row",
                        "score_definition": str(variant_row.get("presence_stage_feature_set", "")),
                        "logistic_n_obs": float(variant_row.get("logistic_n_obs", np.nan)),
                        "logistic_positive_rate": float(variant_row.get("logistic_positive_rate", np.nan)),
                        "logistic_auroc": float(variant_row.get("logistic_auroc", np.nan)),
                        "logistic_tjur_r2": float(variant_row.get("logistic_tjur_r2", np.nan)),
                        "logistic_mcfadden_r2": float(variant_row.get("logistic_mcfadden_r2", np.nan)),
                        "logistic_brier_score": float(variant_row.get("logistic_brier_score", np.nan)),
                        "logistic_fitted_prob_corr": float(variant_row.get("logistic_fitted_prob_corr", np.nan)),
                        "logistic_intercept": float(variant_row.get("logistic_intercept", np.nan)),
                        "logistic_primary_coef": float(variant_row.get(primary_coef_col, np.nan)),
                    }
                )

        if not report_rows:
            return

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(target_metrics_dir / "logistic_regression_comparison_report.tsv", sep="\t", index=False)

        note_lines = [
            "Logistic regression comparison report",
            "",
            "This report compares standalone binary logistic fits and hurdle binary-term logistic fits on the same mutation-row question.",
            "",
            "1. standalone_binary_term rows fit a single-predictor logistic model directly on mutation rows.",
            "   Question: given this specific candidate mutation row, is this mutation observed?",
            "",
            "2. hurdle_binary_term rows use one row per candidate mutation (lineage-position-refAA-AA).",
            "   They answer the same binary question, but inside the hurdle-model export path.",
            "",
            "For the one-predictor PLM-only and mutation-only cases, standalone_binary_term and hurdle_binary_term should now be directly comparable.",
        ]
        (target_metrics_dir / "logistic_regression_comparison_notes.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    def _write_epoch_metric_plot(
        target_dir: Path,
        epoch_summary_df_local: pd.DataFrame,
        metric_col: str,
        baseline_col: Optional[str],
        title: str,
        output_name: str,
    ) -> None:
        if epoch_summary_df_local.empty or metric_col not in epoch_summary_df_local.columns:
            return

        target_dir = ensure_dir(target_dir)
        fig, ax = plt.subplots(figsize=(6, 5))
        epoch_x = epoch_summary_df_local["epoch_value"].to_numpy(dtype=float)
        epoch_y = epoch_summary_df_local[metric_col].to_numpy(dtype=float)
        baseline_y = float(epoch_summary_df_local[baseline_col].mean()) if baseline_col and baseline_col in epoch_summary_df_local else np.nan

        ax.plot(epoch_x, epoch_y, marker="o", linewidth=1.5, color="tab:blue")
        ax.scatter(epoch_x, epoch_y, color="tab:blue", s=35, zorder=3, label="PLM epoch mean")
        if np.isfinite(baseline_y):
            ax.scatter([mutation_baseline_x], [baseline_y], color="tab:red", s=55, zorder=4, label="Mutation baseline")
            ax.axvline(mutation_baseline_x, color="tab:red", linestyle="--", alpha=0.3)
        ax.set_title(_append_sample_note(title, _sample_note_for_epoch_metric(metric_col, epoch_summary_df_local)))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Correlation coefficient")
        ax.grid(alpha=0.25)
        tick_positions = [mutation_baseline_x] + epoch_summary_df_local["epoch_value"].tolist()
        tick_labels = ["mut"] + [
            _format_epoch_tick_label(label, value)
            for label, value in zip(epoch_summary_df_local["epoch_label"].tolist(), epoch_summary_df_local["epoch_value"].tolist())
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)))
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        plt.savefig(target_dir / output_name, dpi=300)
        plt.close(fig)

    def _select_top_level_scatter_frame(plot_frame: pd.DataFrame) -> pd.DataFrame:
        if plot_frame.empty or "model" not in plot_frame.columns:
            return plot_frame

        model_epoch_df = (
            plot_frame.loc[:, ["model", "epoch_label", "epoch_value"]]
            .drop_duplicates()
            .sort_values(["epoch_value", "epoch_label", "model"])
        )
        if model_epoch_df.empty:
            return plot_frame
        if len(model_epoch_df) == 1:
            return plot_frame.loc[plot_frame["model"] == model_epoch_df.iloc[0]["model"]].copy()

        final_rows = model_epoch_df.loc[model_epoch_df["epoch_label"].astype(str) == "final_checkpoint"]
        if not final_rows.empty:
            selected_model = str(final_rows.iloc[-1]["model"])
            return plot_frame.loc[plot_frame["model"] == selected_model].copy()

        non_raw_rows = model_epoch_df.loc[model_epoch_df["epoch_label"].astype(str) != "raw_model"]
        if not non_raw_rows.empty:
            selected_model = str(non_raw_rows.iloc[-1]["model"])
            return plot_frame.loc[plot_frame["model"] == selected_model].copy()

        selected_model = str(model_epoch_df.iloc[-1]["model"])
        return plot_frame.loc[plot_frame["model"] == selected_model].copy()

    def _write_method2_scatter_grid(
        target_dir: Path,
        plot_frame: pd.DataFrame,
        title_suffix: str,
    ) -> None:
        if plot_frame.empty:
            return

        lineage_names = sorted(plot_frame["lineage"].dropna().unique().tolist())
        if not lineage_names or not scatter_alphas:
            return

        target_dir = ensure_dir(target_dir)
        nrows = len(lineage_names)
        ncols = len(scatter_alphas)
        fig_sc, axes_sc = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharey="row")
        axes_sc = np.array(axes_sc, dtype=object)
        if axes_sc.ndim == 0:
            axes_sc = axes_sc.reshape(1, 1)
        elif axes_sc.ndim == 1:
            if nrows == 1:
                axes_sc = axes_sc.reshape(1, -1)
            else:
                axes_sc = axes_sc.reshape(-1, 1)

        for row_idx, lineage_name in enumerate(lineage_names):
            lineage_df = plot_frame.loc[
                plot_frame["lineage"] == lineage_name,
                ["obs_freq", "plm_prob", "mut_prob"],
            ].copy()
            if len(lineage_df) > scatter_max_points:
                lineage_df = lineage_df.sample(scatter_max_points, random_state=0)
            lineage_info = lineage_cache[lineage_name]
            n_seq = int(lineage_info.get("n_sequences", len(lineage_info.get("records", []))))
            n_nonzero_sites = int(
                plot_frame.loc[
                    (plot_frame["lineage"] == lineage_name) & (plot_frame["obs_freq"] > 0),
                    ["position", "ref_aa"],
                ].drop_duplicates().shape[0]
            )
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
                ax.set_title(
                    f"alpha={alpha_value:.2f}\nρ={rho:.3f}, n_seq={n_seq}, nonzero_sites={n_nonzero_sites}"
                )
                ax.grid(alpha=0.25)
                if row_idx == nrows - 1:
                    ax.set_xlabel("log10(PLM × mut^alpha)")
                if col_idx == 0:
                    ax.set_ylabel(f"{lineage_name}\nlog10(observed freq)")

        fig_sc.suptitle(
            f"Observed mutation frequency vs PLM×mutation accessibility\n{title_suffix}\npseudocount={dynamic_pseudocount:.1e}"
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(target_dir / "method2_obsfreq_vs_plm_mut_scatter_grid.png", dpi=300)
        plt.close()

    def _write_plm_vs_mut_outputs(target_dir: Path, plot_frame: pd.DataFrame, title_prefix: str) -> None:
        def _plm_vs_mut_title_text(prefix_text: str, frame: pd.DataFrame) -> str:
            if "all models" in str(prefix_text).lower():
                return "PLM vs mutation probability"

            model_name = str(frame.iloc[0]["model"]) if not frame.empty and "model" in frame.columns and pd.notna(frame.iloc[0]["model"]) else ""
            if model_name == "ESM2_650M_HA80_raw":
                return "PLM vs mutation probability: ESM2_650M"

            clean_name = re.sub(r"(?:_raw| raw)$", "", model_name).strip("_ ") if model_name else ""
            return f"PLM vs mutation probability: {clean_name}" if clean_name else str(prefix_text)

        target_dir = ensure_dir(target_dir)
        id_cols = [
            col for col in [
                "model",
                "model_display_label",
                "base_model",
                "checkpoint_label",
                "epoch_label",
                "epoch_value",
                "lineage",
                "position",
                "ref_aa",
                "aa",
            ]
            if col in plot_frame.columns
        ]
        value_cols = ["plm_prob", "mut_prob"]
        comparison_df = plot_frame[id_cols + value_cols].drop_duplicates().copy()
        comparison_df.to_csv(target_dir / "plm_vs_mut_prob.csv", index=False)

        mean_group_cols = [
            col for col in [
                "model",
                "model_display_label",
                "base_model",
                "checkpoint_label",
                "epoch_label",
                "epoch_value",
                "position",
                "ref_aa",
                "aa",
            ]
            if col in plot_frame.columns
        ]
        if len(mean_group_cols) > 0:
            mean_df = (
                plot_frame.groupby(mean_group_cols, as_index=False)
                .agg(
                    mean_plm_prob=("plm_prob", "mean"),
                    mean_mut_prob=("mut_prob", "mean"),
                    n_rows_averaged=("plm_prob", "size"),
                    n_lineages_averaged=("lineage", "nunique") if "lineage" in plot_frame.columns else ("plm_prob", "size"),
                )
            )
            mean_df.to_csv(target_dir / "plm_vs_mut_prob_mean_by_model.csv", index=False)

        plot_mask = (comparison_df["plm_prob"] > 0) & (comparison_df["mut_prob"] > 0)
        plot_df = comparison_df.loc[plot_mask]
        if plot_df.empty:
            return

        rho, _ = spearmanr(plot_df["plm_prob"], plot_df["mut_prob"])
        pearson_r = safe_pearson(plot_df["plm_prob"], plot_df["mut_prob"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(plot_df["plm_prob"], plot_df["mut_prob"], alpha=0.3, s=10, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _hide_log_minor_ticks(ax)
        ax.set_xlabel("PLM Probability", fontsize=label_fontsize + 2)
        ax.set_ylabel("Mutation Probability", fontsize=label_fontsize + 2)
        ax.set_title(
            f"{_plm_vs_mut_title_text(title_prefix, plot_frame)}\n"
            f"Spearman ρ(plm_prob, mut_prob)={rho:.3f}; "
            f"Pearson r(plm_prob, mut_prob)={pearson_r:.3f}"
        )
        ax.grid(True, which="major", ls="--", alpha=0.4)
        fig.tight_layout()
        export_publication_figure(target_dir / "plm_vs_mut_prob_scatter.png", figure=fig)
        plt.close(fig)

        # Parallel plot by codon mutation distance (1, 2, or 3 nucleotide mutations to that codon)
        try:
            from Functions_HuggingFace import build_codon_aa_mutation_tables
            active_model = mutation_model
            if active_model is None:
                for key, val in lineage_cache.items():
                    if isinstance(val, dict) and "mutation_model" in val:
                        active_model = val["mutation_model"]
                        break
                if active_model is None:
                    out_str = str(output_dir).upper()
                    if "SC2" in out_str:
                        active_model = "SC2"
                    elif "H1N1" in out_str:
                        active_model = "H1N1"
                    elif "H3N2" in out_str:
                        active_model = "H3N2"
            if active_model not in ("SC2", "H1N1", "H3N2"):
                active_model = "SC2"
            tables = build_codon_aa_mutation_tables(active_model)
            aa_to_codons = tables.get("aa_to_codons_all", tables.get("aa_to_codons"))
            from Functions_HuggingFace import compute_codon_distances_for_df
            plot_df_dist = compute_codon_distances_for_df(plot_df, lineage_cache, aa_to_codons)
            # Filter to keep only valid 1, 2, or 3 nucleotide mutations
            plot_df_dist = plot_df_dist.loc[plot_df_dist["nt_mutations"].isin([1, 2, 3])]

            # Print sanity checks and warnings for outliers
            print(f"\n[MUTATION DISTANCE PLOT SANITY CHECK] {title_prefix}:")
            standard_genetic_code = {
                "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
                "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
                "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
                "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
                "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
                "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
                "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
                "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
                "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
                "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
                "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
                "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
                "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
                "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
                "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
                "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
            }
            for lin in sorted(plot_df_dist["lineage"].dropna().unique()):
                lin_df = plot_df_dist.loc[plot_df_dist["lineage"] == lin]
                expected_count = 0
                lin_str = str(lin)
                if lin_str in lineage_cache and lineage_cache[lin_str].get("full_ref_protein"):
                    expected_count = 19 * len(lineage_cache[lin_str]["full_ref_protein"])
                actual_count = len(lin_df)
                print(f"  Lineage {lin_str}: expected {expected_count} datapoints (19 * {len(lineage_cache[lin_str]['full_ref_protein']) if lin_str in lineage_cache and lineage_cache[lin_str].get('full_ref_protein') else 0}) vs {actual_count} actually plotted.")
            
            # Print low-probability codon-amino acid mutation outliers
            cond1 = (plot_df_dist["nt_mutations"] == 1) & (plot_df_dist["mut_prob"] < 1e-6)
            subset1 = plot_df_dist.loc[cond1]
            if not subset1.empty:
                print(f"  WARNING: Found {len(subset1)} entries with mut_num=1 but mut_prob < 1e-6:")
                for _, row in subset1.iterrows():
                    ref_c = row.get("ref_codon", "")
                    ref_c_aa = standard_genetic_code.get(ref_c.upper().replace("U", "T"), "X") if ref_c else "X"
                    print(f"    Lineage {row['lineage']}, Pos {row['position']}: ref_codon={ref_c} ({ref_c_aa}), ref_aa={row['ref_aa']} -> aa={row['aa']} (mut_prob={row['mut_prob']:.2e}, plm_prob={row['plm_prob']:.2e})")

            cond2 = (plot_df_dist["nt_mutations"] == 2) & (plot_df_dist["mut_prob"] < 1e-11)
            subset2 = plot_df_dist.loc[cond2]
            if not subset2.empty:
                print(f"  WARNING: Found {len(subset2)} entries with mut_num=2 but mut_prob < 1e-11:")
                for _, row in subset2.iterrows():
                    ref_c = row.get("ref_codon", "")
                    ref_c_aa = standard_genetic_code.get(ref_c.upper().replace("U", "T"), "X") if ref_c else "X"
                    print(f"    Lineage {row['lineage']}, Pos {row['position']}: ref_codon={ref_c} ({ref_c_aa}), ref_aa={row['ref_aa']} -> aa={row['aa']} (mut_prob={row['mut_prob']:.2e}, plm_prob={row['plm_prob']:.2e})")
            print()

            # Recalculate statistics on the filtered subset for this plot
            valid_df = plot_df_dist.dropna(subset=["plm_prob", "mut_prob"])
            if len(valid_df) > 1:
                sub_rho, _ = spearmanr(valid_df["plm_prob"], valid_df["mut_prob"])
                sub_pearson = safe_pearson(valid_df["plm_prob"], valid_df["mut_prob"])
            else:
                sub_rho, sub_pearson = np.nan, np.nan

            fig_dist, ax_dist = plt.subplots(figsize=(6, 5))
            dark2_colors = sns.color_palette("Dark2", 3)
            color_map = {1: dark2_colors[0], 2: dark2_colors[1], 3: dark2_colors[2]}

            # Plot groups 3, 2, 1 sequentially so 1-nt mutations are on top
            for num_muts in [3, 2, 1]:
                sub_df = plot_df_dist.loc[plot_df_dist["nt_mutations"] == num_muts]
                if not sub_df.empty:
                    ax_dist.scatter(
                        sub_df["plm_prob"],
                        sub_df["mut_prob"],
                        color=color_map[num_muts],
                        alpha=0.3,
                        s=10,
                        edgecolors="none",
                        label=f"{num_muts}",
                    )

            ax_dist.set_xscale("log")
            ax_dist.set_yscale("log")
            _hide_log_minor_ticks(ax_dist)
            ax_dist.set_xlabel("PLM Probability", fontsize=label_fontsize + 2)
            ax_dist.set_ylabel("Mutation Probability", fontsize=label_fontsize + 2)
            ax_dist.set_title(
                f"{_plm_vs_mut_title_text(title_prefix, plot_frame)}\n"
                f"Spearman ρ(x, y)={sub_rho:.3f}; "
                f"Pearson r(x, y)={sub_pearson:.3f}"
            )
            ax_dist.grid(True, which="major", ls="--", alpha=0.4)
            handles, labels = ax_dist.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ordered_labels = [label for label in ["1", "2", "3"] if label in by_label]
            ordered_handles = [by_label[label] for label in ordered_labels]
            leg_dist = ax_dist.legend(
                ordered_handles,
                ordered_labels,
                title="Mut_num",
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=True,
                facecolor="white",
                edgecolor="none",
            )
            # Make the legend points way bigger and opaque
            handles_to_update = getattr(leg_dist, "legend_handles", getattr(leg_dist, "legendHandles", []))
            for lh in handles_to_update:
                lh.set_alpha(1.0)
                lh.set_sizes([50])
            fig_dist.tight_layout()
            export_publication_figure(target_dir / "plm_vs_mut_prob_scatter_by_mutation_distance.png", figure=fig_dist)
            plt.close(fig_dist)
        except Exception as e:
            print(f"Warning: Failed to create parallel plot by mutation distance: {e}")

    def _load_reference_protein_from_path(reference_path: object) -> Optional[str]:
        if reference_path is None or pd.isna(reference_path):
            return None
        path = Path(str(reference_path))
        if not path.exists():
            return None
        records = list(SeqIO.parse(str(path), "fasta"))
        if not records:
            return None
        sequence_text = str(records[0].seq).strip().upper()
        ungapped_text = sequence_text.replace("-", "")
        if not ungapped_text:
            return None
        if set(ungapped_text) <= {"A", "C", "G", "T", "U", "N"}:
            return str(records[0].seq.translate(to_stop=True)).strip().upper()
        return sequence_text

    def _canonical_reference_path_for_mutation_model(model_name: Optional[str]) -> Optional[Path]:
        canonical_refs = {
            "H3N2": REPO_ROOT / "Sequences" / "H3N2_canonical.fa",
        }
        if model_name is None:
            return None
        canonical_path = canonical_refs.get(str(model_name))
        if canonical_path is None or not canonical_path.exists():
            return None
        return canonical_path

    def _lineage_reference_mutations(
        focal_ref_seq: str,
        comparison_ref_seq: str,
    ) -> Tuple[List[Tuple[int, str]], Dict[Tuple[int, str], str]]:
        if not focal_ref_seq or not comparison_ref_seq:
            return [], {}
        try:
            from Functions_HuggingFace import get_mutations

            raw_mutations = [
                mutation for mutation in get_mutations(focal_ref_seq, comparison_ref_seq)
                if "del" not in mutation and "-" not in mutation
            ]
            direct_mutations: List[Tuple[int, str]] = []
            direct_lookup: Dict[Tuple[int, str], str] = {}
            for mutation in raw_mutations:
                match = re.fullmatch(r"([A-Z*-])(\d+)([A-Z*-])", str(mutation))
                if match is None:
                    continue
                mutation_key = (int(match.group(2)) - 1, match.group(3))
                direct_mutations.append(mutation_key)
                direct_lookup[mutation_key] = str(mutation)
            if direct_mutations:
                return direct_mutations, direct_lookup
        except Exception:
            pass
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -0.5
        alignment = next(iter(aligner.align(focal_ref_seq, comparison_ref_seq)), None)
        if alignment is None:
            return [], {}
        aligned_focal, aligned_comparison = str(alignment[0]), str(alignment[1])
        raw_mutations = []
        try:
            from Functions_HuggingFace import get_mutations

            raw_mutations = [
                mutation for mutation in get_mutations(aligned_focal, aligned_comparison)
                if "del" not in mutation and "-" not in mutation
            ]
        except Exception:
            raw_mutations = []
        mutations: List[Tuple[int, str]] = []
        raw_mutation_lookup: Dict[Tuple[int, str], str] = {}
        focal_pos = -1
        mutation_idx = 0
        for focal_char, comparison_char in zip(aligned_focal, aligned_comparison):
            if focal_char != "-":
                focal_pos += 1
            if focal_char == "-" or comparison_char == "-":
                continue
            if focal_char != comparison_char:
                mutation_key = (focal_pos, comparison_char)
                mutations.append(mutation_key)
                if mutation_idx < len(raw_mutations):
                    raw_mutation_lookup[mutation_key] = raw_mutations[mutation_idx]
                    mutation_idx += 1
                else:
                    raw_mutation_lookup[mutation_key] = f"{focal_char}{focal_pos + 1}{comparison_char}"
        return mutations, raw_mutation_lookup

    def _write_lineage_comparison_ranked_plot(
        target_dir: Path,
        model_df: pd.DataFrame,
        model_label: object,
        focal_lineage: str = "J.2_int",
        comparison_lineage: str = "K",
        comparison_title_label: str = "K lineage (J.2.4.1)",
    ) -> None:
        required_cols = {"lineage", "position", "ref_aa", "aa", "plm_prob", "mut_prob"}
        if model_df.empty or not required_cols.issubset(model_df.columns):
            return
        if focal_lineage not in lineage_cache or comparison_lineage not in lineage_cache:
            return

        focal_df = model_df.loc[model_df["lineage"].astype(str) == focal_lineage].copy()
        if focal_df.empty:
            return

        focal_positions_df = (
            focal_df.loc[:, ["position", "ref_aa"]]
            .drop_duplicates()
            .sort_values("position")
        )
        if focal_positions_df.empty:
            return
        focal_positions = focal_positions_df["position"].tolist()
        focal_ref_seq = "".join(focal_positions_df["ref_aa"].astype(str).tolist())
        comparison_ref_seq = _load_reference_protein_from_path(lineage_cache[comparison_lineage].get("reference_path"))
        if not comparison_ref_seq:
            return

        highlighted_mutations, highlighted_mutation_labels = _lineage_reference_mutations(focal_ref_seq, comparison_ref_seq)
        if not highlighted_mutations:
            return

        plm_matrix = (
            focal_df.pivot_table(index="aa", columns="position", values="plm_prob", aggfunc="first")
            .reindex(columns=focal_positions)
            .sort_index()
        )
        mut_matrix = (
            focal_df.pivot_table(index="aa", columns="position", values="mut_prob", aggfunc="first")
            .reindex(columns=focal_positions)
            .sort_index()
        )
        if plm_matrix.empty or mut_matrix.empty:
            return

        model_display_label = _plm_epoch_label(
            focal_df.iloc[0]["epoch_label"] if "epoch_label" in focal_df.columns else str(model_label),
            focal_df.iloc[0]["epoch_value"] if "epoch_value" in focal_df.columns else np.nan,
            model_name=model_label,
            model_display_label=focal_df.iloc[0]["model_display_label"] if "model_display_label" in focal_df.columns else None,
        )
        canonical_mutation_lookup: Dict[str, str] = {}
        canonical_h3_map = None
        canonical_converter = None

        canonical_reference_path = _canonical_reference_path_for_mutation_model(mutation_model)
        if canonical_reference_path is not None:
            try:
                from Functions_HuggingFace import create_h3_numbering_map, mutations_to_canonical

                canonical_ref_seq = _load_reference_protein_from_path(canonical_reference_path)
                if canonical_ref_seq:
                    canonical_h3_map = create_h3_numbering_map(focal_ref_seq, canonical_ref_seq, HA2_start=330)
                    canonical_converter = mutations_to_canonical
            except Exception:
                canonical_mutation_lookup = {}
                canonical_h3_map = None
                canonical_converter = None

        def _add_mutation_labels(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or "Mutation" in frame.columns or not {"Position", "AA"}.issubset(frame.columns):
                return frame
            labelled = frame.copy()
            positions = pd.to_numeric(labelled["Position"], errors="coerce")
            if positions.isna().any():
                return labelled
            ref_indices = positions.astype(int)
            display_positions = ref_indices + 1
            fallback_labels = []
            for ref_index, display_pos, aa in zip(ref_indices, display_positions, labelled["AA"]):
                if 0 <= int(ref_index) < len(focal_ref_seq):
                    fallback_labels.append(f"{focal_ref_seq[int(ref_index)]}{int(display_pos)}{aa}")
                else:
                    fallback_labels.append(f"?{int(display_pos)}{aa}")
            labelled["Mutation"] = fallback_labels
            return labelled

        def _canonicalize_mutation_labels(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty or "Mutation" not in frame.columns:
                return frame
            labelled = frame.copy()
            raw_display_mutations = [
                highlighted_mutation_labels.get((int(pos), str(aa)), str(mutation))
                for pos, aa, mutation in zip(labelled["Position"], labelled["AA"], labelled["Mutation"])
            ]
            if canonical_converter is not None and canonical_h3_map is not None:
                unresolved = [mutation for mutation in raw_display_mutations if mutation not in canonical_mutation_lookup]
                if unresolved:
                    canonical_mutation_lookup.update(dict(zip(unresolved, canonical_converter(unresolved, canonical_h3_map))))
            labelled["DisplayMutation"] = [canonical_mutation_lookup.get(str(mutation), str(mutation)) for mutation in raw_display_mutations]
            return labelled

        # Define explicit font sizes configuration to pass into rc_context if export_publication_figure overrides parameters
        custom_rc_params = {
            "axes.labelsize": 19,
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }

        with plt.rc_context(rc=custom_rc_params):
            fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.5), sharey=True)
            axis_label_fontsize = 20
            tick_label_fontsize = 17
            panel_label_fontsize = 14
            panel_specs = [
                ("PLM probabilities", plm_matrix),
                ("Mutation probabilities", mut_matrix),
            ]
            for ax, (panel_title, matrix) in zip(axes, panel_specs):
                ranked_df, obs_df = get_ranked_mutations(matrix, focal_ref_seq, highlighted_mutations)
                
                if ranked_df.empty:
                    _style_axis_text(ax)
                    ax.set_title(panel_title, fontsize=27, pad=21)
                    ax.set_xlabel("Rank (1 = highest probability)", fontsize=axis_label_fontsize, labelpad=14)
                    ax.set_ylabel("log10(probability)", fontsize=axis_label_fontsize, labelpad=16)
                    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
                    
                    
                    
                    ax.grid(True, which="major", ls="-", alpha=0.2)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
                    continue

                ranked_df = _add_mutation_labels(ranked_df)
                ranked_df = _canonicalize_mutation_labels(ranked_df)
                ranked_df["log10Probability"] = np.log10(np.clip(ranked_df["Probability"], 1e-32, None))
                ax.plot(ranked_df["Rank"], ranked_df["log10Probability"], color="lightgray", linewidth=1.15)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                if not obs_df.empty:
                    obs_df = _add_mutation_labels(obs_df)
                    obs_df = _canonicalize_mutation_labels(obs_df)
                    obs_df["log10Probability"] = np.log10(np.clip(obs_df["Probability"], 1e-32, None))
                    ax.scatter(
                        obs_df["Rank"],
                        obs_df["log10Probability"],
                        color="#d62728",
                        s=30,
                        zorder=5,
                        label=f"{focal_lineage} -> {comparison_title_label}",
                    )
                    ax.set_ylim(top=1.6,bottom=-16)
                    text_artists = []
                    for label_idx, (_, row) in enumerate(obs_df.iterrows()):
                        label_text = f"{row.get('DisplayMutation', row['Mutation'])} (N={int(row['Rank'])})"
                        x_direction = 1.0 if label_idx % 2 == 0 else -1.0
                        x_offset = 0.6 * x_direction
                        y_offset = 0.06 * ((label_idx % 3) + 1)
                        text_artists.append(
                            ax.text(
                                float(row["Rank"]) + x_offset,
                                float(row["log10Probability"]) + y_offset,
                                label_text,
                                fontsize=panel_label_fontsize,
                                color="#8b0000",
                                ha="left" if x_direction > 0 else "right",
                                va="bottom",
                                zorder=6,
                            )
                        )
                    if text_artists:
                      if text_artists:
                        if adjust_text is not None:
                            adjust_text(
                                text_artists,
                                x=obs_df["Rank"].astype(float).to_numpy(),
                                y=obs_df["log10Probability"].astype(float).to_numpy(),
                                ax=ax,
                                only_move={"static": "y", "text": "xy"},
                                expand=(1.35, 1.65),
                                force_static=(0.35, 0.5),
                                force_text=(0.32, 0.5),
                                arrowprops={"arrowstyle": "-", "color": "#8b0000", "lw": 0.5, "alpha": 0.65},
                            )
                        else:
                            for idx, text_artist in enumerate(text_artists):
                                text_artist.set_position(
                                    (
                                        float(obs_df.iloc[idx]["Rank"]) + 0.55,
                                        float(obs_df.iloc[idx]["log10Probability"]) + (0.07 * ((idx % 4) + 1)),
                                    )
                                )

                # 1. Execute styling functions first
                _style_axis_text(ax)
                ax.grid(True, which="major", ls="-", alpha=0.2)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):d}" if float(value).is_integer() else ""))

                # 2. Force modifications over structural style configurations at the absolute end of the axis setup
                ax.set_title(panel_title, fontsize=27, pad=21)
                ax.set_xlabel("Rank (1 = highest probability)", fontsize=axis_label_fontsize, labelpad=14)
                ax.set_ylabel("log10(probability)", fontsize=axis_label_fontsize, labelpad=16)
                ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize)
                

            fig.subplots_adjust(left=0.08, right=0.995, bottom=0.13, top=0.92, wspace=0.18)
            fig.tight_layout(pad=0.8)
            export_publication_figure(target_dir / "ranked_mutations_j2_int_vs_k_lineage.png", figure=fig)
            plt.close(fig)

    if not combined_df.empty:
        _write_plm_vs_mut_outputs(output_dir, combined_df, "PLM vs mutation probability (all models)")

        all_models_dir = ensure_dir(output_dir / "all_models")
        _write_plm_vs_mut_outputs(all_models_dir, combined_df, "PLM vs mutation probability (all models)")

        per_model_dir = ensure_dir(output_dir / "per_model")
        for model_label, model_df in combined_df.groupby("model", sort=True):
            model_plot_dir = ensure_dir(per_model_dir / _safe_output_label(model_label))
            _write_plm_vs_mut_outputs(
                model_plot_dir,
                model_df,
                f"PLM vs mutation probability ({model_label})",
            )
            _write_lineage_comparison_ranked_plot(model_plot_dir, model_df, model_label)

    if not alpha_df.empty:
        mutation_only_metrics = _compute_mutation_only_alpha_baseline(
            combined_df,
            baseline_output_path=output_dir / "alpha_sweep_fit_metrics_mutation_only.tsv",
        )
        _plot_alpha_metric_grid(
            alpha_df,
            combined_df,
            metric_cols=[
                "site_top10pct_mutated_enrichment",
                "site_top10pct_mutated_precision",
                "site_rank_spearman_r",
                "mut_flat_global_spearman_r",
                "mut_flat_global_pearson_r",
                "mut_flat_mean_site_nll",
            ],
            title_map={
                "site_top10pct_mutated_enrichment": "Method A: enrichment of mutated sites in top 10%",
                "site_top10pct_mutated_precision": "Method A: fraction of top 10% sites mutated",
                "site_rank_spearman_r": "Method A: Spearman(site score vs burden)",
                "mut_flat_global_spearman_r": "Method B: Spearman(score vs freq)",
                "mut_flat_global_pearson_r": "Method B: Pearson(score vs freq)",
                "mut_flat_mean_site_nll": "Method B: mean site-level NLL",
            },
            output_name="alpha_sweep_metrics.png",
            nrows=2,
            ncols=3,
            figsize=(18, 9),
            mutation_only_metrics=mutation_only_metrics,
        )
        _plot_alpha_metric_grid(
            alpha_df,
            combined_df,
            metric_cols=[
                "mut_flat_nonzero_spearman_r",
                "mut_flat_nonzero_pearson_r",
                "mut_flat_logfreq_global_pearson_r",
                "mut_flat_logfreq_nonzero_pearson_r",
            ],
            title_map={
                "mut_flat_nonzero_spearman_r": "Method B: Spearman(score vs freq), non-zero obs only",
                "mut_flat_nonzero_pearson_r": "Method B: Pearson(score vs freq), non-zero obs only",
                "mut_flat_logfreq_global_pearson_r": "Method B: Pearson(score vs log(freq + pc)), zeroes included",
                "mut_flat_logfreq_nonzero_pearson_r": "Method B: Pearson(score vs log(freq)), non-zero obs only",
            },
            output_name="alpha_sweep_metrics_nonzero_and_logfreq.png",
            nrows=2,
            ncols=2,
            figsize=(14, 10),
            mutation_only_metrics=mutation_only_metrics,
        )

        per_model_dir = ensure_dir(output_dir / "per_model")
        per_model_metrics_dir = ensure_dir(metrics_output_dir / "per_model")
        for model_label, model_combined_df in combined_df.groupby("model", sort=True):
            model_plot_dir = ensure_dir(per_model_dir / _safe_output_label(model_label))
            model_metrics_dir = ensure_dir(per_model_metrics_dir / _safe_output_label(model_label))
            model_alpha_df = alpha_df.loc[alpha_df["model"] == model_label].copy() if "model" in alpha_df.columns else alpha_df.copy()
            _write_selected_alpha_sweep_plot(
                model_plot_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Focused alpha-sweep metrics ({model_label})",
            )
            _write_selected_alpha_sweep_plot(
                model_plot_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Focused alpha-sweep metrics ({model_label})",
                output_name="alpha_sweep_metrics_selected_pr_auc.png",
                logistic_summary_metric_col="site_logistic_pr_auc",
            )
            _write_selected_alpha_sweep_plot(
                model_plot_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Focused alpha-sweep metrics ({model_label})",
                output_name="alpha_sweep_metrics_selected_mutation_counts.png",
                logistic_output_name="alpha_sweep_logistic_metrics_selected_mutation_counts.png",
                note_style="counts",
            )
            _write_selected_alpha_sweep_plot(
                model_plot_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Focused alpha-sweep metrics ({model_label})",
                output_name="alpha_sweep_metrics_selected_mutation_counts_pr_auc.png",
                logistic_output_name="alpha_sweep_logistic_metrics_selected_mutation_counts.png",
                note_style="counts",
                logistic_summary_metric_col="site_logistic_pr_auc",
            )
            _write_hurdle_alpha_outputs(
                model_plot_dir,
                model_metrics_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Hurdle-model alpha sweep ({model_label})",
            )
            _write_logistic_comparison_report(
                model_metrics_dir,
                model_combined_df,
                model_alpha_df,
            )

        if all(col in alpha_df.columns for col in ["model", "epoch_label", "epoch_value"]):
            alpha_model_meta = alpha_df.loc[:, ["model", "epoch_label", "epoch_value"]].drop_duplicates().copy()
            alpha_model_meta["family_key"] = [
                _model_family_key(model_label, epoch_label)
                for model_label, epoch_label in zip(alpha_model_meta["model"], alpha_model_meta["epoch_label"])
            ]
            for _family_key, family_meta_df in alpha_model_meta.groupby("family_key", sort=False):
                raw_rows = family_meta_df.loc[family_meta_df["epoch_label"].astype(str) == "raw_model"]
                non_raw_rows = family_meta_df.loc[family_meta_df["epoch_label"].astype(str) != "raw_model"]
                if raw_rows.empty or non_raw_rows.empty:
                    continue

                latest_row = non_raw_rows.sort_values(["epoch_value", "epoch_label", "model"]).iloc[-1]
                raw_model_label = str(raw_rows.iloc[0]["model"])
                latest_model_label = str(latest_row["model"])
                comparison_models = {raw_model_label, latest_model_label}
                comparison_alpha_df = alpha_df.loc[alpha_df["model"].isin(comparison_models)].copy()
                comparison_combined_df = combined_df.loc[combined_df["model"].isin(comparison_models)].copy()
                if comparison_alpha_df.empty or comparison_combined_df.empty:
                    continue

                latest_plot_dir = ensure_dir(per_model_dir / _safe_output_label(latest_model_label))
                _write_selected_alpha_sweep_plot(
                    latest_plot_dir,
                    comparison_combined_df,
                    comparison_alpha_df,
                    title_prefix=f"Focused alpha-sweep metrics ({raw_model_label} vs {latest_model_label})",
                    output_name="alpha_sweep_metrics_selected_with_raw.png",
                )
                _write_selected_alpha_sweep_plot(
                    latest_plot_dir,
                    comparison_combined_df,
                    comparison_alpha_df,
                    title_prefix=f"Focused alpha-sweep metrics ({raw_model_label} vs {latest_model_label})",
                    output_name="alpha_sweep_metrics_selected_with_raw_pr_auc.png",
                    logistic_summary_metric_col="site_logistic_pr_auc",
                )
                _write_selected_alpha_sweep_plot(
                    latest_plot_dir,
                    comparison_combined_df,
                    comparison_alpha_df,
                    title_prefix=f"Focused alpha-sweep metrics ({raw_model_label} vs {latest_model_label})",
                    output_name="alpha_sweep_metrics_selected_with_raw_mutation_counts.png",
                    logistic_output_name="alpha_sweep_logistic_metrics_selected_with_raw_mutation_counts.png",
                    note_style="counts",
                )
                _write_selected_alpha_sweep_plot(
                    latest_plot_dir,
                    comparison_combined_df,
                    comparison_alpha_df,
                    title_prefix=f"Focused alpha-sweep metrics ({raw_model_label} vs {latest_model_label})",
                    output_name="alpha_sweep_metrics_selected_with_raw_mutation_counts_pr_auc.png",
                    logistic_output_name="alpha_sweep_logistic_metrics_selected_with_raw_mutation_counts.png",
                    note_style="counts",
                    logistic_summary_metric_col="site_logistic_pr_auc",
                )

        _write_selected_alpha_sweep_plot(
            output_dir,
            combined_df,
            alpha_df,
            title_prefix="Focused alpha-sweep metrics (all models)",
        )
        _write_selected_alpha_sweep_plot(
            output_dir,
            combined_df,
            alpha_df,
            title_prefix="Focused alpha-sweep metrics (all models)",
            output_name="alpha_sweep_metrics_selected_pr_auc.png",
            logistic_summary_metric_col="site_logistic_pr_auc",
        )
        _write_selected_alpha_sweep_plot(
            output_dir,
            combined_df,
            alpha_df,
            title_prefix="Focused alpha-sweep metrics (all models)",
            output_name="alpha_sweep_metrics_selected_mutation_counts.png",
            logistic_output_name="alpha_sweep_logistic_metrics_selected_mutation_counts.png",
            note_style="counts",
        )
        _write_selected_alpha_sweep_plot(
            output_dir,
            combined_df,
            alpha_df,
            title_prefix="Focused alpha-sweep metrics (all models)",
            output_name="alpha_sweep_metrics_selected_mutation_counts_pr_auc.png",
            logistic_output_name="alpha_sweep_logistic_metrics_selected_mutation_counts.png",
            note_style="counts",
            logistic_summary_metric_col="site_logistic_pr_auc",
        )
        _write_hurdle_alpha_outputs(
            output_dir,
            metrics_output_dir,
            combined_df,
            alpha_df,
            title_prefix="Hurdle-model alpha sweep (all models)",
        )
        _write_logistic_comparison_report(
            metrics_output_dir,
            combined_df,
            alpha_df,
        )

    if not combined_df.empty and scatter_alphas:
        per_model_dir = ensure_dir(output_dir / "per_model")
        for model_label, model_df in combined_df.groupby("model", sort=True):
            model_plot_dir = ensure_dir(per_model_dir / _safe_output_label(model_label))
            _write_method2_scatter_grid(
                model_plot_dir,
                model_df,
                title_suffix=str(model_label),
            )

        top_level_scatter_df = _select_top_level_scatter_frame(combined_df)
        if not top_level_scatter_df.empty:
            top_level_label = str(top_level_scatter_df["model"].iloc[0]) if "model" in top_level_scatter_df.columns else "selected model"
            _write_method2_scatter_grid(
                output_dir,
                top_level_scatter_df,
                title_suffix=top_level_label,
            )

    if not epoch_summary_df.empty:
        pooled_plm_mut_metrics = (
            combined_df.groupby(["model", "epoch_label", "epoch_value"], as_index=False)
            .apply(
                lambda grp: pd.Series(
                    {
                        "pooled_spearman_plm_vs_mut": safe_spearman(grp["plm_prob"], grp["mut_prob"]),
                        "pooled_pearson_plm_vs_mut": safe_pearson(grp["plm_prob"], grp["mut_prob"]),
                    }
                )
            )
            .reset_index(drop=True)
            .sort_values(["epoch_value", "epoch_label"])
        )
        pooled_plm_mut_metrics.to_csv(output_dir / "pooled_plm_vs_mut_metrics.tsv", sep="\t", index=False)

        metric_specs = [
            (
                "logistic_site_mutated_vs_plm_corr",
                "logistic_site_mutated_vs_mut_corr_baseline",
                "Logistic regression: site mutated (binary)\nvs PLM probability (pseudo-likelihood)",
            ),
            (
                "spearman_obs_freq_mutated_vs_plm",
                "spearman_obs_freq_mutated_vs_mut_baseline",
                "Spearman r: observed mutation frequency\nvs PLM probability\n(mutated sites only, zeroes excluded)",
            ),
            (
                "pearson_obs_freq_mutated_vs_plm",
                "pearson_obs_freq_mutated_vs_mut_baseline",
                "Pearson r: observed mutation frequency\nvs PLM probability\n(mutated sites only, zeroes excluded)",
            ),
            (
                "spearman_plm_vs_mut",
                None,
                "Mean per-lineage Spearman ρ(plm_prob, mut_prob)\ncorrelation is symmetric in variable order",
            ),
            (
                "pearson_plm_vs_mut",
                None,
                "Mean per-lineage Pearson r(plm_prob, mut_prob)\ncorrelation is symmetric in variable order",
            ),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        epoch_summary_df = epoch_summary_df.sort_values(["epoch_value", "epoch_label"])
        for ax, (metric_col, baseline_col, title) in zip(axes, metric_specs):
            epoch_x = epoch_summary_df["epoch_value"].to_numpy(dtype=float)
            epoch_y = epoch_summary_df[metric_col].to_numpy(dtype=float)
            baseline_y = float(epoch_summary_df[baseline_col].mean()) if baseline_col and baseline_col in epoch_summary_df else np.nan

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
                tick_labels = ["mut"] + [
                    _format_epoch_tick_label(label, value)
                    for label, value in zip(epoch_summary_df["epoch_label"].tolist(), epoch_summary_df["epoch_value"].tolist())
                ]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)))
        plt.tight_layout(rect=(0, 0, 1, 0.92))
        plt.savefig(output_dir / "epoch_metric_summary.png", dpi=300)
        plt.close()

        _write_epoch_metric_plot(
            output_dir,
            epoch_summary_df,
            metric_col="logistic_site_mutated_vs_plm_corr",
            baseline_col="logistic_site_mutated_vs_mut_corr_baseline",
            title="Logistic regression: site mutated (binary) vs PLM probability",
            output_name="epoch_metric_logistic.png",
        )
        _write_epoch_metric_plot(
            output_dir,
            epoch_summary_df,
            metric_col="spearman_obs_freq_mutated_vs_plm",
            baseline_col="spearman_obs_freq_mutated_vs_mut_baseline",
            title="Spearman r: observed mutation frequency vs PLM probability\n(mutated sites only, zeroes excluded)",
            output_name="epoch_metric_spearman_obs_freq_mutated.png",
        )
        _write_epoch_metric_plot(
            output_dir,
            epoch_summary_df,
            metric_col="pearson_obs_freq_mutated_vs_plm",
            baseline_col="pearson_obs_freq_mutated_vs_mut_baseline",
            title="Pearson r: observed mutation frequency vs PLM probability\n(mutated sites only, zeroes excluded)",
            output_name="epoch_metric_pearson_obs_freq_mutated.png",
        )
        _write_epoch_metric_plot(
            output_dir,
            epoch_summary_df,
            metric_col="spearman_plm_vs_mut",
            baseline_col=None,
            title="Mean per-lineage Spearman ρ(plm_prob, mut_prob)",
            output_name="epoch_metric_spearman_plm_vs_mut.png",
        )
        _write_epoch_metric_plot(
            output_dir,
            epoch_summary_df,
            metric_col="pearson_plm_vs_mut",
            baseline_col=None,
            title="Mean per-lineage Pearson r(plm_prob, mut_prob)",
            output_name="epoch_metric_pearson_plm_vs_mut.png",
        )


def _regenerate_figures_from_existing_tables(
    args: argparse.Namespace,
    *,
    tables_dir: Path,
    plots_dir: Path,
    existing_panel_metadata_df: pd.DataFrame,
) -> int:
    combined_path = tables_dir / "combined_long_table.csv"
    if not combined_path.exists():
        raise RuntimeError(f"Cannot regenerate figures: missing combined table at {combined_path}")

    combined_df = pd.read_csv(combined_path)
    alpha_path = tables_dir / "alpha_sweep_fit_metrics.tsv"
    alpha_df = pd.read_csv(alpha_path, sep="\t") if alpha_path.exists() else pd.DataFrame()
    epoch_summary_path = tables_dir / "epoch_metric_summary.tsv"
    if epoch_summary_path.exists():
        epoch_summary_df = pd.read_csv(epoch_summary_path, sep="\t")
    else:
        epoch_summary_df = summarize_epoch_metrics(compute_epoch_lineage_metrics(combined_df))

    lineage_cache = _build_lightweight_lineage_cache_from_metadata(existing_panel_metadata_df)
    if not lineage_cache and "lineage" in combined_df.columns:
        lineage_cache = {
            str(lineage_name): {"n_sequences": 0, "diversity_path": "", "reference_path": ""}
            for lineage_name in combined_df["lineage"].dropna().astype(str).unique().tolist()
        }

    max_depth = pd.to_numeric(combined_df.get("depth"), errors="coerce").max() if "depth" in combined_df.columns else np.nan
    if np.isfinite(max_depth) and max_depth > 0:
        dynamic_pseudocount = float(10 ** -round(np.log10(10 * max(1, max_depth))))
    else:
        dynamic_pseudocount = 1e-16

    mutation_model_val = args.mutation_model
    if mutation_model_val is None and not existing_panel_metadata_df.empty and "mutation_model" in existing_panel_metadata_df.columns:
        mutation_model_val = existing_panel_metadata_df["mutation_model"].dropna().iloc[0]

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
        metrics_output_dir=tables_dir,
        mutation_model=mutation_model_val,
    )
    return 0


def run_analysis(args: argparse.Namespace) -> int:
    from Functions_HuggingFace import build_codon_aa_mutation_tables, evaluate_alpha_sweep

    args = apply_arg_defaults(args)
    output_dir = ensure_dir(args.output_dir)
    group_dir = ensure_dir(output_dir / "groups")
    plm_dir = ensure_dir(output_dir / "plm_cache")
    tables_dir = ensure_dir(output_dir / "tables")
    plots_dir = ensure_dir(output_dir / "plots")
    model_tables_dir = ensure_dir(tables_dir / "per_model")

    existing_panel_metadata_path = tables_dir / "panel_metadata.tsv"
    existing_panel_metadata_df = pd.read_csv(existing_panel_metadata_path, sep="\t") if existing_panel_metadata_path.exists() else pd.DataFrame()
    if getattr(args, "regen_figures_only", False):
        return _regenerate_figures_from_existing_tables(
            args,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            existing_panel_metadata_df=existing_panel_metadata_df,
        )

    model_specs = build_model_specs(args)
    cache_version_matches = (
        not existing_panel_metadata_df.empty
        and (
            "cache_version" not in existing_panel_metadata_df.columns
            or pd.to_numeric(existing_panel_metadata_df["cache_version"], errors="coerce").eq(PANEL_CACHE_VERSION).all()
        )
    )
    mutation_model_matches = (
        not existing_panel_metadata_df.empty
        and (
            "mutation_model" not in existing_panel_metadata_df.columns
            or existing_panel_metadata_df["mutation_model"].astype(str).eq(args.mutation_model).all()
        )
    )
    use_cached_outputs_only = (
        not args.force_recompute_plm
        and not args.diagnostic_exports
        and not existing_panel_metadata_df.empty
        and cache_version_matches
        and mutation_model_matches
        and _all_model_outputs_cached(model_tables_dir, model_specs)
    )

    if use_cached_outputs_only:
        mutation_tables = None
        lineage_cache = _build_lightweight_lineage_cache_from_metadata(existing_panel_metadata_df)
    else:
        mutation_tables = build_codon_aa_mutation_tables(args.mutation_model)
        lineage_cache = build_lineage_cache(args, mutation_tables)

    target_specs = [
        {
            "label": label,
            "diversity_path": data.get("diversity_path", ""),
            "reference_path": data.get("reference_path", ""),
        }
        for label, data in lineage_cache.items()
    ]
    save_run_manifest(args, output_dir, target_specs)

    if args.diagnostic_exports and mutation_tables is not None:
        export_codon_model_diagnostics(tables_dir / "diagnostics" / str(args.mutation_model).lower(), mutation_tables)

    if not lineage_cache:
        raise RuntimeError("No valid targets were resolved for this run")

    runtime_cache: Dict[Tuple[str, str], Dict[str, object]] = {}
    metadata_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []
    all_combined_frames: List[pd.DataFrame] = []
    all_alpha_frames: List[pd.DataFrame] = []
    all_alpha_lineage_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []
    per_group_best_rows: List[Dict[str, object]] = []
    alpha_grid = parse_alpha_grid(args)
    use_parallel = args.alpha_parallel and len(alpha_grid) >= args.alpha_sweep_min_grid

    for model_spec in model_specs:
        model_label = str(model_spec["model_tag"])
        alpha_by_lineage_df = pd.DataFrame()
        cached_outputs = None if args.force_recompute_plm else _load_cached_model_outputs(model_tables_dir, model_spec)
        if cached_outputs is not None:
            model_combined_df, alpha_df = cached_outputs
            alpha_by_lineage_path = model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv"
            if alpha_by_lineage_path.exists():
                alpha_by_lineage_df = pd.read_csv(alpha_by_lineage_path, sep="\t")
            if (
                alpha_by_lineage_df.empty
                or not _alpha_table_has_complete_logistic_metrics(alpha_df)
                or not _alpha_table_has_complete_logistic_metrics(alpha_by_lineage_df)
            ):
                alpha_df, alpha_by_lineage_df = _build_alpha_tables_from_combined(
                    model_combined_df,
                    alpha_grid,
                    model_label=model_label,
                    model_spec=model_spec,
                    parallel=use_parallel,
                    max_workers=args.alpha_sweep_max_workers,
                    alpha_sweep_min_grid=args.alpha_sweep_min_grid,
                    pseudocount=1e-16,
                )
                alpha_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
                if not alpha_by_lineage_df.empty:
                    alpha_by_lineage_df.to_csv(alpha_by_lineage_path, sep="\t", index=False)
            warn_on_excess_mutation_rows(
                model_combined_df,
                context_label=f"cached combined table ({model_label})",
            )
            all_combined_frames.append(model_combined_df)
            all_alpha_frames.append(alpha_df)
            if not alpha_by_lineage_df.empty:
                all_alpha_lineage_frames.append(alpha_by_lineage_df)
            if not existing_panel_metadata_df.empty:
                cached_metadata = existing_panel_metadata_df.loc[existing_panel_metadata_df["model"] == model_label]
                metadata_rows.extend(cached_metadata.to_dict("records"))
            status_rows.append({"model": model_label, "lineage": "all", "status": "completed", "reason": "cached"})
        else:
            model_combined_rows: List[Dict[str, object]] = []

            for lineage_label, lineage_data in lineage_cache.items():
                print(
                    f"Processing {model_label} / {lineage_label}: n_seq={len(lineage_data['records'])}, "
                    f"plm_ref_len={len(lineage_data['plm_ref_protein'])}, full_ref_len={len(lineage_data['full_ref_protein'])}"
                )
                try:
                    plm_result = ensure_plm_matrix(
                        args,
                        model_spec,
                        lineage_label,
                        lineage_data,
                        plm_dir,
                        runtime_cache,
                    )
                    if len(plm_result) == 2:
                        plm_matrix, plm_path = plm_result
                        source_plm_sequence = str(lineage_data["plm_ref_protein"])
                    else:
                        plm_matrix, plm_path, source_plm_sequence = plm_result
                    resolved_coord_map, global_to_lineage_trim, remap_alignment = resolve_plm_coordinate_maps(
                        args,
                        source_plm_sequence,
                        lineage_data,
                    )
                    rows = build_combined_rows(
                        args,
                        model_spec,
                        lineage_label,
                        lineage_data,
                        plm_matrix,
                        coord_map=resolved_coord_map,
                    )
                    model_combined_rows.extend(rows)
                    lineage_data["mut_profile"].to_csv(group_dir / f"{lineage_data['lineage_key']}_mutation_accessibility_profile.csv")
                    lineage_data["obs_freq"].to_csv(group_dir / f"{lineage_data['lineage_key']}_observed_diversity_profile.csv")
                    if args.diagnostic_exports:
                        export_lineage_diagnostics(
                            args=args,
                            plot_dir=plots_dir / "diagnostics",
                            table_dir=tables_dir / "diagnostics",
                            model_label=model_label,
                            lineage_label=lineage_label,
                            lineage_data=lineage_data,
                            plm_matrix=plm_matrix,
                            coord_map=resolved_coord_map,
                            source_plm_sequence=source_plm_sequence,
                            mutation_tables=mutation_tables,
                            global_to_lineage_trim=global_to_lineage_trim,
                            remap_alignment=remap_alignment,
                        )
                    metadata_rows.append(
                        {
                            "model": model_label,
                            "epoch_label": model_spec["epoch_label"],
                            "epoch_value": float(model_spec["epoch_value"]),
                            "mutation_model": args.mutation_model,
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

            warn_on_excess_mutation_rows(
                model_combined_df,
                context_label=f"combined table before alpha sweep ({model_label})",
            )

            all_combined_frames.append(model_combined_df)
            model_combined_df.to_csv(model_tables_dir / f"{model_label}_combined_long_table.csv", index=False)

            alpha_by_lineage_df = evaluate_alpha_sweep_by_lineage(
                model_combined_df,
                alpha_grid,
                parallel=use_parallel,
                max_workers=args.alpha_sweep_max_workers,
                alpha_sweep_min_grid=args.alpha_sweep_min_grid,
                pseudocount=1e-16,
            )
            if alpha_by_lineage_df.empty:
                alpha_df = pd.DataFrame()
            else:
                alpha_df, alpha_by_lineage_df = _build_alpha_tables_from_combined(
                    model_combined_df,
                    alpha_grid,
                    model_label=model_label,
                    model_spec=model_spec,
                    parallel=use_parallel,
                    max_workers=args.alpha_sweep_max_workers,
                    alpha_sweep_min_grid=args.alpha_sweep_min_grid,
                    pseudocount=1e-16,
                )
                all_alpha_lineage_frames.append(alpha_by_lineage_df)
                alpha_by_lineage_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t", index=False)
            alpha_df.to_csv(model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
            all_alpha_frames.append(alpha_df)

        model_epoch_lineage_metrics_df = compute_epoch_lineage_metrics(model_combined_df)
        model_epoch_summary_df = summarize_epoch_metrics(model_epoch_lineage_metrics_df)
        if not model_epoch_summary_df.empty:
            mutation_baseline_cols = [
                "model",
                "epoch_label",
                "epoch_value",
                "logistic_site_mutated_vs_mut_corr_baseline",
                "spearman_obs_freq_vs_mut_baseline",
                "pearson_obs_freq_vs_mut_baseline",
                "spearman_mut_vs_mut_baseline",
                "pearson_mut_vs_mut_baseline",
            ]
            model_epoch_summary_df.loc[:, mutation_baseline_cols].to_csv(
                model_tables_dir / f"{model_label}_mutation_baseline_summary.tsv",
                sep="\t",
                index=False,
            )

        if not alpha_df.empty:
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

        if not alpha_by_lineage_df.empty:
            for lineage_name, lineage_alpha in alpha_by_lineage_df.groupby("lineage"):
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
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df["cache_version"] = PANEL_CACHE_VERSION
    if "mutation_model" not in metadata_df.columns:
        metadata_df["mutation_model"] = args.mutation_model
    metadata_df.to_csv(tables_dir / "panel_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(status_rows).to_csv(tables_dir / "model_run_status.tsv", sep="\t", index=False)

    alpha_df = pd.concat(all_alpha_frames, ignore_index=True) if all_alpha_frames else pd.DataFrame()
    if not alpha_df.empty:
        alpha_df.to_csv(tables_dir / "alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
    alpha_by_lineage_df = pd.concat(all_alpha_lineage_frames, ignore_index=True) if all_alpha_lineage_frames else pd.DataFrame()
    if not alpha_by_lineage_df.empty:
        alpha_by_lineage_df.to_csv(tables_dir / "alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t", index=False)

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
        metrics_output_dir=tables_dir,
        mutation_model=args.mutation_model,
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