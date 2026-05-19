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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import Align, SeqIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IGNORE_ALIGNMENT_CHARS = {"-", "*", "."}
PANEL_CACHE_VERSION = 2


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
    if not (combined_path.exists() and alpha_path.exists()):
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
    from matplotlib.ticker import NullLocator
    from scipy.stats import spearmanr
    from Functions_HuggingFace import evaluate_alpha_sweep

    def _safe_output_label(text: object) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "model"

    def _hide_log_minor_ticks(ax) -> None:
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="minor", bottom=False, top=False, left=False, right=False)

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

        baseline_df = (
            plot_frame.loc[:, sorted(required_cols)]
            .drop_duplicates()
            .copy()
        )
        if baseline_df.empty:
            return None

        baseline_df["plm_prob"] = 1.0
        baseline_metrics = evaluate_alpha_sweep(
            baseline_df,
            np.array([1.0]),
            parallel=False,
            pseudocount=1e-16,
        )
        if baseline_metrics.empty:
            return None
        if baseline_output_path is not None:
            baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_metrics.to_csv(baseline_output_path, sep="\t", index=False)
        return baseline_metrics.iloc[0]

    def _compute_site_logistic_alpha_metrics(
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        required_cols = {"lineage", "position", "ref_aa", "plm_prob", "mut_prob", "obs_present"}
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

            score_base_df = group_plot_df.loc[:, ["lineage", "position", "ref_aa", "plm_prob", "mut_prob", "obs_present"]].copy()
            score_base_df["plm_prob"] = score_base_df["plm_prob"].clip(lower=1e-32)
            score_base_df["mut_prob"] = score_base_df["mut_prob"].clip(lower=1e-32)

            for alpha_value in sorted(group_alpha_df["alpha"].dropna().astype(float).unique().tolist()):
                working_df = score_base_df.copy()
                working_df["combined_prob"] = working_df["plm_prob"] * np.power(working_df["mut_prob"], alpha_value)
                site_df = (
                    working_df.groupby(["lineage", "position", "ref_aa"], as_index=False)
                    .agg(
                        site_score=("combined_prob", "max"),
                        site_mutated=("obs_present", "max"),
                    )
                )
                try:
                    logistic_corr, _, _ = fit_logistic_site_correlation(site_df["site_score"], site_df["site_mutated"])
                except Exception:
                    logistic_corr = np.nan
                logistic_row: Dict[str, object] = {"alpha": float(alpha_value), "site_logistic_mutated_corr": logistic_corr}
                logistic_row.update(group_key)
                logistic_rows.append(logistic_row)

        return pd.DataFrame(logistic_rows)

    def _compute_mutation_only_logistic_baseline(plot_frame: pd.DataFrame) -> float:
        required_cols = {"lineage", "position", "ref_aa", "mut_prob", "obs_present"}
        if plot_frame.empty or not required_cols.issubset(plot_frame.columns):
            return np.nan

        site_df = (
            plot_frame.loc[:, ["lineage", "position", "ref_aa", "mut_prob", "obs_present"]]
            .groupby(["lineage", "position", "ref_aa"], as_index=False)
            .agg(
                site_score=("mut_prob", "max"),
                site_mutated=("obs_present", "max"),
            )
        )
        logistic_corr, _, _ = fit_logistic_site_correlation(site_df["site_score"], site_df["site_mutated"])
        return logistic_corr

    def _build_selected_alpha_metric_frame(
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_cols = [
            col for col in ["alpha", "model", "epoch_label", "epoch_value", "mut_flat_global_spearman_r", "mut_flat_nonzero_pearson_r"]
            if col in alpha_frame.columns
        ]
        selected_alpha_df = alpha_frame.loc[:, selected_cols].copy()
        logistic_alpha_df = _compute_site_logistic_alpha_metrics(plot_frame, alpha_frame)
        if logistic_alpha_df.empty:
            selected_alpha_df["site_logistic_mutated_corr"] = np.nan
            return selected_alpha_df

        merge_cols = [col for col in ["model", "epoch_label", "epoch_value", "alpha"] if col in selected_alpha_df.columns and col in logistic_alpha_df.columns]
        return selected_alpha_df.merge(logistic_alpha_df, on=merge_cols, how="left")

    def _plot_alpha_metric_grid(
        plot_frame: pd.DataFrame,
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
        cmap = plt.get_cmap("coolwarm_r")
        epoch_colours = [cmap(i / max(1, n_epochs - 1)) for i in range(n_epochs)]
        include_model_in_label = bool("model" in plot_frame.columns and plot_frame["model"].nunique() > 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.array(axes, dtype=object).reshape(-1)
        for i, metric_col in enumerate(metric_cols):
            ax = axes[i]
            if epoch_groups is not None:
                for colour_idx, ((ep_val, ep_label, _model_name), grp) in enumerate(epoch_groups):
                    grp_sorted = grp.loc[np.isfinite(grp["alpha"]) & np.isfinite(grp[metric_col])].sort_values("alpha")
                    if grp_sorted.empty:
                        continue
                    tick_label = _format_epoch_tick_label(ep_label, ep_val)
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
                    ax.set_title(title_map.get(metric_col, metric_col))
                    ax.set_xlabel("Alpha")
                    ax.set_ylabel("Metric value")
                    ax.grid(alpha=0.3)
                    continue
                ax.plot(ax_sorted["alpha"], ax_sorted[metric_col], marker="o", linewidth=1.5)

            if (
                mutation_only_metrics is not None
                and metric_col in mutation_only_metrics.index
                and np.isfinite(float(mutation_only_metrics["alpha"]))
                and np.isfinite(float(mutation_only_metrics[metric_col]))
            ):
                ax.scatter(
                    [float(mutation_only_metrics["alpha"])],
                    [float(mutation_only_metrics[metric_col])],
                    marker="s",
                    s=110,
                    color="#d62728",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label="Mutation only",
                )

            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(title_map.get(metric_col, metric_col))
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Metric value")
            ax.grid(alpha=0.3)

        for ax in axes[len(metric_cols):]:
            ax.axis("off")

        handles, labels = _collect_axes_legend_entries(axes)
        if handles:
            fig.legend(handles, labels, title="Epoch", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        plt.tight_layout(rect=(0, 0, 0.84, 1))
        plt.savefig(output_dir / output_name, dpi=300)
        plt.close()

    def _write_selected_alpha_sweep_plot(
        target_dir: Path,
        plot_frame: pd.DataFrame,
        alpha_frame: pd.DataFrame,
        title_prefix: str,
        output_name: str = "alpha_sweep_metrics_selected.png",
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
            mutation_only_metrics["site_logistic_mutated_corr"] = mutation_only_logistic

        plot_metric_cols = [
            metric_col for metric_col in [
                "mut_flat_global_spearman_r",
                "mut_flat_nonzero_pearson_r",
                "site_logistic_mutated_corr",
            ]
            if metric_col in focused_alpha_df.columns
        ]
        if not plot_metric_cols:
            return

        fig, axes = plt.subplots(1, len(plot_metric_cols), figsize=(6 * len(plot_metric_cols), 5), sharex=True)
        axes = np.array(axes, dtype=object).reshape(-1)
        title_map = {
            "mut_flat_global_spearman_r": "Spearman(score vs observed freq), all 19xN mutation rows",
            "mut_flat_nonzero_pearson_r": "Pearson(score vs observed freq), non-zero allele frequencies only",
            "site_logistic_mutated_corr": "Logistic regression: site mutated (>0 obs freq) vs score",
        }

        epoch_groups = (
            focused_alpha_df.groupby(["epoch_value", "epoch_label", "model"], sort=True)
            if all(c in focused_alpha_df.columns for c in ["epoch_value", "epoch_label", "model"])
            else None
        )
        n_groups = len(epoch_groups) if epoch_groups is not None else 1
        cmap = plt.get_cmap("coolwarm_r")
        epoch_colours = [cmap(i / max(1, n_groups - 1)) for i in range(n_groups)]
        for ax, metric_col in zip(axes, plot_metric_cols):
            if epoch_groups is not None:
                for colour_idx, ((ep_val, ep_label, model_name), grp) in enumerate(epoch_groups):
                    grp_sorted = grp.loc[np.isfinite(grp["alpha"]) & np.isfinite(grp[metric_col])].sort_values("alpha")
                    if grp_sorted.empty:
                        continue
                    tick_label = _format_epoch_tick_label(ep_label, ep_val)
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
                grp_sorted = focused_alpha_df.loc[np.isfinite(focused_alpha_df["alpha"]) & np.isfinite(focused_alpha_df[metric_col])].sort_values("alpha")
                if grp_sorted.empty:
                    ax.set_title(title_map[metric_col])
                    ax.set_xlabel("Alpha")
                    ax.set_ylabel("Metric value")
                    ax.grid(alpha=0.3)
                    continue
                ax.plot(grp_sorted["alpha"], grp_sorted[metric_col], marker="o", linewidth=1.5)

            if (
                mutation_only_metrics is not None
                and metric_col in mutation_only_metrics.index
                and np.isfinite(float(mutation_only_metrics["alpha"]))
                and np.isfinite(float(mutation_only_metrics[metric_col]))
            ):
                ax.scatter(
                    [float(mutation_only_metrics["alpha"])],
                    [float(mutation_only_metrics[metric_col])],
                    marker="s",
                    s=110,
                    color="#d62728",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=6,
                    label="Mutation only",
                )

            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_title(title_map[metric_col])
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Metric value")
            ax.grid(alpha=0.3)

        handles, labels = _collect_axes_legend_entries(axes)
        if handles:
            fig.legend(handles, labels, title="Epoch", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.suptitle(title_prefix)
        plt.tight_layout(rect=(0, 0, 0.84, 0.95))
        plt.savefig(target_dir / output_name, dpi=300)
        plt.close(fig)

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
        ax.set_title(title)
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
        ax.set_xlabel("PLM Probability")
        ax.set_ylabel("Mutation Probability")
        ax.set_title(
            f"{title_prefix}\n"
            f"Spearman ρ(plm_prob, mut_prob)={rho:.3f}; "
            f"Pearson r(plm_prob, mut_prob)={pearson_r:.3f}"
        )
        ax.grid(True, which="major", ls="--", alpha=0.4)
        fig.tight_layout()
        export_publication_figure(target_dir / "plm_vs_mut_prob_scatter.png", figure=fig)
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

    if not alpha_df.empty:
        mutation_only_metrics = _compute_mutation_only_alpha_baseline(
            combined_df,
            baseline_output_path=output_dir / "alpha_sweep_fit_metrics_mutation_only.tsv",
        )
        _plot_alpha_metric_grid(
            alpha_df,
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
        for model_label, model_combined_df in combined_df.groupby("model", sort=True):
            model_plot_dir = ensure_dir(per_model_dir / _safe_output_label(model_label))
            model_alpha_df = alpha_df.loc[alpha_df["model"] == model_label].copy() if "model" in alpha_df.columns else alpha_df.copy()
            _write_selected_alpha_sweep_plot(
                model_plot_dir,
                model_combined_df,
                model_alpha_df,
                title_prefix=f"Focused alpha-sweep metrics ({model_label})",
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
            output_dir,
            combined_df,
            alpha_df,
            title_prefix="Focused alpha-sweep metrics (all models)",
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


def run_analysis(args: argparse.Namespace) -> int:
    from Functions_HuggingFace import build_codon_aa_mutation_tables, evaluate_alpha_sweep

    args = apply_arg_defaults(args)
    output_dir = ensure_dir(args.output_dir)
    group_dir = ensure_dir(output_dir / "groups")
    plm_dir = ensure_dir(output_dir / "plm_cache")
    tables_dir = ensure_dir(output_dir / "tables")
    plots_dir = ensure_dir(output_dir / "plots")
    model_tables_dir = ensure_dir(tables_dir / "per_model")

    model_specs = build_model_specs(args)
    existing_panel_metadata_path = tables_dir / "panel_metadata.tsv"
    existing_panel_metadata_df = pd.read_csv(existing_panel_metadata_path, sep="\t") if existing_panel_metadata_path.exists() else pd.DataFrame()
    cache_version_matches = (
        not existing_panel_metadata_df.empty
        and "cache_version" in existing_panel_metadata_df.columns
        and pd.to_numeric(existing_panel_metadata_df["cache_version"], errors="coerce").eq(PANEL_CACHE_VERSION).all()
    )
    mutation_model_matches = (
        not existing_panel_metadata_df.empty
        and "mutation_model" in existing_panel_metadata_df.columns
        and existing_panel_metadata_df["mutation_model"].astype(str).eq(args.mutation_model).all()
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
    best_rows: List[Dict[str, object]] = []
    per_group_best_rows: List[Dict[str, object]] = []
    alpha_grid = parse_alpha_grid(args)
    use_parallel = args.alpha_parallel and len(alpha_grid) >= args.alpha_sweep_min_grid

    for model_spec in model_specs:
        model_label = str(model_spec["model_tag"])
        cached_outputs = None if args.force_recompute_plm else _load_cached_model_outputs(model_tables_dir, model_spec)
        if cached_outputs is not None:
            model_combined_df, alpha_df = cached_outputs
            all_combined_frames.append(model_combined_df)
            all_alpha_frames.append(alpha_df)
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

            all_combined_frames.append(model_combined_df)
            model_combined_df.to_csv(model_tables_dir / f"{model_label}_combined_long_table.csv", index=False)

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
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df["cache_version"] = PANEL_CACHE_VERSION
    if "mutation_model" not in metadata_df.columns:
        metadata_df["mutation_model"] = args.mutation_model
    metadata_df.to_csv(tables_dir / "panel_metadata.tsv", sep="\t", index=False)
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