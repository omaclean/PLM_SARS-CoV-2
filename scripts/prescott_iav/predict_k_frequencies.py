#!/usr/bin/env python3
"""Score ESCOTT (fitted on J) against the allele frequencies actually observed in K.

This is the evaluation ``slurm_mutation_acesiblity.sbatch.sh`` runs for a protein language
model, applied instead to the ESCOTT matrix produced by the J stage-1 run. Nothing here is
reimplemented: the observed-diversity profile, the codon-accessibility baseline and the three
metric helpers are imported from ``run_mutational_accessibility`` / ``Functions_HuggingFace``,
so the numbers are directly comparable to the PLM runs in
``Results/iav_mutational_accessibility/``.

THE THREE METRICS (rma's own definitions)

``AUROC``        ``safe_auroc(score, obs_present)`` -- can the score separate substitutions
                 that were observed AT ALL in the K panel from those that never appear?
                 Computed on the raw score, not on a fitted logistic, so it measures the
                 score itself rather than a model built on top of it.
``Spearman``     ``safe_spearman(score, obs_freq)`` over EVERY substitution, unobserved ones
                 included at frequency 0. Rank-based, so it is dominated by the
                 observed/unobserved split.
``Pearson``      ``safe_pearson(score, obs_freq)`` restricted to ``obs_freq > 0`` -- the
                 mutated-sites-only correlation. This is the hard one: given that a
                 substitution happened, does the score track HOW COMMON it got?

THE FRAME QUESTION, WHICH IS THE WHOLE DESIGN

ESCOTT was fitted on J, so its columns are J's positions and its rows are substitutions AWAY
FROM J. The K panel is therefore read in J's frame: a "mutation" is any residue differing
from J, which includes the eleven lineage-defining J->K substitutions sitting at frequency
~1.0 in every K sequence.

``--filter-fixed-mutations`` (rma's default) drops those, and that is a different question:

    fixed KEPT     "does ESCOTT predict what K became?" -- the eleven are the signal
    fixed DROPPED  "does ESCOTT predict what varies WITHIN K?" -- the eleven are excluded
                   as foregone conclusions and only segregating diversity is scored

Both are reported, because quoting either alone would be misleading.

Every metric is also computed for the codon-accessibility baseline (``mut_prob``), because a
score that merely rediscovers which substitutions are one nucleotide away is not predicting
biology, and only the comparison shows which is happening.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for extra in (str(REPO_ROOT), str(SCRIPT_DIR.parent)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from prescott_iav import common  # noqa: E402
from prescott_iav.jk_impact_report import AA20, load_probability_matrix, load_raw_matrix  # noqa: E402

import run_mutational_accessibility as rma  # noqa: E402
from Functions_HuggingFace import (  # noqa: E402
    _load_single_focal_reference,
    build_codon_aa_mutation_tables,
    build_reference_to_alignment_column_map,
    compute_lineage_mutation_profile,
    compute_observed_diversity_profile_fast,
)

IGNORE = rma.IGNORE_ALIGNMENT_CHARS


def observed_profile(panel_fasta: Path, reference_fasta: Path, label: str,
                     mutation_tables: Dict[str, object], test_mode: bool = False):
    """Observed allele frequencies for one panel, in the given reference's frame."""
    payload = _load_single_focal_reference(str(reference_fasta), label)
    ref_nt, ref_protein = payload["nucleotide"], payload["protein"]
    records, _ = rma.load_diversity_records(
        panel_fasta, expect_protein_diversity=False,
        test_mode=test_mode, test_max_records=2000)
    if not records:
        raise SystemExit(f"no diversity records in {panel_fasta}")
    ref_to_aln, aln_len, matched = build_reference_to_alignment_column_map(
        ref_protein, records, mutation_tables["aa_to_codons"], IGNORE)
    obs_freq, obs_depth, stats = compute_observed_diversity_profile_fast(
        records, ref_protein, ref_to_aln, aln_len,
        mutation_tables["aa_to_codons"], IGNORE)
    mut_profile = compute_lineage_mutation_profile(
        ref_nt, ref_protein, mutation_tables["aa_to_codons"],
        mutation_tables["codon_mutation_df"])
    return {
        "ref_nt": ref_nt, "ref_protein": ref_protein, "n_records": len(records),
        "obs_freq": obs_freq, "obs_depth": obs_depth, "mut_profile": mut_profile,
        "matched_pairs": matched, "stats": stats,
    }


def build_rows(profile: Dict[str, object], escott_raw: pd.DataFrame,
               escott_prob: pd.DataFrame) -> pd.DataFrame:
    """One row per (position, substituted residue), joining score to observation."""
    ref_protein = profile["ref_protein"]
    obs_freq, obs_depth, mut_profile = (profile["obs_freq"], profile["obs_depth"],
                                        profile["mut_profile"])
    rows: List[Dict[str, object]] = []
    for index, ref_aa in enumerate(ref_protein):
        position = index + 1
        if position not in mut_profile.columns or position not in escott_raw.columns:
            continue
        depth = float(obs_depth.get(position, 0))
        for aa in AA20:
            if aa == ref_aa or aa not in mut_profile.index or aa not in obs_freq.index:
                continue
            frequency = float(obs_freq.loc[aa, position])
            rows.append({
                "position": position, "ref_aa": ref_aa, "aa": aa,
                "escott_raw": float(escott_raw.at[aa, position]),
                "escott_prob": float(escott_prob.at[aa, position]),
                "mut_prob": float(mut_profile.loc[aa, position]),
                "obs_freq": frequency,
                "obs_present": 1 if frequency > 0 else 0,
                "depth": depth,
            })
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, score_column: str) -> Dict[str, float]:
    """rma's three headline metrics, using rma's own helpers."""
    mutated = frame.loc[frame["obs_freq"] > 0]
    return {
        "auroc_obs_present": rma.safe_auroc(frame[score_column], frame["obs_present"]),
        "spearman_obs_freq": rma.safe_spearman(frame[score_column], frame["obs_freq"]),
        "pearson_obs_freq_mutated_only": rma.safe_pearson(mutated[score_column],
                                                          mutated["obs_freq"]),
        "spearman_obs_freq_mutated_only": rma.safe_spearman(mutated[score_column],
                                                            mutated["obs_freq"]),
        "n_rows": int(len(frame)),
        "n_mutated": int(len(mutated)),
    }


def alpha_sweep(frame: pd.DataFrame, score_column: str, label: str,
                alphas: Sequence[float]) -> pd.DataFrame:
    """rma's combined score: log10(model) + alpha * log10(mut_prob).

    Neither term wins alone -- ESCOTT knows what is structurally tolerable and the codon
    model knows what is reachable in one nucleotide -- so the pipeline sweeps the weight
    between them rather than picking one. Reproduced here on ESCOTT instead of a PLM.

    ``escott_prob`` is the faithful analogue of ``plm_prob`` (both are per-column softmax
    probabilities). ``escott_raw`` is swept too because the per-column softmax discards each
    site's overall magnitude, which a cross-position evaluation like this one actually needs.
    """
    # escott_raw and the score_* columns are already on a log-like additive scale;
    # only a bare probability column needs logging first.
    values = frame[score_column].to_numpy(dtype=float)
    base = np.log10(np.clip(values, 1e-32, None)) if score_column == "escott_prob" else values
    log_mut = np.log10(np.clip(frame["mut_prob"].to_numpy(dtype=float), 1e-32, None))
    rows = []
    for alpha in alphas:
        combined = pd.Series(base + alpha * log_mut, index=frame.index)
        scored = frame.assign(_combined=combined)
        stats = metrics(scored, "_combined")
        rows.append({"score": label, "alpha": float(alpha), **stats})
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--score-lineage", default="J_int",
                        help="the lineage ESCOTT was fitted on; supplies the score matrix")
    parser.add_argument("--target-lineage", default="K",
                        help="the lineage whose observed allele frequencies are predicted")
    parser.add_argument("--guide-path", type=Path,
                        default=REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv")
    parser.add_argument("--mutation-model", default="H3N2", choices=("SC2", "H1N1", "H3N2"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--extra-score", action="append", default=None, metavar="LABEL=PATH",
                        help="repeatable; another score matrix in plm_cache format, scored on "
                             "the SAME panel in the SAME frame so the numbers are comparable")
    parser.add_argument("--alpha-start", type=float, default=0.0)
    # Upper bound 1.0 on purpose. The combined score is plm_prob * mut_prob^alpha
    # (rma:1895), so alpha is the EXPONENT on the codon term: alpha=1 multiplies two
    # probabilities as independent evidence, and alpha>1 raises mut_prob to a power, which
    # is no longer a probability product and simply slides the score toward the codon
    # baseline. The driver's own sweep is [-1, 1] (run_prescott_diversity.py:536-537).
    parser.add_argument("--alpha-stop", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    out_dir = common.ensure_dir(args.out_dir or run_dir / "JtoK_report")
    tables_dir = common.ensure_dir(out_dir / "tables")

    guide = {row["label"]: row for row in common.read_guide_rows(args.guide_path)}
    if args.target_lineage not in guide:
        raise SystemExit(f"{args.target_lineage} not in {args.guide_path}")

    key = common.safe_label(args.score_lineage)
    escott_raw = load_raw_matrix(run_dir / "scores" / f"{key}_ESCOTT_raw.tsv")
    escott_prob, score_protein = load_probability_matrix(
        run_dir / "scores" / f"{key}_ESCOTT_score_matrix.csv")

    mutation_tables = build_codon_aa_mutation_tables(args.mutation_model)

    # J's frame throughout: ESCOTT's columns ARE J's positions, so the K panel must be read
    # against J's reference or the scores would be joined to the wrong sites.
    profile = observed_profile(
        Path(guide[args.target_lineage]["diversity_path"]),
        Path(guide[args.score_lineage]["reference_path"]),
        f"{args.target_lineage}_in_{args.score_lineage}_frame",
        mutation_tables, test_mode=args.test_mode)
    if profile["ref_protein"] != score_protein:
        raise SystemExit("score matrix and reference protein disagree")
    print(f"@> {args.target_lineage} panel: {profile['n_records']:,} sequences, "
          f"read in {args.score_lineage}'s frame ({profile['matched_pairs']} matched sites)")

    frame = build_rows(profile, escott_raw, escott_prob)

    # Extra matrices are joined onto the SAME rows -- same panel, same frame, same reference --
    # which is the only way these numbers can sit in one table. The per-lineage PLM runs under
    # Results/iav_mutational_accessibility/ evaluate each lineage against ITS OWN reference, so
    # their published metrics answer a different question and must not be pasted in here.
    extra_labels: List[str] = []
    for entry in (args.extra_score or []):
        label, _, path = entry.partition("=")
        matrix, matrix_protein = load_probability_matrix(Path(path))
        if matrix_protein != score_protein:
            raise SystemExit(f"{label}: matrix was fitted on a different reference sequence")
        column = f"score_{label}"
        frame[column] = [float(np.log10(max(matrix.at[r.aa, r.position], 1e-32)))
                         for r in frame.itertuples()]
        extra_labels.append(label)
        print(f"@> joined {label} from {path}")
    fixed = frame["obs_freq"] >= 1.0
    print(f"@> {len(frame):,} substitution rows; {int(frame['obs_present'].sum()):,} observed; "
          f"{int(fixed.sum())} at frequency >= 1.0 (lineage-defining)")

    results: Dict[str, object] = {}
    for view, subset in (("fixed_kept", frame),
                         ("fixed_dropped", frame.loc[~fixed])):
        block = {"ESCOTT (raw score)": metrics(subset, "escott_raw")}
        for label in extra_labels:
            block[label] = metrics(subset, f"score_{label}")
        block["codon accessibility baseline"] = metrics(subset, "mut_prob")
        results[view] = block

    alphas = np.round(np.arange(args.alpha_start, args.alpha_stop + 1e-9, args.alpha_step), 3)
    sweeps = []
    for view, subset in (("fixed_kept", frame), ("fixed_dropped", frame.loc[~fixed])):
        sweep_targets = [("escott_raw", "ESCOTT raw")]
        sweep_targets += [(f"score_{l}", l) for l in extra_labels]
        for column, label in sweep_targets:
            sweep = alpha_sweep(subset, column, label, alphas)
            sweep.insert(0, "view", view)
            sweeps.append(sweep)
    sweep_df = pd.concat(sweeps, ignore_index=True)
    sweep_df.to_csv(tables_dir / "k_frequency_alpha_sweep.tsv", sep="\t", index=False)

    print("\n=== alpha sweep: best alpha per metric ===")
    for view in ("fixed_kept", "fixed_dropped"):
        for label in sweep_df["score"].unique():
            block = sweep_df[(sweep_df["view"] == view) & (sweep_df["score"] == label)]
            best_auc = block.loc[block["auroc_obs_present"].idxmax()]
            best_sp = block.loc[block["spearman_obs_freq"].idxmax()]
            best_pe = block.loc[block["pearson_obs_freq_mutated_only"].idxmax()]
            edge = "  [AT SWEEP BOUNDARY - wants more codon weight than alpha<=1 allows]" \
                if abs(float(best_auc["alpha"]) - float(block["alpha"].max())) < 1e-9 else ""
            print(f"  [{view}] {label}")
            print(f"      AUROC        best {best_auc['auroc_obs_present']:.3f} at a={best_auc['alpha']:.1f}{edge}")
            print(f"      Spearman     best {best_sp['spearman_obs_freq']:.3f} at a={best_sp['alpha']:.1f}")
            print(f"      Pearson(mut) best {best_pe['pearson_obs_freq_mutated_only']:.3f} at a={best_pe['alpha']:.1f}")

    frame.to_csv(tables_dir / "k_frequency_prediction_rows.tsv.gz", sep="\t", index=False)
    (tables_dir / "k_frequency_prediction_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    for view, block in results.items():
        print(f"\n=== {view} ===")
        print(f"{'score':<40}{'AUROC':>9}{'Spearman':>11}{'Pearson(mut)':>14}{'n':>9}{'n_mut':>8}")
        for name, m in block.items():
            print(f"{name:<40}{m['auroc_obs_present']:>9.3f}{m['spearman_obs_freq']:>11.3f}"
                  f"{m['pearson_obs_freq_mutated_only']:>14.3f}"
                  f"{m['n_rows']:>9,}{m['n_mutated']:>8,}")
    print(f"\n@> rows   -> {tables_dir/'k_frequency_prediction_rows.tsv.gz'}")
    print(f"@> metrics-> {tables_dir/'k_frequency_prediction_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
