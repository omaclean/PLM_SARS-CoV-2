#!/usr/bin/env python3
"""Compare ESCOTT's per-substitution scores against a protein language model's, for one lineage.

Both halves of this repository score the same object -- every single amino-acid substitution
away from a lineage reference -- and both write the same file layout, so they can be put on
one pair of axes with no reformatting:

    ESCOTT   scores/<key>_ESCOTT_raw.tsv            -normPred; 0 = tolerated, negative = worse
    PLM      plm_cache/<key>_<tag>_plm_probability_profile.csv   softmax over 20 residues

They are NOT on the same scale and there is no reason they should be, which is the whole
point of the comparison. ESCOTT's value is a log-odds-like quantity already; the PLM's is a
probability in [0, 1]. So the scatter is drawn on four scale treatments at once
(:func:`fig_scale_grid`), and both Spearman and Pearson are reported for each:

* **Spearman is identical across all four** -- it only sees ranks, and every transform here
  is monotone. That is the honest headline number for "do the two models agree".
* **Pearson changes**, and which treatment maximises it is the actual answer to
  "log space or linear": it says which transform makes the relationship straight.

Colour throughout is the **minimum number of nucleotide substitutions** needed to reach that
amino acid from the lineage's own codon (1, 2 or 3). This is not decoration. A model scoring
amino-acid sequences has no idea what the underlying codon was, so any agreement or
disagreement that tracks nt distance is a property of the genetic code leaking into the
comparison rather than of either model's biology.

Positions whose ESCOTT column is constant (``trace == 0`` or fully conserved) are dropped:
they sit at exactly 0.0, the top of the ESCOTT scale, and would otherwise paint a false
horizontal stripe of "perfectly tolerated" through every panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402
from prescott_iav.jk_impact_report import (  # noqa: E402
    AA20, AXIS, CODON_TABLE, GRID, INK, INK2, MUTED, SURFACE,
    apply_style, informative_positions, load_probability_matrix, load_raw_matrix,
)

# Minimum-nucleotide-distance is an ORDINAL variable (1 < 2 < 3), so it takes a single-hue
# ordinal ramp rather than three categorical hues.  Steps 250/450/650 of the reference blue
# ramp: the lightest still clears 2:1 on the light surface, which is the ordinal floor.
NT_COLOURS: Dict[int, str] = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
NT_LABELS: Dict[int, str] = {
    1: "1 nt change",
    2: "2 nt changes",
    3: "3 nt changes",
}


# --------------------------------------------------------------------------- #
# Codon distance
# --------------------------------------------------------------------------- #

def codon_distance_table(cds: str, protein: str) -> pd.DataFrame:
    """Minimum nucleotide substitutions from the lineage's own codon to each amino acid.

    Minimum over every codon encoding the target residue, so this is the cheapest route the
    genetic code allows -- not the distance to some canonical codon.
    """
    by_amino_acid: Dict[str, List[str]] = {}
    for codon, amino_acid in CODON_TABLE.items():
        by_amino_acid.setdefault(amino_acid, []).append(codon)

    records: List[Dict[str, object]] = []
    for index, wt in enumerate(protein):
        position = index + 1
        codon = cds[3 * index: 3 * index + 3].upper()
        if CODON_TABLE.get(codon) != wt:
            raise ValueError(f"codon {codon!r} at position {position} does not encode {wt}")
        for mut in AA20:
            if mut == wt:
                continue
            distance = min(
                sum(1 for a, b in zip(codon, candidate) if a != b)
                for candidate in by_amino_acid[mut]
            )
            records.append({"ha0_pos": position, "wt": wt, "mut": mut,
                            "codon": codon, "nt_distance": int(distance)})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_frame(escott_raw: pd.DataFrame, escott_prob: pd.DataFrame,
                plm: pd.DataFrame, distances: pd.DataFrame,
                keep: Sequence[int]) -> pd.DataFrame:
    allowed = set(int(position) for position in keep)
    frame = distances[distances["ha0_pos"].isin(allowed)].copy()
    frame["escott_raw"] = [escott_raw.at[row.mut, row.ha0_pos] for row in frame.itertuples()]
    frame["escott_prob"] = [escott_prob.at[row.mut, row.ha0_pos] for row in frame.itertuples()]
    frame["plm_prob"] = [plm.at[row.mut, row.ha0_pos] for row in frame.itertuples()]
    # log10 of a probability that can underflow to 0 in float32 -- floor it at the smallest
    # value actually present rather than letting -inf silently drop points from the fit.
    floor = float(frame.loc[frame["plm_prob"] > 0, "plm_prob"].min())
    frame["plm_log10"] = np.log10(frame["plm_prob"].clip(lower=floor))
    frame["escott_prob_log10"] = np.log10(frame["escott_prob"].clip(lower=1e-300))
    return frame.reset_index(drop=True)


def correlations(frame: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    xs = frame[x].to_numpy(dtype=float)
    ys = frame[y].to_numpy(dtype=float)
    good = np.isfinite(xs) & np.isfinite(ys)
    return {
        "n": int(good.sum()),
        "spearman": float(stats.spearmanr(xs[good], ys[good]).statistic),
        "pearson": float(stats.pearsonr(xs[good], ys[good]).statistic),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _scatter_by_distance(axes, frame: pd.DataFrame, x: str, y: str,
                         size: float = 3.4, alpha: float = 0.32) -> None:
    """Draw farthest-first so the 1-nt points -- the ones evolution can actually use -- sit on top."""
    for distance in (3, 2, 1):
        subset = frame[frame["nt_distance"] == distance]
        axes.scatter(subset[x], subset[y], s=size, c=NT_COLOURS[distance], alpha=alpha,
                     linewidths=0, rasterized=True, zorder=distance_zorder(distance))


def distance_zorder(distance: int) -> int:
    return {3: 2, 2: 3, 1: 4}[distance]


def _distance_legend(axes, frame: pd.DataFrame, loc: str = "lower right") -> None:
    counts = frame["nt_distance"].value_counts()
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.5,
               markerfacecolor=NT_COLOURS[distance], markeredgecolor="none",
               label=f"{NT_LABELS[distance]}  (n = {int(counts.get(distance, 0)):,})")
        for distance in (1, 2, 3)
    ]
    axes.legend(handles=handles, loc=loc, fontsize=8, title="minimum route through the code",
                title_fontsize=8, labelspacing=0.35)


def binned_median(frame: pd.DataFrame, x: str, y: str, edges: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and interquartile band of `y` within bins of `x`, for a readable trend line.

    Ten thousand semi-transparent points cannot carry a three-way colour comparison -- the
    largest class simply paints over the others.  The cloud gives density; these lines carry
    the actual contrast between codon classes.
    """
    centres, medians, lower, upper = [], [], [], []
    values_x = frame[x].to_numpy(dtype=float)
    values_y = frame[y].to_numpy(dtype=float)
    for start, stop in zip(edges[:-1], edges[1:]):
        inside = (values_x >= start) & (values_x < stop)
        if inside.sum() < 12:          # too few points to quote a quartile from
            continue
        centres.append((start + stop) / 2)
        medians.append(np.median(values_y[inside]))
        lower.append(np.percentile(values_y[inside], 25))
        upper.append(np.percentile(values_y[inside], 75))
    return (np.array(centres), np.array(medians), np.array(lower), np.array(upper))


def fig_main(frame: pd.DataFrame, plm_label: str, out_path: Path) -> Path:
    """The headline panel plus one density facet per nt class."""
    figure = plt.figure(figsize=(12.6, 6.8))
    grid = figure.add_gridspec(2, 3, width_ratios=[1.75, 1, 1], height_ratios=[1, 1],
                               hspace=0.46, wspace=0.30)
    main = figure.add_subplot(grid[:, 0])

    _scatter_by_distance(main, frame, "escott_raw", "plm_log10", size=3.0, alpha=0.20)
    edges = np.linspace(frame["escott_raw"].min(), frame["escott_raw"].max(), 13)
    for distance in (1, 2, 3):
        subset = frame[frame["nt_distance"] == distance]
        centres, medians, lower, upper = binned_median(subset, "escott_raw", "plm_log10", edges)
        if not len(centres):
            continue
        main.fill_between(centres, lower, upper, color=NT_COLOURS[distance], alpha=0.13,
                          linewidth=0, zorder=5)
        main.plot(centres, medians, color=SURFACE, linewidth=4.4, zorder=6)
        main.plot(centres, medians, color=NT_COLOURS[distance], linewidth=2.4, zorder=7,
                  marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.9)

    overall = correlations(frame, "escott_raw", "plm_log10")
    main.set_xlabel("ESCOTT score   (0 = tolerated, negative = deleterious)")
    main.set_ylabel(f"log₁₀ P(residue)   -   {plm_label}")
    main.set_title(f"ESCOTT vs {plm_label}\n"
                   f"Spearman ρ = {overall['spearman']:.3f}    "
                   f"Pearson r = {overall['pearson']:.3f}    n = {overall['n']:,}",
                   fontsize=10.5, pad=10)
    _distance_legend(main, frame)
    main.spines[["top", "right"]].set_visible(False)

    # Explicit cell placement: the two-row block to the right of the main panel holds the
    # three class facets plus one note. Deriving these from a loop index is how they
    # previously ended up drawn on top of each other.
    cells = {1: grid[0, 1], 2: grid[0, 2], 3: grid[1, 1]}
    for distance, cell in cells.items():
        axis = figure.add_subplot(cell)
        subset = frame[frame["nt_distance"] == distance]
        axis.hexbin(subset["escott_raw"], subset["plm_log10"], gridsize=30, mincnt=1,
                    cmap="Blues", linewidths=0, extent=(*main.get_xlim(), *main.get_ylim()))
        pair = correlations(subset, "escott_raw", "plm_log10")
        axis.set_title(f"{NT_LABELS[distance]}   ρ = {pair['spearman']:.3f}",
                       fontsize=9, color=NT_COLOURS[distance], pad=6)
        axis.set_xlim(main.get_xlim())
        axis.set_ylim(main.get_ylim())
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)

    note = figure.add_subplot(grid[1, 2])
    note.axis("off")
    note.text(0.0, 1.0, "Reading the facets", transform=note.transAxes, fontsize=9.5,
              fontweight="bold", va="top", color=INK)
    note.text(0.0, 0.84,
              "Each facet is the main panel's axes,\n"
              "restricted to one codon class.\n\n"
              "If the three clouds sat in the same\n"
              "place, agreement between the models\n"
              "would be independent of the genetic\n"
              "code. They do not: agreement decays\n"
              "as the codon route gets longer, so\n"
              "part of what looks like model\n"
              "agreement is codon structure.",
              transform=note.transAxes, fontsize=8, va="top", color=INK2, linespacing=1.45)

    figure.savefig(out_path, bbox_inches="tight", dpi=170)
    plt.close(figure)
    return out_path


def fig_scale_grid(frame: pd.DataFrame, plm_label: str, out_path: Path) -> Tuple[Path, List[Dict]]:
    """The log-vs-linear question, answered on all four combinations at once."""
    # The grid is deliberately a 2 x 2 of TWO DIFFERENT THINGS, and the panel titles say which:
    #   rows    = which ESCOTT QUANTITY is on x (raw score vs per-column softmax probability)
    #   columns = which SCALE the PLM probability is drawn on (linear vs log)
    # Only the rows can move Spearman. Calling the second row "both linear/both log" -- as if it
    # were a scale change like the columns -- makes the rho drop look like a broken invariant
    # when it is really a change of variable.
    treatments = [
        ("escott_raw", "plm_prob", "ESCOTT raw score", "P(residue), linear",
         "ESCOTT raw score  ·  PLM linear"),
        ("escott_raw", "plm_log10", "ESCOTT raw score", "log₁₀ P(residue)",
         "ESCOTT raw score  ·  PLM log"),
        ("escott_prob", "plm_prob", "ESCOTT softmax probability", "P(residue), linear",
         "ESCOTT softmax P  ·  PLM linear"),
        ("escott_prob_log10", "plm_log10", "log₁₀ ESCOTT softmax probability",
         "log₁₀ P(residue)", "ESCOTT softmax P  ·  PLM log"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 9.0))
    rows: List[Dict] = []
    for axis, (x, y, xlabel, ylabel, title) in zip(axes.ravel(), treatments):
        _scatter_by_distance(axis, frame, x, y, size=2.6, alpha=0.26)
        stat = correlations(frame, x, y)
        rows.append({"treatment": title, "x": x, "y": y, **stat})
        axis.set_xlabel(xlabel, fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_title(f"{title}\nρ = {stat['spearman']:.3f}    r = {stat['pearson']:.3f}",
                       fontsize=9.5, pad=8)
        axis.tick_params(labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)

    best = max(rows, key=lambda row: abs(row["pearson"]))
    cross = float(stats.spearmanr(frame["escott_raw"], frame["escott_prob"]).statistic)
    figure.suptitle(
        f"Log space or linear?   ESCOTT vs {plm_label}\n"
        f"ρ is identical ACROSS each row - logging a probability cannot reorder it - and r is "
        f"not, peaking at {abs(best['pearson']):.3f}.\n"
        f"ρ differs BETWEEN rows only because the x variable changes: the per-column softmax "
        f"reorders ESCOTT against itself (ρ = {cross:.3f}).",
        fontsize=9.8, fontweight="bold", x=0.012, ha="left", y=0.998)
    _distance_legend(axes[0][0], frame, loc="upper left")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path, rows


def fig_models(frames: Dict[str, pd.DataFrame], out_path: Path) -> Path:
    """The same axes across every PLM available for this lineage."""
    labels = list(frames)
    # 4 models tile as 2x2, not 3+1 with a hole in the second row.
    columns = 2 if len(labels) == 4 else min(3, len(labels))
    rows_count = int(np.ceil(len(labels) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(4.0 * columns, 3.9 * rows_count),
                               squeeze=False)
    for axis, label in zip(axes.ravel(), labels):
        frame = frames[label]
        _scatter_by_distance(axis, frame, "escott_raw", "plm_log10", size=2.2, alpha=0.24)
        stat = correlations(frame, "escott_raw", "plm_log10")
        axis.set_title(f"{label}\nρ = {stat['spearman']:.3f}   r = {stat['pearson']:.3f}",
                       fontsize=9.5)
        axis.set_xlabel("ESCOTT score", fontsize=8.5)
        axis.set_ylabel("log₁₀ P(residue)", fontsize=8.5)
        axis.tick_params(labelsize=7.5)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.ravel()[len(labels):]:
        axis.axis("off")
    figure.suptitle("Every protein language model scored for this lineage, on ESCOTT's axes",
                    fontsize=11, fontweight="bold", x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path


def fig_distance_effect(frame: pd.DataFrame, plm_label: str, out_path: Path) -> Path:
    """What nt distance does to each model separately -- the confound, measured."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    order = [1, 2, 3]
    for axis, column, label in (
        (left, "escott_raw", "ESCOTT score"),
        (right, "plm_log10", f"log₁₀ P(residue)  -  {plm_label}"),
    ):
        data = [frame.loc[frame["nt_distance"] == distance, column].to_numpy(dtype=float)
                for distance in order]
        parts = axis.violinplot(data, positions=order, widths=0.78, showextrema=False)
        for body, distance in zip(parts["bodies"], order):
            body.set_facecolor(NT_COLOURS[distance])
            body.set_alpha(0.75)
            body.set_linewidth(0)
        medians = [float(np.median(values)) for values in data]
        axis.scatter(order, medians, s=34, color=SURFACE, edgecolor=INK, zorder=5, linewidth=1.3)
        for distance, median in zip(order, medians):
            axis.annotate(f"{median:.2f}", (distance, median), textcoords="offset points",
                          xytext=(11, -3), fontsize=8, color=INK2)
        axis.set_xticks(order, [NT_LABELS[distance] for distance in order], fontsize=8.5)
        axis.set_ylabel(label, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)

    kruskal_escott = stats.kruskal(*[frame.loc[frame["nt_distance"] == d, "escott_raw"]
                                     for d in order])
    kruskal_plm = stats.kruskal(*[frame.loc[frame["nt_distance"] == d, "plm_log10"]
                                  for d in order])
    def p_text(value: float) -> str:
        # scipy underflows to exactly 0 well before the true p is 0; say so honestly.
        return "p < 1e-300" if value == 0 else f"p = {value:.1e}"

    left.set_title(f"ESCOTT\nKruskal-Wallis H = {kruskal_escott.statistic:.0f}, "
                   f"{p_text(kruskal_escott.pvalue)}", fontsize=9.5)
    right.set_title(f"{plm_label}\nKruskal-Wallis H = {kruskal_plm.statistic:.0f}, "
                    f"{p_text(kruskal_plm.pvalue)}", fontsize=9.5)
    figure.suptitle("Does codon distance itself predict the score? Both models, separately",
                    fontsize=11, fontweight="bold", x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_PLMS: Tuple[Tuple[str, str], ...] = (
    ("ESM-2 650M, no fine-tuning",
     "Results/iav_mutational_accessibility/Lytras_OG/plm_cache/"
     "J_int_ESM2_650M_HA80_raw_plm_probability_profile.csv"),
    ("ESM-2 650M, HA80 fine-tuned",
     "Results/iav_mutational_accessibility/Lytras_OG/plm_cache/"
     "J_int_ESM2_650M_HA80_plm_probability_profile.csv"),
    ("ESM-2 600M, flu fine-tuned (AdamW)",
     "Results/iav_mutational_accessibility/esm2_flu_full_AdamW/plm_cache/"
     "J_int_ESM2_600M_FLU_AdamW_plm_probability_profile.csv"),
    ("ESM-C 600M, flu fine-tuned",
     "Results/iav_mutational_accessibility/esmc_flu_full_final_only/plm_cache/"
     "J_int_ESMC_600M_FLU_final_checkpoint_plm_probability_profile.csv"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lineage", default="J_int")
    parser.add_argument("--lineage-dir", type=Path,
                        default=REPO_ROOT / "Sequences" / "IAV_lineage_files")
    parser.add_argument("--plm", action="append", default=None, metavar="LABEL=PATH",
                        help="repeatable; default is the four J_int caches in Results/")
    parser.add_argument("--primary-plm", default=None,
                        help="label of the PLM used for the headline figures; default the "
                             "last one given")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default <run-dir>/JtoK_report")
    parser.add_argument("--keep-dead-sites", action="store_true",
                        help="do NOT drop constant ESCOTT columns (they sit at exactly 0.0 and "
                             "paint a false 'perfectly tolerated' stripe)")
    return parser


def resolve_plms(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    if args.plm:
        pairs = []
        for entry in args.plm:
            label, _, path = entry.partition("=")
            if not path:
                raise ValueError(f"--plm expects LABEL=PATH, got {entry!r}")
            pairs.append((label, Path(path)))
        return pairs
    return [(label, REPO_ROOT / path) for label, path in DEFAULT_PLMS]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_style()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "JtoK_report").resolve()
    figures_dir = common.ensure_dir(out_dir / "figures")
    tables_dir = common.ensure_dir(out_dir / "tables")

    key = common.safe_label(args.lineage)
    escott_raw = load_raw_matrix(run_dir / "scores" / f"{key}_ESCOTT_raw.tsv")
    escott_prob, protein = load_probability_matrix(
        run_dir / "scores" / f"{key}_ESCOTT_score_matrix.csv")

    reference = common.load_reference_cds(args.lineage_dir / f"{args.lineage}.nt.fa", args.lineage)
    if reference["protein"] != protein:
        raise ValueError("the ESCOTT matrix was not built from this lineage's reference")
    distances = codon_distance_table(reference["nucleotide"], protein)

    live = (sorted(escott_raw.columns) if args.keep_dead_sites
            else list(informative_positions(escott_raw)))
    if not args.keep_dead_sites:
        print(f"@> using {len(live)}/{len(escott_raw.columns)} positions "
              f"({len(escott_raw.columns) - len(live)} constant ESCOTT columns dropped)")

    frames: Dict[str, pd.DataFrame] = {}
    for label, path in resolve_plms(args):
        if not path.exists():
            print(f"@> skipping {label}: {path} not found")
            continue
        plm, plm_protein = load_probability_matrix(path)
        if plm_protein != protein:
            raise ValueError(f"{path} was scored on a different sequence than the ESCOTT matrix")
        frames[label] = build_frame(escott_raw, escott_prob, plm, distances, live)
        print(f"@> loaded {label}: {len(frames[label]):,} substitutions")
    if not frames:
        raise SystemExit("no PLM profiles found")

    primary_label = args.primary_plm or list(frames)[-1]
    if primary_label not in frames:
        raise SystemExit(f"--primary-plm {primary_label!r} not among {list(frames)}")
    primary = frames[primary_label]

    figures = {
        "main": fig_main(primary, primary_label, figures_dir / "fig8_escott_vs_plm.png"),
        "distance": fig_distance_effect(primary, primary_label,
                                        figures_dir / "fig10_codon_distance_effect.png"),
        "models": fig_models(frames, figures_dir / "fig11_escott_vs_plm_models.png"),
    }
    scale_path, scale_rows = fig_scale_grid(primary, primary_label,
                                            figures_dir / "fig9_scale_grid.png")
    figures["scales"] = scale_path

    summary = {
        "primary_plm": primary_label,
        "n_positions_used": len(live),
        # The diagnostic that explains why Spearman differs between the two ROWS of the scale
        # grid while being invariant along each row. Anything below 1.0 here means the
        # per-column softmax has reordered ESCOTT against its own raw score.
        "escott_raw_vs_softmax_spearman": float(
            stats.spearmanr(primary["escott_raw"], primary["escott_prob"]).statistic),
        "scale_treatments": scale_rows,
        "per_model": {
            label: {
                "overall": correlations(frame, "escott_raw", "plm_log10"),
                "by_nt_distance": {
                    str(distance): correlations(frame[frame["nt_distance"] == distance],
                                                "escott_raw", "plm_log10")
                    for distance in (1, 2, 3)
                },
            }
            for label, frame in frames.items()
        },
    }
    (tables_dir / "escott_vs_plm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary.to_csv(tables_dir / "escott_vs_plm_primary.tsv.gz", sep="\t", index=False)

    print()
    print(json.dumps(summary, indent=2))
    for name, path in figures.items():
        print(f"@> {name:10s} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
