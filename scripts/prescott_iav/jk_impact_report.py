#!/usr/bin/env python3
"""Summarise what ESCOTT (run on the base J lineage) predicts for the J -> K substitutions.

This is a *reader*, not a pipeline stage.  It consumes what stages A/B/C already wrote
under ``--run-dir`` -- the ESCOTT raw score matrix for one base lineage, the JET2-surrogate
residue table and the per-lineage reference CDSs -- and turns them into the figures and
tables behind ``REPORT.md``.  It computes no new model scores, so every number it prints is
traceable to a file ESCOTT itself produced.

The question it answers: the J lineage is the *reference* ESCOTT was run on, so ESCOTT has
scored all 19 x L single substitutions away from J without ever being told which of them
actually happened.  The J -> K substitutions are a subset of those cells.  Where do they sit
in the distribution ESCOTT predicted?

Three nulls are evaluated, because "are these mutations tolerated?" is three different
questions:

    site selection   are the 11 POSITIONS unusually unconstrained?
    residue choice   given those positions, is K's particular residue unusually tolerated?
    joint            is the whole set unusual against a codon-reachable background?

Run under the PRESCOTT env (numpy/pandas/matplotlib only -- no torch, no prody).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402


# --------------------------------------------------------------------------- #
# H3 numbering
#
# The pipeline works in HA0 coordinates: 1..566 over the unprocessed precursor,
# which is what the ESCOTT query FASTA and every score matrix column is indexed by.
# Every published H3N2 antigenic result is in MATURE numbering instead: HA1 starts
# after the 16-residue signal peptide and runs to the cleavage-site arginine, then
# HA2 restarts at 1.  Both offsets are verified against the J reference at import
# time (see verify_numbering) rather than trusted.
# --------------------------------------------------------------------------- #

SIGNAL_PEPTIDE_LEN = 16          # HA0 1..16   -> signal peptide
HA1_LAST_HA0 = 345               # HA0 17..345 -> HA1 1..329 (345 is the cleavage R)

# Classical H3 HA1 antigenic sites (Wiley/Skehel epitopes as tabulated by
# Bush et al. 1999, Mol Biol Evol 16:1457).  HA1 numbering.
ANTIGENIC_SITES: Dict[str, Tuple[int, ...]] = {
    "A": (122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144,
          145, 146, 150, 152, 168),
    "B": (128, 129, 155, 156, 157, 158, 159, 160, 163, 164, 165, 186, 187, 188,
          189, 190, 192, 193, 194, 196, 197, 198),
    "C": (44, 45, 46, 47, 48, 50, 51, 53, 54, 273, 275, 276, 278, 279, 280, 294,
          297, 299, 300, 304, 305, 307, 308, 309, 310, 311, 312),
    "D": (96, 102, 103, 117, 121, 167, 170, 171, 172, 173, 174, 175, 176, 177,
          179, 182, 201, 203, 207, 208, 209, 212, 213, 214, 215, 216, 217, 218,
          219, 226, 227, 228, 229, 230, 238, 240, 242, 244, 246, 247, 248),
    "E": (57, 59, 62, 63, 67, 75, 78, 80, 81, 82, 83, 86, 87, 88, 91, 92, 94,
          109, 260, 261, 262, 263, 265),
}

# Receptor-binding-site elements, HA1 numbering: the three structural loops that
# line the sialic-acid pocket plus the conserved base residues.
RBS_ELEMENTS: Dict[str, Tuple[int, ...]] = {
    "130-loop": tuple(range(134, 139)),
    "190-helix": tuple(range(186, 196)),
    "220-loop": tuple(range(221, 229)),
    "base": (98, 153, 183, 194),
}

# The standard genetic code, used only for the codon-reachability null.
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")

# --------------------------------------------------------------------------- #
# Palette (dataviz reference instance, light surface)
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # blue, orange, aqua, yellow
CRITICAL = "#d03b3b"
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "figure.dpi": 160,
    })


# --------------------------------------------------------------------------- #
# Numbering helpers
# --------------------------------------------------------------------------- #

def h3_label(ha0_pos: int) -> str:
    """HA0 position -> mature-numbering label, e.g. 174 -> 'HA1 158'."""
    if ha0_pos <= SIGNAL_PEPTIDE_LEN:
        return f"SP {ha0_pos}"
    if ha0_pos <= HA1_LAST_HA0:
        return f"HA1 {ha0_pos - SIGNAL_PEPTIDE_LEN}"
    return f"HA2 {ha0_pos - HA1_LAST_HA0}"


def ha1_number(ha0_pos: int) -> Optional[int]:
    if SIGNAL_PEPTIDE_LEN < ha0_pos <= HA1_LAST_HA0:
        return ha0_pos - SIGNAL_PEPTIDE_LEN
    return None


def verify_numbering(protein: str) -> None:
    """Fail loudly rather than silently mislabel every figure.

    Two facts are checked against the actual reference: the signal peptide is 16 aa
    (so HA0 17 is HA1 1), and HA0 345 is the cleavage-site arginine (so HA0 346 is
    HA2 1).  Both are properties of H3 HA, but a differently trimmed CDS would break
    them and every antigenic-site call downstream would be shifted.
    """
    if len(protein) < HA1_LAST_HA0 + 10:
        raise ValueError(f"reference protein is only {len(protein)} aa; expected a full HA0")
    if protein[HA1_LAST_HA0 - 1] != "R":
        raise ValueError(
            f"HA0 {HA1_LAST_HA0} is {protein[HA1_LAST_HA0 - 1]!r}, not the cleavage-site R; "
            "the HA1/HA2 offset in this module does not apply to this reference"
        )
    if protein[SIGNAL_PEPTIDE_LEN] != "Q":
        raise ValueError(
            f"HA0 {SIGNAL_PEPTIDE_LEN + 1} is {protein[SIGNAL_PEPTIDE_LEN]!r}, not the Q that "
            "opens mature H3 HA1; the signal-peptide length in this module does not apply"
        )


def antigenic_site_of(ha0_pos: int) -> Optional[str]:
    number = ha1_number(ha0_pos)
    if number is None:
        return None
    for site, members in ANTIGENIC_SITES.items():
        if number in members:
            return site
    return None


def rbs_element_of(ha0_pos: int) -> Optional[str]:
    number = ha1_number(ha0_pos)
    if number is None:
        return None
    for element, members in RBS_ELEMENTS.items():
        if number in members:
            return element
    return None


# --------------------------------------------------------------------------- #
# Codon reachability and glycosylation
# --------------------------------------------------------------------------- #

def codon_neighbours(cds: str, protein: str) -> Dict[int, frozenset]:
    """Per position, the amino acids reachable by a SINGLE nucleotide change.

    This is the honest background for "was this substitution available to the virus?".
    Roughly 5-7 of the 19 alternatives at a site are one mutation away; the rest need
    two or three, so scoring the K residue against all 19 would compare it to changes
    that essentially never happen in one step.
    """
    reachable: Dict[int, frozenset] = {}
    for index, wt in enumerate(protein):
        codon = cds[3 * index: 3 * index + 3].upper()
        if CODON_TABLE.get(codon) != wt:
            raise ValueError(f"codon {codon!r} at position {index + 1} does not encode {wt}")
        options = set()
        for site in range(3):
            for base in "ACGT":
                if base == codon[site]:
                    continue
                mutated = codon[:site] + base + codon[site + 1:]
                aa = CODON_TABLE[mutated]
                if aa not in ("*", wt):
                    options.add(aa)
        reachable[index + 1] = frozenset(options)
    return reachable


def sequons(protein: str) -> Dict[int, str]:
    """N-glycosylation sequons N-X-S/T (X != P), keyed by the HA0 position of the N."""
    found: Dict[int, str] = {}
    for index in range(len(protein) - 2):
        first, middle, third = protein[index], protein[index + 1], protein[index + 2]
        if first == "N" and middle != "P" and third in ("S", "T"):
            found[index + 1] = f"{first}{middle}{third}"
    return found


# --------------------------------------------------------------------------- #
# Loading what the pipeline wrote
# --------------------------------------------------------------------------- #

def load_raw_matrix(path: Path) -> pd.DataFrame:
    """``scores/<key>_ESCOTT_raw.tsv``: 20 x L of -normPred. Negative = deleterious."""
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.columns = [int(column) for column in frame.columns]
    frame.index = [str(value).upper() for value in frame.index]
    return frame.loc[list(AA20)].astype(float)


def load_probability_matrix(path: Path) -> Tuple[pd.DataFrame, str]:
    """``scores/<key>_<variant>_score_matrix.csv``: sequence row + 20 probability rows."""
    frame = pd.read_csv(path, index_col=0, header=None)
    protein = "".join(str(value) for value in frame.loc["sequence"].tolist())
    probabilities = frame.drop(index="sequence").astype(float)
    probabilities.index = [str(value).upper() for value in probabilities.index]
    probabilities.columns = range(1, probabilities.shape[1] + 1)
    return probabilities.loc[list(AA20)], protein


def load_jet(path: Path) -> pd.DataFrame:
    """Stage B's ``<key>_surrogate_jet.res``: AA pos chain pc tr freq trace cv."""
    frame = pd.read_csv(path, sep=r"\s+")
    return frame.set_index("pos")


# --------------------------------------------------------------------------- #
# The substitution table
# --------------------------------------------------------------------------- #

def clade_steps(proteins: Dict[str, str], path: Sequence[str]) -> Dict[Tuple[int, str], str]:
    """Which edge of the clade path introduced each residue state.

    A substitution is attributed to the LAST edge that changed the residue, so a site
    that flips and flips back is credited to the flip that survives into K.
    """
    attribution: Dict[Tuple[int, str], str] = {}
    for parent, child in zip(path, path[1:]):
        edge = f"{parent}->{child}"
        for index, (before, after) in enumerate(zip(proteins[parent], proteins[child])):
            if before != after:
                attribution[(index + 1, after)] = edge
    return attribution


def build_substitution_table(
    base_protein: str,
    target_protein: str,
    raw: pd.DataFrame,
    probabilities: pd.DataFrame,
    jet: pd.DataFrame,
    reachable: Dict[int, frozenset],
    attribution: Dict[Tuple[int, str], str],
    base_sequons: Dict[int, str],
    target_sequons: Dict[int, str],
) -> pd.DataFrame:
    finite = raw.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]

    rows: List[Dict[str, object]] = []
    for index, (wt, mut) in enumerate(zip(base_protein, target_protein)):
        if wt == mut:
            continue
        position = index + 1
        column = raw[position]
        score = float(column[mut])
        alternatives = column.drop(index=wt).astype(float)
        # rank 1 = the most tolerated of the 19 alternatives ESCOTT scored at this site
        rank = int((alternatives > score).sum()) + 1
        accessible_set = reachable[position]
        accessible_scores = alternatives.reindex(sorted(accessible_set)).dropna()
        rows.append({
            "ha0_pos": position,
            "wt": wt,
            "mut": mut,
            "mutation": f"{wt}{position}{mut}",
            "h3_label": h3_label(position),
            # HA1 substitutions get the bare mature number every H3N2 paper uses; anything
            # outside HA1 keeps its chain prefix so it can never be mistaken for one.
            "h3_mutation": (f"{wt}{ha1_number(position)}{mut}" if ha1_number(position) is not None
                            else f"{wt}({h3_label(position).replace(' ', ':')}){mut}"),
            "clade_step": attribution.get((position, mut), "unattributed"),
            "escott_score": score,
            "escott_probability": float(probabilities.at[mut, position]),
            "site_mean_score": float(alternatives.mean()),
            "site_best_score": float(alternatives.max()),
            "site_worst_score": float(alternatives.min()),
            "rank_in_site": rank,
            "rank_in_site_accessible": int((accessible_scores > score).sum()) + 1,
            "n_accessible": len(accessible_set),
            "codon_accessible": mut in accessible_set,
            # A position whose 20 scores are all equal carries no rank information: either
            # trace == 0 (pred.R:487 zeroes the column) or the site is fully conserved and
            # every substitution gets the same non-zero value.  Such a cell scores 0.0,
            # which is the MAXIMUM of the scale, so counting it as "perfectly tolerated"
            # silently inflates every set-level statistic.  It is excluded from the tests
            # and reported separately instead.
            "informative": bool(np.ptp(column.dropna().to_numpy(dtype=float)) > 1e-12),
            "percentile_all_substitutions": float((finite < score).mean() * 100.0),
            "trace": float(jet.at[position, "trace"]),
            "cv": float(jet.at[position, "cv"]),
            "pc": float(jet.at[position, "pc"]),
            "msa_freq_of_wt": float(jet.at[position, "freq"]),
            "antigenic_site": antigenic_site_of(position) or "",
            "rbs_element": rbs_element_of(position) or "",
            "sequon_change": sequon_change(position, base_sequons, target_sequons),
        })
    return pd.DataFrame(rows).sort_values("ha0_pos").reset_index(drop=True)


def sequon_change(position: int, before: Dict[int, str], after: Dict[int, str]) -> str:
    """Whether this substitution creates or destroys an N-glycosylation sequon.

    A substitution at position p can make or break a sequon whose N sits at p, p-1 or
    p-2, so all three are inspected rather than only the site itself.
    """
    notes = []
    for anchor in (position - 2, position - 1, position):
        if anchor < 1:
            continue
        was, now = anchor in before, anchor in after
        if was and not now:
            notes.append(f"lost N{anchor} ({before[anchor]})")
        elif now and not was:
            notes.append(f"gained N{anchor} ({after[anchor]})")
    return "; ".join(notes)


# --------------------------------------------------------------------------- #
# Nulls
# --------------------------------------------------------------------------- #

def informative_positions(raw: pd.DataFrame) -> np.ndarray:
    """Positions whose ESCOTT column is not constant.

    A constant column -- trace == 0, or full conservation -- softmaxes to a uniform 1/20
    and carries no rank information.  Both the observed set and every null are restricted
    to these, so a dead site can never be counted as a tolerated substitution.
    """
    return np.array([
        position for position in sorted(raw.columns)
        if np.ptp(raw[position].dropna().to_numpy(dtype=float)) > 1e-12
    ])


def accessible_score_pool(raw: pd.DataFrame, protein: str, reachable: Dict[int, frozenset],
                          keep: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Every codon-reachable single substitution, with its ESCOTT score."""
    allowed = None if keep is None else set(int(position) for position in keep)
    records: List[Dict[str, object]] = []
    for index, wt in enumerate(protein):
        position = index + 1
        if allowed is not None and position not in allowed:
            continue
        for mut in sorted(reachable[position]):
            records.append({
                "ha0_pos": position,
                "wt": wt,
                "mut": mut,
                "escott_score": float(raw.at[mut, position]),
            })
    return pd.DataFrame(records)


def _summarise(observed: float, means: np.ndarray, n_draws: int) -> Dict[str, float]:
    """Empirical p-values for one null, from the null distribution of the MEAN.

    Both tails are reported, because a mutation set can be surprising in either
    direction and only the one-sided number is usually quoted.  The +1/+1 is the
    standard finite-sampling correction: a permutation p can never be exactly 0.
    """
    spread = float(means.std(ddof=1))
    p_greater = float(((means >= observed).sum() + 1) / (n_draws + 1))
    p_less = float(((means <= observed).sum() + 1) / (n_draws + 1))
    return {
        "observed_mean": float(observed),
        "null_mean": float(means.mean()),
        "null_sd": spread,
        "p_more_tolerated": p_greater,
        "p_less_tolerated": p_less,
        "p_two_sided": float(min(1.0, 2 * min(p_greater, p_less))),
        "z": float((observed - means.mean()) / spread) if spread else float("nan"),
        "n_draws": int(n_draws),
    }


def permutation_test(observed: float, pool: np.ndarray, size: int,
                     n_draws: int, seed: int) -> Tuple[Dict[str, float], np.ndarray]:
    """Null: `size` values drawn at random from `pool`, `n_draws` times.

    Drawn WITH replacement. The pool is three orders of magnitude larger than `size`,
    so the difference from a without-replacement draw is far below the resolution of
    the p-value, and replacement keeps every draw independent.
    """
    rng = np.random.default_rng(seed)
    means = rng.choice(pool, size=(n_draws, size), replace=True).mean(axis=1)
    return _summarise(observed, means, n_draws), means


def site_choice_test(table: pd.DataFrame, raw: pd.DataFrame, reachable: Dict[int, frozenset],
                     n_draws: int, seed: int) -> Tuple[Dict[str, float], np.ndarray]:
    """Given the observed positions, was K's residue choice unusually tolerated?

    One codon-reachable alternative is drawn per observed position, so site constraint
    is held fixed and only the residue identity varies.  This is the test that
    separates "K mutated tolerant sites" from "K picked the tolerant residue".
    """
    rng = np.random.default_rng(seed)
    pools = [
        np.array([raw.at[aa, int(row["ha0_pos"])] for aa in sorted(reachable[int(row["ha0_pos"])])],
                 dtype=float)
        for _, row in table.iterrows()
    ]
    # One independent column of draws per site, then average across sites.
    columns = np.stack([rng.choice(pool, size=n_draws, replace=True) for pool in pools], axis=1)
    means = columns.mean(axis=1)
    return _summarise(float(table["escott_score"].mean()), means, n_draws), means


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def fig_landscape_position(table: pd.DataFrame, pool: pd.DataFrame, out_path: Path) -> Path:
    """Where the observed substitutions sit against everything ESCOTT scored."""
    figure, axes = plt.subplots(figsize=(7.4, 4.2))
    scores = pool["escott_score"].to_numpy(dtype=float)
    axes.hist(scores, bins=70, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.4,
              label=f"all {len(scores):,} codon-reachable substitutions from J")

    tested = table[table["informative"]]
    dropped = table[~table["informative"]]
    ymax = axes.get_ylim()[1]
    for _, row in tested.iterrows():
        axes.plot([row["escott_score"]], [ymax * 0.045], marker="v", markersize=7,
                  color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
    axes.plot([], [], marker="v", linestyle="none", color=SERIES[1], markersize=7,
              label=f"the {len(tested)} scored J -> K substitutions")
    for _, row in dropped.iterrows():
        axes.plot([row["escott_score"]], [ymax * 0.045], marker="v", markersize=7,
                  markerfacecolor="none", color=MUTED, markeredgewidth=1.4, zorder=5)
    if len(dropped):
        axes.plot([], [], marker="v", linestyle="none", markerfacecolor="none", color=MUTED,
                  markersize=7, markeredgewidth=1.4,
                  label=f"{len(dropped)} on a dead column (no information)")

    axes.axvline(tested["escott_score"].mean(), color=SERIES[1], linewidth=2, linestyle="--",
                 zorder=4, label=f"J -> K mean = {tested['escott_score'].mean():.2f}")
    axes.axvline(scores.mean(), color=INK2, linewidth=2, linestyle=":", zorder=4,
                 label=f"background mean = {scores.mean():.2f}")

    axes.set_xlabel("ESCOTT score  (0 = tolerated, more negative = more deleterious)")
    axes.set_ylabel("number of substitutions")
    axes.set_title("Every substitution ESCOTT scored, and the eleven that happened")
    axes.legend(loc="upper left", fontsize=8)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def fig_per_mutation(table: pd.DataFrame, out_path: Path) -> Path:
    """Per-substitution score and within-site rank, coloured by clade step."""
    steps = list(dict.fromkeys(table["clade_step"]))
    colour_of = {step: SERIES[index % len(SERIES)] for index, step in enumerate(steps)}

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.5, 1.0]}, sharey=True)
    order = table.sort_values("ha0_pos", ascending=False).reset_index(drop=True)
    y = np.arange(len(order))
    labels = [f"{row.h3_mutation}  ({row.antigenic_site or '-'})" for row in order.itertuples()]

    for index, row in order.iterrows():
        colour = colour_of[row["clade_step"]]
        dead = not row["informative"]
        left.plot([0, row["escott_score"]], [index, index], color=colour, linewidth=2, zorder=2)
        left.plot([row["escott_score"]], [index], marker="o", markersize=8,
                  markerfacecolor="none" if dead else colour, color=colour,
                  markeredgecolor=MUTED if dead else SURFACE, markeredgewidth=1.4, zorder=3)
        left.plot([row["site_worst_score"], row["site_best_score"]], [index, index],
                  color=GRID, linewidth=6, zorder=1, solid_capstyle="round")
        if dead:
            left.annotate("dead column - no information", (0, index),
                          textcoords="offset points", xytext=(-6, 0), ha="right", va="center",
                          fontsize=7.2, color=MUTED, style="italic")

    for step in steps:
        left.plot([], [], marker="o", linestyle="-", color=colour_of[step], label=step)
    left.plot([], [], marker="o", linestyle="none", markerfacecolor="none",
              markeredgecolor=MUTED, markersize=8, label="uninformative site (excluded)")
    left.axvline(0, color=AXIS, linewidth=1)
    left.set_yticks(y, labels)
    left.set_xlabel("ESCOTT score")
    left.set_title("Predicted effect of each J -> K substitution")
    left.legend(loc="lower left", fontsize=8, title="introduced on edge", title_fontsize=8)
    left.spines[["top", "right"]].set_visible(False)

    for index, row in order.iterrows():
        colour = colour_of[row["clade_step"]]
        right.barh(index, row["rank_in_site_accessible"], height=0.55, color=colour,
                   edgecolor=SURFACE, linewidth=1.0)
        right.text(row["rank_in_site_accessible"] + 0.12, index,
                   f"{row['rank_in_site_accessible']}/{row['n_accessible']}",
                   va="center", fontsize=8, color=INK2)
    right.set_xlabel("rank among codon-reachable alternatives\n(1 = ESCOTT's top choice at that site)")
    right.set_title("Did K pick the model's preferred residue?")
    right.set_xlim(0, max(table["n_accessible"]) + 1.6)
    right.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def fig_site_heatmap(table: pd.DataFrame, raw: pd.DataFrame, base_protein: str,
                     reachable: Dict[int, frozenset], out_path: Path) -> Path:
    """The full 20-residue ESCOTT column at each mutated site."""
    positions = table["ha0_pos"].tolist()
    block = raw[positions].reindex(list(AA20))
    values = block.to_numpy(dtype=float)

    figure, axes = plt.subplots(figsize=(9.4, 6.6))
    vmin = float(np.nanmin(values))
    image = axes.imshow(values, aspect="auto", cmap=SEQ_CMAP.reversed(),
                        norm=Normalize(vmin=vmin, vmax=0.0), interpolation="nearest")

    axes.set_xticks(range(len(positions)),
                    [f"{row.wt}{row.ha0_pos}\n{row.h3_label}" for row in table.itertuples()],
                    fontsize=7.5, rotation=45, ha="right")
    axes.set_yticks(range(len(AA20)), list(AA20), fontsize=8)
    axes.set_ylabel("substituted residue")
    axes.grid(False)

    for column, (_, row) in enumerate(table.iterrows()):
        wt_row = AA20.index(row["wt"])
        mut_row = AA20.index(row["mut"])
        axes.add_patch(Rectangle((column - 0.5, wt_row - 0.5), 1, 1, fill=False,
                                 edgecolor=MUTED, linewidth=1.4, linestyle=":"))
        axes.add_patch(Rectangle((column - 0.5, mut_row - 0.5), 1, 1, fill=False,
                                 edgecolor=SERIES[1], linewidth=2.4))
        for aa_row, aa in enumerate(AA20):
            if aa not in reachable[int(row["ha0_pos"])] and aa != row["wt"]:
                axes.plot([column], [aa_row], marker=".", markersize=2.0, color=SURFACE, alpha=0.9)

    axes.plot([], [], marker="s", markerfacecolor="none", markeredgecolor=SERIES[1],
              markeredgewidth=2, linestyle="none", markersize=9, label="residue K acquired")
    axes.plot([], [], marker="s", markerfacecolor="none", markeredgecolor=MUTED,
              markeredgewidth=1.4, linestyle="none", markersize=9, label="residue J had")
    axes.plot([], [], marker=".", color=MUTED, linestyle="none",
              label="needs >1 nucleotide change")
    axes.legend(loc="upper left", bbox_to_anchor=(1.16, 1.0), fontsize=8)
    axes.set_title("ESCOTT's full verdict at each of the eleven sites")

    bar = figure.colorbar(image, ax=axes, pad=0.02, fraction=0.045)
    bar.set_label("ESCOTT score", color=INK2)
    bar.outline.set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def fig_constraint_profile(table: pd.DataFrame, raw: pd.DataFrame, jet: pd.DataFrame,
                           out_path: Path) -> Path:
    """Site-level constraint along HA, with the mutated positions marked."""
    positions = np.array(sorted(raw.columns))
    site_mean = np.array([np.nanmean(raw[position].to_numpy(dtype=float)) for position in positions])
    window = 9
    smooth = pd.Series(site_mean).rolling(window, center=True, min_periods=1).mean().to_numpy()

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.2, 5.6), sharex=True,
                                         gridspec_kw={"height_ratios": [1.0, 0.55]})

    for members in ANTIGENIC_SITES.values():
        for member in members:
            top.axvspan(member + SIGNAL_PEPTIDE_LEN - 0.5, member + SIGNAL_PEPTIDE_LEN + 0.5,
                        color=SERIES[3], alpha=0.18, linewidth=0)
    antigenic_proxy = Patch(facecolor=SERIES[3], alpha=0.18, edgecolor="none",
                            label="classical antigenic sites A-E")

    top.plot(positions, site_mean, color=GRID, linewidth=0.8, zorder=1)
    top.plot(positions, smooth, color=SERIES[0], linewidth=2, zorder=2,
             label=f"mean ESCOTT score per site ({window}-residue rolling mean)")
    top.scatter(table["ha0_pos"], table["site_mean_score"], s=52, color=SERIES[1],
                edgecolor=SURFACE, linewidth=1.2, zorder=4, label="J -> K positions")
    # Labels are staggered on two rows: the head positions are close enough together that a
    # single row of annotations collides.
    for offset_index, (_, row) in enumerate(table.iterrows()):
        top.annotate(row["h3_mutation"], (row["ha0_pos"], row["site_mean_score"]),
                     textcoords="offset points", xytext=(0, 10 if offset_index % 2 else 20),
                     ha="center", fontsize=7.2, color=INK2)
    top.axvline(HA1_LAST_HA0 + 0.5, color=INK2, linewidth=1.2, linestyle="--")
    top.text(HA1_LAST_HA0 + 6, top.get_ylim()[0], " HA2", fontsize=8, color=INK2, va="bottom")
    top.set_ylabel("mean ESCOTT score\nat that site")
    top.set_title("Constraint along the HA precursor, with the J -> K positions marked")
    handles, labels = top.get_legend_handles_labels()
    top.legend(handles + [antigenic_proxy], labels + [antigenic_proxy.get_label()],
               loc="lower left", fontsize=8, ncol=3)
    top.spines[["top", "right"]].set_visible(False)

    bottom.fill_between(jet.index, jet["trace"], color=SERIES[2], alpha=0.55, linewidth=0)
    bottom.scatter(table["ha0_pos"], table["trace"], s=42, color=SERIES[1],
                   edgecolor=SURFACE, linewidth=1.1, zorder=4)
    bottom.set_ylabel("JET2-surrogate\nweight (trace)")
    bottom.set_xlabel("HA0 position  (HA1 = HA0 - 16;  HA2 = HA0 - 345)")
    bottom.axvline(HA1_LAST_HA0 + 0.5, color=INK2, linewidth=1.2, linestyle="--")
    bottom.spines[["top", "right"]].set_visible(False)
    bottom.set_xlim(1, positions.max())

    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def fig_null_tests(n_mutations: int, tests: Dict[str, Tuple[Dict[str, float], np.ndarray]],
                   out_path: Path) -> Path:
    """The three null models, each as its own panel.

    The histograms are the SAME draws the reported p-values were computed from -- they
    are handed in rather than resampled here, so a figure can never disagree with the
    number printed beside it.
    """
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    panels = (
        ("joint", f"Joint test\n{n_mutations} reachable substitutions,\nanywhere in HA",
         "p(more tolerated)"),
        ("residue_choice", f"Residue-choice test\nthe same {n_mutations} sites,\na random reachable residue",
         "p(more tolerated)"),
        ("site_selection", f"Site-selection test\n{n_mutations} random sites in HA",
         "p(less constrained)"),
    )
    for axis, (name, title, p_label) in zip(axes, panels):
        stats, draws = tests[name]
        _null_panel(axis, draws, stats["observed_mean"], title,
                    f"{p_label} {_format_p(stats['p_more_tolerated'], stats['n_draws'])}"
                    f"    z = {stats['z']:+.2f}")

    figure.suptitle(
        f"Is the J -> K substitution set unusual? Three nulls, "
        f"{tests['joint'][0]['n_draws']:,} draws each",
        fontsize=11, fontweight="bold", x=0.012, ha="left")
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def _format_p(p_value: float, n_draws: int) -> str:
    """A permutation p is bounded below by 1/(n+1); never print it as '0.0000'."""
    floor = 1.0 / (n_draws + 1)
    if p_value <= floor * 1.5:
        return f"< {floor:.0e}"
    return f"= {p_value:.4f}"


def _null_panel(axes, draws: np.ndarray, observed: float, title: str, annotation: str) -> None:
    axes.hist(draws, bins=60, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=0.3)
    axes.axvline(observed, color=SERIES[1], linewidth=2.4)
    axes.annotate(f"observed\n{observed:.2f}", (observed, axes.get_ylim()[1] * 0.82),
                  textcoords="offset points", xytext=(7, 0), fontsize=8.5, color=SERIES[1],
                  fontweight="bold")
    axes.set_title(title, fontsize=9.5, fontweight="bold")
    axes.set_xlabel("mean ESCOTT score")
    axes.set_ylabel("draws")
    axes.text(0.0, -0.40, annotation, transform=axes.transAxes, fontsize=9, color=INK2)
    axes.spines[["top", "right"]].set_visible(False)


def fig_structure_sensitivity(table: pd.DataFrame, alternate_raw: pd.DataFrame,
                              alternate_label: str, out_path: Path) -> Path:
    """Does the verdict on each substitution survive swapping the structure?

    Same alignment, same query, same GEMME -- only the structure behind the JET2-surrogate
    weight differs.  A conclusion that moves across this panel is a conclusion about the
    structure file, not about evolution.
    """
    alternate = np.array([alternate_raw.at[row["mut"], int(row["ha0_pos"])]
                          for _, row in table.iterrows()], dtype=float)
    primary = table["escott_score"].to_numpy(dtype=float)

    figure, axes = plt.subplots(figsize=(5.6, 5.2))
    limits = [min(primary.min(), alternate.min()) - 0.3, 0.35]
    axes.plot(limits, limits, color=AXIS, linewidth=1.2, linestyle="--", zorder=1)
    axes.scatter(primary, alternate, s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=1.3,
                 zorder=3)
    for (_, row), y in zip(table.iterrows(), alternate):
        axes.annotate(row["h3_mutation"], (row["escott_score"], y), textcoords="offset points",
                      xytext=(7, -2), fontsize=7.6, color=INK2)
    correlation = float(np.corrcoef(primary, alternate)[0, 1])
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_xlabel("ESCOTT score, 6WXB crystal structure (485/566 covered)")
    axes.set_ylabel(f"ESCOTT score, {alternate_label}")
    axes.set_title(f"Structural sensitivity: Pearson r = {correlation:.3f}")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def fig_escott_vs_prescott(comparison: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """How much the population term moves each substitution, and why it mostly does not.

    Plotted as the SHIFT rather than as two overlaid point series: for most of these
    substitutions PRESCOTT and ESCOTT agree to the pixel, and two markers on top of each
    other reads as one model, not as agreement.  The shift, with the reason annotated
    beside it, says the real thing -- the parent panel has no evidence to offer.
    """
    if comparison is None or comparison.empty:
        return None
    order = comparison.sort_values("ha0_pos").reset_index(drop=True)
    shift = order["prescott_score"] - order["escott_score"]

    figure, axes = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(order))
    colours = [SERIES[1] if abs(value) > 1e-6 else GRID for value in shift]
    axes.barh(y, shift, height=0.6, color=colours, edgecolor=SURFACE, linewidth=1.0)
    axes.axvline(0, color=AXIS, linewidth=1.2)
    axes.set_yticks(y, order["h3_mutation"])

    span = max(float(np.abs(shift).max()), 0.02)
    for index, row in order.iterrows():
        moved = abs(shift[index]) > 1e-6
        note = (f"in G.1 panel at {row['parent_frequency'] * 100:.2f}%"
                if row["parent_frequency"] > 0 else
                ("column renormalised by another residue at this site"
                 if moved else "not seen in the G.1 parent panel"))
        # Annotations sit opposite the bar; rows with no bar all take the right-hand side
        # so the "nothing happened" cases line up as one block.
        on_left = moved and shift[index] > 0
        axes.annotate(note, (0, index), textcoords="offset points",
                      xytext=(-8 if on_left else 8, 0),
                      ha="right" if on_left else "left", va="center",
                      fontsize=7.6, color=INK2)
    axes.set_xlim(-span * 1.35, span * 1.35)
    axes.set_xlabel("shift in log-probability when the population term is added\n"
                    "(negative = PRESCOTT judges it less likely than ESCOTT did)")
    axes.set_title("What PRESCOTT's population term adds - and where it has nothing to say")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the output root stages A/B/C wrote into")
    parser.add_argument("--base-lineage", default="J_int",
                        help="the lineage ESCOTT was actually run on")
    parser.add_argument("--target-lineage", default="K")
    parser.add_argument("--clade-path", default="J_int,J.2_int,J.2.4,K",
                        help="comma-separated path used to attribute each substitution to an edge")
    parser.add_argument("--lineage-dir", type=Path,
                        default=REPO_ROOT / "Sequences" / "IAV_lineage_files")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default <run-dir>/JtoK_report")
    parser.add_argument("--sensitivity-dir", type=Path, default=None,
                        help="a second run tree scored against a different structure; adds the "
                             "structural sensitivity panel")
    parser.add_argument("--sensitivity-label", default="contemporary J.2.4.1 model (566/566)")
    parser.add_argument("--prescott-variant", default=None,
                        help="score-matrix variant to compare against, e.g. PRESCOTT_eq2_c1.0_k1; "
                             "default picks the first non-ESCOTT variant in score_variants.tsv")
    parser.add_argument("--n-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_style()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "JtoK_report").resolve()
    figures_dir = common.ensure_dir(out_dir / "figures")
    tables_dir = common.ensure_dir(out_dir / "tables")

    base_key = common.safe_label(args.base_lineage)
    scores_dir = run_dir / "scores"
    raw = load_raw_matrix(scores_dir / f"{base_key}_ESCOTT_raw.tsv")
    probabilities, matrix_protein = load_probability_matrix(
        scores_dir / f"{base_key}_ESCOTT_score_matrix.csv")
    jet = load_jet(run_dir / "inputs" / "jet" / f"{base_key}_surrogate_jet.res")

    clade_path = [name.strip() for name in args.clade_path.split(",") if name.strip()]
    references = {
        name: common.load_reference_cds(args.lineage_dir / f"{name}.nt.fa", name)
        for name in dict.fromkeys(clade_path + [args.base_lineage, args.target_lineage])
    }
    proteins = {name: entry["protein"] for name, entry in references.items()}
    base_protein = proteins[args.base_lineage]
    target_protein = proteins[args.target_lineage]

    verify_numbering(base_protein)
    if base_protein != matrix_protein:
        raise ValueError("the score matrix was not built from this base lineage's reference")

    reachable = codon_neighbours(references[args.base_lineage]["nucleotide"], base_protein)
    attribution = clade_steps(proteins, clade_path)
    table = build_substitution_table(
        base_protein, target_protein, raw, probabilities, jet, reachable, attribution,
        sequons(base_protein), sequons(target_protein))

    # Dead columns are excluded from BOTH the observed set and every null, so the
    # comparison is like-for-like.  The excluded substitutions are still reported.
    live = informative_positions(raw)
    tested = table[table["informative"]].reset_index(drop=True)
    dropped = table[~table["informative"]]
    if len(dropped):
        print(f"@> excluded {len(dropped)}/{len(table)} substitution(s) on constant "
              f"(uninformative) ESCOTT columns: {', '.join(dropped['h3_mutation'])}")

    pool = accessible_score_pool(raw, base_protein, reachable, keep=live)

    # ---- the three nulls -------------------------------------------------------------
    live_site_means = np.array([np.nanmean(raw[position].to_numpy(dtype=float))
                                for position in live])
    tests = {
        "joint": permutation_test(float(tested["escott_score"].mean()),
                                  pool["escott_score"].to_numpy(dtype=float),
                                  len(tested), args.n_draws, args.seed),
        "residue_choice": site_choice_test(tested, raw, reachable, args.n_draws, args.seed + 1),
        "site_selection": permutation_test(float(tested["site_mean_score"].mean()),
                                           live_site_means, len(tested), args.n_draws,
                                           args.seed + 2),
    }
    stats = {name: result[0] for name, result in tests.items()}
    stats["coverage"] = {
        "n_substitutions": int(len(table)),
        "n_tested": int(len(tested)),
        "n_excluded_uninformative": int(len(dropped)),
        "excluded": list(dropped["h3_mutation"]),
        "n_positions_total": int(len(raw.columns)),
        "n_positions_informative": int(len(live)),
    }

    # ---- optional PRESCOTT comparison ------------------------------------------------
    comparison = load_prescott_comparison(scores_dir, base_key, table, args.prescott_variant)

    # ---- write ------------------------------------------------------------------------
    table.to_csv(tables_dir / "jk_substitutions.tsv", sep="\t", index=False)
    pool.to_csv(tables_dir / "codon_reachable_background.tsv.gz", sep="\t", index=False)
    (tables_dir / "null_tests.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    figures = {
        "landscape": fig_landscape_position(table, pool, figures_dir / "fig1_landscape.png"),
        "per_mutation": fig_per_mutation(table, figures_dir / "fig2_per_mutation.png"),
        "heatmap": fig_site_heatmap(table, raw, base_protein, reachable,
                                    figures_dir / "fig3_site_heatmap.png"),
        "profile": fig_constraint_profile(table, raw, jet, figures_dir / "fig4_constraint_profile.png"),
        "nulls": fig_null_tests(len(tested), tests, figures_dir / "fig5_null_tests.png"),
    }
    prescott_figure = fig_escott_vs_prescott(comparison, figures_dir / "fig6_escott_vs_prescott.png")
    if prescott_figure is not None:
        figures["prescott"] = prescott_figure

    if args.sensitivity_dir is not None:
        alternate_raw = load_raw_matrix(
            args.sensitivity_dir.resolve() / "scores" / f"{base_key}_ESCOTT_raw.tsv")
        alternate = np.array([alternate_raw.at[row["mut"], int(row["ha0_pos"])]
                              for _, row in table.iterrows()], dtype=float)
        table["escott_score_alt_structure"] = alternate
        table.to_csv(tables_dir / "jk_substitutions.tsv", sep="\t", index=False)
        stats["structure_sensitivity"] = {
            "alternate_label": args.sensitivity_label,
            "pearson_r": float(np.corrcoef(table["escott_score"], alternate)[0, 1]),
            "spearman_r": float(pd.Series(table["escott_score"]).corr(pd.Series(alternate),
                                                                     method="spearman")),
            "mean_primary": float(table["escott_score"].mean()),
            "mean_alternate": float(alternate.mean()),
            "max_abs_shift": float(np.abs(table["escott_score"] - alternate).max()),
        }
        (tables_dir / "null_tests.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        figures["sensitivity"] = fig_structure_sensitivity(
            table, alternate_raw, args.sensitivity_label,
            figures_dir / "fig7_structure_sensitivity.png")
    if comparison is not None and not comparison.empty:
        comparison.to_csv(tables_dir / "escott_vs_prescott.tsv", sep="\t", index=False)

    print(table.to_string(index=False))
    print()
    print(json.dumps(stats, indent=2))
    print()
    for name, path in figures.items():
        print(f"@> {name:14s} -> {path}")
    return 0


def read_parent_frequencies(path: Path) -> Dict[Tuple[int, str], float]:
    """The parent-panel frequency file PRESCOTT was handed: '<WT><pos><MUT> <freq>' per line."""
    frequencies: Dict[Tuple[int, str], float] = {}
    if not path.exists():
        return frequencies
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", fields[0])
        if match:
            frequencies[(int(match.group(2)), match.group(3))] = float(fields[1])
    return frequencies


def load_prescott_comparison(scores_dir: Path, base_key: str, table: pd.DataFrame,
                             requested: Optional[str]) -> Optional[pd.DataFrame]:
    """Pair each observed substitution's ESCOTT score with one PRESCOTT variant's."""
    variants_path = scores_dir / "score_variants.tsv"
    if not variants_path.exists():
        return None
    variants = pd.read_csv(variants_path, sep="\t")
    variants = variants[(variants["lineage_key"] == base_key) & (variants["variant"] != "ESCOTT")]
    if variants.empty:
        return None
    if requested:
        variants = variants[variants["variant"] == requested]
        if variants.empty:
            raise ValueError(f"--prescott-variant {requested} not present in {variants_path}")
    row = variants.iloc[0]
    probabilities, _ = load_probability_matrix(Path(row["score_matrix_path"]))
    escott_probabilities, _ = load_probability_matrix(
        scores_dir / f"{base_key}_ESCOTT_score_matrix.csv")
    frequencies = read_parent_frequencies(Path(str(row.get("frequency_path") or "")))

    records = []
    for _, entry in table.iterrows():
        position, mut = int(entry["ha0_pos"]), str(entry["mut"])
        records.append({
            "ha0_pos": position,
            "h3_mutation": entry["h3_mutation"],
            "variant": row["variant"],
            # log-probability, so the two live on a comparable additive scale
            "escott_score": float(np.log(escott_probabilities.at[mut, position])),
            "prescott_score": float(np.log(probabilities.at[mut, position])),
            "parent_frequency": float(frequencies.get((position, mut), 0.0)),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    raise SystemExit(main())
