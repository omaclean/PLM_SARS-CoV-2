"""Immune escape against the *standing immunity* at a date, not against one strain.

Why this exists
---------------
``plant_escape_*`` measures every mutation as a displacement from one stationary
point — the start lineage. That answers "how far does this mutation move the
virus", but not "how much of the population's immunity does it actually escape".
Those differ whenever the start lineage is not where the immunity is: a mutation
that moves 0.5 units directly away from a dense recent cloud is worth far more
than one that moves 1.0 units further out from strains the population already has
no cross-immunity to.

This script scores each genotype against the PLANT background cloud
(``backgrounds.csv``: ~150k HA sequences, 1968 → 2024, in the same 3D antigenic
space as the scan's genotype embeddings) restricted to sequences collected
**before a given date**, weighted two ways.

The two weightings
------------------
**Recency (whose immunity is still around).** Each prior sequence is discounted
by ``0.5 ** (age_years / half_life)``. The default half-life of 1 year is a
strong recency bias by design: immunity raised by strains two seasons ago counts
a quarter as much as this season's.

Sequence counts also track surveillance effort far more than they track
prevalence — there are ~200 sequences from 1968 and tens of thousands from 2022.
``--normalise-by year`` (the default) therefore divides by the number of kept
sequences in each calendar year first, so a year of circulation contributes a
weight set by *when* it was, not by how hard anyone was sequencing. Pass
``--normalise-by none`` to weight every deposited sequence equally instead.

**Antigenic distance, with saturation (the strong local effect).**
Cross-immunity from a past strain at distance ``d`` decays as
``sigma(d) = exp(-d / scale)``, so escape from that strain is ``1 - sigma(d)``.
Because ``sigma`` is steepest at ``d = 0``, moving away from something *close*
buys far more escape than moving the same distance away from something already
far — which is the whole point:

    scale = 2.0 antigenic units
    0.5 units gained starting 1.0 away   ->  escape 0.393 -> 0.528   (+0.135)
    1.0 units gained starting 10.0 away  ->  escape 0.993 -> 0.996   (+0.004)

a 34x difference in value for twice the distance. ``--kernel sigmoid`` swaps in a
plateau of near-full protection at short range (more realistic for HI titres,
but it flattens exactly the local gradient this analysis is about);
``--kernel linear`` gives the hard cutoff at ``scale``.

The score
---------
For genotype ``g`` at date ``t``::

    escape(g, t) = sum_s w(s, t) * (1 - sigma(d(g, s)))

with ``w`` normalised to sum to 1 over the prior sequences. So escape is a
fraction in [0, 1]: *the share of the recency-weighted standing immunity that
this genotype is not covered by*. ``escape_gain`` is that minus the start
lineage's own escape — what the mutations actually buy — and the single/pair
decomposition and epistasis are computed on the gain.

Outputs land in a ``population_escape/`` subfolder of the PLANT run directory,
one set per ``--as-of`` date. Needs no model: it reads the scan's
``genotype_embeddings.csv`` and the background CSV.

Caveats this script prints rather than hides
--------------------------------------------
* The background CSV ends in early 2024; an ``--as-of`` date past its last
  collection date silently means "all of history", so the actual date range and
  the gap to the requested date are reported every run.
* Sequence counts are prevalence only to the extent surveillance was uniform.
  ``--normalise-by`` picks which assumption you are making; neither is neutral.
* PLANT coordinates are a model's antigenic space, not measured HI titres. The
  distance scale is calibrated by ``--cross-immunity-scale``, and every number
  here moves with it, so it is recorded in the metadata of every run.

Example
-------
    python scripts/JtoJ24_scan/plant_population_escape.py \\
        Results/JtoJ.2.4_scan/plant \\
        --as-of 2023-01-01 --as-of 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plant_order_scan import (  # noqa: E402
    BLUE_INK,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    INK,
    INK_MUTED,
    _tidy,
    diverging_cmap,
    escape_basis,
    lineage_label,
    place_labels,
    ramp_colour,
    split_genotype_label,
)

DEFAULT_BACKGROUND_CSV = Path(
    "/home3/oml4h/hugging_face_downloads/PLANT_model/code/examples/backgrounds.csv"
)
GENOTYPE_TABLE_NAME = "genotype_embeddings.csv"
METADATA_NAME = "run_metadata.json"
SUBFOLDER = "population_escape"

#: Single-hue ramp for immunity density. Orange because the genotype markers
#: already own the blue ramp, and two sequential contexts in one figure take
#: different hues rather than different steps of the same one.
IMMUNITY_RAMP = ["#fff5ef", "#fbd9c7", "#f2a179", "#eb6834", "#a83c14"]
#: Validated categorical slots, for one line per mutation in the date sweep.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


###############################################################################
# CLI
###############################################################################
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path,
                        help=f"A completed PLANT scan directory containing "
                             f"{GENOTYPE_TABLE_NAME}.")
    parser.add_argument("--as-of", action="append", default=None, metavar="DATE",
                        help="Score against sequences collected strictly before this date "
                             "(YYYY, YYYY-MM or YYYY-MM-DD). Repeatable; with more than one "
                             "date a trend figure is drawn too. Default: the last collection "
                             "date in the background CSV.")
    parser.add_argument("--background-csv", type=Path, default=DEFAULT_BACKGROUND_CSV,
                        help="PLANT background embeddings with a collection date "
                             "(default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Where to write (default: <run_dir>/{SUBFOLDER}).")

    immunity = parser.add_argument_group("immune landscape")
    immunity.add_argument("--half-life", type=float, default=1.0, metavar="YEARS",
                          help="Recency half-life of immunity in years. Smaller means a "
                               "stronger recency bias (default: %(default)s)")
    immunity.add_argument("--normalise-by", choices=("year", "month", "none"), default="year",
                          help="Period whose total weight is set by recency alone, flattening "
                               "surveillance effort ACROSS periods. 'none' weights each "
                               "sequence by recency only (default: %(default)s)")
    immunity.add_argument("--within-period", choices=("abundance", "unique", "density"),
                          default="abundance",
                          help="How a period's weight is split among its own sequences. "
                               "'abundance': equally per sequence, so weight follows the "
                               "sampled composition -- a heavily-sequenced clade dominates its "
                               "period. 'unique': equally per distinct antigenic position, so "
                               "weight follows the period's DIVERSITY. 'density': equally per "
                               "occupied grid cell of --density-radius, a smoothed 'unique'. "
                               "Needs --normalise-by other than 'none' (default: %(default)s)")
    immunity.add_argument("--density-radius", type=float, default=0.25, metavar="UNITS",
                          help="Grid cell size for --within-period density (default: %(default)s)")
    immunity.add_argument("--max-age", type=float, default=None, metavar="YEARS",
                          help="Discard sequences older than this. The recency weight already "
                               "makes old sequences negligible; this is for speed and for "
                               "testing sensitivity (default: keep all history).")

    kernel = parser.add_argument_group("cross-immunity kernel")
    kernel.add_argument("--kernel", choices=("exponential", "sigmoid", "linear"),
                        default="exponential",
                        help="Shape of cross-immunity vs antigenic distance "
                             "(default: %(default)s)")
    kernel.add_argument("--cross-immunity-scale", type=float, default=2.0, metavar="UNITS",
                        help="Antigenic distance at which cross-immunity has fallen to 1/e "
                             "(exponential), to 1/2 (sigmoid midpoint), or to 0 (linear). "
                             "Every escape number scales with this (default: %(default)s)")

    parser.add_argument("--chunk-size", type=int, default=64,
                        help="Genotypes per distance block; trades memory for speed "
                             "(default: %(default)s)")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs only.")
    return parser.parse_args(argv)


###############################################################################
# Dates
###############################################################################
def to_decimal_year(value) -> Optional[float]:
    """Decimal year from ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``.

    The background CSV mixes all three: everything before ~2005 is a bare year.
    A bare year becomes mid-year and a bare month mid-month, because the true
    date is uniform over the period and the midpoint is the unbiased estimate --
    snapping them to 1 January would systematically age those records.
    """
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None

    parts = text.replace("/", "-").split("-")
    try:
        year = int(parts[0])
    except ValueError:
        return None
    if len(parts) == 1:
        return year + 0.5

    try:
        month = int(parts[1])
    except ValueError:
        return year + 0.5
    if not 1 <= month <= 12:
        return year + 0.5
    if len(parts) == 2:
        return year + (month - 0.5) / 12.0

    try:
        day = int(parts[2])
        start = date(year, 1, 1).toordinal()
        span = date(year + 1, 1, 1).toordinal() - start
        return year + (date(year, month, day).toordinal() - start) / span
    except ValueError:
        return year + (month - 0.5) / 12.0


def format_decimal_year(value: float) -> str:
    """Calendar date for a decimal year, for printing."""
    year = int(np.floor(value))
    start = date(year, 1, 1).toordinal()
    span = date(year + 1, 1, 1).toordinal() - start
    return date.fromordinal(start + int(round((value - year) * span))).isoformat()


###############################################################################
# Immune landscape
###############################################################################
def load_backgrounds(path: Path) -> pd.DataFrame:
    """Background embeddings with a usable collection date."""
    frame = pd.read_csv(path)
    columns = {name.strip().lower(): name for name in frame.columns}
    date_column = columns.get("collection date") or columns.get("date")
    if date_column is None:
        raise ValueError(
            f"{path} has no 'collection date' column, so there is no way to restrict the "
            f"immune landscape to a date. Columns present: {list(frame.columns)}"
        )
    missing = {"X", "Y", "Z"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing coordinate columns: {sorted(missing)}")

    frame = frame.rename(columns={date_column: "collection_date"})
    frame["decimal_year"] = frame["collection_date"].map(to_decimal_year)
    undated = int(frame["decimal_year"].isna().sum())
    if undated:
        print(f"[warning] {undated} background sequence(s) have an unparseable collection "
              "date and are dropped from the immune landscape.")
    frame = frame.dropna(subset=["decimal_year", "X", "Y", "Z"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path} yielded no dated background sequences.")
    return frame


def within_period_share(
    coordinates: np.ndarray,
    periods: np.ndarray,
    mode: str,
    radius: float,
) -> np.ndarray:
    """Fraction of its period's weight each sequence receives.

    Three answers to "what does a period's weight represent", all normalised so
    each period's shares sum to 1:

    ``abundance``
        Every sequence an equal share, so weight follows the period's *sampled
        composition*. A clade that was sequenced ten times more than another of
        equal true prevalence carries ten times the immunity.
    ``unique``
        Every distinct antigenic position an equal share, split among the
        sequences sitting on it, so weight follows the period's *diversity*.
        Duplicate deposits of the same phenotype stop counting twice. The
        background CSV is full of exact coordinate ties, so this is a large
        change, not a rounding one.
    ``density``
        As ``unique`` but on a grid of side ``radius``, so near-identical
        sequences also collapse. A dense cluster of 500 similar strains then
        counts about as much as one isolated strain, which is the right thing
        if the cluster is a sequencing artefact and the wrong thing if it is a
        real epidemic. That choice is yours to make, which is why it is a flag.
    """
    if mode == "abundance":
        cells = np.arange(len(coordinates))
    elif mode == "unique":
        cells = np.unique(coordinates, axis=0, return_inverse=True)[1]
    elif mode == "density":
        if radius <= 0:
            raise ValueError("--density-radius must be positive.")
        cells = np.unique(np.floor(coordinates / radius), axis=0, return_inverse=True)[1]
    else:
        raise ValueError(f"Unknown --within-period mode: {mode}")

    grouping = pd.DataFrame({"period": periods, "cell": np.asarray(cells).ravel()})
    cell_size = grouping.groupby(["period", "cell"])["cell"].transform("size").to_numpy(float)
    n_cells = grouping.groupby("period")["cell"].transform("nunique").to_numpy(float)
    return 1.0 / (n_cells * cell_size)


def immune_weights(
    backgrounds: pd.DataFrame,
    as_of: float,
    half_life: float,
    normalise_by: str,
    max_age: Optional[float],
    within_period: str = "abundance",
    density_radius: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Recency weights over the sequences circulating before ``as_of``.

    Returns the kept coordinates, weights summing to 1, and diagnostics. The
    weight is a product of two independent choices, kept separate on purpose::

        w(s)  =  recency(s)  x  share(s)

    ``recency`` says how much a *period* counts and ``share`` how a period
    divides its weight among its own sequences -- surveillance effort across
    periods and composition within one are different biases and are corrected
    by different flags.
    """
    if half_life <= 0:
        raise ValueError("--half-life must be positive.")
    if normalise_by == "none" and within_period != "abundance":
        raise ValueError(
            "--within-period only has meaning inside a period; pass --normalise-by "
            "year or month, or leave --within-period at 'abundance'."
        )

    age = as_of - backgrounds["decimal_year"].to_numpy(float)
    keep = age > 0
    if max_age is not None:
        keep &= age <= max_age
    if not keep.any():
        raise ValueError(
            f"No background sequences fall before {format_decimal_year(as_of)}"
            + (f" and within {max_age} year(s) of it." if max_age is not None else ".")
        )

    kept = backgrounds[keep].reset_index(drop=True)
    age = age[keep]
    weight = 0.5 ** (age / half_life)

    if normalise_by != "none":
        decimal_year = kept["decimal_year"].to_numpy(float)
        periods = np.floor(decimal_year * 12.0) if normalise_by == "month" \
            else np.floor(decimal_year)
        weight = weight * within_period_share(
            kept[["X", "Y", "Z"]].to_numpy(float), periods, within_period, density_radius
        )

    total = float(weight.sum())
    if total <= 0:
        raise ValueError("All immune weights underflowed to zero; raise --half-life.")
    weight = weight / total

    # Kish effective sample size: how many sequences the weighted landscape is
    # really made of. A tiny value means the score rests on a handful of records.
    effective = float(1.0 / np.sum(weight ** 2))
    by_year = (
        pd.DataFrame({"year": np.floor(kept["decimal_year"].to_numpy(float)).astype(int),
                      "weight": weight})
        .groupby("year")["weight"].agg(["sum", "size"])
        .rename(columns={"sum": "weight", "size": "n_sequences"})
        .sort_index()
    )
    coordinates = kept[["X", "Y", "Z"]].to_numpy(float)
    diagnostics = {
        "n_sequences": int(len(kept)),
        "effective_sample_size": effective,
        "earliest_collection": format_decimal_year(float(kept["decimal_year"].min())),
        "latest_collection": format_decimal_year(float(kept["decimal_year"].max())),
        "weighted_mean_age_years": float(np.sum(weight * age)),
        "weighted_centroid": (weight @ coordinates).tolist(),
        "weight_by_year": by_year,
    }
    return coordinates, weight, diagnostics


def cross_immunity(distance: np.ndarray, scale: float, kernel: str) -> np.ndarray:
    """Fraction of immunity to a strain at ``distance`` that still cross-protects."""
    if scale <= 0:
        raise ValueError("--cross-immunity-scale must be positive.")
    if kernel == "exponential":
        return np.exp(-distance / scale)
    if kernel == "linear":
        return np.clip(1.0 - distance / scale, 0.0, 1.0)
    if kernel == "sigmoid":
        # Centred on `scale` with a quarter-scale steepness, so it plateaus near
        # full protection at short range instead of dropping fastest at d = 0.
        return 1.0 / (1.0 + np.exp((distance - scale) / (scale / 4.0)))
    raise ValueError(f"Unknown kernel: {kernel}")


def population_escape(
    points: np.ndarray,
    background: np.ndarray,
    weights: np.ndarray,
    scale: float,
    kernel: str,
    chunk_size: int = 64,
) -> np.ndarray:
    """Weighted share of standing immunity each point escapes, in [0, 1]."""
    background_sq = np.einsum("ij,ij->i", background, background)
    escape = np.empty(len(points), dtype=float)
    for start in range(0, len(points), max(1, chunk_size)):
        block = points[start:start + max(1, chunk_size)]
        # |a-b|^2 expanded, so the big intermediate is n_block x n_background
        # rather than n_block x n_background x 3.
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + background_sq[None, :]
            - 2.0 * block @ background.T
        )
        distance = np.sqrt(np.maximum(squared, 0.0))
        escape[start:start + len(block)] = 1.0 - cross_immunity(distance, scale, kernel) @ weights
    return escape


def check_shared_frame(root: np.ndarray, backgrounds: pd.DataFrame) -> Dict[str, object]:
    """Nearest background sequence to the start lineage, as a frame sanity check.

    The genotype embeddings and ``backgrounds.csv`` are only comparable if they
    came out of the same PLANT checkpoint. If they did, the start lineage sits
    essentially on top of a real sequence; if they did not, this distance is
    large and every escape number below is meaningless.
    """
    coordinates = backgrounds[["X", "Y", "Z"]].to_numpy(float)
    distances = np.linalg.norm(coordinates - root, axis=1)
    nearest = int(np.argmin(distances))
    row = backgrounds.iloc[nearest]
    report = {
        "nearest_background": str(row.get("name", nearest)),
        "nearest_subclade": str(row.get("subclade", "")),
        "nearest_collection_date": str(row.get("collection_date", "")),
        "distance": float(distances[nearest]),
    }
    if report["distance"] > 1.0:
        print(f"[warning] The start lineage is {report['distance']:.3f} antigenic units from "
              f"the nearest background sequence ({report['nearest_background']}). That is far "
              "enough to suspect the genotype embeddings and the background CSV did not come "
              "from the same PLANT checkpoint -- check before trusting these numbers.")
    return report


###############################################################################
# Tables
###############################################################################
def build_population_tables(
    genotypes: pd.DataFrame,
    escape: np.ndarray,
    score=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Per-genotype escape, then the single-mutation and pairwise decomposition.

    ``score`` is a callable mapping an (n, 3) array of coordinates to their
    population escape. When it is supplied the pairwise table gains the
    curvature decomposition described in :func:`add_curvature_decomposition`,
    which is the difference between "these two mutations interact" and "escape
    is a saturating function of distance". Without it only the raw
    ``epistasis`` column is produced.
    """
    table = genotypes.assign(population_escape=escape)
    root_mask = table["n_fixed"] == 0
    if not root_mask.any():
        raise ValueError("No root genotype in the embedding table.")
    root_escape = float(table.loc[root_mask, "population_escape"].iloc[0])
    table["escape_gain"] = table["population_escape"] - root_escape

    by_label = table.set_index("genotype_h3")
    full_label = table.loc[table["n_fixed"].idxmax(), "genotype_h3"]
    names = [name for name in split_genotype_label(full_label) if name in by_label.index]

    singles = pd.DataFrame(
        {
            "mutation_h3": names,
            "population_escape": [float(by_label.loc[n, "population_escape"]) for n in names],
            "escape_gain": [float(by_label.loc[n, "escape_gain"]) for n in names],
        }
    )
    singles["root_escape"] = root_escape
    # What fraction of the immunity still covering the start lineage this one
    # mutation gets out from under. Escape is bounded, so a raw gain of 0.02 is
    # a very different thing at root escape 0.10 than at 0.90.
    remaining = 1.0 - root_escape
    singles["share_of_remaining_immunity"] = (
        singles["escape_gain"] / remaining if remaining > 0 else np.nan
    )

    pair_rows = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            label = f"{first}+{second}"
            if label not in by_label.index:
                continue
            observed = float(by_label.loc[label, "escape_gain"])
            additive = float(by_label.loc[first, "escape_gain"]) + \
                float(by_label.loc[second, "escape_gain"])
            pair_rows.append(
                {
                    "mutation_a": first,
                    "mutation_b": second,
                    "pair_h3": label,
                    "population_escape": float(by_label.loc[label, "population_escape"]),
                    "escape_gain": observed,
                    "additive_gain": additive,
                    "epistasis": observed - additive,
                    "relative_epistasis": (observed - additive) / additive if additive else np.nan,
                }
            )
    pairs = pd.DataFrame(pair_rows) if pair_rows else None
    if pairs is not None and score is not None:
        pairs = add_curvature_decomposition(pairs, by_label, root_escape, score)
    return table, singles, pairs


def add_curvature_decomposition(
    pairs: pd.DataFrame,
    by_label: pd.DataFrame,
    root_escape: float,
    score,
) -> pd.DataFrame:
    """Split each pair's ε into kernel curvature and genuine interaction.

    ``epistasis = observed_gain - (gain_a + gain_b)`` is additivity in *escape*,
    and escape is a bounded, saturating function of distance. So two mutations
    that add perfectly in coordinates still fail to add in escape: the pair
    lands further out, where the kernel is flatter (or, if the immunity is close,
    where it is steeper), and the difference has nothing to do with the
    mutations interacting.

    The fix is a second baseline. Score the genotype that additivity *predicts*
    in coordinate space -- the point ``root + Δa + Δb`` -- against the same
    landscape. Then::

        epistasis            = observed - (gain_a + gain_b)      (total, as before)
        kernel_curvature     = additive_genotype_gain - (gain_a + gain_b)
        epistasis_vs_additive_genotype
                             = observed - additive_genotype_gain (the interaction)

    and the first is exactly the sum of the other two. Only the last one means
    "these two substitutions do something together"; on a landscape sitting close
    to the start lineage the curvature term can be several times larger.
    """
    root = by_label.loc["root", ["X", "Y", "Z"]].to_numpy(float)
    predicted = np.array(
        [
            root
            + (by_label.loc[row.mutation_a, ["X", "Y", "Z"]].to_numpy(float) - root)
            + (by_label.loc[row.mutation_b, ["X", "Y", "Z"]].to_numpy(float) - root)
            for row in pairs.itertuples()
        ]
    )
    additive_genotype_gain = np.asarray(score(predicted), dtype=float) - root_escape

    pairs = pairs.copy()
    pairs["additive_genotype_gain"] = additive_genotype_gain
    pairs["kernel_curvature"] = additive_genotype_gain - pairs["additive_gain"]
    pairs["epistasis_vs_additive_genotype"] = pairs["escape_gain"] - additive_genotype_gain
    pairs["curvature_share_of_epistasis"] = np.where(
        pairs["epistasis"].abs() > 0,
        pairs["kernel_curvature"] / pairs["epistasis"],
        np.nan,
    )
    return pairs


###############################################################################
# Figures
###############################################################################
def plot_population_singles_pairs(
    singles: pd.DataFrame,
    pairs: Optional[pd.DataFrame],
    root_escape: float,
    path: Path,
    start_label: str,
    as_of_text: str,
) -> None:
    """Escape bought by each mutation, and by each pair against its additive sum."""
    n_single = len(singles)
    n_pair = 0 if pairs is None else len(pairs)
    heights = [max(2.2, 0.42 * n_single)] + ([max(2.4, 0.38 * n_pair)] if n_pair else [])

    figure, axes = plt.subplots(
        len(heights), 1, figsize=(11, sum(heights) + 2.0),
        gridspec_kw={"height_ratios": heights}, sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    ranked = singles.sort_values("escape_gain")
    y = np.arange(n_single)
    top = axes[0]
    top.barh(y, ranked["escape_gain"], height=0.58, color=BLUE_INK)
    for index, row in enumerate(ranked.itertuples()):
        offset = 0.012 * max(ranked["escape_gain"].max(), 1e-9)
        top.text(row.escape_gain + offset, index,
                 f"{row.escape_gain:+.4f}   ({row.share_of_remaining_immunity:+.1%} of the "
                 f"immunity still covering {start_label})",
                 va="center", fontsize=8.5, color=INK)
    top.set_yticks(y)
    top.set_yticklabels(ranked["mutation_h3"], fontsize=10, color=INK)
    top.set_title(
        f"Escape from the standing immunity as of {as_of_text}, bought by each mutation\n"
        f"{start_label} itself already escapes {root_escape:.3f} of that immunity",
        fontsize=12, color=INK, loc="left",
    )
    top.axvline(0, color=INK_MUTED, linewidth=0.8)
    top.grid(axis="x", alpha=0.25, linewidth=0.6)
    top.set_axisbelow(True)
    top.margins(x=0.28)
    _tidy(top)

    if n_pair:
        bottom = axes[1]
        ordered = pairs.sort_values("escape_gain")
        split = "additive_genotype_gain" in ordered
        y = np.arange(n_pair)
        span = max(abs(ordered["escape_gain"].max()), 1e-9)
        for index, row in enumerate(ordered.itertuples()):
            # Grey leg: what a coordinate-additive double mutant would score, so
            # the saturating kernel's own contribution is drawn, not folded into
            # the interaction. Coloured leg: what is left, the real interaction.
            middle = row.additive_genotype_gain if split else row.additive_gain
            interaction = (row.epistasis_vs_additive_genotype if split else row.epistasis)
            colour = DIVERGING_HIGH if interaction >= 0 else DIVERGING_LOW
            if split:
                bottom.plot([row.additive_gain, middle], [index, index],
                            color="#b9b8b2", linewidth=2.6, solid_capstyle="round", zorder=2)
                bottom.plot(middle, index, marker="x", markersize=8, color=INK_MUTED,
                            markeredgewidth=1.6, zorder=3)
            bottom.plot([middle, row.escape_gain], [index, index],
                        color=colour, linewidth=2.6, solid_capstyle="round", zorder=2)
            bottom.plot(row.additive_gain, index, marker="o", markersize=8,
                        markerfacecolor="white", markeredgecolor=INK_MUTED,
                        markeredgewidth=1.4, zorder=3)
            bottom.plot(row.escape_gain, index, marker="o", markersize=9, color=BLUE_INK,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=4)
            bottom.text(max(row.escape_gain, row.additive_gain, middle) + 0.012 * span,
                        index, f"interaction {interaction:+.4f}", va="center",
                        fontsize=8, color=colour)
        bottom.set_yticks(y)
        bottom.set_yticklabels(ordered["pair_h3"], fontsize=9.5, color=INK)
        bottom.set_title(
            "Pairwise: observed escape vs the sum of the two single mutants"
            + ("\ngrey leg = the saturating kernel, coloured leg = the mutations interacting"
               if split else ""),
            fontsize=12, color=INK, loc="left",
        )
        bottom.axvline(0, color=INK_MUTED, linewidth=0.8)
        bottom.grid(axis="x", alpha=0.25, linewidth=0.6)
        bottom.set_axisbelow(True)
        bottom.margins(x=0.22)
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="white",
                       markeredgecolor=INK_MUTED, label="additive in escape (gain a + gain b)"),
        ]
        if split:
            handles.append(
                plt.Line2D([], [], marker="x", linestyle="-", color="#b9b8b2",
                           markeredgecolor=INK_MUTED, markersize=8, markeredgewidth=1.6,
                           label="additive in coordinates, scored (kernel curvature)")
            )
        handles += [
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_INK,
                       label="observed double mutant"),
            plt.Line2D([], [], color=DIVERGING_HIGH, linewidth=2.6,
                       label="interaction > 0: super-additive"),
            plt.Line2D([], [], color=DIVERGING_LOW, linewidth=2.6,
                       label="interaction < 0: sub-additive"),
        ]
        bottom.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
        _tidy(bottom)

    axes[-1].set_xlabel(
        "Gain in escaped share of the recency-weighted standing immunity", fontsize=10
    )
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_population_epistasis_matrix(
    singles: pd.DataFrame,
    pairs: pd.DataFrame,
    path: Path,
    as_of_text: str,
) -> None:
    """Symmetric ε matrix on the population-escape scale."""
    names = list(singles.sort_values("escape_gain", ascending=False)["mutation_h3"])
    index = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(names), len(names)), np.nan)
    # Plot the interaction term when it is available. The raw epistasis is
    # dominated by the saturating kernel whenever the immunity sits close to the
    # start lineage, and a matrix of kernel curvature reads as biology.
    column = "epistasis_vs_additive_genotype" \
        if "epistasis_vs_additive_genotype" in pairs else "epistasis"
    for row in pairs.itertuples():
        i, j = index[row.mutation_a], index[row.mutation_b]
        matrix[i, j] = matrix[j, i] = getattr(row, column)

    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = limit if limit > 0 else 1.0

    size = max(4.6, 0.85 * len(names) + 3.2)
    figure, axis = plt.subplots(figsize=(size + 1.8, size))
    image = axis.imshow(matrix, cmap=diverging_cmap(), vmin=-limit, vmax=limit)

    gain = singles.set_index("mutation_h3")["escape_gain"]
    labels = [f"{name}\n({gain[name]:+.4f})" for name in names]
    axis.set_xticks(np.arange(len(names)))
    axis.set_yticks(np.arange(len(names)))
    axis.set_xticklabels(labels, fontsize=9, color=INK)
    axis.set_yticklabels(labels, fontsize=9, color=INK)
    axis.set_xticks(np.arange(len(names) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)
    axis.tick_params(which="major", length=0)

    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                axis.text(j, i, "—", ha="center", va="center", color=INK_MUTED, fontsize=11)
            elif np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:+.4f}", ha="center", va="center",
                          color=INK, fontsize=8.5)

    subtitle = (
        "interaction = double mutant − the coordinate-additive double mutant,\n"
        "i.e. epistasis with the saturating kernel's own curvature removed"
        if column != "epistasis" else
        "ε = double mutant − (single a + single b)"
    )
    axis.set_title(
        f"Pairwise interaction in population escape, immunity as of {as_of_text}\n"
        f"{subtitle}; single-mutant gain in brackets",
        fontsize=11, color=INK,
    )
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label("share of standing immunity", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_immune_landscape(
    table: pd.DataFrame,
    background: np.ndarray,
    weights: np.ndarray,
    path: Path,
    start_label: str,
    end_label: str,
    as_of_text: str,
    diagnostics: Dict[str, object],
) -> None:
    """The weighted immunity the escape is measured against, with the genotypes on it.

    Drawn in the same frame as ``plant_escape_map.png`` -- x along start -> end,
    y the principal off-axis direction of the genotypes -- so the two figures can
    be read side by side.
    """
    from matplotlib.colors import LinearSegmentedColormap

    root, unit, _ = escape_basis(table)
    coordinates = table[["X", "Y", "Z"]].to_numpy(float)
    offsets = coordinates - root
    along = offsets @ unit
    residual = offsets - np.outer(along, unit)
    if np.allclose(residual, 0):
        perpendicular = np.zeros(3)
    else:
        perpendicular = np.linalg.svd(residual, full_matrices=False)[2][0]
        perpendicular = perpendicular - (perpendicular @ unit) * unit
        norm = np.linalg.norm(perpendicular)
        perpendicular = perpendicular / norm if norm else np.zeros(3)

    frame = table.assign(escape_along=along, escape_off=offsets @ perpendicular)
    background_offsets = background - root
    background_x = background_offsets @ unit
    background_y = background_offsets @ perpendicular

    figure, axis = plt.subplots(figsize=(12, 8.5))
    ramp = LinearSegmentedColormap.from_list("immunity", IMMUNITY_RAMP)
    # Hexbin rather than 150k scatter points: the quantity that matters is the
    # local *weight* density, which a scatter of overlapping dots cannot show.
    mesh = axis.hexbin(background_x, background_y, C=weights, reduce_C_function=np.sum,
                       gridsize=70, cmap=ramp, mincnt=1, linewidths=0.0, zorder=1)
    bar = figure.colorbar(mesh, ax=axis, fraction=0.04, pad=0.02)
    bar.set_label("share of standing immunity per cell", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)

    centroid = np.asarray(diagnostics["weighted_centroid"], dtype=float) - root
    axis.plot(centroid @ unit, centroid @ perpendicular, marker="*", markersize=18,
              color=INK, markeredgecolor="white", markeredgewidth=1.2, zorder=5,
              linestyle="", label="immunity centroid")

    top_level = int(frame["n_fixed"].max())
    for level in sorted(int(value) for value in frame["n_fixed"].unique()):
        subset = frame[frame["n_fixed"] == level]
        axis.scatter(subset["escape_along"], subset["escape_off"],
                     s=42 + 26 * (4 * level / top_level if top_level else 0),
                     color=ramp_colour(level, top_level), edgecolor="white",
                     linewidth=1.2, zorder=4,
                     label=f"{level} mutation{'s' if level != 1 else ''} fixed")

    label_pairs = int((frame["n_fixed"] == 2).sum()) <= 12
    keep = (frame["n_fixed"] <= (2 if label_pairs else 1)) | (frame["n_fixed"] == top_level)
    entries = []
    for row in frame[keep].itertuples():
        name = start_label if row.genotype_h3 == "root" else row.genotype_h3
        weight = "bold" if row.n_fixed in (0, top_level) else "normal"
        entries.append((row.escape_along, row.escape_off, name,
                        {"fontsize": 8.5, "weight": weight}))

    axis.set_xlabel(f"Along the {start_label} → {end_label} axis (PLANT units)", fontsize=10)
    axis.set_ylabel("Principal off-axis direction (PLANT units)", fontsize=10)
    axis.set_title(
        f"The immune landscape as of {as_of_text}, and where the mutations sit on it\n"
        f"{diagnostics['n_sequences']:,} prior sequences, effective sample size "
        f"{diagnostics['effective_sample_size']:.0f}, weighted mean age "
        f"{diagnostics['weighted_mean_age_years']:.2f} y",
        fontsize=12, color=INK, loc="left",
    )
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=8.5, frameon=False, loc="best")
    _tidy(axis)

    figure.tight_layout()
    figure.canvas.draw()
    place_labels(axis, entries)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_escape_vs_date(sweep: pd.DataFrame, path: Path, start_label: str) -> None:
    """One line per mutation: how much escape it buys against successive landscapes."""
    order = (
        sweep.groupby("mutation_h3")["escape_gain"].mean()
        .sort_values(ascending=False)
    )
    shown = list(order.index[:len(SERIES_COLOURS)])
    dropped = list(order.index[len(SERIES_COLOURS):])
    if dropped:
        print(f"[note] escape-vs-date figure shows the {len(shown)} mutations with the "
              f"largest mean gain; {len(dropped)} not drawn: {', '.join(dropped)}. "
              "All of them are in the CSV.")

    figure, axis = plt.subplots(figsize=(11, 6))
    for colour, name in zip(SERIES_COLOURS, shown):
        subset = sweep[sweep["mutation_h3"] == name].sort_values("as_of_decimal_year")
        axis.plot(subset["as_of_decimal_year"], subset["escape_gain"], marker="o",
                  markersize=6, linewidth=2.0, color=colour, label=name,
                  markeredgecolor="white", markeredgewidth=1.0)
        last = subset.iloc[-1]
        axis.annotate(name, (last["as_of_decimal_year"], last["escape_gain"]),
                      textcoords="offset points", xytext=(8, 0), va="center",
                      fontsize=9, color=colour)

    axis.axhline(0, color=INK_MUTED, linewidth=0.8)
    axis.set_xlabel("Immune landscape date (decimal year)", fontsize=10)
    axis.set_ylabel("Gain in escaped share of standing immunity", fontsize=10)
    axis.set_title(
        f"What each mutation is worth against successive immune landscapes, relative to "
        f"{start_label}",
        fontsize=12, color=INK, loc="left",
    )
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.margins(x=0.12)
    if len(shown) > 1:
        axis.legend(fontsize=8.5, frameon=False, loc="best")
    _tidy(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


###############################################################################
# Main
###############################################################################
def resolve_labels(run_dir: Path) -> Tuple[str, str]:
    metadata_path = run_dir / METADATA_NAME
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return (lineage_label(metadata.get("start_header", "start")),
                lineage_label(metadata.get("end_header", "end")))
    return "start", "end"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    table_path = args.run_dir / GENOTYPE_TABLE_NAME
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} not found. Is {args.run_dir} a completed PLANT scan?")
    genotypes = pd.read_csv(table_path)
    start_label, end_label = resolve_labels(args.run_dir)

    backgrounds = load_backgrounds(args.background_csv)
    latest = float(backgrounds["decimal_year"].max())
    print(f"Background cloud: {len(backgrounds):,} dated sequences, "
          f"{format_decimal_year(float(backgrounds['decimal_year'].min()))} to "
          f"{format_decimal_year(latest)}.")

    root = genotypes.loc[genotypes["n_fixed"] == 0, ["X", "Y", "Z"]].to_numpy(float)[0]
    frame_check = check_shared_frame(root, backgrounds)
    print(f"Frame check: {start_label} sits {frame_check['distance']:.4f} units from "
          f"{frame_check['nearest_background']} "
          f"[{frame_check['nearest_subclade']}, {frame_check['nearest_collection_date']}].")

    requested = args.as_of or [format_decimal_year(latest)]
    output_dir = args.output_dir or (args.run_dir / SUBFOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = genotypes[["X", "Y", "Z"]].to_numpy(float)
    sweep_rows: List[pd.DataFrame] = []

    for as_of_text in requested:
        as_of = to_decimal_year(as_of_text)
        if as_of is None:
            raise ValueError(f"Could not parse --as-of {as_of_text!r} as a date.")
        stem = str(as_of_text).replace("/", "-")

        background_coordinates, weights, diagnostics = immune_weights(
            backgrounds, as_of, args.half_life, args.normalise_by, args.max_age,
            args.within_period, args.density_radius,
        )
        if as_of > latest:
            print(f"[warning] --as-of {as_of_text} is after the last background collection "
                  f"date ({format_decimal_year(latest)}), so the landscape is everything "
                  "available and is "
                  f"{as_of - latest:.2f} year(s) staler than requested.")

        def score(candidate_points, _weights=weights, _background=background_coordinates):
            return population_escape(candidate_points, _background, _weights,
                                     args.cross_immunity_scale, args.kernel, args.chunk_size)

        escape = score(points)
        table, singles, pairs = build_population_tables(genotypes, escape, score)
        root_escape = float(singles["root_escape"].iloc[0])

        table.to_csv(output_dir / f"genotype_population_escape_{stem}.csv", index=False)
        singles.to_csv(output_dir / f"single_mutation_population_escape_{stem}.csv", index=False)
        if pairs is not None:
            pairs.to_csv(output_dir / f"pairwise_population_escape_{stem}.csv", index=False)
        by_year: pd.DataFrame = diagnostics.pop("weight_by_year")
        by_year.to_csv(output_dir / f"immune_landscape_by_year_{stem}.csv")

        metadata = {
            "as_of": as_of_text,
            "as_of_decimal_year": as_of,
            "half_life_years": args.half_life,
            "normalise_by": args.normalise_by,
            "within_period": args.within_period,
            "density_radius": args.density_radius,
            "max_age_years": args.max_age,
            "kernel": args.kernel,
            "cross_immunity_scale": args.cross_immunity_scale,
            "background_csv": str(args.background_csv),
            "background_latest_collection": format_decimal_year(latest),
            "root_population_escape": root_escape,
            "frame_check": frame_check,
            **diagnostics,
        }
        (output_dir / f"run_metadata_{stem}.json").write_text(json.dumps(metadata, indent=2))

        print(f"\n=== Immunity as of {as_of_text} ===")
        print(f"  {diagnostics['n_sequences']:,} prior sequences "
              f"({diagnostics['earliest_collection']} to {diagnostics['latest_collection']}), "
              f"effective sample size {diagnostics['effective_sample_size']:.0f}, "
              f"weighted mean age {diagnostics['weighted_mean_age_years']:.2f} y")
        print(f"  {start_label} already escapes {root_escape:.4f} of it. Gains:")
        for row in singles.sort_values("escape_gain", ascending=False).itertuples():
            print(f"    {row.mutation_h3:>12}  {row.escape_gain:+.4f}  "
                  f"({row.share_of_remaining_immunity:+.2%} of the immunity still covering "
                  f"{start_label})")
        if pairs is not None:
            interaction = "epistasis_vs_additive_genotype" \
                if "epistasis_vs_additive_genotype" in pairs else "epistasis"
            strongest = pairs.reindex(
                pairs[interaction].abs().sort_values(ascending=False).index
            )
            print("  Largest departures from additivity "
                  "(ε total = kernel curvature + interaction):")
            for row in strongest.head(3).itertuples():
                line = (f"    {row.pair_h3:>26}  observed={row.escape_gain:+.4f}  "
                        f"additive={row.additive_gain:+.4f}  ε={row.epistasis:+.4f}")
                if interaction != "epistasis":
                    line += (f"  = curvature {row.kernel_curvature:+.4f} "
                             f"+ interaction {row.epistasis_vs_additive_genotype:+.4f}")
                print(line)
            if interaction != "epistasis":
                curvature = pairs["kernel_curvature"].abs().sum()
                total = pairs["epistasis"].abs().sum()
                if total > 0:
                    print(f"  Kernel curvature accounts for {curvature / total:.0%} of the "
                          "total |ε| here; only the interaction term means the substitutions "
                          "do something together.")

        if not args.no_plots:
            plot_population_singles_pairs(
                singles, pairs, root_escape,
                output_dir / f"population_escape_singles_pairs_{stem}.png",
                start_label, as_of_text,
            )
            plot_immune_landscape(
                table, background_coordinates, weights,
                output_dir / f"immune_landscape_{stem}.png",
                start_label, end_label, as_of_text, diagnostics,
            )
            if pairs is not None:
                plot_population_epistasis_matrix(
                    singles, pairs,
                    output_dir / f"population_epistasis_matrix_{stem}.png", as_of_text,
                )

        sweep_rows.append(singles.assign(as_of=as_of_text, as_of_decimal_year=as_of))

    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep.to_csv(output_dir / "single_mutation_escape_by_date.csv", index=False)
    if len(requested) > 1 and not args.no_plots:
        plot_escape_vs_date(sweep, output_dir / "population_escape_vs_date.png", start_label)

    print(f"\nWrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
