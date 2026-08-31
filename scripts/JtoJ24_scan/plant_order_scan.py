"""PLANT 3D antigenic-space trajectories for every ordering of a mutation set.

What it answers
---------------
"How much antigenic escape does each mutation buy on its own, how much does each
pair buy together, and how much does the trajectory from J to J.2.4 depend on the
order the mutations fix in?"

Every ordering ends on the same genotype, so the order-dependence is a property
of the *intermediates* only, and the per-ordering summary is close to flat for
small sets. The escape figures below are the ones that separate the mutations:

``plant_escape_singles_pairs.png``
    Each single mutant's antigenic displacement from the start lineage, split
    into the part aimed at the endpoint and the part that is not; below it, each
    double mutant's escape against the sum of its two singles.
``plant_escape_epistasis_matrix.png``
    ε = double − (single a + single b) for every pair, on the endpoint axis.
``plant_escape_map.png``
    PLANT space rotated into the escape frame, with the additive prediction and
    the observed position of every pair joined by an arrow.

Method
------
Same hypercube trick as ``epistasis_order_scan.py``: the ``n!`` orderings visit
only ``2**n`` distinct genotypes, and PLANT's embedding is a function of the
genotype alone. So all ``2**n`` genotypes are embedded once and every ordering is
reconstructed as a polyline through those points. For n = 4 that is 16 embeddings
covering all 24 orderings exactly.

Model loading, trimming and embedding are reused from
``Notebooks/OM_influenza/Plant_batch_fastas.py``, which shares its checkpoint and
pipeline with ``Notebooks/OM_influenza/Plant.run.py`` — so the coordinates here
are directly comparable to the outputs of that script. Unlike those two, every
path is exposed as a command-line argument.

PLANT scores a 329-residue HA1 window, so each genotype is projected onto that
window before embedding and the scan asserts that all requested mutations
survive the projection rather than silently embedding an unmutated sequence.

Example
-------
    python scripts/JtoJ24_scan/plant_order_scan.py \
        --fasta Sequences/huH3N2_HA_CDS.translated.fas \
        --start-id J --end-id J.2.4 \
        --output-dir Results/JtoJ.2.4_scan/plant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from order_scan_common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    Mutation,
    add_common_arguments,
    all_orderings,
    all_subsets,
    apply_mutations,
    build_h3_map,
    describe_mutation_set,
    genotype_id,
    mutations_between,
    order_id,
    path_genotypes,
    read_fasta,
    resolve_mutation_set,
    write_fasta,
)

DEFAULT_PLANT_CODE_DIR = Path("/home3/oml4h/hugging_face_downloads/PLANT_model/code")
DEFAULT_PLANT_MODEL_DIR = Path("/home3/oml4h/hugging_face_downloads/PLANT_model")
DEFAULT_PLANT_SUBFOLDER = "variants/PLANT_fixed"
DEFAULT_PLANT_BASE_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_BACKGROUND_CSV = DEFAULT_PLANT_CODE_DIR / "examples/backgrounds.csv"
PLANT_BATCH_MODULE_DIR = REPO_ROOT / "Notebooks/OM_influenza"


###############################################################################
# CLI
###############################################################################
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)

    plant = parser.add_argument_group("PLANT model")
    plant.add_argument("--plant-code-dir", type=Path, default=DEFAULT_PLANT_CODE_DIR,
                       help="Checkout of the PLANT repo; its src/ is put on sys.path "
                            "(default: %(default)s)")
    plant.add_argument("--plant-model-dir", type=Path, default=DEFAULT_PLANT_MODEL_DIR,
                       help="Local HuggingFace snapshot holding the PLANT weights and "
                            "encoder joblibs (default: %(default)s)")
    plant.add_argument("--plant-subfolder", default=DEFAULT_PLANT_SUBFOLDER,
                       help="Checkpoint subfolder inside --plant-model-dir (default: %(default)s)")
    plant.add_argument("--plant-base-model", default=DEFAULT_PLANT_BASE_MODEL,
                       help="Backbone whose tokeniser/config PLANT was built on "
                            "(default: %(default)s)")
    plant.add_argument("--batch-size", type=int, default=32,
                       help="Sequences per PLANT forward pass (default: %(default)s)")

    context = parser.add_argument_group("context points")
    context.add_argument("--background-csv", type=Path, default=DEFAULT_BACKGROUND_CSV,
                         help="PLANT background embedding CSV drawn behind the paths; pass "
                              "'none' to omit (default: %(default)s)")
    context.add_argument("--no-observed", action="store_true",
                         help="Do not embed the other records in --fasta as reference points.")
    context.add_argument("--restrict-to-window", action="store_true",
                         help="Drop mutations that fall outside PLANT's HA1 window instead of "
                              "erroring. PLANT only models HA1, so HA2 substitutions are "
                              "invisible to it by construction. Dropped mutations are printed "
                              "and recorded in run_metadata.json.")

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "plant",
                        help="Directory for all CSVs and figures (default: %(default)s)")
    output.add_argument("--no-plots", action="store_true", help="Write CSVs only.")
    output.add_argument("--dry-run", action="store_true",
                        help="Enumerate and trim genotypes, verify the mutations survive the "
                             "329-residue projection, then stop without loading PLANT.")
    return parser.parse_args(argv)


###############################################################################
# PLANT runtime, reused from Plant_batch_fastas
###############################################################################
def import_plant_batch(args: argparse.Namespace):
    """Import Plant_batch_fastas with the CLI-supplied paths in force.

    The module resolves its checkpoint paths from module-level globals at call
    time, so overriding them after import is enough to redirect it — and it means
    there is still exactly one implementation of the PLANT loading/embedding path
    in the repo rather than a second copy that can drift.
    """
    plant_src = Path(args.plant_code_dir) / "src"
    if str(plant_src) not in sys.path:
        sys.path.insert(0, str(plant_src))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(PLANT_BATCH_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(PLANT_BATCH_MODULE_DIR))

    import Plant_batch_fastas as plant_batch  # noqa: WPS433

    plant_batch.PLANT_CODE_DIR = Path(args.plant_code_dir)
    plant_batch.LOCAL_HF_DIR = Path(args.plant_model_dir)
    plant_batch.SUBFOLDER = args.plant_subfolder
    plant_batch.MODEL_NAME = args.plant_base_model
    return plant_batch


def project_to_plant_window(sequence: str, plant_batch) -> str:
    """Project a full HA protein onto PLANT's fixed-length HA1 window."""
    sys.path.insert(0, str(REPO_ROOT))
    from Functions_HuggingFace import plant_trim_to_target_length  # noqa: WPS433

    trimmed = plant_trim_to_target_length(
        sequence, plant_batch.REFERENCE_SEQ, plant_batch.TARGET_LENGTH, return_start_pos=False
    )
    if trimmed is None or len(trimmed) != plant_batch.TARGET_LENGTH:
        raise ValueError(
            f"Sequence could not be projected onto PLANT's {plant_batch.TARGET_LENGTH}-residue "
            "HA1 window."
        )
    return trimmed


def verify_window_mutations(
    trimmed_root: str,
    trimmed_full: str,
    mutations: Sequence[Mutation],
) -> Dict[int, Mutation]:
    """Check every mutation survives the HA1 projection; return window positions.

    Without this, a mutation that falls outside PLANT's window (or is lost to the
    trimming alignment) would produce two identical embeddings and a trajectory
    that looks like a genuine "this mutation does nothing" result.
    """
    expected = sorted(mutations)
    observed = mutations_between(trimmed_root, trimmed_full)

    # Both lists run in ascending position order and the projection preserves
    # order, so pair them positionally. Matching on (wt, alt) instead would
    # mis-pair two mutations that happen to share a substitution type.
    if [(m.wt, m.alt) for m in observed] != [(m.wt, m.alt) for m in expected]:
        raise ValueError(
            "Mutations did not survive projection onto PLANT's HA1 window.\n"
            f"  expected ({len(expected)}) {[m.name for m in expected]}\n"
            f"  found in window ({len(observed)}, in window coordinates) "
            f"{[m.name for m in observed]}\n"
            "All scanned mutations must lie inside the window for the trajectories to mean "
            "anything. PLANT only models HA1, so an HA2 substitution is invisible to it by "
            "construction -- pass --restrict-to-window to drop out-of-window mutations and "
            "scan the rest."
        )

    return {original.pos: window for original, window in zip(expected, observed)}


def partition_by_window(
    start_seq: str,
    mutations: Sequence[Mutation],
    plant_batch,
) -> Tuple[List[Mutation], List[Mutation]]:
    """Split mutations into those PLANT's HA1 window can see and those it cannot.

    Tests each mutation on its own: apply it to the start sequence, project, and
    see whether the projection still differs from the projected start. That is
    exact, and it does not care *why* a site is invisible (HA2, a signal-peptide
    residue, or loss to the trimming alignment).
    """
    windowed_root = project_to_plant_window(start_seq, plant_batch)
    inside, outside = [], []
    for mutation in sorted(mutations):
        candidate = project_to_plant_window(
            apply_mutations(start_seq, [mutation]), plant_batch
        )
        (inside if candidate != windowed_root else outside).append(mutation)
    return inside, outside


###############################################################################
# Geometry
###############################################################################
def axis_projection(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    """Position along, and perpendicular distance from, the start->end axis."""
    axis = end - start
    length_squared = float(np.dot(axis, axis))
    if length_squared == 0:
        return np.zeros(len(points)), np.linalg.norm(points - start, axis=1)
    t = ((points - start) @ axis) / length_squared
    projected = start + np.outer(t, axis)
    deviation = np.linalg.norm(points - projected, axis=1)
    return t, deviation


def build_genotype_table(
    subsets: Sequence[Tuple[Mutation, ...]],
    coordinates: Dict[str, np.ndarray],
    h3_map: Dict[int, str],
) -> pd.DataFrame:
    """Per-genotype coordinates plus their placement on the J -> J.2.4 axis."""
    gids = [genotype_id(s) for s in subsets]
    points = np.array([coordinates[g] for g in gids])
    start = coordinates["root"]
    end = coordinates[genotype_id(subsets[-1])]

    t, deviation = axis_projection(points, start, end)
    return pd.DataFrame(
        {
            "genotype_id": gids,
            "genotype_h3": [genotype_id(s, h3_map) for s in subsets],
            "n_fixed": [len(s) for s in subsets],
            "X": points[:, 0],
            "Y": points[:, 1],
            "Z": points[:, 2],
            "dist_to_start": np.linalg.norm(points - start, axis=1),
            "dist_to_end": np.linalg.norm(points - end, axis=1),
            "axis_progress": t,
            "axis_deviation": deviation,
        }
    )


def build_path_tables(
    orderings: Sequence[Tuple[Mutation, ...]],
    coordinates: Dict[str, np.ndarray],
    genotypes: pd.DataFrame,
    h3_map: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-step and per-path geometry for every ordering."""
    by_gid = genotypes.set_index("genotype_id")
    start = coordinates["root"]
    end_gid = genotypes.loc[genotypes["n_fixed"].idxmax(), "genotype_id"]
    end = coordinates[end_gid]
    net_displacement = float(np.linalg.norm(end - start))

    step_rows = []
    summary_rows = []

    for index, ordering in enumerate(orderings):
        pid = f"path{index + 1:03d}"
        nodes = path_genotypes(ordering)
        cumulative = 0.0
        lengths = []

        for step, focal in enumerate(ordering, start=1):
            source = genotype_id(nodes[step - 1])
            target = genotype_id(nodes[step])
            vector = coordinates[target] - coordinates[source]
            length = float(np.linalg.norm(vector))
            cumulative += length
            lengths.append(length)

            step_rows.append(
                {
                    "path_id": pid,
                    "order_h3": order_id(ordering, h3_map),
                    "step": step,
                    "from_genotype_h3": genotype_id(nodes[step - 1], h3_map),
                    "to_genotype_h3": genotype_id(nodes[step], h3_map),
                    "focal_mutation_h3": focal.label(h3_map),
                    "dX": float(vector[0]),
                    "dY": float(vector[1]),
                    "dZ": float(vector[2]),
                    "step_length": length,
                    "cumulative_length": cumulative,
                    "dist_to_end_after_step": float(by_gid.loc[target, "dist_to_end"]),
                    "axis_progress_after_step": float(by_gid.loc[target, "axis_progress"]),
                    "axis_deviation_after_step": float(by_gid.loc[target, "axis_deviation"]),
                }
            )

        progress = [0.0] + [float(by_gid.loc[genotype_id(n), "axis_progress"]) for n in nodes[1:]]
        deviations = [float(by_gid.loc[genotype_id(n), "axis_deviation"]) for n in nodes]
        distances = [float(by_gid.loc[genotype_id(n), "dist_to_end"]) for n in nodes]
        progress_steps = np.diff(progress)

        summary_rows.append(
            {
                "path_id": pid,
                "order_h3": order_id(ordering, h3_map),
                "total_path_length": cumulative,
                "net_displacement": net_displacement,
                # 1.0 means the path runs straight from J to J.2.4; larger means
                # the intermediates detour through space they need not visit.
                "tortuosity": cumulative / net_displacement if net_displacement else np.nan,
                "longest_step": max(lengths),
                "longest_step_mutation_h3": ordering[int(np.argmax(lengths))].label(h3_map),
                "shortest_step": min(lengths),
                "max_axis_deviation": max(deviations),
                "min_axis_progress_step": float(progress_steps.min()),
                "backtracks_along_axis": bool(progress_steps.min() < 0),
                "max_dist_to_end_after_start": max(distances[1:]),
                # distances[0] is the root's distance to the endpoint, so this flags
                # orderings whose intermediates sit further from J.2.4 than J itself.
                "moves_away_from_end": bool(
                    len(distances) > 2 and max(distances[1:-1]) > distances[0]
                ),
            }
        )

    steps = pd.DataFrame(step_rows)
    summary = pd.DataFrame(summary_rows)
    summary["rank_by_path_length"] = summary["total_path_length"].rank(method="min").astype(int)
    summary = summary.sort_values("rank_by_path_length").reset_index(drop=True)
    return steps, summary


###############################################################################
# Figures
###############################################################################
def plot_interactive_3d(
    genotypes: pd.DataFrame,
    orderings: Sequence[Tuple[Mutation, ...]],
    coordinates: Dict[str, np.ndarray],
    observed: Optional[pd.DataFrame],
    background: Optional[pd.DataFrame],
    h3_map: Dict[int, str],
    path: Path,
) -> None:
    import plotly.graph_objects as go

    figure = go.Figure()

    if background is not None and not background.empty:
        figure.add_trace(
            go.Scatter3d(
                x=background["X"], y=background["Y"], z=background["Z"],
                mode="markers",
                marker=dict(size=2, opacity=0.25, symbol="diamond-open", color="#bbbbbb"),
                name="PLANT background",
                hoverinfo="skip",
            )
        )

    # One faint polyline per ordering. They overlap wherever two orderings share a
    # prefix, so the visible spread is exactly the order-dependence.
    for index, ordering in enumerate(orderings):
        nodes = [coordinates[genotype_id(n)] for n in path_genotypes(ordering)]
        arr = np.array(nodes)
        figure.add_trace(
            go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2],
                mode="lines",
                line=dict(width=3, color="rgba(60,90,160,0.35)"),
                name=order_id(ordering, h3_map),
                legendgroup="orderings",
                showlegend=index == 0,
                hovertemplate=f"{order_id(ordering, h3_map)}<extra></extra>",
            )
        )

    figure.add_trace(
        go.Scatter3d(
            x=genotypes["X"], y=genotypes["Y"], z=genotypes["Z"],
            mode="markers+text",
            marker=dict(
                size=6 + 3 * genotypes["n_fixed"],
                color=genotypes["n_fixed"],
                colorscale="Viridis",
                colorbar=dict(title="mutations fixed"),
                opacity=0.95,
            ),
            text=genotypes["genotype_h3"],
            textposition="top center",
            textfont=dict(size=9),
            name="genotypes",
            hovertemplate=(
                "%{text}<br>X %{x:.3f} Y %{y:.3f} Z %{z:.3f}<extra>genotype</extra>"
            ),
        )
    )

    if observed is not None and not observed.empty:
        figure.add_trace(
            go.Scatter3d(
                x=observed["X"], y=observed["Y"], z=observed["Z"],
                mode="markers",
                marker=dict(size=9, symbol="diamond", color="#e41a1c", opacity=0.9),
                text=observed["sequence_id"],
                name="observed sequences",
                hovertemplate="%{text}<extra>observed</extra>",
            )
        )

    figure.update_layout(
        title="PLANT 3D trajectories for every mutation ordering",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        height=1000, width=1200,
    )
    figure.write_html(path)


def plot_path_summary(steps: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5.0, 0.32 * len(summary) + 3)))

    ranked = summary.sort_values("total_path_length")
    y = np.arange(len(ranked))[::-1]
    axes[0].barh(y, ranked["total_path_length"], color="#4c72b0")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ranked["order_h3"], fontsize=8)
    axes[0].axvline(float(ranked["net_displacement"].iloc[0]), color="crimson",
                    linestyle="--", linewidth=1.2,
                    label="straight-line J → J.2.4 distance")
    axes[0].set_xlabel("Total PLANT path length")
    axes[0].set_title("Trajectory length by mutation order")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.3)

    # Step 0 is the J root, whose distance to the endpoint is the straight-line
    # J -> J.2.4 distance, shared by every path.
    root_distance = float(summary["net_displacement"].iloc[0])
    for _, group in steps.groupby("path_id"):
        group = group.sort_values("step")
        axes[1].plot(
            np.concatenate([[0], group["step"].values]),
            np.concatenate([[root_distance], group["dist_to_end_after_step"].values]),
            marker="o", markersize=3, linewidth=1.0, alpha=0.5,
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance to J.2.4 in PLANT space")
    axes[1].set_title("Approach to the endpoint, one line per ordering")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


###############################################################################
# Immune escape: single mutants and pairs
###############################################################################
# Validated categorical/sequential steps (see scripts/JtoJ24_scan/README.md).
# Kept as literals so these figures need only matplotlib -- no seaborn -- and so
# they draw identically from either conda env.
BLUE_FILL = "#9ec5f4"     # sequential blue 200: total displacement
BLUE_INK = "#1c5cab"      # sequential blue 550: on-axis component
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal, n_fixed 0..4
DIVERGING_LOW = "#2a78d6"   # negative epistasis (blue), matching the PLM heatmaps' vlag
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#d03b3b"  # positive epistasis (red)
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def split_genotype_label(label: str) -> Tuple[str, ...]:
    """Mutation labels making up a genotype name; ``root`` is the empty set."""
    return () if label == "root" else tuple(label.split("+"))


def escape_basis(genotypes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    """Root coordinates, the unit vector root -> full mutant, and its length.

    Every escape number below is expressed in this frame: a signed advance along
    the start -> end axis plus an unsigned off-axis residual. The axis component
    is the additive one -- it is a linear functional of the coordinates, so
    "pair minus the two singles" is a meaningful epistasis term on it, which is
    not true of the raw Euclidean displacement.
    """
    coordinates = genotypes[["X", "Y", "Z"]].to_numpy(float)
    counts = genotypes["n_fixed"].to_numpy()
    root = coordinates[int(np.argmin(counts))]
    end = coordinates[int(np.argmax(counts))]
    axis = end - root
    span = float(np.linalg.norm(axis))
    if span == 0:
        raise ValueError(
            "The full mutant embeds on top of the start lineage, so there is no "
            "start -> end axis to project escape onto."
        )
    return root, axis / span, span


def build_escape_tables(genotypes: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Per-mutation and per-pair antigenic escape, plus the pairwise epistasis.

    ``singles`` is always returned. ``pairs`` is ``None`` when the run did not
    embed two-mutation backgrounds (``--max-background-size 1``), because the
    pair terms are measurements, not extrapolations -- there is nothing to draw.
    """
    root, unit, span = escape_basis(genotypes)
    by_label = {
        row["genotype_h3"]: np.array([row["X"], row["Y"], row["Z"]], dtype=float)
        for _, row in genotypes.iterrows()
    }
    offsets = {label: point - root for label, point in by_label.items()}

    full_label = genotypes.loc[genotypes["n_fixed"].idxmax(), "genotype_h3"]
    names = list(split_genotype_label(full_label))

    def components(vector: np.ndarray) -> Tuple[float, float, float]:
        along = float(vector @ unit)
        return float(np.linalg.norm(vector)), along, float(np.linalg.norm(vector - along * unit))

    single_rows = []
    for name in names:
        if name not in offsets:
            continue
        total, along, off = components(offsets[name])
        single_rows.append(
            {
                "mutation_h3": name,
                "escape_total": total,
                "escape_along_axis": along,
                "escape_off_axis": off,
                # What share of the whole start -> end antigenic move this one
                # mutation delivers on its own.
                "frac_of_endpoint": along / span,
                "off_axis_fraction": off / total if total else np.nan,
            }
        )
    singles = pd.DataFrame(single_rows)
    if singles.empty:
        raise ValueError("No single-mutant genotypes found; cannot build escape tables.")

    single_offset = {name: offsets[name] for name in singles["mutation_h3"]}

    pair_rows = []
    for i, first in enumerate(singles["mutation_h3"]):
        for second in singles["mutation_h3"][i + 1:]:
            label = f"{first}+{second}"
            if label not in offsets:
                continue
            observed = offsets[label]
            additive = single_offset[first] + single_offset[second]
            epistasis = observed - additive

            obs_total, obs_along, obs_off = components(observed)
            add_total, add_along, add_off = components(additive)
            eps_total, eps_along, eps_off = components(epistasis)

            pair_rows.append(
                {
                    "mutation_a": first,
                    "mutation_b": second,
                    "pair_h3": label,
                    "escape_total": obs_total,
                    "escape_along_axis": obs_along,
                    "escape_off_axis": obs_off,
                    "additive_total": add_total,
                    "additive_along_axis": add_along,
                    "additive_off_axis": add_off,
                    # Signed, additive, and the number the matrix plots.
                    "epistasis_along_axis": eps_along,
                    # Unsigned size of the full 3D departure from additivity --
                    # a pair can be perfectly additive along the axis and still
                    # land somewhere else entirely.
                    "epistasis_magnitude": eps_total,
                    "epistasis_off_axis": eps_off,
                    "relative_epistasis": eps_along / add_along if add_along else np.nan,
                    "frac_of_endpoint": obs_along / span,
                }
            )

    pairs = pd.DataFrame(pair_rows) if pair_rows else None
    return singles, pairs


def ramp_colour(level: int, top_level: int) -> str:
    """Step on the ordinal blue ramp for ``level`` of ``top_level`` mutations.

    Sampled rather than indexed so a set of 11 mutations still spans the whole
    ramp instead of piling every level past the fourth onto the darkest step.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    ramp = LinearSegmentedColormap.from_list("escape_ordinal", BLUE_RAMP)
    return to_hex(ramp(level / top_level if top_level else 0.0))


def diverging_cmap():
    """blue -> neutral -> red, oriented like the PLM heatmaps' ``vlag``."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "escape_diverging", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH]
    )


def _tidy(axis) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=INK_MUTED, labelsize=9)
    axis.xaxis.label.set_color(INK_MUTED)
    axis.yaxis.label.set_color(INK_MUTED)


def plot_escape_singles_pairs(
    singles: pd.DataFrame,
    pairs: Optional[pd.DataFrame],
    path: Path,
    start_label: str = "start",
    end_label: str = "end",
) -> None:
    """Per-mutation escape on top, per-pair observed-vs-additive below.

    The two panels share an x axis so a pair's escape can be read directly
    against the singles that make it up.
    """
    n_single = len(singles)
    n_pair = 0 if pairs is None else len(pairs)
    heights = [max(2.2, 0.42 * n_single)] + ([max(2.4, 0.38 * n_pair)] if n_pair else [])

    figure, axes = plt.subplots(
        len(heights), 1, figsize=(11, sum(heights) + 1.8),
        gridspec_kw={"height_ratios": heights}, sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    ranked = singles.sort_values("escape_total")
    y = np.arange(n_single)
    top = axes[0]
    # Bullet layout: the wide pale bar is the whole antigenic move, the narrow
    # dark bar the part of it aimed at the endpoint. The visible remainder is
    # the off-axis component, which is escape the endpoint does not explain.
    top.barh(y, ranked["escape_total"], height=0.62, color=BLUE_FILL,
             label="total antigenic displacement  |Δ|")
    top.barh(y, ranked["escape_along_axis"], height=0.30, color=BLUE_INK,
             edgecolor="white", linewidth=0.8,
             label=f"advance along {start_label} → {end_label}")
    for index, row in enumerate(ranked.itertuples()):
        top.text(row.escape_total + 0.012, index, f"{row.escape_total:.3f}",
                 va="center", fontsize=8.5, color=INK)
        top.text(row.escape_along_axis + 0.012, index - 0.30,
                 f"{row.frac_of_endpoint:+.0%} of {end_label}",
                 va="center", fontsize=7.5, color=INK_MUTED)
    top.set_yticks(y)
    top.set_yticklabels(ranked["mutation_h3"], fontsize=10, color=INK)
    top.set_title(f"Single-mutant immune escape from {start_label} (PLANT antigenic space)",
                  fontsize=12, color=INK, loc="left")
    top.axvline(0, color=INK_MUTED, linewidth=0.8)
    top.grid(axis="x", alpha=0.25, linewidth=0.6)
    top.set_axisbelow(True)
    top.legend(fontsize=8.5, frameon=False, loc="lower right")
    _tidy(top)

    if n_pair:
        bottom = axes[1]
        ordered = pairs.sort_values("escape_along_axis")
        y = np.arange(n_pair)
        for index, row in enumerate(ordered.itertuples()):
            colour = DIVERGING_HIGH if row.epistasis_along_axis >= 0 else DIVERGING_LOW
            bottom.plot(
                [row.additive_along_axis, row.escape_along_axis], [index, index],
                color=colour, linewidth=2.6, solid_capstyle="round", zorder=2,
            )
            bottom.plot(row.additive_along_axis, index, marker="o", markersize=8,
                        markerfacecolor="white", markeredgecolor=INK_MUTED,
                        markeredgewidth=1.4, zorder=3)
            bottom.plot(row.escape_along_axis, index, marker="o", markersize=9,
                        color=BLUE_INK, markeredgecolor="white", markeredgewidth=1.2,
                        zorder=4)
            bottom.text(max(row.escape_along_axis, row.additive_along_axis) + 0.012, index,
                        f"ε {row.epistasis_along_axis:+.3f}", va="center",
                        fontsize=8, color=colour)
        bottom.set_yticks(y)
        bottom.set_yticklabels(ordered["pair_h3"], fontsize=9.5, color=INK)
        bottom.set_title(
            "Pairwise escape: observed vs the sum of the two single mutants",
            fontsize=12, color=INK, loc="left",
        )
        bottom.axvline(0, color=INK_MUTED, linewidth=0.8)
        bottom.grid(axis="x", alpha=0.25, linewidth=0.6)
        bottom.set_axisbelow(True)
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                       markerfacecolor="white", markeredgecolor=INK_MUTED,
                       label="additive expectation (single a + single b)"),
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=BLUE_INK,
                       label="observed double mutant"),
            plt.Line2D([], [], color=DIVERGING_HIGH, linewidth=2.6,
                       label="ε > 0: more escape than additive"),
            plt.Line2D([], [], color=DIVERGING_LOW, linewidth=2.6,
                       label="ε < 0: less escape than additive"),
        ]
        bottom.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
        _tidy(bottom)

    axes[-1].set_xlabel(f"Antigenic escape from {start_label} (PLANT units)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_escape_epistasis_matrix(
    singles: pd.DataFrame,
    pairs: pd.DataFrame,
    path: Path,
    start_label: str = "start",
    end_label: str = "end",
) -> None:
    """Symmetric ε matrix; each single mutant's own escape sits in its label."""
    names = list(singles.sort_values("escape_total", ascending=False)["mutation_h3"])
    index = {name: i for i, name in enumerate(names)}
    matrix = np.full((len(names), len(names)), np.nan)
    for row in pairs.itertuples():
        i, j = index[row.mutation_a], index[row.mutation_b]
        matrix[i, j] = matrix[j, i] = row.epistasis_along_axis

    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    limit = limit if limit > 0 else 1.0

    size = max(4.6, 0.85 * len(names) + 3.2)
    figure, axis = plt.subplots(figsize=(size + 1.6, size))
    image = axis.imshow(matrix, cmap=diverging_cmap(), vmin=-limit, vmax=limit)

    escape = singles.set_index("mutation_h3")["escape_along_axis"]
    labels = [f"{name}\n({escape[name]:+.3f})" for name in names]
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
                # Ink stays neutral; the cell fill carries sign and size.
                axis.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                          color=INK, fontsize=9)

    axis.set_title(
        f"Pairwise epistasis in escape along the {start_label} → {end_label} axis\n"
        "ε = double mutant − (single a + single b), PLANT units; "
        "single-mutant escape in brackets",
        fontsize=11, color=INK,
    )
    bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    bar.set_label("ε (PLANT units)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED)
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def place_labels(axis, entries: Sequence[Tuple[float, float, str, dict]]) -> None:
    """Annotate points, nudging each label to the first free slot around it.

    A scatter of genotype names collides badly as soon as two genotypes embed
    close together, which is exactly the interesting case here. Boxes are
    estimated in display space from the font metrics, and the first candidate
    offset that clears everything already placed wins; failing that the label
    still goes down at its first choice rather than being dropped silently.

    The caller must have drawn the figure once already, because the offsets are
    resolved against the final data limits.
    """
    import matplotlib.patheffects as path_effects

    halo = [path_effects.withStroke(linewidth=2.6, foreground="white")]
    candidates = [(0, 11), (0, -17), (13, 4), (-13, 4), (0, 21), (0, -27), (26, -10), (-26, -10)]
    placed: List[Tuple[float, float, float, float]] = []

    def overlaps(box, other) -> bool:
        return not (box[2] <= other[0] or other[2] <= box[0]
                    or box[3] <= other[1] or other[3] <= box[1])

    for x, y, name, style in entries:
        anchor = axis.transData.transform((x, y))
        fontsize = style.get("fontsize", 8.5)
        width = 0.58 * fontsize * len(name)
        height = 1.35 * fontsize
        chosen = candidates[0]
        for offset in candidates:
            left = anchor[0] + offset[0] - width / 2
            bottom = anchor[1] + offset[1] - height / 2
            box = (left, bottom, left + width, bottom + height)
            if not any(overlaps(box, other) for other in placed):
                chosen = offset
                break
        left = anchor[0] + chosen[0] - width / 2
        bottom = anchor[1] + chosen[1] - height / 2
        placed.append((left, bottom, left + width, bottom + height))
        text_kwargs = {"color": INK, "fontsize": 8.5}
        text_kwargs.update(style)
        axis.annotate(name, (x, y), textcoords="offset points", xytext=chosen,
                      ha="center", va="center", path_effects=halo, zorder=6,
                      **text_kwargs)


def plot_escape_map(
    genotypes: pd.DataFrame,
    pairs: Optional[pd.DataFrame],
    observed: Optional[pd.DataFrame],
    path: Path,
    start_label: str = "start",
    end_label: str = "end",
) -> None:
    """2D view of PLANT space in the escape frame.

    x is the advance along start -> end; y is the single off-axis direction that
    carries the most spread among the genotypes, found by SVD on the residuals.
    That keeps the informative plane rather than an arbitrary pair of PLANT's
    own axes, and it makes the additive-vs-observed gap of each pair visible as
    a line segment.
    """
    root, unit, span = escape_basis(genotypes)
    coordinates = genotypes[["X", "Y", "Z"]].to_numpy(float)
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
    off = offsets @ perpendicular

    frame = genotypes.assign(escape_along=along, escape_off=off)
    figure, axis = plt.subplots(figsize=(11, 8))

    axis.axhline(0, color="#d8d7d2", linewidth=1.0, zorder=0)
    axis.plot([0, span], [0, 0], color="#b9b8b2", linewidth=1.4, linestyle="--", zorder=1)

    if pairs is not None:
        # Segment from where additivity predicts the double mutant to where PLANT
        # actually puts it: the epistasis, drawn to scale. Deliberately a plain
        # segment rather than an arrow -- most of these gaps are only a few
        # pixels long, and an arrowhead at that size points wherever it likes.
        lookup = frame.set_index("genotype_h3")
        for row in pairs.itertuples():
            a, b = lookup.loc[row.mutation_a], lookup.loc[row.mutation_b]
            predicted = (a.escape_along + b.escape_along, a.escape_off + b.escape_off)
            actual = lookup.loc[row.pair_h3]
            axis.plot([predicted[0], actual.escape_along], [predicted[1], actual.escape_off],
                      color=INK_MUTED, linewidth=1.1, alpha=0.85, zorder=2)
            axis.plot(*predicted, marker="x", markersize=7, color=INK_MUTED,
                      markeredgewidth=1.4, zorder=3)

    label_entries = []
    if observed is not None and not observed.empty:
        observed_offsets = observed[["X", "Y", "Z"]].to_numpy(float) - root
        observed_x = observed_offsets @ unit
        observed_y = observed_offsets @ perpendicular
        axis.scatter(observed_x, observed_y, marker="D", s=55, facecolor="none",
                     edgecolor=INK_MUTED, linewidth=1.2, zorder=3,
                     label="observed sequences")
        seen = set()
        names = observed["lineage"] if "lineage" in observed else observed["sequence_id"]
        for name, x, y in zip(names, observed_x, observed_y):
            if name in seen:
                continue
            seen.add(name)
            label_entries.append(
                (x, y, str(name), {"fontsize": 8, "color": INK_MUTED, "style": "italic"})
            )

    levels = sorted(int(level) for level in frame["n_fixed"].unique())
    top_level = int(frame["n_fixed"].max())
    for level in levels:
        subset = frame[frame["n_fixed"] == level]
        axis.scatter(subset["escape_along"], subset["escape_off"],
                     s=42 + 26 * (4 * level / top_level if top_level else 0),
                     color=ramp_colour(level, top_level), edgecolor="white",
                     linewidth=1.2, zorder=4,
                     label=f"{level} mutation{'s' if level != 1 else ''} fixed")

    # This figure is about the singles and the pairs; higher-order genotypes are
    # context. Labelling them too is what makes the panel unreadable -- and past
    # a dozen or so pairs even the pairs have to go, or the labels win over the
    # points they name.
    label_pairs = pairs is not None and len(pairs) <= 12
    keep = (frame["n_fixed"] <= (2 if label_pairs else 1)) | (frame["n_fixed"] == top_level)
    for row in frame[keep].itertuples():
        name = start_label if row.genotype_h3 == "root" else row.genotype_h3
        weight = "bold" if row.n_fixed in (0, top_level) else "normal"
        label_entries.append(
            (row.escape_along, row.escape_off, name, {"fontsize": 8.5, "weight": weight})
        )

    axis.set_xlabel(f"Escape along the {start_label} → {end_label} axis (PLANT units)",
                    fontsize=10)
    axis.set_ylabel("Principal off-axis escape (PLANT units)", fontsize=10)
    axis.set_title(
        f"Where each mutation and pair moves {start_label} in PLANT antigenic space\n"
        "× = the pair's additive prediction, joined to where PLANT actually puts it",
        fontsize=12, color=INK, loc="left",
    )
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.set_axisbelow(True)
    # Equal aspect: this is a distance map, so a unit sideways has to look like a
    # unit forwards or the off-axis escape reads as bigger than it is.
    axis.set_aspect("equal", adjustable="datalim")

    handles, _ = axis.get_legend_handles_labels()
    if pairs is not None:
        handles.append(plt.Line2D([], [], marker="x", linestyle="-", color=INK_MUTED,
                                  markersize=7, markeredgewidth=1.4,
                                  label="additive prediction → observed"))
    axis.legend(handles=handles, fontsize=8.5, frameon=False, loc="best")
    _tidy(axis)

    figure.tight_layout()
    # Labels are placed in display space, so the data limits (which equal aspect
    # only settles at draw time) must already be final.
    figure.canvas.draw()
    place_labels(axis, label_entries)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def write_escape_outputs(
    genotypes: pd.DataFrame,
    observed: Optional[pd.DataFrame],
    output_dir: Path,
    start_label: str,
    end_label: str,
    draw: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Build the escape tables, write them, and draw the three escape figures."""
    singles, pairs = build_escape_tables(genotypes)
    singles.to_csv(output_dir / "single_mutation_escape.csv", index=False)
    if pairs is not None:
        pairs.to_csv(output_dir / "pairwise_escape.csv", index=False)

    if draw:
        plot_escape_singles_pairs(singles, pairs,
                                  output_dir / "plant_escape_singles_pairs.png",
                                  start_label, end_label)
        plot_escape_map(genotypes, pairs, observed,
                        output_dir / "plant_escape_map.png", start_label, end_label)
        if pairs is not None:
            plot_escape_epistasis_matrix(singles, pairs,
                                         output_dir / "plant_escape_epistasis_matrix.png",
                                         start_label, end_label)
    return singles, pairs


def lineage_label(header: str) -> str:
    """Trailing lineage field of a pipe-delimited FASTA header."""
    return header.rsplit("|", 1)[-1] if "|" in header else header


###############################################################################
# Main
###############################################################################
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.ha2_start is not None and args.ha2_start < 0:
        args.ha2_start = None

    start_header, start_seq, end_header, end_seq, mutations = resolve_mutation_set(args)
    h3_map = build_h3_map(start_seq, args.h3_reference, args.signal_peptide_length, args.ha2_start)

    # PLANT must be importable before the mutation set is final: deciding which
    # mutations it can actually see needs its own reference window.
    plant_batch = import_plant_batch(args)

    dropped: List[Mutation] = []
    if args.restrict_to_window:
        mutations, dropped = partition_by_window(start_seq, mutations, plant_batch)
        if dropped:
            print(
                f"--restrict-to-window: dropping {len(dropped)} mutation(s) outside PLANT's "
                f"{plant_batch.TARGET_LENGTH}-residue HA1 window: "
                + ", ".join(f"{m.name} [{m.label(h3_map)}]" for m in dropped)
            )
        if not mutations:
            raise ValueError("Every mutation falls outside PLANT's HA1 window; nothing to scan.")

    print(describe_mutation_set(mutations, h3_map, start_header, end_header))

    # As in the PLM scan, capping background size keeps a large mutation set
    # simple. Trajectories walk every background size, so they are skipped when
    # capped -- but the full mutant is still embedded, because it defines the
    # J -> endpoint axis that the per-mutation displacements are measured on.
    pairwise_only = args.max_background_size is not None
    full_set = tuple(sorted(mutations))
    subsets = list(all_subsets(mutations, args.max_background_size))
    if pairwise_only:
        if subsets[-1] != full_set:
            subsets.append(full_set)
        orderings = []
        print(
            f"Embedding {len(subsets)} genotypes (backgrounds of at most "
            f"{args.max_background_size} mutation(s), plus the full mutant as the axis endpoint; "
            f"the full hypercube would be {2 ** len(mutations)}).\n"
            "Trajectory/path tables need every background size, so they are SKIPPED. "
            "Per-genotype coordinates and per-mutation displacements are complete.\n"
            "Drop --max-background-size to get the full hypercube and the ordering trajectories."
        )
    else:
        orderings = all_orderings(mutations, args.max_orders, args.seed)
        print(f"Embedding {len(subsets)} genotypes and reconstructing {len(orderings)} orderings.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genotype_sequences = {
        genotype_id(subset): apply_mutations(start_seq, subset) for subset in subsets
    }
    windowed = {
        gid: project_to_plant_window(sequence, plant_batch)
        for gid, sequence in genotype_sequences.items()
    }

    full_gid = genotype_id(subsets[-1])
    window_positions = verify_window_mutations(windowed["root"], windowed[full_gid], mutations)
    print("All mutations present inside PLANT's HA1 window:")
    for mutation in mutations:
        window = window_positions[mutation.pos]
        print(f"  {mutation.name} [{mutation.label(h3_map)}] -> window position {window.pos}")

    write_fasta(output_dir / "genotypes_plant_window.fasta", list(windowed.items()))

    metadata = {
        "start_header": start_header,
        "end_header": end_header,
        "mutations": [m.name for m in mutations],
        "mutations_h3": [m.label(h3_map) for m in mutations],
        "window_positions": {m.name: window_positions[m.pos].pos for m in mutations},
        "dropped_outside_window": [m.name for m in dropped],
        "dropped_outside_window_h3": [m.label(h3_map) for m in dropped],
        "n_genotypes": len(subsets),
        "n_genotypes_full_hypercube": 2 ** len(mutations),
        "max_background_size": args.max_background_size,
        "pairwise_only": pairwise_only,
        "n_orderings": len(orderings),
        "plant_model_dir": str(args.plant_model_dir),
        "plant_subfolder": args.plant_subfolder,
        "plant_base_model": args.plant_base_model,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.dry_run:
        print(f"Dry run complete. Enumeration and window checks written to {output_dir}")
        return 0

    print("Loading PLANT runtime...")
    tokenizer, model, device, use_fp16 = plant_batch.load_plant_runtime()
    print(f"PLANT loaded on {device} (fp16={use_fp16}).")

    genotype_frame = pd.DataFrame(
        {"sequence_id": list(windowed), "seq": [windowed[g] for g in windowed]}
    )
    embedded = plant_batch.embed_dataframe(
        genotype_frame, tokenizer, model, use_fp16, args.batch_size
    )
    if len(embedded) != len(genotype_frame):
        raise RuntimeError(
            f"PLANT returned {len(embedded)} embeddings for {len(genotype_frame)} genotypes."
        )
    coordinates = {
        row["sequence_id"]: np.array([row["X"], row["Y"], row["Z"]], dtype=float)
        for _, row in embedded.iterrows()
    }

    observed = None
    if not args.no_observed:
        records = read_fasta(args.fasta)
        observed_frame = pd.DataFrame(
            {
                "sequence_id": list(records),
                "seq": [project_to_plant_window(seq, plant_batch) for seq in records.values()],
            }
        )
        observed = plant_batch.embed_dataframe(
            observed_frame, tokenizer, model, use_fp16, args.batch_size
        )
        observed["lineage"] = observed["sequence_id"].str.rsplit("|", n=1).str[-1]
        observed.to_csv(output_dir / "observed_sequence_embeddings.csv", index=False)

    background = None
    if args.background_csv is not None and str(args.background_csv).lower() != "none":
        if Path(args.background_csv).exists():
            background = pd.read_csv(args.background_csv)
        else:
            print(f"[warning] Background CSV not found at {args.background_csv}; skipping.")

    print("Assembling tables...")
    genotypes = build_genotype_table(subsets, coordinates, h3_map)
    genotypes.to_csv(output_dir / "genotype_embeddings.csv", index=False)

    steps = summary = None
    if orderings:
        steps, summary = build_path_tables(orderings, coordinates, genotypes, h3_map)
        steps.to_csv(output_dir / "order_paths_steps.csv", index=False)
        summary.to_csv(output_dir / "order_paths_summary.csv", index=False)

    start_label = lineage_label(start_header)
    end_label = lineage_label(end_header)
    escape_singles, escape_pairs = write_escape_outputs(
        genotypes, observed, output_dir, start_label, end_label, draw=not args.no_plots
    )

    if not args.no_plots:
        print("Drawing figures...")
        plot_interactive_3d(genotypes, orderings, coordinates, observed, background, h3_map,
                            output_dir / "plant_order_paths_3d.html")
        if summary is not None:
            plot_path_summary(steps, summary, output_dir / "plant_order_paths_summary.png")

    print(f"\nSingle-mutant escape from {start_label} (PLANT units):")
    for row in escape_singles.sort_values("escape_total", ascending=False).itertuples():
        print(f"  {row.mutation_h3:>12}  |Δ|={row.escape_total:.4f}  "
              f"along axis={row.escape_along_axis:+.4f} ({row.frac_of_endpoint:+.1%} of "
              f"{start_label} → {end_label})  off-axis={row.escape_off_axis:.4f}")
    if escape_pairs is not None:
        strongest = escape_pairs.reindex(
            escape_pairs["epistasis_along_axis"].abs().sort_values(ascending=False).index
        )
        print("Pairwise epistasis in escape, largest departures from additivity first:")
        for row in strongest.head(5).itertuples():
            print(f"  {row.pair_h3:>26}  observed={row.escape_along_axis:+.4f}  "
                  f"additive={row.additive_along_axis:+.4f}  "
                  f"ε={row.epistasis_along_axis:+.4f} ({row.relative_epistasis:+.1%})")
    else:
        print("Pairwise escape skipped: no two-mutation backgrounds were embedded "
              "(pass --max-background-size 2 or drop the flag).")

    if summary is not None:
        spread = summary["total_path_length"]
        print("\nPath length across orderings: "
              f"min {spread.min():.4f}, max {spread.max():.4f}, "
              f"straight-line start → end {summary['net_displacement'].iloc[0]:.4f}")
        print("Most direct orderings:")
        for _, row in summary.head(5).iterrows():
            print(f"  {row['rank_by_path_length']:>2}. {row['order_h3']}  "
                  f"length={row['total_path_length']:.4f}  "
                  f"tortuosity={row['tortuosity']:.3f}  "
                  f"max deviation={row['max_axis_deviation']:.4f}")
    else:
        singles = genotypes[genotypes["n_fixed"] == 1].sort_values("dist_to_start", ascending=False)
        print("\nPer-mutation displacement from the start lineage in PLANT space:")
        for _, row in singles.iterrows():
            print(f"  {row['genotype_h3']:>12}  |Δ|={row['dist_to_start']:.4f}  "
                  f"axis progress={row['axis_progress']:+.3f}  "
                  f"off-axis={row['axis_deviation']:.4f}")

    print(f"\nWrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
