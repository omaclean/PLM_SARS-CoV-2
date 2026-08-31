"""Redraw the single-mutation and pairwise immune-escape figures for a PLANT scan.

Why this exists
---------------
``plant_order_paths_summary.png`` ranks the n! orderings, but every ordering ends
on the same genotype, so for a small mutation set the bars are all nearly the
same length and the distance-to-endpoint lines all collapse onto zero. The
information that *does* separate the mutations is per-mutation and per-pair, not
per-ordering:

``plant_escape_singles_pairs.png``
    Antigenic displacement from the start lineage for each single mutant, split
    into the component aimed at the endpoint and the off-axis remainder; below
    it, each double mutant's escape against the sum of its two singles.
``plant_escape_epistasis_matrix.png``
    ε = double − (single a + single b), along the start → end axis, for every
    pair. Blue is sub-additive, red super-additive, matching the PLM heatmaps.
``plant_escape_map.png``
    PLANT space rotated so x is the start → end axis and y is the off-axis
    direction carrying the most spread. Each pair's additive prediction (×) is
    joined to its observed position by an arrow, so the epistasis is drawn to
    scale rather than only tabulated.

Alongside them: ``single_mutation_escape.csv`` and ``pairwise_escape.csv``.

Reads ``genotype_embeddings.csv`` and needs no model, so it can be pointed at
runs that finished earlier — including runs made before these figures existed.

Example
-------
    python scripts/JtoJ24_scan/plot_plant_escape.py Results/JtoJ.2.4_scan/plant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plant_order_scan import lineage_label, write_escape_outputs  # noqa: E402

GENOTYPE_TABLE_NAME = "genotype_embeddings.csv"
OBSERVED_TABLE_NAME = "observed_sequence_embeddings.csv"
METADATA_NAME = "run_metadata.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help=f"One or more PLANT scan output directories, each containing "
             f"{GENOTYPE_TABLE_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write all figures here instead of alongside each input table.",
    )
    parser.add_argument(
        "--start-label",
        default=None,
        help="Name for the start lineage (default: read from run_metadata.json).",
    )
    parser.add_argument(
        "--end-label",
        default=None,
        help="Name for the endpoint lineage (default: read from run_metadata.json).",
    )
    parser.add_argument(
        "--no-observed",
        action="store_true",
        help=f"Ignore {OBSERVED_TABLE_NAME} even when the run wrote one.",
    )
    return parser.parse_args(argv)


def resolve_labels(run_dir: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Lineage names for the axis labels, from the CLI or the run's metadata."""
    start, end = args.start_label, args.end_label
    metadata_path = run_dir / METADATA_NAME
    if (start is None or end is None) and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        start = start or lineage_label(metadata.get("start_header", "start"))
        end = end or lineage_label(metadata.get("end_header", "end"))
    return start or "start", end or "end"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    for run_dir in args.run_dirs:
        table_path = run_dir / GENOTYPE_TABLE_NAME
        if not table_path.exists():
            raise FileNotFoundError(
                f"{table_path} not found. Is {run_dir} a completed PLANT scan?"
            )

        genotypes = pd.read_csv(table_path)
        observed = None
        observed_path = run_dir / OBSERVED_TABLE_NAME
        if not args.no_observed and observed_path.exists():
            observed = pd.read_csv(observed_path)

        start_label, end_label = resolve_labels(run_dir, args)
        destination = args.output_dir or run_dir
        destination.mkdir(parents=True, exist_ok=True)

        singles, pairs = write_escape_outputs(
            genotypes, observed, destination, start_label, end_label
        )
        print(f"{run_dir}: {len(singles)} single mutant(s), "
              f"{0 if pairs is None else len(pairs)} pair(s) -> {destination}")
        if pairs is None:
            print("  No two-mutation backgrounds in this run, so the pairwise figures "
                  "were skipped. Re-run the scan with --max-background-size 2 (or no "
                  "cap) to get them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
