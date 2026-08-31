"""Redraw the pairwise-context heatmap on an odds scale.

Two views, both written by default:

``pairwise_odds_with_wt_<scoring>``
    **Raw odds** P(derived)/P(ancestral) on each single-mutant background, with
    ``WT`` (the unmutated start) as the first column. Log colour scale.

``pairwise_delta_odds_<scoring>``
    **Δ odds vs WT**, on a diverging scale centred at zero and scaled only to the
    deltas. The raw-odds view's colour is dominated by which mutation a row is,
    which buries the context effect inside each row; subtracting the WT baseline
    removes that offset. WT is not plotted as a column of zeros — its value goes
    into the row labels instead.

Both differ from `pairwise_epistasis_*.png`, which works in Δ *log*-odds.

Works on any completed scan directory: reads
`site_probabilities_by_background.csv` and needs no model, so it can be pointed at
runs that finished earlier.

Example
-------
    python scripts/JtoJ24_scan/plot_pairwise_odds.py \
        Results/JtoJ.2.4_scan/epistasis Results/JtoJ.2.4.1_scan/epistasis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epistasis_order_scan import plot_delta_odds_vs_wt, plot_odds_with_wt  # noqa: E402

SITE_TABLE_NAME = "site_probabilities_by_background.csv"

# stem -> drawing function, both taking (site_table, scoring, figure_path, csv_path)
VIEWS = {
    "pairwise_odds_with_wt": plot_odds_with_wt,
    "pairwise_delta_odds": plot_delta_odds_vs_wt,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help=f"One or more scan output directories, each containing {SITE_TABLE_NAME}.",
    )
    parser.add_argument(
        "--scoring",
        nargs="+",
        default=None,
        help="Scoring schemes to draw (default: every scheme present in the table).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write all figures here instead of alongside each input table.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=sorted(VIEWS),
        default=sorted(VIEWS),
        help="Which views to draw (default: both).",
    )
    parser.add_argument(
        "--delta-odds-scale",
        choices=("auto", "linear", "symlog"),
        default="auto",
        help="Colour scale for the pairwise_delta_odds view (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    for run_dir in args.run_dirs:
        table_path = run_dir / SITE_TABLE_NAME
        if not table_path.exists():
            raise FileNotFoundError(f"{table_path} not found. Is {run_dir} a completed scan?")

        site_table = pd.read_csv(table_path)
        schemes = args.scoring or sorted(site_table["scoring"].unique())
        destination = args.output_dir or run_dir
        destination.mkdir(parents=True, exist_ok=True)

        for view in args.views:
            # Prefix figures from a shared --output-dir with the run name so two
            # runs cannot silently overwrite each other.
            stem = view if args.output_dir is None else f"{run_dir.parent.name}_{view}"
            draw = VIEWS[view]

            extra = (
                {"scale": args.delta_odds_scale} if view == "pairwise_delta_odds" else {}
            )
            for scoring in schemes:
                figure_path = destination / f"{stem}_{scoring}.png"
                csv_path = destination / f"{stem}_{scoring}.csv"
                draw(site_table, scoring, figure_path, csv_path, **extra)
                print(f"Wrote {figure_path}")
                print(f"Wrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
