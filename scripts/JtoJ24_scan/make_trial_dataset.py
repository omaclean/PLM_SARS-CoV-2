"""Build a small, fast run directory for trialling ``plant_population_escape.py``.

Two modes, for the two different questions you have when trying a new score.

``--mode subsample`` (default) — *is it fast enough, and does it look sane on my data?*
    Takes the real PLANT background cloud, caps it at ``--per-year`` sequences per
    calendar year with a fixed seed, and copies the genotype embeddings from a
    completed PLANT scan alongside it. The result is a self-contained run
    directory of a few thousand background rows instead of ~150k, so the whole
    pipeline runs in seconds and you can iterate on flags.

    The sample is uniform within each year, so a year's antigenic *composition*
    is preserved in expectation — which is what matters, because with the default
    ``--normalise-by year`` each year's total weight comes from recency and not
    from its sequence count. **With ``--normalise-by none`` the cap changes the
    answer directly**, since there the count IS the weight. Trial with the
    default; check the conclusion against the full cloud before believing it.

``--mode synthetic`` — *does the score do what it claims?*
    Writes a planted landscape with a known right answer and no real data at all.
    Four mutations, three of them at exactly the same distance from the start
    lineage, so the stationary displacement measure **cannot rank them**. The
    immunity is placed to one side, so the population measure ranks them
    immediately, and in a printed, predicted order. If the run does not reproduce
    that order, the score is wrong — no judgement about influenza required.

Example
-------
    python scripts/JtoJ24_scan/make_trial_dataset.py
    python scripts/JtoJ24_scan/make_trial_dataset.py --mode synthetic
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from order_scan_common import REPO_ROOT  # noqa: E402

DEFAULT_BACKGROUND_CSV = Path(
    "/home3/oml4h/hugging_face_downloads/PLANT_model/code/examples/backgrounds.csv"
)
DEFAULT_RUN_DIR = REPO_ROOT / "Results/JtoJ.2.4_scan/plant"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results/JtoJ.2.4_scan/plant_trial"

#: Synthetic mutation set: the first three are equidistant from the root on
#: orthogonal axes, so any measure of pure displacement scores them identically.
SYNTHETIC_OFFSETS = {
    "N122D": (1.0, 0.0, 0.0),
    "T135K": (0.0, 1.0, 0.0),
    "K189R": (0.0, 0.0, 1.0),
    "K276E": (0.2, 0.2, 0.2),
}

#: (centre, n sequences, year, subclade). The 2023 cluster sits on -X, so moving
#: along +X escapes it best; the 2021 cluster on -Y breaks the T135K/K189R tie.
SYNTHETIC_CLUSTERS = [
    ((-0.5, 0.0, 0.0), 200, 2023, "RECENT"),
    ((0.0, -0.5, 0.0), 100, 2021, "OLDER"),
    ((20.0, 20.0, 20.0), 50, 1970, "ANCIENT"),
]
SYNTHETIC_JITTER = 0.05


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("subsample", "synthetic"), default="subsample",
                        help="Real data capped per year, or a planted landscape with a "
                             "known right answer (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to build (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling/jitter seed, so the dataset is reproducible "
                             "(default: %(default)s)")

    subsample = parser.add_argument_group("subsample mode")
    subsample.add_argument("--background-csv", type=Path, default=DEFAULT_BACKGROUND_CSV,
                           help="Full PLANT background cloud (default: %(default)s)")
    subsample.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                           help="Completed PLANT scan to copy genotype embeddings from "
                                "(default: %(default)s)")
    subsample.add_argument("--per-year", type=int, default=250,
                           help="Cap on sequences kept per calendar year (default: %(default)s)")
    subsample.add_argument("--since", type=int, default=None, metavar="YEAR",
                           help="Drop everything collected before this year. The recency "
                                "weight already makes old sequences negligible, so this is "
                                "for size, not correctness (default: keep all).")
    return parser.parse_args(argv)


###############################################################################
# subsample mode
###############################################################################
def build_subsample(args: argparse.Namespace) -> Path:
    if not args.background_csv.exists():
        raise FileNotFoundError(f"Background CSV not found: {args.background_csv}")
    genotype_table = args.run_dir / "genotype_embeddings.csv"
    if not genotype_table.exists():
        raise FileNotFoundError(
            f"{genotype_table} not found. Point --run-dir at a completed PLANT scan."
        )

    backgrounds = pd.read_csv(args.background_csv)
    date_column = "collection date" if "collection date" in backgrounds else "date"
    # Year off the leading four characters: the column mixes "1968" and
    # "2006-09-17", and both start with the year.
    year = pd.to_numeric(
        backgrounds[date_column].astype(str).str.slice(0, 4), errors="coerce"
    )
    backgrounds = backgrounds[year.notna()].copy()
    backgrounds["_year"] = year[year.notna()].astype(int)

    if args.since is not None:
        backgrounds = backgrounds[backgrounds["_year"] >= args.since]
        if backgrounds.empty:
            raise ValueError(f"No background sequences collected in or after {args.since}.")

    # Sample positionally rather than with groupby.apply: apply() over the
    # grouping column is deprecated in pandas 2.2, and taking the indices keeps
    # this to one pass with no per-group frame construction.
    generator = np.random.default_rng(args.seed)
    keep: List[int] = []
    for _, block in backgrounds.groupby("_year", sort=True).indices.items():
        if len(block) <= args.per_year:
            keep.extend(block.tolist())
        else:
            keep.extend(generator.choice(block, size=args.per_year, replace=False).tolist())
    sampled = (
        backgrounds.iloc[sorted(keep)].drop(columns="_year").reset_index(drop=True)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = output_dir / "backgrounds_trial.csv"
    sampled.to_csv(trial_csv, index=False)

    shutil.copy(genotype_table, output_dir / "genotype_embeddings.csv")
    for optional in ("observed_sequence_embeddings.csv", "run_metadata.json"):
        source = args.run_dir / optional
        if source.exists():
            shutil.copy(source, output_dir / optional)

    kept_years = sorted(pd.to_numeric(
        sampled[date_column].astype(str).str.slice(0, 4), errors="coerce"
    ).dropna().astype(int).unique())
    (output_dir / "trial_dataset_metadata.json").write_text(
        json.dumps(
            {
                "mode": "subsample",
                "source_background_csv": str(args.background_csv),
                "source_run_dir": str(args.run_dir),
                "per_year_cap": args.per_year,
                "since": args.since,
                "seed": args.seed,
                "n_background_full": int(len(backgrounds)),
                "n_background_trial": int(len(sampled)),
                "years_covered": [int(kept_years[0]), int(kept_years[-1])],
            },
            indent=2,
        )
    )

    print(f"Subsampled {len(backgrounds):,} -> {len(sampled):,} background sequences "
          f"({len(kept_years)} years, cap {args.per_year}/year, seed {args.seed}).")
    print(f"Copied genotype embeddings from {args.run_dir}.")
    print(
        "\nNote: the cap is near-harmless under the default --normalise-by year, where a "
        "year's total weight comes from recency rather than from its sequence count. Under "
        "--normalise-by none the count IS the weight, so the cap changes the answer -- "
        "check any --normalise-by none conclusion against the full cloud."
    )
    return trial_csv


###############################################################################
# synthetic mode
###############################################################################
def build_synthetic(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    names = list(SYNTHETIC_OFFSETS)

    rows = []
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            offset = np.sum([SYNTHETIC_OFFSETS[name] for name in subset], axis=0) \
                if subset else np.zeros(3)
            rows.append(
                {
                    "genotype_id": "root" if not subset else "+".join(subset),
                    "genotype_h3": "root" if not subset else "+".join(subset),
                    "n_fixed": len(subset),
                    "X": float(offset[0]), "Y": float(offset[1]), "Z": float(offset[2]),
                }
            )
    genotypes = pd.DataFrame(rows)

    background_rows: List[dict] = []
    for index, (centre, count, year, subclade) in enumerate(SYNTHETIC_CLUSTERS):
        jitter = rng.normal(0.0, SYNTHETIC_JITTER, size=(count, 3))
        points = np.asarray(centre, dtype=float) + jitter
        # Spread within the year so the month-normalised mode has something to do.
        days = rng.integers(1, 366, size=count)
        for member, (point, day) in enumerate(zip(points, days)):
            stamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(day) - 1)
            background_rows.append(
                {
                    "name": f"A/Synthetic/{index}-{member}/{year}",
                    "collection date": stamp.date().isoformat(),
                    "subclade": subclade,
                    "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2]),
                }
            )
    backgrounds = pd.DataFrame(background_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    genotypes.to_csv(output_dir / "genotype_embeddings.csv", index=False)
    trial_csv = output_dir / "backgrounds_trial.csv"
    backgrounds.to_csv(trial_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["SYN1|HA|A/Synthetic/start/2022|X|J",
                            "SYN2|HA|A/Synthetic/end/2024|X|J.2.4"],
            "X": [0.0, 1.2], "Y": [0.0, 1.2], "Z": [0.0, 1.2],
            "lineage": ["J", "J.2.4"],
        }
    ).to_csv(output_dir / "observed_sequence_embeddings.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "start_header": "SYN1|HA|A/Synthetic/start/2022|SYN|J",
                "end_header": "SYN2|HA|A/Synthetic/end/2024|SYN|J.2.4",
                "mutations_h3": names,
            },
            indent=2,
        )
    )

    # The prediction, derived here rather than by running the thing under test.
    root = np.zeros(3)
    recent_centre = np.asarray(SYNTHETIC_CLUSTERS[0][0], dtype=float)
    predicted = {
        name: float(np.linalg.norm(np.asarray(offset) - recent_centre)
                    - np.linalg.norm(root - recent_centre))
        for name, offset in SYNTHETIC_OFFSETS.items()
    }
    ranked = sorted(predicted, key=predicted.get, reverse=True)

    (output_dir / "trial_dataset_metadata.json").write_text(
        json.dumps(
            {
                "mode": "synthetic",
                "seed": args.seed,
                "mutation_offsets": {k: list(v) for k, v in SYNTHETIC_OFFSETS.items()},
                "clusters": [
                    {"centre": list(centre), "n": count, "year": year, "subclade": subclade}
                    for centre, count, year, subclade in SYNTHETIC_CLUSTERS
                ],
                "n_background": int(len(backgrounds)),
                "predicted_gain_ranking": ranked,
            },
            indent=2,
        )
    )

    print(f"Planted {len(genotypes)} genotypes and {len(backgrounds)} background sequences.")
    print("\nWhat this dataset is built to show")
    print("  N122D, T135K and K189R sit at distance exactly 1.0 from the start lineage on")
    print("  three orthogonal axes, so any measure of pure displacement scores all three")
    print("  identically -- check single_mutation_escape.csv from plot_plant_escape.py and")
    print("  you will see escape_total = 1.0 for each. The 2023 immunity is centred at")
    print(f"  {SYNTHETIC_CLUSTERS[0][0]}, so population escape must separate them:")
    for position, name in enumerate(ranked, start=1):
        print(f"    {position}. {name}   (moves {predicted[name]:+.3f} units away from the "
              "recent cluster)")
    print("  Expect that order in escape_gain. Anything else means the score is wrong.")
    print("  The 1970 cluster is 34 units away and must contribute nothing after recency")
    print("  weighting -- root_population_escape should sit near, but below, 1.0.")
    return trial_csv


###############################################################################
# Main
###############################################################################
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    builder = build_subsample if args.mode == "subsample" else build_synthetic
    trial_csv = builder(args)

    python = sys.executable
    scan_dir = Path(__file__).resolve().parent
    print(f"\nTrial run directory: {args.output_dir}")
    print("\nRun it with:\n")
    print(f"  {python} \\\n"
          f"      {scan_dir / 'plant_population_escape.py'} \\\n"
          f"      {args.output_dir} \\\n"
          f"      --background-csv {trial_csv} \\\n"
          f"      --as-of 2023-01-01 --as-of 2024-01-01")
    print("\nAnd the stationary-point figures on the same genotypes, for comparison:\n")
    print(f"  {python} \\\n"
          f"      {scan_dir / 'plot_plant_escape.py'} {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
