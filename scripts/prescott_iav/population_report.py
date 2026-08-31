#!/usr/bin/env python3
"""What a real population prior does to PRESCOTT, measured on the J -> K substitutions.

The pipeline's default population prior for J is its parent lineage panel, G.1: 229
sequences, 61 mutant records, and exactly ONE of the eleven J -> K substitutions present. The
population term therefore cannot move the substitutions we care about, and PRESCOTT collapses
onto ESCOTT.

``build_population_frequency.py`` replaces that with every human H3N2 HA record in a
collection-date window (14,039 sequences for 2021-2023, median depth 14,037). This module
scores the difference: for each observed substitution it reads the ESCOTT probability and the
PRESCOTT probability under each frequency file, and reports the shift.

Everything is compared in LOG-probability, because that is the additive scale the softmax
lives on -- a shift of +0.94 means the model judges that substitution e^0.94 = 2.6x more
likely than ESCOTT alone did, at any starting probability.
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from prescott_iav import common  # noqa: E402
from prescott_iav.jk_impact_report import (  # noqa: E402
    AXIS, GRID, INK, INK2, MUTED, SERIES, SURFACE, apply_style, h3_label,
    ha1_number, load_probability_matrix,
)

DEFAULT_VARIANT = "PRESCOTT_eq2_c0p25_k1_parentG1"


def read_frequencies(path: Path) -> Dict[Tuple[int, str], float]:
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


def substitutions(base_protein: str, target_protein: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for index, (wt, mut) in enumerate(zip(base_protein, target_protein)):
        if wt == mut:
            continue
        position = index + 1
        number = ha1_number(position)
        out.append({
            "ha0_pos": position, "wt": wt, "mut": mut,
            "mutation": f"{wt}{position}{mut}",
            "h3_mutation": (f"{wt}{number}{mut}" if number is not None
                            else f"{wt}({h3_label(position).replace(' ', ':')}){mut}"),
        })
    return out


def fig_population_effect(table: pd.DataFrame, configs: Sequence[str], out_path: Path) -> Path:
    """Left: the shift each configuration produces. Right: why -- the observed frequency."""
    order = table.sort_values("ha0_pos", ascending=False).reset_index(drop=True)
    y = np.arange(len(order))
    height = 0.8 / len(configs)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.6, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True)

    for index, config in enumerate(configs):
        offset = (index - (len(configs) - 1) / 2) * height
        left.barh(y + offset, order[f"delta_{config}"], height=height * 0.92,
                  color=SERIES[index % len(SERIES)], edgecolor=SURFACE, linewidth=0.8,
                  label=config)
    left.axvline(0, color=AXIS, linewidth=1.2)
    left.set_yticks(y, order["h3_mutation"])
    left.set_xlabel("shift in log-probability vs ESCOTT alone\n"
                    "(positive = the population term makes it MORE likely)")
    left.set_title("What the population prior does to each J -> K substitution")
    left.spines[["top", "right"]].set_visible(False)

    # Frequencies span four orders of magnitude, so the axis has to be log; a substitution
    # absent from the file has no log and is drawn as an explicit marker at the floor.
    present = order[order["pop_frequency"] > 0]
    absent = order[order["pop_frequency"] <= 0]
    floor = float(present["pop_frequency"].min()) / 3 if len(present) else 1e-5
    right.barh(present.index, present["pop_frequency"], height=0.55, color=SERIES[2],
               edgecolor=SURFACE, linewidth=0.8)
    for idx in absent.index:
        right.plot([floor], [idx], marker="x", color=MUTED, markersize=7, markeredgewidth=1.6)
    right.set_xscale("log")
    right.set_xlim(floor / 2, 1.4)
    right.set_xlabel("frequency in the 2021-2023 population  (log scale)")
    right.set_title("Standing variation that made it possible")
    for idx, row in order.iterrows():
        if row["pop_frequency"] > 0:
            right.annotate(f"{row['pop_frequency'] * 100:.2f}%",
                           (row["pop_frequency"], idx), textcoords="offset points",
                           xytext=(5, 0), va="center", fontsize=7.4, color=INK2)
    right.spines[["top", "right"]].set_visible(False)

    # Figure-level legend below both panels: an in-axes legend has nowhere to sit here
    # without covering either the longest bar or the bottom row.
    handles, legend_labels = left.get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=len(configs), fontsize=8.5,
                  frameon=False, title="frequency prior", title_fontsize=8.5,
                  bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-lineage", default="J_int")
    parser.add_argument("--target-lineage", default="K")
    parser.add_argument("--lineage-dir", type=Path,
                        default=REPO_ROOT / "Sequences" / "IAV_lineage_files")
    parser.add_argument("--variant", default=DEFAULT_VARIANT,
                        help="PRESCOTT grid point to compare; default the least-clipped one")
    parser.add_argument("--config", action="append", default=None, metavar="LABEL=SCORES_DIR",
                        help="repeatable; default is the G.1 baseline plus the population runs")
    parser.add_argument("--frequency-file", type=Path, default=None,
                        help="population frequency file used for the right-hand panel; default "
                             "the unfiltered 2021-2023 build")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_style()
    run_dir = args.run_dir.resolve()
    key = common.safe_label(args.base_lineage)
    out_dir = common.ensure_dir(run_dir / "JtoK_report")
    figures_dir = common.ensure_dir(out_dir / "figures")
    tables_dir = common.ensure_dir(out_dir / "tables")

    if args.config:
        configs = [tuple(entry.split("=", 1)) for entry in args.config]
    else:
        configs = [
            ("G.1 parent panel (n=229)", run_dir / "scores"),
            ("2021-23 population, unfiltered", run_dir / "population/score_unfiltered/scores"),
            ("2021-23 population, reversions dropped",
             run_dir / "population/score_reversions/scores"),
        ]

    proteins = {
        name: common.load_reference_cds(args.lineage_dir / f"{name}.nt.fa", name)["protein"]
        for name in (args.base_lineage, args.target_lineage)
    }
    escott, matrix_protein = load_probability_matrix(
        run_dir / "scores" / f"{key}_ESCOTT_score_matrix.csv")
    if matrix_protein != proteins[args.base_lineage]:
        raise ValueError("the ESCOTT matrix was not built from this lineage's reference")

    frequency_path = args.frequency_file or (
        run_dir / "population/build" /
        f"{args.base_lineage}_pop2021_2023_unfiltered_frequency.txt")
    frequencies = read_frequencies(frequency_path)

    rows = substitutions(proteins[args.base_lineage], proteins[args.target_lineage])
    table = pd.DataFrame(rows)
    table["escott_logp"] = [float(np.log(escott.at[r["mut"], r["ha0_pos"]])) for r in rows]
    table["pop_frequency"] = [frequencies.get((r["ha0_pos"], r["mut"]), 0.0) for r in rows]

    labels: List[str] = []
    for label, scores_dir in configs:
        path = Path(scores_dir) / f"{key}_{args.variant}_score_matrix.csv"
        if not path.exists():
            print(f"@> skipping {label}: {path} not found")
            continue
        matrix, _ = load_probability_matrix(path)
        values = [float(np.log(matrix.at[r["mut"], r["ha0_pos"]])) for r in rows]
        table[f"logp_{label}"] = values
        table[f"delta_{label}"] = np.array(values) - table["escott_logp"].to_numpy()
        labels.append(label)

    summary = {
        "variant": args.variant,
        "frequency_file": str(frequency_path),
        "n_substitutions": len(table),
        "configs": {
            label: {
                "mean_abs_shift": float(table[f"delta_{label}"].abs().mean()),
                "n_moved": int((table[f"delta_{label}"].abs() > 1e-6).sum()),
                "n_increased": int((table[f"delta_{label}"] > 1e-6).sum()),
                "max_increase": float(table[f"delta_{label}"].max()),
            }
            for label in labels
        },
        "n_with_population_frequency": int((table["pop_frequency"] > 0).sum()),
    }

    table.to_csv(tables_dir / "population_prescott_comparison.tsv", sep="\t", index=False)
    (tables_dir / "population_prescott_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    figure = fig_population_effect(table, labels,
                                   figures_dir / "fig12_population_prior.png")

    columns = ["h3_mutation", "pop_frequency", "escott_logp"] + [f"delta_{l}" for l in labels]
    print(table[columns].round(4).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print(f"@> figure -> {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
