"""Compute amino-acid proportions at canonical H3 sites from an aligned HA FASTA."""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from Functions_HuggingFace import create_h3_numbering_map


# ---- User parameters ----
FASTA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_AA.fasta"
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/canon_sites"
REFERENCE_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

HA2_START = 330
K_LINEAGE_ACCESSION = "PV511679"    # Accession of a not actually quite K lineage sequence to use for distance calculations

# Canonical sites to measure (H3 numbering; use HA2:## for HA2 sites)
CANONICAL_SITES = ["2", "144", "158", "160", "173", "189", "328", "HA2:49"]

DISTANCE_BINS = [-0.1, 0, 5, 10, 20, 50, 100, 10_000]
DISTANCE_LABELS = ["0", "1-5", "5-10", "10-20", "20-50", "50-100", "100+"]

TEST_MODE = False #True #False
TEST_SAMPLE_SIZE = 500
RANDOM_SEED = 42


def parse_alignment_records(fasta_path: str) -> List:
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        record.id = record.id.strip()
        record.description = ""
        records.append(record)
    return records


def load_reference_sequence(reference_path: str) -> str:
    ref_record = next(SeqIO.parse(reference_path, "fasta"))
    return str(ref_record.seq)


def build_ungapped_to_aligned_index(aligned_seq: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    ungapped_index = 0
    for aligned_index, aa in enumerate(aligned_seq, start=1):
        if aa == "-":
            continue
        mapping[ungapped_index] = aligned_index
        ungapped_index += 1
    return mapping


def compute_alignment_distance(seq_a: str, seq_b: str) -> int:
    if len(seq_a) != len(seq_b):
        raise ValueError("Aligned sequences must have the same length.")
    distance = 0
    for aa_a, aa_b in zip(seq_a, seq_b):
        if aa_a == aa_b:
            continue
        if aa_a == "-" and aa_b == "-":
            continue
        distance += 1
    return distance


def summarize_site_counts(df: pd.DataFrame, site: str) -> pd.DataFrame:
    aa_col = f"aa_{site}"
    counts = df[aa_col].value_counts(dropna=False)
    total = counts.sum()
    out = counts.reset_index()
    out.columns = ["amino_acid", "count"]
    out["total"] = total
    out["proportion"] = out["count"] / total if total else 0.0
    out["canonical_site"] = site
    return out


def plot_stacked_bar(site_summary: pd.DataFrame, output_path: str) -> None:
    pivot = site_summary.pivot_table(
        index="canonical_site",
        columns="amino_acid",
        values="proportion",
        fill_value=0.0,
    )
    aa_order = sorted(pivot.columns)
    colors = [plt.cm.tab20(i % 20) for i in range(len(aa_order))]
    ax = pivot[aa_order].plot(kind="bar", stacked=True, figsize=(10, 6), color=colors)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Canonical site")
    ax.set_ylim(0, 1)
    ax.legend(title="AA", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container, aa in zip(ax.containers, aa_order):
        for patch in container:
            height = patch.get_height()
            if height < 0.05:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + height / 2
            ax.text(x, y, aa, ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_distance_panels(df: pd.DataFrame, output_path: str) -> None:
    df = df.copy()
    df["distance_bin"] = pd.cut(
        df["distance_to_k"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        include_lowest=True,
        right=True,
    )

    num_sites = len(CANONICAL_SITES)
    cols = 3
    rows = (num_sites + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
    axes = axes.flatten()

    for idx, site in enumerate(CANONICAL_SITES):
        ax = axes[idx]
        aa_col = f"aa_{site}"
        grouped = (
            df.groupby(["distance_bin", aa_col])
            .size()
            .reset_index(name="count")
        )
        pivot = grouped.pivot_table(
            index="distance_bin",
            columns=aa_col,
            values="count",
            fill_value=0,
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
        aa_order = sorted(pivot.columns)
        colors = [plt.cm.tab20(i % 20) for i in range(len(aa_order))]
        pivot[aa_order].plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_title(f"Site {site}")
        ax.set_xlabel("Distance to K")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend().remove()

        for container, aa in zip(ax.containers, aa_order):
            for patch in container:
                height = patch.get_height()
                if height < 0.05:
                    continue
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_y() + height / 2
                ax.text(x, y, aa, ha="center", va="center", fontsize=7, color="black")

    for ax in axes[num_sites:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="AA", bbox_to_anchor=(1.02, 0.9), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = parse_alignment_records(FASTA_PATH)
    if TEST_MODE:
        random.seed(RANDOM_SEED)
        records = random.sample(records, min(TEST_SAMPLE_SIZE, len(records)))
    ref_sequence = load_reference_sequence(REFERENCE_PATH)

    k_record = next((r for r in records if r.id == K_LINEAGE_ACCESSION), None)
    if k_record is None:
        raise ValueError(f"K lineage sequence {K_LINEAGE_ACCESSION} not found in alignment.")
    k_aligned = str(k_record.seq)

    rows = []
    for record in records:
        aligned_seq = str(record.seq)
        distance_to_k = compute_alignment_distance(aligned_seq, k_aligned)
        ungapped = aligned_seq.replace("-", "")
        query_map = create_h3_numbering_map(ungapped, ref_sequence, HA2_start=HA2_START)
        label_to_pos = {label: pos for pos, label in query_map.items()}
        ungapped_to_aligned = build_ungapped_to_aligned_index(aligned_seq)

        row = {
            "record_id": record.id,
            "distance_to_k": distance_to_k,
        }
        for site in CANONICAL_SITES:
            pos = label_to_pos.get(site)
            aa = ungapped[pos] if pos is not None and pos < len(ungapped) else None
            aligned_pos = (
                ungapped_to_aligned.get(pos) if pos is not None else None
            )
            row[f"pos_{site}"] = aligned_pos
            row[f"aa_{site}"] = aa
        rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "canonical_site_alignment_table.tsv")
    out_df.to_csv(output_path, sep="\t", index=False)

    summary_frames = [summarize_site_counts(out_df, site) for site in CANONICAL_SITES]
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = os.path.join(OUTPUT_DIR, "canonical_site_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)

    proportions_wide = summary_df.pivot_table(
        index="canonical_site",
        columns="amino_acid",
        values="proportion",
        fill_value=0.0,
    )
    proportions_path = os.path.join(OUTPUT_DIR, "canonical_site_summary_proportions.tsv")
    proportions_wide.to_csv(proportions_path, sep="\t")

    stacked_output = os.path.join(OUTPUT_DIR, "canonical_site_stacked_bar.png")
    plot_stacked_bar(summary_df, stacked_output)

    panel_output = os.path.join(OUTPUT_DIR, "canonical_site_distance_panels.png")
    plot_distance_panels(out_df, panel_output)

    print("Done.")
    print(out_df)
    print(summary_df)
    if TEST_MODE:
        print(f"Test mode: sampled {len(records)} sequences")


if __name__ == "__main__":
    main()
