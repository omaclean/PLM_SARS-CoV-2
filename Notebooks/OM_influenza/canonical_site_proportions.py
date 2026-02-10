"""Compute amino-acid proportions at canonical H3 sites from an aligned HA FASTA."""

from __future__ import annotations

import os
import random
import re
import sys
import warnings
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from Functions_HuggingFace import create_h3_numbering_map, align_sequences


# ---- User parameters ----
FASTA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_AA.fasta"
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/canon_sites_test"
REFERENCE_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

HA2_START = 330
REFERENCE_LINEAGE_LABEL = "J.2_odd"  # Label used in plots/output for the lineage reference sequence
REFERENCE_LINEAGE_ACCESSION = "PV511679"  # Accession of the lineage reference sequence for distance calculations

REFERENCE_LINEAGE_LABEL="oldHA2:49:S_lineage"
REFERENCE_LINEAGE_ACCESSION ="MT186157"
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/canon_sites_test_weird_ha2_49_lineage"
#
# Canonical sites to measure (H3 numbering; use HA2:## for HA2 sites)
CANONICAL_SITES = ["2", "122","135","144", "158", "160", "173", "189", "276","328", "HA2:49"]

DISTANCE_BINS = [-0.1, 0, 5, 10, 20, 50, 100, 150, 200, 10_000]
DISTANCE_LABELS = [
    "0",
    "1-5",
    "5-10",
    "10-20",
    "20-50",
    "50-100",
    "100-150",
    "150-200",
    "200+",
]

TEST_MODE = False #True #False  #False
TEST_SAMPLE_SIZE = 1000
RANDOM_SEED = 42

# Fixed amino-acid palette and ordering
AA_ORDER = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "-", "X", "*",
]
AA_COLORS = {
    "A": "#1f77b4",
    "C": "#ff7f0e",
    "D": "#2ca02c",
    "E": "#d62728",
    "F": "#9467bd",
    "G": "#8c564b",
    "H": "#e377c2",
    "I": "#7f7f7f",
    "K": "#bcbd22",
    "L": "#17becf",
    "M": "#393b79",
    "N": "#637939",
    "P": "#8c6d31",
    "Q": "#843c39",
    "R": "#7b4173",
    "S": "#3182bd",
    "T": "#31a354",
    "V": "#756bb1",
    "W": "#636363",
    "Y": "#e6550d",
    "-": "#bdbdbd",
    "X": "#969696",
    "*": "#525252",
}
UNKNOWN_AA_COLOR = "#c7c7c7"

warnings.filterwarnings(
    "ignore",
    message=(
        r"A NumPy version .* is required for this version of SciPy \(detected version .*\)"
    ),
    category=UserWarning,
)


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _ordered_aas(columns: List) -> List[str]:
    aas = [str(c) for c in columns]
    ordered = [aa for aa in AA_ORDER if aa in aas]
    extra = sorted([aa for aa in aas if aa not in AA_ORDER])
    return ordered + extra


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
    pivot = pivot.reindex(CANONICAL_SITES)
    aa_order = _ordered_aas(pivot.columns)
    colors = [AA_COLORS.get(aa, UNKNOWN_AA_COLOR) for aa in aa_order]
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
            df.groupby(["distance_bin", aa_col], observed=False)
            .size()
            .reset_index(name="count")
        )
        pivot = grouped.pivot_table(
            index="distance_bin",
            columns=aa_col,
            values="count",
            fill_value=0,
            observed=False,
        )
        if pivot.empty:
            ax.set_title(f"Site {site}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        # Calculate counts per distance bin before normalization
        bin_counts = pivot.sum(axis=1)
        pivot = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
        aa_order = _ordered_aas(pivot.columns)
        if not aa_order:
            ax.set_title(f"Site {site}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        colors = [AA_COLORS.get(aa, UNKNOWN_AA_COLOR) for aa in aa_order]
        pivot[aa_order].plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_title(f"Site {site}")
        ax.set_xlabel(f"Distance to {REFERENCE_LINEAGE_LABEL}")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        ax.legend().remove()
        
        # Update x-axis labels with sequence counts
        x_labels = [f"{label}\n(N={int(bin_counts[label])})" for label in pivot.index]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

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


def export_alignment_label_map(
    record_id: str,
    aligned_seq: str,
    ref_sequence: str,
    output_dir: str,
) -> str:
    ungapped = aligned_seq.replace("-", "")
    query_map = create_h3_numbering_map(ungapped, ref_sequence, HA2_start=HA2_START)
    ungapped_to_aligned = build_ungapped_to_aligned_index(aligned_seq)

    rows = []
    for ungapped_pos0, label in query_map.items():
        aligned_pos1 = ungapped_to_aligned.get(ungapped_pos0)
        aa = ungapped[ungapped_pos0] if ungapped_pos0 < len(ungapped) else None
        rows.append(
            {
                "record_id": record_id,
                "ungapped_pos0": ungapped_pos0,
                "ungapped_pos1": ungapped_pos0 + 1,
                "aligned_pos1": aligned_pos1,
                "label": label,
                "aa": aa,
            }
        )

    map_df = pd.DataFrame(rows).sort_values(["aligned_pos1", "ungapped_pos0"])
    safe_id = _sanitize_filename(record_id)
    output_path = os.path.join(output_dir, f"alignment_label_map_{safe_id}.csv")
    map_df.to_csv(output_path, index=False)
    return output_path


def export_full_alignment_table(
    ref_record_id: str,
    ref_sequence: str,
    canonical_sequence: str,
    output_dir: str,
) -> str:
    alignment = align_sequences(
        reference_seq=canonical_sequence,
        query_seq=ref_sequence,
        mode="global",
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )
    canonical_aligned, ref_aligned = _build_aligned_strings(
        alignment, canonical_sequence, ref_sequence
    )

    canonical_map = create_h3_numbering_map(
        canonical_sequence, canonical_sequence, HA2_start=HA2_START
    )
    canonical_ungapped_to_label = {pos: label for pos, label in canonical_map.items()}
    canonical_ungapped_to_aligned = build_ungapped_to_aligned_index(canonical_aligned)
    aligned_pos_to_label = {
        canonical_ungapped_to_aligned[pos0]: label
        for pos0, label in canonical_ungapped_to_label.items()
        if canonical_ungapped_to_aligned.get(pos0) is not None
    }

    rows = []
    for aligned_pos1, (canon_aa, ref_aa) in enumerate(
        zip(canonical_aligned, ref_aligned), start=1
    ):
        row = {
            "aligned_pos1": aligned_pos1,
            "label": aligned_pos_to_label.get(aligned_pos1),
            "canonical_aa": canon_aa,
            f"{REFERENCE_LINEAGE_LABEL}_aa": ref_aa,
        }
        rows.append(row)

    full_df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, "canonical_alignment_table.csv")
    full_df.to_csv(output_path, index=False)
    return output_path


def _build_aligned_strings(alignment, ref_seq: str, query_seq: str) -> tuple[str, str]:
    ref_aligned_parts = []
    query_aligned_parts = []
    ref_pos = 0
    query_pos = 0

    ref_blocks, query_blocks = alignment.aligned
    for (ref_start, ref_end), (query_start, query_end) in zip(ref_blocks, query_blocks):
        if ref_start > ref_pos:
            ref_aligned_parts.append(ref_seq[ref_pos:ref_start])
            query_aligned_parts.append("-" * (ref_start - ref_pos))
        if query_start > query_pos:
            ref_aligned_parts.append("-" * (query_start - query_pos))
            query_aligned_parts.append(query_seq[query_pos:query_start])

        ref_aligned_parts.append(ref_seq[ref_start:ref_end])
        query_aligned_parts.append(query_seq[query_start:query_end])
        ref_pos = ref_end
        query_pos = query_end

    if ref_pos < len(ref_seq):
        ref_aligned_parts.append(ref_seq[ref_pos:])
        query_aligned_parts.append("-" * (len(ref_seq) - ref_pos))
    if query_pos < len(query_seq):
        ref_aligned_parts.append("-" * (len(query_seq) - query_pos))
        query_aligned_parts.append(query_seq[query_pos:])

    return "".join(ref_aligned_parts), "".join(query_aligned_parts)


def export_reference_homology(
    ref_record_id: str,
    ref_sequence: str,
    canonical_sequence: str,
    output_dir: str,
) -> str:
    alignment = align_sequences(
        reference_seq=canonical_sequence,
        query_seq=ref_sequence,
        mode="global",
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )

    canon_aligned, ref_aligned = _build_aligned_strings(
        alignment, canonical_sequence, ref_sequence
    )
    aligned_length = len(canon_aligned)
    ungapped_positions = [
        i for i, (a, b) in enumerate(zip(canon_aligned, ref_aligned)) if a != "-" and b != "-"
    ]
    ungapped_length = len(ungapped_positions)
    matches = sum(
        1
        for i in ungapped_positions
        if canon_aligned[i] == ref_aligned[i]
    )
    identity_ungapped = matches / ungapped_length if ungapped_length else 0.0
    identity_aligned = (
        sum(1 for a, b in zip(canon_aligned, ref_aligned) if a == b) / aligned_length
        if aligned_length
        else 0.0
    )

    summary_df = pd.DataFrame(
        [
            {
                "reference_record_id": ref_record_id,
                "reference_lineage_label": REFERENCE_LINEAGE_LABEL,
                "canonical_reference_path": REFERENCE_PATH,
                "reference_length": len(ref_sequence),
                "canonical_length": len(canonical_sequence),
                "aligned_length": aligned_length,
                "ungapped_aligned_length": ungapped_length,
                "matches": matches,
                "identity_ungapped": identity_ungapped,
                "identity_aligned": identity_aligned,
            }
        ]
    )
    output_path = os.path.join(output_dir, "reference_homology_summary.csv")
    summary_df.to_csv(output_path, index=False)
    return output_path


def build_label_to_msa_aligned_pos(
    canonical_sequence: str,
    ref_sequence: str,
    ref_aligned: str,
) -> Dict[str, int]:
    alignment = align_sequences(
        reference_seq=canonical_sequence,
        query_seq=ref_sequence,
        mode="global",
        open_gap_score=-10,
        extend_gap_score=-0.5,
    )
    canonical_aligned, ref_aligned_pairwise = _build_aligned_strings(
        alignment, canonical_sequence, ref_sequence
    )

    canonical_map = create_h3_numbering_map(
        canonical_sequence, canonical_sequence, HA2_start=HA2_START
    )
    canonical_pos_to_label = {pos0: label for pos0, label in canonical_map.items()}

    label_to_ref_pos0: Dict[str, int] = {}
    canon_pos0 = -1
    ref_pos0 = -1
    for canon_aa, ref_aa in zip(canonical_aligned, ref_aligned_pairwise):
        if canon_aa != "-":
            canon_pos0 += 1
        if ref_aa != "-":
            ref_pos0 += 1
        if canon_aa != "-" and ref_aa != "-":
            label = canonical_pos_to_label.get(canon_pos0)
            if label:
                label_to_ref_pos0[label] = ref_pos0

    ref_ungapped_to_aligned = build_ungapped_to_aligned_index(ref_aligned)
    label_to_aligned_pos1 = {
        label: ref_ungapped_to_aligned.get(pos0)
        for label, pos0 in label_to_ref_pos0.items()
        if ref_ungapped_to_aligned.get(pos0) is not None
    }
    return label_to_aligned_pos1


def run_association_experiment(out_df: pd.DataFrame, output_dir: str) -> None:
    def bin_distance(d: int) -> str:
        if d <= 0:
            return "0"
        if d <= 5:
            return "1-5"
        if d <= 10:
            return "5-10"
        if d <= 20:
            return "10-20"
        if d <= 50:
            return "20-50"
        if d <= 100:
            return "50-100"
        if d <= 150:
            return "100-150"
        if d <= 200:
            return "150-200"
        return "200+"

    df = out_df.copy()
    df["distance_bin"] = df["distance_to_k"].apply(bin_distance)

    s_div = df[(df["aa_HA2:49"] == "S") & (df["distance_bin"] == "200+")].copy()
    n_close = df[(df["aa_HA2:49"] == "N") & (df["distance_bin"] == "200+")].copy()

    site_cols = [c for c in df.columns if c.startswith("aa_") and c != "aa_HA2:49"]

    records = []
    for col in site_cols:
        s_counts = s_div[col].value_counts(dropna=False)
        n_counts = n_close[col].value_counts(dropna=False)
        s_total = len(s_div)
        n_total = len(n_close)
        aas = set(s_counts.index).union(set(n_counts.index))
        for aa in aas:
            s_prop = s_counts.get(aa, 0) / s_total if s_total else 0.0
            n_prop = n_counts.get(aa, 0) / n_total if n_total else 0.0
            records.append(
                {
                    "site": col.replace("aa_", ""),
                    "aa": aa,
                    "s_prop": s_prop,
                    "n_prop": n_prop,
                    "diff": s_prop - n_prop,
                    "s_count": int(s_counts.get(aa, 0)),
                    "n_count": int(n_counts.get(aa, 0)),
                }
            )

    assoc_df = pd.DataFrame(records)
    assoc_path = os.path.join(output_dir, "ha2_49_association_diffs.tsv")
    assoc_df.sort_values("diff", key=lambda s: s.abs(), ascending=False).to_csv(
        assoc_path, sep="\t", index=False
    )

    example_s = (
        s_div.sort_values("distance_to_k", ascending=False)["record_id"].head(5).tolist()
    )
    example_n = (
        n_close.sort_values("distance_to_k", ascending=False)["record_id"].head(5).tolist()
    )

    summary_path = os.path.join(output_dir, "ha2_49_association_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Association experiment: HA2:49 S (200+) vs N (200+)\n")
        handle.write(f"S 200+ count: {len(s_div)}\n")
        handle.write(f"N 200+ count: {len(n_close)}\n")
        handle.write(f"Example S accessions: {', '.join(example_s)}\n")
        handle.write(f"Example N accessions: {', '.join(example_n)}\n")
        handle.write(f"Diffs table: {assoc_path}\n")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = parse_alignment_records(FASTA_PATH)
    ref_sequence = load_reference_sequence(REFERENCE_PATH)

    ref_record = next((r for r in records if r.id == REFERENCE_LINEAGE_ACCESSION), None)
    if ref_record is None:
        raise ValueError(
            f"Reference lineage sequence {REFERENCE_LINEAGE_ACCESSION} not found in alignment."
        )
    ref_aligned = str(ref_record.seq)

    if TEST_MODE:
        random.seed(RANDOM_SEED)
        records = random.sample(records, min(TEST_SAMPLE_SIZE, len(records)))
        if all(r.id != ref_record.id for r in records):
            records.append(ref_record)

    alignment_map_path = export_alignment_label_map(
        ref_record.id,
        ref_aligned,
        ref_sequence,
        OUTPUT_DIR,
    )

    full_alignment_path = export_full_alignment_table(
        ref_record.id,
        ref_aligned.replace("-", ""),
        ref_sequence,
        OUTPUT_DIR,
    )

    homology_path = export_reference_homology(
        ref_record.id,
        ref_aligned.replace("-", ""),
        ref_sequence,
        OUTPUT_DIR,
    )

    site_to_aligned_pos1 = build_label_to_msa_aligned_pos(
        canonical_sequence=ref_sequence,
        ref_sequence=ref_aligned.replace("-", ""),
        ref_aligned=ref_aligned,
    )

    rows = []
    for record in records:
        aligned_seq = str(record.seq)
        distance_to_k = compute_alignment_distance(aligned_seq, ref_aligned)

        row = {
            "record_id": record.id,
            "distance_to_k": distance_to_k,
        }
        for site in CANONICAL_SITES:
            aligned_pos = site_to_aligned_pos1.get(site)
            aa = (
                aligned_seq[aligned_pos - 1]
                if aligned_pos is not None and aligned_pos - 1 < len(aligned_seq)
                else None
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

    run_association_experiment(out_df, OUTPUT_DIR)

    print("Done.")
    print(f"Alignment label map written to: {alignment_map_path}")
    print(f"Full alignment table written to: {full_alignment_path}")
    print(f"Reference homology summary written to: {homology_path}")
    print(out_df)
    print(summary_df)
    if TEST_MODE:
        print(f"Test mode: sampled {len(records)} sequences")


if __name__ == "__main__":
    main()
