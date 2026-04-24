#!/usr/bin/env python3
"""Assign aligned sequences to the nearest lineage reference and write lineage FASTAs.

This CLI generalizes the earlier notebook workflow for aligned protein FASTA inputs.
By default it runs on the SARS-CoV-2 spike reference and monthly snapshot files in
this repository, but all paths and thresholds are configurable.

Two reference-handling modes are supported:
1. Provide aligned references in the same coordinate space as the query alignment.
2. Provide unaligned references plus --align-accession to pad them into the query
   alignment coordinates using pairwise alignment to a chosen query record.

Assignment modes:
- distance: nearest reference by Hamming distance only.
- soft: require current branch mutations and reject only if all next-branch
  mutations are present.
- hard: require current branch mutations and reject if more than N next-branch
  mutations are present.

The branch-aware modes assume the references are ordered along a single lineage path.
For mixed reference sets, use the default distance mode.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from Bio import Align
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


DEFAULT_QUERY_FASTA = Path(
    "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/comb2025_spike_aa_aln.fa"
)
DEFAULT_REFERENCE_FASTA = Path(
    "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references_spike_aln_LoA_subset.fasta"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/lineage_spilts"
)

ALIGN_OPEN_GAP_SCORE = -10.0
ALIGN_EXTEND_GAP_SCORE = -0.5
ALIGN_MATCH_SCORE = 2.0
ALIGN_MISMATCH_SCORE = -1.0
DEFAULT_MAX_MUTATIONS = 10
DEFAULT_HARD_MAX_NEXT_MUTATIONS = 2
DEFAULT_FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH = 1
DEFAULT_RANDOM_SEED = 42
DEFAULT_DEBUG_TOP_N = 15
UNKNOWN_RESIDUES = frozenset({"X", "N", "?"})

BranchMutation = Tuple[int, str, str]


@dataclass
class ClusterRef:
    record_id: str
    lineage: str
    sequence: str
    source_type: str
    order_index: int


@dataclass
class AssignmentResult:
    assignments: List[Dict[str, str]]
    best_matches: List[Dict[str, str]]
    lineage_records: Dict[str, List[SeqRecord]]
    ignored_records: List[SeqRecord]
    assigned_query_sequences_by_reference: Dict[str, List[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign aligned protein sequences to the nearest lineage reference and "
            "write one FASTA per lineage."
        )
    )
    parser.add_argument(
        "--query-fasta",
        type=Path,
        default=DEFAULT_QUERY_FASTA,
        help="Aligned query FASTA file to split by nearest lineage reference.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=DEFAULT_REFERENCE_FASTA,
        help=(
            "Reference FASTA defining lineage labels. If --align-accession is omitted, "
            "these references are assumed to already be aligned to the query coordinate space."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-lineage FASTAs and diagnostics.",
    )
    parser.add_argument(
        "--align-accession",
        default=None,
        help=(
            "Optional query FASTA accession to use as the anchor alignment template. "
            "Use this when the references are not already aligned to the query coordinate space."
        ),
    )
    parser.add_argument(
        "--assignment-mode",
        choices=("distance", "soft", "hard"),
        default="distance",
        help=(
            "Assignment rule. Use distance for arbitrary reference sets; soft/hard assume "
            "references are ordered along a single lineage path."
        ),
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=DEFAULT_MAX_MUTATIONS,
        help="Maximum Hamming distance allowed for assignment. Higher distances are ignored.",
    )
    parser.add_argument(
        "--hard-max-next-mutations",
        type=int,
        default=DEFAULT_HARD_MAX_NEXT_MUTATIONS,
        help="For hard mode, reject assignments with more than this many next-branch mutations.",
    )
    parser.add_argument(
        "--final-lineage-max-missing-current-branch",
        type=int,
        default=DEFAULT_FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH,
        help="For terminal references in branch-aware modes, allow up to this many missing branch mutations.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional random subsample size for query sequences. The anchor record is always kept when used.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for --sample-size.",
    )
    parser.add_argument(
        "--debug-top-n",
        type=int,
        default=DEFAULT_DEBUG_TOP_N,
        help="Number of closest assignments to include in the debug FASTA.",
    )
    parser.add_argument(
        "--lineage-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Optional lineage alias mapping. Repeatable.",
    )
    return parser


def parse_aliases(raw_aliases: Sequence[str]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for raw_alias in raw_aliases:
        if "=" not in raw_alias:
            raise ValueError(f"Invalid --lineage-alias value: {raw_alias}")
        source, target = raw_alias.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid --lineage-alias value: {raw_alias}")
        aliases[source] = target
    return aliases


def parse_alignment_records(fasta_path: Path) -> List[SeqRecord]:
    records: List[SeqRecord] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        record.id = record.id.strip()
        record.description = ""
        records.append(record)
    return records


def is_probably_nucleotide(seq: str) -> bool:
    cleaned = seq.replace("-", "").replace(".", "").upper()
    if not cleaned:
        return False
    nuc_chars = set("ACGTUN")
    nuc_count = sum(1 for char in cleaned if char in nuc_chars)
    return (nuc_count / len(cleaned)) >= 0.95


def is_unknown_residue(residue: str) -> bool:
    return residue.upper() in UNKNOWN_RESIDUES


def configure_aligner_end_gap_scores(aligner: Align.PairwiseAligner, score: float) -> None:
    aligner_dir = dir(aligner)
    if "end_insertion_score" in aligner_dir and "end_deletion_score" in aligner_dir:
        aligner.end_insertion_score = score
        aligner.end_deletion_score = score
        return
    aligner.target_end_gap_score = score
    aligner.query_end_gap_score = score


def to_protein_sequence(seq: str) -> str:
    cleaned = seq.replace("-", "").replace(".", "").upper()
    if not cleaned:
        return ""

    best_orf = ""
    best_has_start = False

    for frame in range(3):
        frame_seq = cleaned[frame:]
        trimmed = frame_seq[: (len(frame_seq) // 3) * 3]
        if not trimmed:
            continue

        translated = str(Seq(trimmed).translate(to_stop=False))
        for peptide in translated.split("*"):
            if not peptide:
                continue

            m_index = peptide.find("M")
            has_start = m_index != -1
            candidate = peptide[m_index:] if has_start else peptide
            if not candidate:
                continue

            if (
                not best_orf
                or (has_start and not best_has_start)
                or (has_start == best_has_start and len(candidate) > len(best_orf))
            ):
                best_orf = candidate
                best_has_start = has_start

    return best_orf


def extract_lineage(header: str, aliases: Dict[str, str]) -> str:
    lineage = header.split("|")[-1].strip() if "|" in header else header.strip()
    return aliases.get(lineage, lineage)


def parse_cluster_references(cluster_path: Path, aliases: Dict[str, str]) -> List[ClusterRef]:
    refs: List[ClusterRef] = []
    for idx, record in enumerate(SeqIO.parse(str(cluster_path), "fasta"), start=1):
        header = record.id.strip()
        lineage = extract_lineage(header, aliases)
        raw_seq = str(record.seq)
        if is_probably_nucleotide(raw_seq):
            parsed_seq = to_protein_sequence(raw_seq)
            source_type = "nt_translated"
        else:
            parsed_seq = raw_seq.upper().replace(".", "-")
            source_type = "aa"
        refs.append(
            ClusterRef(
                record_id=header,
                lineage=lineage,
                sequence=parsed_seq,
                source_type=source_type,
                order_index=idx,
            )
        )
    if not refs:
        raise ValueError(f"No reference records found in {cluster_path}")
    return refs


def build_anchor_aligned_sequence(records: Sequence[SeqRecord], accession: str) -> str:
    for record in records:
        if record.id == accession:
            return str(record.seq).upper().replace(".", "-")
    raise ValueError(f"align accession {accession} not found in query FASTA")


def pad_cluster_to_anchor_alignment(
    cluster_seq: str,
    anchor_alignment: str,
    match_score: float = ALIGN_MATCH_SCORE,
    mismatch_score: float = ALIGN_MISMATCH_SCORE,
    gap_open: float = ALIGN_OPEN_GAP_SCORE,
    gap_extend: float = ALIGN_EXTEND_GAP_SCORE,
) -> str:
    anchor_ungapped = anchor_alignment.replace("-", "")
    cluster_ungapped = cluster_seq.replace("-", "")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    configure_aligner_end_gap_scores(aligner, 0.0)
    alignment = aligner.align(anchor_ungapped, cluster_ungapped)[0]

    aligned_fasta = alignment.format("fasta")
    aligned_lines = [
        line.strip()
        for line in aligned_fasta.splitlines()
        if line.strip() and not line.startswith(">")
    ]
    if len(aligned_lines) < 2:
        raise ValueError("Failed to reconstruct pairwise alignment strings")
    aligned_anchor = aligned_lines[0]
    aligned_cluster = aligned_lines[1]

    cluster_by_anchor_pos: List[str] = []
    for anchor_char, cluster_char in zip(aligned_anchor, aligned_cluster):
        if anchor_char != "-":
            cluster_by_anchor_pos.append(cluster_char)

    if len(cluster_by_anchor_pos) < len(anchor_ungapped):
        cluster_by_anchor_pos.extend(["-"] * (len(anchor_ungapped) - len(cluster_by_anchor_pos)))

    padded: List[str] = []
    anchor_pos = 0
    for anchor_char in anchor_alignment:
        if anchor_char == "-":
            padded.append("-")
            continue
        padded.append(cluster_by_anchor_pos[anchor_pos] if anchor_pos < len(cluster_by_anchor_pos) else "-")
        anchor_pos += 1
    return "".join(padded)


def hamming_distance(seq_a: str, seq_b: str) -> int:
    return sum(
        1
        for a, b in zip(seq_a, seq_b)
        if not is_unknown_residue(a) and not is_unknown_residue(b) and a != b
    )


def normalize_length(seq: str, target_len: int) -> str:
    seq = seq.upper().replace(".", "-")
    if len(seq) == target_len:
        return seq
    if len(seq) < target_len:
        return seq + ("-" * (target_len - len(seq)))
    return seq[:target_len]


def safe_label(label: str) -> str:
    cleaned = label.strip().replace(" ", "_")
    return cleaned.replace("/", "-")


def deduplicate_records(records: Iterable[SeqRecord]) -> List[SeqRecord]:
    unique_by_sequence: Dict[str, SeqRecord] = {}
    for record in records:
        seq_key = str(record.seq)
        if seq_key not in unique_by_sequence:
            unique_by_sequence[seq_key] = record
    return list(unique_by_sequence.values())


def branch_mutations(parent_seq: str, child_seq: str) -> List[BranchMutation]:
    muts: List[BranchMutation] = []
    for pos, (parent_aa, child_aa) in enumerate(zip(parent_seq, child_seq)):
        if is_unknown_residue(parent_aa) or is_unknown_residue(child_aa):
            continue
        if parent_aa != child_aa:
            muts.append((pos, parent_aa, child_aa))
    return muts


def has_all_mutations(sequence: str, mutations: Sequence[BranchMutation]) -> bool:
    for pos, _, child_aa in mutations:
        if pos >= len(sequence):
            return False
        if is_unknown_residue(sequence[pos]) or is_unknown_residue(child_aa):
            continue
        if sequence[pos] != child_aa:
            return False
    return True


def count_missing_mutations(sequence: str, mutations: Sequence[BranchMutation]) -> int:
    missing = 0
    for pos, _, child_aa in mutations:
        if pos >= len(sequence):
            missing += 1
            continue
        if is_unknown_residue(sequence[pos]) or is_unknown_residue(child_aa):
            continue
        if sequence[pos] != child_aa:
            missing += 1
    return missing


def count_present_mutations(sequence: str, mutations: Sequence[BranchMutation]) -> int:
    present = 0
    for pos, _, child_aa in mutations:
        if pos < len(sequence) and not is_unknown_residue(sequence[pos]) and not is_unknown_residue(child_aa) and sequence[pos] == child_aa:
            present += 1
    return present


def mode_tag(mode: str, hard_max_next: int) -> str:
    if mode == "hard":
        return f"hard_nextle{hard_max_next}"
    if mode == "soft":
        return "soft"
    return "distance"


def validate_mode(mode: str) -> None:
    if mode not in {"distance", "soft", "hard"}:
        raise ValueError("assignment mode must be one of: distance, soft, hard")


def build_branch_mutation_profiles(
    aligned_refs: Sequence[ClusterRef],
) -> Tuple[List[List[BranchMutation]], List[List[BranchMutation]], List[Dict[str, str]]]:
    branch_by_ref: List[List[BranchMutation]] = []
    cumulative_by_ref: List[List[BranchMutation]] = []
    inheritance_rows: List[Dict[str, str]] = []

    root_seq = aligned_refs[0].sequence if aligned_refs else ""
    for idx, ref in enumerate(aligned_refs):
        if idx == 0:
            branch_by_ref.append([])
            cumulative_by_ref.append([])
            inheritance_rows.append(
                {
                    "order_index": str(ref.order_index),
                    "record_id": ref.record_id,
                    "lineage": ref.lineage,
                    "status": "root",
                    "missing_previous_branch_mutations": "",
                    "previous_branch_size": "0",
                }
            )
            continue

        parent = aligned_refs[idx - 1]
        branch = branch_mutations(parent.sequence, ref.sequence)
        branch_by_ref.append(branch)
        cumulative_by_ref.append(branch_mutations(root_seq, ref.sequence))

        if idx >= 2:
            prior_branch = branch_by_ref[idx - 1]
            missing_prior = [
                f"{src}{pos + 1}{dst}"
                for pos, src, dst in prior_branch
                if ref.sequence[pos] != dst
            ]
            status = "ok" if not missing_prior else "violation"
            inheritance_rows.append(
                {
                    "order_index": str(ref.order_index),
                    "record_id": ref.record_id,
                    "lineage": ref.lineage,
                    "status": status,
                    "missing_previous_branch_mutations": ",".join(missing_prior),
                    "previous_branch_size": str(len(prior_branch)),
                }
            )
        else:
            inheritance_rows.append(
                {
                    "order_index": str(ref.order_index),
                    "record_id": ref.record_id,
                    "lineage": ref.lineage,
                    "status": "ok",
                    "missing_previous_branch_mutations": "",
                    "previous_branch_size": "0",
                }
            )

    return branch_by_ref, cumulative_by_ref, inheritance_rows


def passes_mode_criteria(
    query_seq: str,
    ref_index: int,
    branch_by_ref: Sequence[Sequence[BranchMutation]],
    cumulative_by_ref: Sequence[Sequence[BranchMutation]],
    mode: str,
    hard_max_next: int,
    final_lineage_max_missing_current_branch: int,
) -> Tuple[bool, int, int]:
    if mode == "distance":
        return True, 0, 0

    current_branch = branch_by_ref[ref_index]
    current_missing = count_missing_mutations(query_seq, current_branch)
    is_terminal = ref_index == (len(branch_by_ref) - 1)
    if is_terminal:
        if current_missing > final_lineage_max_missing_current_branch:
            return False, 0, len(current_branch)
    elif current_missing > 0:
        return False, 0, len(current_branch)

    if not has_all_mutations(query_seq, current_branch) and not is_terminal:
        return False, 0, len(current_branch)

    if not has_all_mutations(query_seq, cumulative_by_ref[ref_index]):
        return False, 0, 0

    next_index = ref_index + 1
    if next_index >= len(branch_by_ref):
        return True, 0, 0

    next_branch = branch_by_ref[next_index]
    next_present = count_present_mutations(query_seq, next_branch)
    next_total = len(next_branch)

    if mode == "soft":
        if next_total > 0 and next_present == next_total:
            return False, next_present, next_total
        return True, next_present, next_total

    if next_present > hard_max_next:
        return False, next_present, next_total
    return True, next_present, next_total


def maybe_subsample_records(
    records_all: Sequence[SeqRecord],
    align_accession: Optional[str],
    sample_size: Optional[int],
    random_seed: int,
) -> List[SeqRecord]:
    if sample_size is None:
        return list(records_all)

    if sample_size <= 0:
        raise ValueError("sample size must be positive")

    records = list(records_all)
    if sample_size >= len(records):
        return records

    random.seed(random_seed)
    if align_accession is None:
        return random.sample(records, sample_size)

    anchor_record: Optional[SeqRecord] = None
    non_anchor_records: List[SeqRecord] = []
    for record in records:
        if record.id == align_accession:
            anchor_record = record
        else:
            non_anchor_records.append(record)

    if anchor_record is None:
        raise ValueError(f"align accession {align_accession} not found in query FASTA")

    if sample_size == 1:
        return [anchor_record]

    sampled_non_anchor = random.sample(non_anchor_records, min(sample_size - 1, len(non_anchor_records)))
    return [anchor_record, *sampled_non_anchor]


def choose_alignment_accession(
    query_records: Sequence[SeqRecord],
    cluster_refs: Sequence[ClusterRef],
    align_accession: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if align_accession:
        return align_accession, None

    if not query_records or not cluster_refs:
        return None, None

    query_lengths = {len(str(record.seq)) for record in query_records}
    reference_lengths = {len(ref.sequence) for ref in cluster_refs}
    if query_lengths != reference_lengths:
        inferred_accession = str(query_records[0].id)
        reason = (
            "query and reference sequence lengths differ, so references will be padded "
            f"to the query coordinate space using the first query record: {inferred_accession}"
        )
        return inferred_accession, reason

    return None, None


def prepare_aligned_references(
    cluster_refs: Sequence[ClusterRef],
    query_records: Sequence[SeqRecord],
    align_accession: Optional[str],
) -> Tuple[List[ClusterRef], str, int]:
    if align_accession:
        anchor_alignment = build_anchor_aligned_sequence(query_records, align_accession)
        aligned_refs: List[ClusterRef] = []
        for ref in cluster_refs:
            padded_seq = pad_cluster_to_anchor_alignment(ref.sequence, anchor_alignment)
            aligned_refs.append(
                ClusterRef(
                    record_id=ref.record_id,
                    lineage=ref.lineage,
                    sequence=padded_seq,
                    source_type=ref.source_type,
                    order_index=ref.order_index,
                )
            )
        return aligned_refs, anchor_alignment, len(anchor_alignment)

    alignment_len = max(
        max(len(str(record.seq)) for record in query_records),
        max(len(ref.sequence) for ref in cluster_refs),
    )
    aligned_refs = [
        ClusterRef(
            record_id=ref.record_id,
            lineage=ref.lineage,
            sequence=normalize_length(ref.sequence, alignment_len),
            source_type=ref.source_type,
            order_index=ref.order_index,
        )
        for ref in cluster_refs
    ]
    anchor_alignment = "-" * alignment_len
    return aligned_refs, anchor_alignment, alignment_len


def prepare_query_records(
    query_records: Sequence[SeqRecord],
    anchor_alignment: str,
    alignment_len: int,
    align_accession: Optional[str],
) -> List[SeqRecord]:
    aligned_records: List[SeqRecord] = []
    should_realign = align_accession is not None or len({len(str(record.seq)) for record in query_records}) > 1

    for record in query_records:
        raw_seq = str(record.seq).upper().replace(".", "-")
        if should_realign:
            if align_accession is not None and record.id == align_accession:
                aligned_seq = anchor_alignment
            elif len(raw_seq) == alignment_len:
                aligned_seq = normalize_length(raw_seq, alignment_len)
            else:
                aligned_seq = pad_cluster_to_anchor_alignment(raw_seq, anchor_alignment)
        else:
            aligned_seq = normalize_length(raw_seq, alignment_len)

        aligned_records.append(SeqRecord(Seq(aligned_seq), id=record.id, description=""))

    return aligned_records


def assign_records(
    records: Sequence[SeqRecord],
    aligned_cluster_refs: Sequence[ClusterRef],
    assignment_mode: str,
    max_mutations: int,
    hard_max_next_mutations: int,
    final_lineage_max_missing_current_branch: int,
) -> AssignmentResult:
    lineages = list(dict.fromkeys(ref.lineage for ref in aligned_cluster_refs))
    lineage_records: Dict[str, List[SeqRecord]] = {lineage: [] for lineage in lineages}
    assignments: List[Dict[str, str]] = []
    ignored_records: List[SeqRecord] = []
    best_matches: List[Dict[str, str]] = []
    assigned_query_sequences_by_reference: Dict[str, List[str]] = {
        ref.record_id: [] for ref in aligned_cluster_refs
    }

    branch_by_ref: List[List[BranchMutation]] = []
    cumulative_by_ref: List[List[BranchMutation]] = []
    if assignment_mode != "distance":
        branch_by_ref, cumulative_by_ref, _ = build_branch_mutation_profiles(aligned_cluster_refs)

    anchor_alignment_len = len(aligned_cluster_refs[0].sequence)
    for record in records:
        query_seq = normalize_length(str(record.seq), anchor_alignment_len)
        nearest_ref: Optional[ClusterRef] = None
        nearest_distance: Optional[int] = None

        best_ref: Optional[ClusterRef] = None
        best_distance: Optional[int] = None
        best_next_present = 0
        best_next_total = 0

        for ref_index, ref in enumerate(aligned_cluster_refs):
            distance = hamming_distance(query_seq, ref.sequence)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_ref = ref

            passes, next_present, next_total = passes_mode_criteria(
                query_seq,
                ref_index,
                branch_by_ref,
                cumulative_by_ref,
                assignment_mode,
                hard_max_next_mutations,
                final_lineage_max_missing_current_branch,
            )
            if not passes:
                continue

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_ref = ref
                best_next_present = next_present
                best_next_total = next_total

        if nearest_ref is None or nearest_distance is None:
            ignored_records.append(record)
            continue

        if best_ref is None or best_distance is None:
            ignored_records.append(record)
            assignments.append(
                {
                    "record_id": str(record.id),
                    "assigned_lineage": "",
                    "best_reference": nearest_ref.record_id,
                    "mutation_count": str(nearest_distance),
                    "status": "ignored_mode_rules",
                    "assignment_mode": assignment_mode,
                    "hard_max_next_mutations": str(hard_max_next_mutations),
                    "next_branch_present": "",
                    "next_branch_total": "",
                }
            )
            continue

        best_matches.append(
            {
                "record_id": str(record.id),
                "best_reference": best_ref.record_id,
                "assigned_lineage": best_ref.lineage,
                "mutation_count": str(best_distance),
                "assignment_mode": assignment_mode,
                "hard_max_next_mutations": str(hard_max_next_mutations),
                "next_branch_present": str(best_next_present),
                "next_branch_total": str(best_next_total),
                "query_sequence": query_seq,
                "reference_sequence": best_ref.sequence,
            }
        )

        if best_distance > max_mutations:
            ignored_records.append(record)
            assignments.append(
                {
                    "record_id": str(record.id),
                    "assigned_lineage": "",
                    "best_reference": best_ref.record_id,
                    "mutation_count": str(best_distance),
                    "status": "ignored",
                    "assignment_mode": assignment_mode,
                    "hard_max_next_mutations": str(hard_max_next_mutations),
                    "next_branch_present": str(best_next_present),
                    "next_branch_total": str(best_next_total),
                }
            )
            continue

        lineage_records[best_ref.lineage].append(record)
        assigned_query_sequences_by_reference[best_ref.record_id].append(query_seq)
        assignments.append(
            {
                "record_id": str(record.id),
                "assigned_lineage": best_ref.lineage,
                "best_reference": best_ref.record_id,
                "mutation_count": str(best_distance),
                "status": "assigned",
                "assignment_mode": assignment_mode,
                "hard_max_next_mutations": str(hard_max_next_mutations),
                "next_branch_present": str(best_next_present),
                "next_branch_total": str(best_next_total),
            }
        )

    return AssignmentResult(
        assignments=assignments,
        best_matches=best_matches,
        lineage_records=lineage_records,
        ignored_records=ignored_records,
        assigned_query_sequences_by_reference=assigned_query_sequences_by_reference,
    )


def write_reference_exports(
    output_dir: Path,
    cluster_refs: Sequence[ClusterRef],
    assignment_tag: str,
) -> Tuple[Path, Path]:
    translated_cluster_path = output_dir / f"reference_sequences_all_{assignment_tag}.fasta"
    with translated_cluster_path.open("w", encoding="utf-8") as handle:
        for ref in cluster_refs:
            handle.write(
                f">{ref.record_id}|lineage={ref.lineage}|source={ref.source_type}|order={ref.order_index}\n"
            )
            handle.write(f"{ref.sequence}\n")

    dedup_translated_cluster_path = output_dir / f"reference_sequences_unique_{assignment_tag}.fasta"
    unique_proteins: Dict[str, List[ClusterRef]] = {}
    for ref in cluster_refs:
        unique_proteins.setdefault(ref.sequence, []).append(ref)
    with dedup_translated_cluster_path.open("w", encoding="utf-8") as handle:
        for idx, (protein_seq, refs_for_seq) in enumerate(unique_proteins.items(), start=1):
            representative = refs_for_seq[0]
            handle.write(
                f">unique_{idx}|rep={representative.record_id}|lineage={representative.lineage}|duplicates={len(refs_for_seq)}\n"
            )
            handle.write(f"{protein_seq}\n")

    return translated_cluster_path, dedup_translated_cluster_path


def write_branch_outputs(
    output_dir: Path,
    aligned_cluster_refs: Sequence[ClusterRef],
    assigned_query_sequences_by_reference: Dict[str, List[str]],
    assignment_tag: str,
) -> Optional[Path]:
    branch_by_ref, cumulative_by_ref, inheritance_rows = build_branch_mutation_profiles(aligned_cluster_refs)

    defining_rows: List[Dict[str, str]] = []
    for idx, ref in enumerate(aligned_cluster_refs):
        for pos, parent_aa, child_aa in branch_by_ref[idx]:
            defining_rows.append(
                {
                    "order_index": str(ref.order_index),
                    "record_id": ref.record_id,
                    "lineage": ref.lineage,
                    "mutation": f"{parent_aa}{pos + 1}{child_aa}",
                    "position_1_based": str(pos + 1),
                    "from_aa": parent_aa,
                    "to_aa": child_aa,
                }
            )

    pd.DataFrame(defining_rows).to_csv(
        output_dir / f"cluster_branch_defining_mutations_{assignment_tag}.tsv",
        sep="\t",
        index=False,
    )

    inheritance_df = pd.DataFrame(inheritance_rows)
    inheritance_df.to_csv(
        output_dir / f"cluster_branch_inheritance_check_{assignment_tag}.tsv",
        sep="\t",
        index=False,
    )

    mutation_audit_rows: List[Dict[str, str]] = []
    for idx, ref in enumerate(aligned_cluster_refs):
        ref_id = ref.record_id
        assigned_queries = assigned_query_sequences_by_reference.get(ref_id, [])
        n_assigned = len(assigned_queries)
        is_terminal = idx == (len(aligned_cluster_refs) - 1)
        for pos, parent_aa, child_aa in branch_by_ref[idx]:
            present_n = sum(1 for query in assigned_queries if pos < len(query) and query[pos] == child_aa)
            support_fraction = (present_n / n_assigned) if n_assigned > 0 else float("nan")
            always_ignored = n_assigned > 0 and present_n == 0
            mutation_audit_rows.append(
                {
                    "order_index": str(ref.order_index),
                    "record_id": ref_id,
                    "lineage": ref.lineage,
                    "mutation": f"{parent_aa}{pos + 1}{child_aa}",
                    "position_1_based": str(pos + 1),
                    "assigned_sequences_to_reference": str(n_assigned),
                    "present_in_assigned_n": str(present_n),
                    "support_fraction": "" if pd.isna(support_fraction) else f"{support_fraction:.6f}",
                    "always_ignored_flag": "yes" if always_ignored else "no",
                    "is_terminal_reference": "yes" if is_terminal else "no",
                    "manual_alignment_check_flag": "yes" if always_ignored else "no",
                }
            )

    mutation_audit_path = output_dir / f"defining_mutation_assignment_audit_{assignment_tag}.tsv"
    pd.DataFrame(mutation_audit_rows).to_csv(mutation_audit_path, sep="\t", index=False)
    return mutation_audit_path


def write_assignment_outputs(
    output_dir: Path,
    result: AssignmentResult,
    assignment_tag: str,
    max_mutations: int,
    debug_top_n: int,
    anchor_alignment: str,
    align_accession: Optional[str],
) -> Dict[str, Path]:
    summary_rows: List[Dict[str, object]] = []
    for lineage, recs in result.lineage_records.items():
        safe_lineage = safe_label(lineage)
        output_path = output_dir / f"{safe_lineage}_{assignment_tag}_max{max_mutations}.fasta"
        SeqIO.write(recs, str(output_path), "fasta")

        unique_recs = deduplicate_records(recs)
        unique_output_path = output_dir / f"{safe_lineage}_{assignment_tag}_max{max_mutations}_unique.fasta"
        SeqIO.write(unique_recs, str(unique_output_path), "fasta")

        summary_rows.append(
            {
                "lineage": lineage,
                "count": len(recs),
                "unique_count": len(unique_recs),
                "output": str(output_path),
                "unique_output": str(unique_output_path),
            }
        )

    ignored_path = output_dir / f"ignored_{assignment_tag}_over_max{max_mutations}.fasta"
    SeqIO.write(result.ignored_records, str(ignored_path), "fasta")
    ignored_unique_records = deduplicate_records(result.ignored_records)
    ignored_unique_path = output_dir / f"ignored_{assignment_tag}_over_max{max_mutations}_unique.fasta"
    SeqIO.write(ignored_unique_records, str(ignored_unique_path), "fasta")

    summary_path = output_dir / f"lineage_subalignment_summary_{assignment_tag}.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)

    assignments_path = output_dir / f"lineage_assignments_{assignment_tag}.tsv"
    pd.DataFrame(result.assignments).to_csv(assignments_path, sep="\t", index=False)

    outputs = {
        "summary": summary_path,
        "assignments": assignments_path,
        "ignored": ignored_path,
        "ignored_unique": ignored_unique_path,
    }

    best_df = pd.DataFrame(result.best_matches)
    if not best_df.empty:
        best_df = best_df.sort_values("mutation_count", key=lambda s: pd.to_numeric(s, errors="coerce"))
        diagnostics_path = output_dir / f"distance_diagnostics_{assignment_tag}.tsv"
        best_df.to_csv(diagnostics_path, sep="\t", index=False)
        outputs["distance_diagnostics"] = diagnostics_path

        top_n = best_df.head(debug_top_n)
        debug_alignment_path = output_dir / f"debug_top{debug_top_n}_nearest_pairs_{assignment_tag}_max{max_mutations}.fasta"
        with debug_alignment_path.open("w", encoding="utf-8") as handle:
            if align_accession is not None:
                handle.write(f">ANCHOR|{align_accession}|aligned_template\n{anchor_alignment}\n")
            for _, row in top_n.iterrows():
                handle.write(
                    f">{row['record_id']}|query|d={row['mutation_count']}|lineage={row['assigned_lineage']}\n{row['query_sequence']}\n"
                )
                handle.write(
                    f">{row['record_id']}|best_ref|{row['best_reference']}\n{row['reference_sequence']}\n"
                )
        outputs["debug_pairs"] = debug_alignment_path

    return outputs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_mode(args.assignment_mode)
    aliases = parse_aliases(args.lineage_alias)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_tag = mode_tag(args.assignment_mode, args.hard_max_next_mutations)

    cluster_refs = parse_cluster_references(args.reference_fasta, aliases)
    translated_cluster_path, dedup_translated_cluster_path = write_reference_exports(
        args.output_dir,
        cluster_refs,
        assignment_tag,
    )

    records_all = parse_alignment_records(args.query_fasta)
    if not records_all:
        raise ValueError(f"No query records found in {args.query_fasta}")
    effective_align_accession, inferred_alignment_reason = choose_alignment_accession(
        records_all,
        cluster_refs,
        args.align_accession,
    )
    records = maybe_subsample_records(
        records_all,
        effective_align_accession,
        args.sample_size,
        args.random_seed,
    )

    aligned_cluster_refs, anchor_alignment, anchor_alignment_len = prepare_aligned_references(
        cluster_refs,
        records_all,
        effective_align_accession,
    )

    aligned_query_records = prepare_query_records(
        records,
        anchor_alignment,
        anchor_alignment_len,
        effective_align_accession,
    )

    result = assign_records(
        aligned_query_records,
        aligned_cluster_refs,
        args.assignment_mode,
        args.max_mutations,
        args.hard_max_next_mutations,
        args.final_lineage_max_missing_current_branch,
    )

    outputs = write_assignment_outputs(
        args.output_dir,
        result,
        assignment_tag,
        args.max_mutations,
        args.debug_top_n,
        anchor_alignment,
        effective_align_accession,
    )

    mutation_audit_path: Optional[Path] = None
    if args.assignment_mode != "distance":
        mutation_audit_path = write_branch_outputs(
            args.output_dir,
            aligned_cluster_refs,
            result.assigned_query_sequences_by_reference,
            assignment_tag,
        )

    translated_n = sum(1 for ref in cluster_refs if ref.source_type == "nt_translated")
    print(
        f"Reference records: {len(cluster_refs)} total; translated from nucleotide: {translated_n}; already AA: {len(cluster_refs) - translated_n}"
    )
    print(f"Query records processed: {len(records)} of {len(records_all)} total")
    if inferred_alignment_reason:
        print(f"Auto-detected reference realignment: {inferred_alignment_reason}")
    if effective_align_accession:
        print(
            f"Reference alignment mode: pad to query accession {effective_align_accession}; aligned length: {anchor_alignment_len}"
        )
    else:
        print(
            f"Reference alignment mode: assume pre-aligned references; normalized length: {anchor_alignment_len}"
        )
    print(f"Assignment mode: {args.assignment_mode}")
    print(f"Summary table: {outputs['summary']}")
    print(f"Assignment table: {outputs['assignments']}")
    print(f"Ignored FASTA: {outputs['ignored']}")
    print(f"Ignored unique FASTA: {outputs['ignored_unique']}")
    if "distance_diagnostics" in outputs:
        print(f"Distance diagnostics: {outputs['distance_diagnostics']}")
    if "debug_pairs" in outputs:
        print(f"Debug nearest-pairs FASTA: {outputs['debug_pairs']}")
    print(f"Exported reference FASTA: {translated_cluster_path}")
    print(f"Exported unique reference FASTA: {dedup_translated_cluster_path}")
    if mutation_audit_path is not None:
        print(f"Defining-mutation assignment audit: {mutation_audit_path}")


if __name__ == "__main__":
    main()