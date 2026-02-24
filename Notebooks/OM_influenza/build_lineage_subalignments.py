"""Assign aligned HA sequences to nearest lineage references by Hamming distance.

Workflow
--------
1) Parse lineage labels from CLUSTER_PATH FASTA headers (text after final "|").
2) Use ALIGN_ACCESSION from FASTA_PATH as the anchor alignment template.
3) Pairwise-align each cluster reference to the anchor (ungapped), then pad into the
	anchor's gapped coordinate space.
4) For each sequence in FASTA_PATH, pick the nearest padded cluster reference by
	Hamming distance; keep only assignments with distance <= MAX_MUTATIONS.
5) Write one output FASTA per assigned lineage plus ignored sequences.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import Align
from Bio import SeqIO
from Bio.Seq import Seq


# ---- User parameters ----
FASTA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_AA.fasta"
# Accession in FASTA_PATH used as the anchor gapped alignment coordinate system.
ALIGN_ACCESSION = "PV511679"
CLUSTER_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/OM_list_cluster_nuc_plus_internal_only.fa"
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_19feb26/hard"
ASSIGNMENT_MODE = "hard"  # "soft" or "hard"
HARD_MAX_NEXT_MUTATIONS = 2
# Maximum allowed Hamming distance to assign a sequence to a lineage reference.
MAX_MUTATIONS = 10
LINEAGE_ALIAS = {
	"J.2.4.1": "K",
}

# Assignment policy for branch-defining mutations:
# - "soft": require lineage-defining mutations for the candidate lineage and
#   reject assignment only if all defining mutations of the NEXT lineage are present.
# - "hard": require lineage-defining mutations for the candidate lineage and
#   reject assignment if more than HARD_MAX_NEXT_MUTATIONS defining mutations of the
#   NEXT lineage are present.

# Allow limited missing defining mutations only for the terminal lineage reference
# (useful when terminal branch contains uncertain/singleton-like private changes).
FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH = 1

ALIGN_OPEN_GAP_SCORE = -10
ALIGN_EXTEND_GAP_SCORE = -0.5
ALIGN_MATCH_SCORE = 2
ALIGN_MISMATCH_SCORE = -1

TEST_MODE = False #True #False #True #False
TEST_SAMPLE_SIZE = 20000
RANDOM_SEED = 42

DEBUG_TOP_N = 15


BranchMutation = Tuple[int, str, str]


@dataclass
class ClusterRef:
	record_id: str
	lineage: str
	sequence: str
	source_type: str
	order_index: int


def parse_alignment_records(fasta_path: str) -> List:
	records = []
	for record in SeqIO.parse(fasta_path, "fasta"):
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


def parse_cluster_references(cluster_path: str) -> List[ClusterRef]:
	refs: List[ClusterRef] = []
	for idx, record in enumerate(SeqIO.parse(cluster_path, "fasta"), start=1):
		header = record.id.strip()
		parts = header.split("|")
		# Take lineage label from header suffix after final '|'.
		lineage = parts[-1] if parts else header
		lineage = LINEAGE_ALIAS.get(lineage or "", lineage or "")
		raw_seq = str(record.seq)
		if is_probably_nucleotide(raw_seq):
			parsed_seq = to_protein_sequence(raw_seq)
			source_type = "nt_translated"
		else:
			parsed_seq = raw_seq.replace("-", "")
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
	return refs


def build_anchor_aligned_sequence(records: List, accession: str) -> str:
	for record in records:
		if record.id == accession:
			return str(record.seq)
	raise ValueError(f"ALIGN_ACCESSION {accession} not found in FASTA_PATH alignment.")


def pad_cluster_to_anchor_alignment(
	cluster_seq: str,
	anchor_alignment: str,
	match_score: float = ALIGN_MATCH_SCORE,
	mismatch_score: float = ALIGN_MISMATCH_SCORE,
	gap_open: float = -10,
	gap_extend: float = -0.5,
) -> str:
	anchor_ungapped = anchor_alignment.replace("-", "")
	aligner = Align.PairwiseAligner()
	aligner.mode = "global"
	aligner.match_score = match_score
	aligner.mismatch_score = mismatch_score
	aligner.open_gap_score = gap_open
	aligner.extend_gap_score = gap_extend
	# Semiglobal/overlap-like behavior: avoid over-penalizing unmatched ends
	# when references are partial relative to the anchor.
	aligner.target_end_gap_score = 0.0
	aligner.query_end_gap_score = 0.0
	alignment = aligner.align(anchor_ungapped, cluster_seq)[0]

	aligned_fasta = alignment.format("fasta")
	aligned_lines = [
		line.strip()
		for line in aligned_fasta.splitlines()
		if line.strip() and not line.startswith(">")
	]
	if len(aligned_lines) < 2:
		raise ValueError("Failed to reconstruct pairwise alignment strings.")
	aligned_anchor = aligned_lines[0]
	aligned_cluster = aligned_lines[1]

	cluster_by_anchor_pos: List[str] = []
	for anchor_char, cluster_char in zip(aligned_anchor, aligned_cluster):
		if anchor_char != "-":
			cluster_by_anchor_pos.append(cluster_char)
		else:
			continue

	if len(cluster_by_anchor_pos) < len(anchor_ungapped):
		cluster_by_anchor_pos.extend(["-"] * (len(anchor_ungapped) - len(cluster_by_anchor_pos)))

	padded: List[str] = []
	anchor_pos = 0
	for anchor_char in anchor_alignment:
		if anchor_char == "-":
			padded.append("-")
		else:
			padded.append(
				cluster_by_anchor_pos[anchor_pos]
				if anchor_pos < len(cluster_by_anchor_pos)
				else "-"
			)
			anchor_pos += 1
	return "".join(padded)


def hamming_distance(seq_a: str, seq_b: str) -> int:
	return sum(1 for a, b in zip(seq_a, seq_b) if a != b)


def normalize_length(seq: str, target_len: int) -> str:
	if len(seq) == target_len:
		return seq
	if len(seq) < target_len:
		return seq + ("-" * (target_len - len(seq)))
	return seq[:target_len]


def safe_label(label: str) -> str:
	cleaned = label.strip().replace(" ", "_")
	return cleaned.replace("/", "-")


def deduplicate_records(records: List) -> List:
	unique_by_sequence: Dict[str, object] = {}
	for record in records:
		seq_key = str(record.seq)
		if seq_key not in unique_by_sequence:
			unique_by_sequence[seq_key] = record
	return list(unique_by_sequence.values())


def branch_mutations(parent_seq: str, child_seq: str) -> List[BranchMutation]:
	muts: List[BranchMutation] = []
	for pos, (parent_aa, child_aa) in enumerate(zip(parent_seq, child_seq)):
		if parent_aa != child_aa:
			muts.append((pos, parent_aa, child_aa))
	return muts


def has_all_mutations(sequence: str, mutations: List[BranchMutation]) -> bool:
	for pos, _, child_aa in mutations:
		if pos >= len(sequence) or sequence[pos] != child_aa:
			return False
	return True


def count_missing_mutations(sequence: str, mutations: List[BranchMutation]) -> int:
	missing = 0
	for pos, _, child_aa in mutations:
		if pos >= len(sequence) or sequence[pos] != child_aa:
			missing += 1
	return missing


def count_present_mutations(sequence: str, mutations: List[BranchMutation]) -> int:
	present = 0
	for pos, _, child_aa in mutations:
		if pos < len(sequence) and sequence[pos] == child_aa:
			present += 1
	return present


def mode_tag(mode: str, hard_max_next: int) -> str:
	if mode == "hard":
		return f"hard_nextle{hard_max_next}"
	return "soft"


def validate_mode(mode: str) -> None:
	if mode not in {"soft", "hard"}:
		raise ValueError("ASSIGNMENT_MODE must be 'soft' or 'hard'.")


def build_branch_mutation_profiles(
	aligned_refs: List[ClusterRef],
) -> Tuple[List[List[BranchMutation]], List[List[BranchMutation]], List[Dict[str, str]]]:
	branch_by_ref: List[List[BranchMutation]] = []
	cumulative_by_ref: List[List[BranchMutation]] = []
	inheritance_rows: List[Dict[str, str]] = []

	root_seq = aligned_refs[0].sequence if aligned_refs else ""
	for idx, ref in enumerate(aligned_refs):
		if idx == 0:
			branch_by_ref.append([])
			cumulative_by_ref.append([])
			inheritance_rows.append({
				"order_index": str(ref.order_index),
				"record_id": ref.record_id,
				"lineage": ref.lineage,
				"status": "root",
				"missing_previous_branch_mutations": "",
				"previous_branch_size": "0",
			})
			continue

		parent = aligned_refs[idx - 1]
		branch = branch_mutations(parent.sequence, ref.sequence)
		branch_by_ref.append(branch)
		# Net defining states from root -> current reference.
		# This avoids impossible constraints when the same site changes multiple times
		# along the branch path (e.g., A->B then B->C).
		cumulative_by_ref.append(branch_mutations(root_seq, ref.sequence))

		if idx >= 2:
			prior_branch = branch_by_ref[idx - 1]
			missing_prior = [
				f"{src}{pos + 1}{dst}"
				for pos, src, dst in prior_branch
				if ref.sequence[pos] != dst
			]
			status = "ok" if not missing_prior else "violation"
			inheritance_rows.append({
				"order_index": str(ref.order_index),
				"record_id": ref.record_id,
				"lineage": ref.lineage,
				"status": status,
				"missing_previous_branch_mutations": ",".join(missing_prior),
				"previous_branch_size": str(len(prior_branch)),
			})
		else:
			inheritance_rows.append({
				"order_index": str(ref.order_index),
				"record_id": ref.record_id,
				"lineage": ref.lineage,
				"status": "ok",
				"missing_previous_branch_mutations": "",
				"previous_branch_size": "0",
			})

	return branch_by_ref, cumulative_by_ref, inheritance_rows


def passes_mode_criteria(
	query_seq: str,
	ref_index: int,
	branch_by_ref: List[List[BranchMutation]],
	cumulative_by_ref: List[List[BranchMutation]],
	mode: str,
	hard_max_next: int,
	final_lineage_max_missing_current_branch: int,
) -> Tuple[bool, int, int]:
	current_branch = branch_by_ref[ref_index]
	current_missing = count_missing_mutations(query_seq, current_branch)
	is_terminal = ref_index == (len(branch_by_ref) - 1)
	if is_terminal:
		if current_missing > final_lineage_max_missing_current_branch:
			return False, 0, len(current_branch)
	else:
		if current_missing > 0:
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

	if mode == "hard":
		if next_present > hard_max_next:
			return False, next_present, next_total
		return True, next_present, next_total

	return False, next_present, next_total


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	validate_mode(ASSIGNMENT_MODE)
	assignment_tag = mode_tag(ASSIGNMENT_MODE, HARD_MAX_NEXT_MUTATIONS)

	cluster_refs = parse_cluster_references(CLUSTER_PATH)
	lineages = list(dict.fromkeys(ref.lineage for ref in cluster_refs))
	lineage_records: Dict[str, List] = {ln: [] for ln in lineages}
	assignments: List[Dict[str, str]] = []
	ignored_records: List = []
	best_matches: List[Dict[str, str]] = []
	assigned_query_sequences_by_reference: Dict[str, List[str]] = {
		ref.record_id: [] for ref in cluster_refs
	}

	translated_cluster_path = os.path.join(
		OUTPUT_DIR,
		f"cluster_references_protein_translated_all_{assignment_tag}.fasta",
	)
	with open(translated_cluster_path, "w", encoding="utf-8") as handle:
		for ref in cluster_refs:
			handle.write(
				f">{ref.record_id}|lineage={ref.lineage}|source={ref.source_type}|order={ref.order_index}\n"
			)
			handle.write(f"{ref.sequence}\n")

	dedup_translated_cluster_path = os.path.join(
		OUTPUT_DIR,
		f"cluster_references_protein_translated_unique_{assignment_tag}.fasta",
	)
	unique_proteins: Dict[str, List[ClusterRef]] = {}
	for ref in cluster_refs:
		unique_proteins.setdefault(ref.sequence, []).append(ref)
	with open(dedup_translated_cluster_path, "w", encoding="utf-8") as handle:
		for idx, (protein_seq, refs_for_seq) in enumerate(unique_proteins.items(), start=1):
			representative = refs_for_seq[0]
			handle.write(
				f">unique_{idx}|rep={representative.record_id}|lineage={representative.lineage}|"
				f"duplicates={len(refs_for_seq)}\n"
			)
			handle.write(f"{protein_seq}\n")

	records_all = parse_alignment_records(FASTA_PATH)
	anchor_alignment = build_anchor_aligned_sequence(records_all, ALIGN_ACCESSION)
	anchor_ungapped_len = len(anchor_alignment.replace("-", ""))
	translated_n = sum(1 for ref in cluster_refs if ref.source_type == "nt_translated")
	print(
		f"Cluster refs: {len(cluster_refs)} total; translated from nucleotide: {translated_n}; "
		f"already AA: {len(cluster_refs) - translated_n}"
	)
	print(
		f"Anchor aligned length: {len(anchor_alignment)} (ungapped: {anchor_ungapped_len})"
	)

	records = records_all
	if TEST_MODE:
		random.seed(RANDOM_SEED)
		anchor_record = None
		non_anchor_records: List = []
		for record in records_all:
			if record.id == ALIGN_ACCESSION:
				anchor_record = record
			else:
				non_anchor_records.append(record)

		if anchor_record is None:
			raise ValueError(
				f"ALIGN_ACCESSION {ALIGN_ACCESSION} not found in FASTA_PATH alignment."
			)

		target_n = min(TEST_SAMPLE_SIZE, len(records_all))
		if target_n <= 1:
			records = [anchor_record]
		else:
			sampled_non_anchor = random.sample(
				non_anchor_records,
				min(target_n - 1, len(non_anchor_records)),
			)
			records = [anchor_record, *sampled_non_anchor]

	anchor_alignment_len = len(anchor_alignment)

	aligned_cluster_refs: List[ClusterRef] = []
	for ref in cluster_refs:
		padded_seq = pad_cluster_to_anchor_alignment(
			ref.sequence,
			anchor_alignment,
			gap_open=ALIGN_OPEN_GAP_SCORE,
			gap_extend=ALIGN_EXTEND_GAP_SCORE,
		)
		aligned_cluster_refs.append(
			ClusterRef(
				record_id=ref.record_id,
				lineage=ref.lineage,
				sequence=padded_seq,
				source_type=ref.source_type,
				order_index=ref.order_index,
			)
		)

	branch_by_ref, cumulative_by_ref, inheritance_rows = build_branch_mutation_profiles(
		aligned_cluster_refs
	)

	defining_rows: List[Dict[str, str]] = []
	for idx, ref in enumerate(aligned_cluster_refs):
		for pos, parent_aa, child_aa in branch_by_ref[idx]:
			defining_rows.append({
				"order_index": str(ref.order_index),
				"record_id": ref.record_id,
				"lineage": ref.lineage,
				"mutation": f"{parent_aa}{pos + 1}{child_aa}",
				"position_1_based": str(pos + 1),
				"from_aa": parent_aa,
				"to_aa": child_aa,
			})

	defining_df = pd.DataFrame(defining_rows)
	defining_df.to_csv(
		os.path.join(
			OUTPUT_DIR,
			f"cluster_branch_defining_mutations_{assignment_tag}.tsv",
		),
		sep="\t",
		index=False,
	)

	inheritance_df = pd.DataFrame(inheritance_rows)
	inheritance_df.to_csv(
		os.path.join(
			OUTPUT_DIR,
			f"cluster_branch_inheritance_check_{assignment_tag}.tsv",
		),
		sep="\t",
		index=False,
	)

	violations = inheritance_df.loc[inheritance_df["status"] == "violation"]
	print(
		f"Branch inheritance checks: {len(inheritance_df)} rows, violations: {len(violations)}"
	)
	if len(violations) > 0:
		print("Warning: some references do not carry all mutations from the previous branch.")

	for record in records:
		query_seq = normalize_length(str(record.seq), anchor_alignment_len)
		nearest_ref: Optional[ClusterRef] = None
		nearest_distance: Optional[int] = None
		nearest_ref_index: Optional[int] = None

		best_ref: Optional[ClusterRef] = None
		best_distance: Optional[int] = None
		best_ref_index: Optional[int] = None
		best_next_present = 0
		best_next_total = 0

		for ref_index, ref in enumerate(aligned_cluster_refs):
			distance = hamming_distance(query_seq, ref.sequence)
			if nearest_distance is None or distance < nearest_distance:
				nearest_distance = distance
				nearest_ref = ref
				nearest_ref_index = ref_index

			passes, next_present, next_total = passes_mode_criteria(
				query_seq,
				ref_index,
				branch_by_ref,
				cumulative_by_ref,
				ASSIGNMENT_MODE,
				HARD_MAX_NEXT_MUTATIONS,
				FINAL_LINEAGE_MAX_MISSING_CURRENT_BRANCH,
			)
			if not passes:
				continue

			if best_distance is None or distance < best_distance:
				best_distance = distance
				best_ref = ref
				best_ref_index = ref_index
				best_next_present = next_present
				best_next_total = next_total

		if nearest_distance is None or nearest_ref is None or nearest_ref_index is None:
			ignored_records.append(record)
			continue

		if best_distance is None or best_ref is None or best_ref_index is None:
			ignored_records.append(record)
			assignments.append({
				"record_id": record.id,
				"assigned_lineage": "",
				"best_reference": nearest_ref.record_id,
				"mutation_count": str(nearest_distance),
				"status": "ignored_mode_rules",
				"assignment_mode": ASSIGNMENT_MODE,
				"hard_max_next_mutations": str(HARD_MAX_NEXT_MUTATIONS),
				"next_branch_present": "",
				"next_branch_total": "",
			})
			continue

		best_matches.append({
			"record_id": record.id,
			"best_reference": best_ref.record_id,
			"assigned_lineage": best_ref.lineage,
			"mutation_count": str(best_distance),
			"assignment_mode": ASSIGNMENT_MODE,
			"hard_max_next_mutations": str(HARD_MAX_NEXT_MUTATIONS),
			"next_branch_present": str(best_next_present),
			"next_branch_total": str(best_next_total),
			"query_sequence": query_seq,
			"reference_sequence": best_ref.sequence,
		})

		if best_distance > MAX_MUTATIONS:
			ignored_records.append(record)
			assignments.append({
				"record_id": record.id,
				"assigned_lineage": "",
				"best_reference": best_ref.record_id,
				"mutation_count": str(best_distance),
				"status": "ignored",
				"assignment_mode": ASSIGNMENT_MODE,
				"hard_max_next_mutations": str(HARD_MAX_NEXT_MUTATIONS),
				"next_branch_present": str(best_next_present),
				"next_branch_total": str(best_next_total),
			})
			continue

		lineage_records[best_ref.lineage].append(record)
		assigned_query_sequences_by_reference.setdefault(best_ref.record_id, []).append(query_seq)
		assignments.append({
			"record_id": record.id,
			"assigned_lineage": best_ref.lineage,
			"best_reference": best_ref.record_id,
			"mutation_count": str(best_distance),
			"status": "assigned",
			"assignment_mode": ASSIGNMENT_MODE,
			"hard_max_next_mutations": str(HARD_MAX_NEXT_MUTATIONS),
			"next_branch_present": str(best_next_present),
			"next_branch_total": str(best_next_total),
		})

	summary_rows = []
	for lineage, recs in lineage_records.items():
		safe_lineage = safe_label(lineage)
		output_path = os.path.join(
			OUTPUT_DIR,
			f"H3N2_{safe_lineage}_{assignment_tag}_max{MAX_MUTATIONS}.fasta",
		)
		SeqIO.write(recs, output_path, "fasta")
		unique_recs = deduplicate_records(recs)
		unique_output_path = os.path.join(
			OUTPUT_DIR,
			f"H3N2_{safe_lineage}_{assignment_tag}_max{MAX_MUTATIONS}_unique.fasta",
		)
		SeqIO.write(unique_recs, unique_output_path, "fasta")
		summary_rows.append({
			"lineage": lineage,
			"count": len(recs),
			"unique_count": len(unique_recs),
			"assignment_mode": ASSIGNMENT_MODE,
			"hard_max_next_mutations": HARD_MAX_NEXT_MUTATIONS,
			"output": output_path,
			"unique_output": unique_output_path,
		})

	ignored_path = os.path.join(
		OUTPUT_DIR,
		f"ignored_{assignment_tag}_over_max{MAX_MUTATIONS}.fasta",
	)
	SeqIO.write(ignored_records, ignored_path, "fasta")
	ignored_unique_path = os.path.join(
		OUTPUT_DIR,
		f"ignored_{assignment_tag}_over_max{MAX_MUTATIONS}_unique.fasta",
	)
	ignored_unique_records = deduplicate_records(ignored_records)
	SeqIO.write(ignored_unique_records, ignored_unique_path, "fasta")

	summary_df = pd.DataFrame(summary_rows)
	summary_df.to_csv(
		os.path.join(
			OUTPUT_DIR,
			f"lineage_subalignment_summary_{assignment_tag}.tsv",
		),
		sep="\t",
		index=False,
	)

	assign_df = pd.DataFrame(assignments)
	assign_df.to_csv(
		os.path.join(
			OUTPUT_DIR,
			f"lineage_assignments_{assignment_tag}.tsv",
		),
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
			present_n = sum(
				1
				for query in assigned_queries
				if pos < len(query) and query[pos] == child_aa
			)
			support_fraction = (present_n / n_assigned) if n_assigned > 0 else float("nan")
			always_ignored = n_assigned > 0 and present_n == 0
			mutation_audit_rows.append({
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
			})

	mutation_audit_df = pd.DataFrame(mutation_audit_rows)
	mutation_audit_path = os.path.join(
		OUTPUT_DIR,
		f"defining_mutation_assignment_audit_{assignment_tag}.tsv",
	)
	mutation_audit_df.to_csv(
		mutation_audit_path,
		sep="\t",
		index=False,
	)

	always_ignored_df = mutation_audit_df.loc[
		mutation_audit_df["always_ignored_flag"] == "yes"
	]
	print(
		"Defining-mutation assignment audit: "
		f"{len(mutation_audit_df)} rows; always-ignored flags: {len(always_ignored_df)}"
	)
	if len(always_ignored_df) > 0:
		print(
			"Warning: some defining mutations are never present in sequences assigned "
			"to that reference. Check: "
			f"{mutation_audit_path}"
		)

	best_df = pd.DataFrame(best_matches)
	if not best_df.empty:
		best_df = best_df.sort_values("mutation_count", key=lambda s: pd.to_numeric(s, errors="coerce"))
		best_df.to_csv(
			os.path.join(OUTPUT_DIR, f"distance_diagnostics_{assignment_tag}.tsv"),
			sep="\t",
			index=False,
		)

		top_n = best_df.head(DEBUG_TOP_N)
		print(f"Top {len(top_n)} closest sequences by Hamming distance:")
		for _, row in top_n.iterrows():
			print(
				f"  {row['record_id']} -> {row['assigned_lineage']} "
				f"[{row['best_reference']}] d={row['mutation_count']}"
			)

		debug_alignment_path = os.path.join(
			OUTPUT_DIR,
			f"debug_top{DEBUG_TOP_N}_nearest_pairs_{assignment_tag}_max{MAX_MUTATIONS}.fasta",
		)
		with open(debug_alignment_path, "w", encoding="utf-8") as handle:
			handle.write(
				f">ANCHOR|{ALIGN_ACCESSION}|aligned_template\n{anchor_alignment}\n"
			)
			for _, row in top_n.iterrows():
				handle.write(
					f">{row['record_id']}|query|d={row['mutation_count']}|lineage={row['assigned_lineage']}\n"
					f"{row['query_sequence']}\n"
				)
				handle.write(
					f">{row['record_id']}|best_ref|{row['best_reference']}\n"
					f"{row['reference_sequence']}\n"
				)
		print(f"Saved debug alignment FASTA: {debug_alignment_path}")
	else:
		print("No best-match diagnostics were produced.")

	print("Done.")
	print(summary_df)
	print(f"Ignored (>{MAX_MUTATIONS} mutations): {len(ignored_records)}")
	print(
		f"Ignored unique (>{MAX_MUTATIONS} mutations): {len(ignored_unique_records)}"
	)
	print(f"Anchor accession used for coordinate padding: {ALIGN_ACCESSION}")
	print(
		f"Assignment mode: {ASSIGNMENT_MODE} "
		f"(hard max next-defining muts: {HARD_MAX_NEXT_MUTATIONS})"
	)
	print(f"Exported translated cluster proteins: {translated_cluster_path}")
	print(f"Exported unique translated proteins: {dedup_translated_cluster_path}")
	print(f"Exported defining-mutation assignment audit: {mutation_audit_path}")
	if TEST_MODE:
		print(f"Test mode: sampled {len(records)} sequences")


if __name__ == "__main__":
	main()
