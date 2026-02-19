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
from typing import Dict, List, Optional

import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.Seq import Seq


# ---- User parameters ----
FASTA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_AA.fasta"
# Accession in FASTA_PATH used as the anchor gapped alignment coordinate system.
ALIGN_ACCESSION = "PV511679"
CLUSTER_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/OM_list_cluster_nuc_plus.fa"
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_16feb26_dryrun"

# Maximum allowed Hamming distance to assign a sequence to a lineage reference.
MAX_MUTATIONS = 5
LINEAGE_ALIAS = {
	"J.2.4.1": "K",
}

ALIGN_OPEN_GAP_SCORE = -10
ALIGN_EXTEND_GAP_SCORE = -0.5

TEST_MODE = False #True #False
TEST_SAMPLE_SIZE = 20000
RANDOM_SEED = 42

DEBUG_TOP_N = 15


@dataclass
class ClusterRef:
	record_id: str
	lineage: str
	sequence: str
	source_type: str


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
	trimmed = cleaned[: (len(cleaned) // 3) * 3]
	if not trimmed:
		return ""
	protein = str(Seq(trimmed).translate(to_stop=False))
	return protein.replace("*", "")


def parse_cluster_references(cluster_path: str) -> List[ClusterRef]:
	refs: List[ClusterRef] = []
	for record in SeqIO.parse(cluster_path, "fasta"):
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
	match_score: float = 2,
	mismatch_score: float = -1,
	gap_open: float = -10,
	gap_extend: float = -0.5,
) -> str:
	anchor_ungapped = anchor_alignment.replace("-", "")
	globalms = getattr(pairwise2.align, "globalms")
	alignment = globalms(
		anchor_ungapped,
		cluster_seq,
		match_score,
		mismatch_score,
		gap_open,
		gap_extend,
		one_alignment_only=True,
	)[0]
	aligned_anchor = alignment.seqA
	aligned_cluster = alignment.seqB

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


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	cluster_refs = parse_cluster_references(CLUSTER_PATH)
	lineages = sorted({ref.lineage for ref in cluster_refs})
	lineage_records: Dict[str, List] = {ln: [] for ln in lineages}
	assignments: List[Dict[str, str]] = []
	ignored_records: List = []
	best_matches: List[Dict[str, str]] = []

	translated_cluster_path = os.path.join(
		OUTPUT_DIR,
		"cluster_references_protein_translated_all.fasta",
	)
	with open(translated_cluster_path, "w", encoding="utf-8") as handle:
		for ref in cluster_refs:
			handle.write(
				f">{ref.record_id}|lineage={ref.lineage}|source={ref.source_type}\n"
			)
			handle.write(f"{ref.sequence}\n")

	dedup_translated_cluster_path = os.path.join(
		OUTPUT_DIR,
		"cluster_references_protein_translated_unique.fasta",
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
		padded_seq = pad_cluster_to_anchor_alignment(ref.sequence, anchor_alignment)
		aligned_cluster_refs.append(
			ClusterRef(
				record_id=ref.record_id,
				lineage=ref.lineage,
				sequence=padded_seq,
				source_type=ref.source_type,
			)
		)

	for record in records:
		query_seq = normalize_length(str(record.seq), anchor_alignment_len)
		best_ref: Optional[ClusterRef] = None
		best_distance: Optional[int] = None

		for ref in aligned_cluster_refs:
			distance = hamming_distance(query_seq, ref.sequence)
			if best_distance is None or distance < best_distance:
				best_distance = distance
				best_ref = ref

		if best_distance is None or best_ref is None:
			ignored_records.append(record)
			continue

		best_matches.append({
			"record_id": record.id,
			"best_reference": best_ref.record_id,
			"assigned_lineage": best_ref.lineage,
			"mutation_count": str(best_distance),
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
			})
			continue

		lineage_records[best_ref.lineage].append(record)
		assignments.append({
			"record_id": record.id,
			"assigned_lineage": best_ref.lineage,
			"best_reference": best_ref.record_id,
			"mutation_count": str(best_distance),
			"status": "assigned",
		})

	summary_rows = []
	for lineage, recs in lineage_records.items():
		safe_lineage = safe_label(lineage)
		output_path = os.path.join(
			OUTPUT_DIR,
			f"H3N2_{safe_lineage}_max{MAX_MUTATIONS}.fasta",
		)
		SeqIO.write(recs, output_path, "fasta")
		unique_recs = deduplicate_records(recs)
		unique_output_path = os.path.join(
			OUTPUT_DIR,
			f"H3N2_{safe_lineage}_max{MAX_MUTATIONS}_unique.fasta",
		)
		SeqIO.write(unique_recs, unique_output_path, "fasta")
		summary_rows.append({
			"lineage": lineage,
			"count": len(recs),
			"unique_count": len(unique_recs),
			"output": output_path,
			"unique_output": unique_output_path,
		})

	ignored_path = os.path.join(OUTPUT_DIR, f"ignored_over_max{MAX_MUTATIONS}.fasta")
	SeqIO.write(ignored_records, ignored_path, "fasta")
	ignored_unique_path = os.path.join(
		OUTPUT_DIR,
		f"ignored_over_max{MAX_MUTATIONS}_unique.fasta",
	)
	ignored_unique_records = deduplicate_records(ignored_records)
	SeqIO.write(ignored_unique_records, ignored_unique_path, "fasta")

	summary_df = pd.DataFrame(summary_rows)
	summary_df.to_csv(
		os.path.join(OUTPUT_DIR, "lineage_subalignment_summary.tsv"),
		sep="\t",
		index=False,
	)

	assign_df = pd.DataFrame(assignments)
	assign_df.to_csv(
		os.path.join(OUTPUT_DIR, "lineage_assignments.tsv"),
		sep="\t",
		index=False,
	)

	best_df = pd.DataFrame(best_matches)
	if not best_df.empty:
		best_df = best_df.sort_values("mutation_count", key=lambda s: pd.to_numeric(s, errors="coerce"))
		best_df.to_csv(
			os.path.join(OUTPUT_DIR, "distance_diagnostics.tsv"),
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
			f"debug_top{DEBUG_TOP_N}_nearest_pairs_max{MAX_MUTATIONS}.fasta",
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
	print(f"Exported translated cluster proteins: {translated_cluster_path}")
	print(f"Exported unique translated proteins: {dedup_translated_cluster_path}")
	if TEST_MODE:
		print(f"Test mode: sampled {len(records)} sequences")


if __name__ == "__main__":
	main()
