"""Create H3N2 lineage subalignments from an AA alignment and metadata.

How to use
----------
1) Set the file paths in the user-parameter section:
	- FASTA_PATH: aligned HA amino-acid FASTA (record IDs should match metadata accession values)
	- METADATA_PATH: tab-separated metadata with clade/subtype/accession columns
	- OUTPUT_DIR: destination directory for FASTA/TSV outputs
	- REFERENCE_PATH: canonical H3 reference FASTA used for position labeling
2) Adjust filtering options (lineage aliases, subtype, and forbidden mutation rules).
3) Run: python build_lineage_subalignments_with_metadata.py

What it produces
----------------
- Per-lineage raw FASTA files (all matched lineage records)
- Per-lineage filtered FASTA files (records after forbidden-mutation exclusion)
- unmatched_accessions.fasta
- lineage_subalignment_summary.tsv
- record_matches.tsv
- filtered_sequences.tsv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from Bio import SeqIO
import sys
# add the path to Functions_HuggingFace.py if it's not in the same directory

sys.path.append('/home3/oml4h/PLM_SARS-CoV-2/')

from Functions_HuggingFace import create_h3_numbering_map


# ---- User parameters ----
# Input aligned protein FASTA (record.id is used as the accession key for metadata matching).
FASTA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/Protein-alignment/sgt_4_HA_CDS.fasta"
# Input metadata TSV (must contain clade + subtype + at least one accession column).
METADATA_PATH = "/home4/lm305z/IAV_DB/flu_vgtk_integrations/tmp/gisaid-data/metadata.tsv"
# Output folder where lineage FASTAs and reports are written.
OUTPUT_DIR = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_and_genbank_data"
# Canonical H3 reference sequence used to map canonical mutation labels (e.g., K189R).
REFERENCE_PATH = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

# Keep only records whose metadata subtype exactly equals this value.
SUBTYPE_FILTER = "H3N2"
# Canonical output lineage label -> accepted clade/lineage aliases.
#
# Matching behavior:
# - Metadata Clade values are split on "/" (e.g., "3C... / K" -> ["3C...", "K"]).
# - All whitespace is removed from tokens before matching.
# - Matching is exact against aliases after normalization.
# - Output files are named by the dict key (canonical label).
#
# Example:
# "K": ["K", "3C.2a1b.2a.2a.3a.1"]
TARGET_LINEAGE_ALIASES = {
	"J.2": ["J.2"],
	"J.2.4": ["J.2.4" ],
	"K": ["K","J.2.4.1", "3C.2a1b.2a.2a.3a.1"],
}
# HA2 boundary passed to create_h3_numbering_map for canonical position labeling.
HA2_START = 330

# Optional per-lineage exclusion rules in canonical-mutation notation (OrigPosNew).
# If a sequence carries any listed mutation for that lineage, it is excluded from *_filtered.fasta
# and logged in filtered_sequences.tsv.
FORBIDDEN_CANONICAL_MUTATIONS = {
	"J.2": ["K189R"],
	"J.2.4": [ "S144N","I160K", "N158D", "T328A"],
}

# Candidate metadata columns that may contain the accession used in FASTA headers.
# The script checks these names after normalization (case/space-insensitive).
ACCESSION_COLUMNS = [
	"Isolate_Id",
	"HA INSDC_Upload",
	"HA_INSDC_Upload",
	"HA INSDC Upload",
]


@dataclass
class MatchResult:
	record_id: str
	matched_accession: Optional[str]
	clade: Optional[str]
	subtype: Optional[str]
	matched_targets: str


def normalize_column(col: str) -> str:
	return col.strip().lower().replace(" ", "_")


def normalize_lineage_label(label: str) -> str:
	return "".join(label.split()).lower()


def find_metadata_columns(df: pd.DataFrame) -> Dict[str, str]:
	normalized = {normalize_column(c): c for c in df.columns}
	result: Dict[str, str] = {}
	for col in ACCESSION_COLUMNS:
		norm = normalize_column(col)
		if norm in normalized:
			result[col] = normalized[norm]
	for key in ["Clade", "Subtype"]:
		norm = normalize_column(key)
		if norm in normalized:
			result[key] = normalized[norm]
	return result


def split_clade_tokens(clade_value: str) -> List[str]:
	tokens = [token.strip() for token in clade_value.split("/")]
	return [token for token in tokens if token]


def build_lineage_alias_lookup(
	target_aliases: Dict[str, List[str]]
) -> Dict[str, str]:
	alias_to_target: Dict[str, str] = {}
	for target, aliases in target_aliases.items():
		for alias in [target, *aliases]:
			norm_alias = normalize_lineage_label(alias)
			existing = alias_to_target.get(norm_alias)
			if existing is not None and existing != target:
				raise ValueError(
					f"Alias '{alias}' is assigned to both '{existing}' and '{target}'."
				)
			alias_to_target[norm_alias] = target
	return alias_to_target


def build_accession_lookup(
	df: pd.DataFrame, accession_cols: Iterable[str]
) -> Dict[str, Dict[str, str]]:
	lookup: Dict[str, Dict[str, str]] = {}
	for _, row in df.iterrows():
		for col in accession_cols:
			value = row.get(col)
			if pd.isna(value):
				continue
			accession = str(value).strip()
			if not accession:
				continue
			if accession not in lookup:
				lookup[accession] = row.to_dict()
	return lookup


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


def build_reference_label_map(ref_sequence: str) -> Dict[str, str]:
	ref_map = create_h3_numbering_map(ref_sequence, ref_sequence, HA2_start=HA2_START)
	label_to_residue: Dict[str, str] = {}
	for ref_pos, label in ref_map.items():
		label_to_residue[label] = ref_sequence[ref_pos]
	return label_to_residue


def parse_canonical_mutation(mutation: str) -> Optional[Dict[str, str]]:
	if len(mutation) < 3:
		return None
	orig = mutation[0]
	new = mutation[-1]
	pos = mutation[1:-1]
	if not pos.isdigit():
		return None
	return {"orig": orig, "pos": pos, "new": new}


def sequence_has_forbidden_mutation(
	sequence: str,
	ref_sequence: str,
	ref_label_map: Dict[str, str],
	forbidden: List[str],
) -> Tuple[bool, List[str]]:
	ungapped = sequence.replace("-", "")
	query_map = create_h3_numbering_map(ungapped, ref_sequence, HA2_start=HA2_START)
	label_to_query_residue = {
		label: ungapped[pos]
		for pos, label in query_map.items()
		if pos < len(ungapped)
	}

	triggered: List[str] = []
	for mutation in forbidden:
		parsed = parse_canonical_mutation(mutation)
		if not parsed:
			continue
		label = parsed["pos"]
		ref_residue = ref_label_map.get(label)
		query_residue = label_to_query_residue.get(label)
		if ref_residue is None or query_residue is None:
			continue
		if ref_residue != parsed["orig"]:
			continue
		if query_residue == parsed["new"]:
			triggered.append(mutation)

	return len(triggered) > 0, triggered


def main() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)

	metadata = pd.read_csv(METADATA_PATH, sep="\t", dtype=str, low_memory=False)
	metadata_cols = find_metadata_columns(metadata)
	alias_lookup = build_lineage_alias_lookup(TARGET_LINEAGE_ALIASES)
	target_lineages = list(TARGET_LINEAGE_ALIASES.keys())

	if "Clade" not in metadata_cols or "Subtype" not in metadata_cols:
		raise ValueError(
			"Metadata must include Clade and Subtype columns. Found: "
			f"{list(metadata_cols.values())}"
		)

	accession_cols = [metadata_cols[c] for c in ACCESSION_COLUMNS if c in metadata_cols]
	if not accession_cols:
		raise ValueError(
			"No accession columns found in metadata. Checked: "
			f"{ACCESSION_COLUMNS}"
		)

	lookup = build_accession_lookup(metadata, accession_cols)

	ref_sequence = load_reference_sequence(REFERENCE_PATH)
	ref_label_map = build_reference_label_map(ref_sequence)

	records = parse_alignment_records(FASTA_PATH)

	lineage_records: Dict[str, List] = {ln: [] for ln in target_lineages}
	filtered_records: Dict[str, List] = {ln: [] for ln in target_lineages}
	unmatched_records: List = []
	match_results: List[MatchResult] = []
	filtered_log: List[Dict[str, str]] = []

	for record in records:
		accession = record.id
		row = lookup.get(accession)
		if row is None:
			unmatched_records.append(record)
			match_results.append(MatchResult(accession, None, None, None, ""))
			continue

		clade = str(row.get(metadata_cols["Clade"], "")).strip()
		subtype = str(row.get(metadata_cols["Subtype"], "")).strip()

		matched_targets = []
		for token in split_clade_tokens(clade):
			target = alias_lookup.get(normalize_lineage_label(token))
			if target and target not in matched_targets:
				matched_targets.append(target)

		match_results.append(
			MatchResult(accession, accession, clade, subtype, ",".join(matched_targets))
		)

		if subtype != SUBTYPE_FILTER:
			continue

		for target in matched_targets:
			lineage_records[target].append(record)
			forbidden = FORBIDDEN_CANONICAL_MUTATIONS.get(target, [])
			if forbidden:
				has_forbidden, triggered = sequence_has_forbidden_mutation(
					str(record.seq),
					ref_sequence,
					ref_label_map,
					forbidden,
				)
				if has_forbidden:
					filtered_log.append({
						"record_id": record.id,
						"lineage": target,
						"forbidden_mutations": ",".join(triggered),
					})
					continue
			filtered_records[target].append(record)

	summary_rows = []
	for lineage, recs in lineage_records.items():
		output_path = os.path.join(OUTPUT_DIR, f"H3N2_{lineage}_raw.fasta")
		SeqIO.write(recs, output_path, "fasta")
		filtered_path = os.path.join(OUTPUT_DIR, f"H3N2_{lineage}_filtered.fasta")
		SeqIO.write(filtered_records[lineage], filtered_path, "fasta")
		summary_rows.append({
			"lineage": lineage,
			"count": len(recs),
			"filtered_count": len(filtered_records[lineage]),
			"output": output_path,
			"filtered_output": filtered_path,
		})

	unmatched_path = os.path.join(OUTPUT_DIR, "unmatched_accessions.fasta")
	SeqIO.write(unmatched_records, unmatched_path, "fasta")

	summary_df = pd.DataFrame(summary_rows)
	summary_df.to_csv(
		os.path.join(OUTPUT_DIR, "lineage_subalignment_summary.tsv"),
		sep="\t",
		index=False,
	)

	match_df = pd.DataFrame([
		{
			"record_id": m.record_id,
			"matched_accession": m.matched_accession,
			"clade": m.clade,
			"subtype": m.subtype,
			"matched_targets": m.matched_targets,
		}
		for m in match_results
	])
	match_df.to_csv(
		os.path.join(OUTPUT_DIR, "record_matches.tsv"),
		sep="\t",
		index=False,
	)

	filtered_df = pd.DataFrame(filtered_log)
	filtered_df.to_csv(
		os.path.join(OUTPUT_DIR, "filtered_sequences.tsv"),
		sep="\t",
		index=False,
	)

	print("Done.")
	print(summary_df)
	print(f"Unmatched records: {len(unmatched_records)}")
	print(f"Filtered sequences: {len(filtered_log)}")


if __name__ == "__main__":
	main()
