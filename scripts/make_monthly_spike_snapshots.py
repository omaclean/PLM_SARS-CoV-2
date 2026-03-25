#!/usr/bin/env python3
"""Create month-based FASTA snapshots by filtering record IDs from metadata dates.

Supported modes:
- `spike`: input FASTA already contains spike CDS nucleotide sequences; the
    script writes nucleotide and translated protein snapshots
- `nucleotide_full`: input FASTA contains full-genome nucleotide sequences; the
    script writes matching nucleotide sequences unchanged, with no extraction or
    transformation

Examples
--------
Create the default June 2025 full-genome snapshot:
    python scripts/make_monthly_spike_snapshots.py

Create June and July 2025 snapshots:
    python scripts/make_monthly_spike_snapshots.py --month 2025-06 --month 2025-07

Run on spike CDS input instead:
    python scripts/make_monthly_spike_snapshots.py \
        --sequence-mode spike \
        --sequence-path Sequences/cog_2025-07-17_spike.fa \
        --metadata-path Sequences/cog_2025-07-17_spike_metadata.csv

Relax the protein length filter in spike mode:
    python scripts/make_monthly_spike_snapshots.py --min-protein-length 1150
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, IO, Iterator, List, Optional, Sequence, Set, Tuple, cast



DEFAULT_SEQUENCE_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_alignment.fa.gz")
DEFAULT_METADATA_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_metadata.csv.gz")
DEFAULT_OUTPUT_DIR = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots/full_genome")
DEFAULT_SEQUENCE_MODE = "nucleotide_full"

DEFAULT_SEQUENCE_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_spike.fa")
DEFAULT_METADATA_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_spike_metadata.csv")
DEFAULT_OUTPUT_DIR = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_month_snapshots")
DEFAULT_SEQUENCE_MODE = "spike"


DEFAULT_MONTHS = ("2025-06",)
DEFAULT_ID_COLUMN = "sequence_name"
DEFAULT_DATE_COLUMN = "sample_date"
DEFAULT_MIN_PROTEIN_LENGTH = 1200
FASTA_WRAP = 100

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class MonthWindow:
    label: str
    start: date
    end: date

    def contains(self, value: date) -> bool:
        return self.start <= value < self.end


@dataclass
class SnapshotStats:
    metadata_rows_selected: int = 0
    duplicate_metadata_ids: int = 0
    fasta_records_matched: int = 0
    sequences_written: int = 0
    skipped_missing_fasta: int = 0
    skipped_short_protein: int = 0
    skipped_bad_coding_length: int = 0


@dataclass
class SnapshotOutputs:
    nucleotide_path: Path
    protein_path: Optional[Path]
    metadata_path: Path
    nucleotide_handle: IO[str]
    protein_handle: Optional[IO[str]]
    metadata_handle: IO[str]
    metadata_writer: csv.DictWriter
    written_ids: Set[str] = field(default_factory=set)


@dataclass
class MetadataSelection:
    fieldnames: List[str]
    rows_by_id: Dict[str, Dict[str, str]]
    ids_by_month: Dict[str, Set[str]]
    stats_by_month: Dict[str, SnapshotStats]
    total_rows: int
    missing_id_rows: int
    missing_date_rows: int
    invalid_date_rows: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create month-based FASTA snapshots by matching metadata dates to FASTA headers."
    )
    parser.add_argument(
        "--sequence-mode",
        choices=("spike", "nucleotide_full"),
        default=DEFAULT_SEQUENCE_MODE,
        help=(
            "Interpret input FASTA as spike CDS (`spike`) or full-genome nucleotide sequences "
            "to be copied unchanged (`nucleotide_full`)."
        ),
    )
    parser.add_argument(
        "--sequence-path",
        type=Path,
        default=DEFAULT_SEQUENCE_PATH,
        help="Input FASTA file to subset by matching metadata-selected IDs.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Metadata CSV matching FASTA record IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for monthly snapshot outputs.",
    )
    parser.add_argument(
        "--month",
        dest="months",
        action="append",
        default=None,
        metavar="YYYY-MM",
        help=(
            "Month snapshot to create in YYYY-MM format. "
            "May be provided multiple times or as a comma-separated list. Default: 2025-06."
        ),
    )
    parser.add_argument(
        "--metadata-id-column",
        default=DEFAULT_ID_COLUMN,
        help="Metadata column matching FASTA IDs. Default: sequence_name.",
    )
    parser.add_argument(
        "--metadata-date-column",
        default=DEFAULT_DATE_COLUMN,
        help="Metadata date column in ISO format. Default: sample_date.",
    )
    parser.add_argument(
        "--min-protein-length",
        type=int,
        default=DEFAULT_MIN_PROTEIN_LENGTH,
        help="Minimum translated protein length to keep. Default: 1200.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on FASTA records scanned, useful for quick tests.",
    )
    return parser


def open_text(path: Path, mode: str) -> IO[str]:
    if "b" in mode:
        raise ValueError("Binary mode is not supported.")
    if path.suffix == ".gz":
        return cast(IO[str], gzip.open(path, mode, newline=""))
    return cast(IO[str], path.open(mode, newline=""))


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalise_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def resolve_column_name(requested: str, fieldnames: Sequence[str]) -> str:
    mapping = {normalise_column_name(name): name for name in fieldnames}
    key = normalise_column_name(requested)
    if key not in mapping:
        available = ", ".join(fieldnames)
        raise KeyError(f"Column {requested!r} not found. Available columns: {available}")
    return mapping[key]


def parse_month(month_text: str) -> MonthWindow:
    try:
        month_start = datetime.strptime(month_text, "%Y-%m").date().replace(day=1)
    except ValueError as exc:
        raise ValueError(f"Invalid month {month_text!r}; expected YYYY-MM.") from exc

    if month_start.month == 12:
        month_end = date(month_start.year + 1, 1, 1)
    else:
        month_end = date(month_start.year, month_start.month + 1, 1)
    return MonthWindow(label=month_text, start=month_start, end=month_end)


def parse_months(values: Optional[Sequence[str]]) -> List[MonthWindow]:
    if not values:
        month_values = list(DEFAULT_MONTHS)
    else:
        month_values = []
        for value in values:
            month_values.extend(part.strip() for part in value.split(",") if part.strip())

    parsed = [parse_month(value) for value in month_values]
    seen: Set[str] = set()
    deduplicated: List[MonthWindow] = []
    for month in parsed:
        if month.label in seen:
            continue
        seen.add(month.label)
        deduplicated.append(month)
    return deduplicated


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_metadata_selection(
    metadata_path: Path,
    months: Sequence[MonthWindow],
    id_column_name: str,
    date_column_name: str,
) -> MetadataSelection:
    stats_by_month = {month.label: SnapshotStats() for month in months}
    ids_by_month = {month.label: set() for month in months}
    rows_by_id: Dict[str, Dict[str, str]] = {}
    total_rows = 0
    missing_id_rows = 0
    missing_date_rows = 0
    invalid_date_rows = 0

    with open_text(metadata_path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file {metadata_path} does not contain a header row.")
        fieldnames = list(reader.fieldnames)
        id_column = resolve_column_name(id_column_name, fieldnames)
        date_column = resolve_column_name(date_column_name, fieldnames)

        for row in reader:
            total_rows += 1
            record_id = (row.get(id_column) or "").strip()
            if not record_id:
                missing_id_rows += 1
                continue

            date_text = (row.get(date_column) or "").strip()
            if not date_text:
                missing_date_rows += 1
                continue

            try:
                sample_date = parse_iso_date(date_text)
            except ValueError:
                invalid_date_rows += 1
                continue

            matched_month: Optional[MonthWindow] = None
            for month in months:
                if month.contains(sample_date):
                    matched_month = month
                    break
            if matched_month is None:
                continue

            month_stats = stats_by_month[matched_month.label]
            month_stats.metadata_rows_selected += 1
            if record_id in rows_by_id:
                month_stats.duplicate_metadata_ids += 1
                continue

            rows_by_id[record_id] = row
            ids_by_month[matched_month.label].add(record_id)

    return MetadataSelection(
        fieldnames=fieldnames,
        rows_by_id=rows_by_id,
        ids_by_month=ids_by_month,
        stats_by_month=stats_by_month,
        total_rows=total_rows,
        missing_id_rows=missing_id_rows,
        missing_date_rows=missing_date_rows,
        invalid_date_rows=invalid_date_rows,
    )


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    with open_text(path, "rt") as handle:
        header: Optional[str] = None
        chunks: List[str] = []

        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

        if header is not None:
            yield header, "".join(chunks)


def fasta_identifier(header: str) -> str:
    return header.split()[0]


def write_fasta_record(handle: IO[str], header: str, sequence: str) -> None:
    handle.write(f">{header}\n")
    for start in range(0, len(sequence), FASTA_WRAP):
        handle.write(sequence[start : start + FASTA_WRAP] + "\n")


def translate_spike(sequence: str) -> str:
    cleaned = sequence.upper().replace("U", "T")
    amino_acids: List[str] = []
    for start in range(0, len(cleaned), 3):
        codon = cleaned[start : start + 3]
        amino_acids.append(CODON_TABLE.get(codon, "X"))
    protein = "".join(amino_acids)
    if protein.endswith("*"):
        protein = protein[:-1]
    return protein


def make_snapshot_prefix(sequence_path: Path, month_label: str) -> str:
    name = sequence_path.name
    for suffix in (".fasta.gz", ".fa.gz", ".fna.gz", ".fasta", ".fa", ".fna", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    name = re.sub(r"^[^_]+_\d{4}-\d{2}-\d{2}_", "", name)
    if not name:
        name = "spike"

    return f"{name}_{month_label}"


def create_snapshot_outputs(
    output_dir: Path,
    prefix: str,
    metadata_fieldnames: Sequence[str],
    sequence_mode: str,
) -> SnapshotOutputs:
    ensure_parent_directory(output_dir / "placeholder")
    suffix = "full_nt" if sequence_mode == "nucleotide_full" else "nt"
    nucleotide_path = output_dir / f"{prefix}_{suffix}.fa"
    protein_path = output_dir / f"{prefix}_aa.fa" if sequence_mode == "spike" else None
    metadata_path = output_dir / f"{prefix}_metadata.csv"

    nucleotide_handle = open_text(nucleotide_path, "wt")
    protein_handle = open_text(protein_path, "wt") if protein_path is not None else None
    metadata_handle = open_text(metadata_path, "wt")
    metadata_writer = csv.DictWriter(metadata_handle, fieldnames=list(metadata_fieldnames))
    metadata_writer.writeheader()

    return SnapshotOutputs(
        nucleotide_path=nucleotide_path,
        protein_path=protein_path,
        metadata_path=metadata_path,
        nucleotide_handle=nucleotide_handle,
        protein_handle=protein_handle,
        metadata_handle=metadata_handle,
        metadata_writer=metadata_writer,
    )


def close_snapshot_outputs(outputs_by_month: Dict[str, SnapshotOutputs]) -> None:
    for outputs in outputs_by_month.values():
        outputs.nucleotide_handle.close()
        if outputs.protein_handle is not None:
            outputs.protein_handle.close()
        outputs.metadata_handle.close()


def write_summary(
    output_dir: Path,
    months: Sequence[MonthWindow],
    stats_by_month: Dict[str, SnapshotStats],
) -> Path:
    summary_path = output_dir / "snapshot_summary.csv"
    with summary_path.open("wt", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "month",
                "metadata_rows_selected",
                "duplicate_metadata_ids",
                "fasta_records_matched",
                "sequences_written",
                "skipped_missing_fasta",
                "skipped_short_protein",
                "skipped_bad_coding_length",
            ],
        )
        writer.writeheader()
        for month in months:
            stats = stats_by_month[month.label]
            writer.writerow(
                {
                    "month": month.label,
                    "metadata_rows_selected": stats.metadata_rows_selected,
                    "duplicate_metadata_ids": stats.duplicate_metadata_ids,
                    "fasta_records_matched": stats.fasta_records_matched,
                    "sequences_written": stats.sequences_written,
                    "skipped_missing_fasta": stats.skipped_missing_fasta,
                    "skipped_short_protein": stats.skipped_short_protein,
                    "skipped_bad_coding_length": stats.skipped_bad_coding_length,
                }
            )
    return summary_path


def run_snapshotting(args: argparse.Namespace) -> int:
    months = parse_months(args.months)
    if args.min_protein_length < 0:
        raise ValueError("--min-protein-length must be >= 0.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be a positive integer.")
    if not args.sequence_path.exists():
        raise FileNotFoundError(f"Sequence FASTA not found: {args.sequence_path}")
    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_path}")

    selection = load_metadata_selection(
        metadata_path=args.metadata_path,
        months=months,
        id_column_name=args.metadata_id_column,
        date_column_name=args.metadata_date_column,
    )

    id_to_month: Dict[str, str] = {}
    for month_label, ids in selection.ids_by_month.items():
        for record_id in ids:
            id_to_month[record_id] = month_label

    outputs_by_month = {
        month.label: create_snapshot_outputs(
            output_dir=args.output_dir,
            prefix=make_snapshot_prefix(args.sequence_path, month.label),
            metadata_fieldnames=selection.fieldnames,
            sequence_mode=args.sequence_mode,
        )
        for month in months
    }

    seen_fasta_ids: Set[str] = set()
    try:
        for index, (header, nucleotide_sequence) in enumerate(iter_fasta_records(args.sequence_path), start=1):
            if args.max_records is not None and index > args.max_records:
                break

            record_id = fasta_identifier(header)
            month_label = id_to_month.get(record_id)
            if month_label is None:
                continue

            seen_fasta_ids.add(record_id)
            stats = selection.stats_by_month[month_label]
            stats.fasta_records_matched += 1

            outputs = outputs_by_month[month_label]
            if args.sequence_mode == "nucleotide_full":
                write_fasta_record(outputs.nucleotide_handle, header, nucleotide_sequence)
            else:
                cleaned_nt = nucleotide_sequence.strip().upper().replace("U", "T")
                if len(cleaned_nt) % 3 != 0:
                    stats.skipped_bad_coding_length += 1
                    continue

                protein_sequence = translate_spike(cleaned_nt)
                if len(protein_sequence) < args.min_protein_length:
                    stats.skipped_short_protein += 1
                    continue

                write_fasta_record(outputs.nucleotide_handle, header, cleaned_nt)
                if outputs.protein_handle is not None:
                    write_fasta_record(outputs.protein_handle, header, protein_sequence)

            if record_id not in outputs.written_ids:
                outputs.metadata_writer.writerow(selection.rows_by_id[record_id])
                outputs.written_ids.add(record_id)
            stats.sequences_written += 1
    finally:
        close_snapshot_outputs(outputs_by_month)

    for month in months:
        missing_ids = selection.ids_by_month[month.label] - seen_fasta_ids
        selection.stats_by_month[month.label].skipped_missing_fasta = len(missing_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = write_summary(args.output_dir, months, selection.stats_by_month)

    print(f"Metadata rows scanned: {selection.total_rows}")
    if selection.missing_id_rows:
        print(f"Metadata rows skipped with missing IDs: {selection.missing_id_rows}")
    if selection.missing_date_rows:
        print(f"Metadata rows skipped with missing dates: {selection.missing_date_rows}")
    if selection.invalid_date_rows:
        print(f"Metadata rows skipped with invalid dates: {selection.invalid_date_rows}")

    for month in months:
        stats = selection.stats_by_month[month.label]
        outputs = outputs_by_month[month.label]
        print(
            f"{month.label}: selected={stats.metadata_rows_selected}, "
            f"matched_fasta={stats.fasta_records_matched}, written={stats.sequences_written}, "
            f"short_protein={stats.skipped_short_protein}, bad_coding_length={stats.skipped_bad_coding_length}"
        )
        print(f"  nucleotide: {outputs.nucleotide_path}")
        if outputs.protein_path is not None:
            print(f"  protein: {outputs.protein_path}")
        print(f"  metadata: {outputs.metadata_path}")

    print(f"Summary: {summary_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_snapshotting(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
