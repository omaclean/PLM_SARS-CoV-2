#!/usr/bin/env python3
"""Extract Wuhan-Hu-1 annotated gene sequences from a whole-genome alignment.

The default configuration is set up for the COG-UK SARS-CoV-2 alignment in this
repository and extracts the spike CDS (`S`) using Wuhan-Hu-1 annotation
coordinates from MN908947.3.

The input alignment is expected to already be in Wuhan-Hu-1 genomic coordinates
(e.g. the provided COG alignment has aligned length 29903), so the requested
feature can be sliced directly from each aligned genome.

Examples
--------
Extract all spike nucleotide sequences (default behaviour):
    python scripts/extract_spike_from_alignment.py

Keep alignment gaps in the output spike region:
    python scripts/extract_spike_from_alignment.py --keep-gaps

Filter to one or more metadata values:
    python scripts/extract_spike_from_alignment.py \
        --where country=UK \
        --in lineage=NB.1.8.1,XFG.3

Filter with substring matching and write matched metadata rows:
    python scripts/extract_spike_from_alignment.py \
        --contains scorpio_call=Omicron \
        --output-metadata Sequences/cog_2025-07-17_spike_metadata.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple

DEFAULT_ALIGNMENT_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_alignment.fa.gz")
DEFAULT_METADATA_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_metadata.csv.gz")
DEFAULT_GFF3_PATH = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/wuhan-hu-1-sequence.gff3")
OUTPUT_FASTA = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_spike.fa")
OUTPUT_METADATA = Path("/home3/oml4h/PLM_SARS-CoV-2/Sequences/cog_2025-07-17_spike_metadata.csv")

DEFAULT_GENE = "S"
FASTA_WRAP = 80


@dataclass(frozen=True)
class FeatureCoordinates:
    gene: str
    seqid: str
    start: int
    end: int
    strand: str
    feature_type: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class FilterSpec:
    column: str
    value: str


@dataclass(frozen=True)
class InFilterSpec:
    column: str
    values: Tuple[str, ...]


@dataclass
class MetadataConfig:
    id_column: str
    exact_filters: List[FilterSpec]
    contains_filters: List[FilterSpec]
    in_filters: List[InFilterSpec]
    case_insensitive: bool

    @property
    def has_filters(self) -> bool:
        return bool(self.exact_filters or self.contains_filters or self.in_filters)


@dataclass
class MetadataIndex:
    rows_by_id: Dict[str, Dict[str, str]]
    fieldnames: List[str]
    total_rows: int
    matched_rows: int
    duplicate_ids: int
    missing_ids: int


@dataclass
class ExtractionStats:
    examined: int = 0
    written: int = 0
    skipped_by_metadata: int = 0
    skipped_missing_metadata: int = 0
    skipped_short_sequences: int = 0
    metadata_rows_written: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract spike (or another Wuhan-Hu-1 annotated gene) from a SARS-CoV-2 "
            "whole-genome alignment and optionally filter records using metadata."
        )
    )
    parser.add_argument(
        "--alignment-path",
        type=Path,
        default=DEFAULT_ALIGNMENT_PATH,
        help="Input FASTA alignment (.fa, .fasta, optionally .gz).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Optional metadata CSV (.csv, optionally .gz).",
    )
    parser.add_argument(
        "--gff3-path",
        type=Path,
        default=DEFAULT_GFF3_PATH,
        help="Wuhan-Hu-1 GFF3 annotation file (MN908947.3).",
    )
    parser.add_argument(
        "--gene",
        default=DEFAULT_GENE,
        help="Gene symbol to extract from the GFF3 annotation. Default: S.",
    )
    parser.add_argument(
        "--metadata-id-column",
        default="sequence_name",
        help="Metadata column used to match FASTA record IDs. Default: sequence_name.",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        default=OUTPUT_FASTA,
        help="Output FASTA path. Defaults to a gzip-compressed file next to the alignment.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=OUTPUT_METADATA,
        help="Optional CSV/CSV.GZ file containing metadata rows for written sequences.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Exact-match metadata filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        metavar="COLUMN=TEXT",
        help="Substring metadata filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--in",
        dest="in_filters",
        action="append",
        default=[],
        metavar="COLUMN=VALUE1,VALUE2",
        help="Set-membership metadata filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Apply metadata filter comparisons case-insensitively.",
    )
    parser.add_argument(
        "--require-metadata-match",
        action="store_true",
        help="Only write sequences that have a matching metadata row, even if no filters are set.",
    )
    parser.add_argument(
        "--keep-gaps",
        action="store_true",
        help="Keep alignment gaps in the extracted feature. Default: remove '-' and '.'.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap on the number of FASTA records to process. Useful for quick tests.",
    )
    return parser


def open_text(path: Path, mode: str) -> TextIO:
    if "b" in mode:
        raise ValueError("Binary mode is not supported.")
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def strip_known_suffixes(path: Path) -> str:
    name = path.name
    for suffix in (".fasta.gz", ".fa.gz", ".fna.gz", ".fas.gz", ".fasta", ".fa", ".fna", ".fas", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def default_output_path(alignment_path: Path, gene: str, keep_gaps: bool) -> Path:
    base = strip_known_suffixes(alignment_path)
    gap_label = "aligned" if keep_gaps else "ungapped"
    return alignment_path.with_name(f"{base}_{gene.lower()}_{gap_label}.fa.gz")


def parse_attribute_field(attribute_field: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for item in attribute_field.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            attributes[key] = value
    return attributes


def find_feature_coordinates(gff3_path: Path, gene: str) -> FeatureCoordinates:
    gene_normalised = gene.strip().lower()
    candidates: Dict[str, List[Tuple[str, int, int, str]]] = {"CDS": [], "gene": []}

    with gff3_path.open("rt", newline="") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            seqid, _source, feature_type, start, end, _score, strand, _phase, attributes_text = fields
            if feature_type not in candidates:
                continue
            attributes = parse_attribute_field(attributes_text)
            possible_names = {
                attributes.get("gene", ""),
                attributes.get("Name", ""),
                attributes.get("gene_name", ""),
            }
            possible_names = {name.strip().lower() for name in possible_names if name.strip()}
            if gene_normalised not in possible_names:
                continue
            candidates[feature_type].append((seqid, int(start), int(end), strand))

    for preferred_type in ("CDS", "gene"):
        if not candidates[preferred_type]:
            continue
        seqids = {item[0] for item in candidates[preferred_type]}
        strands = {item[3] for item in candidates[preferred_type]}
        if len(seqids) != 1 or len(strands) != 1:
            raise ValueError(
                f"Found inconsistent {preferred_type} annotations for gene {gene!r} in {gff3_path}."
            )
        return FeatureCoordinates(
            gene=gene,
            seqid=next(iter(seqids)),
            start=min(item[1] for item in candidates[preferred_type]),
            end=max(item[2] for item in candidates[preferred_type]),
            strand=next(iter(strands)),
            feature_type=preferred_type,
        )

    raise ValueError(f"Could not find gene {gene!r} in {gff3_path}.")


def normalise_column_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def parse_key_value_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid filter {spec!r}; expected COLUMN=VALUE.")
    column, value = spec.split("=", 1)
    column = column.strip()
    if not column:
        raise ValueError(f"Invalid filter {spec!r}; column name is empty.")
    return column, value.strip()


def parse_metadata_config(args: argparse.Namespace) -> MetadataConfig:
    exact_filters = [FilterSpec(*parse_key_value_spec(spec)) for spec in args.where]
    contains_filters = [FilterSpec(*parse_key_value_spec(spec)) for spec in args.contains]

    in_filters: List[InFilterSpec] = []
    for spec in args.in_filters:
        column, values_text = parse_key_value_spec(spec)
        values = tuple(value.strip() for value in values_text.split(",") if value.strip())
        if not values:
            raise ValueError(f"Invalid --in filter {spec!r}; provide at least one value.")
        in_filters.append(InFilterSpec(column=column, values=values))

    return MetadataConfig(
        id_column=args.metadata_id_column,
        exact_filters=exact_filters,
        contains_filters=contains_filters,
        in_filters=in_filters,
        case_insensitive=args.case_insensitive,
    )


def resolve_column_name(requested: str, fieldnames: Sequence[str]) -> str:
    by_normalised = {normalise_column_name(name): name for name in fieldnames}
    key = normalise_column_name(requested)
    if key not in by_normalised:
        available = ", ".join(fieldnames)
        raise KeyError(f"Column {requested!r} not found in metadata. Available columns: {available}")
    return by_normalised[key]


def normalise_value(value: str, case_insensitive: bool) -> str:
    return value.casefold() if case_insensitive else value


def row_matches_filters(
    row: Dict[str, str],
    exact_filters: Sequence[FilterSpec],
    contains_filters: Sequence[FilterSpec],
    in_filters: Sequence[InFilterSpec],
    case_insensitive: bool,
) -> bool:
    for spec in exact_filters:
        cell_value = row.get(spec.column, "")
        if normalise_value(cell_value, case_insensitive) != normalise_value(spec.value, case_insensitive):
            return False

    for spec in contains_filters:
        cell_value = row.get(spec.column, "")
        if normalise_value(spec.value, case_insensitive) not in normalise_value(cell_value, case_insensitive):
            return False

    for spec in in_filters:
        cell_value = normalise_value(row.get(spec.column, ""), case_insensitive)
        allowed = {normalise_value(value, case_insensitive) for value in spec.values}
        if cell_value not in allowed:
            return False

    return True


def load_metadata_index(metadata_path: Path, config: MetadataConfig) -> MetadataIndex:
    with open_text(metadata_path, "rt") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file {metadata_path} does not contain a header row.")

        fieldnames = list(reader.fieldnames)
        id_column = resolve_column_name(config.id_column, fieldnames)
        exact_filters = [
            FilterSpec(resolve_column_name(spec.column, fieldnames), spec.value)
            for spec in config.exact_filters
        ]
        contains_filters = [
            FilterSpec(resolve_column_name(spec.column, fieldnames), spec.value)
            for spec in config.contains_filters
        ]
        in_filters = [
            InFilterSpec(resolve_column_name(spec.column, fieldnames), spec.values)
            for spec in config.in_filters
        ]

        rows_by_id: Dict[str, Dict[str, str]] = {}
        total_rows = 0
        matched_rows = 0
        duplicate_ids = 0
        missing_ids = 0

        for row in reader:
            total_rows += 1
            row_id = (row.get(id_column) or "").strip()
            if not row_id:
                missing_ids += 1
                continue
            if not row_matches_filters(
                row=row,
                exact_filters=exact_filters,
                contains_filters=contains_filters,
                in_filters=in_filters,
                case_insensitive=config.case_insensitive,
            ):
                continue
            matched_rows += 1
            if row_id in rows_by_id:
                duplicate_ids += 1
                continue
            rows_by_id[row_id] = row

    return MetadataIndex(
        rows_by_id=rows_by_id,
        fieldnames=fieldnames,
        total_rows=total_rows,
        matched_rows=matched_rows,
        duplicate_ids=duplicate_ids,
        missing_ids=missing_ids,
    )


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    with open_text(path, "rt") as handle:
        header: Optional[str] = None
        sequence_chunks: List[str] = []

        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_chunks)
                header = line[1:].strip()
                sequence_chunks = []
                continue
            sequence_chunks.append(line)

        if header is not None:
            yield header, "".join(sequence_chunks)


def reverse_complement(sequence: str) -> str:
    translation = str.maketrans(
        "ACGTRYMKBDHVNacgtrymkbdhvn.-",
        "TGCAYRKMVHDBNtgcayrkmvhd bn.-".replace(" ", ""),
    )
    return sequence.translate(translation)[::-1]


def extract_feature_sequence(sequence: str, feature: FeatureCoordinates, keep_gaps: bool) -> str:
    if len(sequence) < feature.end:
        raise ValueError(
            f"Sequence length {len(sequence)} is shorter than required end coordinate {feature.end}."
        )

    extracted = sequence[feature.start - 1 : feature.end]
    if feature.strand == "-":
        extracted = reverse_complement(extracted)
    if not keep_gaps:
        extracted = extracted.replace("-", "").replace(".", "")
    return extracted.upper()


def fasta_identifier(header: str) -> str:
    return header.split()[0]


def write_fasta_record(handle: TextIO, header: str, sequence: str, line_width: int = FASTA_WRAP) -> None:
    handle.write(f">{header}\n")
    for index in range(0, len(sequence), line_width):
        handle.write(sequence[index : index + line_width] + "\n")


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_sequences(
    alignment_path: Path,
    output_fasta: Path,
    feature: FeatureCoordinates,
    metadata_index: Optional[MetadataIndex],
    output_metadata: Optional[Path],
    require_metadata_match: bool,
    keep_gaps: bool,
    max_records: Optional[int],
    filters_active: bool,
) -> ExtractionStats:
    stats = ExtractionStats()
    written_metadata_ids: set[str] = set()

    metadata_writer = None
    metadata_handle: Optional[TextIO] = None
    try:
        if output_metadata is not None:
            if metadata_index is None:
                raise ValueError("--output-metadata requires a metadata file.")
            ensure_parent_directory(output_metadata)
            metadata_handle = open_text(output_metadata, "wt")
            metadata_writer = csv.DictWriter(metadata_handle, fieldnames=metadata_index.fieldnames)
            metadata_writer.writeheader()

        ensure_parent_directory(output_fasta)
        with open_text(output_fasta, "wt") as fasta_handle:
            for header, aligned_sequence in iter_fasta_records(alignment_path):
                if max_records is not None and stats.examined >= max_records:
                    break
                stats.examined += 1
                record_id = fasta_identifier(header)
                metadata_row = metadata_index.rows_by_id.get(record_id) if metadata_index else None

                if filters_active:
                    if metadata_row is None:
                        stats.skipped_by_metadata += 1
                        continue
                elif require_metadata_match and metadata_index is not None and metadata_row is None:
                    stats.skipped_missing_metadata += 1
                    continue

                try:
                    extracted_sequence = extract_feature_sequence(
                        sequence=aligned_sequence,
                        feature=feature,
                        keep_gaps=keep_gaps,
                    )
                except ValueError:
                    stats.skipped_short_sequences += 1
                    continue

                write_fasta_record(fasta_handle, header, extracted_sequence)
                stats.written += 1

                if metadata_writer is not None and metadata_row is not None and record_id not in written_metadata_ids:
                    metadata_writer.writerow(metadata_row)
                    written_metadata_ids.add(record_id)
                    stats.metadata_rows_written += 1
    finally:
        if metadata_handle is not None:
            metadata_handle.close()

    return stats


def validate_inputs(args: argparse.Namespace, metadata_config: MetadataConfig) -> None:
    if not args.alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {args.alignment_path}")
    if not args.gff3_path.exists():
        raise FileNotFoundError(f"GFF3 file not found: {args.gff3_path}")
    needs_metadata = bool(
        args.metadata_path
        and (metadata_config.has_filters or args.output_metadata is not None or args.require_metadata_match)
    )
    if needs_metadata and not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    if args.output_metadata is not None and not args.metadata_path:
        raise ValueError("--output-metadata requires --metadata-path.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be a positive integer.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        metadata_config = parse_metadata_config(args)
        validate_inputs(args, metadata_config)
        feature = find_feature_coordinates(args.gff3_path, args.gene)
        output_fasta = args.output_fasta or default_output_path(
            alignment_path=args.alignment_path,
            gene=args.gene,
            keep_gaps=args.keep_gaps,
        )

        metadata_index: Optional[MetadataIndex] = None
        if args.metadata_path and (
            metadata_config.has_filters or args.output_metadata is not None or args.require_metadata_match
        ):
            metadata_index = load_metadata_index(args.metadata_path, metadata_config)

        stats = extract_sequences(
            alignment_path=args.alignment_path,
            output_fasta=output_fasta,
            feature=feature,
            metadata_index=metadata_index,
            output_metadata=args.output_metadata,
            require_metadata_match=args.require_metadata_match,
            keep_gaps=args.keep_gaps,
            max_records=args.max_records,
            filters_active=metadata_config.has_filters,
        )
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        "Loaded feature "
        f"{feature.gene} ({feature.feature_type}) from {feature.seqid}:{feature.start}-{feature.end} "
        f"[{feature.strand}]"
    )
    if metadata_index is not None:
        print(
            "Metadata rows: "
            f"{metadata_index.total_rows} total, "
            f"{metadata_index.matched_rows} matched filters, "
            f"{len(metadata_index.rows_by_id)} unique IDs retained"
        )
        if metadata_index.duplicate_ids:
            print(f"Duplicate metadata IDs ignored after first occurrence: {metadata_index.duplicate_ids}")
        if metadata_index.missing_ids:
            print(f"Metadata rows missing the ID column were skipped: {metadata_index.missing_ids}")

    print(f"Processed FASTA records: {stats.examined}")
    print(f"Sequences written: {stats.written}")
    if stats.skipped_by_metadata:
        print(f"Skipped by metadata filters: {stats.skipped_by_metadata}")
    if stats.skipped_missing_metadata:
        print(f"Skipped due to missing metadata matches: {stats.skipped_missing_metadata}")
    if stats.skipped_short_sequences:
        print(f"Skipped because the alignment was shorter than the feature coordinates: {stats.skipped_short_sequences}")
    print(f"Output FASTA: {output_fasta}")
    if args.output_metadata is not None:
        print(f"Output metadata rows written: {stats.metadata_rows_written}")
        print(f"Output metadata: {args.output_metadata}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
