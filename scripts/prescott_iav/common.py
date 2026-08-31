#!/usr/bin/env python3
"""Shared, dependency-light helpers for the PRESCOTT/ESCOTT influenza HA stage-1 code.

Everything here must import cleanly under ``/home3/oml4h/miniconda3/envs/PRESCOTT``,
which has no torch and no esm.  That constraint is the whole reason this module
exists: ``Functions_HuggingFace`` imports ``torch`` and ``esm`` at module scope
(lines 7-8), so stage-1 cannot simply reuse its algorithms even though it needs a
handful of them.

Where a function below mirrors one in ``Functions_HuggingFace`` the docstring
names the original.  Those mirrors are deliberately byte-faithful, not merely
"equivalent": ``run_prescott_diversity.py`` asserts that the reference protein it
gets back from ``build_lineage_cache`` is identical to the one baked into the
ESCOTT score matrix, and the frequency files have to be built on exactly the same
reference->alignment coordinate map that the evaluation half uses, or the
population-frequency prior would be silently shifted against the scores.

Kept to pathlib + Bio + numpy + pandas on purpose (see the task brief); no prody,
no biotite, no scipy, so that ``run_escott.py`` can import it on a node without a
structural stack.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq

# ``constants.py`` is THE source of truth for anything the stage-2 driver also has
# an opinion about (the parent map, the input-only set, the escott name tokens).
# It is dependency-free on purpose so it imports under both interpreters; we
# re-export from it rather than redefining, so the two halves of the pipeline
# cannot drift.  The import is tried three ways because this package is used as a
# package (``from prescott_iav import common``), as a bare module on sys.path
# (``import common``) and as a script directory.
try:  # pragma: no cover - exercised implicitly by every import path
    from . import constants as _constants  # type: ignore
except ImportError:  # not imported as a package member
    try:
        import constants as _constants  # type: ignore
    except ImportError:  # last resort: load it straight off disk
        import importlib.util as _importlib_util

        _spec = _importlib_util.spec_from_file_location(
            "prescott_iav_constants", Path(__file__).resolve().parent / "constants.py"
        )
        if _spec is None or _spec.loader is None:  # pragma: no cover - unreachable in-tree
            raise
        _constants = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_constants)


# --------------------------------------------------------------------------- #
# Constants shared across stage-1 modules
# --------------------------------------------------------------------------- #

# Mirrors run_mutational_accessibility.py:77-78 so consensus/diversity handling
# is identical on both sides of the pipeline.
IGNORE_ALIGNMENT_CHARS = frozenset({"-", "*", "."})
STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

# ESCOTT/PRESCOTT emit and consume mutants in this order (prescott.py:916,
# ``aaOrderList = list('ACDEFGHIKLMNPQRSTVWY')``).
GEMME_AA_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")

# The order the SC2 script writes its PLM probability profiles in; the score
# matrices we hand to stage 2 must use it (run_mutational_accessibility.py:1033).
PLM_AA_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")

# The clade topology, the input-only set and the escott tags all live in
# constants.py now -- see its module docstring for why.  These names are kept as
# aliases because every stage-1 module already imports them from here, but they
# are *bindings*, not copies: there is exactly one dict, and it is the one the
# stage-2 driver imports too.
PARENT_MAP_PRESETS: Dict[str, Dict[str, str]] = _constants.DEFAULT_PARENT_MAPS
DEFAULT_PARENT_MAP_PRESET = _constants.DEFAULT_PARENT_MAP_PRESET

# G.1 has no basal panel underneath it, so it can only ever be an input.
INPUT_ONLY_LINEAGES = _constants.INPUT_ONLY_LINEAGES

# escott derives its `prot` token by splitting the FASTA header on the first
# non-alphanumeric character (escott.py:76), then names *every* output after it.
LINEAGE_TAGS: Dict[str, str] = _constants.LINEAGE_TAGS

# Naming helpers stage 1 and stage 2 must agree on byte-for-byte (the driver reads
# the `_parent<TOK>` suffix back off a variant name to tell a primary model from a
# parent-sensitivity row).
variant_parent_token = _constants.variant_parent_token
alternate_frequency_basename = _constants.alternate_frequency_basename
sensitivity_edges_between_presets = _constants.sensitivity_edges_between_presets
parse_edge_spec = _constants.parse_edge_spec

# jet_surrogate's tuned defaults live in constants.py too, so that the driver, the
# slurm script and any hand-run of jet_surrogate.py all quote the same number.
DEFAULT_TRACE_TOP_FRACTION = _constants.DEFAULT_TRACE_TOP_FRACTION
MAX_ZERO_TRACE_FRACTION = _constants.MAX_ZERO_TRACE_FRACTION
WARN_ZERO_TRACE_FRACTION = _constants.WARN_ZERO_TRACE_FRACTION

# escott.py:76 verbatim.
_ESCOTT_NAME_SPLIT_RE = re.compile("[^A-Z0-9a-z]")


# --------------------------------------------------------------------------- #
# Labels, keys and hashing
# --------------------------------------------------------------------------- #

def safe_label(label: str) -> str:
    """Mirror of ``Functions_HuggingFace._safe_label`` (line 3505).

    Note that dots survive, so ``J.2_int`` stays ``J.2_int``.  That matters:
    ``prescott.py:901`` runs ``os.path.splitext`` on its ``-o`` value and would
    truncate at the last dot, so anything handed to prescott/escott must go
    through :func:`dot_free_key` first.
    """
    return label.strip().replace(" ", "_").replace("/", "-")


def dot_free_key(lineage_key: str) -> str:
    """Filesystem key with dots removed, for escott/prescott output basenames."""
    return lineage_key.replace(".", "_")


def lineage_tag(label: str) -> str:
    """Short alphanumeric escott token for a lineage label (``J.2.4`` -> ``J24``)."""
    if label in LINEAGE_TAGS:
        return LINEAGE_TAGS[label]
    tag = re.sub(r"[^A-Za-z0-9]", "", label)
    if not tag:
        raise ValueError(f"Cannot derive an escott tag from lineage label {label!r}")
    return tag


def escott_prot_token(header: str) -> str:
    """Reproduce escott's own ``prot`` derivation (escott.py:76).

    ``header`` is the FASTA description with the leading '>' already stripped.
    We call this to *assert* that our chosen header yields the token we expect,
    rather than guessing what escott will name its outputs.
    """
    return _ESCOTT_NAME_SPLIT_RE.split(header)[0]


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def read_json(path: Path) -> Optional[Dict[str, object]]:
    """Return the parsed JSON, or None if the file is missing/corrupt.

    Cache manifests are advisory; a truncated one from a killed job must degrade
    to "recompute", never to a crash.
    """
    if not path.exists():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# FASTA I/O
#
# Deliberately hand-rolled rather than Bio.SeqIO: the deep set is 6433 records
# and the J.2_int panel is 27452 x 594, and we only ever want (header, str) pairs.
# Constructing SeqRecord objects for those costs several seconds and ~10x the RAM.
# --------------------------------------------------------------------------- #

def read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield ``(header_without_gt, sequence)`` pairs, sequence uppercased."""
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks).upper()


def read_fasta_sequences(path: Path) -> List[str]:
    """Sequences only, for panel work where headers are never used."""
    return [seq for _header, seq in read_fasta(path)]


def write_fasta(path: Path, records: Iterable[Tuple[str, str]], line_width: int = 0) -> None:
    """Write records as FASTA.

    ``line_width=0`` writes each sequence on a single line, which is what mafft,
    escott and prescott all cope with best and what makes the query files
    trivially diffable.
    """
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            if line_width and line_width > 0:
                for start in range(0, len(seq), line_width):
                    handle.write(seq[start:start + line_width] + "\n")
            else:
                handle.write(seq + "\n")


def count_fasta_records(path: Path) -> int:
    with open(path) as handle:
        return sum(1 for line in handle if line.startswith(">"))


# --------------------------------------------------------------------------- #
# Reference CDS handling
# --------------------------------------------------------------------------- #

def translate_reference_cds(raw_seq: str) -> str:
    """Byte-faithful mirror of ``Functions_HuggingFace._translate_nt_to_protein`` (2468).

    In particular it strips *all* '*' characters rather than truncating at the
    first stop.  For our five HA references that is a distinction without a
    difference (each has exactly one, terminal, stop), but the driver asserts
    ``score_matrix_source_sequence == lineage_data['full_ref_protein']`` and that
    assertion only holds if we translate identically.
    """
    cleaned = raw_seq.replace("-", "").replace(".", "").upper().replace("U", "T")
    trimmed = cleaned[: (len(cleaned) // 3) * 3]
    if not trimmed:
        return ""
    return str(Seq(trimmed).translate(to_stop=False)).replace("*", "")


def load_reference_cds(path: Path, label: str = "") -> Dict[str, str]:
    """Load a single-record nucleotide CDS FASTA and translate it.

    Mirrors ``Functions_HuggingFace._load_single_focal_reference`` (3466) minus the
    nucleotide sniffing, which is replaced by an explicit alphabet check so the
    failure message names the offending file.
    """
    records = list(read_fasta(path))
    if not records:
        raise ValueError(f"No records found in reference FASTA: {path}")
    if len(records) > 1:
        print(f"@> Warning: {path} contains {len(records)} records; using the first only.")

    header, raw_seq = records[0]
    nucleotide = raw_seq.replace("-", "").replace(".", "").upper().replace("U", "T")
    if not nucleotide or set(nucleotide) - set("ACGTNRYKMSWBDHV"):
        raise ValueError(
            f"Reference {path} does not look like a nucleotide CDS "
            f"(unexpected characters: {sorted(set(nucleotide) - set('ACGTNRYKMSWBDHV'))})"
        )
    protein = translate_reference_cds(raw_seq)
    return {
        "header": header,
        "lineage": label,
        "nucleotide": nucleotide,
        "protein": protein,
        "path": str(path),
    }


# --------------------------------------------------------------------------- #
# Guide file
# --------------------------------------------------------------------------- #

def read_guide_rows(guide_path: Path) -> List[Dict[str, str]]:
    """Mirror of ``Functions_HuggingFace.load_analysis_targets`` MONTHLY_GUIDE branch (186-208).

    Returns dicts with keys ``label``, ``diversity_path``, ``reference_path``.
    """
    if not guide_path.exists():
        raise FileNotFoundError(f"Guide file not found: {guide_path}")

    rows: List[Dict[str, str]] = []
    with open(guide_path, "r", newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("month") or row.get("label") or "").strip()
            fasta_path = (row.get("fasta") or row.get("path") or "").strip()
            reference = (row.get("reference") or "").strip()
            if not label or not fasta_path:
                continue
            rows.append(
                {
                    "label": label,
                    "diversity_path": fasta_path,
                    "reference_path": reference,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Parent map resolution
# --------------------------------------------------------------------------- #

def parse_parent_map_override(spec: Optional[str]) -> Dict[str, str]:
    """Parse ``"child=parent,child=parent"`` into a dict."""
    if not spec:
        return {}
    override: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"--parent-map entry {chunk!r} is not of the form child=parent")
        child, parent = chunk.split("=", 1)
        override[child.strip()] = parent.strip()
    return override


def resolve_parent_map(
    preset: str,
    override: Optional[str],
    known_labels: Sequence[str],
) -> Dict[str, str]:
    """Resolve the preset + per-edge overrides, then validate against the guide.

    Validation is worth doing eagerly because a bad edge does not fail loudly
    later -- it just silently produces a frequency prior computed from the wrong
    panel, which is exactly the sort of error that survives to a figure.
    """
    if preset not in PARENT_MAP_PRESETS:
        raise ValueError(f"Unknown parent-map preset {preset!r}; expected one of {sorted(PARENT_MAP_PRESETS)}")

    resolved = dict(PARENT_MAP_PRESETS[preset])
    resolved.update(parse_parent_map_override(override))

    known = set(known_labels)
    for child, parent in resolved.items():
        if child not in known:
            raise ValueError(f"Parent map names child {child!r}, which has no guide row")
        if parent not in known:
            raise ValueError(f"Parent map names parent {parent!r} (of {child!r}), which has no guide row")
        if child == parent:
            raise ValueError(f"Parent map makes {child!r} its own parent")

    # Cycle check: walk each child up to the root, bounded by the map size.
    for child in resolved:
        seen = {child}
        node = child
        while node in resolved:
            node = resolved[node]
            if node in seen:
                raise ValueError(f"Parent map contains a cycle through {node!r}")
            seen.add(node)

    return resolved


# --------------------------------------------------------------------------- #
# Alignment / consensus mirrors
# --------------------------------------------------------------------------- #

def aligned_byte_matrix(sequences: Sequence[str]) -> Tuple[np.ndarray, int]:
    """Pack aligned sequences into an (n_seq, aln_len) uint8 ASCII matrix.

    ``Functions_HuggingFace`` builds ``np.array([list(seq) ...])``, i.e. a '<U1'
    array costing 4 bytes per character.  For the 27452 x 594 J.2_int panel that
    is ~65 MB and several seconds; the uint8 packing is ~16 MB and ~0.1 s and is
    exactly equivalent for our purposes (all characters are ASCII, and ``np.unique``
    on uint8 sorts by code point, which for uppercase letters is alphabetical --
    so the ``argmax``-on-ties consensus tie-break is preserved).
    """
    if not sequences:
        return np.zeros((0, 0), dtype=np.uint8), 0
    aln_len = max(len(seq) for seq in sequences)
    padded = "".join(seq.ljust(aln_len, "-") for seq in sequences)
    matrix = np.frombuffer(padded.encode("ascii", errors="replace"), dtype=np.uint8)
    return matrix.reshape(len(sequences), aln_len), aln_len


def build_consensus_and_column_map(
    sequences: Sequence[str],
    valid_residues: Sequence[str] = STANDARD_AMINO_ACIDS,
    ignore_chars: Iterable[str] = IGNORE_ALIGNMENT_CHARS,
) -> Tuple[str, List[int], int]:
    """Mirror of ``Functions_HuggingFace._build_lineage_consensus_and_column_map`` (2571).

    Returns ``(consensus_seq, consensus_to_alignment_col, aln_len)`` where the
    column list is 1-based.  Columns with no valid residue at all are dropped from
    the consensus, exactly as in the original.
    """
    matrix, aln_len = aligned_byte_matrix(sequences)
    if aln_len == 0:
        return "", [], 0

    allowed = sorted(set(valid_residues) - set(ignore_chars))
    allowed_codes = np.array([ord(aa) for aa in allowed], dtype=np.uint8)

    # counts[i, j] = number of rows with residue `allowed[i]` in column j.
    counts = np.zeros((len(allowed), aln_len), dtype=np.int64)
    for idx, code in enumerate(allowed_codes):
        counts[idx] = (matrix == code).sum(axis=0)

    totals = counts.sum(axis=0)
    consensus_chars: List[str] = []
    consensus_to_alignment_col: List[int] = []
    # np.argmax breaks ties toward the lowest index; `allowed` is sorted, so this
    # reproduces np.unique(...)+argmax in the original (alphabetically first wins).
    winners = counts.argmax(axis=0)
    for col_idx in range(aln_len):
        if totals[col_idx] == 0:
            continue
        consensus_chars.append(allowed[int(winners[col_idx])])
        consensus_to_alignment_col.append(col_idx + 1)

    return "".join(consensus_chars), consensus_to_alignment_col, aln_len


def map_reference_to_alignment_columns(
    reference_protein: str,
    sequences: Sequence[str],
    valid_residues: Sequence[str] = STANDARD_AMINO_ACIDS,
    ignore_chars: Iterable[str] = IGNORE_ALIGNMENT_CHARS,
) -> Tuple[Dict[int, int], int, int, str]:
    """Mirror of ``Functions_HuggingFace.build_reference_to_alignment_column_map`` (2598).

    Same globalms parameters (2.0/-1.0/-10.0/-0.5, one_alignment_only) on purpose:
    the frequency prior must be laid out on the identical reference->column map
    that ``compute_observed_diversity_profile_fast`` uses on the evaluation side.

    Returns ``(mapping_1based, aln_len, matched_pairs, consensus_seq)``.
    """
    consensus_seq, consensus_to_alignment_col, aln_len = build_consensus_and_column_map(
        sequences, valid_residues=valid_residues, ignore_chars=ignore_chars
    )
    if not consensus_seq:
        return {}, aln_len, 0, ""

    alignments = pairwise2.align.globalms(
        reference_protein,
        consensus_seq,
        2.0,
        -1.0,
        -10.0,
        -0.5,
        one_alignment_only=True,
    )
    if len(alignments) == 0:
        return {}, aln_len, 0, consensus_seq

    ref_aln, cons_aln = alignments[0].seqA, alignments[0].seqB

    ref_pos = 0
    cons_pos = 0
    mapping: Dict[int, int] = {}
    matched_pairs = 0
    for ref_char, cons_char in zip(ref_aln, cons_aln):
        if ref_char != "-":
            ref_pos += 1
        if cons_char != "-":
            cons_pos += 1
        if ref_char != "-" and cons_char != "-":
            if 1 <= cons_pos <= len(consensus_to_alignment_col):
                mapping[ref_pos] = consensus_to_alignment_col[cons_pos - 1]
                matched_pairs += 1

    return mapping, aln_len, matched_pairs, consensus_seq
