#!/usr/bin/env python3
"""Shared fixtures for the ``prescott_iav`` test suite.

DESIGN RULE FOR EVERYTHING IN THIS FILE
=======================================
A fixture whose expected values have to be computed by the code under test is
worthless: it can only ever assert that the code agrees with itself.  Every
fixture here therefore ships **independently-derived ground truth** alongside
the artefact, and the ground truth is either

* a literal (the 72-aa query protein, the planted mutation counts), or
* arithmetic done by hand in the docstring (the Henikoff weights of
  :data:`HANDWORKED_MSA_ROWS`, the circular variance of the CA ladder), or
* a closed form a reader can check in one line (a 50/50 column has exactly
  1.0 bit of entropy; an isolated glycine clips to RSA 1.0; a column of
  identical values softmaxes to exactly 1/20).

Denominators are chosen to make this possible: the synthetic panels hold
exactly 100 records, so every planted frequency is an exact decimal, and the
one column with gaps has depth exactly 95.

WHAT IS HERE
============
Query / sequence
    :func:`query_protein_fasta`, :func:`query_cds_fasta`, :data:`QUERY_PROTEIN`,
    :data:`QUERY_CDS`

Alignments
    :func:`tiny_msa` (controlled conservation: perfectly conserved, all-gap and
    hypervariable columns), :func:`uniform_msa` (weights are exactly 1.0),
    :func:`handworked_msa` (Henikoff weights derived by hand)

Structures
    :func:`cv_ladder_pdb` / :func:`cv_context_pdb` (circular variance is exactly
    0.0 or exactly 1.0 by symmetry), :func:`sasa_monomer_pdb` /
    :func:`sasa_context_pdb` (an isolated residue clips to RSA 1.0; the same
    residue inside a 12-atom shell is buried)

Panels
    :func:`frequency_panels` (planted per-position counts, so
    ``build_parent_frequency_file`` output is known exactly),
    :func:`leakage_panels` (a deliberately planted duplicate, plus a deep set)

Pipeline artefacts
    :func:`fake_jet_res` / :func:`jet_res_factory`,
    :func:`fake_escott_matrix` / :func:`escott_matrix_factory` (real R
    ``write.table`` format), :func:`frequency_file_factory`,
    :func:`guide_factory` / :func:`five_lineage_guide`,
    :func:`prepared_inputs_tree`, :func:`output_dir_factory`

Environment
    :func:`prescott_python`, :func:`subprocess_env`, plus marker-driven skips
    (see :data:`MARKER_REQUIREMENTS` and the README).

IMPORTING THE CODE UNDER TEST
=============================
Two pieces of wiring happen exactly once, at the top of this module.

``sys.path``
    the repository root and ``<repo>/scripts`` are prepended if absent.

``PATH``
    the PRESCOTT env's ``bin`` is moved to the front.  See
    :func:`_prepend_prescott_bin_to_path` -- with an unprepared PATH the system
    ``makeblastdb`` is picked up and *hangs* rather than failing, which turns a
    test run into a non-responding process.

Test modules may therefore just do::

    from prescott_iav import common, constants, run_escott   # package form
    import run_mutational_accessibility                      # scripts/ form

and the ``driver_module`` fixture loads ``scripts/run_prescott_diversity.py``
(which is a script, not a module) by file location.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

# --------------------------------------------------------------------------- #
# sys.path wiring -- done ONCE, at import, before anything else in the suite.
#
# This suite lives outside tests/, so nothing else puts the repo on the path.
# --------------------------------------------------------------------------- #

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PRESCOTT_IAV_DIR = SCRIPTS_DIR / "prescott_iav"
DRIVER_PATH = SCRIPTS_DIR / "run_prescott_diversity.py"
RMA_PATH = SCRIPTS_DIR / "run_mutational_accessibility.py"

for _path in (SCRIPTS_DIR, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Every figure-producing code path in the driver must be head-safe.  Set before
# any test imports matplotlib (directly or through run_mutational_accessibility).
os.environ.setdefault("MPLBACKEND", "Agg")

PRESCOTT_ENV_BIN = Path("/home3/oml4h/miniconda3/envs/PRESCOTT/bin")
PRESCOTT_PYTHON = PRESCOTT_ENV_BIN / "python"


def _prepend_prescott_bin_to_path() -> None:
    """Put the PRESCOTT env's ``bin`` FIRST on ``PATH``, for this process and its children.

    This is not a convenience.  The modules under test resolve their external
    tools by bare name (``makeblastdb``, ``blastp``, ``mafft``, ``mkdssp``,
    ``Rscript``, ``escott``), and on this machine the bare names resolve to
    ``/software/blast-v2.11.0/bin`` unless the env is in front.  That system
    ``makeblastdb`` does not fail -- it **hangs**, indefinitely, inside
    ``leakage_check.blast_records``.  A test suite that inherits an unprepared
    PATH therefore does not report a red test, it stops responding, and no
    ``--timeout`` short of the per-test one will tell you why.

    Doing it here rather than in each fixture means it also covers subprocesses
    the modules spawn themselves, which is where the hang actually lives.
    """
    if not PRESCOTT_ENV_BIN.is_dir():
        return
    entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if entries and entries[0] == str(PRESCOTT_ENV_BIN):
        return
    entries = [p for p in entries if p != str(PRESCOTT_ENV_BIN)]
    os.environ["PATH"] = os.pathsep.join([str(PRESCOTT_ENV_BIN), *entries])


_prepend_prescott_bin_to_path()


# --------------------------------------------------------------------------- #
# Real production inputs -- referenced only by @pytest.mark.requires_real_data
# tests, which are opt-in (see --run-slow).
# --------------------------------------------------------------------------- #

REAL_GUIDE = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"
REAL_LINEAGE_FILES = REPO_ROOT / "Sequences" / "IAV_lineage_files"
BLAT_REFERENCE_JET_RES = Path("/home3/oml4h/PRESCOTT/data/BLAT_jet.res")
BLAT_REFERENCE_MSA = Path("/home3/oml4h/PRESCOTT/data/aliBLAT.fasta")
BLAT_REFERENCE_PDB = Path("/home3/oml4h/PRESCOTT/data/blat-af2.pdb")


# --------------------------------------------------------------------------- #
# The lineage ladder.
#
# Repeated here as a LITERAL rather than imported from prescott_iav.constants on
# purpose: a test that reads the map from the module can never notice the module
# regressing to the old, wrong K <- J.2_int edge.  This is the corrected map --
# K descends from J.2.4 (K's own reference FASTA header carries the clade call
# J.2.4.1, corroborated by Sequences/nextclade_id_clade.tsv).
# --------------------------------------------------------------------------- #

EXPECTED_PARENT_MAP: Dict[str, str] = {
    "J_int": "G.1",
    "J.2_int": "J_int",
    "J.2.4": "J.2_int",
    "K": "J.2.4",
}
"""The `clade_evidence` preset: the single linear ladder G.1 -> J_int -> J.2_int -> J.2.4 -> K."""

EXPECTED_SENSITIVITY_PARENT_MAP: Dict[str, str] = {
    "J_int": "G.1",
    "J.2_int": "J_int",
    "J.2.4": "J.2_int",
    "K": "J.2_int",
}
"""The `brief_as_stated` preset.  It exists ONLY as the labelled sensitivity
alternative; the single contested edge is K's parent."""

CONTESTED_EDGE = ("K", "J.2.4", "J.2_int")  # (child, default parent, sensitivity parent)

LINEAGE_ORDER: Tuple[str, ...] = ("G.1", "J_int", "J.2_int", "J.2.4", "K")
EXPECTED_INPUT_ONLY_LINEAGES = frozenset({"G.1"})
EXPECTED_LINEAGE_TAGS: Dict[str, str] = {
    "G.1": "G1",
    "J_int": "J",
    "J.2_int": "J2",
    "J.2.4": "J24",
    "K": "K",
}
EXPECTED_PROT_TOKENS: Dict[str, str] = {k: f"HA{v}" for k, v in EXPECTED_LINEAGE_TAGS.items()}

# jet_surrogate's tuned defaults, literal for the same reason as the parent map.
EXPECTED_TRACE_TOP_FRACTION = 0.90
EXPECTED_MAX_ZERO_TRACE_FRACTION = 0.10
EXPECTED_WARN_ZERO_TRACE_FRACTION = 0.05


# --------------------------------------------------------------------------- #
# Alphabets and small independent lookups.
#
# Defined here rather than imported so a test comparing against them is testing
# the module, not comparing the module with itself.
# --------------------------------------------------------------------------- #

AA20: str = "ACDEFGHIKLMNPQRSTVWY"
"""The 20 standard amino acids, alphabetical."""

ESCOTT_ROW_ORDER: Tuple[str, ...] = tuple("acdefghiklmnpqrstvwy")
"""computePred.R:12 -- ESCOTT's own row order: lowercase, alphabetical."""

PLM_CACHE_ROW_ORDER: Tuple[str, ...] = (
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
)
"""Functions_HuggingFace.py:821 -- the row order a PLM probability cache uses."""

THREE_LETTER: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}

ONE_CODON_PER_AA: Dict[str, str] = {
    "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT", "G": "GGT",
    "H": "CAT", "I": "ATT", "K": "AAA", "L": "CTT", "M": "ATG", "N": "AAT",
    "P": "CCT", "Q": "CAA", "R": "CGT", "S": "TCT", "T": "ACT", "V": "GTT",
    "W": "TGG", "Y": "TAT",
}
STOP_CODON = "TGA"


# --------------------------------------------------------------------------- #
# The synthetic query: 72 aa of HA-like protein and its CDS.
# --------------------------------------------------------------------------- #

QUERY_PROTEIN: str = (
    "MKTIIALSYI"   #  1-10
    "LCLVFAQKLP"   # 11-20
    "GNDNSTATLC"   # 21-30
    "LGHHAVPNGT"   # 31-40
    "IVKTITNDQI"   # 41-50
    "EVTNATELVQ"   # 51-60
    "SSSTGEICDS"   # 61-70
    "PH"           # 71-72
)
"""72 aa: the N-terminus of H3 HA (signal peptide + start of HA1).

Short enough that a 20 x L matrix is readable in a failure message, long enough
that the structure fixtures have somewhere to put a burial gradient.  Residues
referenced by the panel and MSA ground truth below:

    1 M   5 I  10 I  12 C  15 F  18 K  20 P  25 S  30 C
   33 H  35 A  40 T  45 I  50 I  60 Q  72 H
"""

QUERY_LENGTH: int = len(QUERY_PROTEIN)

QUERY_CDS: str = "".join(ONE_CODON_PER_AA[aa] for aa in QUERY_PROTEIN) + STOP_CODON
"""219 nt = 72 codons + TGA.  ``common.translate_reference_cds`` strips the '*',
so translating this must give :data:`QUERY_PROTEIN` back exactly."""

QUERY_HEADER = "HAK"
"""escott splits the FASTA description at the first non-alphanumeric character
(escott.py:76) and names every output after the result, so a purely alphanumeric
header is required: ``escott_prot_token('HAK') == 'HAK'``."""

PARENT_PROTEIN: str = QUERY_PROTEIN[:39] + "I" + QUERY_PROTEIN[40:]
"""The parent lineage's reference: identical to the child except T40 -> I40.

That single difference is what makes ``drop_parent_reversions`` testable in
isolation.  The planted frequency of the reversion T40I is 0.12 -- far below the
0.95 ``freq_max`` cutoff -- so the frequency threshold CANNOT catch it and only
the identity-based reversion filter can.
"""
assert len(PARENT_PROTEIN) == QUERY_LENGTH
assert sum(a != b for a, b in zip(PARENT_PROTEIN, QUERY_PROTEIN)) == 1
assert QUERY_PROTEIN[39] == "T" and PARENT_PROTEIN[39] == "I"


# --------------------------------------------------------------------------- #
# Small IO helpers (independent of the code under test).
# --------------------------------------------------------------------------- #

def write_fasta(path: Path, records: Sequence[Tuple[str, str]], line_width: int = 0) -> Path:
    """Write records as FASTA.  ``line_width=0`` means one line per sequence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            if line_width and line_width > 0:
                for start in range(0, len(sequence), line_width):
                    handle.write(sequence[start:start + line_width] + "\n")
            else:
                handle.write(sequence + "\n")
    return path


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Minimal FASTA reader, so parser tests are not validated by the parser under test."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def md5_file(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def translate_cds(cds: str) -> str:
    """Independent translation, for asserting against ``common.translate_reference_cds``."""
    table = {codon: aa for aa, codon in ONE_CODON_PER_AA.items()}
    protein = []
    for i in range(0, len(cds) - len(cds) % 3, 3):
        codon = cds[i:i + 3]
        if codon == STOP_CODON:
            continue
        protein.append(table[codon])
    return "".join(protein)


assert translate_cds(QUERY_CDS) == QUERY_PROTEIN


# --------------------------------------------------------------------------- #
# MSA fixtures with controlled conservation.
# --------------------------------------------------------------------------- #

MSA_N_ROWS = 12
"""Query plus 11 homologues.  11 is chosen so a hypervariable column can hold 11
DISTINCT non-wild-type residues, i.e. the maximum variability the row count
allows, without repeating a residue."""

MSA_CONSERVED_POSITIONS: Tuple[int, ...] = tuple(range(1, 9)) + tuple(range(61, 73))
"""Columns where all 12 rows carry the query residue: exactly 1 residue type,
occupancy 1.0 (exact, whatever the sequence weights turn out to be)."""

MSA_ALL_GAP_POSITIONS: Tuple[int, ...] = (9, 19, 29)
"""Columns where the 11 non-query rows are '-' and only the query has a residue.

The query row is gap-free by construction (``build_jet_table`` refuses an MSA
whose first row has gaps), so 'all-gap' means all-gap-except-the-query -- which
is exactly the shape that makes an unweighted conservation score claim a column
is perfectly conserved when it is really just empty."""

MSA_HYPERVARIABLE_POSITIONS: Tuple[int, ...] = (10, 20, 30)
"""Columns holding 12 distinct residues: the query's plus 11 others."""

MSA_SEMI_CONSERVED_POSITIONS: Tuple[int, ...] = (15, 25, 35)
"""Columns holding 9 wild-type and 3 of a single alternative residue."""

MSA_SEMI_ALTERNATIVE = "V"
MSA_MILD_ALTERNATIVE = "G"
MSA_MILD_COUNT = 1
"""Every column not named above is 11 wild-type plus 1 :data:`MSA_MILD_ALTERNATIVE`."""


def _other_residues(wt: str, n: int) -> List[str]:
    """The first ``n`` residues of :data:`AA20` that are not ``wt``.  Deterministic."""
    return [aa for aa in AA20 if aa != wt][:n]


def _build_tiny_msa_rows() -> List[str]:
    """Assemble the 12 aligned rows from the literal column recipe above."""
    rows = [list(QUERY_PROTEIN) for _ in range(MSA_N_ROWS)]
    for pos in range(1, QUERY_LENGTH + 1):
        wt = QUERY_PROTEIN[pos - 1]
        if pos in MSA_CONSERVED_POSITIONS:
            continue  # already all-wild-type
        if pos in MSA_ALL_GAP_POSITIONS:
            for row in range(1, MSA_N_ROWS):
                rows[row][pos - 1] = "-"
        elif pos in MSA_HYPERVARIABLE_POSITIONS:
            for row, aa in enumerate(_other_residues(wt, MSA_N_ROWS - 1), start=1):
                rows[row][pos - 1] = aa
        elif pos in MSA_SEMI_CONSERVED_POSITIONS:
            alt = MSA_SEMI_ALTERNATIVE if MSA_SEMI_ALTERNATIVE != wt else "L"
            for row in (1, 2, 3):
                rows[row][pos - 1] = alt
        else:
            alt = MSA_MILD_ALTERNATIVE if MSA_MILD_ALTERNATIVE != wt else "A"
            for row in range(1, 1 + MSA_MILD_COUNT):
                rows[row][pos - 1] = alt
    return ["".join(row) for row in rows]


TINY_MSA_ROWS: Tuple[str, ...] = tuple(_build_tiny_msa_rows())
TINY_MSA_HEADERS: Tuple[str, ...] = (QUERY_HEADER,) + tuple(
    f"homolog_{i:02d}" for i in range(1, MSA_N_ROWS)
)


def _column_counts(rows: Sequence[str]) -> Dict[int, Dict[str, int]]:
    """{1-based column -> {character -> count}} over ALL rows, gaps included as '-'."""
    return {
        pos: dict(Counter(row[pos - 1] for row in rows))
        for pos in range(1, len(rows[0]) + 1)
    }


TINY_MSA_COLUMN_COUNTS: Dict[int, Dict[str, int]] = _column_counts(TINY_MSA_ROWS)
"""Ground truth derived from the literal rows, not from the encoder under test."""

TINY_MSA_OCCUPANCY_COUNTS: Dict[int, int] = {
    pos: sum(count for char, count in counts.items() if char != "-")
    for pos, counts in TINY_MSA_COLUMN_COUNTS.items()
}

TINY_MSA_N_RESIDUE_TYPES: Dict[int, int] = {
    pos: len([char for char in counts if char != "-"])
    for pos, counts in TINY_MSA_COLUMN_COUNTS.items()
}

TINY_MSA_COLUMN_CLASS: Dict[int, str] = {
    pos: (
        "conserved" if pos in MSA_CONSERVED_POSITIONS else
        "all_gap" if pos in MSA_ALL_GAP_POSITIONS else
        "hypervariable" if pos in MSA_HYPERVARIABLE_POSITIONS else
        "semi_conserved" if pos in MSA_SEMI_CONSERVED_POSITIONS else
        "mild"
    )
    for pos in range(1, QUERY_LENGTH + 1)
}

# Self-checks: the recipe must actually produce the classes it claims.  If one of
# these fires the fixture is lying to every test that trusts it.
assert TINY_MSA_ROWS[0] == QUERY_PROTEIN, "MSA row 1 must be the ungapped query"
assert all(len(row) == QUERY_LENGTH for row in TINY_MSA_ROWS)
assert all(TINY_MSA_N_RESIDUE_TYPES[p] == 1 for p in MSA_CONSERVED_POSITIONS)
assert all(TINY_MSA_OCCUPANCY_COUNTS[p] == MSA_N_ROWS for p in MSA_CONSERVED_POSITIONS)
assert all(TINY_MSA_OCCUPANCY_COUNTS[p] == 1 for p in MSA_ALL_GAP_POSITIONS)
assert all(TINY_MSA_N_RESIDUE_TYPES[p] == MSA_N_ROWS for p in MSA_HYPERVARIABLE_POSITIONS)
assert all(TINY_MSA_N_RESIDUE_TYPES[p] == 2 for p in MSA_SEMI_CONSERVED_POSITIONS)


UNIFORM_MSA_ROWS: Tuple[str, ...] = tuple([QUERY_PROTEIN] * MSA_N_ROWS)
"""Every row identical to the query.

Ground truth, by hand: in every column there is r=1 residue type present in
n=12 rows, so each row's Henikoff contribution per column is 1/(1*12) and its
raw weight is L/12 -- the SAME for every row.  Normalising to mean 1 therefore
gives **exactly 1.0 for all 12 rows**, and every column's occupancy is exactly
1.0.  Any deviation is a bug in the weighting, not in the alignment."""

UNIFORM_MSA_EXPECTED_WEIGHTS: Tuple[float, ...] = tuple([1.0] * MSA_N_ROWS)


HANDWORKED_MSA_ROWS: Tuple[str, ...] = ("AAAA", "AAAC", "AACC", "ACCC")
r"""A 4 x 4 alignment whose Henikoff & Henikoff weights are computed by hand.

``w_k = sum_i 1 / (r_i * n_{i,x_ik})`` over the columns where row k has a residue,
with ``r_i`` = number of distinct residue types in column i:

    col 0  = A A A A   -> r=1, n_A=4          -> every row gets 1/(1*4) = 1/4
    col 1  = A A A C   -> r=2, n_A=3, n_C=1   -> rows 0,1,2: 1/6   row 3: 1/2
    col 2  = A A C C   -> r=2, n_A=2, n_C=2   -> every row gets 1/4
    col 3  = A C C C   -> r=2, n_A=1, n_C=3   -> row 0: 1/2   rows 1,2,3: 1/6

    raw w0 = 1/4 + 1/6 + 1/4 + 1/2 = 7/6
    raw w1 = 1/4 + 1/6 + 1/4 + 1/6 = 5/6
    raw w2 = 1/4 + 1/6 + 1/4 + 1/6 = 5/6
    raw w3 = 1/4 + 1/2 + 1/4 + 1/6 = 7/6
    total  = 4  ==  n_rows, so the mean-1 normalisation is the identity here.

Hence the weights below are exact, and the symmetry w0 == w3, w1 == w2 is the
structural fact (rows 0 and 3 are the two extremes of the A/C gradient) that any
correct implementation must reproduce."""

HANDWORKED_MSA_EXPECTED_WEIGHTS: Tuple[float, ...] = (7 / 6, 5 / 6, 5 / 6, 7 / 6)
HANDWORKED_MSA_HEADERS: Tuple[str, ...] = ("query", "h1", "h2", "h3")


# --------------------------------------------------------------------------- #
# Synthetic structures with exactly-known geometry.
# --------------------------------------------------------------------------- #

def pdb_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str = "C",
    occupancy: float = 1.0,
    bfactor: float = 0.0,
) -> str:
    """One PDB ATOM record in strict column-format (78 characters)."""
    return (
        f"ATOM  {serial:5d} {name:^4s}"
        f"{'':1s}{resname:>3s} {chain:1s}{resseq:4d}{'':1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}"
        f"{'':10s}{element:>2s}"
    )


def build_pdb(atoms: Sequence[Tuple[str, str, str, int, float, float, float]]) -> str:
    """``atoms`` = [(atom_name, resname, chain, resseq, x, y, z), ...] -> PDB text."""
    lines = []
    for serial, (name, resname, chain, resseq, x, y, z) in enumerate(atoms, start=1):
        lines.append(pdb_atom_line(serial, name, resname, chain, resseq, x, y, z))
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


CV_LADDER_N_RESIDUES = 8
CV_LADDER_SPACING = 3.8
"""Ca-Ca spacing.  With the default 7.0 A circular-variance radius this puts the
i+-1 neighbours inside the sphere (3.8) and the i+-2 neighbours outside it
(7.6 > 7.0), with 0.6 A of margin on both sides -- so the neighbour set of every
residue is exact and not a floating-point coin flip."""

CV_LADDER_EXPECTED_MONOMER: Dict[int, float] = {
    **{pos: 1.0 for pos in range(2, CV_LADDER_N_RESIDUES)},
    1: 0.0,
    CV_LADDER_N_RESIDUES: 0.0,
}
r"""Circular variance ``cv_i = 1 - || mean_j (r_j - r_i)/||r_j - r_i|| ||``
on a straight one-atom-per-residue ladder along +x:

* an INTERIOR residue sees exactly two neighbours, at +x and -x.  The two unit
  vectors are exact opposites, their mean is the zero vector, so cv = **1.0**;
* a TERMINAL residue sees exactly one neighbour.  The mean of a single unit
  vector has norm 1, so cv = **0.0**.

Both values are exact for any radius in (3.8, 7.6), which is why the assertion
is ``== 1.0`` and not ``approx``."""

CV_CONTEXT_PARTNER_X = -CV_LADDER_SPACING
CV_LADDER_EXPECTED_CONTEXT: Dict[int, float] = {
    **CV_LADDER_EXPECTED_MONOMER,
    1: 1.0,
}
r"""The same chain A, evaluated inside a two-chain context.

Chain B contributes a single atom at x = -3.8, i.e. residue 1's missing
neighbour.  Residue 1 therefore becomes interior (cv 0.0 -> **1.0**) while every
other residue is unchanged: chain B's atom is 7.6 A from residue 2 and further
from the rest, so it is outside the 7.0 A sphere of all of them.

This is the monomer-versus-trimer question in miniature.  Score chain A alone
and residue 1 reads as maximally protruding; score it in context and it reads as
buried.  A test that only ever builds the monomer cannot tell the two apart."""


def cv_ladder_atoms(chain: str = "A") -> List[Tuple[str, str, str, int, float, float, float]]:
    return [
        ("CA", "GLY", chain, i, round((i - 1) * CV_LADDER_SPACING, 3), 0.0, 0.0)
        for i in range(1, CV_LADDER_N_RESIDUES + 1)
    ]


def _icosahedron_unit_vectors() -> List[Tuple[float, float, float]]:
    """The 12 vertices of a regular icosahedron, normalised.  Deterministic."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    norm = math.sqrt(1.0 + phi * phi)
    raw: List[Tuple[float, float, float]] = []
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            raw.append((0.0, s1 * 1.0, s2 * phi))
            raw.append((s1 * 1.0, s2 * phi, 0.0))
            raw.append((s2 * phi, 0.0, s1 * 1.0))
    return [(x / norm, y / norm, z / norm) for (x, y, z) in raw]


SASA_SHELL_RADIUS = 3.0
"""Shell radius.  A carbon's van der Waals radius is ~1.7-1.9 A and the solvent
probe is 1.4 A, so 12 atoms at 3.0 A occlude the central atom completely for any
plausible radius set: the buried residue's SASA is 0."""

SASA_ISOLATED_POSITION = (100.0, 0.0, 0.0)
SASA_BURIED_RESNUM = 1
SASA_ISOLATED_RESNUM = 2

SASA_EXPECTED_ISOLATED_RSA = 1.0
r"""An isolated glycine CA is a lone sphere: SASA = 4*pi*(r + 1.4)^2, which is
>= 109 A^2 even for the smallest carbon radius in common use (1.55) and 135 A^2
for the ProtOr radius (1.88).  Tien et al.'s theoretical maximum ASA for glycine
is 104 A^2, so the ratio exceeds 1 under every radius set and ``relative_sasa``
clips it to **exactly 1.0**.  The assertion is therefore an equality, and it does
not depend on which classifier freesasa picks."""

SASA_EXPECTED_BURIED_MAX_RSA = 0.05
"""The shelled residue must come back essentially zero.  Stated as a ceiling
rather than an equality because the exact residual depends on the radius set;
the *qualitative* fact -- fully enclosed means not solvent accessible -- does
not."""


def sasa_monomer_atoms() -> List[Tuple[str, str, str, int, float, float, float]]:
    """Chain A only: residue 1 at the origin, residue 2 far away.  Both exposed."""
    return [
        ("CA", "GLY", "A", SASA_BURIED_RESNUM, 0.0, 0.0, 0.0),
        ("CA", "GLY", "A", SASA_ISOLATED_RESNUM, *SASA_ISOLATED_POSITION),
    ]


def sasa_context_atoms() -> List[Tuple[str, str, str, int, float, float, float]]:
    """Chain A as above, plus a 12-residue chain B shell around residue 1."""
    atoms = sasa_monomer_atoms()
    for idx, (ux, uy, uz) in enumerate(_icosahedron_unit_vectors(), start=1):
        atoms.append((
            "CA", "GLY", "B", idx,
            round(ux * SASA_SHELL_RADIUS, 3),
            round(uy * SASA_SHELL_RADIUS, 3),
            round(uz * SASA_SHELL_RADIUS, 3),
        ))
    return atoms


def full_length_query_pdb_atoms(
    chain: str = "A",
    protein: str = QUERY_PROTEIN,
    spacing: float = CV_LADDER_SPACING,
    covered: Optional[Iterable[int]] = None,
) -> List[Tuple[str, str, str, int, float, float, float]]:
    """A CA-only backbone numbered 1..len(protein) in QUERY numbering.

    ``covered`` restricts which residue numbers get coordinates, which is how a
    partial-coverage structure (the real one covers 566/566, but earlier ones did
    not) is simulated: the uncovered positions must fall back to pure trace in
    ``combine_weight``.
    """
    wanted = set(covered) if covered is not None else set(range(1, len(protein) + 1))
    return [
        ("CA", THREE_LETTER[protein[i - 1]], chain, i, round((i - 1) * spacing, 3), 0.0, 0.0)
        for i in range(1, len(protein) + 1)
        if i in wanted
    ]


# --------------------------------------------------------------------------- #
# Panels with planted, exactly-known composition.
#
# Every panel holds exactly 100 records so a planted count IS the percentage.
# --------------------------------------------------------------------------- #

PANEL_N_RECORDS = 100

PARENT_PANEL_SPEC: Dict[int, Dict[str, int]] = {
    10: {"K": 10},              # I10K  -> 10/100 = 0.10
    15: {"D": 25},              # F15D  -> 25/100 = 0.25
    20: {"E": 19, "-": 5},      # P20E  -> 19/95  = 0.20  (depth 95, not 100)
    25: {"W": 1},               # S25W  -> 1/100  = 0.01  (single observation)
    30: {"Y": 98},              # C30Y  -> 98/100 = 0.98  (>= freq_max 0.95)
    35: {"D": 30, "S": 20},     # A35D 0.30, A35S 0.20   (multi-allelic site)
    40: {"I": 12},              # T40I  -> 12/100 = 0.12  (the parent reversion)
}
"""Non-wild-type counts planted into the PARENT panel, keyed by 1-based query
position.  ``'-'`` plants gaps (they reduce the column depth but are not
mutants).  Every other position is 100/100 wild-type.

Each entry exercises exactly one branch of ``build_parent_frequency_file``:

    pos 10, 15, 35   ordinary standing variation, kept
    pos 20           gaps, so depth is 95 and the frequency denominator is NOT
                     the record count
    pos 25           a single observation: kept at --min-count 1, dropped at 2
    pos 30           frequency 0.98 >= freq_max, i.e. the ancestral residue at a
                     lineage-defining site, dropped by the frequency cutoff
    pos 40           the parent's own residue at a site where parent and child
                     differ.  At 0.12 the frequency cutoff cannot see it; only
                     drop_parent_reversions removes it.
"""

TARGET_PANEL_SPEC: Dict[int, Dict[str, int]] = {
    5:  {"V": 20},              # I5V   -> 20/100 = 0.20
    18: {"R": 50},              # K18R  -> 50/100 = 0.50  (exactly 1 bit)
    33: {"N": 10, "Q": 5},      # H33N 0.10, H33Q 0.05
    50: {"T": 1},               # I50T  -> 1/100  = 0.01
    60: {"H": 9, "-": 10},      # Q60H  -> 9/90   = 0.10  (depth 90)
}
"""Non-wild-type counts planted into the TARGET (evaluation) panel.

Position 18 is the important one: 50 K and 50 R is exactly two equally-frequent
states, so its Shannon entropy is **exactly 1.0 bit** and its 'fraction mutated'
is exactly 0.5.  Position 5 (80/20) has the closed form
-0.8*log2(0.8) - 0.2*log2(0.2) = 0.7219280948873623 bits.  Neither number has to
be produced by the code under test to be known."""


def _panel_records(
    protein: str,
    spec: Mapping[int, Mapping[str, int]],
    n_records: int,
    prefix: str,
) -> List[Tuple[str, str]]:
    """Materialise a panel with EXACTLY the planted counts.

    Variants are laid into disjoint row blocks per position, starting at row 0,
    which makes the composition exact and the records reproducible.
    """
    rows = [list(protein) for _ in range(n_records)]
    for pos, alleles in sorted(spec.items()):
        row = 0
        for char, count in sorted(alleles.items()):
            for _ in range(count):
                rows[row][pos - 1] = char
                row += 1
        if row > n_records:
            raise ValueError(f"panel spec over-subscribes position {pos}: {row} > {n_records}")
    return [(f"{prefix}{i:04d}", "".join(row)) for i, row in enumerate(rows)]


PARENT_PANEL_RECORDS: Tuple[Tuple[str, str], ...] = tuple(
    _panel_records(PARENT_PROTEIN, PARENT_PANEL_SPEC, PANEL_N_RECORDS, "PAR")
)
TARGET_PANEL_RECORDS: Tuple[Tuple[str, str], ...] = tuple(
    _panel_records(QUERY_PROTEIN, TARGET_PANEL_SPEC, PANEL_N_RECORDS, "TGT")
)


def _panel_ground_truth(
    reference_protein: str,
    records: Sequence[Tuple[str, str]],
) -> Dict[str, object]:
    """Per-position counts, depths and mutant frequencies, computed from the records."""
    length = len(reference_protein)
    counts = _column_counts([seq for _, seq in records])
    depths = {
        pos: sum(c for char, c in counts[pos].items() if char in AA20)
        for pos in range(1, length + 1)
    }
    frequencies: Dict[str, float] = {}
    for pos in range(1, length + 1):
        wt = reference_protein[pos - 1]
        for char, count in counts[pos].items():
            if char == wt or char not in AA20:
                continue
            frequencies[f"{wt}{pos}{char}"] = count / depths[pos]
    return {"counts": counts, "depths": depths, "frequencies": frequencies}


PARENT_PANEL_TRUTH = _panel_ground_truth(PARENT_PROTEIN, PARENT_PANEL_RECORDS)
TARGET_PANEL_TRUTH = _panel_ground_truth(QUERY_PROTEIN, TARGET_PANEL_RECORDS)

PARENT_PANEL_DEPTHS: Dict[int, int] = PARENT_PANEL_TRUTH["depths"]  # type: ignore[assignment]
TARGET_PANEL_DEPTHS: Dict[int, int] = TARGET_PANEL_TRUTH["depths"]  # type: ignore[assignment]

PARENT_PANEL_MEDIAN_DEPTH = 100.0
"""71 of the 72 columns have depth 100 and one has 95, so the median is exactly 100."""

# The mutants build_parent_frequency_file must emit for child=QUERY_PROTEIN,
# parent=PARENT_PROTEIN, at min_depth=50, freq_max=0.95.  Frequencies are quoted
# against the CHILD's wild-type letters, which is what PRESCOTT keys on.
EXPECTED_FREQUENCY_FILE_MIN_COUNT_1: Dict[str, float] = {
    "I10K": 0.10,
    "F15D": 0.25,
    "P20E": 19 / 95,   # == 0.2 exactly
    "S25W": 0.01,
    "A35D": 0.30,
    "A35S": 0.20,
}
EXPECTED_FREQUENCY_FILE_MIN_COUNT_2: Dict[str, float] = {
    key: value for key, value in EXPECTED_FREQUENCY_FILE_MIN_COUNT_1.items() if key != "S25W"
}
EXPECTED_FREQUENCY_FILE_DROPPED = {
    "C30Y": "frequency 0.98 >= freq_max 0.95 (ancestral residue at a defining site)",
    "T40I": "parent reversion: parent reference has I at 40 where the child has T",
    "S25W": "below --min-count when that is 2 (count 1)",
}
EXPECTED_N_REVERTED_ANCESTRAL_MUTANTS = 1     # C30Y
EXPECTED_N_PARENT_REVERSION_MUTANTS = 1       # T40I

assert abs(EXPECTED_FREQUENCY_FILE_MIN_COUNT_1["P20E"] - 0.2) < 1e-15
assert PARENT_PANEL_DEPTHS[20] == 95
assert TARGET_PANEL_DEPTHS[60] == 90

# Closed-form diversity ground truth for the TARGET panel.
TARGET_PANEL_EXPECTED_ENTROPY_BITS: Dict[int, float] = {
    18: 1.0,                                                    # 50/50
    5: -(0.8 * math.log2(0.8)) - (0.2 * math.log2(0.2)),        # 80/20
    1: 0.0,                                                     # invariant column
}
TARGET_PANEL_EXPECTED_MUTANT_FRACTION: Dict[int, float] = {
    5: 0.20, 18: 0.50, 33: 0.15, 50: 0.01, 60: 0.10, 1: 0.0,
}
assert abs(TARGET_PANEL_EXPECTED_ENTROPY_BITS[5] - 0.7219280948873623) < 1e-15


# --------------------------------------------------------------------------- #
# Leakage: panels with a deliberately planted duplicate.
#
# Kept separate from the frequency panels on purpose.  Planting a duplicate into
# a panel whose composition is ground truth would perturb that ground truth, and
# a fixture that has to be corrected for its own contamination is not ground
# truth any more.
# --------------------------------------------------------------------------- #

LEAKAGE_SIGNAL_PEPTIDE = "MKTIIALSYILCLVFA"
"""The first 16 residues of :data:`QUERY_PROTEIN`.

The real geometry the leakage checker has to survive: the deep pre-cutoff set
carries HA's signal peptide and the contemporary panels start at the mature
N-terminus, so a genuine duplicate is NOT byte-identical and NOT in a common
frame.  Hashing alone can never see it; only alignment can.  The synthetic sets
below reproduce that offset."""

LEAKAGE_MATURE_QUERY = QUERY_PROTEIN[len(LEAKAGE_SIGNAL_PEPTIDE):]

LEAKAGE_PLANTED_DUPLICATE_TARGET_ROW = 3
LEAKAGE_PLANTED_DUPLICATE_PARENT_ROW = 7
LEAKAGE_PLANTED_DEEP_ROW = 5
"""Row indices (0-based) of the planted duplicates.  A test that finds a leak at
any other index has found a false positive."""


def _mutated(seq: str, changes: Mapping[int, str]) -> str:
    """Apply ``{1-based position: residue}`` to ``seq``."""
    chars = list(seq)
    for pos, aa in changes.items():
        chars[pos - 1] = aa
    return "".join(chars)


def _divergent(seq: str, n: int, seed: int) -> str:
    """``n`` substitutions at deterministic, evenly-spread positions."""
    chars = list(seq)
    for i in range(n):
        pos = (seed * 7 + i * 11) % len(chars)
        chars[pos] = "W" if chars[pos] != "W" else "C"
    return "".join(chars)


# --------------------------------------------------------------------------- #
# Pipeline artefact writers.
# --------------------------------------------------------------------------- #

JET_COLUMNS: Tuple[str, ...] = ("AA", "pos", "chain", "pc", "tr", "freq", "trace", "cv")
"""The exact column layout escott's parser expects (escott.py:1124).  Note the
absence of ``traceMax``: computePred.R:61 prefers it over ``trace`` if present,
so emitting it silently changes which weight is used."""

JET_ZERO_TRACE_POSITIONS: Tuple[int, ...] = (9, 19, 29, 41, 52, 63, 70)
"""7 of 72 positions = 9.722% zero trace.

Deliberately inside the warning band: above WARN_ZERO_TRACE_FRACTION (5%) and
below MAX_ZERO_TRACE_FRACTION (10%), so the default table must WARN and still be
written.  ``jet_res_factory(n_zero=...)`` produces tables on either side of the
refusal ceiling."""

JET_EXPECTED_FRAC_ZERO = len(JET_ZERO_TRACE_POSITIONS) / QUERY_LENGTH
assert 0.05 < JET_EXPECTED_FRAC_ZERO < 0.10


def _jet_value(pos: int, offset: int) -> float:
    """Deterministic value in [0, 1], varying with position; 4 dp like the writer."""
    return round(((pos * 7 + offset * 13) % 11) / 10.0, 4)


def _jet_trace_value(pos: int) -> float:
    """Deterministic trace in [0.1, 1.0] -- STRICTLY positive.

    The trace column is the one place a coincidental zero would be a bug in the
    fixture rather than a feature of it: the zero-trace count is ground truth
    (:data:`JET_ZERO_TRACE_POSITIONS`), so only the positions named there may be
    zero.  Hence the 0.1 floor.
    """
    return round(0.1 + ((pos * 7) % 10) / 10.0 * 0.9, 4)


def write_jet_res(
    path: Path,
    protein: str = QUERY_PROTEIN,
    zero_trace_positions: Sequence[int] = JET_ZERO_TRACE_POSITIONS,
    chain: str = "A",
) -> Path:
    """Write a valid ``<prot>_jet.res``.

    Tab separated with a header, which is what ``jet_surrogate.write_jet_res``
    emits; the shipped JET2 file is space-padded instead.  Both parse under
    ``pd.read_table(sep=r'\\s+')``, and a test that cares about the difference
    should assert on both.
    """
    zeros = set(int(p) for p in zero_trace_positions)
    lines = ["\t".join(JET_COLUMNS)]
    for pos in range(1, len(protein) + 1):
        trace = 0.0 if pos in zeros else _jet_trace_value(pos)
        row = [
            THREE_LETTER[protein[pos - 1]],
            str(pos),
            chain,
            f"{_jet_value(pos, 0):.4f}",
            f"{_jet_value(pos, 1):.4f}",
            f"{_jet_value(pos, 2):.4f}",
            f"{trace:.4f}",
            f"{_jet_value(pos, 4):.4f}",
        ]
        lines.append("\t".join(row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def escott_value(aa_index: int, position: int) -> float:
    """The value ESCOTT is pretended to have produced for (row, column).

    ``-0.5 * ((aa_index + position) mod 20)``.  The modular shift means every
    column holds the SAME multiset of 20 values, just permuted, which gives two
    invariants a test can assert without recomputing anything:

    * every non-flat column has an identical set of softmax probabilities
      (identical multiset in, identical multiset out);
    * the column sums of ``exp(v/T)`` are identical across all non-flat columns,
      for every temperature.

    Values are negative-or-zero, matching the real convention: ESCOTT emits
    ``-normPred``, so **negative is deleterious**.
    """
    return -0.5 * ((aa_index + position) % 20)


ESCOTT_FLAT_POSITIONS: Tuple[int, ...] = JET_ZERO_TRACE_POSITIONS
"""Columns written as identically zero.

pred.R:487 multiplies column i by ``trace[i]``, so a zero-trace position comes
back as an all-zero column.  After the wild-type fill all 20 entries are equal,
so the per-column softmax is **exactly 1/20 = 0.05 for every residue** at any
temperature -- a site that contributes pure noise to every site-level metric.
Keeping these aligned with :data:`JET_ZERO_TRACE_POSITIONS` means the jet and
ESCOTT fixtures tell a consistent story."""

ESCOTT_EXPECTED_FLAT_PROBABILITY = 1.0 / 20.0


def escott_matrix_values(
    protein: str = QUERY_PROTEIN,
    flat_positions: Sequence[int] = ESCOTT_FLAT_POSITIONS,
) -> Dict[Tuple[str, int], Optional[float]]:
    """``{(uppercase_aa, 1-based position): value or None for the wild-type NA}``."""
    flat = set(int(p) for p in flat_positions)
    out: Dict[Tuple[str, int], Optional[float]] = {}
    for pos in range(1, len(protein) + 1):
        wt = protein[pos - 1]
        for aa_index, aa in enumerate(ESCOTT_ROW_ORDER):
            upper = aa.upper()
            if upper == wt:
                out[(upper, pos)] = None
            elif pos in flat:
                out[(upper, pos)] = 0.0
            else:
                out[(upper, pos)] = escott_value(aa_index, pos)
    return out


def write_escott_normpred(
    path: Path,
    protein: str = QUERY_PROTEIN,
    flat_positions: Sequence[int] = ESCOTT_FLAT_POSITIONS,
    values: Optional[Mapping[Tuple[str, int], Optional[float]]] = None,
) -> Path:
    r"""Write ``<prot>_normPred_evolCombi.txt`` in R's exact ``write.table`` format.

    Verified against real output (``HAJ_normPred_evolCombi.txt``).  The format is
    ``write.table(x)`` with all defaults, which means:

    * a header line of L **quoted** column names, space separated, with NO
      leading field for the row-name column;
    * L+1 fields per data row: a **quoted** row name then L unquoted numbers;
    * the wild-type cell is the bare token ``NA`` (unquoted);
    * row names are the 20 amino acids lowercase and alphabetical
      (computePred.R:12), column names are ``"<WT><pos>"`` with no offset.

    That header/row asymmetry is the whole point: it is what makes
    ``pd.read_table(sep=r"\s+")`` put the row names in the index, and a parser
    that assumes a square header silently shifts every column by one.
    """
    cells = dict(values) if values is not None else escott_matrix_values(protein, flat_positions)
    header = " ".join(f'"{protein[pos - 1]}{pos}"' for pos in range(1, len(protein) + 1))
    lines = [header]
    for aa in ESCOTT_ROW_ORDER:
        upper = aa.upper()
        fields = [f'"{aa}"']
        for pos in range(1, len(protein) + 1):
            value = cells[(upper, pos)]
            fields.append("NA" if value is None else repr(float(value)))
        lines.append(" ".join(fields))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_frequency_file(path: Path, frequencies: Mapping[str, float]) -> Path:
    """Write a PRESCOTT custom frequency file: ``<MUTANT> <freq>``, no header.

    The extension must NOT be ``.csv`` -- prescott.py:982 switches to the gnomAD
    parser on that suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{mutant} {freq:.10e}" for mutant, freq in sorted(frequencies.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_frequency_meta(path: Path, frequencies: Mapping[str, float], parent_label: str) -> Path:
    """The ``*_meta.tsv`` sidecar that records how each frequency was derived."""
    import re

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["mutant\tposition\twt\tmut\tcount\tdepth\tfrequency\tparent_lineage"]
    for mutant, freq in sorted(frequencies.items()):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant)
        assert match, mutant
        wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
        depth = PANEL_N_RECORDS
        rows.append(
            f"{mutant}\t{pos}\t{wt}\t{mut}\t{round(freq * depth)}\t{depth}\t{freq:.10e}\t{parent_label}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def write_score_matrix(path: Path, protein: str, probabilities: Mapping[str, Sequence[float]]) -> Path:
    """Write a ``plm_probability_profile.csv``-format score matrix.

    Row 0 is labelled ``sequence`` and holds the L residue letters; the next 20
    rows are the probabilities in :data:`PLM_CACHE_ROW_ORDER`; written with no
    header, exactly as ``run_mutational_accessibility.py:1033-1043`` does, so the
    SC2 loader reads it with ``index_col=0, header=None`` and needs no new code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence"] + list(protein))
        for aa in PLM_CACHE_ROW_ORDER:
            writer.writerow([aa] + [repr(float(v)) for v in probabilities[aa]])
    return path


def uniform_score_matrix_probabilities(length: int) -> Dict[str, List[float]]:
    """A matrix in which every column is the uniform 1/20 distribution."""
    return {aa: [1.0 / 20.0] * length for aa in PLM_CACHE_ROW_ORDER}


def write_guide_csv(path: Path, rows: Sequence[Tuple[str, Path, Path]]) -> Path:
    """Write a MONTHLY_GUIDE lineage guide: ``month,fasta,reference``.

    ``month`` is the lineage label, ``fasta`` the aligned protein diversity
    panel, ``reference`` the single-record NUCLEOTIDE CDS (the driver translates
    it).  Column names match ``Functions_HuggingFace.load_analysis_targets``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["month", "fasta", "reference"])
        for label, fasta, reference in rows:
            writer.writerow([label, str(fasta), str(reference)])
    return path


# --------------------------------------------------------------------------- #
# Capability detection and marker-driven skipping.
# --------------------------------------------------------------------------- #

def _have_binaries(*names: str) -> bool:
    return all(shutil.which(name) is not None for name in names)


def _have_import(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _have_paths(*paths: Path) -> bool:
    return all(Path(p).exists() for p in paths)


CAPABILITIES: Dict[str, Tuple[Callable[[], bool], str]] = {
    "requires_escott": (
        lambda: _have_binaries("escott", "prescott"),
        "the escott/prescott entry points are not on PATH "
        "(activate the PRESCOTT conda env)",
    ),
    "requires_blast": (
        lambda: _have_binaries("blastp", "makeblastdb"),
        "BLAST+ (blastp/makeblastdb) is not on PATH",
    ),
    "requires_mafft": (lambda: _have_binaries("mafft"), "mafft is not on PATH"),
    "requires_muscle": (lambda: _have_binaries("muscle"), "muscle is not on PATH"),
    "requires_dssp": (lambda: _have_binaries("mkdssp"), "mkdssp is not on PATH"),
    "requires_r": (lambda: _have_binaries("Rscript"), "Rscript is not on PATH"),
    "requires_prody": (lambda: _have_import("prody"), "prody is not importable"),
    "requires_freesasa": (lambda: _have_import("freesasa"), "freesasa is not importable"),
    "requires_scipy": (lambda: _have_import("scipy"), "scipy is not importable"),
    "requires_torch": (lambda: _have_import("torch"), "torch is not importable"),
    "requires_rma": (
        lambda: RMA_PATH.exists() and _have_import("torch"),
        "run_mutational_accessibility.py needs torch, which is not importable",
    ),
    "requires_real_data": (
        lambda: _have_paths(REAL_GUIDE, REAL_LINEAGE_FILES),
        "the real multi-GB production inputs are not present",
    ),
    "requires_blat_reference": (
        lambda: _have_paths(BLAT_REFERENCE_JET_RES, BLAT_REFERENCE_MSA, BLAT_REFERENCE_PDB),
        "the shipped PRESCOTT BLAT reference data is not present",
    ),
    "requires_prescott_python": (
        lambda: PRESCOTT_PYTHON.exists(),
        f"{PRESCOTT_PYTHON} does not exist",
    ),
}
"""marker name -> (availability probe, reason shown when it fails).

Probes are evaluated ONCE per session (see :func:`pytest_collection_modifyitems`)
and cached in :data:`_CAPABILITY_CACHE`, so a suite of 500 tests does not run 500
``shutil.which`` calls."""

_CAPABILITY_CACHE: Dict[str, bool] = {}

MARKER_REQUIREMENTS = CAPABILITIES  # documented alias used by the README


def capability_available(marker: str) -> bool:
    """Is the external dependency behind ``marker`` present?  Cached per session."""
    if marker not in _CAPABILITY_CACHE:
        probe, _ = CAPABILITIES[marker]
        try:
            _CAPABILITY_CACHE[marker] = bool(probe())
        except Exception:  # a broken probe must not take the suite down
            _CAPABILITY_CACHE[marker] = False
    return _CAPABILITY_CACHE[marker]


OPT_IN_MARKERS = ("slow", "requires_real_data")
"""Markers whose tests are skipped unless ``--run-slow`` is given.

These are the two categories that stop the default suite being fast and offline:
minute-scale work, and anything touching the multi-GB production inputs."""


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="also run tests marked `slow` or `requires_real_data` "
             "(minutes, and the real production data)",
    )


def pytest_configure(config):
    """Register every marker here as well as in pytest.ini.

    pytest.ini is the declaration a reader looks at; this makes ``--strict-markers``
    hold even when the suite is invoked with an explicit ``-c`` pointing somewhere
    else, which is how a CI job usually gets it wrong.
    """
    descriptions = {
        "unit": "pure, fast, no subprocess and no external binary",
        "integration": "several modules together, still fully synthetic",
        "cli": "exercises a module's argparse surface / main()",
        "slow": "minute-scale; skipped unless --run-slow",
        "blast": "legacy alias for requires_blast",
    }
    for name, (_, reason) in CAPABILITIES.items():
        descriptions[name] = f"skipped when: {reason}"
    for name, text in descriptions.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def pytest_collection_modifyitems(config, items):
    """Apply the capability skips and the ``--run-slow`` opt-in."""
    run_slow = config.getoption("--run-slow")
    opt_in_skip = pytest.mark.skip(
        reason="needs --run-slow (slow, or uses the real production data)"
    )
    for item in items:
        own_markers = set(item.keywords)
        if not run_slow and any(marker in own_markers for marker in OPT_IN_MARKERS):
            item.add_marker(opt_in_skip)
            continue
        for marker in CAPABILITIES:
            if marker in own_markers and not capability_available(marker):
                item.add_marker(pytest.mark.skip(reason=CAPABILITIES[marker][1]))
                break
        if "blast" in own_markers and not capability_available("requires_blast"):
            item.add_marker(pytest.mark.skip(reason=CAPABILITIES["requires_blast"][1]))


# --------------------------------------------------------------------------- #
# Environment fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, already on ``sys.path``."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def scripts_dir() -> Path:
    """``<repo>/scripts``, already on ``sys.path``."""
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def prescott_python() -> Path:
    """The one interpreter that can run the whole pipeline.

    Never use a bare ``python``/``pytest``/``Rscript`` in a subprocess: the system
    ones lack this project's dependencies.
    """
    if not PRESCOTT_PYTHON.exists():
        pytest.skip(f"{PRESCOTT_PYTHON} does not exist")
    return PRESCOTT_PYTHON


@pytest.fixture(scope="session")
def subprocess_env() -> Dict[str, str]:
    """``os.environ`` with the PRESCOTT env's ``bin`` first on PATH.

    Hand this to every ``subprocess.run``: it is what makes ``escott``,
    ``prescott``, ``mafft``, ``Rscript`` and ``blastp`` resolve to the versions
    the pipeline was verified against.
    """
    env = dict(os.environ)
    env["PATH"] = f"{PRESCOTT_ENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPTS_DIR), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["MPLBACKEND"] = "Agg"
    return env


@pytest.fixture(scope="session")
def run_module_cli(subprocess_env):
    """Run a stage-1 module's CLI in a subprocess and return the CompletedProcess.

    ``run_module_cli('jet_surrogate', ['--validate-only'])``.  Nothing is asserted
    about the return code: a test that expects a refusal wants to inspect it.
    """
    def _run(module: str, argv: Sequence[str], timeout: int = 600, cwd: Optional[Path] = None):
        script = PRESCOTT_IAV_DIR / f"{module}.py"
        if not script.exists():
            script = SCRIPTS_DIR / f"{module}.py"
        return subprocess.run(
            [str(PRESCOTT_PYTHON), str(script), *[str(a) for a in argv]],
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    return _run


# --------------------------------------------------------------------------- #
# Module-import fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def prescott_modules():
    """The five stage-1 modules plus ``constants``, imported once per session.

    Returns a dict keyed by module name.  Importing through the package
    (``prescott_iav.common``) rather than as bare modules is deliberate: it is the
    path the driver's subprocesses use, so an ``__init__``-level breakage is
    caught here rather than at run time.
    """
    modules = {}
    for name in ("constants", "common", "jet_surrogate", "prepare_inputs",
                 "run_escott", "leakage_check"):
        try:
            modules[name] = importlib.import_module(f"prescott_iav.{name}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            pytest.skip(f"cannot import prescott_iav.{name}: {exc}")
    return modules


@pytest.fixture(scope="session")
def driver_module():
    """``scripts/run_prescott_diversity.py``, loaded by file location.

    It is a script rather than an importable module, and it pulls in
    ``run_mutational_accessibility`` (hence torch), so this is session-scoped and
    skips rather than errors when torch is unavailable.
    """
    if not DRIVER_PATH.exists():
        pytest.skip(f"{DRIVER_PATH} does not exist")
    spec = importlib.util.spec_from_file_location("run_prescott_diversity", DRIVER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"cannot build an import spec for {DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_prescott_diversity", module)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"cannot import {DRIVER_PATH.name}: {exc}")
    return module


# --------------------------------------------------------------------------- #
# Sequence fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def query_protein() -> str:
    """The 72-aa synthetic HA-like query.  See :data:`QUERY_PROTEIN`."""
    return QUERY_PROTEIN


@pytest.fixture(scope="session")
def parent_protein() -> str:
    """The parent lineage reference: :data:`QUERY_PROTEIN` with T40I."""
    return PARENT_PROTEIN


@pytest.fixture()
def query_protein_fasta(tmp_path: Path) -> Path:
    """Single-record protein FASTA whose header yields the escott token ``HAK``."""
    return write_fasta(tmp_path / "query.fasta", [(QUERY_HEADER, QUERY_PROTEIN)])


@pytest.fixture()
def query_cds_fasta(tmp_path: Path) -> Path:
    """Single-record nucleotide CDS: 72 codons + TGA, translating to the query."""
    return write_fasta(
        tmp_path / "query_cds.nt.fa",
        [("EPI0000001|HA|A/Synthetic/1/2025|EPI_ISL_0000001|K", QUERY_CDS)],
    )


# --------------------------------------------------------------------------- #
# Alignment fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def tiny_msa(tmp_path: Path) -> Dict[str, object]:
    """A 12 x 72 alignment with deliberately mixed conservation.

    Returns ``path``, ``rows``, ``headers`` and the ground-truth dictionaries
    (``column_counts``, ``occupancy_counts``, ``n_residue_types``,
    ``column_class``) plus the position tuples for each class.

    Row 1 is the ungapped query, as ``build_jet_table`` requires; the all-gap
    columns are therefore all-gap-except-the-query, which is the shape that
    actually occurs and the shape that fools an unweighted conservation score.
    """
    path = write_fasta(
        tmp_path / "msa_tiny.fasta",
        list(zip(TINY_MSA_HEADERS, TINY_MSA_ROWS)),
    )
    return {
        "path": path,
        "rows": list(TINY_MSA_ROWS),
        "headers": list(TINY_MSA_HEADERS),
        "n_rows": MSA_N_ROWS,
        "n_columns": QUERY_LENGTH,
        "query": QUERY_PROTEIN,
        "column_counts": TINY_MSA_COLUMN_COUNTS,
        "occupancy_counts": TINY_MSA_OCCUPANCY_COUNTS,
        "n_residue_types": TINY_MSA_N_RESIDUE_TYPES,
        "column_class": TINY_MSA_COLUMN_CLASS,
        "conserved_positions": MSA_CONSERVED_POSITIONS,
        "all_gap_positions": MSA_ALL_GAP_POSITIONS,
        "hypervariable_positions": MSA_HYPERVARIABLE_POSITIONS,
        "semi_conserved_positions": MSA_SEMI_CONSERVED_POSITIONS,
    }


@pytest.fixture()
def uniform_msa(tmp_path: Path) -> Dict[str, object]:
    """12 identical rows: Henikoff weights are exactly 1.0 and occupancy exactly 1.0."""
    path = write_fasta(
        tmp_path / "msa_uniform.fasta",
        [(f"row_{i:02d}", seq) for i, seq in enumerate(UNIFORM_MSA_ROWS)],
    )
    return {
        "path": path,
        "rows": list(UNIFORM_MSA_ROWS),
        "expected_weights": list(UNIFORM_MSA_EXPECTED_WEIGHTS),
        "expected_occupancy": 1.0,
    }


@pytest.fixture()
def handworked_msa(tmp_path: Path) -> Dict[str, object]:
    """The 4 x 4 alignment whose Henikoff weights are (7/6, 5/6, 5/6, 7/6) by hand."""
    path = write_fasta(
        tmp_path / "msa_handworked.fasta",
        list(zip(HANDWORKED_MSA_HEADERS, HANDWORKED_MSA_ROWS)),
    )
    return {
        "path": path,
        "rows": list(HANDWORKED_MSA_ROWS),
        "expected_weights": list(HANDWORKED_MSA_EXPECTED_WEIGHTS),
    }


@pytest.fixture()
def gapped_query_msa(tmp_path: Path) -> Path:
    """An MSA whose FIRST row carries gaps -- ``build_jet_table`` must refuse it.

    Every downstream index (jet row, GEMME column, escott mutation label) is a
    query position, so a gapped query row silently shifts all of them.
    """
    rows = ["MK-TIIALSYI" + QUERY_PROTEIN[10:]] + list(TINY_MSA_ROWS[1:])
    padded = [row + "-" * (len(rows[0]) - len(row)) for row in rows]
    return write_fasta(
        tmp_path / "msa_gapped_query.fasta",
        list(zip(TINY_MSA_HEADERS, padded)),
    )


# --------------------------------------------------------------------------- #
# Structure fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def cv_ladder_pdb(tmp_path: Path) -> Dict[str, object]:
    """Chain A only: an 8-residue CA ladder at 3.8 A spacing.

    Ground truth (:data:`CV_LADDER_EXPECTED_MONOMER`): interior residues have
    circular variance exactly 1.0 (two exactly opposite neighbours), termini
    exactly 0.0 (one neighbour).
    """
    path = tmp_path / "cv_ladder.pdb"
    path.write_text(build_pdb(cv_ladder_atoms("A")), encoding="utf-8")
    return {
        "path": path,
        "chain": "A",
        "n_residues": CV_LADDER_N_RESIDUES,
        "spacing": CV_LADDER_SPACING,
        "expected_cv": dict(CV_LADDER_EXPECTED_MONOMER),
    }


@pytest.fixture()
def cv_context_pdb(tmp_path: Path) -> Dict[str, object]:
    """The same chain A ladder plus a one-atom chain B at x = -3.8.

    Ground truth (:data:`CV_LADDER_EXPECTED_CONTEXT`): residue 1 goes from 0.0 in
    the monomer to exactly 1.0 in context; nothing else moves.  Use this together
    with :func:`cv_ladder_pdb` to prove that the context structure is really being
    used -- computing circular variance on the monomer alone would report the
    interface residue as protruding.
    """
    atoms = cv_ladder_atoms("A")
    atoms.append(("CA", "GLY", "B", 1, CV_CONTEXT_PARTNER_X, 0.0, 0.0))
    path = tmp_path / "cv_context.pdb"
    path.write_text(build_pdb(atoms), encoding="utf-8")
    return {
        "path": path,
        "chain": "A",
        "context_chains": ["A", "B"],
        "expected_cv": dict(CV_LADDER_EXPECTED_CONTEXT),
        "changed_position": 1,
    }


@pytest.fixture()
def sasa_monomer_pdb(tmp_path: Path) -> Dict[str, object]:
    """Chain A with two isolated glycines; BOTH clip to RSA exactly 1.0."""
    path = tmp_path / "sasa_monomer.pdb"
    path.write_text(build_pdb(sasa_monomer_atoms()), encoding="utf-8")
    return {
        "path": path,
        "chain": "A",
        "expected_rsa": {
            SASA_BURIED_RESNUM: SASA_EXPECTED_ISOLATED_RSA,
            SASA_ISOLATED_RESNUM: SASA_EXPECTED_ISOLATED_RSA,
        },
    }


@pytest.fixture()
def sasa_context_pdb(tmp_path: Path) -> Dict[str, object]:
    """The same chain A, now with a 12-atom chain B shell around residue 1.

    Ground truth: residue 1 is fully enclosed, so its RSA drops to ~0
    (:data:`SASA_EXPECTED_BURIED_MAX_RSA`); residue 2, 100 A away, stays at
    exactly 1.0.  Pair with :func:`sasa_monomer_pdb`: the difference between the
    two is the whole reason freesasa is run on the trimer and not the monomer.
    """
    path = tmp_path / "sasa_context.pdb"
    path.write_text(build_pdb(sasa_context_atoms()), encoding="utf-8")
    return {
        "path": path,
        "chain": "A",
        "buried_resnum": SASA_BURIED_RESNUM,
        "isolated_resnum": SASA_ISOLATED_RESNUM,
        "expected_buried_max_rsa": SASA_EXPECTED_BURIED_MAX_RSA,
        "expected_isolated_rsa": SASA_EXPECTED_ISOLATED_RSA,
        "shell_radius": SASA_SHELL_RADIUS,
        "n_shell_atoms": 12,
    }


@pytest.fixture()
def query_numbered_pdb_factory(tmp_path: Path):
    """Build a CA-only structure numbered in QUERY coordinates (1..72).

    ``factory(covered=range(1, 51))`` yields an 50/72-coverage structure, which is
    how the 'positions with no coordinates fall back to pure trace' branch of
    ``combine_weight`` is reached without a real PDB.
    """
    counter = {"n": 0}

    def _factory(
        covered: Optional[Iterable[int]] = None,
        chains: Sequence[str] = ("A",),
        protein: str = QUERY_PROTEIN,
        name: Optional[str] = None,
    ) -> Path:
        counter["n"] += 1
        atoms: List[Tuple[str, str, str, int, float, float, float]] = []
        for offset, chain in enumerate(chains):
            for atom in full_length_query_pdb_atoms(chain, protein, covered=covered):
                name_, resname, chain_, resseq, x, y, z = atom
                atoms.append((name_, resname, chain_, resseq, x, y + offset * 12.0, z))
        path = tmp_path / (name or f"query_numbered_{counter['n']}.pdb")
        path.write_text(build_pdb(atoms), encoding="utf-8")
        return path

    return _factory


# --------------------------------------------------------------------------- #
# Panel fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def frequency_panels(tmp_path: Path) -> Dict[str, object]:
    """Parent and target panels with planted, exactly-known composition.

    100 records each, so a planted count is a percentage.  The returned dict
    carries every number a frequency test needs to assert without recomputing
    anything:

    ``expected_frequency_file_min_count_1`` / ``_2``
        exactly which mutants ``build_parent_frequency_file`` must emit, and at
        what frequency, for the two interesting ``--min-count`` settings;
    ``expected_dropped``
        the three mutants that must NOT appear, and why each is dropped;
    ``expected_median_depth``
        100.0 (71 columns at depth 100, one at 95);
    ``target_expected_entropy_bits`` / ``target_expected_mutant_fraction``
        closed-form observed-diversity values for the evaluation panel.
    """
    parent_fasta = write_fasta(
        tmp_path / "panel_parent.fasta", list(PARENT_PANEL_RECORDS), line_width=60
    )
    target_fasta = write_fasta(
        tmp_path / "panel_target.fasta", list(TARGET_PANEL_RECORDS), line_width=60
    )
    return {
        "parent_fasta": parent_fasta,
        "target_fasta": target_fasta,
        "child_protein": QUERY_PROTEIN,
        "parent_protein": PARENT_PROTEIN,
        "n_records": PANEL_N_RECORDS,
        "parent_spec": PARENT_PANEL_SPEC,
        "target_spec": TARGET_PANEL_SPEC,
        "parent_counts": PARENT_PANEL_TRUTH["counts"],
        "parent_depths": PARENT_PANEL_DEPTHS,
        "parent_frequencies": PARENT_PANEL_TRUTH["frequencies"],
        "target_counts": TARGET_PANEL_TRUTH["counts"],
        "target_depths": TARGET_PANEL_DEPTHS,
        "target_frequencies": TARGET_PANEL_TRUTH["frequencies"],
        "expected_median_depth": PARENT_PANEL_MEDIAN_DEPTH,
        "expected_frequency_file_min_count_1": EXPECTED_FREQUENCY_FILE_MIN_COUNT_1,
        "expected_frequency_file_min_count_2": EXPECTED_FREQUENCY_FILE_MIN_COUNT_2,
        "expected_dropped": EXPECTED_FREQUENCY_FILE_DROPPED,
        "expected_n_reverted_ancestral": EXPECTED_N_REVERTED_ANCESTRAL_MUTANTS,
        "expected_n_parent_reversions": EXPECTED_N_PARENT_REVERSION_MUTANTS,
        "reversion_position": 40,
        "reversion_mutant": "T40I",
        "target_expected_entropy_bits": TARGET_PANEL_EXPECTED_ENTROPY_BITS,
        "target_expected_mutant_fraction": TARGET_PANEL_EXPECTED_MUTANT_FRACTION,
    }


@pytest.fixture()
def leakage_panels(tmp_path: Path) -> Dict[str, object]:
    """Target panel, parent panel and deep set with ONE planted duplicate each.

    Layout reproduces the real frame mismatch: the deep (pre-cutoff) set carries
    the 16-aa signal peptide and the panels start at the mature N-terminus.  A
    genuine duplicate is therefore NOT byte-identical to its deep-set twin, which
    is why hashing alone can never find it.

    Planted, with row indices as ground truth:

    * ``parent`` row 7 is a byte-identical copy of ``target`` row 3 -- one shared
      exact sequence, and zero shared accessions (the headers differ);
    * ``deep`` row 5 is ``target`` row 0 with the signal peptide prepended -- a
      real leak that only an alignment-based check can see;
    * ``deep`` row 0 is the query itself with the signal peptide, i.e. the row
      that must NEVER be purged (it is GEMME's epistatic reference).

    ``deep_clean`` is the same set with row 5 replaced by a divergent sequence,
    so a test can assert that a clean input yields zero removals.
    """
    target = [(f"TGT{i:04d}", seq) for i, seq in enumerate(
        [LEAKAGE_MATURE_QUERY]
        + [_divergent(LEAKAGE_MATURE_QUERY, 2 + i, i) for i in range(1, 12)]
    )]
    parent = [(f"PAR{i:04d}", seq) for i, seq in enumerate(
        [_divergent(LEAKAGE_MATURE_QUERY, 3 + i, 100 + i) for i in range(12)]
    )]
    parent[LEAKAGE_PLANTED_DUPLICATE_PARENT_ROW] = (
        f"PAR{LEAKAGE_PLANTED_DUPLICATE_PARENT_ROW:04d}",
        target[LEAKAGE_PLANTED_DUPLICATE_TARGET_ROW][1],
    )

    deep = [(f"DEEP{i:04d}", seq) for i, seq in enumerate(
        [LEAKAGE_SIGNAL_PEPTIDE + QUERY_PROTEIN[len(LEAKAGE_SIGNAL_PEPTIDE):]]
        + [LEAKAGE_SIGNAL_PEPTIDE + _divergent(LEAKAGE_MATURE_QUERY, 18 + i, 200 + i)
           for i in range(1, 12)]
    )]
    deep_clean = list(deep)
    deep[LEAKAGE_PLANTED_DEEP_ROW] = (
        f"DEEP{LEAKAGE_PLANTED_DEEP_ROW:04d}",
        LEAKAGE_SIGNAL_PEPTIDE + target[0][1],
    )

    return {
        "target_fasta": write_fasta(tmp_path / "leak_target.fasta", target),
        "parent_fasta": write_fasta(tmp_path / "leak_parent.fasta", parent),
        "deep_fasta": write_fasta(tmp_path / "leak_deep.fasta", deep),
        "deep_clean_fasta": write_fasta(tmp_path / "leak_deep_clean.fasta", deep_clean),
        "query_fasta": write_fasta(
            tmp_path / "leak_query.fasta", [(QUERY_HEADER, LEAKAGE_MATURE_QUERY)]
        ),
        "target_records": target,
        "parent_records": parent,
        "deep_records": deep,
        "signal_peptide": LEAKAGE_SIGNAL_PEPTIDE,
        "planted_parent_row": LEAKAGE_PLANTED_DUPLICATE_PARENT_ROW,
        "planted_target_row": LEAKAGE_PLANTED_DUPLICATE_TARGET_ROW,
        "planted_deep_row": LEAKAGE_PLANTED_DEEP_ROW,
        "planted_parent_accession": f"PAR{LEAKAGE_PLANTED_DUPLICATE_PARENT_ROW:04d}",
        "planted_deep_accession": f"DEEP{LEAKAGE_PLANTED_DEEP_ROW:04d}",
        "expected_shared_exact_sequences": 1,
        "expected_shared_accessions": 0,
        "expected_deep_removals": 1,
        "query_row_index": 0,
    }


@pytest.fixture()
def panel_factory(tmp_path: Path):
    """Build an arbitrary panel from a ``{position: {residue: count}}`` spec.

    ``factory(spec, protein=..., n_records=100)`` returns
    ``(path, ground_truth)`` where ``ground_truth`` holds ``counts``, ``depths``
    and ``frequencies`` -- all derived from the literal spec, never from the code
    under test.
    """
    counter = {"n": 0}

    def _factory(
        spec: Mapping[int, Mapping[str, int]],
        protein: str = QUERY_PROTEIN,
        n_records: int = PANEL_N_RECORDS,
        prefix: str = "SYN",
        name: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, object]]:
        counter["n"] += 1
        records = _panel_records(protein, spec, n_records, prefix)
        path = write_fasta(
            tmp_path / (name or f"panel_{counter['n']}.fasta"), records, line_width=60
        )
        return path, _panel_ground_truth(protein, records)

    return _factory


# --------------------------------------------------------------------------- #
# Pipeline artefact fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def fake_jet_res(tmp_path: Path) -> Dict[str, object]:
    """A valid 72-row ``.res`` with 7 zero-trace positions (9.72%, in the warn band)."""
    path = write_jet_res(tmp_path / f"{QUERY_HEADER}_jet.res")
    return {
        "path": path,
        "columns": list(JET_COLUMNS),
        "n_rows": QUERY_LENGTH,
        "zero_trace_positions": list(JET_ZERO_TRACE_POSITIONS),
        "n_zero_trace": len(JET_ZERO_TRACE_POSITIONS),
        "expected_frac_zero": JET_EXPECTED_FRAC_ZERO,
        "warn_fraction": EXPECTED_WARN_ZERO_TRACE_FRACTION,
        "max_fraction": EXPECTED_MAX_ZERO_TRACE_FRACTION,
    }


@pytest.fixture()
def jet_res_factory(tmp_path: Path):
    """``factory(n_zero=30)`` -> ``(path, {'n_zero':.., 'frac_zero':..})``.

    Use it to straddle the refusal ceiling: ``n_zero=7`` warns and writes,
    ``n_zero=30`` (41.7%) must be refused with ``ZeroTraceError``.
    """
    counter = {"n": 0}

    def _factory(
        n_zero: int = len(JET_ZERO_TRACE_POSITIONS),
        protein: str = QUERY_PROTEIN,
        name: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, object]]:
        counter["n"] += 1
        positions = tuple(range(1, n_zero + 1))
        path = write_jet_res(
            tmp_path / (name or f"jet_{counter['n']}.res"), protein, positions
        )
        return path, {
            "n_zero": n_zero,
            "frac_zero": n_zero / len(protein),
            "zero_positions": positions,
        }

    return _factory


@pytest.fixture()
def fake_escott_matrix(tmp_path: Path) -> Dict[str, object]:
    """An ESCOTT ``normPred`` file in R's exact ``write.table`` format.

    Ground truth returned alongside it:

    ``values``
        ``{(AA, position): value or None}`` -- the matrix that was written, so a
        parser test is a genuine round trip;
    ``flat_positions``
        columns written as identically zero (the zero-trace case).  After the
        wild-type fill every entry in such a column is equal, so its softmax is
        exactly ``1/20`` for all 20 residues, at any temperature;
    ``wt_sequence``
        what a parser must recover from the ``"<WT><pos>"`` column labels.
    """
    path = write_escott_normpred(tmp_path / f"{QUERY_HEADER}_normPred_evolCombi.txt")
    return {
        "path": path,
        "protein": QUERY_PROTEIN,
        "wt_sequence": QUERY_PROTEIN,
        "n_positions": QUERY_LENGTH,
        "row_order": list(ESCOTT_ROW_ORDER),
        "plm_row_order": list(PLM_CACHE_ROW_ORDER),
        "values": escott_matrix_values(),
        "flat_positions": list(ESCOTT_FLAT_POSITIONS),
        "expected_flat_probability": ESCOTT_EXPECTED_FLAT_PROBABILITY,
    }


@pytest.fixture()
def escott_matrix_factory(tmp_path: Path):
    """``factory(protein=..., flat_positions=..., values=...)`` -> ``(path, values)``.

    Pass ``values`` to write a deliberately malformed matrix (two NAs in a column,
    an NA off the wild-type row, a non-contiguous position) and check the parser
    rejects it rather than misreading it.
    """
    counter = {"n": 0}

    def _factory(
        protein: str = QUERY_PROTEIN,
        flat_positions: Sequence[int] = ESCOTT_FLAT_POSITIONS,
        values: Optional[Mapping[Tuple[str, int], Optional[float]]] = None,
        name: Optional[str] = None,
    ) -> Tuple[Path, Dict[Tuple[str, int], Optional[float]]]:
        counter["n"] += 1
        cells = dict(values) if values is not None else escott_matrix_values(protein, flat_positions)
        path = write_escott_normpred(
            tmp_path / (name or f"escott_{counter['n']}_normPred_evolCombi.txt"),
            protein,
            flat_positions,
            cells,
        )
        return path, cells

    return _factory


@pytest.fixture()
def frequency_file_factory(tmp_path: Path):
    """``factory({'I10K': 0.1, ...})`` -> path to a PRESCOTT custom frequency file.

    Defaults to :data:`EXPECTED_FREQUENCY_FILE_MIN_COUNT_1`.  The extension is
    ``.txt``, never ``.csv``: prescott.py:982 switches parsers on that suffix.
    """
    counter = {"n": 0}

    def _factory(
        frequencies: Optional[Mapping[str, float]] = None,
        name: Optional[str] = None,
        with_meta: bool = False,
        parent_label: str = "J.2.4",
    ) -> Path:
        counter["n"] += 1
        values = dict(frequencies) if frequencies is not None else dict(
            EXPECTED_FREQUENCY_FILE_MIN_COUNT_1
        )
        path = write_frequency_file(
            tmp_path / (name or f"frequency_{counter['n']}.txt"), values
        )
        if with_meta:
            write_frequency_meta(
                path.with_name(path.stem + "_meta.tsv"), values, parent_label
            )
        return path

    return _factory


@pytest.fixture()
def score_matrix_factory(tmp_path: Path):
    """Write a ``plm_probability_profile``-format score matrix.

    Defaults to a uniform 1/20 matrix, which is what a fully zero-trace protein
    would produce -- i.e. the pathological case, on purpose.
    """
    counter = {"n": 0}

    def _factory(
        protein: str = QUERY_PROTEIN,
        probabilities: Optional[Mapping[str, Sequence[float]]] = None,
        name: Optional[str] = None,
    ) -> Path:
        counter["n"] += 1
        values = probabilities or uniform_score_matrix_probabilities(len(protein))
        return write_score_matrix(
            tmp_path / (name or f"score_matrix_{counter['n']}.csv"), protein, values
        )

    return _factory


# --------------------------------------------------------------------------- #
# Directory / guide fixtures.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def output_dir_factory(tmp_path: Path):
    """``factory('run1')`` -> a fresh output tree with the driver's subdirectories.

    Creates ``tables/``, ``tables/diagnostics/``, ``figures/``, ``scores/`` and
    ``inputs/`` and returns the root.  ``tables/diagnostics`` matters: it is where
    ``--diagnostics-dir`` and the jet-surrogate validation table land, and a test
    that only creates the root will not notice the driver writing them elsewhere.
    """
    counter = {"n": 0}

    def _factory(name: Optional[str] = None, subdirs: Sequence[str] = (
        "tables", "tables/diagnostics", "figures", "scores", "inputs",
    )) -> Path:
        counter["n"] += 1
        root = tmp_path / (name or f"output_{counter['n']}")
        for sub in subdirs:
            (root / sub).mkdir(parents=True, exist_ok=True)
        return root

    return _factory


@pytest.fixture()
def guide_factory(tmp_path: Path):
    """Build a MONTHLY_GUIDE CSV plus the panel and reference files it names.

    ``factory(['K', 'J.2.4'])`` writes a diversity panel and a nucleotide CDS for
    each label and returns ``{'path':.., 'rows':.., 'panels':.., 'references':..}``.
    Labels are written to the ``month`` column, which is what
    ``read_guide_rows`` reads.
    """
    counter = {"n": 0}

    def _factory(
        labels: Sequence[str] = LINEAGE_ORDER,
        n_records: int = 12,
        name: Optional[str] = None,
    ) -> Dict[str, object]:
        counter["n"] += 1
        root = tmp_path / f"guide_{counter['n']}"
        root.mkdir(parents=True, exist_ok=True)
        rows: List[Tuple[str, Path, Path]] = []
        panels: Dict[str, Path] = {}
        references: Dict[str, Path] = {}
        for index, label in enumerate(labels):
            safe = label.replace("/", "-").replace(" ", "_")
            panel = write_fasta(
                root / f"panel_{safe}.fasta",
                [(f"{safe}_{i:03d}", _divergent(QUERY_PROTEIN, index + (i % 3), index * 5 + i))
                 for i in range(n_records)],
                line_width=60,
            )
            reference = write_fasta(
                root / f"{safe}.nt.fa",
                [(f"EPI{index:07d}|HA|A/Synthetic/{index}/2025|EPI_ISL_{index:07d}|{label}",
                  QUERY_CDS)],
            )
            panels[label] = panel
            references[label] = reference
            rows.append((label, panel, reference))
        path = write_guide_csv(root / (name or "guide.csv"), rows)
        return {
            "path": path,
            "root": root,
            "labels": list(labels),
            "rows": rows,
            "panels": panels,
            "references": references,
        }

    return _factory


@pytest.fixture()
def five_lineage_guide(guide_factory) -> Dict[str, object]:
    """The production topology in miniature: G.1, J_int, J.2_int, J.2.4, K.

    All five labels are present, which is what ``common.resolve_parent_map``
    validates against -- a parent map naming a lineage with no guide row must be
    rejected eagerly, because a bad edge does not fail later, it just silently
    builds the frequency prior from the wrong panel.
    """
    guide = guide_factory(LINEAGE_ORDER)
    guide["expected_parent_map"] = dict(EXPECTED_PARENT_MAP)
    guide["expected_sensitivity_parent_map"] = dict(EXPECTED_SENSITIVITY_PARENT_MAP)
    guide["input_only"] = set(EXPECTED_INPUT_ONLY_LINEAGES)
    guide["evaluable"] = [l for l in LINEAGE_ORDER if l not in EXPECTED_INPUT_ONLY_LINEAGES]
    return guide


@pytest.fixture()
def prepared_inputs_tree(tmp_path: Path, five_lineage_guide) -> Dict[str, object]:
    """A complete stage-A/B ``inputs/`` tree, without running stage A or B.

    Layout is exactly what ``run_escott.resolve_lineage_inputs`` looks for::

        inputs/query/<key>_query.fasta
        inputs/msa/msa_<key>.fasta
        inputs/jet/<key>_surrogate_jet.res
        inputs/frequency/<key>_parent_frequency.txt        (primary edge)
        inputs/frequency/K_parentJ2int_frequency.txt       (sensitivity edge)
        inputs/structure/*.pdb
        inputs/inputs_manifest.json

    The manifest carries ``parent_map`` (the corrected ladder), ``frequency_index``
    (with ``is_primary_parent`` set per edge, which is how the alternate parent is
    discovered -- the filename alone cannot be inverted back to a label),
    ``queries``, ``structures``, ``lineage_msas`` and ``drop_parent_reversions``.
    """
    inputs = tmp_path / "inputs"
    for sub in ("query", "msa", "jet", "frequency", "structure", "leakage"):
        (inputs / sub).mkdir(parents=True, exist_ok=True)

    queries: Dict[str, object] = {}
    lineage_msas: Dict[str, object] = {}
    for label in LINEAGE_ORDER:
        key = label
        token = EXPECTED_PROT_TOKENS[label]
        query_path = write_fasta(inputs / "query" / f"{key}_query.fasta",
                                 [(token, QUERY_PROTEIN)])
        msa_path = write_fasta(
            inputs / "msa" / f"msa_{key}.fasta",
            [(token, QUERY_PROTEIN)] + list(zip(TINY_MSA_HEADERS[1:], TINY_MSA_ROWS[1:])),
        )
        write_jet_res(inputs / "jet" / f"{key}_surrogate_jet.res")
        queries[label] = {
            "lineage_key": key,
            "tag": EXPECTED_LINEAGE_TAGS[label],
            "header": token,
            "prot_token": token,
            "path": str(query_path),
            "length": QUERY_LENGTH,
            "md5": md5_text(QUERY_PROTEIN),
            "nucleotide_length": len(QUERY_CDS),
            "reference_path": str(five_lineage_guide["references"][label]),
            "reference_header": f"EPI0000000|HA|A/Synthetic/0/2025|EPI_ISL_0000000|{label}",
        }
        lineage_msas[label] = {
            "path": str(msa_path),
            "anchor_lineage": "G.1",
            "n_rows": MSA_N_ROWS,
            "n_columns": QUERY_LENGTH,
            "md5": md5_file(msa_path),
            "n_query_residues_changed_vs_anchor": 0,
        }

    frequency: Dict[str, object] = {}
    frequency_index: Dict[str, object] = {}
    for child, parent in EXPECTED_PARENT_MAP.items():
        freq_path = write_frequency_file(
            inputs / "frequency" / f"{child}_parent_frequency.txt",
            EXPECTED_FREQUENCY_FILE_MIN_COUNT_1,
        )
        meta_path = write_frequency_meta(
            inputs / "frequency" / f"{child}_parent_frequency_meta.tsv",
            EXPECTED_FREQUENCY_FILE_MIN_COUNT_1,
            parent,
        )
        entry = {
            "child_lineage": child,
            "lineage_key": child,
            "parent_lineage": parent,
            "parent_token": parent.replace(".", "").replace("_", ""),
            "is_primary_parent": True,
            "frequency_path": str(freq_path),
            "meta_path": str(meta_path),
            "median_mapped_depth": PARENT_PANEL_MEDIAN_DEPTH,
            "n_mutants": len(EXPECTED_FREQUENCY_FILE_MIN_COUNT_1),
            "drop_parent_reversions": True,
        }
        frequency[child] = dict(entry, freq_max=0.95, min_count=1, min_depth=50)
        frequency_index[child] = entry

    # The single contested edge, prepared as the labelled sensitivity alternative.
    child, _default_parent, alt_parent = CONTESTED_EDGE
    alt_token = alt_parent.replace(".", "").replace("_", "")
    alt_stem = f"{child}_parent{alt_token}_frequency"
    alt_path = write_frequency_file(
        inputs / "frequency" / f"{alt_stem}.txt", EXPECTED_FREQUENCY_FILE_MIN_COUNT_2
    )
    alt_meta = write_frequency_meta(
        inputs / "frequency" / f"{alt_stem}_meta.tsv",
        EXPECTED_FREQUENCY_FILE_MIN_COUNT_2,
        alt_parent,
    )
    frequency_index[alt_stem] = {
        "child_lineage": child,
        "lineage_key": child,
        "parent_lineage": alt_parent,
        "parent_token": alt_token,
        "is_primary_parent": False,
        "frequency_path": str(alt_path),
        "meta_path": str(alt_meta),
        "median_mapped_depth": 27452.0,
        "n_mutants": len(EXPECTED_FREQUENCY_FILE_MIN_COUNT_2),
        "drop_parent_reversions": True,
    }

    monomer = tmp_path / "inputs" / "structure" / "model_chainA_qnum.pdb"
    monomer.write_text(build_pdb(full_length_query_pdb_atoms("A")), encoding="utf-8")
    trimer_atoms: List[Tuple[str, str, str, int, float, float, float]] = []
    for offset, chain in enumerate("ABC"):
        for name, resname, chain_, resseq, x, y, z in full_length_query_pdb_atoms(chain):
            trimer_atoms.append((name, resname, chain_, resseq, x, y + offset * 12.0, z))
    trimer = tmp_path / "inputs" / "structure" / "model_trimer_qnum.pdb"
    trimer.write_text(build_pdb(trimer_atoms), encoding="utf-8")

    structure_record = {
        "chain": "A",
        "offset": 0,
        "query_length": QUERY_LENGTH,
        "n_covered": QUERY_LENGTH,
        "coverage_fraction": 1.0,
        "covered_positions": list(range(1, QUERY_LENGTH + 1)),
        "uncovered_runs": [],
        "construct_truncated_at_author_resnum": None,
        "source_path": str(monomer),
        "source_md5": md5_file(monomer),
        "monomer": {"path": str(monomer), "chains": ["A"],
                    "per_chain": {"A": {"n_residues": QUERY_LENGTH, "resnum_min": 1,
                                        "resnum_max": QUERY_LENGTH, "gaps": []}}},
        "trimer": {"path": str(trimer), "chains": ["A", "B", "C"],
                   "per_chain": {c: {"n_residues": QUERY_LENGTH, "resnum_min": 1,
                                     "resnum_max": QUERY_LENGTH, "gaps": []}
                                 for c in "ABC"}},
    }

    manifest = {
        "anchor_lineage": "G.1",
        "generated_by": "tests_prescott_iav.conftest.prepared_inputs_tree",
        "generated_at": "2026-08-05T00:00:00",
        "python": str(PRESCOTT_PYTHON),
        "seed": 20260805,
        "guide_path": str(five_lineage_guide["path"]),
        "guide_md5": md5_file(five_lineage_guide["path"]),
        "parent_map": dict(EXPECTED_PARENT_MAP),
        "parent_map_preset": "clade_evidence",
        "sensitivity_parent_map": {child: alt_parent},
        "sensitivity_preset": "brief_as_stated",
        "input_only_lineages": sorted(EXPECTED_INPUT_ONLY_LINEAGES),
        "drop_parent_reversions": True,
        "n_parent_reversion_mutants_dropped": {c: EXPECTED_N_PARENT_REVERSION_MUTANTS
                                               for c in EXPECTED_PARENT_MAP},
        "parent_freq_max": 0.95,
        "queries": queries,
        "lineage_msas": lineage_msas,
        "frequency": frequency,
        "frequency_index": frequency_index,
        "structures": {"primary": structure_record, "extra": structure_record},
        "msa": {
            "n_rows": MSA_N_ROWS,
            "n_columns": QUERY_LENGTH,
            "anchor_lineage": "G.1",
            "anchor_header": EXPECTED_PROT_TOKENS["G.1"],
        },
        "args": {},
    }
    manifest_path = inputs / "inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    return {
        "inputs_dir": inputs,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "guide": five_lineage_guide,
        "monomer_pdb": monomer,
        "trimer_pdb": trimer,
        "lineages": list(LINEAGE_ORDER),
        "evaluable": [l for l in LINEAGE_ORDER if l not in EXPECTED_INPUT_ONLY_LINEAGES],
        "parent_map": dict(EXPECTED_PARENT_MAP),
        "sensitivity_edge": {child: alt_parent},
        "alternate_frequency_path": alt_path,
        "prot_tokens": dict(EXPECTED_PROT_TOKENS),
    }


# --------------------------------------------------------------------------- #
# Convenience assertion helpers.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def expected_constants() -> Dict[str, object]:
    """Literal copies of the pinned constants, for testing the module against.

    Deliberately NOT imported from ``prescott_iav.constants``.  A test that reads
    the parent map out of the module and compares it with itself would pass
    happily if the module regressed to the old, wrong ``K <- J.2_int`` edge.
    """
    return {
        "parent_map": dict(EXPECTED_PARENT_MAP),
        "sensitivity_parent_map": dict(EXPECTED_SENSITIVITY_PARENT_MAP),
        "default_preset": "clade_evidence",
        "sensitivity_preset": "brief_as_stated",
        "contested_edge": CONTESTED_EDGE,
        "input_only_lineages": frozenset(EXPECTED_INPUT_ONLY_LINEAGES),
        "lineage_tags": dict(EXPECTED_LINEAGE_TAGS),
        "trace_top_fraction": EXPECTED_TRACE_TOP_FRACTION,
        "max_zero_trace_fraction": EXPECTED_MAX_ZERO_TRACE_FRACTION,
        "warn_zero_trace_fraction": EXPECTED_WARN_ZERO_TRACE_FRACTION,
        "escott_row_order": tuple(ESCOTT_ROW_ORDER),
        "plm_cache_row_order": tuple(PLM_CACHE_ROW_ORDER),
        "no_frequency_sentinel": 999.0,
        "supported_prescott_equations": (1, 2, 3, 5),
    }
