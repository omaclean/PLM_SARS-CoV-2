#!/usr/bin/env python3
r"""Tests for ``scripts/prescott_iav/prepare_inputs.py`` (stage A).

Stage A builds every input ESCOTT/PRESCOTT will ever see, and each of its four
products fails silently rather than loudly when it is wrong:

* a **structure** that still carries the stabilising construct (GGGGT linker,
  foldon trimerisation domain, His6 tag) hands the JET2 surrogate 50 extra
  "residues" that are not HA at all, and they read as conserved;
* a **query** translated even slightly differently from the evaluation half
  shifts the whole reference->column map by one and every correlation with it;
* an **MSA** whose first row is not the ungapped query breaks GEMME's frame
  (computePred.R indexes predictions by row 1's columns) with no error;
* a **frequency file** that emits an unobserved mutant as ``0.0`` becomes
  ``log10(0) = -inf`` at prescott.py:1113 and drives every such variant to
  maximal deleteriousness.

So the tests are organised around those four failure modes, not around the
module's public surface, and every expected value is a literal or a closed form
(``19/95``, ``log10(1/100) = -2``, residues ``1..60``) rather than something the
code under test computed.

Run with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_prepare_inputs.py -q

Markers: everything is offline and sub-second except the ``requires_mafft``
alignment tests (real mafft, ~1 s) and the ``requires_real_data`` 6WXB test.
The rest of the MSA coverage uses a *fake* mafft written into ``tmp_path``, which
is what lets the error branches (ragged output, dropped row, corrupted anchor,
non-zero exit) be reached at all -- real mafft never produces them.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prescott_iav import common  # noqa: E402
from prescott_iav import prepare_inputs as pi  # noqa: E402

from tests_prescott_iav.conftest import (  # noqa: E402
    AA20,
    LINEAGE_ORDER,
    ONE_CODON_PER_AA,
    PANEL_N_RECORDS,
    PARENT_PROTEIN,
    QUERY_CDS,
    QUERY_LENGTH,
    QUERY_PROTEIN,
    STOP_CODON,
    THREE_LETTER,
    _divergent,
    _panel_records,
    build_pdb,
    full_length_query_pdb_atoms,
    write_fasta,
    write_guide_csv,
)


# --------------------------------------------------------------------------- #
# Literals.  Never imported from the module under test: a test that reads the
# motif list out of prepare_inputs.CONSTRUCT_MOTIFS and then asserts the motif
# was stripped is comparing the module with itself.
# --------------------------------------------------------------------------- #

LINKER = "GGGGTGGGGT"
FOLDON = "GRMKQIEDKIEEILSKIYHIENEIARIKKLIGER"
HIS6 = "HHHHHH"

CORE = QUERY_PROTEIN[:60]
"""The 'real' part of the synthetic construct: query residues 1..60."""

CONSTRUCT_CHAIN = CORE + LINKER + FOLDON + HIS6
CONSTRUCT_FIRST_ARTEFACT_RESNUM = len(CORE) + 1          # 61, the linker's first G
CONSTRUCT_FOLDON_RESNUM = len(CORE) + len(LINKER) + 1    # 71
CONSTRUCT_HIS_RESNUM = CONSTRUCT_FOLDON_RESNUM + len(FOLDON)   # 105
CONSTRUCT_TOTAL = len(CONSTRUCT_CHAIN)                   # 110

# Sanity: the construct motifs must not occur inside the core, or the fixture
# would be testing the wrong thing.
assert LINKER not in CORE and FOLDON not in CORE and HIS6 not in CORE
assert CONSTRUCT_CHAIN.find(LINKER) == CONSTRUCT_FIRST_ARTEFACT_RESNUM - 1
assert CONSTRUCT_CHAIN.find(FOLDON) == CONSTRUCT_FOLDON_RESNUM - 1
assert CONSTRUCT_CHAIN.find(HIS6) == CONSTRUCT_HIS_RESNUM - 1
assert CONSTRUCT_TOTAL == 110

EXPECTED_CONSTRUCT_MOTIFS = {
    "linker": LINKER,
    "foldon": FOLDON,
    "his_tag": HIS6,
}

# 6WXB, measured on the real files (see TestRealStructure).  Author numbering is
# the mature-HA1 convention, so the shift into the 566-aa HA0 query frame is +16.
REAL_6WXB_OFFSET = 16
REAL_6WXB_IDENTITY = 0.845360824742268
REAL_6WXB_MATCHED = 410
REAL_6WXB_RESNUM_MIN = 25
REAL_6WXB_RESNUM_MAX = 517
REAL_6WXB_N_COVERED = 485
REAL_6WXB_GAPS = [[341, 350]]
REAL_6WXB_UNCOVERED = [[1, 24], [342, 349], [518, 566]]
REAL_6WXB_QUERY_LENGTH = 566


# --------------------------------------------------------------------------- #
# Independent readers/writers.
#
# The PDB reader below is deliberately hand-rolled off the column spec rather
# than prody: prody wrote the file, so parsing it back with prody could only
# prove prody round-trips.
# --------------------------------------------------------------------------- #

def read_pdb_atoms(path: Path):
    """[(record, atom_name, resname, chain, resnum, icode)] from PDB columns."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        out.append((
            line[0:6].strip(),
            line[12:16].strip(),
            line[17:20].strip(),
            line[21:22].strip(),
            int(line[22:26]),
            line[26:27].strip(),
        ))
    return out


def read_pdb_ca_sequence(path: Path, chain: str = "A") -> str:
    inverse = {three: one for one, three in THREE_LETTER.items()}
    return "".join(
        inverse[resname]
        for record, name, resname, chid, _num, _icode in read_pdb_atoms(path)
        if record == "ATOM" and name == "CA" and chid == chain
    )


def read_pdb_ca_resnums(path: Path, chain: str = "A"):
    return [
        num for record, name, _resname, chid, num, _icode in read_pdb_atoms(path)
        if record == "ATOM" and name == "CA" and chid == chain
    ]


CIF_FIELDS = (
    "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id "
    "label_asym_id label_entity_id label_seq_id pdbx_PDB_ins_code "
    "Cartn_x Cartn_y Cartn_z occupancy B_iso_or_equiv "
    "auth_seq_id auth_asym_id pdbx_PDB_model_num"
).split()


def write_min_cif(path: Path, rows) -> Path:
    """Minimal single-``_atom_site``-loop mmCIF.

    ``rows`` = [(group_PDB, atom_name, resname, chain, resnum, icode, x, y, z)].
    Enough for ``prody.parseMMCIF``, and small enough to read in a failure
    message -- which the 2.4 MB real 6WXB assembly is not.
    """
    lines = ["data_TEST", "#", "loop_"] + [f"_atom_site.{field}" for field in CIF_FIELDS]
    for serial, (group, name, resname, chain, resnum, icode, x, y, z) in enumerate(rows, start=1):
        lines.append(
            f"{group} {serial} C {name} . {resname} {chain} 1 {resnum} {icode or '?'} "
            f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00 {resnum} {chain} 1"
        )
    lines.append("#")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ca_rows(sequence: str, chain: str = "A", start: int = 1, spacing: float = 3.8, y: float = 0.0):
    """CA-only rows for ``write_min_cif``, numbered ``start..start+len-1``."""
    return [
        ("ATOM", "CA", THREE_LETTER[aa], chain, start + i, "?", round(i * spacing, 3), y, 0.0)
        for i, aa in enumerate(sequence)
    ]


# --------------------------------------------------------------------------- #
# Fixtures local to this module (nothing here duplicates conftest).
# --------------------------------------------------------------------------- #

FAKE_MAFFT_PREAMBLE = '''
import sys


def read(path):
    records = []
    header = None
    chunks = []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:]
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


argv = sys.argv[1:]
if "--version" in argv:
    sys.stderr.write("v0.0 (fake)\\n")
    raise SystemExit(0)
recs = read(argv[-1])
if "--add" in argv:
    recs = recs + read(argv[argv.index("--add") + 1])
'''

FAKE_MAFFT_EMIT = '''
for _h, _s in recs:
    sys.stdout.write(">" + _h + "\\n" + _s + "\\n")
'''

FAKE_MAFFT_BODIES = {
    # Perfect aligner: the inputs are already equal-length and ungapped.
    "passthrough": "",
    # One insertion relative to the query: the query gets a gap column that
    # _strip_query_gap_columns must remove, every deep row an extra residue.
    "insertion": (
        "recs = [(h, s[:10] + ('-' if i == 0 else 'W') + s[10:])\n"
        "        for i, (h, s) in enumerate(recs)]\n"
    ),
    # Non-standard residues in the deep rows only (row 0 must never be touched).
    "nonstandard": (
        "recs = [(h, s if i == 0 else 'B' + s[1:3] + 'Z' + s[4:])\n"
        "        for i, (h, s) in enumerate(recs)]\n"
    ),
    "fail": "sys.stderr.write('fake mafft exploded\\n')\nraise SystemExit(3)\n",
    "ragged": "recs = [(recs[0][0], recs[0][1])] + [(h, s[:-3]) for h, s in recs[1:]]\n",
    "drop_row": "recs = recs[:-1]\n",
    "corrupt_anchor": "recs = [(recs[0][0], 'W' + recs[0][1][1:])] + list(recs[1:])\n",
}


@pytest.fixture()
def fake_mafft(tmp_path):
    """``fake_mafft('ragged')`` -> path to an executable stand-in for mafft.

    Real mafft cannot be made to emit ragged output, drop a row or corrupt the
    anchor, so the only way to reach ``build_deep_msa``'s four defensive raises
    is to substitute the aligner.  ``--mafft-bin`` exists for exactly this.
    """
    counter = {"n": 0}

    def _factory(mode: str = "passthrough") -> Path:
        counter["n"] += 1
        path = tmp_path / f"fake_mafft_{mode}_{counter['n']}.py"
        path.write_text(
            f"#!{sys.executable}\n"
            + FAKE_MAFFT_PREAMBLE
            + FAKE_MAFFT_BODIES[mode]
            + FAKE_MAFFT_EMIT,
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    return _factory


@pytest.fixture()
def anchor_info():
    """The ``queries[label]`` payload ``build_deep_msa`` actually consumes."""
    return {
        "header": "HAG1",
        "protein": QUERY_PROTEIN,
        "md5": common.md5_text(QUERY_PROTEIN),
        "length": QUERY_LENGTH,
    }


@pytest.fixture()
def deep_fasta(tmp_path):
    """Six 72-aa 'pre-cutoff' homologues, 2..7 substitutions from the query."""
    return write_fasta(
        tmp_path / "deep.fasta",
        [(f"DEEP{i:03d}", _divergent(QUERY_PROTEIN, 2 + i, i)) for i in range(6)],
    )


@pytest.fixture()
def freq_paths(tmp_path):
    """``(out_txt, out_meta)`` inside a directory that does not exist yet.

    ``build_parent_frequency_file`` is expected to create it (``ensure_dir``).
    """
    root = tmp_path / "frequency_out"
    return root / "K_parent_frequency.txt", root / "K_parent_frequency_meta.tsv"


def build_freq(panel_path, out_txt, out_meta, **kwargs):
    """Call ``build_parent_frequency_file`` with this suite's usual defaults."""
    params = {
        "child_label": "K",
        "parent_label": "J.2.4",
        "child_protein": QUERY_PROTEIN,
        "min_count": 1,
        "min_depth": 50,
        "freq_max": 0.95,
        "parent_protein": PARENT_PROTEIN,
        "drop_parent_reversions": True,
    }
    params.update(kwargs)
    return pi.build_parent_frequency_file(
        parent_panel_fasta=Path(panel_path), out_txt=Path(out_txt), out_meta=Path(out_meta),
        **params,
    )


def parse_frequency_file(path: Path):
    """``{mutant: float}`` parsed off the two-field PRESCOTT custom format."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        fields = line.split(" ")
        assert len(fields) == 2, f"frequency line must be '<MUTANT> <freq>': {line!r}"
        out[fields[0]] = float(fields[1])
    return out


def make_guide(root: Path, labels=LINEAGE_ORDER, panel_spec=None, n_records=PANEL_N_RECORDS,
               cds_by_label=None, panel_protein=None):
    """Guide CSV + panels + nucleotide references, all under ``root``.

    Distinct from ``conftest.guide_factory`` in the two ways ``main`` cares
    about: the panels are deep enough (100 records) to clear ``--parent-min-depth``,
    and the per-lineage CDS can be overridden so a single lineage can be given a
    frameshifted or prematurely-stopped reference.
    """
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = {}
    references = {}
    for index, label in enumerate(labels):
        safe = label.replace("/", "-")
        panel = write_fasta(
            root / f"panel_{safe}.fasta",
            _panel_records(panel_protein or QUERY_PROTEIN, panel_spec or {5: {"V": 20}},
                           n_records, safe[:3]),
            line_width=60,
        )
        cds = (cds_by_label or {}).get(label, QUERY_CDS)
        reference = write_fasta(
            root / f"{safe}.nt.fa",
            [(f"EPI{index:07d}|HA|A/Synthetic/{index}/2025|EPI_ISL_{index:07d}|{label}", cds)],
        )
        panels[label] = panel
        references[label] = reference
        rows.append((label, panel, reference))
    return {
        "path": write_guide_csv(root / "guide.csv", rows),
        "panels": panels,
        "references": references,
        "labels": list(labels),
    }


def guide_rows_for(guide, labels=None):
    """``read_guide_rows`` output restricted to ``labels`` (order preserved)."""
    rows = common.read_guide_rows(Path(guide["path"]))
    if labels is None:
        return rows
    wanted = list(labels)
    return [row for row in rows if row["label"] in wanted]


# =========================================================================== #
# CLI surface
# =========================================================================== #

@pytest.mark.unit
class TestParser:
    """The argparse surface is a contract: the driver and the slurm wrapper both
    build these argv lists by hand, so a renamed flag or a changed default is a
    silent behaviour change in three places at once."""

    def test_inputs_dir_is_required(self):
        with pytest.raises(SystemExit) as excinfo:
            pi.build_parser().parse_args([])
        assert excinfo.value.code == 2

    def test_defaults(self, tmp_path):
        args = pi.build_parser().parse_args(["--inputs-dir", str(tmp_path)])
        assert args.structure_chain == "A"
        assert args.structure_offset == "auto"
        assert args.structure_min_identity == 0.60
        assert args.msa_anchor == "G.1"
        assert args.mafft_mode == "auto"
        assert args.msa_nonstandard == "to-x"
        assert args.mafft_threads == 8
        assert args.parent_min_count == 1
        assert args.parent_min_depth == 50
        assert args.parent_freq_max == 0.95
        assert args.frequency_cutoff_mode == "depth_scaled"
        assert args.frequency_cutoff_k == "1"
        assert args.frequency_cutoff == -4.0
        assert args.seed == 20260805
        assert args.only_lineage is None
        assert args.force is False
        assert args.no_extra_structure is False
        assert args.leakage_hash_suffix_length == 500

    def test_drop_parent_reversions_defaults_on_and_can_be_switched_off(self, tmp_path):
        """It defaults ON because --parent-freq-max 0.95 provably does NOT catch
        the real reversions (K's N160S sits at 0.932, K176I at 0.897)."""
        parser = pi.build_parser()
        assert parser.parse_args(["--inputs-dir", str(tmp_path)]).drop_parent_reversions is True
        assert parser.parse_args(
            ["--inputs-dir", str(tmp_path), "--no-drop-parent-reversions"]
        ).drop_parent_reversions is False
        assert parser.parse_args(
            ["--inputs-dir", str(tmp_path), "--no-drop-parent-reversions",
             "--drop-parent-reversions"]
        ).drop_parent_reversions is True

    def test_leakage_flags_are_attached_and_default_on(self, tmp_path):
        args = pi.build_parser().parse_args(["--inputs-dir", str(tmp_path)])
        assert args.leakage_check is True
        assert args.purge_leakage is True
        off = pi.build_parser().parse_args(
            ["--inputs-dir", str(tmp_path), "--no-leakage-check", "--no-purge-leakage"]
        )
        assert off.leakage_check is False and off.purge_leakage is False

    def test_parent_map_preset_default_is_the_clade_evidence_ladder(self, tmp_path):
        args = pi.build_parser().parse_args(["--inputs-dir", str(tmp_path)])
        assert args.parent_map_preset == "clade_evidence"
        assert args.parent_map is None
        assert args.sensitivity_parent_map is None
        assert args.sensitivity_preset is None

    @pytest.mark.parametrize("flag,value", [
        ("--mafft-mode", "elsewhere"),
        ("--msa-nonstandard", "to-z"),
        ("--frequency-cutoff-mode", "guessing"),
        ("--parent-map-preset", "not_a_preset"),
    ])
    def test_choice_flags_reject_unknown_values(self, tmp_path, flag, value):
        with pytest.raises(SystemExit):
            pi.build_parser().parse_args(["--inputs-dir", str(tmp_path), flag, value])

    def test_only_lineage_is_repeatable(self, tmp_path):
        args = pi.build_parser().parse_args(
            ["--inputs-dir", str(tmp_path), "--only-lineage", "K", "--only-lineage", "J.2.4"]
        )
        assert args.only_lineage == ["K", "J.2.4"]

    def test_construct_motifs_are_the_three_we_expect(self):
        """Asserted against literals, not against the module's own dict."""
        assert pi.CONSTRUCT_MOTIFS == EXPECTED_CONSTRUCT_MOTIFS


@pytest.mark.cli
@pytest.mark.requires_prescott_python
class TestModuleCli:
    def test_help_exits_zero_and_documents_the_reversion_filter(self, run_module_cli):
        proc = run_module_cli("prepare_inputs", ["--help"])
        assert proc.returncode == 0, proc.stderr
        assert "--drop-parent-reversions" in proc.stdout
        assert "--no-drop-parent-reversions" in proc.stdout
        assert "--sensitivity-parent-map" in proc.stdout

    def test_missing_inputs_dir_exits_two(self, run_module_cli):
        proc = run_module_cli("prepare_inputs", [])
        assert proc.returncode == 2
        assert "--inputs-dir" in proc.stderr


# =========================================================================== #
# Structure: parsing, construct stripping, renumbering
# =========================================================================== #

@pytest.mark.unit
@pytest.mark.requires_prody
class TestLoadStructure:
    def test_parses_mmcif(self, tmp_path):
        path = write_min_cif(tmp_path / "mini.cif", ca_rows("MKT"))
        struct = pi.load_structure(path)
        assert struct.numAtoms() == 3
        assert list(struct.getResnums()) == [1, 2, 3]

    def test_parses_pdb(self, tmp_path):
        path = tmp_path / "mini.pdb"
        path.write_text(build_pdb(full_length_query_pdb_atoms("A")), encoding="utf-8")
        struct = pi.load_structure(path)
        assert struct.numAtoms() == QUERY_LENGTH

    def test_mmcif_and_pdb_of_the_same_chain_agree(self, tmp_path):
        """cif -> PDB conversion must not renumber or reorder anything by itself."""
        cif = write_min_cif(tmp_path / "same.cif", ca_rows(CORE))
        pdb = tmp_path / "same.pdb"
        pdb.write_text(build_pdb([
            ("CA", THREE_LETTER[aa], "A", i + 1, round(i * 3.8, 3), 0.0, 0.0)
            for i, aa in enumerate(CORE)
        ]), encoding="utf-8")
        from_cif = pi._chain_ca(pi.load_structure(cif), "A")
        from_pdb = pi._chain_ca(pi.load_structure(pdb), "A")
        assert from_cif.getSequence() == from_pdb.getSequence() == CORE
        assert list(from_cif.getResnums()) == list(from_pdb.getResnums())

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif"])
    def test_cif_suffixes_are_case_insensitive(self, tmp_path, suffix):
        path = write_min_cif(tmp_path / f"upper{suffix.upper()}", ca_rows("MKT"))
        assert pi.load_structure(path).numAtoms() == 3

    def test_unsupported_suffix_is_refused_by_name(self, tmp_path):
        path = tmp_path / "structure.xyz"
        path.write_text("nonsense\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported structure format"):
            pi.load_structure(path)

    def test_chain_ca_refuses_an_absent_chain(self, tmp_path):
        struct = pi.load_structure(write_min_cif(tmp_path / "a.cif", ca_rows("MKT", chain="A")))
        with pytest.raises(ValueError, match="No protein CA atoms in chain Z"):
            pi._chain_ca(struct, "Z")


@pytest.mark.unit
@pytest.mark.requires_prody
class TestConstructArtefactStripping:
    """6WXB is a stabilised ectodomain construct.  If the GGGGT linker, the
    foldon trimerisation domain or the His6 tag survived into the renumbered PDB
    they would be read as 50 extra, highly conserved HA2 residues -- so the test
    is on the *emitted residue range*, not on the truncation bookkeeping."""

    @pytest.fixture()
    def construct_cif(self, tmp_path):
        return write_min_cif(tmp_path / "construct.cif", ca_rows(CONSTRUCT_CHAIN))

    def test_truncation_point_is_the_first_artefact_residue(self, construct_cif):
        struct = pi.load_structure(construct_cif)
        assert pi._find_construct_truncation(struct, "A") == CONSTRUCT_FIRST_ARTEFACT_RESNUM

    @pytest.mark.parametrize("motif,expected_resnum", [
        (LINKER, len(CORE) + 1),
        (FOLDON, len(CORE) + 1),
        (HIS6, len(CORE) + 1),
    ])
    def test_each_motif_alone_is_enough_to_truncate(self, tmp_path, motif, expected_resnum):
        cif = write_min_cif(tmp_path / f"one_{len(motif)}.cif", ca_rows(CORE + motif))
        struct = pi.load_structure(cif)
        assert pi._find_construct_truncation(struct, "A") == expected_resnum

    def test_clean_chain_reports_no_truncation(self, tmp_path):
        cif = write_min_cif(tmp_path / "clean.cif", ca_rows(QUERY_PROTEIN))
        assert pi._find_construct_truncation(pi.load_structure(cif), "A") is None

    def test_emitted_residue_range_stops_at_the_last_real_residue(self, construct_cif, tmp_path):
        report = pi.prepare_structure(
            construct_cif, "A", "auto", QUERY_PROTEIN, tmp_path / "out", "construct", 0.60,
        )
        assert report["construct_truncated_at_author_resnum"] == CONSTRUCT_FIRST_ARTEFACT_RESNUM
        assert report["offset"] == 0

        mono = Path(report["monomer"]["path"])
        resnums = read_pdb_ca_resnums(mono)
        assert resnums == list(range(1, len(CORE) + 1))
        assert min(resnums) == 1 and max(resnums) == len(CORE) == 60

        per_chain = report["monomer"]["per_chain"]["A"]
        assert per_chain == {"n_residues": 60, "resnum_min": 1, "resnum_max": 60, "gaps": []}
        assert report["n_covered"] == 60
        assert report["coverage_fraction"] == 60 / QUERY_LENGTH
        assert report["uncovered_runs"] == [[61, QUERY_LENGTH]]

    def test_no_construct_motif_survives_into_the_pdb(self, construct_cif, tmp_path):
        report = pi.prepare_structure(
            construct_cif, "A", "auto", QUERY_PROTEIN, tmp_path / "out", "construct", 0.60,
        )
        for path in (Path(report["monomer"]["path"]), Path(report["trimer"]["path"])):
            emitted = read_pdb_ca_sequence(path)
            assert emitted == CORE
            for name, motif in EXPECTED_CONSTRUCT_MOTIFS.items():
                assert motif not in emitted, f"{name} survived into {path.name}"
            # His6 is the one that can hide: HA legitimately contains HH.
            assert emitted.count("H") == CORE.count("H")

    def test_trimer_strips_the_construct_from_every_chain(self, tmp_path):
        rows = []
        for offset, chain in enumerate("ABC"):
            rows.extend(ca_rows(CONSTRUCT_CHAIN, chain=chain, y=offset * 40.0))
        cif = write_min_cif(tmp_path / "trimer.cif", rows)
        report = pi.prepare_structure(
            cif, "A", "auto", QUERY_PROTEIN, tmp_path / "out", "trimer", 0.60,
        )
        assert report["trimer"]["chains"] == ["A", "B", "C"]
        for chain in "ABC":
            assert read_pdb_ca_sequence(Path(report["trimer"]["path"]), chain) == CORE
            assert report["trimer"]["per_chain"][chain]["resnum_max"] == 60

    def test_residues_numbered_past_the_query_are_dropped_even_without_a_motif(self, tmp_path):
        """The generic guard: an unrecognised tag numbered off the end of the
        real sequence still cannot reach the surrogate."""
        junk = "WWWWWWWW"
        assert not any(m in QUERY_PROTEIN + junk for m in EXPECTED_CONSTRUCT_MOTIFS.values())
        cif = write_min_cif(tmp_path / "tagged.cif", ca_rows(QUERY_PROTEIN + junk))
        report = pi.prepare_structure(
            cif, "A", "auto", QUERY_PROTEIN, tmp_path / "out", "tagged", 0.60,
        )
        assert report["construct_truncated_at_author_resnum"] is None
        assert read_pdb_ca_resnums(Path(report["monomer"]["path"])) == list(range(1, QUERY_LENGTH + 1))
        assert read_pdb_ca_sequence(Path(report["monomer"]["path"])) == QUERY_PROTEIN

    def test_glycans_and_other_heteroatoms_are_removed(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN)
        rows.append(("HETATM", "C1", "NAG", "A", 601, "?", 5.0, 5.0, 5.0))
        rows.append(("HETATM", "C1", "BMA", "A", 602, "?", 6.0, 6.0, 6.0))
        cif = write_min_cif(tmp_path / "glyco.cif", rows)
        report = pi.prepare_structure(
            cif, "A", "auto", QUERY_PROTEIN, tmp_path / "out", "glyco", 0.60,
        )
        emitted = read_pdb_atoms(Path(report["monomer"]["path"]))
        assert {r[2] for r in emitted}.isdisjoint({"NAG", "BMA"})
        assert all(record == "ATOM" for record, *_rest in emitted)

    def test_an_n_terminal_tag_is_refused_loudly_rather_than_stripped(self, tmp_path):
        """LIMITATION, pinned deliberately.

        ``_find_construct_truncation`` drops everything from the motif ONWARD, so
        a construct whose His6 is at the N-terminus (numbered 1..6, real protein
        from 7) makes ``resnum < 1`` select nothing.  The module then raises
        rather than silently emitting an empty structure, which is the right
        failure mode -- but it is a refusal, not a strip.  See the report.
        """
        cif = write_min_cif(tmp_path / "ntag.cif", ca_rows(HIS6 + CORE))
        struct = pi.load_structure(cif)
        assert pi._find_construct_truncation(struct, "A") == 1
        with pytest.raises(ValueError, match="Construct truncation removed the entire chain"):
            pi.write_renumbered_pdb(struct, ["A"], 0, QUERY_LENGTH, tmp_path / "ntag.pdb",
                                    truncate_at=1)


@pytest.mark.unit
@pytest.mark.requires_prody
class TestResolveStructureOffset:
    def test_finds_a_positive_shift_and_reports_exact_counts(self, tmp_path):
        # Author numbering 17..88 for query residues 1..72 -> offset -16.
        cif = write_min_cif(tmp_path / "shift.cif", ca_rows(QUERY_PROTEIN, start=17))
        info = pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60)
        assert info == {
            "offset": -16,
            "identity": 1.0,
            "matched_residues": QUERY_LENGTH,
            "compared_residues": QUERY_LENGTH,
        }

    def test_zero_offset_for_an_already_query_numbered_model(self, tmp_path):
        cif = write_min_cif(tmp_path / "zero.cif", ca_rows(QUERY_PROTEIN))
        info = pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60)
        assert info["offset"] == 0 and info["identity"] == 1.0

    def test_ranking_is_by_match_count_not_fraction(self, tmp_path):
        """A short overhang can be 100% identical by luck; the true offset is the
        one that maximises MATCHED RESIDUES, which is why the sort key is the
        count and only then the fraction.

        The trap is built explicitly.  The query is a tandem repeat ``P + P``
        (P = the first 36 residues); the structure is ``P + P'`` where P' carries
        9 substitutions, numbered 1..72.  Two candidate offsets:

            offset   0 -> 72 residues compared, 63 match, identity 0.875
            offset +36 -> 36 residues compared, 36 match, identity 1.000

        A fraction-first ranking picks +36 and shifts every structural term by
        half the protein; a count-first ranking picks 0, which is the truth.
        """
        half = QUERY_PROTEIN[:36]
        query = half + half
        decoy = list(half)
        substituted = list(range(0, 36, 4))
        for index in substituted:
            decoy[index] = "W" if decoy[index] != "W" else "C"
        structure_seq = half + "".join(decoy)
        assert len(substituted) == 9
        assert sum(a != b for a, b in zip(query, structure_seq)) == 9

        cif = write_min_cif(tmp_path / "rank.cif", ca_rows(structure_seq))
        info = pi.resolve_structure_offset(pi.load_structure(cif), "A", query, 0.60)
        assert info["offset"] == 0
        assert info["matched_residues"] == 63
        assert info["compared_residues"] == 72
        assert info["identity"] == pytest.approx(63 / 72)

    def test_partial_coverage_only_compares_residues_inside_the_frame(self, tmp_path):
        cif = write_min_cif(tmp_path / "partial.cif", ca_rows(QUERY_PROTEIN[:30], start=1))
        info = pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60)
        assert info["offset"] == 0
        assert info["compared_residues"] == 30
        assert info["matched_residues"] == 30

    def test_wrong_protein_is_refused_by_the_identity_gate(self, tmp_path):
        wrong = "W" * QUERY_LENGTH
        cif = write_min_cif(tmp_path / "wrong.cif", ca_rows(wrong))
        with pytest.raises(ValueError, match=r"only reaches .* identity"):
            pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60)

    def test_identity_gate_can_be_relaxed(self, tmp_path):
        half = QUERY_PROTEIN[:36] + "W" * 36
        cif = write_min_cif(tmp_path / "half.cif", ca_rows(half))
        info = pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.40)
        assert info["offset"] == 0
        assert info["matched_residues"] >= 36

    def test_no_offset_inside_the_frame_at_all(self, tmp_path):
        """Author numbers so far off that no shift in (-40, 41) lands in 1..72."""
        cif = write_min_cif(tmp_path / "far.cif", ca_rows(QUERY_PROTEIN, start=1000))
        with pytest.raises(ValueError, match="No offset places any residue inside the query frame"):
            pi.resolve_structure_offset(pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60)

    def test_search_window_is_configurable(self, tmp_path):
        cif = write_min_cif(tmp_path / "far2.cif", ca_rows(QUERY_PROTEIN, start=1000))
        info = pi.resolve_structure_offset(
            pi.load_structure(cif), "A", QUERY_PROTEIN, 0.60, search=(-1000, -900)
        )
        assert info["offset"] == -999 and info["identity"] == 1.0


@pytest.mark.unit
@pytest.mark.requires_prody
class TestWriteRenumberedPdb:
    def test_offset_is_applied_before_the_frame_filter(self, tmp_path):
        cif = write_min_cif(tmp_path / "shift.cif", ca_rows(QUERY_PROTEIN, start=17))
        out = tmp_path / "shifted.pdb"
        report = pi.write_renumbered_pdb(
            pi.load_structure(cif), ["A"], -16, QUERY_LENGTH, out, collapse_to_chain_a=True
        )
        assert read_pdb_ca_resnums(out) == list(range(1, QUERY_LENGTH + 1))
        assert report["per_chain"]["A"]["resnum_min"] == 1
        assert report["per_chain"]["A"]["resnum_max"] == QUERY_LENGTH

    def test_residues_outside_the_frame_are_dropped(self, tmp_path):
        cif = write_min_cif(tmp_path / "over.cif", ca_rows(QUERY_PROTEIN + "WWWW"))
        out = tmp_path / "clipped.pdb"
        pi.write_renumbered_pdb(pi.load_structure(cif), ["A"], 0, QUERY_LENGTH, out)
        assert max(read_pdb_ca_resnums(out)) == QUERY_LENGTH

    def test_collapse_to_chain_a_renames_every_chain(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN, chain="B")
        cif = write_min_cif(tmp_path / "chainb.cif", rows)
        out = tmp_path / "collapsed.pdb"
        report = pi.write_renumbered_pdb(
            pi.load_structure(cif), ["B"], 0, QUERY_LENGTH, out, collapse_to_chain_a=True
        )
        assert {atom[3] for atom in read_pdb_atoms(out)} == {"A"}
        assert list(report["per_chain"]) == ["A"]
        assert report["chains"] == ["B"]

    def test_without_collapse_the_original_chain_ids_survive(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN, chain="A") + ca_rows(QUERY_PROTEIN, chain="B", y=40.0)
        cif = write_min_cif(tmp_path / "two.cif", rows)
        out = tmp_path / "two.pdb"
        report = pi.write_renumbered_pdb(
            pi.load_structure(cif), ["A", "B"], 0, QUERY_LENGTH, out, collapse_to_chain_a=False
        )
        assert {atom[3] for atom in read_pdb_atoms(out)} == {"A", "B"}
        assert sorted(report["per_chain"]) == ["A", "B"]

    def test_absent_chain_is_refused(self, tmp_path):
        cif = write_min_cif(tmp_path / "only_a.cif", ca_rows(QUERY_PROTEIN))
        with pytest.raises(ValueError, match="No protein atoms in chains Z"):
            pi.write_renumbered_pdb(pi.load_structure(cif), ["Z"], 0, QUERY_LENGTH,
                                    tmp_path / "nope.pdb")

    def test_offset_that_empties_the_frame_is_refused(self, tmp_path):
        cif = write_min_cif(tmp_path / "empty.cif", ca_rows(QUERY_PROTEIN))
        with pytest.raises(ValueError, match="No residues fall inside the query frame"):
            pi.write_renumbered_pdb(pi.load_structure(cif), ["A"], -1000, QUERY_LENGTH,
                                    tmp_path / "nope.pdb")

    def test_insertion_codes_break_the_renumbering_assumption(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN)
        rows.append(("ATOM", "CA", "GLY", "A", 30, "A", 1.0, 1.0, 1.0))
        cif = write_min_cif(tmp_path / "icode.cif", rows)
        with pytest.raises(ValueError, match="Insertion codes present"):
            pi.write_renumbered_pdb(pi.load_structure(cif), ["A"], 0, QUERY_LENGTH,
                                    tmp_path / "nope.pdb")

    def test_two_cas_on_one_residue_number_are_refused(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN)
        rows.append(("ATOM", "CA", "GLY", "A", 30, "?", 1.0, 1.0, 1.0))
        cif = write_min_cif(tmp_path / "dup.cif", rows)
        with pytest.raises(ValueError, match="more than one CA per residue number"):
            pi.write_renumbered_pdb(pi.load_structure(cif), ["A"], 0, QUERY_LENGTH,
                                    tmp_path / "nope.pdb")

    def test_output_directory_is_created(self, tmp_path):
        cif = write_min_cif(tmp_path / "mk.cif", ca_rows(QUERY_PROTEIN))
        out = tmp_path / "deep" / "deeper" / "out.pdb"
        pi.write_renumbered_pdb(pi.load_structure(cif), ["A"], 0, QUERY_LENGTH, out)
        assert out.exists()


@pytest.mark.unit
@pytest.mark.requires_prody
class TestPrepareStructure:
    def test_explicit_offset_skips_the_scan(self, tmp_path):
        cif = write_min_cif(tmp_path / "exp.cif", ca_rows(QUERY_PROTEIN, start=17))
        report = pi.prepare_structure(cif, "A", "-16", QUERY_PROTEIN, tmp_path / "o", "exp", 0.60)
        assert report["offset"] == -16
        assert report["offset_identity"] is None
        assert report["offset_matched_residues"] is None
        assert read_pdb_ca_resnums(Path(report["monomer"]["path"])) == list(range(1, 73))

    def test_explicit_offset_is_not_validated_against_the_sequence(self, tmp_path):
        """A deliberately wrong --structure-offset is honoured: 'auto' is the only
        mode that proves the structure and the query are the same protein."""
        cif = write_min_cif(tmp_path / "bad.cif", ca_rows(QUERY_PROTEIN, start=17))
        report = pi.prepare_structure(cif, "A", "0", QUERY_PROTEIN, tmp_path / "o", "bad", 0.60)
        assert report["offset"] == 0
        assert report["monomer"]["per_chain"]["A"]["resnum_min"] == 17

    def test_file_names_and_coverage(self, tmp_path):
        cif = write_min_cif(tmp_path / "cov.cif", ca_rows(QUERY_PROTEIN[:36]))
        out = tmp_path / "structure"
        report = pi.prepare_structure(cif, "A", "auto", QUERY_PROTEIN, out, "6WXB", 0.60)
        assert Path(report["monomer"]["path"]) == out / "6WXB_chainA_qnum.pdb"
        assert Path(report["trimer"]["path"]) == out / "6WXB_trimer_qnum.pdb"
        assert report["n_covered"] == 36
        assert report["covered_positions"] == list(range(1, 37))
        assert report["uncovered_runs"] == [[37, QUERY_LENGTH]]
        assert report["coverage_fraction"] == 0.5
        assert report["query_length"] == QUERY_LENGTH
        assert report["source_md5"] == common.md5_file(cif)

    def test_trimer_keeps_every_protein_chain_monomer_keeps_one(self, tmp_path):
        rows = []
        for offset, chain in enumerate("ABC"):
            rows.extend(ca_rows(QUERY_PROTEIN, chain=chain, y=offset * 40.0))
        cif = write_min_cif(tmp_path / "tri.cif", rows)
        report = pi.prepare_structure(cif, "A", "auto", QUERY_PROTEIN, tmp_path / "o", "tri", 0.60)
        assert sorted(report["monomer"]["per_chain"]) == ["A"]
        assert sorted(report["trimer"]["per_chain"]) == ["A", "B", "C"]
        assert len(read_pdb_ca_resnums(Path(report["trimer"]["path"]), "C")) == QUERY_LENGTH

    def test_coverage_gap_is_reported_as_a_run(self, tmp_path):
        rows = ca_rows(QUERY_PROTEIN[:20]) + ca_rows(QUERY_PROTEIN[30:], start=31)
        cif = write_min_cif(tmp_path / "gap.cif", rows)
        report = pi.prepare_structure(cif, "A", "auto", QUERY_PROTEIN, tmp_path / "o", "gap", 0.60)
        assert report["monomer"]["per_chain"]["A"]["gaps"] == [[20, 31]]
        assert report["uncovered_runs"] == [[21, 30]]
        assert report["n_covered"] == QUERY_LENGTH - 10


@pytest.mark.unit
class TestResidueReportHelpers:
    """Pure arithmetic; no prody, so these run everywhere."""

    def test_residue_report_contiguous(self):
        assert pi._residue_report([3, 1, 2]) == {
            "n_residues": 3, "resnum_min": 1, "resnum_max": 3, "gaps": []
        }

    def test_residue_report_gaps_are_the_flanking_pair(self):
        report = pi._residue_report([1, 2, 5, 6, 10])
        assert report["gaps"] == [[2, 5], [6, 10]]
        assert report["n_residues"] == 5

    def test_residue_report_deduplicates(self):
        assert pi._residue_report([1, 1, 2, 2])["n_residues"] == 2

    def test_residue_report_empty(self):
        assert pi._residue_report([]) == {
            "n_residues": 0, "resnum_min": None, "resnum_max": None, "gaps": []
        }

    @pytest.mark.parametrize("covered,length,expected", [
        ([1, 2, 3], 3, []),
        ([], 3, [[1, 3]]),
        ([2], 3, [[1, 1], [3, 3]]),
        ([1, 2, 5], 6, [[3, 4], [6, 6]]),
        ([3, 4], 5, [[1, 2], [5, 5]]),
    ])
    def test_uncovered_runs(self, covered, length, expected):
        assert pi._uncovered_runs(covered, length) == expected

    def test_uncovered_runs_ignores_out_of_range_coverage(self):
        assert pi._uncovered_runs([1, 2, 99], 3) == [[3, 3]]


# =========================================================================== #
# Queries: nt -> protein
# =========================================================================== #

@pytest.mark.unit
class TestBuildQueryFastas:
    def test_translation_header_token_and_file_layout(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["K"])
        out = tmp_path / "query"
        queries = pi.build_query_fastas(guide_rows_for(guide), out)

        info = queries["K"]
        assert info["protein"] == QUERY_PROTEIN
        assert info["length"] == QUERY_LENGTH
        assert info["tag"] == "K"
        assert info["header"] == "HAK"
        assert info["prot_token"] == "HAK"
        assert info["lineage_key"] == "K"
        assert info["md5"] == common.md5_text(QUERY_PROTEIN)
        assert info["nucleotide_length"] == len(QUERY_CDS)

        path = out / "K_query.fasta"
        assert Path(info["path"]) == path
        assert path.read_text(encoding="utf-8") == f">HAK\n{QUERY_PROTEIN}\n"

    def test_every_lineage_gets_its_own_file_and_tag(self, tmp_path):
        guide = make_guide(tmp_path / "g")
        out = tmp_path / "query"
        queries = pi.build_query_fastas(guide_rows_for(guide), out)
        assert sorted(queries) == sorted(LINEAGE_ORDER)
        assert {label: info["header"] for label, info in queries.items()} == {
            "G.1": "HAG1", "J_int": "HAJ", "J.2_int": "HAJ2", "J.2.4": "HAJ24", "K": "HAK",
        }
        for label, info in queries.items():
            assert Path(info["path"]).name == f"{common.safe_label(label)}_query.fasta"
            assert Path(info["path"]).exists()

    def test_headers_never_collide_with_escotts_own_output_name(self, tmp_path):
        """escott writes ``<prot>.fasta`` into the CWD, so the query file must NOT
        be called ``HAK.fasta`` -- only the header carries the tag."""
        guide = make_guide(tmp_path / "g")
        out = tmp_path / "query"
        queries = pi.build_query_fastas(guide_rows_for(guide), out)
        names = {Path(info["path"]).name for info in queries.values()}
        assert names.isdisjoint({f"{info['prot_token']}.fasta" for info in queries.values()})

    def test_query_report_tsv(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["K", "J.2.4"])
        out = tmp_path / "query"
        pi.build_query_fastas(guide_rows_for(guide), out)
        lines = (out / "query_report.tsv").read_text(encoding="utf-8").strip().split("\n")
        assert lines[0].split("\t") == [
            "lineage", "lineage_key", "tag", "prot_token", "length", "md5", "reference_path"
        ]
        assert len(lines) == 3
        row = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
        assert row["lineage"] == "K" and row["tag"] == "K" and row["length"] == "72"
        assert row["md5"] == common.md5_text(QUERY_PROTEIN)

    def test_missing_reference_names_the_lineage(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["K"])
        Path(guide["references"]["K"]).unlink()
        with pytest.raises(FileNotFoundError, match="Reference CDS for K not found"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_non_multiple_of_three_drops_the_trailing_partial_codon(self, tmp_path):
        """A CDS with 2 dangling bases must translate to the SAME protein, not
        shift the frame -- ``translate_reference_cds`` trims to floor(len/3)*3."""
        ragged = QUERY_CDS + "AC"
        assert len(ragged) % 3 == 2
        guide = make_guide(tmp_path / "g", labels=["K"], cds_by_label={"K": ragged})
        queries = pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")
        assert queries["K"]["protein"] == QUERY_PROTEIN
        assert queries["K"]["length"] == QUERY_LENGTH
        # The recorded nucleotide length is the FULL cleaned CDS, not the trimmed one.
        assert queries["K"]["nucleotide_length"] == len(ragged)

    @pytest.mark.parametrize("dangling", [1, 2])
    def test_dangling_bases_never_change_the_protein(self, tmp_path, dangling):
        cds = QUERY_CDS + "A" * dangling
        guide = make_guide(tmp_path / f"g{dangling}", labels=["K"], cds_by_label={"K": cds})
        queries = pi.build_query_fastas(guide_rows_for(guide), tmp_path / f"q{dangling}")
        assert queries["K"]["protein"] == QUERY_PROTEIN

    def test_internal_stop_codon_is_deleted_not_truncated(self, tmp_path):
        """HAZARD, pinned deliberately.

        ``translate_reference_cds`` strips every '*' rather than truncating, so an
        internal stop silently yields a protein ONE residue shorter with every
        downstream residue shifted up by one.  Nothing in this module notices when
        only one lineage is being prepared; the cross-lineage length check
        (below) is the only guard, and it does not fire for a single lineage.
        """
        codons = [QUERY_CDS[i:i + 3] for i in range(0, len(QUERY_CDS) - 3, 3)]
        codons[30] = STOP_CODON
        cds = "".join(codons) + STOP_CODON
        guide = make_guide(tmp_path / "g", labels=["K"], cds_by_label={"K": cds})
        queries = pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")
        protein = queries["K"]["protein"]
        assert len(protein) == QUERY_LENGTH - 1
        assert protein == QUERY_PROTEIN[:30] + QUERY_PROTEIN[31:]
        assert "*" not in protein

    def test_internal_stop_in_one_lineage_trips_the_shared_length_guard(self, tmp_path):
        codons = [QUERY_CDS[i:i + 3] for i in range(0, len(QUERY_CDS) - 3, 3)]
        codons[30] = STOP_CODON
        cds = "".join(codons) + STOP_CODON
        guide = make_guide(tmp_path / "g", labels=["K", "J.2.4"], cds_by_label={"K": cds})
        with pytest.raises(ValueError, match=r"differing lengths: \[71, 72\]"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_ambiguous_nucleotide_becomes_x_and_is_refused(self, tmp_path):
        """An N in the CDS translates to X, which ESCOTT's alphabet cannot hold."""
        cds = "AAN" + QUERY_CDS[3:]
        guide = make_guide(tmp_path / "g", labels=["K"], cds_by_label={"K": cds})
        with pytest.raises(ValueError, match=r"non-standard residues \['X'\]"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_protein_fasta_as_a_reference_is_refused_by_the_alphabet_check(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["K"])
        write_fasta(Path(guide["references"]["K"]), [("K_protein", QUERY_PROTEIN)])
        with pytest.raises(ValueError, match="does not look like a nucleotide CDS"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_empty_reference_is_refused(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["K"])
        Path(guide["references"]["K"]).write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No records found in reference FASTA"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_guide_row_with_an_empty_reference_column_should_name_the_lineage(self, tmp_path):
        """Regression test for a fixed guard.

        ``read_guide_rows`` does not require the ``reference`` column, so an empty
        one arrives as ``""`` -- and ``Path("")`` is ``PosixPath(".")``, whose
        ``.exists()`` is True.  The ``.exists()`` guard was therefore bypassed and
        the run died with ``IsADirectoryError('.')`` from inside
        ``load_reference_cds``, naming neither the lineage nor the guide.  The
        guard now tests the raw cell and demands a regular file.
        """
        guide_path = tmp_path / "no_reference.csv"
        guide_path.write_text("month,fasta,reference\nK,panel.fasta,\n", encoding="utf-8")
        rows = common.read_guide_rows(guide_path)
        assert rows == [{"label": "K", "diversity_path": "panel.fasta", "reference_path": ""}]
        with pytest.raises(FileNotFoundError, match="Reference CDS for K not found"):
            pi.build_query_fastas(rows, tmp_path / "query")

    def test_lineage_tag_refuses_a_label_with_no_alphanumerics(self, tmp_path):
        guide = make_guide(tmp_path / "g", labels=["---"])
        with pytest.raises(ValueError, match="Cannot derive an escott tag"):
            pi.build_query_fastas(guide_rows_for(guide), tmp_path / "query")

    def test_translation_matches_the_evaluation_half_exactly(self):
        """The driver asserts score_matrix_source_sequence == full_ref_protein, so
        the two translations must agree character for character."""
        independent = "".join(
            {codon: aa for aa, codon in ONE_CODON_PER_AA.items()}[QUERY_CDS[i:i + 3]]
            for i in range(0, len(QUERY_CDS) - 3, 3)
        )
        assert common.translate_reference_cds(QUERY_CDS) == independent == QUERY_PROTEIN


# =========================================================================== #
# MSA
# =========================================================================== #

@pytest.mark.unit
class TestSanitiseNonstandard:
    def test_keep_is_a_no_op(self):
        rows = [("q", QUERY_PROTEIN), ("d", "B" + QUERY_PROTEIN[1:])]
        out, n = pi._sanitise_nonstandard(list(rows), "keep")
        assert out == rows and n == 0

    def test_to_x_replaces_every_offender(self):
        rows = [("q", QUERY_PROTEIN), ("d", "BZ" + QUERY_PROTEIN[2:] + "JUO")]
        out, n = pi._sanitise_nonstandard(rows, "to-x")
        assert out[1][1] == "XX" + QUERY_PROTEIN[2:] + "XXX"
        assert n == 5

    def test_to_gap_replaces_with_dashes(self):
        rows = [("q", QUERY_PROTEIN), ("d", "B" + QUERY_PROTEIN[1:])]
        out, n = pi._sanitise_nonstandard(rows, "to-gap")
        assert out[1][1] == "-" + QUERY_PROTEIN[1:]
        assert n == 1

    def test_gaps_are_never_offenders(self):
        rows = [("q", QUERY_PROTEIN), ("d", "---" + QUERY_PROTEIN[3:])]
        out, n = pi._sanitise_nonstandard(rows, "to-x")
        assert out[1][1] == "---" + QUERY_PROTEIN[3:] and n == 0

    def test_row_zero_is_never_touched(self):
        """The query is asserted pure-20AA upstream; if a B ever reached row 0 the
        alignment frame check would catch it, so the sanitiser must leave it be."""
        rows = [("q", "B" + QUERY_PROTEIN[1:]), ("d", "B" + QUERY_PROTEIN[1:])]
        out, n = pi._sanitise_nonstandard(rows, "to-x")
        assert out[0][1] == "B" + QUERY_PROTEIN[1:]
        assert out[1][1] == "X" + QUERY_PROTEIN[1:]
        assert n == 1

    def test_x_counts_as_non_standard_under_to_x(self):
        """Documented quirk: X is not in the 20-letter alphabet, so an X already
        present is counted as 'replaced' even though the substitution is X -> X.
        The sequence is unchanged; only the tally moves."""
        rows = [("q", QUERY_PROTEIN), ("d", "X" + QUERY_PROTEIN[1:])]
        out, n = pi._sanitise_nonstandard(rows, "to-x")
        assert out[1][1] == "X" + QUERY_PROTEIN[1:]
        assert n == 1

    def test_rows_without_offenders_are_passed_through_unmodified(self):
        rows = [("q", QUERY_PROTEIN), ("d", QUERY_PROTEIN)]
        out, n = pi._sanitise_nonstandard(rows, "to-x")
        assert out == rows and n == 0


@pytest.mark.unit
class TestStripQueryGapColumns:
    def test_projects_onto_the_query_frame(self):
        rows = [("q", "AB-CD"), ("d1", "ABXCD"), ("d2", "--XC-")]
        assert pi._strip_query_gap_columns(rows) == [
            ("q", "ABCD"), ("d1", "ABCD"), ("d2", "--C-")
        ]

    def test_ungapped_query_is_the_identity(self):
        rows = [("q", QUERY_PROTEIN), ("d", "-" * QUERY_LENGTH)]
        assert pi._strip_query_gap_columns(rows) == rows

    def test_every_row_keeps_its_alignment_to_the_query(self):
        """The property that matters: after stripping, column j of every row is
        still the residue that was aligned to query residue j."""
        query = "AB-CD--E"
        rows = [("q", query), ("d1", "ABQCDRSE"), ("d2", "-BQ-D-SE")]
        stripped = pi._strip_query_gap_columns(rows)
        keep = [i for i, char in enumerate(query) if char != "-"]
        for (_h0, original), (_h1, projected) in zip(rows, stripped):
            assert projected == "".join(original[i] for i in keep)

    def test_headers_are_preserved_in_order(self):
        rows = [("q", "A-B"), ("x", "AZB"), ("y", "A-B")]
        assert [h for h, _s in pi._strip_query_gap_columns(rows)] == ["q", "x", "y"]

    def test_all_gap_query_yields_empty_rows(self):
        rows = [("q", "---"), ("d", "ABC")]
        assert pi._strip_query_gap_columns(rows) == [("q", ""), ("d", "")]


@pytest.mark.integration
class TestBuildDeepMsaWithFakeMafft:
    """Everything except the alignment itself.  A fake aligner is the only way to
    reach the four defensive raises, and it keeps the default suite offline."""

    def test_happy_path_products_and_report(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(
            deep_fasta, "G.1", anchor_info, out, fake_mafft("passthrough"),
            threads=2, mode="auto", nonstandard_policy="to-x", force=False,
        )
        assert report["n_rows"] == 7
        assert report["n_deep_sequences"] == 6
        assert report["n_columns"] == QUERY_LENGTH
        assert report["raw_columns"] == QUERY_LENGTH
        assert report["anchor_lineage"] == "G.1"
        assert report["anchor_header"] == "HAG1"
        assert report["mafft_mode"] == "auto"
        assert report["mafft_threads"] == 2
        assert report["msa_nonstandard"] == "to-x"
        assert report["n_nonstandard_residues_replaced"] == 0
        assert report["deep_fasta_md5"] == common.md5_file(deep_fasta)

        msa = out / "deep_msa_566.fasta"
        assert Path(report["msa_566_path"]) == msa
        assert report["msa_566_md5"] == common.md5_file(msa)
        rows = list(common.read_fasta(msa))
        assert len(rows) == 7
        assert (out / "msa_report.json").exists()

    def test_query_is_row_zero_and_ungapped(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        """GEMME/ESCOTT index every prediction by row 1's columns, so row 1 IS the
        query frame: first, and gap-free."""
        out = tmp_path / "msa"
        pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("insertion"),
                          2, "auto", "to-x", False)
        rows = list(common.read_fasta(out / "deep_msa_566.fasta"))
        header, sequence = rows[0]
        assert header == "HAG1"
        assert "-" not in sequence
        assert sequence == QUERY_PROTEIN

    def test_insertion_columns_are_stripped_and_every_other_row_stays_aligned(
        self, tmp_path, anchor_info, deep_fasta, fake_mafft
    ):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("insertion"),
                                   2, "auto", "to-x", False)
        assert report["raw_columns"] == QUERY_LENGTH + 1
        assert report["n_columns"] == QUERY_LENGTH

        raw = list(common.read_fasta(out / "deep_msa_raw.fasta"))
        stripped = list(common.read_fasta(out / "deep_msa_566.fasta"))
        keep = [i for i, char in enumerate(raw[0][1]) if char != "-"]
        assert len(keep) == QUERY_LENGTH
        for (raw_header, raw_seq), (out_header, out_seq) in zip(raw, stripped):
            assert raw_header == out_header
            assert out_seq == "".join(raw_seq[i] for i in keep)
        # The inserted W sat in the query-gap column, so no deep row keeps it.
        deep_original = {h: s for h, s in common.read_fasta(deep_fasta)}
        for header, sequence in stripped[1:]:
            assert sequence == deep_original[header]

    def test_all_rows_have_the_query_width(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        out = tmp_path / "msa"
        pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("insertion"),
                          2, "auto", "to-x", False)
        widths = {len(seq) for _h, seq in common.read_fasta(out / "deep_msa_566.fasta")}
        assert widths == {QUERY_LENGTH}

    def test_nonstandard_residues_are_normalised_and_counted(
        self, tmp_path, anchor_info, deep_fasta, fake_mafft
    ):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("nonstandard"),
                                   2, "auto", "to-x", False)
        assert report["n_nonstandard_residues_replaced"] == 12  # 2 per deep row x 6 rows
        rows = list(common.read_fasta(out / "deep_msa_566.fasta"))
        assert rows[0][1] == QUERY_PROTEIN
        for _header, sequence in rows[1:]:
            assert set(sequence) <= set(AA20) | {"X", "-"}
            assert sequence[0] == "X" and sequence[3] == "X"

    def test_to_gap_policy_writes_dashes(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("nonstandard"),
                                   2, "auto", "to-gap", False)
        assert report["msa_nonstandard"] == "to-gap"
        rows = list(common.read_fasta(out / "deep_msa_566.fasta"))
        assert rows[1][1][0] == "-" and rows[1][1][3] == "-"

    def test_keep_policy_leaves_b_and_z_in_place(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("nonstandard"),
                                   2, "auto", "keep", False)
        assert report["n_nonstandard_residues_replaced"] == 0
        rows = list(common.read_fasta(out / "deep_msa_566.fasta"))
        assert rows[1][1][0] == "B" and rows[1][1][3] == "Z"

    def test_keeplength_mode_builds_the_add_command(self, tmp_path, anchor_info, deep_fasta,
                                                    fake_mafft, capsys):
        out = tmp_path / "msa"
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("passthrough"),
                                   2, "keeplength", "to-x", False)
        printed = capsys.readouterr().out
        assert "--keeplength" in printed and "--add" in printed
        assert (out / "deep_only.fasta").exists()
        assert report["mafft_mode"] == "keeplength"
        assert report["n_rows"] == 7
        rows = list(common.read_fasta(out / "deep_msa_566.fasta"))
        assert rows[0][1] == QUERY_PROTEIN

    def test_cache_hit_skips_mafft_entirely(self, tmp_path, anchor_info, deep_fasta, fake_mafft,
                                            capsys):
        out = tmp_path / "msa"
        good = fake_mafft("passthrough")
        first = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, good,
                                  2, "auto", "to-x", False)
        capsys.readouterr()
        exploding = fake_mafft("fail")
        second = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, exploding,
                                   2, "auto", "to-x", False)
        assert "MSA cache hit" in capsys.readouterr().out
        assert second["msa_566_md5"] == first["msa_566_md5"]
        assert second["mafft_bin"] == str(good)

    def test_force_ignores_the_cache(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        out = tmp_path / "msa"
        pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("passthrough"),
                          2, "auto", "to-x", False)
        with pytest.raises(RuntimeError, match="mafft failed"):
            pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("fail"),
                              2, "auto", "to-x", force=True)

    @pytest.mark.parametrize("changed", ["mode", "policy", "threads", "anchor"])
    def test_cache_key_covers_every_input_that_changes_the_output(
        self, tmp_path, anchor_info, deep_fasta, fake_mafft, changed
    ):
        out = tmp_path / "msa"
        pi.build_deep_msa(deep_fasta, "G.1", anchor_info, out, fake_mafft("passthrough"),
                          2, "auto", "to-x", False)
        kwargs = {"mode": "auto", "nonstandard_policy": "to-x", "threads": 2}
        info = dict(anchor_info)
        if changed == "mode":
            kwargs["mode"] = "keeplength"
        elif changed == "policy":
            kwargs["nonstandard_policy"] = "to-gap"
        elif changed == "threads":
            kwargs["threads"] = 4
        else:
            info["md5"] = common.md5_text(QUERY_PROTEIN + "A")
        # A stale cache would return without invoking mafft; the exploding fake
        # proves the cache was correctly invalidated.
        with pytest.raises(RuntimeError, match="mafft failed"):
            pi.build_deep_msa(deep_fasta, "G.1", info, out, fake_mafft("fail"),
                              force=False, **kwargs)

    def test_mafft_failure_surfaces_the_return_code_and_stderr(
        self, tmp_path, anchor_info, deep_fasta, fake_mafft
    ):
        with pytest.raises(RuntimeError, match=r"mafft failed \(3\)"):
            pi.build_deep_msa(deep_fasta, "G.1", anchor_info, tmp_path / "msa",
                              fake_mafft("fail"), 2, "auto", "to-x", False)

    def test_ragged_output_is_refused(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        with pytest.raises(ValueError, match="mafft output is ragged"):
            pi.build_deep_msa(deep_fasta, "G.1", anchor_info, tmp_path / "msa",
                              fake_mafft("ragged"), 2, "auto", "to-x", False)

    def test_a_dropped_row_is_refused(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        with pytest.raises(ValueError, match=r"mafft returned 6 rows, expected 7"):
            pi.build_deep_msa(deep_fasta, "G.1", anchor_info, tmp_path / "msa",
                              fake_mafft("drop_row"), 2, "auto", "to-x", False)

    def test_a_corrupted_anchor_is_refused(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        with pytest.raises(ValueError, match="does not reproduce the anchor query"):
            pi.build_deep_msa(deep_fasta, "G.1", anchor_info, tmp_path / "msa",
                              fake_mafft("corrupt_anchor"), 2, "auto", "to-x", False)

    def test_empty_deep_set_is_refused(self, tmp_path, anchor_info, fake_mafft):
        empty = tmp_path / "empty.fasta"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Deep MSA source is empty"):
            pi.build_deep_msa(empty, "G.1", anchor_info, tmp_path / "msa",
                              fake_mafft("passthrough"), 2, "auto", "to-x", False)

    def test_mafft_version_is_recorded(self, tmp_path, anchor_info, deep_fasta, fake_mafft):
        report = pi.build_deep_msa(deep_fasta, "G.1", anchor_info, tmp_path / "msa",
                                   fake_mafft("passthrough"), 2, "auto", "to-x", False)
        assert report["mafft_version"] == "v0.0 (fake)"

    def test_mafft_version_degrades_to_unknown(self, tmp_path):
        assert pi._mafft_version(tmp_path / "definitely_not_here") == "unknown"


@pytest.mark.integration
@pytest.mark.requires_mafft
class TestBuildDeepMsaWithRealMafft:
    """One test with the real aligner, because the fake one cannot prove that
    mafft keeps the query first or that the projection survives real gaps."""

    def test_real_alignment_keeps_the_query_first_ungapped_and_in_frame(
        self, tmp_path, anchor_info
    ):
        # Deep rows with a genuine 5-residue insertion and a deletion, so mafft
        # must gap the query and _strip_query_gap_columns must undo it.
        deep = write_fasta(tmp_path / "deep.fasta", [
            ("ins", QUERY_PROTEIN[:30] + "WWWWW" + QUERY_PROTEIN[30:]),
            ("del", QUERY_PROTEIN[:20] + QUERY_PROTEIN[26:]),
            ("sub", _divergent(QUERY_PROTEIN, 4, 1)),
            ("both", QUERY_PROTEIN[:10] + "YY" + QUERY_PROTEIN[10:40] + QUERY_PROTEIN[44:]),
        ])
        out = tmp_path / "msa"
        report = pi.build_deep_msa(
            deep, "G.1", anchor_info, out, Path("mafft"), 2, "auto", "to-x", False
        )
        assert report["n_rows"] == 5
        assert report["n_columns"] == QUERY_LENGTH
        assert report["raw_columns"] > QUERY_LENGTH  # the insertion widened it

        raw = list(common.read_fasta(out / "deep_msa_raw.fasta"))
        stripped = list(common.read_fasta(out / "deep_msa_566.fasta"))
        assert stripped[0][0] == "HAG1"
        assert stripped[0][1] == QUERY_PROTEIN
        assert "-" not in stripped[0][1]

        keep = [i for i, char in enumerate(raw[0][1]) if char != "-"]
        for (raw_header, raw_seq), (out_header, out_seq) in zip(raw, stripped):
            assert raw_header == out_header
            assert len(out_seq) == QUERY_LENGTH
            assert out_seq == "".join(raw_seq[i] for i in keep)

        # The substitution-only row must be recoverable position by position.
        by_header = dict(stripped)
        sub = by_header["sub"]
        assert sub.replace("-", "") != ""
        assert sum(1 for a, b in zip(sub, QUERY_PROTEIN) if a != b) <= 8

    def test_keeplength_mode_needs_no_stripping(self, tmp_path, anchor_info):
        deep = write_fasta(tmp_path / "deep.fasta", [
            ("ins", QUERY_PROTEIN[:30] + "WWWWW" + QUERY_PROTEIN[30:]),
            ("sub", _divergent(QUERY_PROTEIN, 3, 2)),
        ])
        report = pi.build_deep_msa(
            deep, "G.1", anchor_info, tmp_path / "msa", Path("mafft"),
            2, "keeplength", "to-x", False,
        )
        assert report["raw_columns"] == QUERY_LENGTH
        assert report["n_columns"] == QUERY_LENGTH


@pytest.mark.unit
class TestMaterialiseLineageMsa:
    @pytest.fixture()
    def shared_msa(self, tmp_path):
        rows = [("HAG1", QUERY_PROTEIN)] + [
            (f"DEEP{i}", _divergent(QUERY_PROTEIN, 2 + i, i)) for i in range(5)
        ]
        return write_fasta(tmp_path / "deep_msa_566.fasta", rows), rows

    def test_row_zero_is_replaced_and_every_other_row_is_byte_identical(self, tmp_path, shared_msa):
        source, rows = shared_msa
        out = tmp_path / "msa_K.fasta"
        info = pi.materialise_lineage_msa(
            source, "K", {"protein": PARENT_PROTEIN, "header": "HAK"}, out, "G.1"
        )
        new_rows = list(common.read_fasta(out))
        assert new_rows[0] == ("HAK", PARENT_PROTEIN)
        assert new_rows[1:] == rows[1:]
        assert info["n_rows"] == len(rows)
        assert info["n_columns"] == QUERY_LENGTH
        assert info["anchor_lineage"] == "G.1"
        assert info["md5"] == common.md5_file(out)
        assert Path(info["path"]) == out

    def test_residue_difference_count_is_exact(self, tmp_path, shared_msa):
        source, _rows = shared_msa
        info = pi.materialise_lineage_msa(
            source, "K", {"protein": PARENT_PROTEIN, "header": "HAK"},
            tmp_path / "out.fasta", "G.1",
        )
        # PARENT_PROTEIN is QUERY_PROTEIN with exactly one substitution (T40I).
        assert info["n_query_residues_changed_vs_anchor"] == 1

    def test_identical_query_reports_zero_changes(self, tmp_path, shared_msa):
        source, _rows = shared_msa
        info = pi.materialise_lineage_msa(
            source, "G.1", {"protein": QUERY_PROTEIN, "header": "HAG1"},
            tmp_path / "out.fasta", "G.1",
        )
        assert info["n_query_residues_changed_vs_anchor"] == 0

    def test_length_mismatch_is_refused(self, tmp_path, shared_msa):
        source, _rows = shared_msa
        with pytest.raises(ValueError, match=r"query is 71 aa but the MSA has 72 columns"):
            pi.materialise_lineage_msa(
                source, "K", {"protein": QUERY_PROTEIN[:-1], "header": "HAK"},
                tmp_path / "out.fasta", "G.1",
            )


# =========================================================================== #
# Frequency files
# =========================================================================== #

@pytest.mark.unit
class TestColumnResidueCounts:
    def test_counts_are_per_column_over_the_20_letters(self):
        counts, allowed, aln_len = pi._column_residue_counts(["AC", "AD", "GC"])
        assert allowed == list(AA20)
        assert aln_len == 2
        assert counts[allowed.index("A"), 0] == 2
        assert counts[allowed.index("G"), 0] == 1
        assert counts[allowed.index("C"), 1] == 2
        assert counts[allowed.index("D"), 1] == 1
        assert counts.sum(axis=0).tolist() == [3, 3]

    def test_gaps_x_and_stops_do_not_count_towards_depth(self):
        counts, _allowed, _len = pi._column_residue_counts(["A", "-", "X", "*", "."])
        assert counts.sum(axis=0).tolist() == [1]

    def test_ragged_rows_are_right_padded_with_gaps(self):
        counts, allowed, aln_len = pi._column_residue_counts(["AC", "A"])
        assert aln_len == 2
        assert counts.sum(axis=0).tolist() == [2, 1]


@pytest.mark.unit
class TestBuildParentFrequencyFile:
    """The PRESCOTT custom-format file.  Three decisions are load-bearing:
    unobserved mutants are OMITTED (never 0.0), near-fixed mutants are dropped,
    and reversions to the parent's own residue are dropped by IDENTITY."""

    def test_emitted_set_and_frequencies_are_exactly_the_planted_ones(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        emitted = parse_frequency_file(out_txt)
        expected = frequency_panels["expected_frequency_file_min_count_1"]
        assert set(emitted) == set(expected)
        for mutant, frequency in expected.items():
            assert emitted[mutant] == pytest.approx(frequency, abs=1e-12)
        assert report["n_mutants"] == len(expected)

    def test_output_directory_is_created(self, frequency_panels, freq_paths):
        out_txt, out_meta = freq_paths
        assert not out_txt.parent.exists()
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        assert out_txt.exists() and out_meta.exists()

    def test_file_format_is_two_space_separated_fields_in_scientific_notation(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        text = out_txt.read_text(encoding="utf-8")
        assert text.endswith("\n")
        for line in text.splitlines():
            assert re.fullmatch(r"[A-Z]\d+[A-Z] -?\d\.\d{10}e[+-]\d\d", line), line
        assert "I10K 1.0000000000e-01" in text
        assert out_txt.suffix == ".txt"  # .csv would switch prescott to gnomAD parsing

    def test_gapped_column_uses_depth_not_record_count_as_the_denominator(
        self, frequency_panels, freq_paths
    ):
        """Position 20 has 5 gaps, so P20E is 19/95, not 19/100."""
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        emitted = parse_frequency_file(out_txt)
        assert emitted["P20E"] == pytest.approx(19 / 95, abs=1e-12)
        assert emitted["P20E"] != pytest.approx(19 / 100, abs=1e-12)

    def test_unobserved_mutants_are_omitted_never_written_as_zero(
        self, frequency_panels, freq_paths
    ):
        """log10(0) = -inf at prescott.py:1113 would drive every unobserved
        variant to maximal deleteriousness under equation 3."""
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        emitted = parse_frequency_file(out_txt)
        assert all(value > 0.0 for value in emitted.values())
        assert len(emitted) == 6
        assert len(emitted) < 19 * QUERY_LENGTH
        assert "M1A" not in emitted  # invariant column: nothing observed

    def test_one_over_n_floor(self, panel_factory, freq_paths):
        """A single observation in a 100-deep column is 1/100 and must be kept:
        it is the smallest frequency the panel can express."""
        out_txt, out_meta = freq_paths
        panel, truth = panel_factory({50: {"T": 1}})
        report = build_freq(panel, out_txt, out_meta, min_count=1)
        emitted = parse_frequency_file(out_txt)
        assert emitted == {"I50T": 0.01}
        assert truth["depths"][50] == PANEL_N_RECORDS
        assert min(emitted.values()) == 1 / PANEL_N_RECORDS
        assert report["n_mutants_below_min_count"] == 0
        assert out_txt.read_text(encoding="utf-8") == "I50T 1.0000000000e-02\n"

    def test_min_count_two_drops_the_singleton_and_counts_it(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta, min_count=2)
        emitted = parse_frequency_file(out_txt)
        assert set(emitted) == set(frequency_panels["expected_frequency_file_min_count_2"])
        assert "S25W" not in emitted
        assert report["n_mutants_below_min_count"] == 1
        assert report["min_count"] == 2

    def test_frequency_at_or_above_freq_max_is_dropped_as_ancestral(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        assert "C30Y" not in parse_frequency_file(out_txt)
        assert report["n_reverted_ancestral_mutants"] == 1
        assert report["reverted_ancestral_mutants"] == [
            {"mutant": "C30Y", "frequency": 0.98, "count": 98, "depth": 100}
        ]

    def test_raising_freq_max_above_the_observation_keeps_it(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta, freq_max=0.99)
        assert parse_frequency_file(out_txt)["C30Y"] == pytest.approx(0.98)
        assert report["n_reverted_ancestral_mutants"] == 0

    def test_frequency_exactly_one_is_dropped_at_freq_max_one(self, panel_factory, freq_paths):
        """The cutoff is ``>=``, so a mutant fixed in the parent panel is dropped
        even when freq_max is set to exactly 1.0."""
        out_txt, out_meta = freq_paths
        panel, truth = panel_factory({12: {"Y": PANEL_N_RECORDS}})
        assert truth["frequencies"]["C12Y"] == 1.0
        report = build_freq(panel, out_txt, out_meta, freq_max=1.0)
        assert parse_frequency_file(out_txt) == {}
        assert out_txt.read_text(encoding="utf-8") == ""
        assert report["n_mutants"] == 0
        assert report["reverted_ancestral_mutants"] == [
            {"mutant": "C12Y", "frequency": 1.0, "count": 100, "depth": 100}
        ]

    def test_frequency_exactly_one_is_emitted_when_freq_max_exceeds_one(
        self, panel_factory, freq_paths
    ):
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({12: {"Y": PANEL_N_RECORDS}})
        build_freq(panel, out_txt, out_meta, freq_max=1.1)
        assert out_txt.read_text(encoding="utf-8") == "C12Y 1.0000000000e+00\n"
        assert parse_frequency_file(out_txt)["C12Y"] == 1.0

    def test_zero_depth_column_produces_no_nan_and_no_division(self, panel_factory, freq_paths):
        """An all-gap column has depth 0.  It must contribute nothing -- not a NaN,
        not an inf, and not a ZeroDivisionError."""
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({30: {"-": PANEL_N_RECORDS}, 15: {"D": 25}})
        report = build_freq(panel, out_txt, out_meta, min_depth=0)
        emitted = parse_frequency_file(out_txt)
        assert emitted == {"F15D": 0.25}
        assert all(math.isfinite(value) for value in emitted.values())
        assert "nan" not in out_txt.read_text(encoding="utf-8").lower()
        # The dead column drops out of the consensus, so it is simply unmapped.
        assert report["alignment_length"] == QUERY_LENGTH
        assert report["consensus_length"] == QUERY_LENGTH - 1
        assert report["mapped_ref_sites"] == QUERY_LENGTH - 1

    def test_zero_depth_column_does_not_shift_the_other_positions(self, panel_factory, freq_paths):
        """The dangerous version of the bug above: an off-by-one in the
        reference->column map would move every mutant downstream of the dead
        column by one residue."""
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({30: {"-": PANEL_N_RECORDS}, 35: {"D": 30}, 15: {"D": 25}})
        build_freq(panel, out_txt, out_meta, min_depth=0)
        emitted = parse_frequency_file(out_txt)
        assert set(emitted) == {"F15D", "A35D"}
        assert "A34D" not in emitted and "A36D" not in emitted

    def test_shallow_columns_are_skipped_and_counted(self, panel_factory, freq_paths):
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({15: {"D": 25}}, n_records=40)
        report = build_freq(panel, out_txt, out_meta, min_depth=50)
        assert report["n_mutants"] == 0
        assert report["n_positions_below_min_depth"] == QUERY_LENGTH
        assert parse_frequency_file(out_txt) == {}

    def test_min_depth_boundary_is_inclusive(self, panel_factory, freq_paths):
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({15: {"D": 25}}, n_records=50)
        assert build_freq(panel, out_txt, out_meta, min_depth=50)["n_mutants"] == 1
        assert build_freq(panel, out_txt, out_meta, min_depth=51)["n_mutants"] == 0

    def test_non_standard_residues_are_excluded_from_the_denominator(
        self, panel_factory, freq_paths
    ):
        out_txt, out_meta = freq_paths
        panel, truth = panel_factory({45: {"X": 10, "V": 5}})
        build_freq(panel, out_txt, out_meta)
        assert truth["depths"][45] == 90
        assert parse_frequency_file(out_txt)["I45V"] == pytest.approx(5 / 90, abs=1e-12)

    def test_multi_allelic_position_emits_every_allele(self, frequency_panels, freq_paths):
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        emitted = parse_frequency_file(out_txt)
        assert emitted["A35D"] == pytest.approx(0.30)
        assert emitted["A35S"] == pytest.approx(0.20)

    def test_wild_type_is_never_emitted_as_a_mutant(self, frequency_panels, freq_paths):
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        for mutant in parse_frequency_file(out_txt):
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant)
            wt, pos, mut = match.group(1), int(match.group(2)), match.group(3)
            assert wt == QUERY_PROTEIN[pos - 1]
            assert mut != wt

    def test_report_bookkeeping(self, frequency_panels, freq_paths):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        assert report["child_lineage"] == "K"
        assert report["parent_lineage"] == "J.2.4"
        assert report["parent_panel"] == str(frequency_panels["parent_fasta"])
        assert report["parent_panel_records"] == PANEL_N_RECORDS
        assert report["alignment_length"] == QUERY_LENGTH
        assert report["consensus_length"] == QUERY_LENGTH
        assert report["mapped_ref_sites"] == QUERY_LENGTH
        assert report["matched_pairs"] == QUERY_LENGTH
        assert report["median_mapped_depth"] == frequency_panels["expected_median_depth"]
        assert report["min_depth"] == 50 and report["freq_max"] == 0.95
        assert report["frequency_md5"] == common.md5_file(out_txt)
        assert Path(report["frequency_path"]) == out_txt
        assert Path(report["meta_path"]) == out_meta

    def test_meta_tsv_mirrors_the_frequency_file_row_for_row(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        build_freq(frequency_panels["parent_fasta"], out_txt, out_meta)
        lines = out_meta.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0].split("\t") == [
            "mutant", "position", "wt", "mut", "count", "depth", "frequency", "parent_lineage"
        ]
        emitted = parse_frequency_file(out_txt)
        assert len(lines) == len(emitted) + 1
        for line in lines[1:]:
            mutant, position, wt, mut, count, depth, frequency, parent = line.split("\t")
            assert parent == "J.2.4"
            assert mutant == f"{wt}{position}{mut}"
            assert float(frequency) == pytest.approx(int(count) / int(depth), abs=1e-12)
            assert emitted[mutant] == pytest.approx(float(frequency), abs=1e-15)

    def test_meta_tsv_is_written_even_when_nothing_is_emitted(self, panel_factory, freq_paths):
        out_txt, out_meta = freq_paths
        panel, _truth = panel_factory({})
        build_freq(panel, out_txt, out_meta)
        assert out_txt.read_text(encoding="utf-8") == ""
        assert out_meta.read_text(encoding="utf-8").strip().startswith("mutant\t")

    def test_empty_panel_is_refused(self, tmp_path, freq_paths):
        out_txt, out_meta = freq_paths
        empty = tmp_path / "empty.fasta"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="contains no records"):
            build_freq(empty, out_txt, out_meta)

    def test_all_gap_panel_maps_nothing_and_reports_zero_median_depth(self, tmp_path, freq_paths):
        out_txt, out_meta = freq_paths
        panel = write_fasta(
            tmp_path / "gaps.fasta",
            [(f"G{i}", "-" * QUERY_LENGTH) for i in range(10)],
        )
        report = build_freq(panel, out_txt, out_meta, min_depth=0)
        assert report["mapped_ref_sites"] == 0
        assert report["consensus_length"] == 0
        assert report["median_mapped_depth"] == 0.0
        assert report["n_mutants"] == 0


@pytest.mark.unit
class TestDropParentReversions:
    """--parent-freq-max is the wrong instrument for a categorical problem: on the
    real data K's reversion N160S sits at 0.932 and K176I at 0.897, so a 0.95
    cutoff lets two of the seven K-defining sites through.  The identity filter is
    exact and threshold-free -- and must not remove anything else."""

    @pytest.fixture()
    def low_frequency_reversion_panel(self, panel_factory):
        """A parent panel whose T40I reversion sits at 0.12, far below freq_max.

        NOTE: this is built on ``QUERY_PROTEIN`` (the CHILD's residues) rather
        than reusing ``frequency_panels``, whose parent panel carries I at 40 in
        every record and therefore puts T40I at 1.0 -- a frequency the 0.95 cutoff
        would catch on its own, which is exactly what this test must rule out.
        """
        return panel_factory(
            {40: {"I": 12}, 10: {"K": 10}, 35: {"D": 30}}, protein=QUERY_PROTEIN
        )

    def test_reversion_is_dropped_although_the_frequency_cutoff_cannot_see_it(
        self, low_frequency_reversion_panel, freq_paths
    ):
        panel, truth = low_frequency_reversion_panel
        out_txt, out_meta = freq_paths
        assert truth["frequencies"]["T40I"] == pytest.approx(0.12)
        assert truth["frequencies"]["T40I"] < 0.95

        report = build_freq(panel, out_txt, out_meta, drop_parent_reversions=True)
        emitted = parse_frequency_file(out_txt)
        assert "T40I" not in emitted
        assert report["n_parent_reversion_mutants"] == 1
        assert report["parent_reversion_mutants"] == [
            {"mutant": "T40I", "frequency": 0.12, "count": 12, "depth": 100}
        ]
        assert report["n_reverted_ancestral_mutants"] == 0  # the cutoff caught nothing
        assert report["drop_parent_reversions"] is True

    def test_it_drops_the_reversion_and_nothing_else(
        self, low_frequency_reversion_panel, freq_paths
    ):
        panel, _truth = low_frequency_reversion_panel
        out_txt, out_meta = freq_paths
        with_filter = parse_frequency_file(
            (build_freq(panel, out_txt, out_meta, drop_parent_reversions=True), out_txt)[1]
        )
        without_filter = parse_frequency_file(
            (build_freq(panel, out_txt, out_meta, drop_parent_reversions=False), out_txt)[1]
        )
        assert set(without_filter) - set(with_filter) == {"T40I"}
        assert with_filter == {k: v for k, v in without_filter.items() if k != "T40I"}
        assert set(with_filter) == {"I10K", "A35D"}

    def test_disabled_filter_emits_the_reversion_at_its_true_frequency(
        self, low_frequency_reversion_panel, freq_paths
    ):
        panel, _truth = low_frequency_reversion_panel
        out_txt, out_meta = freq_paths
        report = build_freq(panel, out_txt, out_meta, drop_parent_reversions=False)
        assert parse_frequency_file(out_txt)["T40I"] == pytest.approx(0.12)
        assert report["n_parent_reversion_mutants"] == 0
        assert report["drop_parent_reversions"] is False

    def test_filter_is_inert_without_a_parent_reference(
        self, low_frequency_reversion_panel, freq_paths
    ):
        """``parent_protein=None`` must not silently behave like the filter is on."""
        panel, _truth = low_frequency_reversion_panel
        out_txt, out_meta = freq_paths
        report = build_freq(panel, out_txt, out_meta, parent_protein=None,
                            drop_parent_reversions=True)
        assert report["drop_parent_reversions"] is False
        assert parse_frequency_file(out_txt)["T40I"] == pytest.approx(0.12)

    def test_only_sites_where_parent_and_child_differ_are_eligible(
        self, panel_factory, freq_paths
    ):
        """At a site where parent and child agree, the 'reversion' is the wild
        type itself and can never be a mutant; nothing may be dropped there."""
        panel, _truth = panel_factory({10: {"K": 10}, 35: {"D": 30}}, protein=QUERY_PROTEIN)
        out_txt, out_meta = freq_paths
        report = build_freq(panel, out_txt, out_meta, parent_protein=QUERY_PROTEIN)
        assert report["n_parent_reversion_mutants"] == 0
        assert set(parse_frequency_file(out_txt)) == {"I10K", "A35D"}

    def test_a_non_reversion_mutation_at_the_defining_site_survives(
        self, panel_factory, freq_paths
    ):
        """Position 40: parent has I, child has T.  T40I is a reversion and goes;
        T40V is ordinary standing variation at the same site and must stay."""
        panel, _truth = panel_factory({40: {"I": 12, "V": 8}}, protein=QUERY_PROTEIN)
        out_txt, out_meta = freq_paths
        report = build_freq(panel, out_txt, out_meta)
        emitted = parse_frequency_file(out_txt)
        assert emitted == {"T40V": pytest.approx(0.08)}
        assert report["n_parent_reversion_mutants"] == 1

    def test_reversion_is_dropped_before_the_frequency_cutoff_can_claim_it(
        self, frequency_panels, freq_paths
    ):
        """Order matters for the audit trail: on this panel T40I is at 1.0, so both
        rules would fire.  The identity rule runs first, so it is booked as a
        parent reversion, not as a near-fixed mutant."""
        out_txt, out_meta = freq_paths
        on = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta,
                        drop_parent_reversions=True)
        assert on["n_parent_reversion_mutants"] == 1
        assert [m["mutant"] for m in on["parent_reversion_mutants"]] == ["T40I"]
        assert [m["mutant"] for m in on["reverted_ancestral_mutants"]] == ["C30Y"]

        off = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta,
                         drop_parent_reversions=False)
        assert off["n_parent_reversion_mutants"] == 0
        assert sorted(m["mutant"] for m in off["reverted_ancestral_mutants"]) == ["C30Y", "T40I"]
        assert "T40I" not in parse_frequency_file(out_txt)

    def test_parent_of_a_different_length_is_refused(self, frequency_panels, freq_paths):
        out_txt, out_meta = freq_paths
        with pytest.raises(ValueError, match=r"parent reference is 71 aa but child is 72 aa"):
            build_freq(frequency_panels["parent_fasta"], out_txt, out_meta,
                       parent_protein=PARENT_PROTEIN[:-1])

    def test_length_mismatch_is_tolerated_when_the_filter_is_off(
        self, frequency_panels, freq_paths
    ):
        out_txt, out_meta = freq_paths
        report = build_freq(frequency_panels["parent_fasta"], out_txt, out_meta,
                            parent_protein=PARENT_PROTEIN[:-1], drop_parent_reversions=False)
        assert report["drop_parent_reversions"] is False


@pytest.mark.unit
class TestResolveFrequencyCutoffs:
    def test_depth_scaled_is_log10_k_over_median_depth(self):
        cutoffs = pi.resolve_frequency_cutoffs("depth_scaled", [1], -4.0, 100.0)
        assert cutoffs == {"1": -2.0}

    def test_depth_scaled_over_several_k(self):
        cutoffs = pi.resolve_frequency_cutoffs("depth_scaled", [1, 2, 10], -4.0, 1000.0)
        assert cutoffs["1"] == pytest.approx(-3.0)
        assert cutoffs["2"] == pytest.approx(math.log10(2 / 1000))
        assert cutoffs["10"] == pytest.approx(-2.0)
        assert list(cutoffs) == ["1", "2", "10"]

    def test_depth_scaling_makes_the_penalty_depth_independent(self):
        """Fc = log10(k/N) is what makes the v2 penalty exactly c*log_N(count):
        0 for a singleton, c for a fixed variant, at ANY panel depth."""
        for depth in (229.0, 877.0, 27452.0):
            fc = pi.resolve_frequency_cutoffs("depth_scaled", [1], -4.0, depth)["1"]
            singleton = math.log10(1 / depth)
            assert singleton / fc == pytest.approx(1.0)
            assert math.log10(1.0) / fc == pytest.approx(0.0)

    def test_fixed_ignores_the_depth(self):
        assert pi.resolve_frequency_cutoffs("fixed", [1, 3], -4.0, 100.0) == {
            "1": -4.0, "3": -4.0
        }
        assert pi.resolve_frequency_cutoffs("fixed", [1], -4.0, 0.0) == {"1": -4.0}

    @pytest.mark.parametrize("median", [0.0, -1.0])
    def test_depth_scaling_refuses_a_non_positive_median_depth(self, median):
        with pytest.raises(ValueError, match="median parent depth is zero"):
            pi.resolve_frequency_cutoffs("depth_scaled", [1], -4.0, median)

    def test_empty_k_list_gives_an_empty_mapping(self):
        assert pi.resolve_frequency_cutoffs("depth_scaled", [], -4.0, 100.0) == {}


# =========================================================================== #
# main(): the whole stage, offline
# =========================================================================== #

@pytest.mark.integration
@pytest.mark.requires_prody
class TestMain:
    """A complete stage-A run with a fake aligner, a synthetic query-numbered
    structure and the leakage stage off, so it is fully offline and ~1 s."""

    @pytest.fixture()
    def workspace(self, tmp_path, fake_mafft):
        root = tmp_path / "ws"
        guide = make_guide(root / "data")
        deep = write_fasta(
            root / "data" / "deep.fasta",
            [(f"DEEP{i:03d}", _divergent(QUERY_PROTEIN, 2 + i, i)) for i in range(6)],
        )
        pdb = root / "data" / "synthstruct.pdb"
        pdb.parent.mkdir(parents=True, exist_ok=True)
        pdb.write_text(build_pdb(full_length_query_pdb_atoms("A")), encoding="utf-8")
        inputs = root / "inputs"

        def run(extra=()):
            extra = [str(item) for item in extra]
            argv = [
                "--guide-path", str(guide["path"]),
                "--deep-fasta", str(deep),
                "--inputs-dir", str(inputs),
                "--structure", str(pdb),
                "--mafft-bin", str(fake_mafft("passthrough")),
                "--no-leakage-check", "--no-purge-leakage",
            ]
            # --no-extra-structure is store_true, so it can never be undone by a
            # later --extra-structure; only add it when the caller wants neither.
            if "--extra-structure" not in extra:
                argv.append("--no-extra-structure")
            return pi.main(argv + extra)

        return {"root": root, "guide": guide, "deep": deep, "pdb": pdb,
                "inputs": inputs, "run": run}

    def read_manifest(self, workspace):
        return json.loads((workspace["inputs"] / "inputs_manifest.json").read_text())

    def test_full_run_writes_every_product(self, workspace):
        assert workspace["run"]() == 0
        inputs = workspace["inputs"]
        assert (inputs / "inputs_manifest.json").exists()
        for label in LINEAGE_ORDER:
            assert (inputs / "query" / f"{label}_query.fasta").exists()
            assert (inputs / "msa" / f"msa_{label}.fasta").exists()
        assert (inputs / "query" / "query_report.tsv").exists()
        assert (inputs / "msa" / "msa_report.json").exists()
        assert (inputs / "msa" / "deep_msa_566.fasta").exists()
        assert (inputs / "structure" / "structure_report.json").exists()
        assert (inputs / "structure" / "synthstruct_chainA_qnum.pdb").exists()
        assert (inputs / "structure" / "synthstruct_trimer_qnum.pdb").exists()
        assert (inputs / "frequency" / "frequency_report.json").exists()

    def test_parent_map_is_the_corrected_ladder(self, workspace):
        """K descends from J.2.4, NOT from J.2_int.  Compared against a literal on
        purpose: reading the map out of the module could not notice a regression."""
        workspace["run"]()
        manifest = self.read_manifest(workspace)
        assert manifest["parent_map"] == {
            "J_int": "G.1", "J.2_int": "J_int", "J.2.4": "J.2_int", "K": "J.2.4",
        }
        assert manifest["parent_map_preset"] == "clade_evidence"
        assert manifest["parent_map"]["K"] != "J.2_int"

    def test_frequency_files_are_written_for_every_evaluable_lineage_only(self, workspace):
        workspace["run"]()
        freq = workspace["inputs"] / "frequency"
        written = sorted(p.name for p in freq.glob("*_parent_frequency.txt"))
        assert written == [
            "J.2.4_parent_frequency.txt", "J.2_int_parent_frequency.txt",
            "J_int_parent_frequency.txt", "K_parent_frequency.txt",
        ]
        # G.1 is input-only: it has no basal panel, so no frequency file.
        assert not (freq / "G.1_parent_frequency.txt").exists()
        manifest = self.read_manifest(workspace)
        assert manifest["input_only_lineages"] == ["G.1"]
        assert "G.1" not in manifest["frequency_index"]

    def test_frequency_index_records_the_edge_each_file_belongs_to(self, workspace):
        workspace["run"]()
        index = self.read_manifest(workspace)["frequency_index"]
        assert set(index) == {"J_int", "J.2_int", "J.2.4", "K"}
        assert index["K"]["parent_lineage"] == "J.2.4"
        assert index["K"]["parent_token"] == "J24"
        assert index["K"]["is_primary_parent"] is True
        assert index["K"]["child_lineage"] == "K"
        assert Path(index["K"]["frequency_path"]).exists()
        assert Path(index["K"]["meta_path"]).exists()

    def test_manifest_hoists_the_reversion_settings_to_the_top_level(self, workspace):
        workspace["run"]()
        manifest = self.read_manifest(workspace)
        assert manifest["drop_parent_reversions"] is True
        assert manifest["parent_freq_max"] == 0.95
        assert set(manifest["n_parent_reversion_mutants_dropped"]) == {
            "J_int", "J.2_int", "J.2.4", "K"
        }
        assert manifest["seed"] == 20260805
        assert manifest["guide_md5"] == common.md5_file(Path(workspace["guide"]["path"]))

    def test_no_drop_parent_reversions_is_recorded(self, workspace):
        workspace["run"](["--no-drop-parent-reversions"])
        manifest = self.read_manifest(workspace)
        assert manifest["drop_parent_reversions"] is False
        assert manifest["frequency"]["K"]["drop_parent_reversions"] is False
        assert manifest["frequency_index"]["K"]["drop_parent_reversions"] is False

    def test_frequency_cutoffs_are_depth_scaled_per_lineage(self, workspace):
        workspace["run"](["--frequency-cutoff-k", "1,5"])
        manifest = self.read_manifest(workspace)
        entry = manifest["frequency"]["K"]
        assert entry["median_mapped_depth"] == 100.0
        assert entry["frequency_cutoff_mode"] == "depth_scaled"
        assert entry["frequency_cutoffs"]["1"] == pytest.approx(-2.0)
        assert entry["frequency_cutoffs"]["5"] == pytest.approx(math.log10(5 / 100))

    def test_fixed_cutoff_mode(self, workspace):
        workspace["run"](["--frequency-cutoff-mode", "fixed", "--frequency-cutoff", "-3.5"])
        entry = self.read_manifest(workspace)["frequency"]["K"]
        assert entry["frequency_cutoffs"] == {"1": -3.5}

    def test_leakage_stage_off_is_recorded_as_such(self, workspace, capsys):
        workspace["run"]()
        assert "Leakage stage DISABLED" in capsys.readouterr().out
        assert self.read_manifest(workspace)["leakage"] == {
            "enabled": False, "status": "SKIPPED"
        }

    def test_only_lineage_pulls_in_the_parent_it_needs(self, workspace):
        workspace["run"](["--only-lineage", "K"])
        inputs = workspace["inputs"]
        # K and its parent J.2.4 are prepared; the rest are not.
        assert (inputs / "query" / "K_query.fasta").exists()
        assert (inputs / "query" / "J.2.4_query.fasta").exists()
        assert not (inputs / "query" / "G.1_query.fasta").exists()
        assert not (inputs / "query" / "J_int_query.fasta").exists()
        manifest = self.read_manifest(workspace)
        assert sorted(manifest["queries"]) == ["J.2.4", "K"]
        # Only the selected lineage gets a frequency file.
        assert set(manifest["frequency_index"]) == {"K"}
        # The parent map is still resolved against the FULL guide.
        assert manifest["parent_map"]["J_int"] == "G.1"

    def test_only_lineage_falls_back_to_an_available_anchor(self, workspace):
        """--msa-anchor defaults to G.1; when G.1 is not prepared the anchor falls
        back to the alphabetically first prepared lineage rather than crashing."""
        workspace["run"](["--only-lineage", "K"])
        assert self.read_manifest(workspace)["anchor_lineage"] == "J.2.4"

    def test_unknown_only_lineage_is_refused(self, workspace):
        with pytest.raises(ValueError, match="names labels absent from the guide"):
            workspace["run"](["--only-lineage", "Z.9"])

    def test_parent_map_override_reaches_the_frequency_index(self, workspace):
        workspace["run"](["--parent-map", "K=J.2_int"])
        manifest = self.read_manifest(workspace)
        assert manifest["parent_map"]["K"] == "J.2_int"
        assert manifest["frequency_index"]["K"]["parent_lineage"] == "J.2_int"
        assert manifest["frequency_index"]["K"]["parent_token"] == "J2int"

    def test_sensitivity_edge_writes_a_second_independently_named_file(self, workspace):
        workspace["run"](["--sensitivity-parent-map", "K=J.2_int"])
        freq = workspace["inputs"] / "frequency"
        assert (freq / "K_parent_frequency.txt").exists()
        assert (freq / "K_parentJ2int_frequency.txt").exists()
        assert (freq / "K_parentJ2int_frequency_meta.tsv").exists()

        manifest = self.read_manifest(workspace)
        assert manifest["sensitivity_parent_map"] == {"K": "J.2_int"}
        index = manifest["frequency_index"]
        assert index["K"]["is_primary_parent"] is True
        assert index["K_parentJ2int_frequency"]["is_primary_parent"] is False
        assert index["K_parentJ2int_frequency"]["parent_lineage"] == "J.2_int"
        assert index["K_parentJ2int_frequency"]["child_lineage"] == "K"
        # The primary map is untouched by the sensitivity pass.
        assert manifest["parent_map"]["K"] == "J.2.4"

    def test_sensitivity_preset_derives_only_the_contested_edge(self, workspace):
        workspace["run"](["--sensitivity-preset", "brief_as_stated"])
        manifest = self.read_manifest(workspace)
        assert manifest["sensitivity_parent_map"] == {"K": "J.2_int"}
        assert manifest["sensitivity_preset"] == "brief_as_stated"
        # J_int, J.2_int and J.2.4 agree between the presets, so no duplicate file.
        assert not (workspace["inputs"] / "frequency" / "J.2.4_parentJint_frequency.txt").exists()

    def test_redundant_sensitivity_edge_is_skipped(self, workspace, capsys):
        workspace["run"](["--sensitivity-parent-map", "K=J.2.4"])
        printed = capsys.readouterr().out
        assert "equals the primary map; skipping" in printed
        manifest = self.read_manifest(workspace)
        assert manifest["sensitivity_parent_map"] == {}
        assert not (workspace["inputs"] / "frequency" / "K_parentJ24_frequency.txt").exists()

    def test_sensitivity_edge_naming_an_unknown_child_is_refused(self, workspace):
        with pytest.raises(ValueError, match="unknown child lineage"):
            workspace["run"](["--sensitivity-parent-map", "Q=J.2.4"])

    def test_sensitivity_edge_naming_an_unknown_parent_is_refused(self, workspace):
        with pytest.raises(ValueError, match="unknown parent lineage"):
            workspace["run"](["--sensitivity-parent-map", "K=Q.9"])

    def test_sensitivity_edge_on_an_input_only_lineage_is_refused(self, workspace):
        with pytest.raises(ValueError, match="no primary parent"):
            workspace["run"](["--sensitivity-parent-map", "G.1=K"])

    def test_msa_row_zero_is_the_lineage_query_ungapped(self, workspace):
        workspace["run"]()
        for label in LINEAGE_ORDER:
            rows = list(common.read_fasta(workspace["inputs"] / "msa" / f"msa_{label}.fasta"))
            header, sequence = rows[0]
            assert header == f"HA{common.lineage_tag(label)}"
            assert "-" not in sequence
            assert sequence == QUERY_PROTEIN
            assert len({len(seq) for _h, seq in rows}) == 1

    def test_structure_report_records_offset_and_coverage(self, workspace):
        workspace["run"]()
        report = json.loads(
            (workspace["inputs"] / "structure" / "structure_report.json").read_text()
        )
        assert set(report) == {"primary"}
        assert report["primary"]["offset"] == 0
        assert report["primary"]["n_covered"] == QUERY_LENGTH
        assert report["primary"]["coverage_fraction"] == 1.0
        assert report["primary"]["construct_truncated_at_author_resnum"] is None

    def test_extra_structure_is_prepared_when_present(self, workspace, tmp_path):
        extra = tmp_path / "extra_model.cif"
        write_min_cif(extra, ca_rows(QUERY_PROTEIN))
        workspace["run"](["--extra-structure", str(extra)])
        report = json.loads(
            (workspace["inputs"] / "structure" / "structure_report.json").read_text()
        )
        assert set(report) == {"primary", "extra"}
        assert Path(report["extra"]["monomer"]["path"]).name == "model_J241_chainA_qnum.pdb"

    def test_a_missing_extra_structure_is_simply_skipped(self, workspace, tmp_path):
        workspace["run"](["--extra-structure", str(tmp_path / "does_not_exist.cif")])
        report = json.loads(
            (workspace["inputs"] / "structure" / "structure_report.json").read_text()
        )
        assert set(report) == {"primary"}

    def test_manifest_args_block_round_trips_paths_as_strings(self, workspace):
        workspace["run"]()
        args = self.read_manifest(workspace)["args"]
        assert args["inputs_dir"] == str(workspace["inputs"].resolve())
        assert args["parent_min_depth"] == 50
        assert args["drop_parent_reversions"] is True
        assert args["leakage_check"] is False

    def test_queries_block_omits_the_protein_but_keeps_its_md5(self, workspace):
        workspace["run"]()
        queries = self.read_manifest(workspace)["queries"]
        assert "protein" not in queries["K"]
        assert queries["K"]["md5"] == common.md5_text(QUERY_PROTEIN)
        assert queries["K"]["length"] == QUERY_LENGTH
        assert queries["K"]["prot_token"] == "HAK"

    def test_second_run_reuses_the_msa_cache(self, workspace, capsys):
        workspace["run"]()
        capsys.readouterr()
        workspace["run"]()
        assert "MSA cache hit" in capsys.readouterr().out

    def test_force_rebuilds_the_msa(self, workspace, capsys):
        workspace["run"]()
        capsys.readouterr()
        workspace["run"](["--force"])
        printed = capsys.readouterr().out
        assert "MSA cache hit" not in printed
        assert "@> mafft:" in printed

    def test_empty_guide_is_refused(self, tmp_path, workspace):
        empty = tmp_path / "empty_guide.csv"
        empty.write_text("month,fasta,reference\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No usable rows in guide"):
            workspace["run"](["--guide-path", str(empty)])

    def test_missing_parent_panel_names_the_child(self, workspace):
        Path(workspace["guide"]["panels"]["J.2.4"]).unlink()
        with pytest.raises(FileNotFoundError, match="Parent panel for K not found"):
            workspace["run"](["--only-lineage", "K"])

    def test_leakage_stage_is_skipped_when_only_parents_are_selected(self, workspace, capsys):
        """G.1 is never an evaluation target, so purging the deep set against its
        panel would cost MSA depth for nothing.  Note the leakage flags are left
        at their defaults (both ON) here: the skip must come from there being no
        target, not from the stage being disabled."""
        workspace["run"](["--only-lineage", "G.1"])
        printed = capsys.readouterr().out
        assert "Leakage stage skipped: no evaluation targets selected" in printed
        manifest = self.read_manifest(workspace)
        assert manifest["leakage"] == {"enabled": False, "status": "SKIPPED"}
        assert manifest["frequency_index"] == {}
        assert (workspace["inputs"] / "msa" / "msa_G.1.fasta").exists()


@pytest.mark.integration
@pytest.mark.requires_prody
class TestMainStructureIsStrippedEndToEnd:
    """The construct check, through ``main`` rather than through
    ``prepare_structure``: the flag path and the stem derivation are part of the
    contract too (``6WXB-assembly1.cif`` -> ``6WXB_chainA_qnum.pdb``)."""

    def test_linker_foldon_and_his_tag_never_reach_the_inputs_tree(self, tmp_path, fake_mafft):
        root = tmp_path / "ws"
        guide = make_guide(root / "data")
        deep = write_fasta(root / "data" / "deep.fasta",
                           [(f"D{i}", _divergent(QUERY_PROTEIN, 2 + i, i)) for i in range(4)])
        cif = write_min_cif(root / "data" / "6WXB-assembly1.cif", ca_rows(CONSTRUCT_CHAIN))
        inputs = root / "inputs"
        assert pi.main([
            "--guide-path", str(guide["path"]), "--deep-fasta", str(deep),
            "--inputs-dir", str(inputs), "--structure", str(cif), "--no-extra-structure",
            "--mafft-bin", str(fake_mafft("passthrough")),
            "--no-leakage-check", "--no-purge-leakage",
        ]) == 0

        mono = inputs / "structure" / "6WXB_chainA_qnum.pdb"
        assert mono.exists()
        assert read_pdb_ca_resnums(mono) == list(range(1, 61))
        emitted = read_pdb_ca_sequence(mono)
        assert emitted == CORE
        for motif in EXPECTED_CONSTRUCT_MOTIFS.values():
            assert motif not in emitted

        report = json.loads((inputs / "structure" / "structure_report.json").read_text())
        assert report["primary"]["construct_truncated_at_author_resnum"] == 61
        assert report["primary"]["uncovered_runs"] == [[61, QUERY_LENGTH]]


@pytest.mark.integration
@pytest.mark.requires_prody
@pytest.mark.requires_blast
class TestLeakagePurgeIsWiredIn:
    """The purge rewrites ``msa/msa_<key>.fasta`` in place, so every md5 and row
    count recorded before it is stale.  The module refreshes them afterwards --
    which is the difference between a manifest that describes what ESCOTT read
    and one that describes what mafft produced.  This is that wiring, not the
    purge algorithm (``test_leakage_check.py`` owns that)."""

    def test_purge_rewrites_the_msa_and_the_manifest_follows(self, tmp_path, fake_mafft):
        root = tmp_path / "ws"
        guide = make_guide(root / "data")
        # One deep row is a byte-identical copy of a K panel record: a leak that
        # the purge must remove from K's alignment and only from K's.
        leaked = _panel_records(QUERY_PROTEIN, {5: {"V": 20}}, PANEL_N_RECORDS, "K__")[0][1]
        deep_rows = [(f"D{i}", _divergent(QUERY_PROTEIN, 12 + i, i)) for i in range(6)]
        deep_rows.insert(3, ("LEAKED_ROW", leaked))
        deep = write_fasta(root / "data" / "deep.fasta", deep_rows)
        pdb = root / "data" / "s.pdb"
        pdb.write_text(build_pdb(full_length_query_pdb_atoms("A")), encoding="utf-8")
        inputs = root / "inputs"

        assert pi.main([
            "--guide-path", str(guide["path"]), "--deep-fasta", str(deep),
            "--inputs-dir", str(inputs), "--structure", str(pdb), "--no-extra-structure",
            "--mafft-bin", str(fake_mafft("passthrough")),
            "--only-lineage", "K",
            # This alignment is 8 rows deep, so the production floors would
            # (correctly) refuse it; they are relaxed to let the WIRING be tested.
            "--leakage-min-depth-after", "1", "--leakage-max-removed-fraction", "1.0",
        ]) == 0

        manifest = json.loads((inputs / "inputs_manifest.json").read_text())
        entry = manifest["lineage_msas"]["K"]
        assert entry["purged"] is True
        assert entry["n_rows_before_purge"] == 8
        assert entry["n_rows_removed_by_purge"] == 1
        assert entry["n_rows"] == 7
        purged = inputs / "msa" / "msa_K.fasta"
        assert entry["md5"] == common.md5_file(purged), "manifest md5 is the PRE-purge one"
        assert Path(entry["prepurge_path"]) == inputs / "msa" / "msa_K_prepurge.fasta"
        assert Path(entry["drop_manifest_path"]).exists()

        rows = list(common.read_fasta(purged))
        assert len(rows) == 7
        assert rows[0] == ("HAK", QUERY_PROTEIN), "the query row must never be purged"
        assert "LEAKED_ROW" not in {header for header, _seq in rows}
        assert len(list(common.read_fasta(inputs / "msa" / "msa_K_prepurge.fasta"))) == 8

        # J.2.4 is a frequency parent, not an evaluation target: untouched.
        parent_entry = manifest["lineage_msas"]["J.2.4"]
        assert "purged" not in parent_entry
        assert parent_entry["n_rows"] == 8

        assert manifest["leakage"]["enabled"] is True
        assert manifest["leakage"]["purge"] is True
        assert manifest["leakage"]["purges"]["K"]["depth_before"] == 8
        assert manifest["leakage"]["purges"]["K"]["depth_after"] == 7


@pytest.mark.requires_real_data
@pytest.mark.requires_prody
class TestRealStructure:
    """The real 6WXB assembly against the real 566-aa G.1 reference.

    Opt-in (``--run-slow``) because it reads the production guide, but it is the
    only test that proves the +16 mature-HA1 offset and the real coverage.
    """

    @pytest.fixture(scope="class")
    def real_g1_protein(self):
        guide = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"
        rows = common.read_guide_rows(guide)
        row = next(r for r in rows if r["label"] == "G.1")
        payload = common.load_reference_cds(Path(row["reference_path"]), "G.1")
        return payload["protein"]

    def test_reference_is_566_aa(self, real_g1_protein):
        assert len(real_g1_protein) == REAL_6WXB_QUERY_LENGTH
        assert set(real_g1_protein) <= set(AA20)

    def test_offset_coverage_and_absence_of_construct(self, tmp_path, real_g1_protein):
        cif = REPO_ROOT / "Sequences" / "6WXB-assembly1.cif"
        if not cif.exists():
            pytest.skip(f"{cif} is not present")
        report = pi.prepare_structure(
            cif, "A", "auto", real_g1_protein, tmp_path / "structure", "6WXB", 0.60
        )
        assert report["offset"] == REAL_6WXB_OFFSET
        assert report["offset_identity"] == pytest.approx(REAL_6WXB_IDENTITY)
        assert report["offset_matched_residues"] == REAL_6WXB_MATCHED
        # 6WXB's coordinates stop before the linker, so there is nothing to strip
        # -- which is exactly why the motif check must not be trusted to be a
        # no-op: it is the guard, not the observation.
        assert report["construct_truncated_at_author_resnum"] is None
        assert report["monomer"]["per_chain"]["A"] == {
            "n_residues": REAL_6WXB_N_COVERED,
            "resnum_min": REAL_6WXB_RESNUM_MIN,
            "resnum_max": REAL_6WXB_RESNUM_MAX,
            "gaps": REAL_6WXB_GAPS,
        }
        assert report["n_covered"] == REAL_6WXB_N_COVERED
        assert report["uncovered_runs"] == REAL_6WXB_UNCOVERED
        assert sorted(report["trimer"]["per_chain"]) == ["A", "B", "C"]

        emitted = read_pdb_ca_sequence(Path(report["monomer"]["path"]))
        for motif in EXPECTED_CONSTRUCT_MOTIFS.values():
            assert motif not in emitted
        assert max(read_pdb_ca_resnums(Path(report["monomer"]["path"]))) <= len(real_g1_protein)
