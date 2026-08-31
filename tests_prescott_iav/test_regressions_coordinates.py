#!/usr/bin/env python3
"""Coordinate, indexing and identity regressions.

WHY THIS FILE EXISTS
====================
A residue index passes through seven frames before it becomes a number in a
table:

===========================  ==================================================
lineage nucleotide CDS       1701 nt, 1-based codons
translated HA0 protein       566 aa, INCLUDING the 16-residue signal peptide
6WXB author numbering        mature-HA convention; chain A spans author 9..501
query ("HA0") numbering      author + 16, i.e. 25..517, with a 342..349 gap
jet.res rows                 ``pos`` 1..566, one row per MSA column, no gaps
ESCOTT matrix columns        1..L, labelled ``<WT><pos>``, WT cell = NA
diversity-panel columns      594 gapped columns of MATURE HA (HA0 17 -> col 1)
===========================  ==================================================

Every arrow between two of those is a place an off-by-one is invisible: a
one-residue shift still yields a full table of plausible numbers.  The tests
below pin each conversion against ground truth derived WITHOUT the code under
test -- a literal, a hand-checked offset, or the *other* half of the pipeline
(``Functions_HuggingFace``), which is where the evaluation numbers come from and
therefore the only meaningful authority on the panel frame.

Sections
--------
1.  REGRESSION: ``jet_surrogate``'s structure cache compared PATHS, not content.
2.  REGRESSION: a frequency file in the wrong frame silently turned every
    PRESCOTT variant into a numerical clone of the ESCOTT baseline.
3.  IDENTITY: dotted lineage labels through keys, tokens, filenames and back.
4.  Real-data coordinate invariants (opt-in: ``--run-slow``).

Nothing here duplicates ``test_prepare_inputs.py`` / ``test_jet_surrogate.py`` /
``test_run_escott.py``: those cover each function in isolation, this file covers
the seams BETWEEN them and the two defects found at those seams.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from prescott_iav import common as CM
from prescott_iav import constants as K
from prescott_iav import jet_surrogate as J
from prescott_iav import run_escott as R
from tests_prescott_iav import conftest as C


REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Real-data landmarks.  Every one is a literal, verified by an opt-in test
# below; none is produced by the code under test.
# --------------------------------------------------------------------------- #

REAL_GUIDE = REPO_ROOT / "Sequences" / "IAV_lineage_guide.csv"
REAL_6WXB = REPO_ROOT / "Sequences" / "6WXB-assembly1.cif"
REAL_REFERENCE_DIR = REPO_ROOT / "Sequences" / "IAV_lineage_files"

HA0_LENGTH = 566
"""All five lineage CDSs are 1701 nt = 567 codons, the last a stop."""

SIGNAL_PEPTIDE_LENGTH = 16
"""``MKTIIALSNILCLVFA`` (G.1) / ``MKAIIALSNILCLVFA`` (J/K).  Mature HA1 begins at
HA0 position 17 with Q -- N in K, which carries Q17N."""

SIX_WXB_OFFSET = 16
"""Author resnum + 16 = query position.  Numerically equal to the signal peptide
length because 6WXB is numbered in the mature-HA convention, but the tests below
derive it from the residues rather than assuming the coincidence."""

SIX_WXB_AUTHOR_RANGE = (9, 501)
SIX_WXB_QUERY_RANGE = (25, 517)
SIX_WXB_N_COVERED = 485
SIX_WXB_UNCOVERED_RUNS = [[1, 24], [342, 349], [518, 566]]
"""1..24 = signal peptide plus the first 8 mature residues; 342..349 = the
disordered HA1/HA2 cleavage loop; 518..566 = the transmembrane anchor and
cytoplasmic tail, absent from the ectodomain construct."""

PANEL_ALIGNMENT_COLUMNS = 594
PANEL_FIRST_MAPPED = (17, 1)
"""HA0 position 17 (the mature N-terminus) is alignment column 1."""

SIGNAL_PEPTIDE_SHIFT = 16
"""The offset between HA0 numbering and mature (H3) numbering -- the single most
likely coordinate accident in this pipeline."""


# --------------------------------------------------------------------------- #
# Local fixtures.  Deliberately not imported from the other test modules: this
# file must stay runnable if one of them is refactored.
# --------------------------------------------------------------------------- #

@pytest.fixture()
def parsed_escott_matrix(tmp_path: Path) -> pd.DataFrame:
    """The fixture ESCOTT product, parsed by the module under test."""
    path = C.write_escott_normpred(tmp_path / f"{C.QUERY_HEADER}_normPred_evolCombi.txt")
    return R.read_escott_matrix(path)


@pytest.fixture()
def stub_dssp_runs(monkeypatch):
    """Replace :func:`jet_surrogate.dssp_runs`.

    Every synthetic structure in this suite is CA-only and mkdssp needs a full
    backbone (N, CA, C, O).  The real DSSP path is covered in
    ``test_jet_surrogate.py``; here it is irrelevant to what is being tested.
    """
    def _fake(pdb_path):
        return {pos: ("H", 1) for pos in range(1, C.QUERY_LENGTH + 1)}

    monkeypatch.setattr(J, "dssp_runs", _fake)
    monkeypatch.setattr(J, "dssp_version", lambda: "stub 0.0")


class _Completed:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def fake_escott(monkeypatch):
    """Stand in for the escott subprocess, reproducing only its products.

    escott derives ``prot`` from the first FASTA header (escott.py:76) and writes
    ``<prot>.fasta`` and ``<prot>_normPred_evolCombi.txt`` as bare CWD filenames.
    """
    def _run(cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        header = (cwd / cmd[1]).read_text(encoding="utf-8").splitlines()[0][1:]
        prot = CM.escott_prot_token(header)
        (cwd / f"{prot}.fasta").write_text(
            f">{prot}\n{C.QUERY_PROTEIN}\n", encoding="utf-8"
        )
        C.write_escott_normpred(cwd / f"{prot}_normPred_evolCombi.txt")
        return _Completed(0, "escott finished", "")

    monkeypatch.setattr(R.subprocess, "run", _run)


def _shift_frequency_frame(frequencies: Dict[str, float], shift: int) -> Dict[str, float]:
    """Rewrite ``{wt}{pos}{mut}`` keys into a frame ``shift`` residues away."""
    return {
        f"{mutant[0]}{int(mutant[1:-1]) - shift}{mutant[-1]}": value
        for mutant, value in frequencies.items()
    }


# =========================================================================== #
# 1. REGRESSION -- the jet cache compared structure PATHS, not structure content
# =========================================================================== #

def _query_frame_pdb(path: Path, shift: int = 0, scale: float = 1.0) -> Path:
    """A CA-only backbone in query numbering, optionally renumbered or squashed."""
    atoms = [
        (name, resname, chain, resseq + shift, x * scale, y * scale, z * scale)
        for name, resname, chain, resseq, x, y, z in C.full_length_query_pdb_atoms("A")
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(C.build_pdb(atoms), encoding="utf-8")
    return path


@pytest.mark.integration
@pytest.mark.requires_prody
@pytest.mark.requires_scipy
@pytest.mark.requires_freesasa
class TestJetCacheKeysOnStructureContent:
    """``--pdb`` names a path prepare_inputs REWRITES, so the cache must hash it.

    THE DEFECT
    ----------
    ``jet_surrogate.main``'s cache short-circuit compared the *path strings* of
    ``--pdb`` and ``--context-pdb`` against the manifest::

        structure_matches = (
            str(cached_structure.get("pdb") or "") == str(args.pdb or "")
            and str(cached_structure.get("context_pdb") or "")
            == str(args.context_pdb or args.pdb or "")
        )

    Its own comment says the guard exists because "``--out-jet`` has one name per
    lineage, so a second run with a different structure writes to the same path
    and would otherwise be served the previous structure's table verbatim" -- but
    a path string is exactly what does NOT change in that scenario.
    ``prepare_inputs.py`` writes ``inputs/structure/<stem>_chain<C>_qnum.pdb`` and
    rewrites it, uncached, on every run, so the path is invariant under

    * a changed ``--structure-offset`` (which renumbers every residue),
    * an edited, re-downloaded or re-prepared source structure,
    * two ``--structure`` files whose stem token collides -- and it is only a
      token: ``args.structure.stem.split("-")[0].split("_")[0]`` maps BOTH
      ``Sequences/6WXB.cif`` and ``Sequences/6WXB-assembly1.cif`` to ``6WXB``.

    Worse than staleness: the cache hit returns before ``build_jet_table`` is
    called, so it also skips that function's own frame check -- the one that
    refuses a structure matching the query at under 60% of covered positions.  A
    structure in the WRONG RESIDUE FRAME was therefore accepted silently with
    exit code 0, and every ``pc``/``cv``/``ss`` value in the served table belonged
    to different residues than the ones ESCOTT would go on to score.

    THE FIX
    -------
    ``build_jet_table`` records ``structure.pdb_md5`` and
    ``structure.context_pdb_md5``; the cache predicate compares them.
    """

    def _argv(self, tmp_path: Path, pdb: Optional[Path], out: Path) -> List[str]:
        msa = C.write_fasta(
            tmp_path / "msa.fasta",
            [(C.QUERY_HEADER, C.QUERY_PROTEIN)]
            + list(zip(C.TINY_MSA_HEADERS[1:], C.TINY_MSA_ROWS[1:])),
        )
        argv = [
            "--msa", str(msa),
            "--out-jet", str(out / "jet.res"),
            "--out-manifest", str(out / "jet_manifest.json"),
            "--trace-definition", "direct",
            "--max-zero-trace-fraction", "1.0",
        ]
        if pdb is not None:
            argv += ["--pdb", str(pdb)]
        return argv

    def test_a_rewritten_structure_at_the_same_path_is_a_cache_miss(
        self, tmp_path, stub_dssp_runs
    ):
        """The defect end to end: same path, different (wrong-frame) coordinates."""
        out = tmp_path / "out"
        out.mkdir()
        live = tmp_path / "structure" / "model_chainA_qnum.pdb"
        _query_frame_pdb(live)
        argv = self._argv(tmp_path, live, out)

        assert J.main(list(argv)) == 0
        first_md5 = CM.md5_file(out / "jet.res")

        # prepare_inputs.py rewrites this exact path with a differently numbered
        # structure (e.g. --structure-offset out by 30).  Before the fix this was a
        # cache hit returning 0; now the frame check gets to see it.
        _query_frame_pdb(live, shift=30)
        with pytest.raises(ValueError, match="not in the query frame"):
            J.main(list(argv))
        assert CM.md5_file(out / "jet.res") == first_md5, (
            "the refusal must not leave a half-written table behind"
        )

    def test_a_same_path_coordinate_change_is_also_a_miss(self, tmp_path, stub_dssp_runs):
        """Not only numbering: moved atoms change circular variance and RSA."""
        out = tmp_path / "out"
        out.mkdir()
        live = tmp_path / "structure" / "model_chainA_qnum.pdb"
        _query_frame_pdb(live)
        argv = self._argv(tmp_path, live, out)
        assert J.main(list(argv)) == 0
        before = CM.md5_file(out / "jet.res")

        _query_frame_pdb(live, scale=0.35)  # same residues, packed 3x tighter
        assert J.main(list(argv)) == 0
        assert CM.md5_file(out / "jet.res") != before, (
            "a structure compressed to a third of its size must change cv/RSA"
        )

    def test_identical_inputs_are_still_a_cache_hit(self, tmp_path, stub_dssp_runs, capsys):
        """The fix must not turn every rerun into a rebuild."""
        out = tmp_path / "out"
        out.mkdir()
        pdb = _query_frame_pdb(tmp_path / "structure" / "model_chainA_qnum.pdb")
        argv = self._argv(tmp_path, pdb, out)
        assert J.main(list(argv)) == 0
        mtime = (out / "jet.res").stat().st_mtime_ns
        capsys.readouterr()
        assert J.main(list(argv)) == 0
        assert "cache hit" in capsys.readouterr().out
        assert (out / "jet.res").stat().st_mtime_ns == mtime

    def test_the_manifest_records_both_structure_hashes(self, tmp_path, stub_dssp_runs):
        out = tmp_path / "out"
        out.mkdir()
        pdb = _query_frame_pdb(tmp_path / "structure" / "model_chainA_qnum.pdb")

        context = tmp_path / "structure" / "model_trimer_qnum.pdb"
        atoms: List[Tuple[str, str, str, int, float, float, float]] = []
        for index, chain in enumerate("ABC"):
            for name, resname, _c, resseq, x, y, z in C.full_length_query_pdb_atoms(chain):
                atoms.append((name, resname, chain, resseq, x, y + index * 14.0, z))
        context.write_text(C.build_pdb(atoms), encoding="utf-8")

        argv = self._argv(tmp_path, pdb, out) + ["--context-pdb", str(context)]
        assert J.main(argv) == 0
        structure = json.loads((out / "jet_manifest.json").read_text())["structure"]
        assert structure["pdb_md5"] == CM.md5_file(pdb)
        assert structure["context_pdb_md5"] == CM.md5_file(context)

    def test_a_rewritten_context_alone_is_a_miss(self, tmp_path, stub_dssp_runs):
        """The SASA/CV environment is half the structural signal."""
        out = tmp_path / "out"
        out.mkdir()
        pdb = _query_frame_pdb(tmp_path / "structure" / "model_chainA_qnum.pdb")
        context = tmp_path / "structure" / "model_trimer_qnum.pdb"

        def _write_context(gap: float) -> None:
            atoms: List[Tuple[str, str, str, int, float, float, float]] = []
            for index, chain in enumerate("ABC"):
                for name, resname, _c, resseq, x, y, z in C.full_length_query_pdb_atoms(chain):
                    atoms.append((name, resname, chain, resseq, x, y + index * gap, z))
            context.write_text(C.build_pdb(atoms), encoding="utf-8")

        _write_context(40.0)  # neighbours far away
        argv = self._argv(tmp_path, pdb, out) + ["--context-pdb", str(context)]
        assert J.main(list(argv)) == 0
        before = CM.md5_file(out / "jet.res")

        _write_context(3.0)  # neighbours packed against the target chain
        assert J.main(list(argv)) == 0
        assert CM.md5_file(out / "jet.res") != before

    def test_a_manifest_written_before_the_hashes_existed_is_a_miss(
        self, tmp_path, stub_dssp_runs
    ):
        """Old trees must recompute, never be trusted."""
        out = tmp_path / "out"
        out.mkdir()
        pdb = _query_frame_pdb(tmp_path / "structure" / "model_chainA_qnum.pdb")
        argv = self._argv(tmp_path, pdb, out)
        assert J.main(list(argv)) == 0

        manifest_path = out / "jet_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["structure"].pop("pdb_md5")
        manifest["structure"].pop("context_pdb_md5", None)
        manifest_path.write_text(json.dumps(manifest))
        (out / "jet.res").unlink()

        assert J.main(list(argv)) == 0
        assert (out / "jet.res").exists(), "a hash-less manifest must be recomputed"

    def test_a_sequence_only_run_has_no_hashes_and_still_caches(self, tmp_path, capsys):
        out = tmp_path / "out"
        out.mkdir()
        argv = self._argv(tmp_path, None, out)
        assert J.main(list(argv)) == 0
        structure = json.loads((out / "jet_manifest.json").read_text())["structure"]
        # No structure, so no hashes: the historical three-key shape is preserved
        # and the predicate compares None against None on both sides.
        assert structure == {"pdb": None, "context_pdb": None, "covered": 0}
        capsys.readouterr()
        assert J.main(list(argv)) == 0
        assert "cache hit" in capsys.readouterr().out


@pytest.mark.unit
class TestStructureStemTokenIsLossy:
    """Why the path is not an identity: two structures reduce to one filename."""

    def test_two_shipped_structures_collide_on_one_output_name(self):
        """``prepare_inputs`` names its output from a TOKEN, not the filename."""
        def stem_token(name: str) -> str:
            return Path(name).stem.split("-")[0].split("_")[0]

        assert stem_token("6WXB.cif") == stem_token("6WXB-assembly1.cif") == "6WXB"
        # ... so both write inputs/structure/6WXB_chainA_qnum.pdb, and the jet
        # cache can only tell them apart by content.
        assert (
            f"{stem_token('6WXB.cif')}_chainA_qnum.pdb"
            == f"{stem_token('6WXB-assembly1.cif')}_chainA_qnum.pdb"
        )


# =========================================================================== #
# 2. REGRESSION -- a wrong-frame frequency file was silently inert
# =========================================================================== #

@pytest.mark.unit
class TestFrequencyFileMustBeInTheEscottFrame:
    """PRESCOTT's entire contribution vanishes if the frequency frame is wrong.

    THE DEFECT
    ----------
    ``build_log10_frequency_matrix`` drops any record whose position or wild-type
    letter disagrees with the ESCOTT column labels, leaving
    :data:`run_escott.NO_FREQUENCY_SENTINEL`, which every PRESCOTT equation skips.
    With NO record landing, ``prescott_v2_scores`` reduces to the identity:
    measured max |PRESCOTT - ESCOTT| on the 20 x 72 fixture is 5.6e-17, i.e. float
    noise.

    Stage C nonetheless wrote each grid point out under its own name, recorded it
    in ``score_variants.tsv`` and stage D reported it as a separate model with its
    own alpha sweep and its own "best alpha" row.  The only trace was a warning on
    stdout.

    The frame is not hypothetical.  HA is universally published in MATURE (H3)
    numbering while every frame in this pipeline is the 566-aa HA0 translation, so
    a frequency file built one signal peptide out (16 residues) matches nothing at
    all.

    THE FIX
    -------
    ``run_escott.assert_frequency_frame``, called by ``process_lineage`` for every
    parent edge.  A self-consistent tree matches 100% -- ``prepare_inputs``
    derives every ``wt`` letter from the same translated CDS ESCOTT is handed --
    a whole-frame error matches 0%, and a one-residue shift matches only where two
    adjacent residues happen to be equal.  The floor is 50%.
    """

    def test_a_signal_peptide_shift_makes_prescott_a_clone_of_escott(
        self, parsed_escott_matrix
    ):
        """The defect itself, measured.  This is what used to pass silently."""
        good = dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        bad = _shift_frequency_frame(good, SIGNAL_PEPTIDE_SHIFT)

        good_log10, good_report = R.build_log10_frequency_matrix(good, parsed_escott_matrix)
        bad_log10, bad_report = R.build_log10_frequency_matrix(bad, parsed_escott_matrix)
        assert good_report["n_matched"] == len(good)
        assert bad_report["n_matched"] == 0

        baseline = R.escott_to_probability(parsed_escott_matrix).to_numpy()
        with_bad = R.escott_to_probability(
            R.prescott_v2_scores(parsed_escott_matrix, bad_log10, 0.5, -2.0, equation=2)
        ).to_numpy()
        with_good = R.escott_to_probability(
            R.prescott_v2_scores(parsed_escott_matrix, good_log10, 0.5, -2.0, equation=2)
        ).to_numpy()

        assert np.nanmax(np.abs(with_bad - baseline)) < 1e-12, (
            "an unmatched frequency file makes PRESCOTT numerically identical to ESCOTT"
        )
        assert np.nanmax(np.abs(with_good - baseline)) > 1e-3, (
            "the correctly framed file must actually move the scores"
        )

    def test_the_guard_names_the_frame_and_the_signal_peptide(self, parsed_escott_matrix):
        bad = _shift_frequency_frame(
            dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1), SIGNAL_PEPTIDE_SHIFT
        )
        _, report = R.build_log10_frequency_matrix(bad, parsed_escott_matrix)
        with pytest.raises(ValueError) as excinfo:
            R.assert_frequency_frame(
                report, "K", "J.2.4", Path("K_parent_frequency.txt"), parsed_escott_matrix
            )
        message = str(excinfo.value)
        assert "not in the ESCOTT column frame" in message
        assert "0/6" in message
        assert "signal peptide" in message
        assert str(C.QUERY_LENGTH) in message

    def test_a_one_residue_shift_is_caught_too(self, parsed_escott_matrix):
        """The near miss, not just the obvious one."""
        bad = _shift_frequency_frame(dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1), 1)
        _, report = R.build_log10_frequency_matrix(bad, parsed_escott_matrix)
        assert report["n_matched"] < report["n_frequency_records"] / 2
        with pytest.raises(ValueError, match="not in the ESCOTT column frame"):
            R.assert_frequency_frame(
                report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix
            )

    def test_a_correctly_framed_file_passes(self, parsed_escott_matrix):
        good = dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        _, report = R.build_log10_frequency_matrix(good, parsed_escott_matrix)
        R.assert_frequency_frame(report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix)

    def test_an_empty_frequency_file_is_not_a_frame_error(self, parsed_escott_matrix):
        """A parent panel can legitimately yield no mutants; that is not a frame bug."""
        _, report = R.build_log10_frequency_matrix({}, parsed_escott_matrix)
        R.assert_frequency_frame(report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix)

    def test_a_handful_of_strays_is_tolerated(self, parsed_escott_matrix):
        """A third-party file may carry records this reference does not have."""
        frequencies = dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        frequencies["A200K"] = 0.1  # beyond the reference
        frequencies["W3K"] = 0.1    # wild-type letter disagrees
        _, report = R.build_log10_frequency_matrix(frequencies, parsed_escott_matrix)
        assert 0 < report["n_unmatched"] < report["n_matched"]
        R.assert_frequency_frame(report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix)

    def test_the_floor_is_the_documented_fifty_percent(self, parsed_escott_matrix):
        assert R.MIN_FREQUENCY_MATCH_FRACTION == 0.5
        frequencies = dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)  # 6 records, all land
        frequencies.update({f"A{200 + i}K": 0.1 for i in range(6)})  # exactly 6/12
        _, report = R.build_log10_frequency_matrix(frequencies, parsed_escott_matrix)
        assert (report["n_matched"], report["n_frequency_records"]) == (6, 12)
        R.assert_frequency_frame(report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix)

        frequencies["A300K"] = 0.1  # 6/13, just under the floor
        _, report = R.build_log10_frequency_matrix(frequencies, parsed_escott_matrix)
        with pytest.raises(ValueError, match="not in the ESCOTT column frame"):
            R.assert_frequency_frame(
                report, "K", "J.2.4", Path("f.txt"), parsed_escott_matrix
            )


@pytest.mark.integration
class TestProcessLineageRefusesAWrongFrameFrequencyFile:
    """The guard has to bite in the pipeline, not only in isolation."""

    def test_a_shifted_frequency_file_stops_the_lineage(
        self, tmp_path, prepared_inputs_tree, fake_escott
    ):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        C.write_frequency_file(
            inputs_dir / "frequency" / "K_parent_frequency.txt",
            _shift_frequency_frame(
                dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1), SIGNAL_PEPTIDE_SHIFT
            ),
        )
        with pytest.raises(ValueError, match="not in the ESCOTT column frame"):
            R.process_lineage(
                "K", inputs_dir, tmp_path / "escott", tmp_path / "scores",
                parent_lineage="J.2.4", coefficients=(0.5,), equations=(2,),
                frequency_cutoff_ks=(1,),
            )
        # No PRESCOTT matrix was left behind claiming to be a real variant.
        written = sorted(p.name for p in (tmp_path / "scores").glob("*_score_matrix.csv"))
        assert written == ["K_ESCOTT_score_matrix.csv"]

    def test_the_untouched_tree_still_scores(
        self, tmp_path, prepared_inputs_tree, fake_escott
    ):
        rows = R.process_lineage(
            "K", Path(prepared_inputs_tree["inputs_dir"]),
            tmp_path / "escott", tmp_path / "scores",
            parent_lineage="J.2.4", coefficients=(0.5,), equations=(2,),
            frequency_cutoff_ks=(1,),
        )
        assert [row["variant"] for row in rows] == [
            "ESCOTT", "PRESCOTT_eq2_c0p50_k1_parentJ24",
        ]
        assert rows[1]["n_mutants_with_frequency"] > 0

    def test_a_sensitivity_edge_is_checked_too(
        self, tmp_path, prepared_inputs_tree, fake_escott
    ):
        """The alternate parent's file is a separate file and a separate risk."""
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        C.write_frequency_file(
            Path(prepared_inputs_tree["alternate_frequency_path"]),
            _shift_frequency_frame(
                dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_2), SIGNAL_PEPTIDE_SHIFT
            ),
        )
        with pytest.raises(ValueError, match="not in the ESCOTT column frame"):
            R.process_lineage(
                "K", inputs_dir, tmp_path / "escott", tmp_path / "scores",
                parent_lineage="J.2.4", alternate_parents=("J.2_int",),
                coefficients=(0.5,), equations=(2,), frequency_cutoff_ks=(1,),
            )


# =========================================================================== #
# 3. IDENTITY -- dotted labels through keys, tokens, filenames and back
# =========================================================================== #

@pytest.mark.unit
class TestLineageIdentityRoundTrips:
    """A lineage label is transformed five ways; none of them may collide."""

    @pytest.mark.parametrize(
        "label, key, dot_free, token, tag",
        [
            ("G.1", "G.1", "G_1", "G1", "G1"),
            ("J_int", "J_int", "J_int", "Jint", "J"),
            ("J.2_int", "J.2_int", "J_2_int", "J2int", "J2"),
            ("J.2.4", "J.2.4", "J_2_4", "J24", "J24"),
            ("K", "K", "K", "K", "K"),
        ],
    )
    def test_the_five_production_labels(self, label, key, dot_free, token, tag):
        """Literals, not re-derivations: this IS the cross-stage naming contract."""
        assert CM.safe_label(label) == key
        assert K.dot_free_key(label) == dot_free
        assert K.variant_parent_token(label) == token
        assert CM.lineage_tag(label) == tag
        # prescott.py:902 runs os.path.splitext on its -o value: no dot may reach it.
        assert "." not in dot_free
        # The token becomes part of a filename AND of a DataFrame `model` value.
        assert token.isalnum()

    def test_no_two_production_labels_share_any_derived_identity(self):
        labels = list(C.LINEAGE_ORDER)
        for name, function in (
            ("safe_label", CM.safe_label),
            ("dot_free_key", K.dot_free_key),
            ("variant_parent_token", K.variant_parent_token),
            ("lineage_tag", CM.lineage_tag),
        ):
            derived = [function(label) for label in labels]
            assert len(set(derived)) == len(labels), f"{name} collides: {derived}"

    def test_the_driver_and_stage_one_agree_on_every_variant_name(self, driver_module):
        """``build_variant_name`` (stage C) vs ``stage1_variant_name`` (stage D).

        Stage D predicts the filename before stage 1 has ever run; a
        one-character drift makes every requested variant read as missing.
        """
        for parent in C.LINEAGE_ORDER:
            for equation in (1, 2, 3, 5):
                for coefficient in (0.0, 0.25, 0.5, 1.0, 2.0):
                    for k_value in (1, 2, 10):
                        assert R.build_variant_name(
                            equation, coefficient, k_value, parent
                        ) == driver_module.stage1_variant_name(
                            equation, coefficient, k_value, parent
                        )

    def test_the_parent_token_matches_the_drivers_own_token(self, driver_module):
        for label in C.LINEAGE_ORDER:
            assert K.variant_parent_token(label) == driver_module.variant_token(label)

    def test_a_variant_name_never_carries_a_dot_or_a_separator(self):
        for parent in C.LINEAGE_ORDER:
            name = R.build_variant_name(2, 0.5, 1, parent)
            assert "." not in name
            assert "/" not in name
            assert name == CM.safe_label(name)

    def test_the_alternate_frequency_stem_cannot_collide_with_the_primary(self):
        """``K_parent_frequency`` vs ``K_parentJ2int_frequency``: the primary always
        has an underscore exactly where the alternate has its token."""
        for child in C.LINEAGE_ORDER:
            primary = f"{child}_parent_frequency"
            for parent in C.LINEAGE_ORDER:
                alternate = K.alternate_frequency_basename(child, parent)
                assert alternate != primary
                assert alternate.startswith(f"{child}_parent")
                assert alternate.endswith("_frequency")
                token = alternate[len(f"{child}_parent"): -len("_frequency")]
                assert token == K.variant_parent_token(parent)
                assert token  # non-empty is what distinguishes it from the primary

    def test_a_dotted_score_matrix_name_round_trips_through_the_driver(
        self, tmp_path, driver_module
    ):
        """``scores/<key>_<variant>_score_matrix.csv`` with dots in the key."""
        scores = tmp_path / "scores"
        scores.mkdir()
        for label in C.LINEAGE_ORDER:
            key = CM.safe_label(label)
            variant = R.build_variant_name(2, 0.5, 1, label)
            path = scores / f"{key}_{variant}_score_matrix.csv"
            path.write_text("x", encoding="utf-8")
            assert driver_module.score_matrix_path(scores, key, variant) == path
        assert len(list(scores.glob("*_score_matrix.csv"))) == len(C.LINEAGE_ORDER)

    def test_escott_would_truncate_a_bare_label_but_never_the_ha_header(self):
        """escott.py:76 splits the header at the first non-alphanumeric character."""
        for label in C.LINEAGE_ORDER:
            header = f"HA{CM.lineage_tag(label)}"
            assert CM.escott_prot_token(header) == header
        # ... which is exactly why the raw label is never used as a header.
        assert CM.escott_prot_token("J.2_int") == "J"
        assert CM.escott_prot_token("J.2.4") == "J"

    def test_two_lineages_never_share_an_escott_product_filename(self):
        products = {
            f"HA{CM.lineage_tag(label)}_normPred_evolCombi.txt"
            for label in C.LINEAGE_ORDER
        }
        assert len(products) == len(C.LINEAGE_ORDER)

    def test_a_header_with_slashes_never_reaches_a_filename(self, tmp_path):
        """Real reference headers are ``EPI...|HA|A/England/.../2025|...``."""
        header = "EPI4748783|HA|A/England/01837755/2025|EPI_ISL_20210731|J.2.4.1"
        assert CM.escott_prot_token(header) == "EPI4748783"
        path = C.write_fasta(tmp_path / "ref.fasta", [(header, "MKTIIALS")])
        assert R.escott_prot_token(path) == "EPI4748783"
        assert "/" not in R.escott_prot_token(path)

    def test_every_prepared_inputs_filename_is_distinct_per_lineage(
        self, prepared_inputs_tree
    ):
        """One tree, five lineages: no two share a query, MSA, jet or frequency file."""
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        for subdir, pattern in (
            ("query", "{key}_query.fasta"),
            ("msa", "msa_{key}.fasta"),
            ("jet", "{key}_surrogate_jet.res"),
        ):
            paths = {
                (inputs_dir / subdir / pattern.format(key=CM.safe_label(label)))
                for label in C.LINEAGE_ORDER
            }
            assert len(paths) == len(C.LINEAGE_ORDER), subdir
            assert all(path.exists() for path in paths), subdir


# =========================================================================== #
# 4. Real-data coordinate invariants (opt-in behind --run-slow)
# =========================================================================== #

@pytest.fixture(scope="module")
def real_proteins() -> Dict[str, str]:
    if not REAL_REFERENCE_DIR.is_dir():
        pytest.skip(f"{REAL_REFERENCE_DIR} is not present")
    out: Dict[str, str] = {}
    for label in C.LINEAGE_ORDER:
        path = REAL_REFERENCE_DIR / f"{label}.nt.fa"
        if not path.exists():
            pytest.skip(f"{path} is not present")
        out[label] = CM.load_reference_cds(path, label)["protein"]
    return out


@pytest.mark.slow
@pytest.mark.requires_real_data
class TestRealCdsToProteinFrame:
    """CDS -> HA0 protein: the frame every other frame is measured against."""

    def test_every_lineage_translates_to_566_aa(self, real_proteins):
        assert {len(protein) for protein in real_proteins.values()} == {HA0_LENGTH}, (
            "build_query_fastas refuses differing lengths, and every downstream "
            "assertion (566 MSA columns, 566 jet rows, 566 ESCOTT columns) rests "
            "on this single number"
        )

    def test_the_signal_peptide_is_sixteen_residues(self, real_proteins):
        for label, protein in real_proteins.items():
            assert protein[:2] == "MK", label
            # The mature N-terminus: Q in every lineage except K, which carries Q17N.
            assert protein[SIGNAL_PEPTIDE_LENGTH] in "QN", label
            assert protein[SIGNAL_PEPTIDE_LENGTH:SIGNAL_PEPTIDE_LENGTH + 4] in (
                "QKIP", "QNIP",
            ), label

    def test_k_is_labelled_j241_in_its_own_header(self):
        """The evidence for the corrected ladder, read off disk rather than asserted."""
        header, _sequence = next(iter(CM.read_fasta(REAL_REFERENCE_DIR / "K.nt.fa")))
        assert header.endswith("|J.2.4.1"), header
        assert K.DEFAULT_PARENT_MAPS["clade_evidence"]["K"] == "J.2.4"


@pytest.mark.slow
@pytest.mark.requires_real_data
@pytest.mark.requires_prody
class TestReal6WXBRenumbering:
    """Author numbering -> query numbering on the real construct."""

    @pytest.fixture(scope="class")
    def prepared(self, tmp_path_factory) -> Dict[str, object]:
        from prescott_iav import prepare_inputs as PI

        if not REAL_6WXB.exists() or not (REAL_REFERENCE_DIR / "G.1.nt.fa").exists():
            pytest.skip("the production structure or reference is not present")
        protein = CM.load_reference_cds(REAL_REFERENCE_DIR / "G.1.nt.fa", "G.1")["protein"]
        out = tmp_path_factory.mktemp("real6wxb")
        report = PI.prepare_structure(REAL_6WXB, "A", "auto", protein, out, "6WXB", 0.60)
        return {"report": report, "protein": protein, "dir": out}

    def test_the_author_range_is_what_the_deposition_says(self, prepared):
        from prescott_iav import prepare_inputs as PI

        struct = PI.load_structure(REAL_6WXB)
        resnums = PI._chain_ca(struct, "A").getResnums()
        assert (int(resnums.min()), int(resnums.max())) == SIX_WXB_AUTHOR_RANGE

    def test_the_resolved_offset_is_the_signal_peptide_length(self, prepared):
        assert prepared["report"]["offset"] == SIX_WXB_OFFSET

    def test_the_offset_is_unambiguous_not_merely_the_best(self, prepared):
        """A wrong shift scores at chance (~9%); the right one at 84.5%.

        If the runner-up were close the scan would be a coin flip and a
        one-residue slip would be invisible.
        """
        from prescott_iav import prepare_inputs as PI

        struct = PI.load_structure(REAL_6WXB)
        calpha = PI._chain_ca(struct, "A")
        resnums, sequence = calpha.getResnums(), calpha.getSequence()
        protein = prepared["protein"]
        scored = []
        for offset in range(-40, 41):
            matches = sum(
                1
                for resnum, aa in zip(resnums, sequence)
                if 1 <= int(resnum) + offset <= len(protein)
                and protein[int(resnum) + offset - 1] == aa
            )
            scored.append((matches, offset))
        scored.sort(reverse=True)
        assert scored[0][1] == SIX_WXB_OFFSET
        assert scored[0][0] > 8 * scored[1][0], f"runner-up too close: {scored[:2]}"

    def test_every_covered_position_carries_the_right_residue_index(self, prepared):
        """Author resnum + 16 must index the query, not the query +/- 1."""
        from prescott_iav import prepare_inputs as PI

        protein = prepared["protein"]
        monomer = PI.load_structure(Path(prepared["report"]["monomer"]["path"]))
        calpha = monomer.select("name CA")
        resnums = [int(r) for r in calpha.getResnums()]
        sequence = calpha.getSequence()
        identity = sum(
            1 for resnum, aa in zip(resnums, sequence) if protein[resnum - 1] == aa
        ) / len(resnums)
        assert identity == pytest.approx(prepared["report"]["offset_identity"])
        assert identity > 0.80
        for slip in (-1, 1):
            slipped = sum(
                1
                for resnum, aa in zip(resnums, sequence)
                if 1 <= resnum + slip <= len(protein) and protein[resnum + slip - 1] == aa
            ) / len(resnums)
            assert slipped < 0.15, f"slip {slip:+d} scored {slipped:.1%}"

    def test_coverage_gaps_are_the_three_biologically_expected_runs(self, prepared):
        report = prepared["report"]
        assert (
            report["covered_positions"][0],
            report["covered_positions"][-1],
        ) == SIX_WXB_QUERY_RANGE
        assert report["n_covered"] == SIX_WXB_N_COVERED
        assert report["uncovered_runs"] == SIX_WXB_UNCOVERED_RUNS
        assert report["construct_truncated_at_author_resnum"] is None, (
            "6WXB's coordinates stop at author 501, before any linker/foldon/His6"
        )

    def test_no_residue_escapes_the_query_frame(self, prepared):
        for role in ("monomer", "trimer"):
            for chain_report in prepared["report"][role]["per_chain"].values():
                assert chain_report["resnum_min"] >= 1
                assert chain_report["resnum_max"] <= HA0_LENGTH


@pytest.mark.slow
@pytest.mark.requires_real_data
@pytest.mark.requires_prody
@pytest.mark.requires_scipy
@pytest.mark.requires_freesasa
@pytest.mark.requires_dssp
class TestRealStructureLandsOnTheRightJetRows:
    """Structure -> jet.res: the seam where a shift would poison pc, cv and ss."""

    def test_has_structure_is_exactly_the_covered_set(self, tmp_path):
        from prescott_iav import prepare_inputs as PI

        if not REAL_6WXB.exists() or not (REAL_REFERENCE_DIR / "G.1.nt.fa").exists():
            pytest.skip("the production structure or reference is not present")
        protein = CM.load_reference_cds(REAL_REFERENCE_DIR / "G.1.nt.fa", "G.1")["protein"]
        report = PI.prepare_structure(
            REAL_6WXB, "A", "auto", protein, tmp_path / "structure", "6WXB", 0.60
        )

        msa = C.write_fasta(
            tmp_path / "msa.fasta",
            [("HAG1", protein)] + [(f"row{i:03d}", protein) for i in range(6)],
        )
        components, meta = J.build_jet_table(
            msa,
            None,
            Path(report["monomer"]["path"]),
            Path(report["trimer"]["path"]),
            trace_definition="direct",
            max_zero_trace_fraction=None,
        )
        assert len(components) == HA0_LENGTH
        assert list(components["pos"]) == list(range(1, HA0_LENGTH + 1))

        covered = set(components.loc[components["has_structure"], "pos"].astype(int))
        assert covered == set(report["covered_positions"])

        # The uncovered set is exactly the three known runs, so a frame slip would
        # show up as an off-by-one on a run boundary.
        uncovered = sorted(set(range(1, HA0_LENGTH + 1)) - covered)
        assert uncovered[0] == 1 and uncovered[-1] == HA0_LENGTH
        assert 24 in uncovered and 25 not in uncovered
        # the cleavage-loop gap is 342..349 inclusive, so its flanks are COVERED
        assert 341 not in uncovered and 342 in uncovered
        assert 349 in uncovered and 350 not in uncovered
        assert 517 not in uncovered and 518 in uncovered

        # pc/cv/ss are defined exactly where there are coordinates and nowhere else.
        with_structure = components.loc[components["has_structure"]]
        without = components.loc[~components["has_structure"]]
        assert with_structure["cv"].notna().all()
        assert with_structure["pc"].notna().all()
        assert with_structure["ss"].notna().all()
        assert without["cv"].isna().all()
        assert without["pc"].isna().all()
        assert without["ss"].isna().all()
        assert meta["structure"]["covered"] == SIX_WXB_N_COVERED

        # ... and the residue named on each row is the QUERY residue at that index.
        assert "".join(components["aa1"]) == protein


@pytest.mark.slow
@pytest.mark.requires_real_data
class TestRealPanelAlignmentFrame:
    """The panel frame the EVALUATION half uses must be the one stage 1 builds on.

    ``common.map_reference_to_alignment_columns`` is a hand-written mirror of
    ``Functions_HuggingFace.build_reference_to_alignment_column_map``.  The
    frequency prior is laid out on the first and the observed diversity on the
    second, so any divergence shifts the prior against the very data it is scored
    against -- silently, because both halves keep producing full tables.
    """

    @pytest.fixture(scope="class")
    def guide_rows(self) -> List[Dict[str, str]]:
        if not REAL_GUIDE.exists():
            pytest.skip(f"{REAL_GUIDE} is not present")
        rows = CM.read_guide_rows(REAL_GUIDE)
        if not rows or not all(Path(row["diversity_path"]).exists() for row in rows):
            pytest.skip("the production GISAID panels are not present")
        return rows

    def test_the_two_halves_produce_byte_identical_maps(self, guide_rows):
        pytest.importorskip("Functions_HuggingFace")
        from Bio import SeqIO

        import Functions_HuggingFace as FH

        tables = FH.build_codon_aa_mutation_tables("H3N2")
        for row in guide_rows[:3]:
            label = row["label"]
            reference = CM.load_reference_cds(Path(row["reference_path"]), label)["protein"]
            panel = Path(row["diversity_path"])

            ours, aln_len, matched, _consensus = CM.map_reference_to_alignment_columns(
                reference, CM.read_fasta_sequences(panel)
            )
            theirs, their_len, their_matched = FH.build_reference_to_alignment_column_map(
                reference,
                list(SeqIO.parse(str(panel), "fasta")),
                tables["aa_to_codons"],
                CM.IGNORE_ALIGNMENT_CHARS,
            )
            assert ours == theirs, label
            assert (aln_len, matched) == (their_len, their_matched), label

    def test_the_signal_peptide_is_unmapped_and_position_17_is_column_1(self, guide_rows):
        for row in guide_rows[:3]:
            label = row["label"]
            reference = CM.load_reference_cds(Path(row["reference_path"]), label)["protein"]
            mapping, aln_len, _matched, _consensus = CM.map_reference_to_alignment_columns(
                reference, CM.read_fasta_sequences(Path(row["diversity_path"]))
            )
            assert aln_len == PANEL_ALIGNMENT_COLUMNS, label
            # The panels are MATURE HA; the reference is HA0.
            assert not (set(range(1, SIGNAL_PEPTIDE_LENGTH + 1)) & set(mapping)), label
            assert mapping[PANEL_FIRST_MAPPED[0]] == PANEL_FIRST_MAPPED[1], label
            assert min(mapping) == SIGNAL_PEPTIDE_LENGTH + 1, label
            assert max(mapping) == HA0_LENGTH, label
            assert len(mapping) == HA0_LENGTH - SIGNAL_PEPTIDE_LENGTH, label
            # 1-based alignment columns, never 0-based, never past the end.
            assert min(mapping.values()) >= 1
            assert max(mapping.values()) <= aln_len

    def test_the_frequency_file_lands_completely_on_the_escott_frame(self, guide_rows):
        """Stage A's own output must satisfy stage C's frame guard on real data."""
        from prescott_iav import prepare_inputs as PI

        by_label = {row["label"]: row for row in guide_rows}
        checked = 0
        for child, parent in K.DEFAULT_PARENT_MAPS["clade_evidence"].items():
            if child not in by_label or parent not in by_label:
                continue
            child_protein = CM.load_reference_cds(
                Path(by_label[child]["reference_path"]), child
            )["protein"]
            parent_protein = CM.load_reference_cds(
                Path(by_label[parent]["reference_path"]), parent
            )["protein"]
            with tempfile.TemporaryDirectory() as scratch:
                out = Path(scratch)
                PI.build_parent_frequency_file(
                    child_label=child,
                    parent_label=parent,
                    child_protein=child_protein,
                    parent_panel_fasta=Path(by_label[parent]["diversity_path"]),
                    out_txt=out / "f.txt",
                    out_meta=out / "f_meta.tsv",
                    min_count=1,
                    min_depth=50,
                    freq_max=0.95,
                    parent_protein=parent_protein,
                )
                frequencies = R.load_frequency_file(out / "f.txt")
            if not frequencies:
                continue
            checked += 1
            for mutant in frequencies:
                position = int(mutant[1:-1])
                assert 1 <= position <= HA0_LENGTH, (child, mutant)
                # Every label must index the CHILD's own HA0 protein ...
                assert child_protein[position - 1] == mutant[0], (child, mutant)
                # ... and never the signal peptide, which the panels do not cover.
                assert position > SIGNAL_PEPTIDE_LENGTH, (child, mutant)
        assert checked, "no evaluable edge had both panels on disk"


@pytest.mark.unit
class TestClippingDenominatorWithNoFrequencies:
    """The intent rescued from ``test_a_frequency_file_that_matches_nothing_...``.

    That test used a wholly unmatched frequency file as a convenient way to reach
    a zero clipping denominator; ``process_lineage`` now refuses such a file, so
    the property is pinned here at the level it actually belongs to.
    """

    def test_an_all_sentinel_frequency_matrix_has_no_denominator(
        self, parsed_escott_matrix
    ):
        log10_frequency, report = R.build_log10_frequency_matrix({}, parsed_escott_matrix)
        assert report["n_matched"] == 0
        ranked = R.escott_rank_scores(parsed_escott_matrix)
        penalised = R.apply_prescott_equation(ranked, log10_frequency, 1.0, -2.0, equation=2)
        clipping = R.count_clipped_to_zero(ranked, penalised, log10_frequency)
        assert clipping == {"n_mutants_with_frequency": 0, "n_clipped_to_zero": 0}
        # ... and with nothing to penalise the equation is a no-op.
        assert np.array_equal(penalised.to_numpy(), ranked.to_numpy())
