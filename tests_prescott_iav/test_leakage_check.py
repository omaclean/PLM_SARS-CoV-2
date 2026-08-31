#!/usr/bin/env python3
"""Tests for ``scripts/prescott_iav/leakage_check.py`` -- detection AND the purge.

The module's job is to stop a 2025/26 GISAID sequence sitting inside a 2024-cutoff
evolutionary set and inflating every reported correlation.  The tests are organised
around the ways that job silently fails rather than around the public API surface:

* the metric definitions (identity / coverage / Hamming) must behave correctly for
  sequences that are NOT in a common frame -- the deep set keeps HA's 16-residue
  signal peptide and the panels start at the mature N-terminus;
* the two purge thresholds combine with **OR**, are independently disableable, and
  which of them is the stricter depends on sequence LENGTH (at 550 aa the Hamming
  rule governs; at 1500 aa the identity rule does).  Both regimes are pinned;
* the coverage floor is a GATE: it must reject an exact 200-aa substring while
  accepting a signal-peptide-offset full-length duplicate;
* **the query row must never be purged** -- it is the lineage reference and GEMME's
  epistatic reference, it is ~identical to the evaluation target by construction,
  and it trips both thresholds every single time.  This is the most important
  property in the file: a regression makes ESCOTT score the wrong protein;
* every removal must be auditable, and removed + retained must reconstruct the
  input exactly.

Offline vs BLAST
----------------
Threshold-boundary work is done through ``fake_blast_records`` -- a deterministic
ungapped-overlay aligner that reproduces BLAST exactly for substitution-only
synthetic data (``TestFakeBlastMatchesRealBlast`` proves that against real blastp).
That keeps the boundary assertions exact, keeps the default suite offline, and
still exercises every branch of the purge.  Anything that must prove the module
works against the real aligner is marked ``requires_blast``.

Run with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_leakage_check.py -q
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prescott_iav import common  # noqa: E402
from prescott_iav import leakage_check as lc  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic sequence helpers.
#
# These mirror the real geometry: ~550 aa "mature" cores, deep rows carrying a
# 16-aa signal peptide the panels lack, so anything that assumes a common frame
# fails here.
# --------------------------------------------------------------------------- #

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

SIGNAL_PEPTIDE = "MKTIIALSYILCLVVA"
"""16 residues, the length of HA's real signal peptide.

The whole frame problem in one constant: the deep set has it, the panels do not,
so a genuine duplicate is neither byte-identical nor the same length.
"""

MATURE_LENGTH = 550
"""Length of a mature HA. Not cosmetic.

Coverage on the default ``both`` basis is ``aln / max(qlen, slen)``, so a 16-residue
signal peptide costs ``16/(L+16)`` of it: at L=550 a perfect duplicate still scores
97.17% and clears the 95% gate, but at L=300 it scores 94.9% and does NOT.  Testing
at 300 aa would exercise a coverage regime the real data never sees and would fail
for the wrong reason.
"""


def make_protein(seed: int, length: int = MATURE_LENGTH) -> str:
    """Deterministic pseudo-protein of ``length`` residues."""
    state = seed * 2654435761 % (2 ** 32)
    out = []
    for _ in range(length):
        state = (1103515245 * state + 12345) % (2 ** 31)
        out.append(_ALPHABET[state % len(_ALPHABET)])
    return "".join(out)


def substitute(seq: str, n: int, start: int = 25, stride: int = 9) -> str:
    """Introduce EXACTLY ``n`` substitutions, none within 25 residues of either end.

    Exactness matters: every threshold assertion in this file is stated as a
    literal mismatch count, so a helper that silently mutates the same position
    twice (or nudges a terminus and makes BLAST trim the alignment) would turn an
    exact test into an approximate one.  The post-condition is asserted here rather
    than trusted.
    """
    chars = list(seq)
    for i in range(n):
        pos = start + i * stride
        assert 24 < pos < len(seq) - 24, f"substitution {i} at {pos} is too close to a terminus"
        chars[pos] = "W" if chars[pos] != "W" else "C"
    out = "".join(chars)
    assert sum(a != b for a, b in zip(seq, out)) == n
    return out


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:]]


def _headers(path: Path) -> List[str]:
    return [header for header, _seq in common.read_fasta(Path(path))]


# --------------------------------------------------------------------------- #
# Hit construction helpers.
# --------------------------------------------------------------------------- #

def make_hit(**kwargs) -> lc.Hit:
    """A Hit describing a perfect signal-peptide-offset duplicate, overridable."""
    defaults = dict(
        qid="q0", sid="s0", qheader="q", sheader="s",
        aln_len=550, nident=550, mismatch=0, gaps=0, qlen=566, slen=550,
        qstart=17, qend=566, sstart=1, send=550,
        bitscore=1136.0, evalue=0.0, pident=100.0,
    )
    defaults.update(kwargs)
    return lc.Hit(**defaults)


class ForcedIdentityHit(lc.Hit):
    """A Hit whose ``identity`` is an exact float chosen by the test.

    No integer ``nident/aln_len`` ratio lands just below 99.0 in IEEE754 (checked),
    so the ``+1e-9`` tolerance in the purge rule cannot be probed with a real hit.
    This subclass is the only way to pin that tolerance, and pinning it matters: it
    is the difference between "99% means 99%" and "99% means 99% unless the division
    rounded down".
    """

    forced_identity: float = 0.0

    @property
    def identity(self) -> float:  # type: ignore[override]
        return self.forced_identity


def forced_identity_hit(identity: float, **kwargs) -> ForcedIdentityHit:
    defaults = dict(
        qid="q0", sid="s0", qheader="0", sheader="TGT0001",
        aln_len=550, nident=545, mismatch=5, gaps=0, qlen=566, slen=550,
        qstart=17, qend=566, sstart=1, send=550,
        bitscore=1000.0, evalue=0.0, pident=identity,
    )
    defaults.update(kwargs)
    hit = ForcedIdentityHit(**defaults)
    hit.forced_identity = identity
    return hit


# --------------------------------------------------------------------------- #
# The offline aligner.
# --------------------------------------------------------------------------- #

def _overlay(query: str, subject: str) -> Tuple[int, int, int]:
    """Best ungapped overlay of ``subject`` inside ``query``.

    Returns ``(nident, aln_len, offset)``.  For substitution-only data with an
    N-terminal offset -- which is exactly the geometry of the real deep set vs the
    real panels -- this reproduces blastp's ``nident``/``length`` exactly; that
    equivalence is asserted against real blastp in
    ``TestFakeBlastMatchesRealBlast``.
    """
    if not query or not subject:
        return 0, 0, 0
    best = (0, 0, 0)
    if len(query) >= len(subject):
        for offset in range(len(query) - len(subject) + 1):
            window = query[offset:offset + len(subject)]
            nident = sum(1 for a, b in zip(window, subject) if a == b)
            if nident > best[0]:
                best = (nident, len(subject), offset)
    else:
        for offset in range(len(subject) - len(query) + 1):
            window = subject[offset:offset + len(query)]
            nident = sum(1 for a, b in zip(window, query) if a == b)
            if nident > best[0]:
                best = (nident, len(query), offset)
    return best


def fake_blast_records(
    query_records: Sequence[Tuple[str, str]],
    subject_records: Sequence[Tuple[str, str]],
    workdir: Path,
    tag: str,
    *,
    min_identity_to_report: float = 25.0,
    **_kwargs,
) -> List[lc.Hit]:
    """Deterministic stand-in for :func:`leakage_check.blast_records`."""
    hits: List[lc.Hit] = []
    if not query_records or not subject_records:
        return hits
    for qi, (qheader, qseq) in enumerate(query_records):
        for si, (sheader, sseq) in enumerate(subject_records):
            nident, aln_len, offset = _overlay(qseq, sseq)
            if not aln_len:
                continue
            identity = 100.0 * nident / aln_len
            if identity < min_identity_to_report:
                continue
            hits.append(lc.Hit(
                qid=f"q{qi:07d}", sid=f"s{si:07d}",
                qheader=qheader, sheader=sheader,
                aln_len=aln_len, nident=nident,
                mismatch=aln_len - nident, gaps=0,
                qlen=len(qseq), slen=len(sseq),
                qstart=offset + 1, qend=offset + aln_len,
                sstart=1, send=aln_len,
                bitscore=float(nident) * 2.0, evalue=1e-99,
                pident=round(identity, 3),
            ))
    return hits


@pytest.fixture()
def fake_blast(monkeypatch):
    """Replace ``lc.blast_records`` with the offline aligner; record every call.

    Returns the call list, so a test can assert what was handed to BLAST -- which is
    how the "panel input must be ungapped before BLAST" requirement is checked
    directly rather than inferred from the result.
    """
    calls: List[Dict[str, object]] = []

    def _fake(query_records, subject_records, workdir, tag, **kwargs):
        common.ensure_dir(Path(workdir))          # blast_records does this too
        calls.append({
            "query": list(query_records),
            "subject": list(subject_records),
            "workdir": Path(workdir),
            "tag": tag,
            "kwargs": dict(kwargs),
        })
        return fake_blast_records(list(query_records), list(subject_records), workdir, tag)

    monkeypatch.setattr(lc, "blast_records", _fake)
    return calls


@pytest.fixture()
def scripted_blast(monkeypatch):
    """Replace ``lc.blast_records`` with a callable returning hand-built Hits.

    ``scripted_blast([hit, hit, ...])`` -- used where the point of the test is an
    exact metric value (identity exactly 99.0, coverage exactly 95.0) that no real
    sequence pair conveniently produces.
    """
    def _install(hits: Sequence[lc.Hit]):
        captured: List[Dict[str, object]] = []

        def _fake(query_records, subject_records, workdir, tag, **kwargs):
            captured.append({"query": list(query_records), "subject": list(subject_records)})
            return list(hits)

        monkeypatch.setattr(lc, "blast_records", _fake)
        return captured

    return _install


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #

DISTANCES = (0, 1, 5, 10, 11, 20, 50)
"""Mismatch counts planted into the deep set, chosen to bracket both defaults.

At L=550 with ``min_identity=99`` / ``max_hamming=10``:

    d    identity   identity>=99?   hamming<=10?   dropped?
    0    100.0000   yes             yes            yes  (identity+hamming)
    1     99.8182   yes             yes            yes  (identity+hamming)
    5     99.0909   yes             yes            yes  (identity+hamming)
   10     98.1818   NO              yes            yes  (hamming ONLY)
   11     98.0000   NO              NO             no
   20     96.3636   NO              NO             no
   50     90.9091   NO              NO             no
"""

EXPECTED_DROPPED_AT_DEFAULTS = ("D000", "D001", "D005", "D010")
EXPECTED_KEPT_AT_DEFAULTS = ("QUERY", "D011", "D020", "D050")


@pytest.fixture()
def distance_panel(tmp_path: Path) -> Dict[str, object]:
    """One 550-aa target sequence and a deep set at KNOWN mismatch distances.

    Row 0 is the query -- the lineage reference, byte-identical to the target's
    mature core apart from the signal peptide -- exactly the real situation.
    """
    core = make_protein(1)
    panel = tmp_path / "panel.fasta"
    common.write_fasta(panel, [("TGT0001", core)])

    rows = [("QUERY", SIGNAL_PEPTIDE + core)]
    rows += [(f"D{d:03d}", SIGNAL_PEPTIDE + substitute(core, d)) for d in DISTANCES]
    msa = tmp_path / "msa_dist.fasta"
    common.write_fasta(msa, rows)
    return {
        "panel": panel,
        "msa": msa,
        "core": core,
        "headers": [header for header, _ in rows],
        "records": rows,
    }


@pytest.fixture()
def long_panel(tmp_path: Path) -> Dict[str, object]:
    """A 1500-aa target: the regime where the IDENTITY rule is the stricter one.

    At 1500 residues ``identity >= 99`` tolerates 15 mismatches while
    ``hamming <= 10`` tolerates 10, so the 13-mismatch row is dropped by identity
    ALONE -- the exact mirror image of the 550-aa case, where the Hamming rule is
    the one that governs.  Both directions are pinned so the module docstring's
    "which rule is stricter depends on length" cannot silently become false.
    """
    core = make_protein(5, length=1500)
    panel = tmp_path / "long_panel.fasta"
    common.write_fasta(panel, [("LTGT", core)])
    rows = [
        ("QUERY", SIGNAL_PEPTIDE + core),
        ("L013", SIGNAL_PEPTIDE + substitute(core, 13, start=40, stride=37)),
        ("L018", SIGNAL_PEPTIDE + substitute(core, 18, start=40, stride=37)),
    ]
    msa = tmp_path / "msa_LONG.fasta"
    common.write_fasta(msa, rows)
    return {"panel": panel, "msa": msa, "core": core}


@pytest.fixture()
def panel_and_deep(tmp_path: Path) -> Tuple[Path, Path, List[str]]:
    """A target panel and a deep set overlapping in a realistic, tricky way."""
    core = make_protein(1)
    panel = tmp_path / "panel.fasta"
    common.write_fasta(panel, [
        ("TGT0001", core),
        ("TGT0002", substitute(core, 25)),
        ("TGT0003", substitute(core, 40)),
    ])
    deep_rows = [
        ("QUERY", SIGNAL_PEPTIDE + core),                  # row 0 -- never purged
        ("DEEP_exact", SIGNAL_PEPTIDE + core),             # 0 mismatches -> both rules
        ("DEEP_mut2", SIGNAL_PEPTIDE + substitute(core, 2)),   # 99.6% -> both rules
        ("DEEP_mut8", SIGNAL_PEPTIDE + substitute(core, 8)),   # 98.5% -> hamming only
        ("DEEP_mut60", SIGNAL_PEPTIDE + substitute(core, 60, stride=8)),  # neither
        ("DEEP_frag", core[50:250]),                       # exact 200-aa substring
        ("DEEP_other", make_protein(99)),                  # unrelated
    ]
    deep = tmp_path / "deep.fasta"
    common.write_fasta(deep, deep_rows)
    return panel, deep, [header for header, _ in deep_rows]


def purge(tmp_path: Path, panel: Path, msa: Path, **kwargs) -> Dict[str, object]:
    """``purge_msa_against_target`` with the depth guards relaxed for tiny fixtures."""
    options: Dict[str, object] = dict(
        target_lineage="TEST",
        min_depth_after=1,
        max_removed_fraction=1.0,
        threads=2,
        workdir=tmp_path / "work",
    )
    options.update(kwargs)
    return lc.purge_msa_against_target(
        Path(msa), Path(panel),
        tmp_path / "purged.fasta", tmp_path / "dropped.tsv",
        **options,
    )


# =========================================================================== #
# Module surface
# =========================================================================== #

@pytest.mark.unit
class TestModuleSurface:
    def test_every_exported_name_exists(self):
        missing = [name for name in lc.__all__ if not hasattr(lc, name)]
        assert missing == []

    def test_defaults_are_the_documented_values(self):
        """These numbers are quoted in --help, in CAVEATS and in the manifest."""
        assert lc.DEFAULT_MIN_IDENTITY == 99.0
        assert lc.DEFAULT_MAX_HAMMING == 10
        assert lc.DEFAULT_MIN_COVERAGE == 95.0
        assert lc.DEFAULT_FLAG_IDENTITY == 99.0
        assert lc.DEFAULT_FLAG_COVERAGE == 95.0
        assert lc.DEFAULT_MAX_REMOVED_FRACTION == 0.25
        assert lc.DEFAULT_MIN_DEPTH_AFTER == 500
        assert lc.DEFAULT_BLAST_TASK == "blastp-fast"
        assert lc.DEFAULT_MAX_TARGET_SEQS == 5

    def test_blast_outfmt_fields_cover_every_metric_input(self):
        """identity/coverage/hamming need exactly these columns from BLAST."""
        for field in ("nident", "length", "qlen", "slen", "qseqid", "sseqid"):
            assert field in lc.BLAST_OUTFMT_FIELDS

    def test_exception_types_are_distinct_and_catchable(self):
        assert issubclass(lc.LeakageError, RuntimeError)
        assert issubclass(lc.MsaDepthError, RuntimeError)
        assert issubclass(lc.BlastNotFoundError, RuntimeError)
        assert not issubclass(lc.MsaDepthError, lc.LeakageError)


# =========================================================================== #
# Sequence primitives
# =========================================================================== #

@pytest.mark.unit
class TestSequencePrimitives:
    def test_ungap_strips_every_alignment_character(self):
        assert lc.ungap("AC-DE.FG*HI~JK LM\tNP") == "ACDEFGHIJKLMNP"

    def test_ungap_uppercases(self):
        assert lc.ungap("ac-de") == "ACDE"

    def test_ungap_of_an_all_gap_row_is_empty(self):
        assert lc.ungap("-----") == ""

    def test_ungap_accepts_non_str_input(self):
        assert lc.ungap(12345) == "12345"

    def test_sequence_hash_ignores_gaps_and_case(self):
        assert lc.sequence_hash("AC-DE") == lc.sequence_hash("acde")

    def test_sequence_hash_separates_a_one_residue_difference(self):
        """The single most important negative: 1 substitution is NOT a duplicate."""
        assert lc.sequence_hash("ACDE") != lc.sequence_hash("ACDF")

    def test_full_length_hash_is_NOT_invariant_to_the_signal_peptide(self):
        """Documents why check C is a floor and not the instrument.

        This is the measured reality of the real files -- zero hash collisions
        between the deep set and any panel, while BLAST finds near neighbours -- so
        it is asserted rather than left as a comment.
        """
        core = make_protein(3)
        assert lc.sequence_hash(SIGNAL_PEPTIDE + core) != lc.sequence_hash(core)

    def test_suffix_hash_is_invariant_to_a_leading_signal_peptide(self):
        core = make_protein(3)
        assert lc.suffix_hash(SIGNAL_PEPTIDE + core, 400) == lc.suffix_hash(core, 400)

    def test_suffix_hash_returns_none_when_too_short(self):
        assert lc.suffix_hash("ACDEF", 200) is None

    def test_suffix_hash_of_the_whole_sequence_equals_the_full_hash(self):
        core = make_protein(4, length=60)
        assert lc.suffix_hash(core, 60) == lc.sequence_hash(core)

    @pytest.mark.parametrize("length", [0, -1])
    def test_suffix_hash_is_disabled_by_a_non_positive_length(self, length):
        assert lc.suffix_hash(make_protein(4, 60), length) is None

    def test_suffix_hash_counts_ungapped_residues_not_columns(self):
        aligned = "-" * 50 + make_protein(6, length=60)
        assert lc.suffix_hash(aligned, 60) == lc.sequence_hash(make_protein(6, length=60))

    @pytest.mark.parametrize("header,expected", [
        ("OQ233153", "OQ233153"),
        ("EPI4748783|HA|A/England/01837755/2025|EPI_ISL_20210731|J.2.4.1", "EPI4748783"),
        (">QBM69670 some description", "QBM69670"),
        ("", ""),
        ("   ", ""),
        ("A/England/415/2024 | extra", "A/England/415/2024"),
    ])
    def test_accession_of(self, header, expected):
        assert lc.accession_of(header) == expected

    @pytest.mark.parametrize("header", [">", "  >  ", ">>"])
    def test_accession_of_a_gt_only_header_must_not_crash(self, header):
        """Regression test for a fixed IndexError in ``leakage_check.accession_of``.

        The original was::

            token = str(header).strip().lstrip(">").split()[0] if str(header).strip() else ""

        The guard tests the ORIGINAL string for truthiness but indexes the string
        that has had its ``>`` stripped, so any header consisting only of ``>`` and
        whitespace passed the guard and then indexed an empty list.  It is reachable
        from a FASTA whose defline is ``>>`` (``common.read_fasta`` hands on the
        header ``">"``), and it killed the whole leakage stage with a traceback
        rather than skipping one malformed record.  Fixed to
        ``(… .split() or [""])[0]``; this test pins the ``""`` result.
        """
        assert lc.accession_of(header) == ""

    def test_a_gt_gt_defline_reaches_accession_of_through_the_real_reader(
        self, tmp_path: Path
    ):
        """The end-to-end reachability the fix above matters for.

        ``>>`` on disk is not hypothetical -- it is what a stray extra ``>`` from a
        concatenation produces.  This drives the real reader so the test fails if
        ``read_fasta`` ever stops yielding ``">"`` for it *or* if ``accession_of``
        regresses.
        """
        path = tmp_path / "malformed.fasta"
        path.write_text(">>\nACDEFGHIK\n>OQ233153\nACDEFGHIL\n", encoding="utf-8")
        records = list(common.read_fasta(path))
        assert [h for h, _ in records] == [">", "OQ233153"]
        assert [lc.accession_of(h) for h, _ in records] == ["", "OQ233153"]

    def test_load_ungapped_drops_empty_records_and_keeps_order(self, tmp_path: Path):
        path = tmp_path / "x.fasta"
        path.write_text(">a\nAC-DE\n>empty\n----\n>b\nFGHI\n", encoding="utf-8")
        assert lc.load_ungapped(path) == [("a", "ACDE"), ("b", "FGHI")]

    def test_load_ungapped_of_an_empty_file_is_empty(self, tmp_path: Path):
        path = tmp_path / "empty.fasta"
        path.write_text("", encoding="utf-8")
        assert lc.load_ungapped(path) == []

    def test_dedupe_by_hash_collapses_and_records_multiplicity(self):
        records = [("a", "ACDE"), ("b", "ACDE"), ("c", "ACDF"), ("d", "ACDE")]
        unique, members = lc.dedupe_by_hash(records)
        assert [header for header, _ in unique] == ["a", "c"]
        assert len(members) == 2
        assert sorted(next(v for v in members.values() if len(v) == 3)) == ["a", "b", "d"]

    def test_dedupe_by_hash_keeps_the_first_header_as_representative(self):
        unique, _ = lc.dedupe_by_hash([("second", "AAAA"), ("first", "AAAA")])
        assert unique == [("second", "AAAA")]

    def test_dedupe_by_hash_of_nothing_is_nothing(self):
        assert lc.dedupe_by_hash([]) == ([], {})

    def test_dedupe_by_hash_does_not_ungap(self):
        """It is fed ``load_ungapped`` output, so it must treat its input literally.

        If it silently re-ungapped, an aligned and an unaligned copy of the same
        sequence would collapse and the panel multiplicity in every report would be
        wrong.
        """
        unique, _ = lc.dedupe_by_hash([("a", "AC-DE"), ("b", "ACDE")])
        assert len(unique) == 2


# =========================================================================== #
# Threshold parsing
# =========================================================================== #

@pytest.mark.unit
class TestParseThreshold:
    @pytest.mark.parametrize("value", ["none", "NONE", " None ", "off", "disable",
                                       "disabled", "inf", "nan", ""])
    def test_sentinels_disable_the_rule(self, value):
        assert lc.parse_threshold(value, "--x") is None

    def test_none_disables_the_rule(self):
        assert lc.parse_threshold(None, "--x") is None

    def test_numbers_parse(self):
        assert lc.parse_threshold("99.0", "--x") == pytest.approx(99.0)
        assert lc.parse_threshold("10", "--x", int) == 10

    def test_a_numeric_value_is_accepted_not_only_a_string(self):
        assert lc.parse_threshold(99.5, "--x") == pytest.approx(99.5)

    def test_zero_is_a_real_value_not_a_sentinel(self):
        """0 must never be overloaded as 'off' -- it reads as 'purge at >= 0%'."""
        assert lc.parse_threshold("0", "--x") == 0.0
        assert lc.parse_threshold("0", "--x", int) == 0

    def test_negative_infinity_also_disables(self):
        assert lc.parse_threshold("-inf", "--x") is None

    def test_garbage_names_the_flag_and_lists_the_sentinels(self):
        with pytest.raises(ValueError, match=r"--leakage-min-identity") as excinfo:
            lc.parse_threshold("abc", "--leakage-min-identity")
        assert "none" in str(excinfo.value)

    def test_a_float_string_is_rejected_for_an_int_threshold(self):
        """--leakage-max-hamming 10.5 is a typo, not 10; it must not be truncated."""
        with pytest.raises(ValueError, match="--leakage-max-hamming"):
            lc.parse_threshold("10.5", "--leakage-max-hamming", int)


# =========================================================================== #
# Metric definitions (trap 2: the two sets are not in a common frame)
# =========================================================================== #

@pytest.mark.unit
class TestHitMetrics:
    def test_identity_is_nident_over_aligned_length(self):
        assert make_hit(aln_len=550, nident=545).identity == pytest.approx(100 * 545 / 550)

    def test_gap_columns_count_against_identity(self):
        """``length`` includes gap columns, so a gapped alignment cannot score 100%."""
        assert make_hit(aln_len=560, nident=550, gaps=10).identity < 100.0

    def test_hamming_is_shorter_length_minus_nident(self):
        assert make_hit(qlen=566, slen=550, nident=545).hamming == 5

    def test_hamming_does_not_charge_the_signal_peptide(self):
        """A perfect duplicate offset by a 16-aa signal peptide is Hamming ZERO.

        ``max(qlen,slen)-nident`` would score 16 and blow straight past a threshold
        of 10, i.e. every real duplicate would survive the purge.
        """
        assert make_hit(qlen=566, slen=550, aln_len=550, nident=550).hamming == 0

    def test_hamming_is_symmetric_in_which_side_is_longer(self):
        assert make_hit(qlen=550, slen=566, aln_len=550, nident=550).hamming == 0

    def test_hamming_is_an_int_not_a_float(self):
        assert isinstance(make_hit(nident=540).hamming, int)

    def test_coverage_both_accepts_the_signal_peptide_duplicate(self):
        hit = make_hit(qlen=566, slen=550, aln_len=550)
        assert hit.coverage("both") == pytest.approx(100 * 550 / 566)
        assert hit.coverage("both") > lc.DEFAULT_MIN_COVERAGE

    def test_coverage_both_rejects_an_exact_substring_fragment(self):
        """The failure mode the 'shorter' basis has: an exact 200-aa substring."""
        fragment = make_hit(qlen=200, slen=550, aln_len=200, nident=200)
        assert fragment.coverage("shorter") == pytest.approx(100.0)
        assert fragment.coverage("both") == pytest.approx(100 * 200 / 550)
        assert fragment.coverage("both") < lc.DEFAULT_MIN_COVERAGE

    def test_coverage_both_is_min_of_the_two_one_sided_coverages(self):
        hit = make_hit(qlen=566, slen=550, aln_len=540)
        assert hit.coverage("both") == pytest.approx(
            min(hit.coverage("query"), hit.coverage("subject"))
        )

    def test_coverage_bases_differ_as_documented(self):
        hit = make_hit(qlen=566, slen=550, aln_len=550)
        assert hit.coverage("query") == pytest.approx(100 * 550 / 566)
        assert hit.coverage("subject") == pytest.approx(100.0)
        assert hit.coverage("shorter") == pytest.approx(100.0)

    def test_an_unknown_basis_falls_back_to_both(self):
        hit = make_hit(qlen=566, slen=550, aln_len=550)
        assert hit.coverage("nonsense") == pytest.approx(hit.coverage("both"))

    def test_zero_length_does_not_divide_by_zero(self):
        assert make_hit(qlen=0, slen=0, aln_len=0).coverage("both") == 0.0
        assert make_hit(qlen=0, aln_len=10).coverage("query") == 0.0
        assert make_hit(slen=0, aln_len=10).coverage("subject") == 0.0
        assert make_hit(qlen=0, slen=0, aln_len=10).coverage("shorter") == 0.0
        assert make_hit(aln_len=0).identity == 0.0

    def test_hit_metrics_rounds_and_carries_every_audited_field(self):
        metrics = lc.hit_metrics(make_hit(aln_len=550, nident=543, qlen=566, slen=550))
        assert metrics["identity"] == round(100 * 543 / 550, 4)
        assert metrics["coverage"] == round(100 * 550 / 566, 4)
        assert metrics["hamming"] == 7
        assert set(metrics) == {"identity", "coverage", "hamming", "aln_len",
                                "nident", "qlen", "slen", "bitscore", "evalue"}

    def test_hit_metrics_honours_the_coverage_basis(self):
        hit = make_hit(qlen=566, slen=550, aln_len=550)
        assert lc.hit_metrics(hit, "subject")["coverage"] == pytest.approx(100.0)

    def test_the_two_thresholds_are_not_equivalent_on_a_real_ha(self):
        """The documented asymmetry, asserted so the --help text cannot go stale."""
        hamming_as_identity = 100 * (550 - lc.DEFAULT_MAX_HAMMING) / 550
        assert hamming_as_identity == pytest.approx(98.18, abs=0.01)
        assert hamming_as_identity < lc.DEFAULT_MIN_IDENTITY
        identity_as_hamming = 550 * (1 - lc.DEFAULT_MIN_IDENTITY / 100)
        assert identity_as_hamming == pytest.approx(5.5, abs=0.01)
        assert identity_as_hamming < lc.DEFAULT_MAX_HAMMING

    def test_which_rule_is_stricter_flips_with_length(self):
        """At 1500 aa the identity rule tolerates 15 mismatches, the Hamming rule 10."""
        assert 1500 * (1 - lc.DEFAULT_MIN_IDENTITY / 100) == pytest.approx(15.0)
        assert 1500 * (1 - lc.DEFAULT_MIN_IDENTITY / 100) > lc.DEFAULT_MAX_HAMMING


@pytest.mark.unit
class TestIdentityDistribution:
    def test_empty_is_reported_as_empty_not_zero(self):
        dist = lc.identity_distribution([])
        assert dist["n"] == 0
        assert dist["min"] is None and dist["max"] is None
        assert dist["percentiles"] == {} and dist["histogram"] == {}

    def test_percentiles_and_histogram(self):
        dist = lc.identity_distribution([90.0, 95.0, 99.5, 100.0])
        assert dist["n"] == 4
        assert dist["max"] == 100.0 and dist["min"] == 90.0
        assert dist["percentiles"]["p50"] == pytest.approx(97.25)
        assert dist["histogram"]["[100,100]"] == 1
        assert dist["histogram"]["[99.5,99.9)"] == 1

    def test_a_single_value_is_every_percentile(self):
        dist = lc.identity_distribution([99.4])
        assert dist["n"] == 1
        assert set(dist["percentiles"].values()) == {99.4}
        assert dist["mean"] == 99.4

    def test_histogram_bins_partition_the_input(self):
        values = [12.0, 71.0, 96.0, 99.2, 99.7, 99.95, 100.0]
        dist = lc.identity_distribution(values)
        assert sum(dist["histogram"].values()) == len(values)

    def test_bin_edges_are_left_inclusive(self):
        dist = lc.identity_distribution([99.0, 99.5, 99.9])
        assert dist["histogram"]["[99,99.5)"] == 1
        assert dist["histogram"]["[99.5,99.9)"] == 1
        assert dist["histogram"]["[99.9,100)"] == 1

    def test_mean_and_ordering_are_independent_of_input_order(self):
        forward = lc.identity_distribution([90.0, 95.0, 100.0])
        backward = lc.identity_distribution([100.0, 95.0, 90.0])
        assert forward == backward
        assert forward["mean"] == 95.0

    def test_values_at_or_above_100_land_in_the_closed_top_bin(self):
        dist = lc.identity_distribution([100.0, 100.0])
        assert dist["histogram"]["[100,100]"] == 2
        assert dist["histogram"]["[99.9,100)"] == 0


@pytest.mark.unit
class TestBestHitPerQuery:
    def test_ranks_by_nident_not_bitscore(self):
        """A longer, less identical alignment can outscore the real near-duplicate.

        The question is "how close is the nearest target sequence", so the count of
        identical residues is the right ranking key.
        """
        near = make_hit(qid="q1", sheader="NEAR", aln_len=550, nident=549, bitscore=10.0)
        long_but_worse = make_hit(qid="q1", sheader="FAR", aln_len=900, nident=500,
                                  bitscore=9999.0)
        best = lc.best_hit_per_query([long_but_worse, near])
        assert best["q1"].sheader == "NEAR"

    def test_bitscore_breaks_an_nident_tie(self):
        a = make_hit(qid="q1", sheader="A", nident=500, bitscore=10.0)
        b = make_hit(qid="q1", sheader="B", nident=500, bitscore=20.0)
        assert lc.best_hit_per_query([a, b])["q1"].sheader == "B"

    def test_one_entry_per_query(self):
        hits = [make_hit(qid="q1"), make_hit(qid="q2"), make_hit(qid="q1", nident=10)]
        assert sorted(lc.best_hit_per_query(hits)) == ["q1", "q2"]

    def test_no_hits_is_an_empty_mapping(self):
        assert lc.best_hit_per_query([]) == {}


# =========================================================================== #
# blast_records plumbing
# =========================================================================== #

@pytest.mark.unit
class TestBlastRecordsPlumbing:
    def test_empty_query_short_circuits_without_touching_blast(self, tmp_path, monkeypatch):
        def _explode(*_a, **_k):  # pragma: no cover - must never run
            raise AssertionError("blast must not be invoked for an empty input")

        monkeypatch.setattr(lc.subprocess, "run", _explode)
        assert lc.blast_records([], [("s", "ACDE")], tmp_path, "t") == []

    def test_empty_subject_short_circuits(self, tmp_path, monkeypatch):
        def _explode(*_a, **_k):  # pragma: no cover - must never run
            raise AssertionError("blast must not be invoked for an empty database")

        monkeypatch.setattr(lc.subprocess, "run", _explode)
        assert lc.blast_records([("q", "ACDE")], [], tmp_path, "t") == []

    def test_missing_binaries_raise_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lc.shutil, "which", lambda _name: None)
        with pytest.raises(lc.BlastNotFoundError) as excinfo:
            lc.blast_records([("q", "ACDE")], [("s", "ACDE")], tmp_path, "t")
        message = str(excinfo.value)
        assert "makeblastdb" in message and "blastp" in message
        assert "--no-leakage-check" in message
        assert "PRESCOTT" in message

    def test_indexed_fasta_hides_pipe_headers_from_blast(self, tmp_path):
        """BLAST rewrites ``|``-containing deflines, so records go in numbered.

        Without this the ``qseqid`` that comes back is mangled and every hit is
        attributed to the wrong sequence.
        """
        path = tmp_path / "indexed.fasta"
        mapping = lc._write_indexed_fasta(
            path, [("EPI1|HA|A/England/1/2025", "ACDE"), ("plain", "FGHI")], "q"
        )
        assert mapping == {"q0000000": "EPI1|HA|A/England/1/2025", "q0000001": "plain"}
        assert _headers(path) == ["q0000000", "q0000001"]
        assert "|" not in path.read_text(encoding="utf-8")

    def test_indexed_fasta_gives_duplicate_headers_distinct_ids(self, tmp_path):
        """makeblastdb rejects duplicate IDs; panels genuinely repeat headers."""
        path = tmp_path / "dupes.fasta"
        mapping = lc._write_indexed_fasta(path, [("same", "AC"), ("same", "DE")], "s")
        assert sorted(mapping) == ["s0000000", "s0000001"]

    def test_makeblastdb_failure_is_reported_with_its_stderr(self, tmp_path, monkeypatch):
        import subprocess as sp

        monkeypatch.setattr(lc.shutil, "which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(
            lc.subprocess, "run",
            lambda *a, **k: sp.CompletedProcess(a[0], 1, "", "database build exploded"),
        )
        with pytest.raises(RuntimeError, match="makeblastdb failed"):
            lc.blast_records([("q", "ACDE")], [("s", "ACDE")], tmp_path, "t")

    def test_blastp_failure_is_reported_with_its_stderr(self, tmp_path, monkeypatch):
        import subprocess as sp

        calls = {"n": 0}

        def _run(cmd, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return sp.CompletedProcess(cmd, 0, "", "")
            return sp.CompletedProcess(cmd, 2, "", "blastp exploded")

        monkeypatch.setattr(lc.shutil, "which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(lc.subprocess, "run", _run)
        with pytest.raises(RuntimeError, match="blastp failed"):
            lc.blast_records([("q", "ACDE")], [("s", "ACDE")], tmp_path, "t")

    def test_malformed_output_rows_are_skipped_not_crashed_on(self, tmp_path, monkeypatch):
        import subprocess as sp

        good = "\t".join(["q0000000", "s0000000", "100.0", "4", "0", "0",
                          "1", "4", "1", "4", "4", "4", "4", "8.0", "1e-3"])

        def _run(cmd, **_kwargs):
            if cmd[0].endswith("blastp"):
                out = Path(cmd[cmd.index("-out") + 1])
                out.write_text(f"truncated\trow\n{good}\n", encoding="utf-8")
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(lc.shutil, "which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(lc.subprocess, "run", _run)
        hits = lc.blast_records([("q", "ACDE")], [("s", "ACDE")], tmp_path, "t")
        assert len(hits) == 1
        assert hits[0].qheader == "q" and hits[0].sheader == "s"
        assert hits[0].nident == 4

    def test_max_hsps_one_is_not_negotiable(self, tmp_path, monkeypatch):
        """With several HSPs per pair, nident/length describe a fragment of the pair.

        Both identity and Hamming would then be computed from that fragment, so the
        flag is part of the metric definition rather than an optimisation.
        """
        import subprocess as sp

        seen: List[List[str]] = []

        def _run(cmd, **_kwargs):
            seen.append(list(cmd))
            if cmd[0].endswith("blastp"):
                Path(cmd[cmd.index("-out") + 1]).write_text("", encoding="utf-8")
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(lc.shutil, "which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(lc.subprocess, "run", _run)
        lc.blast_records([("q", "ACDE")], [("s", "ACDE")], tmp_path, "t",
                         blast_task="blastp", threads=3)
        blast_cmd = seen[-1]
        assert blast_cmd[blast_cmd.index("-max_hsps") + 1] == "1"
        assert blast_cmd[blast_cmd.index("-task") + 1] == "blastp"
        assert blast_cmd[blast_cmd.index("-num_threads") + 1] == "3"
        assert blast_cmd[blast_cmd.index("-comp_based_stats") + 1] == "0"

    def test_threads_are_floored_at_one(self, tmp_path, monkeypatch):
        import subprocess as sp

        seen: List[List[str]] = []

        def _run(cmd, **_kwargs):
            seen.append(list(cmd))
            if cmd[0].endswith("blastp"):
                Path(cmd[cmd.index("-out") + 1]).write_text("", encoding="utf-8")
            return sp.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(lc.shutil, "which", lambda name: f"/fake/{name}")
        monkeypatch.setattr(lc.subprocess, "run", _run)
        lc.blast_records([("q", "AC")], [("s", "AC")], tmp_path, "t", threads=0)
        assert seen[-1][seen[-1].index("-num_threads") + 1] == "1"

    @pytest.mark.requires_blast
    @pytest.mark.integration
    def test_real_blast_cleans_up_after_itself(self, tmp_path):
        core = make_protein(2)
        lc.blast_records([("q", core)], [("s", core)], tmp_path, "cleanup", threads=2)
        leftovers = sorted(p.name for p in tmp_path.iterdir())
        assert leftovers == [], f"blast_records left {leftovers} behind"

    @pytest.mark.requires_blast
    @pytest.mark.integration
    def test_keep_files_preserves_the_inputs_for_debugging(self, tmp_path):
        core = make_protein(2)
        lc.blast_records([("q", core)], [("s", core)], tmp_path, "keep",
                         threads=2, keep_files=True)
        names = {p.name for p in tmp_path.iterdir()}
        assert {"keep_query.fasta", "keep_subject.fasta", "keep_blast.tsv"} <= names


# =========================================================================== #
# The offline aligner is only trustworthy if it matches the real one
# =========================================================================== #

@pytest.mark.requires_blast
@pytest.mark.integration
class TestFakeBlastMatchesRealBlast:
    def test_identical_metrics_on_the_distance_fixture(self, tmp_path, distance_panel):
        """Every offline threshold assertion rests on this equivalence."""
        deep = [(h, lc.ungap(s)) for h, s in distance_panel["records"]]
        panel = lc.load_ungapped(distance_panel["panel"])
        real = {h.qheader: h for h in lc.blast_records(deep, panel, tmp_path / "w", "x",
                                                       threads=2)}
        fake = {h.qheader: h for h in fake_blast_records(deep, panel, tmp_path, "x")}
        assert sorted(real) == sorted(fake)
        for header, hit in real.items():
            other = fake[header]
            assert (other.nident, other.aln_len, other.qlen, other.slen) == \
                   (hit.nident, hit.aln_len, hit.qlen, hit.slen), header
            assert other.identity == pytest.approx(hit.identity)
            assert other.hamming == hit.hamming
            assert other.coverage("both") == pytest.approx(hit.coverage("both"))


# =========================================================================== #
# Check C -- exact duplicates by hash, across namespaces
# =========================================================================== #

@pytest.mark.unit
class TestHashDuplicatesCheckC:
    def test_finds_an_exact_cross_namespace_duplicate(self, tmp_path: Path):
        """Different accessions, same protein: the hash is namespace-blind."""
        shared = make_protein(7)
        a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
        common.write_fasta(a, [("QBM69670", shared), ("QBM00001", make_protein(8))])
        common.write_fasta(b, [("OQ233153", shared)])
        report = lc.check_hash_duplicates({"deep": a, "panel:K": b})
        pair = report["pairs"][0]
        assert pair["set_a"] == "deep" and pair["set_b"] == "panel:K"
        assert pair["n_shared_exact_sequences"] == 1
        assert pair["shared_fraction_of_a"] == pytest.approx(0.5)
        assert pair["shared_fraction_of_b"] == pytest.approx(1.0)
        assert report["n_pairs_with_exact_overlap"] == 1

    def test_one_residue_apart_is_not_an_exact_duplicate(self, tmp_path: Path):
        """The negative control for the whole of check C."""
        core = make_protein(7)
        a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
        common.write_fasta(a, [("A1", core)])
        common.write_fasta(b, [("B1", substitute(core, 1))])
        report = lc.check_hash_duplicates({"A": a, "B": b}, suffix_length=0)
        assert report["pairs"][0]["n_shared_exact_sequences"] == 0
        assert report["n_pairs_with_exact_overlap"] == 0

    def test_alignment_gaps_do_not_hide_a_duplicate(self, tmp_path: Path):
        """Panels arrive ALIGNED (594 columns); the hash is taken after ungapping."""
        core = make_protein(9)
        aligned = core[:100] + "-" * 44 + core[100:]
        a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
        common.write_fasta(a, [("A1", aligned)])
        common.write_fasta(b, [("B1", core)])
        assert lc.check_hash_duplicates({"A": a, "B": b})["pairs"][0][
            "n_shared_exact_sequences"] == 1

    def test_signal_peptide_offset_hides_from_the_full_hash_not_the_suffix_hash(
        self, tmp_path: Path,
    ):
        core = make_protein(11)
        a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
        common.write_fasta(a, [("deep", SIGNAL_PEPTIDE + core)])
        common.write_fasta(b, [("panel", core)])
        report = lc.check_hash_duplicates({"deep": a, "panel": b}, suffix_length=400)
        pair = report["pairs"][0]
        assert pair["n_shared_exact_sequences"] == 0
        assert pair["n_shared_cterm_suffix"] == 1

    def test_suffix_length_zero_disables_the_suffix_hash(self, tmp_path: Path):
        core = make_protein(11)
        a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
        common.write_fasta(a, [("deep", SIGNAL_PEPTIDE + core)])
        common.write_fasta(b, [("panel", core)])
        report = lc.check_hash_duplicates({"deep": a, "panel": b}, suffix_length=0)
        assert report["pairs"][0]["n_shared_cterm_suffix"] == 0
        assert report["suffix_length"] == 0

    def test_reports_duplication_factor_and_lengths(self, tmp_path: Path):
        seq = make_protein(13)
        path = tmp_path / "p.fasta"
        common.write_fasta(path, [("x", seq), ("y", seq), ("z", seq[:100])])
        info = lc.check_hash_duplicates({"P": path})["sets"]["P"]
        assert info["n_records"] == 3 and info["n_unique_sequences"] == 2
        assert info["duplication_factor"] == 1.5
        assert info["min_length"] == 100 and info["max_length"] == MATURE_LENGTH
        assert info["path"] == str(path)

    def test_an_empty_set_is_described_not_divided_by(self, tmp_path: Path):
        empty = tmp_path / "empty.fasta"
        empty.write_text("", encoding="utf-8")
        other = tmp_path / "o.fasta"
        common.write_fasta(other, [("x", make_protein(1))])
        report = lc.check_hash_duplicates({"E": empty, "O": other})
        assert report["sets"]["E"]["n_records"] == 0
        assert report["sets"]["E"]["duplication_factor"] == 0.0
        assert report["sets"]["E"]["min_length"] == 0
        assert report["pairs"][0]["shared_fraction_of_a"] == 0.0

    def test_a_single_set_has_no_pairs(self, tmp_path: Path):
        path = tmp_path / "p.fasta"
        common.write_fasta(path, [("x", make_protein(1))])
        report = lc.check_hash_duplicates({"P": path})
        assert report["pairs"] == []
        assert report["n_pairs_with_exact_overlap"] == 0

    def test_three_sets_give_three_unordered_pairs(self, tmp_path: Path):
        paths = {}
        for name in ("A", "B", "C"):
            path = tmp_path / f"{name}.fasta"
            common.write_fasta(path, [(name, make_protein(ord(name)))])
            paths[name] = path
        pairs = lc.check_hash_duplicates(paths)["pairs"]
        assert [(p["set_a"], p["set_b"]) for p in pairs] == [
            ("A", "B"), ("A", "C"), ("B", "C")
        ]

    def test_carries_the_caveat_that_a_null_result_proves_nothing(self, tmp_path: Path):
        path = tmp_path / "p.fasta"
        common.write_fasta(path, [("x", make_protein(1))])
        assert "NOT evidence of no leakage" in lc.check_hash_duplicates({"P": path})["caveat"]

    def test_finds_the_planted_duplicate_in_the_conftest_fixture(self, leakage_panels):
        """Ground truth from the shared fixture: exactly ONE shared exact sequence.

        The fixture plants parent row 7 as a byte-identical copy of target row 3 and
        gives it a different accession, which is the real-world pattern: the same
        virus deposited twice under two IDs.
        """
        report = lc.check_hash_duplicates({
            "panel:parent": leakage_panels["parent_fasta"],
            "panel:target": leakage_panels["target_fasta"],
        }, suffix_length=0)
        pair = report["pairs"][0]
        assert pair["n_shared_exact_sequences"] == \
            leakage_panels["expected_shared_exact_sequences"] == 1

    def test_the_planted_deep_leak_is_invisible_to_the_full_hash(self, leakage_panels):
        """Why the purge cannot be built on hashing.

        The fixture's deep row 5 IS target row 0, with the signal peptide prepended.
        It is a genuine leak and check C cannot see it -- only alignment can.
        """
        report = lc.check_hash_duplicates({
            "deep_msa": leakage_panels["deep_fasta"],
            "panel:target": leakage_panels["target_fasta"],
        }, suffix_length=0)
        assert report["pairs"][0]["n_shared_exact_sequences"] == 0

    def test_the_suffix_hash_does_see_the_planted_deep_leak(self, leakage_panels):
        report = lc.check_hash_duplicates({
            "deep_msa": leakage_panels["deep_fasta"],
            "panel:target": leakage_panels["target_fasta"],
        }, suffix_length=40)
        assert report["pairs"][0]["n_shared_cterm_suffix"] >= 1


# =========================================================================== #
# THE PURGE -- threshold boundaries, offline and exact
# =========================================================================== #

@pytest.mark.unit
class TestPurgeThresholdBoundaries:
    """Exactly 99%, just under, just over -- and the same for Hamming and coverage.

    These use hand-built hits because no convenient sequence pair lands exactly on
    a threshold, and "approximately at the boundary" is not a boundary test.
    """

    def _msa(self, tmp_path: Path) -> Tuple[Path, Path]:
        core = make_protein(1)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT0001", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core), ("CAND", make_protein(2))])
        return panel, msa

    def _run(self, tmp_path, scripted_blast, hit, **kwargs):
        panel, msa = self._msa(tmp_path)
        scripted_blast([hit] if hit is not None else [])
        options = dict(protect_indices=(), min_identity=None, max_hamming=None)
        options.update(kwargs)
        return purge(tmp_path, panel, msa, **options)

    # ---- identity ---------------------------------------------------------- #

    def test_identity_exactly_at_the_threshold_drops(self, tmp_path, scripted_blast):
        """495/500 is exactly 99.0 in IEEE754; ``>=`` must include it."""
        hit = make_hit(qheader="1", aln_len=500, nident=495, qlen=500, slen=500)
        assert hit.identity == 99.0
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0)
        assert report["n_removed"] == 1
        assert report["n_removed_by_identity_only"] == 1

    def test_identity_just_under_the_threshold_survives(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=500, nident=494, qlen=500, slen=500)
        assert hit.identity == 98.8
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0)
        assert report["n_removed"] == 0

    def test_identity_just_over_the_threshold_drops(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=500, nident=496, qlen=500, slen=500)
        assert hit.identity == 99.2
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0)
        assert report["n_removed"] == 1

    def test_a_float_rounding_shortfall_still_counts_as_at_the_threshold(
        self, tmp_path, scripted_blast,
    ):
        """The ``+1e-9`` tolerance: 99.0 minus a rounding error is still 99%."""
        hit = forced_identity_hit(99.0 - 1e-10, qheader="1", qlen=500, slen=500,
                                  aln_len=500, nident=495)
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0)
        assert report["n_removed"] == 1

    def test_the_tolerance_is_not_a_licence_to_round_down_a_real_gap(
        self, tmp_path, scripted_blast,
    ):
        hit = forced_identity_hit(99.0 - 1e-6, qheader="1", qlen=500, slen=500,
                                  aln_len=500, nident=495)
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0)
        assert report["n_removed"] == 0

    # ---- hamming ----------------------------------------------------------- #

    def test_hamming_exactly_at_the_threshold_drops(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=540, qlen=566, slen=550)
        assert hit.hamming == 10
        report = self._run(tmp_path, scripted_blast, hit, max_hamming=10)
        assert report["n_removed"] == 1
        assert report["n_removed_by_hamming_only"] == 1

    def test_hamming_one_over_the_threshold_survives(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=539, qlen=566, slen=550)
        assert hit.hamming == 11
        assert self._run(tmp_path, scripted_blast, hit, max_hamming=10)["n_removed"] == 0

    def test_hamming_zero_drops(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=550, qlen=566, slen=550)
        report = self._run(tmp_path, scripted_blast, hit, max_hamming=0)
        assert report["n_removed"] == 1

    # ---- coverage gate ------------------------------------------------------ #

    def test_coverage_exactly_at_the_floor_passes_the_gate(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=475, nident=475, qlen=500, slen=475)
        assert hit.coverage("both") == 95.0
        report = self._run(tmp_path, scripted_blast, hit, max_hamming=0, min_coverage=95.0)
        assert report["n_removed"] == 1
        assert report["n_coverage_gated_near_misses"] == 0

    def test_coverage_just_under_the_floor_is_gated(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=474, nident=474, qlen=500, slen=474)
        assert hit.coverage("both") == 94.8
        report = self._run(tmp_path, scripted_blast, hit, max_hamming=0, min_coverage=95.0)
        assert report["n_removed"] == 0
        assert report["n_coverage_gated_near_misses"] == 1
        assert report["coverage_gated_near_miss_max_coverage"] == pytest.approx(94.8)

    def test_a_gated_hit_that_would_not_have_fired_is_not_a_near_miss(
        self, tmp_path, scripted_blast,
    ):
        """Near-misses report what the GATE decided, not everything below coverage."""
        hit = make_hit(qheader="1", aln_len=200, nident=100, qlen=500, slen=200)
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0, max_hamming=10)
        assert report["n_removed"] == 0
        assert report["n_coverage_gated_near_misses"] == 0

    def test_one_near_miss_per_row_not_per_hsp(self, tmp_path, scripted_blast):
        """Two gated hits on the same row read as one row the gate saved."""
        weak = make_hit(qheader="1", sheader="S1", aln_len=474, nident=470,
                        qlen=500, slen=474)
        strong = make_hit(qheader="1", sheader="S2", aln_len=474, nident=474,
                          qlen=500, slen=474)
        panel, msa = self._msa(tmp_path)
        scripted_blast([weak, strong])
        report = purge(tmp_path, panel, msa, protect_indices=(), min_identity=None,
                       max_hamming=10, min_coverage=95.0)
        assert report["n_coverage_gated_near_misses"] == 1
        assert report["coverage_gated_near_misses"][0]["matched_target_sequence"] == "S2"

    # ---- the OR combination -------------------------------------------------- #

    def test_identity_alone_fires_when_hamming_does_not(self, tmp_path, scripted_blast):
        """A long protein: 13 mismatches is >99% identity but >10 Hamming."""
        hit = make_hit(qheader="1", aln_len=1500, nident=1487, qlen=1516, slen=1500)
        assert hit.identity > 99.0 and hit.hamming == 13
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0, max_hamming=10)
        assert report["n_removed"] == 1
        assert report["n_removed_by_identity_only"] == 1
        assert report["n_removed_by_hamming_only"] == 0

    def test_hamming_alone_fires_when_identity_does_not(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=540, qlen=566, slen=550)
        assert hit.identity < 99.0 and hit.hamming == 10
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0, max_hamming=10)
        assert report["n_removed_by_hamming_only"] == 1
        assert report["n_removed_by_identity_only"] == 0

    def test_both_rules_together_are_labelled_as_such(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=550, qlen=566, slen=550)
        report = self._run(tmp_path, scripted_blast, hit, min_identity=99.0, max_hamming=10)
        assert report["n_removed_by_both"] == 1
        assert _read_tsv(tmp_path / "dropped.tsv")[0]["rule"] == "identity+hamming"

    def test_neither_rule_keeps_the_row(self, tmp_path, scripted_blast):
        hit = make_hit(qheader="1", aln_len=550, nident=500, qlen=566, slen=550)
        assert self._run(tmp_path, scripted_blast, hit,
                         min_identity=99.0, max_hamming=10)["n_removed"] == 0

    def test_the_closest_offender_is_the_one_recorded(self, tmp_path, scripted_blast):
        """The audit row must name the strongest evidence, not the first hit printed."""
        weak = make_hit(qheader="1", sheader="WEAK", aln_len=550, nident=545,
                        qlen=566, slen=550)
        strong = make_hit(qheader="1", sheader="STRONG", aln_len=550, nident=550,
                          qlen=566, slen=550)
        panel, msa = self._msa(tmp_path)
        scripted_blast([weak, strong])
        report = purge(tmp_path, panel, msa, protect_indices=(),
                       min_identity=99.0, max_hamming=10)
        assert report["n_removed"] == 1
        row = _read_tsv(tmp_path / "dropped.tsv")[0]
        assert row["matched_target_sequence"] == "STRONG"
        assert float(row["identity"]) == 100.0


# =========================================================================== #
# THE PURGE -- exact survivor sets at known distances (offline)
# =========================================================================== #

@pytest.mark.unit
class TestPurgeAtKnownDistances:
    """Deep rows at 0, 1, 5, 10, 11, 20 and 50 mismatches from the target.

    Every assertion is a literal survivor set, because "roughly the right number
    were removed" is exactly the kind of statement a purge bug survives.
    """

    def test_defaults_drop_up_to_ten_mismatches_and_no_more(self, tmp_path,
                                                            distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        kept = _headers(tmp_path / "purged.fasta")
        assert kept == list(EXPECTED_KEPT_AT_DEFAULTS)
        assert report["n_removed"] == 4
        dropped = {row["msa_header"] for row in _read_tsv(tmp_path / "dropped.tsv")}
        assert dropped == set(EXPECTED_DROPPED_AT_DEFAULTS)

    def test_identity_rule_alone_keeps_the_ten_mismatch_row(self, tmp_path,
                                                            distance_panel, fake_blast):
        """99% of 550 aa is 5.5 mismatches, so D010 (98.18%) survives."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       max_hamming=None)
        kept = _headers(tmp_path / "purged.fasta")
        assert kept == ["QUERY", "D010", "D011", "D020", "D050"]
        assert report["n_removed"] == 3
        assert report["n_removed_by_identity_only"] == 3
        assert report["n_removed_by_hamming_only"] == 0
        assert report["n_removed_by_both"] == 0

    def test_hamming_rule_alone_reproduces_the_default_result(self, tmp_path,
                                                              distance_panel, fake_blast):
        """The documented claim, as an equality: at 550 aa Hamming governs."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       min_identity=None)
        assert _headers(tmp_path / "purged.fasta") == list(EXPECTED_KEPT_AT_DEFAULTS)
        assert report["n_removed"] == 4
        assert report["n_removed_by_hamming_only"] == 4

    def test_the_identity_only_set_is_a_strict_subset_of_the_default_set(
        self, tmp_path, distance_panel, fake_blast,
    ):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"], max_hamming=None)
        identity_only = {r["msa_header"] for r in _read_tsv(tmp_path / "dropped.tsv")}
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        default = {r["msa_header"] for r in _read_tsv(tmp_path / "dropped.tsv")}
        assert identity_only < default
        assert default - identity_only == {"D010"}

    @pytest.mark.parametrize("max_hamming,expected_removed", [
        (0, 1), (1, 2), (5, 3), (10, 4), (11, 5), (20, 6), (50, 7),
    ])
    def test_the_hamming_threshold_sweeps_the_planted_distances_exactly(
        self, tmp_path, distance_panel, fake_blast, max_hamming, expected_removed,
    ):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       min_identity=None, max_hamming=max_hamming)
        assert report["n_removed"] == expected_removed

    @pytest.mark.parametrize("min_identity,expected_removed", [
        (100.0, 1),      # only the exact copy
        (99.5, 2),       # + the 1-mismatch row (99.8182)
        (99.0, 3),       # + the 5-mismatch row (99.0909)
        (98.0, 5),       # + 10 (98.1818) and 11 (98.0000)
        (96.0, 6),       # + 20 (96.3636)
    ])
    def test_the_identity_threshold_sweeps_the_planted_distances_exactly(
        self, tmp_path, distance_panel, fake_blast, min_identity, expected_removed,
    ):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       min_identity=min_identity, max_hamming=None)
        assert report["n_removed"] == expected_removed

    def test_at_1500_aa_the_identity_rule_is_the_stricter_one(self, tmp_path,
                                                              long_panel, fake_blast):
        """The mirror image of the 550-aa case, and the reason OR is not redundant.

        13 mismatches in 1500 residues is 99.13% identity -- caught by the identity
        rule, missed by the Hamming rule.  At 550 aa the same two thresholds behave
        in exactly the opposite order.
        """
        report = purge(tmp_path, long_panel["panel"], long_panel["msa"])
        assert _headers(tmp_path / "purged.fasta") == ["QUERY", "L018"]
        assert report["n_removed"] == 1
        assert report["n_removed_by_identity_only"] == 1
        assert _read_tsv(tmp_path / "dropped.tsv")[0]["rule"] == "identity"

    def test_at_1500_aa_the_hamming_rule_alone_removes_nothing(self, tmp_path,
                                                               long_panel, fake_blast):
        report = purge(tmp_path, long_panel["panel"], long_panel["msa"],
                       min_identity=None, max_hamming=10)
        assert report["n_removed"] == 0

    def test_one_residue_apart_is_not_an_exact_duplicate_but_is_purged(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """The headline requirement, stated as one test.

        D001 differs from the target by a single residue: no hash sees it, and the
        near-identity rule must remove it anyway.
        """
        core = distance_panel["core"]
        assert lc.sequence_hash(SIGNAL_PEPTIDE + substitute(core, 1)) != \
            lc.sequence_hash(core)
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        row = next(r for r in _read_tsv(tmp_path / "dropped.tsv") if r["msa_header"] == "D001")
        assert float(row["identity"]) == pytest.approx(100 * 549 / 550, abs=1e-4)
        assert int(row["hamming"]) == 1

    def test_disabling_both_rules_is_refused(self, tmp_path, distance_panel, fake_blast):
        with pytest.raises(ValueError, match="--no-purge-leakage"):
            purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                  min_identity=None, max_hamming=None)


@pytest.mark.requires_blast
@pytest.mark.integration
class TestPurgeAtKnownDistancesWithRealBlast:
    """The same survivor sets, through real blastp rather than the offline aligner."""

    def test_defaults_drop_exactly_the_first_four_distances(self, tmp_path, distance_panel):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert _headers(tmp_path / "purged.fasta") == list(EXPECTED_KEPT_AT_DEFAULTS)
        assert report["n_removed"] == 4
        assert report["n_removed_by_both"] == 3
        assert report["n_removed_by_hamming_only"] == 1
        assert report["n_removed_exact_full_length"] == 1

    def test_measured_identities_match_the_planted_distances(self, tmp_path, distance_panel):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        rows = {r["msa_header"]: r for r in _read_tsv(tmp_path / "dropped.tsv")}
        for header, distance in (("D000", 0), ("D001", 1), ("D005", 5), ("D010", 10)):
            assert int(rows[header]["hamming"]) == distance
            assert float(rows[header]["identity"]) == pytest.approx(
                100 * (550 - distance) / 550, abs=1e-3
            )

    def test_identity_rule_alone_keeps_the_ten_mismatch_row(self, tmp_path, distance_panel):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"], max_hamming=None)
        assert "D010" in _headers(tmp_path / "purged.fasta")

    def test_at_1500_aa_the_identity_rule_governs(self, tmp_path, long_panel):
        report = purge(tmp_path, long_panel["panel"], long_panel["msa"])
        assert _headers(tmp_path / "purged.fasta") == ["QUERY", "L018"]
        assert _read_tsv(tmp_path / "dropped.tsv")[0]["rule"] == "identity"


# =========================================================================== #
# THE QUERY-EXEMPTION TRAP -- the most important behaviour in the module
# =========================================================================== #

@pytest.mark.unit
class TestQueryExemption:
    """Row 0 is the lineage reference: ~identical to the target, and never purged.

    In the real pipeline the query is the evaluation lineage's own reference
    protein, so it trips both thresholds against the evaluation panel every single
    time.  If it is removed, ESCOTT/GEMME scores a different protein than the one
    being evaluated -- or dies for want of a query.
    """

    def test_the_query_survives_stays_at_index_zero_and_is_unaltered(
        self, tmp_path, distance_panel, fake_blast,
    ):
        original = list(common.read_fasta(distance_panel["msa"]))
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        kept = list(common.read_fasta(tmp_path / "purged.fasta"))
        assert kept[0] == original[0]
        assert kept[0][0] == "QUERY"
        assert kept[0][1] == SIGNAL_PEPTIDE + distance_panel["core"]

    def test_the_query_is_exempted_not_simply_missed(self, tmp_path,
                                                     distance_panel, fake_blast):
        """It must be recorded as "would have been dropped", with its metrics.

        A query that is quietly not near anything is a different (and alarming)
        situation from a query that tripped both rules and was protected.
        """
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        exempted = report["query_exempted"]
        assert exempted is not None
        assert exempted["exempted"] is True
        assert "protected index" in exempted["exemption_reason"]
        assert exempted["msa_index"] == 0
        assert exempted["msa_header"] == "QUERY"
        assert exempted["identity"] == pytest.approx(100.0)
        assert exempted["hamming"] == 0
        assert exempted["rule"] == "identity+hamming"
        assert report["query_protected"] is True
        assert report["query_header"] == "QUERY"

    def test_the_query_is_absent_from_the_drop_manifest(self, tmp_path,
                                                        distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert "QUERY" not in {r["msa_header"] for r in _read_tsv(tmp_path / "dropped.tsv")}

    def test_the_query_counts_towards_the_surviving_depth(self, tmp_path,
                                                          distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert report["depth_after"] == len(_headers(tmp_path / "purged.fasta")) == 4

    def test_without_the_exemption_the_query_WOULD_be_purged(self, tmp_path,
                                                             distance_panel, fake_blast):
        """The control that makes the exemption test meaningful.

        With ``protect_indices=()`` the query is removed, which is precisely the
        regression the default guards against.
        """
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       protect_indices=())
        assert "QUERY" not in _headers(tmp_path / "purged.fasta")
        assert report["n_removed"] == 5
        assert report["query_exempted"] is None

    def test_a_gapped_query_keeps_its_alignment_columns(self, tmp_path, fake_blast):
        """The MSA is an alignment; the purge drops rows and must never touch columns."""
        core = make_protein(21)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        gapped_query = SIGNAL_PEPTIDE + core[:100] + "-" * 10 + core[100:]
        far = make_protein(22)
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [
            ("QUERY", gapped_query),
            ("DUP", SIGNAL_PEPTIDE + core[:100] + "-" * 10 + core[100:]),
            ("FAR", SIGNAL_PEPTIDE + far[:100] + "-" * 10 + far[100:]),
        ])
        purge(tmp_path, panel, msa)
        rows = list(common.read_fasta(tmp_path / "purged.fasta"))
        assert rows[0] == ("QUERY", gapped_query)
        assert all(len(seq) == len(gapped_query) for _h, seq in rows)

    def test_ungapped_output_still_asserts_the_query_survived(self, tmp_path, fake_blast):
        """``keep_alignment_columns=False`` degaps every row INCLUDING row 0.

        The survival assertion compares row 0 before and after, so it has to compare
        against the same transform; otherwise it fires on any gapped query and the
        mode is unusable.
        """
        core = make_protein(91)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        gapped_query = SIGNAL_PEPTIDE + core[:100] + "-" * 5 + core[100:]
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", gapped_query), ("FAR", make_protein(92))])
        purge(tmp_path, panel, msa, keep_alignment_columns=False)
        rows = list(common.read_fasta(tmp_path / "purged.fasta"))
        assert rows[0][0] == "QUERY"
        assert "-" not in rows[0][1]
        assert rows[0][1] == lc.ungap(gapped_query)
        assert all("-" not in seq for _h, seq in rows)

    def test_several_rows_can_be_protected(self, tmp_path, distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       protect_indices=(0, 1))
        kept = _headers(tmp_path / "purged.fasta")
        assert kept[:2] == ["QUERY", "D000"]
        assert report["n_removed"] == 3

    def test_only_the_first_protected_row_is_reported_as_the_query(
        self, tmp_path, distance_panel, fake_blast,
    ):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       protect_indices=(0, 1))
        assert report["query_exempted"]["msa_header"] == "QUERY"

    def test_a_negative_protect_index_wraps_to_the_last_row(self, tmp_path, fake_blast):
        core = make_protein(31)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("FIRST", make_protein(32)),
                                 ("LAST", SIGNAL_PEPTIDE + core)])
        report = purge(tmp_path, panel, msa, protect_indices=(-1,))
        assert _headers(tmp_path / "purged.fasta") == ["FIRST", "LAST"]
        assert report["n_removed"] == 0
        assert report["query_exempted"]["msa_header"] == "LAST"

    def test_the_protected_row_is_excluded_from_the_pre_purge_distribution(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """Otherwise every target reports a spurious 100% at the top of its distribution."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        distribution = report["msa_best_hit_identity_distribution"]
        assert report["n_msa_rows_with_any_hit"] == 7      # 8 rows minus the query
        assert distribution["n"] == 7
        assert distribution["max"] == pytest.approx(100.0)  # D000, not the query

    def test_the_row_zero_guard_is_live_not_decorative(self, tmp_path,
                                                       distance_panel, fake_blast,
                                                       monkeypatch):
        """Corrupt the degapping of row 0 between the two passes and the purge refuses.

        ``keep_alignment_columns=False`` degaps the survivors and then the inputs, and
        compares row 0 across the two.  Making those two disagree is the only way row
        0 can be silently altered, and the module must abort rather than hand ESCOTT a
        query it did not verify.
        """
        real_ungap = lc.ungap
        row0 = SIGNAL_PEPTIDE + distance_panel["core"]
        seen = {"n": 0}

        def _flaky(seq):
            if str(seq) == row0:
                seen["n"] += 1
                if seen["n"] == 3:            # 1: pre-BLAST, 2: survivors, 3: inputs
                    return real_ungap(seq)[:-1]
            return real_ungap(seq)

        monkeypatch.setattr(lc, "ungap", _flaky)
        with pytest.raises(AssertionError, match="removed or altered row 0"):
            purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                  keep_alignment_columns=False)

    def test_a_query_that_is_near_nothing_is_reported_as_such(self, tmp_path, fake_blast):
        core = make_protein(41)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(42)), ("FAR", make_protein(43))])
        report = purge(tmp_path, panel, msa)
        assert report["query_exempted"] is None
        assert report["n_removed"] == 0


@pytest.mark.requires_blast
@pytest.mark.integration
class TestQueryExemptionWithRealBlast:
    def test_the_query_survives_a_real_purge_unchanged_and_first(self, tmp_path,
                                                                 distance_panel):
        original = list(common.read_fasta(distance_panel["msa"]))
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        kept = list(common.read_fasta(tmp_path / "purged.fasta"))
        assert kept[0] == original[0]
        assert report["query_exempted"]["identity"] == pytest.approx(100.0)
        assert report["query_exempted"]["hamming"] == 0

    # The shared fixture is 56 residues of mature core, not 550, so BOTH production
    # thresholds have to be rescaled to it and the reason is worth stating once:
    #   * coverage -- a 16-aa signal peptide is 22.2% of a 72-aa deep row (on a real
    #     HA it is 2.8%), so a genuine duplicate reaches only 77.78% coverage and the
    #     95% floor would gate every hit.  Floor lowered to 70%.
    #   * hamming  -- 10 mismatches in 56 residues is 82% identity, i.e. the rule
    #     would fire on unrelated sequences.  The scale-equivalent of the production
    #     value is 0, which is also what makes "exactly the planted row" assertable.
    # Neither is a workaround for a module defect; the production geometry is tested
    # at 550 aa in TestPurgeAtKnownDistancesWithRealBlast.
    FIXTURE_SCALED = dict(min_coverage=70.0, max_hamming=0, min_identity=99.0)

    def test_the_conftest_planted_query_survives(self, tmp_path, leakage_panels):
        """The shared fixture's row 0 is the query with the signal peptide prepended.

        Its mature core is byte-identical to target row 0, so it trips both rules and
        must be exempted rather than removed.
        """
        report = lc.purge_msa_against_target(
            leakage_panels["deep_fasta"], leakage_panels["target_fasta"],
            tmp_path / "purged.fasta", tmp_path / "dropped.tsv",
            target_lineage="TEST", min_depth_after=1,
            max_removed_fraction=1.0, threads=2, workdir=tmp_path / "work",
            **self.FIXTURE_SCALED,
        )
        kept = _headers(tmp_path / "purged.fasta")
        assert kept[0] == "DEEP0000"
        assert report["query_exempted"]["msa_index"] == leakage_panels["query_row_index"]
        assert report["query_exempted"]["identity"] == pytest.approx(100.0)

    def test_the_planted_deep_leak_is_the_only_removal(self, tmp_path, leakage_panels):
        """Ground truth from the fixture: exactly one planted leak, at row 5."""
        report = lc.purge_msa_against_target(
            leakage_panels["deep_fasta"], leakage_panels["target_fasta"],
            tmp_path / "purged.fasta", tmp_path / "dropped.tsv",
            target_lineage="TEST", min_depth_after=1,
            max_removed_fraction=1.0, threads=2, workdir=tmp_path / "work",
            **self.FIXTURE_SCALED,
        )
        assert report["n_removed"] == leakage_panels["expected_deep_removals"] == 1
        rows = _read_tsv(tmp_path / "dropped.tsv")
        assert [row["msa_header"] for row in rows] == [
            leakage_panels["planted_deep_accession"]
        ]
        assert int(rows[0]["msa_index"]) == leakage_panels["planted_deep_row"]
        assert float(rows[0]["identity"]) == 100.0

    def test_a_clean_deep_set_loses_nothing(self, tmp_path, leakage_panels):
        """The false-positive control: without the plant, nothing is removed."""
        report = lc.purge_msa_against_target(
            leakage_panels["deep_clean_fasta"], leakage_panels["target_fasta"],
            tmp_path / "purged.fasta", tmp_path / "dropped.tsv",
            target_lineage="TEST", min_depth_after=1,
            max_removed_fraction=1.0, threads=2, workdir=tmp_path / "work",
            **self.FIXTURE_SCALED,
        )
        assert report["n_removed"] == 0
        assert _read_tsv(tmp_path / "dropped.tsv") == []
        assert _headers(tmp_path / "purged.fasta") == [f"DEEP{i:04d}" for i in range(12)]

    def test_the_conftest_geometry_is_governed_by_the_coverage_gate_at_defaults(
        self, tmp_path, leakage_panels,
    ):
        """Documents a real limit of the shared fixture rather than hiding it.

        The fixture's mature core is 56 aa, so a signal-peptide-offset duplicate
        reaches only 56/72 = 77.78% coverage and the 95% floor gates it.  The
        planted leak is therefore NOT removed at the production defaults; that is a
        property of the fixture's scale, not of the module, and the near-miss
        counter is what makes it visible instead of silent.
        """
        report = lc.purge_msa_against_target(
            leakage_panels["deep_fasta"], leakage_panels["target_fasta"],
            tmp_path / "purged.fasta", tmp_path / "dropped.tsv",
            target_lineage="TEST", min_depth_after=1, max_removed_fraction=1.0,
            threads=2, workdir=tmp_path / "work",
        )
        assert report["n_removed"] == 0
        assert report["n_coverage_gated_near_misses"] >= 1
        assert report["coverage_gated_near_miss_max_coverage"] == pytest.approx(
            100 * 56 / 72, abs=0.01
        )
        assert any("coverage gate" in w for w in report["warnings"])


# =========================================================================== #
# THE PURGE -- the coverage gate
# =========================================================================== #

@pytest.mark.unit
class TestCoverageGate:
    def test_a_short_high_identity_match_is_not_a_duplicate(self, tmp_path,
                                                            panel_and_deep, fake_blast):
        """An exact 200-aa substring is a fragment, not a leak."""
        panel, deep, _ = panel_and_deep
        purge(tmp_path, panel, deep)
        assert "DEEP_frag" in _headers(tmp_path / "purged.fasta")

    def test_the_shorter_basis_WOULD_drop_the_fragment(self, tmp_path,
                                                       panel_and_deep, fake_blast):
        """Demonstrates why 'both' is the default rather than merely asserting it."""
        panel, deep, _ = panel_and_deep
        purge(tmp_path, panel, deep, coverage_basis="shorter")
        assert "DEEP_frag" not in _headers(tmp_path / "purged.fasta")

    def test_the_query_basis_lets_a_real_duplicate_escape_at_a_tight_floor(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """A 566-aa deep row covering a 550-aa panel row is only 97.2% of the query.

        Raise the floor to 98% and the 'query' basis loses the exact duplicate --
        the second failure mode that made 'both' the default.
        """
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       coverage_basis="query", min_coverage=98.0)
        assert report["n_removed"] == 0
        assert report["n_coverage_gated_near_misses"] >= 1

    def test_the_gate_is_reported_when_it_and_not_the_thresholds_decides(
        self, tmp_path, panel_and_deep, fake_blast, capsys,
    ):
        panel, deep, _ = panel_and_deep
        report = purge(tmp_path, panel, deep)
        assert report["n_coverage_gated_near_misses"] == 1
        assert report["coverage_gated_near_miss_max_coverage"] == pytest.approx(
            100 * 200 / 550, abs=0.1
        )
        assert report["coverage_gated_near_misses"][0]["msa_header"] == "DEEP_frag"
        assert any("coverage gate" in w for w in report["warnings"])
        assert "coverage gate" in capsys.readouterr().err

    def test_no_near_misses_when_nothing_is_gated(self, tmp_path, fake_blast):
        core = make_protein(71)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core), ("FAR", make_protein(72))])
        report = purge(tmp_path, panel, msa)
        assert report["n_coverage_gated_near_misses"] == 0
        assert report["coverage_gated_near_miss_max_coverage"] is None
        assert report["warnings"] == []

    def test_near_misses_are_capped_in_the_report_but_not_in_the_count(
        self, tmp_path, scripted_blast,
    ):
        core = make_protein(1)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT0001", core)])
        rows = [(f"R{i:03d}", core[50:250]) for i in range(60)]
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, rows)
        scripted_blast([
            make_hit(qheader=str(i), sheader="TGT0001", aln_len=200, nident=200,
                     qlen=200, slen=550)
            for i in range(60)
        ])
        report = purge(tmp_path, panel, msa, protect_indices=(), max_hamming=0)
        assert report["n_coverage_gated_near_misses"] == 60
        assert len(report["coverage_gated_near_misses"]) == 50

    @pytest.mark.requires_blast
    @pytest.mark.integration
    def test_real_blast_agrees_that_a_fragment_is_gated(self, tmp_path, panel_and_deep):
        panel, deep, _ = panel_and_deep
        report = purge(tmp_path, panel, deep)
        assert "DEEP_frag" in _headers(tmp_path / "purged.fasta")
        assert report["coverage_gated_near_misses"][0]["msa_header"] == "DEEP_frag"


# =========================================================================== #
# THE PURGE -- indels
# =========================================================================== #

@pytest.mark.requires_blast
@pytest.mark.integration
class TestIndels:
    """A 3-residue deletion, through the real aligner, on both metrics.

    Both metrics must agree on the DECISION (drop), and the numbers they report
    must be the documented ones.  The asymmetry between a deletion and an insertion
    is real and is asserted rather than glossed: ``hamming = min(qlen,slen) -
    nident`` charges gap columns only when the gap falls in the sequence that is
    being used as the denominator.  That is the same property that stops a
    16-residue signal peptide being charged as 16 phantom mismatches, so it cannot
    be "fixed" without breaking the frame handling.
    """

    @pytest.fixture()
    def indel_fixture(self, tmp_path: Path) -> Dict[str, object]:
        core = make_protein(15)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [
            ("QUERY", SIGNAL_PEPTIDE + core),
            ("DEL3", SIGNAL_PEPTIDE + core[:200] + core[203:]),
            ("INS3", SIGNAL_PEPTIDE + core[:200] + "WWW" + core[200:]),
        ])
        return {"panel": panel, "msa": msa, "core": core}

    def test_a_three_residue_deletion_is_dropped_by_both_metrics(self, tmp_path,
                                                                 indel_fixture):
        report = purge(tmp_path, indel_fixture["panel"], indel_fixture["msa"])
        rows = {r["msa_header"]: r for r in _read_tsv(tmp_path / "dropped.tsv")}
        assert set(rows) == {"DEL3", "INS3"}
        assert rows["DEL3"]["rule"] == "identity+hamming"
        assert report["n_removed"] == 2

    def test_deletion_gap_columns_count_toward_both_identity_and_hamming(
        self, tmp_path, indel_fixture,
    ):
        """qlen 563 (16 + 547), slen 550, nident 547 -> hamming 3, identity 547/550."""
        purge(tmp_path, indel_fixture["panel"], indel_fixture["msa"])
        row = next(r for r in _read_tsv(tmp_path / "dropped.tsv") if r["msa_header"] == "DEL3")
        assert int(row["hamming"]) == 3
        assert int(row["qlen"]) == 563 and int(row["slen"]) == 550
        assert int(row["nident"]) == 547
        assert float(row["identity"]) == pytest.approx(100 * 547 / 550, abs=1e-3)

    def test_an_insertion_costs_identity_but_not_hamming(self, tmp_path, indel_fixture):
        """The documented consequence of the min() denominator, pinned explicitly.

        The three inserted residues exist only in the longer sequence, so they are
        overhang, and overhang in the LONGER sequence is exactly what the signal
        peptide is.  Identity still records the 3 gap columns.
        """
        purge(tmp_path, indel_fixture["panel"], indel_fixture["msa"])
        row = next(r for r in _read_tsv(tmp_path / "dropped.tsv") if r["msa_header"] == "INS3")
        assert int(row["hamming"]) == 0
        assert int(row["nident"]) == 550 and int(row["aln_len"]) == 553
        assert float(row["identity"]) == pytest.approx(100 * 550 / 553, abs=1e-3)
        assert row["rule"] == "identity+hamming"

    def test_both_indels_pass_the_coverage_gate(self, tmp_path, indel_fixture):
        report = purge(tmp_path, indel_fixture["panel"], indel_fixture["msa"])
        assert report["n_coverage_gated_near_misses"] == 0
        for row in _read_tsv(tmp_path / "dropped.tsv"):
            assert float(row["coverage"]) >= lc.DEFAULT_MIN_COVERAGE

    def test_an_indel_only_difference_survives_a_tighter_hamming_rule_consistently(
        self, tmp_path, indel_fixture,
    ):
        """With identity off and hamming 0, the insertion (0) goes and the deletion (3) stays."""
        purge(tmp_path, indel_fixture["panel"], indel_fixture["msa"],
              min_identity=None, max_hamming=0)
        kept = _headers(tmp_path / "purged.fasta")
        assert kept == ["QUERY", "DEL3"]


# =========================================================================== #
# THE PURGE -- the audit trail
# =========================================================================== #

@pytest.mark.unit
class TestAuditTrail:
    def test_removed_plus_retained_reconstructs_the_input_exactly(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """No losses, no duplicates, no reordering."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        kept = _headers(tmp_path / "purged.fasta")
        dropped = [r["msa_header"] for r in _read_tsv(tmp_path / "dropped.tsv")]
        assert len(kept) == len(set(kept))
        assert len(dropped) == len(set(dropped))
        assert set(kept).isdisjoint(dropped)
        assert sorted(kept + dropped) == sorted(distance_panel["headers"])
        assert len(kept) + len(dropped) == report["depth_before"]

    def test_survivors_keep_their_input_order(self, tmp_path, distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        kept = _headers(tmp_path / "purged.fasta")
        original = distance_panel["headers"]
        assert kept == [h for h in original if h in set(kept)]

    def test_every_manifest_row_names_its_evidence(self, tmp_path,
                                                   distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        rows = _read_tsv(tmp_path / "dropped.tsv")
        assert len(rows) == 4
        for row in rows:
            assert row["msa_header"].startswith("D")
            assert row["matched_target_sequence"] == "TGT0001"
            assert row["matched_target_accession"] == "TGT0001"
            assert row["target_lineage"] == "TEST"
            assert row["rule"] in {"identity", "hamming", "identity+hamming"}
            assert float(row["identity"]) >= 98.0
            assert 0 <= int(row["hamming"]) <= 10
            assert float(row["coverage"]) >= lc.DEFAULT_MIN_COVERAGE
            assert int(row["nident"]) > 0 and int(row["aln_len"]) > 0
            assert float(row["bitscore"]) > 0

    def test_the_manifest_index_matches_the_input_row_number(self, tmp_path,
                                                             distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        original = distance_panel["headers"]
        for row in _read_tsv(tmp_path / "dropped.tsv"):
            assert original[int(row["msa_index"])] == row["msa_header"]

    def test_the_manifest_has_every_documented_column_in_order(self, tmp_path,
                                                               distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        header = (tmp_path / "dropped.tsv").read_text(
            encoding="utf-8").splitlines()[0].split("\t")
        assert header == [
            "msa_index", "msa_header", "msa_accession", "target_lineage",
            "matched_target_sequence", "matched_target_accession",
            "matched_target_panel_copies", "rule", "identity", "coverage", "hamming",
            "aln_len", "nident", "qlen", "slen", "bitscore", "evalue",
        ]

    def test_the_manifest_is_sorted_closest_first(self, tmp_path,
                                                  distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        identities = [float(r["identity"]) for r in _read_tsv(tmp_path / "dropped.tsv")]
        assert identities == sorted(identities, reverse=True)
        assert identities[0] == 100.0

    def test_the_manifest_is_written_even_when_nothing_was_removed(self, tmp_path,
                                                                   fake_blast):
        core = make_protein(61)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core), ("FAR", make_protein(62))])
        purge(tmp_path, panel, msa)
        text = (tmp_path / "dropped.tsv").read_text(encoding="utf-8")
        assert text.startswith("msa_index\tmsa_header")
        assert len(text.strip().splitlines()) == 1

    def test_panel_multiplicity_is_reported_not_deduped_away(self, tmp_path, fake_blast):
        """The panel collapses ~5x before BLAST; a hit must still report its true count."""
        core = make_protein(63)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core), ("TGT2", core), ("TGT3", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(64)), ("DUP", SIGNAL_PEPTIDE + core)])
        report = purge(tmp_path, panel, msa)
        assert report["target_panel_records"] == 3
        assert report["target_panel_unique"] == 1
        row = _read_tsv(tmp_path / "dropped.tsv")[0]
        assert int(row["matched_target_panel_copies"]) == 3

    def test_the_report_records_both_file_digests(self, tmp_path, distance_panel,
                                                  fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert report["msa_md5_before"] == common.md5_file(distance_panel["msa"])
        assert report["purged_md5"] == common.md5_file(tmp_path / "purged.fasta")
        assert report["msa_md5_before"] != report["purged_md5"]
        assert report["drop_manifest_path"] == str(tmp_path / "dropped.tsv")
        assert report["purged_path"] == str(tmp_path / "purged.fasta")

    def test_the_rule_counters_add_up_to_the_removals(self, tmp_path,
                                                      distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert (report["n_removed_by_identity_only"]
                + report["n_removed_by_hamming_only"]
                + report["n_removed_by_both"]) == report["n_removed"]

    def test_removed_hamming_range_and_distribution_describe_the_removals(
        self, tmp_path, distance_panel, fake_blast,
    ):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert report["removed_hamming_min"] == 0
        assert report["removed_hamming_max"] == 10
        assert report["removed_identity_distribution"]["n"] == 4
        assert report["removed_identity_distribution"]["max"] == 100.0

    def test_an_empty_removal_set_reports_none_not_zero(self, tmp_path, fake_blast):
        core = make_protein(65)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(66)), ("FAR", make_protein(67))])
        report = purge(tmp_path, panel, msa)
        assert report["removed_hamming_min"] is None
        assert report["removed_hamming_max"] is None
        assert report["removed_identity_distribution"]["n"] == 0

    def test_thresholds_are_echoed_into_the_report(self, tmp_path,
                                                   distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        thresholds = report["thresholds"]
        assert thresholds["combination"].startswith("OR")
        assert "STRICTER" in thresholds["note"]
        assert thresholds["min_identity"] == lc.DEFAULT_MIN_IDENTITY
        assert thresholds["max_hamming"] == lc.DEFAULT_MAX_HAMMING
        assert thresholds["min_coverage"] == lc.DEFAULT_MIN_COVERAGE
        assert thresholds["coverage_basis"] == "both"

    def test_the_report_is_json_serialisable(self, tmp_path, distance_panel, fake_blast):
        """It is replayed verbatim into inputs_manifest.json."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert json.loads(json.dumps(report))["n_removed"] == 4


# =========================================================================== #
# THE PURGE -- depth guards
# =========================================================================== #

@pytest.mark.unit
class TestDepthGuards:
    def test_depth_accounting_is_exact(self, tmp_path, distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"])
        assert report["depth_before"] == 8
        assert report["depth_after"] == 4
        assert report["n_removed"] == report["depth_before"] - report["depth_after"]
        assert report["removed_fraction"] == pytest.approx(0.5)

    def test_over_removal_warns_loudly_on_stderr_and_in_the_report(
        self, tmp_path, distance_panel, fake_blast, capsys,
    ):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       max_removed_fraction=0.25)
        assert any("above --leakage-max-removed-fraction" in w for w in report["warnings"])
        stderr = capsys.readouterr().err
        assert "LEAKAGE PURGE" in stderr and "GEMME/ESCOTT quality" in stderr

    def test_a_removal_fraction_exactly_at_the_limit_does_not_warn(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """The warning is ``>``, not ``>=``: 50% removed at a 50% limit is allowed."""
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       max_removed_fraction=0.5)
        assert report["removed_fraction"] == pytest.approx(0.5)
        assert not any("max-removed-fraction" in w for w in report["warnings"])

    def test_the_depth_floor_raises_rather_than_returning_a_thin_alignment(
        self, tmp_path, distance_panel, fake_blast,
    ):
        with pytest.raises(lc.MsaDepthError) as excinfo:
            purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                  min_depth_after=5)
        message = str(excinfo.value)
        assert "purge left 4 rows" in message
        assert "5" in message

    def test_the_depth_floor_is_inclusive(self, tmp_path, distance_panel, fake_blast):
        report = purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                       min_depth_after=4)
        assert report["depth_after"] == 4

    def test_the_purged_file_is_still_written_before_the_depth_error(
        self, tmp_path, distance_panel, fake_blast,
    ):
        """So an operator can inspect what the thresholds actually did."""
        with pytest.raises(lc.MsaDepthError):
            purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                  min_depth_after=5)
        assert (tmp_path / "purged.fasta").exists()
        assert _read_tsv(tmp_path / "dropped.tsv")

    def test_both_warnings_can_fire_together(self, tmp_path, panel_and_deep,
                                             fake_blast, capsys):
        panel, deep, _ = panel_and_deep
        report = purge(tmp_path, panel, deep, max_removed_fraction=0.01)
        assert len(report["warnings"]) == 2
        assert any("coverage gate" in w for w in report["warnings"])
        assert any("max-removed-fraction" in w for w in report["warnings"])


# =========================================================================== #
# THE PURGE -- edge cases
# =========================================================================== #

@pytest.mark.unit
class TestPurgeEdgeCases:
    def test_an_empty_msa_is_refused(self, tmp_path, fake_blast):
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", make_protein(1))])
        msa = tmp_path / "empty.fasta"
        msa.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            purge(tmp_path, panel, msa)

    def test_an_empty_target_panel_is_refused(self, tmp_path, fake_blast):
        panel = tmp_path / "panel.fasta"
        panel.write_text("", encoding="utf-8")
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(1))])
        with pytest.raises(ValueError, match="contains no records"):
            purge(tmp_path, panel, msa)

    def test_a_panel_of_pure_gap_rows_counts_as_empty(self, tmp_path, fake_blast):
        """``load_ungapped`` drops empty records, so this must not become a BLAST call."""
        panel = tmp_path / "panel.fasta"
        panel.write_text(">TGT1\n-----\n>TGT2\n.....\n", encoding="utf-8")
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(1))])
        with pytest.raises(ValueError, match="contains no records"):
            purge(tmp_path, panel, msa)

    def test_a_single_row_msa_that_is_the_query_survives_intact(self, tmp_path, fake_blast):
        """The degenerate but real case: an alignment of one, which must not be emptied."""
        core = make_protein(81)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core)])
        report = purge(tmp_path, panel, msa)
        assert report["depth_before"] == report["depth_after"] == 1
        assert report["n_removed"] == 0
        assert report["query_exempted"]["identity"] == pytest.approx(100.0)
        assert _headers(tmp_path / "purged.fasta") == ["QUERY"]

    def test_a_single_unprotected_duplicate_empties_the_alignment_and_raises(
        self, tmp_path, fake_blast,
    ):
        core = make_protein(82)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("DUP", SIGNAL_PEPTIDE + core)])
        with pytest.raises(lc.MsaDepthError, match="purge left 0 rows"):
            purge(tmp_path, panel, msa, protect_indices=())
        assert _headers(tmp_path / "purged.fasta") == []

    def test_a_single_sequence_panel_is_a_valid_database(self, tmp_path,
                                                         distance_panel, fake_blast):
        assert len(lc.load_ungapped(distance_panel["panel"])) == 1
        assert purge(tmp_path, distance_panel["panel"],
                     distance_panel["msa"])["target_panel_unique"] == 1

    def test_msa_rows_are_ungapped_before_blast_but_written_back_aligned(
        self, tmp_path, fake_blast,
    ):
        core = make_protein(83)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        aligned_row = SIGNAL_PEPTIDE + core[:100] + "-" * 25 + core[100:]
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(84)), ("DUP", aligned_row)])
        purge(tmp_path, panel, msa)
        sent = fake_blast[0]["query"]
        assert all("-" not in seq for _h, seq in sent)
        assert dict(sent)["1"] == lc.ungap(aligned_row)
        assert dict(common.read_fasta(msa))["DUP"] == aligned_row

    def test_the_target_panel_is_ungapped_before_blast(self, tmp_path, fake_blast):
        """Panels arrive as 594-column alignments; makeblastdb must never see a gap."""
        core = make_protein(85)
        aligned_panel_row = core[:200] + "-" * 44 + core[200:]
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", aligned_panel_row)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(86)), ("DUP", SIGNAL_PEPTIDE + core)])
        report = purge(tmp_path, panel, msa)
        subject = fake_blast[0]["subject"]
        assert subject == [("TGT1", core)]
        assert all("-" not in seq for _h, seq in subject)
        assert report["n_removed"] == 1

    @pytest.mark.requires_blast
    @pytest.mark.integration
    def test_an_aligned_panel_gives_the_same_answer_as_an_unaligned_one(
        self, tmp_path, distance_panel,
    ):
        core = distance_panel["core"]
        aligned = tmp_path / "aligned_panel.fasta"
        common.write_fasta(aligned, [("TGT0001", core[:300] + "-" * 44 + core[300:])])
        plain = purge(tmp_path / "plain", distance_panel["panel"], distance_panel["msa"])
        gapped = purge(tmp_path / "gapped", aligned, distance_panel["msa"])
        assert plain["n_removed"] == gapped["n_removed"] == 4
        assert _headers(tmp_path / "plain" / "purged.fasta") == \
            _headers(tmp_path / "gapped" / "purged.fasta")

    def test_a_temporary_workdir_is_created_and_removed(self, tmp_path,
                                                        distance_panel, fake_blast):
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"], workdir=None)
        used = fake_blast[0]["workdir"]
        assert "leakpurge_" in used.name
        assert not used.exists(), "the module must clean up the temp dir it owns"

    def test_a_supplied_workdir_is_left_alone(self, tmp_path, distance_panel, fake_blast):
        workdir = tmp_path / "keepme"
        purge(tmp_path, distance_panel["panel"], distance_panel["msa"], workdir=workdir)
        assert workdir.exists()

    def test_the_manifest_directory_is_created_if_missing(self, tmp_path,
                                                          distance_panel, fake_blast):
        out = tmp_path / "nested" / "deeper" / "dropped.tsv"
        lc.purge_msa_against_target(
            distance_panel["msa"], distance_panel["panel"],
            tmp_path / "nested" / "purged.fasta", out,
            target_lineage="TEST", min_depth_after=1, max_removed_fraction=1.0,
            workdir=tmp_path / "w",
        )
        assert out.exists()

    def test_rows_with_no_hit_at_all_are_kept(self, tmp_path, scripted_blast):
        core = make_protein(87)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        msa = tmp_path / "msa.fasta"
        common.write_fasta(msa, [("QUERY", make_protein(88)), ("A", make_protein(89)),
                                 ("B", make_protein(90))])
        scripted_blast([])
        report = purge(tmp_path, panel, msa)
        assert report["n_removed"] == 0
        assert report["n_msa_rows_with_any_hit"] == 0
        assert report["msa_best_hit_identity_distribution"]["n"] == 0
        assert _headers(tmp_path / "purged.fasta") == ["QUERY", "A", "B"]

    def test_the_target_lineage_falls_back_to_the_msa_stem_in_messages(
        self, tmp_path, distance_panel, fake_blast,
    ):
        with pytest.raises(lc.MsaDepthError, match="msa_dist"):
            purge(tmp_path, distance_panel["panel"], distance_panel["msa"],
                  target_lineage="", min_depth_after=99)


# =========================================================================== #
# purge_lineage_msa -- in-place purge with a cache
# =========================================================================== #

@pytest.mark.unit
class TestPurgeLineageMsa:
    def _run(self, tmp_path, panel, msa, **kwargs):
        options = dict(min_depth_after=1, max_removed_fraction=1.0, threads=2)
        options.update(kwargs)
        return lc.purge_lineage_msa(msa, panel, "TEST", tmp_path / "leakage", **options)

    def test_purges_in_place_and_keeps_the_original(self, tmp_path,
                                                    distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        report = self._run(tmp_path, distance_panel["panel"], msa)

        prepurge = tmp_path / "msa_TEST_prepurge.fasta"
        assert prepurge.exists()
        assert _headers(prepurge) == distance_panel["headers"]
        assert _headers(msa) == list(EXPECTED_KEPT_AT_DEFAULTS)
        assert report["prepurge_path"] == str(prepurge)
        assert report["prepurge_md5"] == common.md5_file(prepurge)
        assert report["purged_path"] == str(msa)
        assert report["purged_md5"] == common.md5_file(msa)
        assert report["cached"] is False

    def test_no_temporary_file_is_left_next_to_the_msa(self, tmp_path,
                                                       distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        assert not (tmp_path / "msa_TEST.purged.tmp").exists()

    def test_the_audit_manifest_lands_in_the_leakage_dir(self, tmp_path,
                                                         distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        assert (tmp_path / "leakage" / "TEST_purge_dropped.tsv").exists()
        assert (tmp_path / "leakage" / "TEST_purge.json").exists()

    def test_cache_hit_on_a_second_identical_call(self, tmp_path,
                                                  distance_panel, fake_blast, capsys):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        first = self._run(tmp_path, distance_panel["panel"], msa)
        capsys.readouterr()
        second = self._run(tmp_path, distance_panel["panel"], msa)
        assert second["cached"] is True
        assert first["purged_md5"] == second["purged_md5"]
        assert "cache hit" in capsys.readouterr().out
        assert len(fake_blast) == 1, "a cache hit must not re-run BLAST"

    def test_force_bypasses_the_cache(self, tmp_path, distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        again = self._run(tmp_path, distance_panel["panel"], msa, force=True)
        assert again["cached"] is False
        assert len(fake_blast) == 2

    def test_changed_thresholds_invalidate_the_cache(self, tmp_path,
                                                     distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        loosened = self._run(tmp_path, distance_panel["panel"], msa, max_hamming=None)
        assert loosened["cached"] is False
        assert loosened["n_removed"] == 3

    def test_cache_key_records_defaulted_thresholds_too(self, tmp_path,
                                                        distance_panel, fake_blast):
        """Regression: thresholds taken from the DEFAULTS must reach the cache key.

        They used to be harvested out of ``**kwargs``, so a call that relied on the
        defaults cached under an empty threshold set -- and would then be served that
        cache after leakage_check.py's defaults changed, silently handing ESCOTT an
        alignment purged at the old stringency.
        """
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)   # no threshold passed
        cached = json.loads((tmp_path / "leakage" / "TEST_purge.json").read_text())
        thresholds = cached["cache_key"]["thresholds"]
        assert thresholds["min_identity"] == lc.DEFAULT_MIN_IDENTITY
        assert thresholds["max_hamming"] == lc.DEFAULT_MAX_HAMMING
        assert thresholds["min_coverage"] == lc.DEFAULT_MIN_COVERAGE
        assert thresholds["coverage_basis"] == "both"
        assert thresholds["protect_indices"] == [0]
        assert cached["cache_key"]["version"] == 2

    def test_the_cache_key_is_versioned_against_schema_drift(self, tmp_path,
                                                             distance_panel, fake_blast):
        """A v1 cache replayed into v2 code would produce a manifest missing fields."""
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        cache_json = tmp_path / "leakage" / "TEST_purge.json"
        payload = json.loads(cache_json.read_text())
        payload["cache_key"]["version"] = 1
        cache_json.write_text(json.dumps(payload), encoding="utf-8")
        assert self._run(tmp_path, distance_panel["panel"], msa)["cached"] is False

    def test_a_changed_source_msa_invalidates_the_cache(self, tmp_path,
                                                        distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        first = self._run(tmp_path, distance_panel["panel"], msa)
        rows = list(common.read_fasta(tmp_path / "msa_TEST_prepurge.fasta"))
        common.write_fasta(tmp_path / "msa_TEST_prepurge.fasta",
                           rows + [("EXTRA", make_protein(81))])
        second = self._run(tmp_path, distance_panel["panel"], msa)
        assert second["cached"] is False
        assert second["depth_before"] == first["depth_before"] + 1

    def test_a_changed_panel_invalidates_the_cache(self, tmp_path,
                                                   distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        common.write_fasta(distance_panel["panel"],
                           [("TGT0001", distance_panel["core"]), ("TGT0002", make_protein(7))])
        assert self._run(tmp_path, distance_panel["panel"], msa)["cached"] is False

    def test_a_tampered_output_invalidates_the_cache(self, tmp_path,
                                                     distance_panel, fake_blast):
        """The cache also verifies the md5 of the file it claims to have produced."""
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        msa.write_text(msa.read_text(encoding="utf-8") + ">JUNK\nACDEF\n", encoding="utf-8")
        assert self._run(tmp_path, distance_panel["panel"], msa)["cached"] is False

    def test_a_corrupt_cache_file_degrades_to_recompute(self, tmp_path,
                                                        distance_panel, fake_blast):
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        self._run(tmp_path, distance_panel["panel"], msa)
        (tmp_path / "leakage" / "TEST_purge.json").write_text("{ not json",
                                                             encoding="utf-8")
        assert self._run(tmp_path, distance_panel["panel"], msa)["cached"] is False

    def test_the_prepurge_copy_is_the_source_on_every_rerun(self, tmp_path,
                                                            distance_panel, fake_blast):
        """Re-purging a purged file would shrink the alignment on every run."""
        msa = tmp_path / "msa_TEST.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        first = self._run(tmp_path, distance_panel["panel"], msa)
        second = self._run(tmp_path, distance_panel["panel"], msa, force=True)
        assert second["depth_before"] == first["depth_before"] == 8
        assert second["depth_after"] == first["depth_after"]

    def test_the_key_is_safe_labelled(self, tmp_path, distance_panel, fake_blast):
        msa = tmp_path / "msa_x.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        lc.purge_lineage_msa(msa, distance_panel["panel"], "J.2/int",
                             tmp_path / "leakage", min_depth_after=1,
                             max_removed_fraction=1.0)
        assert (tmp_path / "leakage" / "J.2-int_purge.json").exists()


# =========================================================================== #
# Checks A and B
# =========================================================================== #

@pytest.mark.unit
class TestCheckAAndB:
    def test_check_a_counts_hits_flags_and_exact_matches(self, tmp_path,
                                                         panel_and_deep, fake_blast):
        panel, deep, headers = panel_and_deep
        check = lc.check_msa_vs_target(lc.load_ungapped(deep), panel, "TEST",
                                       tmp_path / "work")
        assert check.name == "A_msa_vs_target"
        assert check.target_lineage == "TEST"
        assert check.subject_label == "panel:TEST"
        assert check.n_query == len(headers)
        assert check.n_subject == 3 and check.n_subject_unique == 3
        assert check.distribution["max"] == pytest.approx(100.0)
        assert check.n_flagged >= 2
        assert all(row["identity"] >= lc.DEFAULT_FLAG_IDENTITY for row in check.flagged)

    def test_check_a_exact_full_length_needs_both_lengths_to_match(self, tmp_path,
                                                                   fake_blast):
        """A signal-peptide-offset duplicate is NOT an exact full-length match."""
        core = make_protein(101)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        offset = lc.check_msa_vs_target([("deep", SIGNAL_PEPTIDE + core)], panel,
                                        "TEST", tmp_path / "w1")
        identical = lc.check_msa_vs_target([("deep", core)], panel, "TEST",
                                           tmp_path / "w2")
        assert offset.n_exact_full_length == 0
        assert identical.n_exact_full_length == 1

    def test_check_a_report_rows_are_capped_but_the_count_is_not(self, tmp_path,
                                                                 fake_blast):
        core = make_protein(102)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        deep = [(f"D{i:03d}", core) for i in range(10)]
        check = lc.check_msa_vs_target(deep, panel, "TEST", tmp_path / "w",
                                       max_report_rows=3)
        assert check.n_flagged == 10
        assert len(check.flagged) == 3

    def test_check_a_with_the_identity_filter_off_flags_everything_over_coverage(
        self, tmp_path, fake_blast,
    ):
        core = make_protein(103)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        deep = [("close", core), ("far", substitute(core, 60, stride=8))]
        check = lc.check_msa_vs_target(deep, panel, "TEST", tmp_path / "w",
                                       flag_identity=None)
        assert check.n_flagged == 2

    def test_check_a_coverage_gate_excludes_a_fragment_from_the_flags(self, tmp_path,
                                                                      fake_blast):
        core = make_protein(104)
        panel = tmp_path / "panel.fasta"
        common.write_fasta(panel, [("TGT1", core)])
        check = lc.check_msa_vs_target([("frag", core[50:250])], panel, "TEST",
                                       tmp_path / "w")
        assert check.n_flagged == 0
        assert check.n_exact_full_length == 0
        assert check.flagged == []

    def test_check_a_thresholds_are_echoed(self, tmp_path, panel_and_deep, fake_blast):
        panel, deep, _ = panel_and_deep
        check = lc.check_msa_vs_target(lc.load_ungapped(deep), panel, "TEST",
                                       tmp_path / "w", flag_identity=98.0,
                                       flag_coverage=90.0, coverage_basis="shorter")
        assert check.thresholds == {"flag_identity": 98.0, "flag_coverage": 90.0,
                                    "coverage_basis": "shorter"}

    def test_check_b_finds_shared_accessions_in_the_one_valid_namespace(
        self, tmp_path, fake_blast,
    ):
        """Parent and target panels are cut from the same alignment, so IDs match."""
        core = make_protein(31)
        parent, target = tmp_path / "parent.fasta", tmp_path / "target.fasta"
        common.write_fasta(parent, [("SHARED1", core), ("PONLY", make_protein(32))])
        common.write_fasta(target, [("SHARED1", core), ("TONLY", substitute(core, 30))])
        check, accessions = lc.check_parent_vs_target(parent, target, "CHILD",
                                                      "PARENT", tmp_path / "work")
        assert accessions["n_shared_accessions"] == 1
        assert accessions["shared_accessions_sample"] == ["SHARED1"]
        assert accessions["n_parent_accessions"] == 2
        assert accessions["n_target_accessions"] == 2
        assert accessions["shared_accession_fraction_of_target"] == pytest.approx(0.5)
        assert accessions["n_shared_exact_sequences"] == 1
        assert accessions["shared_exact_sequence_fraction_of_target"] == pytest.approx(0.5)
        assert check.name == "B_parent_vs_target"
        assert check.query_label == "panel:PARENT"
        assert check.n_flagged >= 1

    def test_check_b_is_clean_when_the_panels_are_disjoint(self, tmp_path, fake_blast):
        parent, target = tmp_path / "parent.fasta", tmp_path / "target.fasta"
        common.write_fasta(parent, [("P1", make_protein(41))])
        common.write_fasta(target, [("T1", make_protein(42))])
        _check, accessions = lc.check_parent_vs_target(parent, target, "CHILD",
                                                       "PARENT", tmp_path / "work")
        assert accessions["n_shared_accessions"] == 0
        assert accessions["n_shared_exact_sequences"] == 0
        assert accessions["shared_accessions_sample"] == []

    def test_check_b_catches_the_same_sequence_under_a_different_accession(
        self, tmp_path, leakage_panels, fake_blast,
    ):
        """The conftest plant: no shared ID, one shared sequence.

        Accession matching alone would call this clean, which is exactly why the
        hash and the alignment are run as well.
        """
        _check, accessions = lc.check_parent_vs_target(
            leakage_panels["parent_fasta"], leakage_panels["target_fasta"],
            "target", "parent", tmp_path / "work",
        )
        assert accessions["n_shared_accessions"] == \
            leakage_panels["expected_shared_accessions"] == 0
        assert accessions["n_shared_exact_sequences"] == \
            leakage_panels["expected_shared_exact_sequences"] == 1

    def test_check_b_accession_sample_is_capped_at_100(self, tmp_path, fake_blast):
        records = [(f"ACC{i:04d}", make_protein(i % 7 + 1)) for i in range(150)]
        parent, target = tmp_path / "p.fasta", tmp_path / "t.fasta"
        common.write_fasta(parent, records)
        common.write_fasta(target, records)
        _check, accessions = lc.check_parent_vs_target(parent, target, "C", "P",
                                                       tmp_path / "work")
        assert accessions["n_shared_accessions"] == 150
        assert len(accessions["shared_accessions_sample"]) == 100

    def test_check_b_deduplicates_both_sides_before_blast(self, tmp_path, fake_blast):
        core = make_protein(105)
        parent, target = tmp_path / "p.fasta", tmp_path / "t.fasta"
        common.write_fasta(parent, [("P1", core), ("P2", core), ("P3", core)])
        common.write_fasta(target, [("T1", core), ("T2", core)])
        check, _accessions = lc.check_parent_vs_target(parent, target, "C", "P",
                                                       tmp_path / "work")
        assert check.n_query == 3 and check.n_subject == 2
        assert check.n_subject_unique == 1
        assert len(fake_blast[0]["query"]) == 1
        assert len(fake_blast[0]["subject"]) == 1

    def test_check_b_reports_panel_multiplicity_of_the_matched_target(self, tmp_path,
                                                                      fake_blast):
        core = make_protein(106)
        parent, target = tmp_path / "p.fasta", tmp_path / "t.fasta"
        common.write_fasta(parent, [("P1", core)])
        common.write_fasta(target, [("T1", core), ("T2", core), ("T3", core)])
        check, _ = lc.check_parent_vs_target(parent, target, "C", "P", tmp_path / "work")
        assert check.flagged[0]["subject_panel_copies"] == 3


# =========================================================================== #
# run_leakage_stage -- what prepare_inputs.py calls
# =========================================================================== #

@pytest.mark.unit
class TestRunLeakageStage:
    def _panels(self, tmp_path: Path):
        core = make_protein(51)
        target = tmp_path / "target.fasta"
        parent = tmp_path / "parent.fasta"
        common.write_fasta(target, [("TGT1", core), ("TGT2", substitute(core, 30))])
        common.write_fasta(parent, [("PAR1", substitute(core, 45, stride=11))])
        msa = tmp_path / "msa_CHILD.fasta"
        common.write_fasta(msa, [
            ("QUERY", SIGNAL_PEPTIDE + core),
            ("D_exact", SIGNAL_PEPTIDE + core),
            ("D_far", make_protein(52)),
        ])
        return target, parent, msa

    def _stage(self, tmp_path, target, parent, msa, **kwargs):
        options = dict(
            inputs_dir=tmp_path,
            panel_by_label={"CHILD": target, "PARENT": parent},
            parent_map={"CHILD": "PARENT"},
            targets=["CHILD"],
            msa_paths={"CHILD": msa},
            min_depth_after=1, max_removed_fraction=1.0,
        )
        options.update(kwargs)
        return lc.run_leakage_stage(**options)

    def test_purges_reports_and_passes(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa)
        assert result["status"] == "PASS"
        assert result["enabled"] is True
        assert result["leakage_check"] is True and result["purge"] is True
        assert result["purges"]["CHILD"]["n_removed"] == 1
        assert result["checks"]["B_parent_vs_target"]["CHILD"]["parent"] == "PARENT"
        assert result["checks"]["B_parent_vs_target"]["CHILD"]["accessions"][
            "n_shared_accessions"] == 0
        assert _headers(msa) == ["QUERY", "D_far"]

    def test_writes_both_reports_into_the_default_directory(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa)
        report_dir = tmp_path / "leakage"
        assert result["report_dir"] == str(report_dir)
        assert (report_dir / "leakage_report.json").exists()
        summary = (report_dir / "leakage_summary.txt").read_text(encoding="utf-8")
        assert "status: PASS" in summary
        assert "identity >= 99.0 OR hamming <= 10" in summary
        assert "purge CHILD" in summary

    def test_an_explicit_report_dir_is_honoured(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        elsewhere = tmp_path / "custom" / "leak"
        result = self._stage(tmp_path, target, parent, msa, report_dir=elsewhere)
        assert result["report_dir"] == str(elsewhere)
        assert (elsewhere / "leakage_report.json").exists()

    def test_the_blast_workdir_is_removed_at_the_end(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        self._stage(tmp_path, target, parent, msa)
        assert not (tmp_path / "leakage" / "blast_work").exists()

    def test_check_a_runs_instead_of_the_purge_when_purging_is_off(self, tmp_path,
                                                                   fake_blast):
        target, parent, msa = self._panels(tmp_path)
        before = common.md5_file(msa)
        result = self._stage(tmp_path, target, parent, msa, purge=False)
        assert common.md5_file(msa) == before
        assert result["purges"] == {}
        assert result["checks"]["A_msa_vs_target"]["CHILD"]["n_flagged"] >= 1
        assert (tmp_path / "leakage" / "leakage_A_CHILD_hits.tsv").exists()

    def test_detection_only_fails_the_gate_and_says_why(self, tmp_path, fake_blast):
        result = self._stage(tmp_path, *self._panels(tmp_path), purge=False)
        assert result["status"] == "FAIL"
        assert any("--no-purge-leakage" in f for f in result["failures"])

    def test_the_query_row_is_subtracted_from_the_check_a_failure_count(self, tmp_path,
                                                                        fake_blast):
        """Otherwise every target "fails" purely because its own query is in the MSA."""
        core = make_protein(53)
        target = tmp_path / "t.fasta"
        common.write_fasta(target, [("TGT1", core)])
        parent = tmp_path / "p.fasta"
        common.write_fasta(parent, [("PAR1", make_protein(54))])
        msa = tmp_path / "msa_CHILD.fasta"
        common.write_fasta(msa, [("QUERY", core), ("FAR", make_protein(55))])
        result = self._stage(tmp_path, target, parent, msa, purge=False)
        assert result["checks"]["A_msa_vs_target"]["CHILD"]["n_flagged"] == 1
        assert result["status"] == "PASS"

    def test_fail_on_leakage_raises_with_the_report_path(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        with pytest.raises(lc.LeakageError) as excinfo:
            self._stage(tmp_path, target, parent, msa, purge=False, fail_on_leakage=True)
        assert "leakage_report.json" in str(excinfo.value)

    def test_a_parent_that_shares_a_sequence_with_its_target_is_a_failure(self,
                                                                          tmp_path,
                                                                          fake_blast):
        core = make_protein(61)
        target, parent = tmp_path / "t.fasta", tmp_path / "p.fasta"
        common.write_fasta(target, [("SHARED", core)])
        common.write_fasta(parent, [("SHARED", core)])
        msa = tmp_path / "msa_CHILD.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core), ("D", make_protein(62))])
        result = self._stage(tmp_path, target, parent, msa)
        assert result["status"] == "FAIL"
        assert any("cannot fix" in failure for failure in result["failures"])

    def test_an_exact_deep_set_collision_fails_check_c(self, tmp_path, fake_blast):
        core = make_protein(71)
        target, parent = tmp_path / "t.fasta", tmp_path / "p.fasta"
        common.write_fasta(target, [("TGT1", core)])
        common.write_fasta(parent, [("PAR1", make_protein(72))])
        deep = tmp_path / "deep.fasta"
        common.write_fasta(deep, [("DEEP1", core)])      # byte-identical to the target
        msa = tmp_path / "msa_CHILD.fasta"
        common.write_fasta(msa, [("QUERY", SIGNAL_PEPTIDE + core), ("D", make_protein(73))])
        result = self._stage(tmp_path, target, parent, msa, deep_fasta=deep)
        assert "deep_msa" in result["checks"]["C_hash_duplicates"]["sets"]
        assert result["status"] == "FAIL"
        assert any("byte-identical" in failure for failure in result["failures"])

    def test_a_missing_deep_fasta_is_simply_absent_from_check_c(self, tmp_path,
                                                               fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa,
                             deep_fasta=tmp_path / "nope.fasta")
        assert "deep_msa" not in result["checks"]["C_hash_duplicates"]["sets"]

    def test_a_target_without_a_parent_skips_check_b_but_is_still_purged(self,
                                                                        tmp_path,
                                                                        fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa, parent_map={})
        assert result["checks"]["B_parent_vs_target"] == {}
        assert result["purges"]["CHILD"]["n_removed"] == 1

    def test_a_target_whose_parent_panel_is_absent_skips_check_b(self, tmp_path,
                                                                 fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa,
                             panel_by_label={"CHILD": target},
                             parent_map={"CHILD": "MISSING"})
        assert result["checks"]["B_parent_vs_target"] == {}

    def test_a_target_with_no_msa_is_skipped(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa, msa_paths={})
        assert result["purges"] == {}
        assert result["status"] == "PASS"

    def test_a_target_with_no_panel_is_skipped(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa,
                             targets=["CHILD", "GHOST"],
                             msa_paths={"CHILD": msa, "GHOST": msa})
        assert list(result["purges"]) == ["CHILD"]

    def test_disabled_is_recorded_not_silent(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa,
                             leakage_check=False, purge=False)
        assert result["status"] == "SKIPPED"
        assert result["enabled"] is False
        assert result["checks"] == {} and result["purges"] == {}
        assert not (tmp_path / "leakage" / "leakage_report.json").exists()

    def test_purge_only_still_records_the_thresholds_it_used(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa, leakage_check=False,
                             min_identity=98.0, max_hamming=None, min_coverage=90.0,
                             coverage_basis="shorter")
        assert result["checks"] == {}
        assert result["thresholds"]["min_identity"] == 98.0
        assert result["thresholds"]["max_hamming"] is None
        assert result["thresholds"]["coverage_basis"] == "shorter"
        assert result["thresholds"]["combination"].startswith("OR")

    def test_blast_options_are_echoed_into_the_manifest_block(self, tmp_path,
                                                              fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa,
                             blast_task="blastp", threads=3, evalue=1e-10)
        assert result["blast"] == {"blast_task": "blastp", "threads": 3, "evalue": 1e-10}

    def test_the_manifest_block_is_json_serialisable(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa)
        assert json.loads(json.dumps(result, default=str))["status"] == "PASS"

    def test_the_written_json_matches_the_returned_block(self, tmp_path, fake_blast):
        target, parent, msa = self._panels(tmp_path)
        result = self._stage(tmp_path, target, parent, msa)
        on_disk = json.loads((tmp_path / "leakage" / "leakage_report.json").read_text())
        assert on_disk["status"] == result["status"]
        assert on_disk["purges"]["CHILD"]["n_removed"] == 1


@pytest.mark.requires_blast
@pytest.mark.integration
class TestRunLeakageStageWithRealBlast:
    def test_end_to_end_purge_of_a_prepared_lineage_msa(self, tmp_path, distance_panel):
        msa = tmp_path / "msa_CHILD.fasta"
        shutil.copy2(distance_panel["msa"], msa)
        parent = tmp_path / "parent.fasta"
        common.write_fasta(parent, [("PAR1", substitute(distance_panel["core"], 45,
                                                        stride=11))])
        result = lc.run_leakage_stage(
            inputs_dir=tmp_path,
            panel_by_label={"CHILD": distance_panel["panel"], "PARENT": parent},
            parent_map={"CHILD": "PARENT"},
            targets=["CHILD"],
            msa_paths={"CHILD": msa},
            min_depth_after=1, max_removed_fraction=1.0, threads=2,
        )
        assert result["status"] == "PASS"
        assert result["purges"]["CHILD"]["n_removed"] == 4
        assert _headers(msa) == list(EXPECTED_KEPT_AT_DEFAULTS)
        assert (tmp_path / "msa_CHILD_prepurge.fasta").exists()


# =========================================================================== #
# Report writers
# =========================================================================== #

@pytest.mark.unit
class TestReportWriters:
    def test_write_report_emits_all_three_artefacts(self, tmp_path):
        rows = [{"check": "C_hash_duplicates", "target_lineage": "", "query_set": "a",
                 "subject_set": "b", "n_query": 1, "n_subject": 2,
                 "metric": "n_shared_exact_sequences", "value": 0, "note": ""}]
        paths = lc.write_report(tmp_path / "rep", rows, {"status": "PASS"}, ["line one"])
        assert set(paths) == {"tsv", "json", "txt"}
        header = paths["tsv"].read_text(encoding="utf-8").splitlines()[0]
        assert header.split("\t") == list(lc.REPORT_COLUMNS)
        assert paths["tsv"].read_text(encoding="utf-8").splitlines()[1].split("\t")[0] == \
            "C_hash_duplicates"
        assert json.loads(paths["json"].read_text())["status"] == "PASS"
        assert paths["txt"].read_text(encoding="utf-8") == "line one\n"

    def test_write_report_fills_missing_columns_with_blanks(self, tmp_path):
        paths = lc.write_report(tmp_path / "rep", [{"check": "X"}], {}, [])
        fields = paths["tsv"].read_text(encoding="utf-8").splitlines()[1].split("\t")
        assert fields[0] == "X"
        assert fields[1:] == [""] * (len(lc.REPORT_COLUMNS) - 1)

    def test_hits_tsv_is_written_with_a_header_even_when_empty(self, tmp_path):
        path = tmp_path / "nested" / "hits.tsv"
        lc._write_hits_tsv(path, [])
        assert path.read_text(encoding="utf-8").startswith("query\tquery_accession")
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1

    def test_hits_tsv_writes_one_row_per_flag(self, tmp_path):
        rows = [{"query": "q", "query_accession": "q", "subject": "s",
                 "subject_accession": "s", "subject_panel_copies": 2,
                 "exact_full_length": True, "identity": 99.9, "coverage": 97.2,
                 "hamming": 1, "aln_len": 550, "nident": 549, "qlen": 566,
                 "slen": 550, "bitscore": 1000.0, "evalue": 0.0}]
        path = tmp_path / "hits.tsv"
        lc._write_hits_tsv(path, rows)
        parsed = _read_tsv(path)
        assert len(parsed) == 1
        assert parsed[0]["identity"] == "99.9"
        assert parsed[0]["exact_full_length"] == "True"

    def test_report_rows_from_check_expand_the_distribution(self):
        check = lc.BlastCheck(
            name="A_msa_vs_target", target_lineage="K", query_label="deep_msa",
            subject_label="panel:K", n_query=10, n_subject=5, n_subject_unique=4,
            n_query_with_hit=9, n_exact_full_length=0, n_flagged=2,
            distribution=lc.identity_distribution([90.0, 99.5]),
            thresholds={"flag_identity": 99.0, "flag_coverage": 95.0},
        )
        rows = lc._report_rows_from_check(check)
        metrics = {row["metric"]: row["value"] for row in rows}
        assert metrics["n_subject_unique"] == 4
        assert metrics["n_query_with_hit"] == 9
        assert metrics["n_exact_full_length"] == 0
        assert metrics["n_flagged"] == 2
        assert metrics["best_hit_identity_max"] == 99.5
        assert metrics["best_hit_identity_p50"] == pytest.approx(94.75)
        assert metrics["best_hit_identity_hist_[99.5,99.9)"] == 1
        assert all(row["check"] == "A_msa_vs_target" for row in rows)

    def test_report_rows_survive_an_empty_distribution(self):
        check = lc.BlastCheck(
            name="A", target_lineage="K", query_label="q", subject_label="s",
            n_query=0, n_subject=0, n_subject_unique=0, n_query_with_hit=0,
            n_exact_full_length=0, n_flagged=0,
        )
        rows = lc._report_rows_from_check(check)
        assert {row["metric"] for row in rows} == {
            "n_subject_unique", "n_query_with_hit", "n_exact_full_length", "n_flagged"
        }


# =========================================================================== #
# CLI
# =========================================================================== #

@pytest.mark.unit
@pytest.mark.cli
class TestParser:
    def test_help_states_the_threshold_interaction(self):
        text = lc.build_parser().format_help()
        assert "OR" in text
        assert "not equivalent" in text.lower()
        assert "98.18% identity" in text or "98.2%" in text

    def test_defaults_match_the_module_constants(self):
        args = lc.build_parser().parse_args(["--report-dir", "/tmp/x"])
        assert lc.parse_threshold(args.leakage_min_identity, "x") == lc.DEFAULT_MIN_IDENTITY
        assert lc.parse_threshold(args.leakage_max_hamming, "x", int) == lc.DEFAULT_MAX_HAMMING
        assert args.leakage_min_coverage == lc.DEFAULT_MIN_COVERAGE
        assert args.leakage_coverage_basis == "both"
        assert args.leakage_max_removed_fraction == lc.DEFAULT_MAX_REMOVED_FRACTION
        assert args.leakage_min_depth_after == lc.DEFAULT_MIN_DEPTH_AFTER
        assert args.blast_task == lc.DEFAULT_BLAST_TASK
        assert args.purge_leakage is True and args.leakage_check is True
        assert args.fail_on_leakage is False

    def test_report_dir_is_required(self):
        with pytest.raises(SystemExit):
            lc.build_parser().parse_args([])

    def test_the_rules_can_be_switched_off_individually(self):
        args = lc.build_parser().parse_args([
            "--report-dir", "/tmp/x", "--leakage-min-identity", "none",
            "--leakage-max-hamming", "3",
        ])
        assert lc.parse_threshold(args.leakage_min_identity, "x") is None
        assert lc.parse_threshold(args.leakage_max_hamming, "x", int) == 3

    def test_boolean_optional_actions_have_negations(self):
        args = lc.build_parser().parse_args([
            "--report-dir", "/tmp/x", "--no-purge-leakage", "--no-leakage-check",
        ])
        assert args.purge_leakage is False and args.leakage_check is False

    def test_check_only_and_purge_only_are_expressible(self):
        parser = lc.build_parser()
        assert parser.parse_args(["--report-dir", "/tmp/x", "--check-only"]).check_only
        assert parser.parse_args(["--report-dir", "/tmp/x", "--purge-only"]).purge_only

    def test_an_unknown_coverage_basis_is_rejected(self):
        with pytest.raises(SystemExit):
            lc.build_parser().parse_args(["--report-dir", "/tmp/x",
                                          "--leakage-coverage-basis", "nonsense"])

    def test_purge_kwargs_translate_the_sentinels(self):
        args = lc.build_parser().parse_args([
            "--report-dir", "/tmp/x", "--leakage-max-hamming", "none",
            "--leakage-min-identity", "97.5", "--leakage-coverage-basis", "shorter",
        ])
        kwargs = lc._purge_kwargs(args)
        assert kwargs == {
            "min_identity": 97.5, "max_hamming": None, "min_coverage": 95.0,
            "coverage_basis": "shorter",
            "max_removed_fraction": lc.DEFAULT_MAX_REMOVED_FRACTION,
            "min_depth_after": lc.DEFAULT_MIN_DEPTH_AFTER,
        }

    def test_blast_kwargs_are_exactly_what_blast_records_accepts(self):
        args = lc.build_parser().parse_args(["--report-dir", "/tmp/x"])
        kwargs = lc._blast_kwargs(args)
        assert set(kwargs) == {"blast_task", "evalue", "max_target_seqs", "threads",
                               "makeblastdb_bin", "blastp_bin"}
        import inspect

        accepted = set(inspect.signature(lc.blast_records).parameters)
        assert set(kwargs) <= accepted

    def test_add_leakage_arguments_can_be_prefixed(self):
        parser = argparse.ArgumentParser()
        lc.add_leakage_arguments(parser, prefix="stage1-")
        args = parser.parse_args(["--stage1-leakage-min-identity", "98"])
        assert args.leakage_min_identity == "98"

    def test_prepare_inputs_exposes_the_same_flags(self):
        from prescott_iav import prepare_inputs

        args = prepare_inputs.build_parser().parse_args([
            "--inputs-dir", "/tmp/x", "--no-purge-leakage",
            "--leakage-min-identity", "none", "--leakage-max-hamming", "3",
        ])
        assert args.purge_leakage is False
        assert args.leakage_check is True
        assert lc.parse_threshold(args.leakage_min_identity, "x") is None
        assert lc.parse_threshold(args.leakage_max_hamming, "x", int) == 3


@pytest.mark.unit
@pytest.mark.cli
class TestMainExitCodes:
    """PASS/FAIL exit codes, offline (``lc.blast_records`` is stubbed out)."""

    def _deep(self, tmp_path: Path, records) -> Path:
        return _write(tmp_path / "deep.fasta", records)

    def test_a_clean_run_passes_and_exits_zero(self, tmp_path, five_lineage_guide,
                                               fake_blast, capsys):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(201))])
        code = lc.main([
            "--report-dir", str(tmp_path / "rep"),
            "--guide-path", str(five_lineage_guide["path"]),
            "--deep-fasta", str(deep),
            "--check-only",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "VERDICT: PASS" in out
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["status"] == "PASS" and payload["failures"] == []
        assert payload["deep_fasta_md5"] == common.md5_file(deep)

    def test_a_hash_collision_fails_but_still_exits_zero_without_the_gate(
        self, tmp_path, five_lineage_guide, fake_blast, capsys,
    ):
        panel_seq = next(iter(common.read_fasta(five_lineage_guide["panels"]["K"])))[1]
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", panel_seq)])
        code = lc.main([
            "--report-dir", str(tmp_path / "rep"),
            "--guide-path", str(five_lineage_guide["path"]),
            "--deep-fasta", str(deep), "--check-only",
        ])
        assert code == 0
        assert "VERDICT: FAIL" in capsys.readouterr().out
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["status"] == "FAIL"
        assert any("check C" in f for f in payload["failures"])

    def test_the_same_failure_exits_three_with_fail_on_leakage(self, tmp_path,
                                                              five_lineage_guide,
                                                              fake_blast):
        panel_seq = next(iter(common.read_fasta(five_lineage_guide["panels"]["K"])))[1]
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", panel_seq)])
        code = lc.main([
            "--report-dir", str(tmp_path / "rep"),
            "--guide-path", str(five_lineage_guide["path"]),
            "--deep-fasta", str(deep), "--check-only", "--fail-on-leakage",
        ])
        assert code == 3

    def test_an_unknown_target_is_refused_before_any_work(self, tmp_path,
                                                          five_lineage_guide, fake_blast):
        with pytest.raises(SystemExit, match="absent from the guide"):
            lc.main([
                "--report-dir", str(tmp_path / "rep"),
                "--guide-path", str(five_lineage_guide["path"]),
                "--deep-fasta", str(_write(tmp_path / "d.fasta", [("D", make_protein(9))])),
                "--target", "NOT_A_LINEAGE", "--check-only",
            ])

    def test_check_only_leaves_the_purge_section_empty(self, tmp_path,
                                                       five_lineage_guide, fake_blast):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(202))])
        lc.main(["--report-dir", str(tmp_path / "rep"),
                 "--guide-path", str(five_lineage_guide["path"]),
                 "--deep-fasta", str(deep), "--check-only"])
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["purges"] == {}
        assert "A_msa_vs_target" in payload["checks"]
        assert "C_hash_duplicates" in payload["checks"]

    def test_purge_only_skips_the_checks_and_purges_the_named_msa(
        self, tmp_path, five_lineage_guide, distance_panel, fake_blast,
    ):
        report_dir = tmp_path / "rep"
        code = lc.main([
            "--report-dir", str(report_dir),
            "--guide-path", str(five_lineage_guide["path"]),
            "--purge-only", "--target", "K",
            "--msa", str(distance_panel["msa"]),
            "--out-purged", str(tmp_path / "purged.fasta"),
            "--leakage-min-depth-after", "1",
            "--leakage-max-removed-fraction", "1.0",
        ])
        assert code == 0
        payload = json.loads((report_dir / "leakage_report.json").read_text())
        assert payload["checks"] == {}
        # The guide's K panel is unrelated to the distance fixture, so nothing goes.
        assert payload["purges"]["K"]["depth_before"] == 8
        assert (tmp_path / "purged.fasta").exists()

    def test_purge_only_on_the_deep_set_protects_nothing(self, tmp_path,
                                                         five_lineage_guide, fake_blast):
        """With no prepared MSA there is no query row, so ``protect_indices=()``."""
        panel_seq = next(iter(common.read_fasta(five_lineage_guide["panels"]["K"])))[1]
        deep = _write(tmp_path / "deep.fasta",
                      [("DEEP1", panel_seq), ("DEEP2", make_protein(203))])
        lc.main([
            "--report-dir", str(tmp_path / "rep"),
            "--guide-path", str(five_lineage_guide["path"]),
            "--deep-fasta", str(deep), "--purge-only", "--target", "K",
            "--leakage-min-depth-after", "1", "--leakage-max-removed-fraction", "1.0",
        ])
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        purge_report = payload["purges"]["K"]
        assert purge_report["query_exempted"] is None
        assert purge_report["n_removed"] == 1
        assert purge_report["depth_after"] == 1

    def test_the_summary_names_the_rule_and_the_parent_map(self, tmp_path,
                                                           five_lineage_guide, fake_blast):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(204))])
        lc.main(["--report-dir", str(tmp_path / "rep"),
                 "--guide-path", str(five_lineage_guide["path"]),
                 "--deep-fasta", str(deep), "--check-only"])
        summary = (tmp_path / "rep" / "leakage_summary.txt").read_text(encoding="utf-8")
        assert "purge rule" in summary
        assert "combined with OR" in summary
        assert "'K': 'J.2.4'" in summary, "the corrected K <- J.2.4 edge must be recorded"

    def test_a_parent_map_override_reaches_check_b(self, tmp_path, five_lineage_guide,
                                                   fake_blast):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(205))])
        lc.main(["--report-dir", str(tmp_path / "rep"),
                 "--guide-path", str(five_lineage_guide["path"]),
                 "--deep-fasta", str(deep), "--check-only",
                 "--parent-map", "K=J.2_int", "--target", "K"])
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["parent_map"]["K"] == "J.2_int"
        assert payload["checks"]["B_parent_vs_target"]["K"]["parent"] == "J.2_int"

    def test_the_report_tsv_is_machine_readable(self, tmp_path, five_lineage_guide,
                                               fake_blast):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(206))])
        lc.main(["--report-dir", str(tmp_path / "rep"),
                 "--guide-path", str(five_lineage_guide["path"]),
                 "--deep-fasta", str(deep), "--check-only"])
        rows = _read_tsv(tmp_path / "rep" / "leakage_report.tsv")
        assert rows
        assert {row["check"] for row in rows} >= {"C_hash_duplicates", "A_msa_vs_target"}
        assert all(set(row) == set(lc.REPORT_COLUMNS) for row in rows)

    def test_a_parentless_target_skips_check_b_without_failing(self, tmp_path,
                                                               five_lineage_guide,
                                                               fake_blast):
        """G.1 is input-only: it has no parent, so there is no frequency prior to audit."""
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(208))])
        code = lc.main(["--report-dir", str(tmp_path / "rep"),
                        "--guide-path", str(five_lineage_guide["path"]),
                        "--deep-fasta", str(deep), "--check-only", "--target", "G.1"])
        assert code == 0
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["targets"] == ["G.1"]
        assert "G.1" in payload["checks"]["A_msa_vs_target"]
        assert "B_parent_vs_target" not in payload["checks"]

    def test_a_parent_sharing_an_accession_with_its_target_is_a_failure(
        self, tmp_path, five_lineage_guide, fake_blast,
    ):
        """The circularity check B exists for: the frequency prior contains the answer."""
        target_panel = five_lineage_guide["panels"]["K"]
        parent_panel = five_lineage_guide["panels"]["J.2.4"]
        shared = list(common.read_fasta(target_panel))[0]
        common.write_fasta(parent_panel,
                           list(common.read_fasta(parent_panel)) + [shared])
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(209))])
        code = lc.main(["--report-dir", str(tmp_path / "rep"),
                        "--guide-path", str(five_lineage_guide["path"]),
                        "--deep-fasta", str(deep), "--check-only",
                        "--target", "K", "--fail-on-leakage"])
        assert code == 3
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        accessions = payload["checks"]["B_parent_vs_target"]["K"]["accessions"]
        assert accessions["n_shared_accessions"] == 1
        assert any("frequency prior contains the answer" in f for f in payload["failures"])

    def test_the_blast_workdir_is_cleaned_up(self, tmp_path, five_lineage_guide,
                                             fake_blast):
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", make_protein(207))])
        lc.main(["--report-dir", str(tmp_path / "rep"),
                 "--guide-path", str(five_lineage_guide["path"]),
                 "--deep-fasta", str(deep), "--check-only"])
        assert not (tmp_path / "rep" / "blast_work").exists()


@pytest.mark.cli
@pytest.mark.integration
class TestScriptEntryPoint:
    """The module as a script -- the ``__main__`` block and its exit codes."""

    def test_help_exits_zero_and_documents_the_or_rule(self, run_module_cli):
        proc = run_module_cli("leakage_check", ["--help"])
        assert proc.returncode == 0
        assert "combined with OR" in proc.stdout
        assert "--leakage-max-hamming" in proc.stdout

    def test_a_missing_report_dir_exits_two(self, run_module_cli):
        proc = run_module_cli("leakage_check", [])
        assert proc.returncode == 2
        assert "--report-dir" in proc.stderr

    @pytest.mark.requires_blast
    def test_a_planted_collision_exits_three_under_the_gate(self, tmp_path,
                                                            five_lineage_guide,
                                                            run_module_cli):
        panel_seq = next(iter(common.read_fasta(five_lineage_guide["panels"]["K"])))[1]
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", panel_seq)])
        proc = run_module_cli("leakage_check", [
            "--report-dir", tmp_path / "rep",
            "--guide-path", five_lineage_guide["path"],
            "--deep-fasta", deep,
            "--check-only", "--fail-on-leakage", "--target", "K",
            "--leakage-threads", "2",
        ], timeout=300)
        assert proc.returncode == 3, proc.stderr[-2000:]
        assert "VERDICT: FAIL" in proc.stdout
        payload = json.loads((tmp_path / "rep" / "leakage_report.json").read_text())
        assert payload["status"] == "FAIL"

    @pytest.mark.requires_blast
    def test_the_same_run_without_the_gate_exits_zero(self, tmp_path,
                                                      five_lineage_guide,
                                                      run_module_cli):
        panel_seq = next(iter(common.read_fasta(five_lineage_guide["panels"]["K"])))[1]
        deep = _write(tmp_path / "deep.fasta", [("DEEP1", panel_seq)])
        proc = run_module_cli("leakage_check", [
            "--report-dir", tmp_path / "rep",
            "--guide-path", five_lineage_guide["path"],
            "--deep-fasta", deep, "--check-only", "--target", "K",
            "--leakage-threads", "2",
        ], timeout=300)
        assert proc.returncode == 0
        assert "VERDICT: FAIL" in proc.stdout

    @pytest.mark.requires_blast
    def test_a_bad_threshold_is_a_clean_error_not_a_traceback(self, tmp_path,
                                                              five_lineage_guide,
                                                              run_module_cli):
        proc = run_module_cli("leakage_check", [
            "--report-dir", tmp_path / "rep",
            "--guide-path", five_lineage_guide["path"],
            "--check-only", "--leakage-max-hamming", "ten",
        ], timeout=300)
        assert proc.returncode == 1
        assert "ERROR:" in proc.stderr
        assert "--leakage-max-hamming" in proc.stderr
        assert "Traceback" not in proc.stderr


def _write(path: Path, records) -> Path:
    common.write_fasta(path, records)
    return path


# =========================================================================== #
# Driver / manifest integration
# =========================================================================== #

@pytest.mark.requires_torch
@pytest.mark.integration
class TestDriverIntegration:
    def test_the_driver_defaults_every_threshold_to_none(self, driver_module):
        """The driver must not re-declare the numbers; leakage_check.py owns them."""
        args = driver_module.build_parser().parse_args([
            "--output-dir", "/tmp/x", "--analysis-mode", "MONTHLY_GUIDE",
        ])
        assert args.leakage_check is True
        assert args.purge_leakage is True
        assert args.fail_on_leakage is False
        assert args.leakage_min_identity is None
        assert args.leakage_max_hamming is None
        assert args.leakage_min_coverage is None
        assert args.leakage_coverage_basis is None

    def test_the_driver_help_quotes_the_stage1_defaults(self, driver_module):
        assert driver_module.leakage_default("min_identity") == lc.DEFAULT_MIN_IDENTITY
        assert driver_module.leakage_default("max_hamming") == lc.DEFAULT_MAX_HAMMING
        assert driver_module.leakage_default("min_depth_after") == lc.DEFAULT_MIN_DEPTH_AFTER

    def test_an_absent_leakage_block_is_reported_as_unaudited(self, driver_module):
        args = argparse.Namespace(leakage_check=True, purge_leakage=True,
                                  fail_on_leakage=False)
        record = driver_module.leakage_manifest_record(args, {})
        assert record["leakage_stage_ran"] is False
        assert "UNAUDITED" in record["leakage_note"]

    def test_a_present_leakage_block_is_summarised(self, driver_module):
        args = argparse.Namespace(leakage_check=True, purge_leakage=True,
                                  fail_on_leakage=False)
        block = {
            "status": "PASS",
            "purge": True,
            "thresholds": {"min_identity": 99.0, "max_hamming": 10},
            "report_dir": "/tmp/leakage",
            "purges": {"K": {"depth_before": 6434, "n_removed": 3, "depth_after": 6431,
                             "removed_fraction": 0.000466,
                             "removed_identity_distribution": {"max": 99.8},
                             "removed_hamming_min": 1,
                             "query_exempted": {"identity": 100.0}}},
            "checks": {"B_parent_vs_target": {"K": {"parent": "J.2.4", "n_flagged": 0,
                                                    "accessions": {
                                                        "n_shared_accessions": 0,
                                                        "n_shared_exact_sequences": 0}}}},
        }
        record = driver_module.leakage_manifest_record(args, {"leakage": block})
        assert record["leakage_stage_ran"] is True
        assert record["leakage_status"] == "PASS"
        assert record["leakage_per_target"]["K"]["depth_after"] == 6431
        assert record["leakage_per_target"]["K"]["query_would_have_been_purged"] is True
        assert record["leakage_parent_vs_target"]["K"]["parent"] == "J.2.4"
