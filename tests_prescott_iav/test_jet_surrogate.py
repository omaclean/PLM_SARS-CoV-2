#!/usr/bin/env python3
"""Tests for ``scripts/prescott_iav/jet_surrogate.py``.

WHAT THIS MODULE HAS TO GET RIGHT, AND WHY EACH FAILURE IS SILENT
=================================================================
``jet_surrogate`` writes the one file that stands in for JET2.  Every way it can
be wrong is a way that produces a complete run with valid checksums and a
quietly meaningless answer, so the tests are organised around the failure, not
around the API:

``trace`` is multiplied into every prediction column
    ``pred.R:487`` does ``normPred[,i] <- normPred[,i] * trace[i]``.  A ``NaN``
    there poisons the entire matrix; a ``0`` turns that site into an
    identically-zero column, hence a uniform 1/20 softmax, hence pure noise.
    :class:`TestTraceIsSafeForPredR` and :class:`TestZeroTraceGuard` are the two
    halves of that: no NaN and no out-of-range value under *any* configuration,
    and a hard refusal when too many values are exactly zero.

the output is parsed by column NAME, by two different parsers
    ``escott.py:1124`` uses ``pd.read_table(sep=r"\\s+")``; ``computePred.R:46``
    uses R's ``read.table(head=TRUE)`` and at line 61 prefers a ``traceMax``
    column over ``trace`` if one exists.  :class:`TestEscottOutputFormat` pins
    the layout against the shipped JET2 reference
    ``/home3/oml4h/PRESCOTT/data/BLAT_jet.res`` and round-trips the written file
    through *both* real parsers.

the row count is a hard constraint
    ``computePred.R:140`` evaluates ``binAli[2:N[1],] %*% trace^2``, so
    ``nrow(jet)`` must equal the number of MSA columns exactly.  A residue that
    is in the query but *absent from the structure* -- the signal peptide, the
    HA0 cleavage loop, an unresolved tail -- must therefore still get a row.
    :class:`TestResiduesAbsentFromStructure` covers the documented fallback.

GROUND TRUTH
============
Nothing here is compared against the module's own output.  The Henikoff weights,
the KL divergence, the circular variances and the interface propensities are
either conftest literals derived by hand or recomputed in this file in plain
Python from the published formula (:func:`kl_bits_reference`,
:func:`henikoff_reference`).

Run with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \\
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_jet_surrogate.py -q

BUGS FOUND HERE AND SINCE FIXED IN ``jet_surrogate.py`` -- the two tests below
were ``xfail(strict=True)`` until the fix landed and are now live regression
tests:

* ``--structure-chain`` was neither recorded in the manifest nor compared by the
  cache short-circuit, so a rerun on chain B was served chain A's table --
  :meth:`TestCacheShortCircuit.test_structure_chain_invalidates_the_cache`.
* a cache hit returned 0 without producing ``--out-components`` / ``--out-dssp``
  even when they were requested and did not exist --
  :meth:`TestCacheShortCircuit.test_cache_hit_still_writes_requested_side_outputs`.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from prescott_iav import constants, jet_surrogate as js
from tests_prescott_iav import conftest as C


# --------------------------------------------------------------------------- #
# Independent re-implementations of the published formulae.
#
# These exist so a test compares the module against the LITERATURE, not against
# itself.  They are deliberately written in plain Python (dicts, math.log2) with
# no numpy vectorisation, so they share no code path with the implementation.
# --------------------------------------------------------------------------- #

AA20 = "ACDEFGHIKLMNPQRSTVWY"

ROBINSON_BACKGROUND_LITERAL = {
    "A": 0.07805, "C": 0.01925, "D": 0.05364, "E": 0.06295, "F": 0.03856,
    "G": 0.07377, "H": 0.02199, "I": 0.05142, "K": 0.05744, "L": 0.09019,
    "M": 0.02243, "N": 0.04487, "P": 0.05203, "Q": 0.04264, "R": 0.05129,
    "S": 0.07120, "T": 0.05841, "V": 0.06441, "W": 0.01330, "Y": 0.03216,
}
"""Robinson & Robinson (1991), retyped from the paper rather than imported."""

# Jones & Thornton (1996) interface propensities for the three residues this
# file asserts exactly.  W is the maximum of the table and D the minimum, so
# after the module's documented min-max rescaling they are exactly 1.0 and 0.0.
JT_W, JT_D, JT_A = 0.83, -0.38, -0.17


def kl_bits_reference(counts: Dict[str, float], w_sum: float, pseudocount: float = 0.5) -> float:
    """``sum_a p_a log2(p_a / b_a)`` with ``p = (n + c*b) / (w_sum + c)``.

    The docstring formula of :func:`jet_surrogate.column_conservation`, written
    out longhand.  ``counts`` maps a residue letter to its *weighted* count.
    """
    total = 0.0
    for aa in AA20:
        b = ROBINSON_BACKGROUND_LITERAL[aa]
        p = (counts.get(aa, 0.0) + pseudocount * b) / (w_sum + pseudocount)
        total += p * math.log2(p / b)
    return total


def henikoff_reference(rows: Sequence[str]) -> List[float]:
    """Henikoff & Henikoff (1994) position-based weights, normalised to mean 1.

    ``w_k = sum_i 1 / (r_i * n_{i,x_ik})`` over the columns where row k has a
    residue.  Gaps (anything not in :data:`AA20`) contribute nothing.
    """
    n_rows = len(rows)
    n_cols = len(rows[0])
    raw = [0.0] * n_rows
    for col in range(n_cols):
        column = [row[col].upper() for row in rows]
        present = [c for c in column if c in AA20]
        if not present:
            continue
        counts: Dict[str, int] = {}
        for c in present:
            counts[c] = counts.get(c, 0) + 1
        n_types = len(counts)
        for k, c in enumerate(column):
            if c in AA20:
                raw[k] += 1.0 / (n_types * counts[c])
    total = sum(raw)
    if total <= 0:
        return [1.0] * n_rows
    return [w * (n_rows / total) for w in raw]


def circular_variance_reference(points: Sequence[Tuple[float, float, float]],
                                origin: Tuple[float, float, float]) -> float:
    """``1 - || mean_j unit(r_j - origin) ||`` over the given neighbour points."""
    vectors = []
    for p in points:
        d = (p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
        n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        if n > 1e-6:
            vectors.append((d[0] / n, d[1] / n, d[2] / n))
    if not vectors:
        return 0.0
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(3)]
    return 1.0 - math.sqrt(sum(m * m for m in mean))


# --------------------------------------------------------------------------- #
# Local fixtures / helpers
# --------------------------------------------------------------------------- #

BLAT_JET_RES = Path("/home3/oml4h/PRESCOTT/data/BLAT_jet.res")

# The seven columns the shipped JET2 reference actually carries, retyped from
# the first line of BLAT_jet.res rather than imported from the module.
BLAT_REFERENCE_COLUMNS = ["AA", "pos", "chain", "pc", "tr", "freq", "trace"]
EXPECTED_JET_COLUMNS = BLAT_REFERENCE_COLUMNS + ["cv"]


def escott_parser(path: Path) -> pd.DataFrame:
    """``escott.py:1124`` verbatim: ``pd.read_table(<file>, sep=r"\\s+")``.

    Written out here rather than calling ``jet_surrogate.read_jet_res`` so the
    round-trip test does not depend on the module's own wrapper being right.
    """
    return pd.read_table(path, sep=r"\s+")


@pytest.fixture()
def msa_path(tiny_msa) -> Path:
    """The 12 x 72 mixed-conservation alignment (conserved / all-gap / hypervariable)."""
    return tiny_msa["path"]


@pytest.fixture()
def stub_dssp(monkeypatch):
    """Replace :func:`jet_surrogate.dssp_runs` with a fixed ``pos -> (ss, run)`` map.

    mkdssp needs a full backbone (N, CA, C, O); every synthetic structure in this
    suite is CA-only, so the real DSSP call is exercised separately (see
    :class:`TestDsspRuns`, marked ``requires_dssp``) and stubbed everywhere else.
    The stub also records which path it was handed, because
    ``build_jet_table`` must run DSSP on ``--pdb`` and never on ``--context-pdb``.
    """
    calls: List[Path] = []

    def _install(mapping: Optional[Dict[int, Tuple[str, int]]] = None,
                 letter: str = "H",
                 runlength: int = 1,
                 n: int = C.QUERY_LENGTH):
        table = dict(mapping) if mapping is not None else {
            pos: (letter, runlength) for pos in range(1, n + 1)
        }

        def _fake(pdb_path):
            calls.append(Path(pdb_path))
            return table

        monkeypatch.setattr(js, "dssp_runs", _fake)
        monkeypatch.setattr(js, "dssp_version", lambda: "stub 0.0")
        return calls

    return _install


def build(msa, pdb=None, context=None, query=None, **kwargs):
    """``build_jet_table`` with the fast, deterministic defaults this file uses."""
    kwargs.setdefault("trace_definition", "direct")
    return js.build_jet_table(msa, query, pdb, context, **kwargs)


# =========================================================================== #
# 1.  The exact output format escott and computePred.R parse
# =========================================================================== #

@pytest.mark.unit
class TestEscottOutputFormat:
    """The layout is a contract with two parsers that both index by column NAME."""

    def test_jet_columns_constant_is_the_blat_layout_plus_cv(self):
        assert js.JET_COLUMNS == EXPECTED_JET_COLUMNS
        assert js.JET_COLUMNS[:7] == BLAT_REFERENCE_COLUMNS
        assert "traceMax" not in js.JET_COLUMNS

    @pytest.mark.requires_blat_reference
    def test_shipped_jet2_reference_has_exactly_the_first_seven_columns(self):
        """The layout claim is pinned against the real JET2 file, not a literal."""
        ref = escott_parser(BLAT_JET_RES)
        assert list(ref.columns) == BLAT_REFERENCE_COLUMNS
        assert len(ref) == 286
        assert ref["pos"].tolist() == list(range(1, 287))
        assert set(ref["chain"].unique()) == {"A"}

    @pytest.mark.requires_blat_reference
    def test_reference_satisfies_trace_equals_tr_times_freq(self):
        """The identity our surrogate must reproduce, verified on real JET2 output."""
        ref = escott_parser(BLAT_JET_RES)
        product = ref["tr"].to_numpy() * ref["freq"].to_numpy()
        assert np.allclose(ref["trace"].to_numpy(), product, atol=1.1e-4)

    def test_written_header_is_the_column_order_verbatim(self, msa_path, tmp_path):
        components, _ = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        header = out.read_text().splitlines()[0]
        assert header.split("\t") == EXPECTED_JET_COLUMNS

    def test_round_trips_through_escotts_own_parser(self, msa_path, tmp_path):
        components, _ = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)

        parsed = escott_parser(out)
        assert list(parsed.columns) == EXPECTED_JET_COLUMNS
        assert len(parsed) == len(components)
        assert parsed["pos"].tolist() == list(range(1, len(components) + 1))
        assert parsed["AA"].tolist() == components["AA"].tolist()
        for col, source in (("pc", "pc"), ("tr", "tr"), ("freq", "freq"),
                            ("trace", "trace_emitted"), ("cv", "cv")):
            expected = np.round(
                np.nan_to_num(components[source].to_numpy(dtype=float)), 4
            )
            assert np.allclose(parsed[col].to_numpy(dtype=float), expected, atol=1e-9), col

    def test_space_padded_and_tab_separated_parse_identically(self, msa_path, tmp_path):
        """BLAT_jet.res is space-padded, ours is tab-separated; ``sep=r"\\s+"``
        must not be able to tell them apart."""
        components, _ = build(msa_path)
        tabbed = tmp_path / "tabbed.res"
        js.write_jet_res(components, tabbed)

        padded = tmp_path / "padded.res"
        lines = tabbed.read_text().splitlines()
        padded.write_text(
            "\n".join("  ".join(f"{f:<7s}" for f in line.split("\t")) for line in lines) + "\n"
        )
        left, right = escott_parser(tabbed), escott_parser(padded)
        assert list(left.columns) == list(right.columns)
        pd.testing.assert_frame_equal(left, right)

    @pytest.mark.requires_r
    @pytest.mark.integration
    def test_R_read_table_sees_trace_and_no_traceMax(self, msa_path, tmp_path,
                                                     subprocess_env):
        """The *other* consumer: ``computePred.R:46`` + its line-61 column choice.

        Emitting a ``traceMax`` column would silently change which weight R uses,
        so the R side is asked directly which column it picked and whether the
        vector it would square at line 140 has any NA in it.
        """
        components, _ = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)

        script = tmp_path / "read_jet.R"
        script.write_text(textwrap.dedent(
            """
            args <- commandArgs(TRUE)
            jet <- read.table(args[1], head = TRUE)
            cat("NCOL", ncol(jet), "\\n")
            cat("NAMES", paste(names(jet), collapse = ","), "\\n")
            cat("NROW", nrow(jet), "\\n")
            cat("HAS_traceMax", "traceMax" %in% names(jet), "\\n")
            tr <- if ("traceMax" %in% names(jet)) jet$traceMax else jet$trace
            cat("ANY_NA", any(is.na(tr)), "\\n")
            cat("CLASS", class(tr), "\\n")
            cat("MIN", min(tr), "MAX", max(tr), "\\n")
            """
        ))
        proc = subprocess.run(
            ["Rscript", str(script), str(out)],
            capture_output=True, text=True, env=subprocess_env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        fields = dict(
            (line.split(None, 1)[0], line.split(None, 1)[1].strip())
            for line in proc.stdout.splitlines() if line.strip()
        )
        assert fields["NCOL"] == "8"
        assert fields["NAMES"] == ",".join(EXPECTED_JET_COLUMNS)
        assert fields["NROW"] == str(len(components))
        assert fields["HAS_traceMax"] == "FALSE"
        assert fields["ANY_NA"] == "FALSE"
        assert fields["CLASS"] == "numeric"

    def test_every_numeric_field_is_written_with_exactly_four_decimals(
        self, msa_path, tmp_path
    ):
        components, _ = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        for line in out.read_text().splitlines()[1:]:
            fields = line.split("\t")
            assert len(fields) == 8
            for value in fields[3:]:
                assert len(value.split(".")[1]) == 4, line

    def test_pos_is_1_to_N_and_row_count_equals_msa_columns(self, msa_path, tmp_path):
        """``computePred.R:140`` needs exactly one row per MSA column."""
        components, meta = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        assert len(parsed) == meta["msa_n_columns"] == C.QUERY_LENGTH
        assert parsed["pos"].tolist() == list(range(1, C.QUERY_LENGTH + 1))

    def test_AA_column_is_the_three_letter_query_sequence(self, msa_path, tmp_path):
        components, _ = build(msa_path)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        expected = [C.THREE_LETTER[aa] for aa in C.QUERY_PROTEIN]
        assert parsed["AA"].tolist() == expected

    def test_chain_column_is_the_requested_chain(self, msa_path, tmp_path):
        components, _ = build(msa_path, structure_chain="Q")
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        assert set(escott_parser(out)["chain"].unique()) == {"Q"}

    def test_nan_pc_and_cv_are_written_as_zero_not_nan(self, msa_path, tmp_path):
        """R's ``read.table`` turns ``nan`` into ``NA`` and NA propagates silently."""
        components, _ = build(msa_path)          # no --pdb: pc and cv are all NaN
        assert components["pc"].isna().all()
        assert components["cv"].isna().all()

        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        text = out.read_text()
        assert "nan" not in text.lower()
        parsed = escott_parser(out)
        assert (parsed["pc"] == 0.0).all()
        assert (parsed["cv"] == 0.0).all()

    def test_unknown_query_residue_becomes_UNK_and_keeps_its_row(self, tmp_path):
        """An ``X`` in the query is a real occurrence and must not drop a row."""
        protein = "MKX" + C.QUERY_PROTEIN[3:]
        path = C.write_fasta(tmp_path / "msa_x.fasta",
                             [("query", protein), ("h1", protein)])
        components, _ = build(path)
        assert len(components) == len(protein)
        assert components.loc[2, "AA"] == "UNK"
        assert components.loc[2, "occupancy"] == 0.0


# =========================================================================== #
# 2.  trace must never be NaN and never leave [0, 1]  (pred.R:487)
# =========================================================================== #

@pytest.mark.unit
class TestTraceIsSafeForPredR:
    """``normPred[,i] <- normPred[,i] * trace[i]``: one NaN poisons everything."""

    CONFIGS = [
        dict(weight_mode="structural", trace_definition="direct"),
        dict(weight_mode="tjet", trace_definition="direct"),
        dict(weight_mode="structural", trace_definition="bootstrap", trace_bootstraps=6),
        dict(weight_mode="tjet", trace_definition="bootstrap", trace_bootstraps=6),
        dict(weight_mode="structural", trace_definition="direct", sequence_weighting="none"),
        dict(weight_mode="structural", trace_definition="direct", pc_mode="zero"),
        dict(weight_mode="structural", trace_definition="direct", pc_mode="constant"),
    ]

    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "-".join(
        f"{k}={v}" for k, v in sorted(c.items())))
    def test_no_nan_and_within_unit_interval_without_structure(
        self, msa_path, tmp_path, config
    ):
        components, _ = js.build_jet_table(msa_path, None, None, None, **config)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        for col in ("pc", "tr", "freq", "trace", "cv"):
            values = parsed[col].to_numpy(dtype=float)
            assert np.all(np.isfinite(values)), f"{col} has a non-finite value"
            assert values.min() >= -1e-9 and values.max() <= 1.0 + 1e-9, col

    @pytest.mark.requires_prody
    @pytest.mark.requires_freesasa
    @pytest.mark.requires_scipy
    @pytest.mark.parametrize("config", CONFIGS, ids=lambda c: "-".join(
        f"{k}={v}" for k, v in sorted(c.items())))
    def test_no_nan_and_within_unit_interval_with_partial_structure(
        self, msa_path, tmp_path, query_numbered_pdb_factory, stub_dssp, config
    ):
        stub_dssp(letter="C", runlength=99)      # the long-coil fallback branch too
        pdb = query_numbered_pdb_factory(covered=range(1, 51))
        components, _ = js.build_jet_table(msa_path, None, pdb, None, **config)
        out = tmp_path / "x_jet.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        for col in ("pc", "tr", "freq", "trace", "cv"):
            values = parsed[col].to_numpy(dtype=float)
            assert np.all(np.isfinite(values)), f"{col} has a non-finite value"
            assert values.min() >= -1e-9 and values.max() <= 1.0 + 1e-9, col

    def test_trace_equals_tr_times_freq(self, msa_path):
        """JET2's own identity.  Both factors are rounded to 4 dp in the frame, so
        the product can differ from the rounded product by at most 1e-4."""
        components, _ = build(msa_path, trace_definition="bootstrap",
                              trace_bootstraps=8, seed=11)
        product = components["tr"].to_numpy() * components["freq"].to_numpy()
        assert np.allclose(components["trace_jet"].to_numpy(), product, atol=1.1e-4)

    def test_meta_zero_trace_count_matches_the_emitted_column(self, msa_path):
        components, meta = build(msa_path, trace_definition="bootstrap",
                                 trace_bootstraps=8, trace_top_fraction=0.9, seed=3)
        emitted = components["trace_emitted"].to_numpy(dtype=float)
        assert meta["n_zero_trace_columns"] == int((emitted <= 1e-9).sum())
        assert meta["frac_zero_trace_columns"] == round(
            meta["n_zero_trace_columns"] / len(components), 4
        )


# =========================================================================== #
# 3.  The zero-trace guard added in the fix round
# =========================================================================== #

@pytest.mark.unit
class TestZeroTraceGuard:
    """A bad ``--trace-top-fraction`` must be impossible to select silently."""

    def test_module_defaults_come_from_constants(self):
        assert js.DEFAULT_TRACE_TOP_FRACTION is constants.DEFAULT_TRACE_TOP_FRACTION
        assert js.MAX_ZERO_TRACE_FRACTION is constants.MAX_ZERO_TRACE_FRACTION
        assert js.WARN_ZERO_TRACE_FRACTION is constants.WARN_ZERO_TRACE_FRACTION
        # Independent literals, so a regression to the design plan's 0.30 fails here.
        assert js.DEFAULT_TRACE_TOP_FRACTION == 0.90
        assert js.MAX_ZERO_TRACE_FRACTION == 0.10
        assert js.WARN_ZERO_TRACE_FRACTION == 0.05

    def test_parser_default_top_fraction_is_the_measured_090(self):
        args = js.build_parser().parse_args(["--msa", "m", "--out-jet", "o"])
        assert args.trace_top_fraction == 0.90
        assert args.max_zero_trace_fraction == 0.10

    def test_bad_top_fraction_raises_zero_trace_error(self, msa_path):
        with pytest.raises(js.ZeroTraceError) as excinfo:
            build(msa_path, trace_definition="bootstrap", trace_bootstraps=6,
                  trace_top_fraction=0.30, seed=5)
        message = str(excinfo.value)
        assert "pred.R:487" in message
        assert "trace_top_fraction=0.3" in message
        assert "--max-zero-trace-fraction" in message
        assert "1.0 disables the check" in message

    def test_zero_trace_error_is_a_runtime_error(self):
        assert issubclass(js.ZeroTraceError, RuntimeError)

    def test_refusal_writes_nothing(self, msa_path, tmp_path):
        out = tmp_path / "never.res"
        with pytest.raises(js.ZeroTraceError):
            js.main(["--msa", str(msa_path), "--out-jet", str(out),
                     "--trace-bootstraps", "6", "--trace-top-fraction", "0.30"])
        assert not out.exists()
        assert not out.with_suffix(".manifest.json").exists()

    def test_none_disables_the_ceiling_but_still_warns(self, msa_path):
        with pytest.warns(RuntimeWarning, match="pure noise"):
            components, meta = build(msa_path, trace_definition="bootstrap",
                                     trace_bootstraps=6, trace_top_fraction=0.30,
                                     max_zero_trace_fraction=None, seed=5)
        assert meta["frac_zero_trace_columns"] > js.MAX_ZERO_TRACE_FRACTION
        assert meta["max_zero_trace_fraction"] is None
        assert len(components) == C.QUERY_LENGTH

    def test_one_point_zero_is_the_escape_hatch(self, msa_path):
        _, meta = build(msa_path, trace_definition="bootstrap", trace_bootstraps=6,
                        trace_top_fraction=0.30, max_zero_trace_fraction=1.0, seed=5)
        assert meta["max_zero_trace_fraction"] == 1.0
        assert meta["frac_zero_trace_columns"] > 0.10

    def test_warn_band_writes_the_table_and_emits_both_warnings(
        self, msa_path, tmp_path, capsys
    ):
        """5% < frac <= 10%: warn on stderr AND raise a RuntimeWarning, then write."""
        with pytest.warns(RuntimeWarning) as record:
            components, meta = build(msa_path, trace_definition="bootstrap",
                                     trace_bootstraps=8, trace_top_fraction=0.90,
                                     seed=3)
        frac = meta["frac_zero_trace_columns"]
        assert js.WARN_ZERO_TRACE_FRACTION < frac <= js.MAX_ZERO_TRACE_FRACTION, frac
        assert any("pred.R:487" in str(w.message) for w in record)
        assert "[jet_surrogate] WARNING" in capsys.readouterr().err
        out = tmp_path / "warned.res"
        js.write_jet_res(components, out)
        assert out.exists()

    def test_no_warning_below_the_warn_fraction(self, msa_path, recwarn, capsys):
        _, meta = build(msa_path, trace_definition="direct")
        assert meta["frac_zero_trace_columns"] == 0.0
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
        assert "WARNING" not in capsys.readouterr().err

    def test_threshold_is_strictly_greater_than(self, msa_path):
        """A fraction exactly equal to the ceiling is accepted, not refused.

        The comparison uses the *unrounded* fraction; ``meta`` reports it to 4 dp,
        so the exact value has to be recomputed from the counts.
        """
        _, meta = build(msa_path, trace_definition="bootstrap", trace_bootstraps=6,
                        trace_top_fraction=0.30, max_zero_trace_fraction=1.0, seed=5)
        exact = meta["n_zero_trace_columns"] / C.QUERY_LENGTH
        assert exact > js.MAX_ZERO_TRACE_FRACTION
        _, again = build(msa_path, trace_definition="bootstrap", trace_bootstraps=6,
                         trace_top_fraction=0.30, max_zero_trace_fraction=exact, seed=5)
        assert again["n_zero_trace_columns"] == meta["n_zero_trace_columns"]
        with pytest.raises(js.ZeroTraceError):
            build(msa_path, trace_definition="bootstrap", trace_bootstraps=6,
                  trace_top_fraction=0.30, max_zero_trace_fraction=exact - 1e-9, seed=5)

    def test_cache_hit_reapplies_the_ceiling(self, msa_path, tmp_path, capsys):
        """A stale bad table must not be served either."""
        out = tmp_path / "stale.res"
        argv = ["--msa", str(msa_path), "--out-jet", str(out),
                "--trace-bootstraps", "6", "--trace-top-fraction", "0.30"]
        assert js.main(argv + ["--max-zero-trace-fraction", "1.0"]) == 0
        capsys.readouterr()
        cached = json.loads(out.with_suffix(".manifest.json").read_text())
        assert cached["frac_zero_trace_columns"] > 0.10

        with pytest.raises(js.ZeroTraceError) as excinfo:
            js.main(argv)
        assert "cached" in str(excinfo.value)
        assert "--force" in str(excinfo.value)

    def test_cache_hit_ceiling_can_be_disabled(self, msa_path, tmp_path, capsys):
        out = tmp_path / "stale2.res"
        argv = ["--msa", str(msa_path), "--out-jet", str(out),
                "--trace-bootstraps", "6", "--trace-top-fraction", "0.30",
                "--max-zero-trace-fraction", "1.0"]
        assert js.main(argv) == 0
        capsys.readouterr()
        assert js.main(argv) == 0
        assert "cache hit" in capsys.readouterr().out

    @pytest.mark.cli
    @pytest.mark.requires_prescott_python
    def test_cli_refusal_is_a_nonzero_exit(self, msa_path, tmp_path, run_module_cli):
        out = tmp_path / "cli_refused.res"
        proc = run_module_cli("jet_surrogate", [
            "--msa", str(msa_path), "--out-jet", str(out),
            "--trace-bootstraps", "6", "--trace-top-fraction", "0.30",
        ], timeout=300)
        assert proc.returncode != 0
        assert "ZeroTraceError" in proc.stderr
        assert not out.exists()


# =========================================================================== #
# 4.  A residue in the query but absent from the structure
# =========================================================================== #

@pytest.mark.unit
@pytest.mark.requires_prody
@pytest.mark.requires_freesasa
@pytest.mark.requires_scipy
class TestResiduesAbsentFromStructure:
    """Signal peptide, HA0 cleavage loop, TM tail: no coordinates, still a row."""

    @pytest.fixture()
    def partial(self, msa_path, query_numbered_pdb_factory, stub_dssp):
        covered = list(range(10, 51))            # 1-9 and 51-72 have no coordinates
        stub_dssp(mapping={p: ("H", 41) for p in covered})
        pdb = query_numbered_pdb_factory(covered=covered)
        components, meta = build(msa_path, pdb)
        return components, meta, covered

    def test_every_query_position_still_gets_a_row(self, partial):
        components, meta, _ = partial
        assert len(components) == C.QUERY_LENGTH
        assert components["pos"].tolist() == list(range(1, C.QUERY_LENGTH + 1))
        assert meta["msa_n_columns"] == C.QUERY_LENGTH

    def test_coverage_accounting_is_exact(self, partial):
        components, meta, covered = partial
        assert meta["structure"]["covered"] == len(covered)
        assert meta["n_positions_without_structure"] == C.QUERY_LENGTH - len(covered)
        assert components["has_structure"].sum() == len(covered)
        assert sorted(components.loc[components["has_structure"], "pos"]) == covered

    def test_uncovered_positions_have_nan_pc_and_cv_in_the_components_table(self, partial):
        components, _, covered = partial
        uncovered = components[~components["has_structure"]]
        assert len(uncovered) == C.QUERY_LENGTH - len(covered)
        assert uncovered["pc"].isna().all()
        assert uncovered["cv"].isna().all()
        assert uncovered["ss"].isna().all()

    def test_uncovered_positions_fall_back_to_pure_trace(self, partial):
        """The documented fallback: weight == trace exactly, not an imputed value."""
        components, _, _ = partial
        uncovered = components[~components["has_structure"]]
        assert np.array_equal(
            uncovered["weight"].to_numpy(dtype=float),
            np.round(uncovered["trace_jet"].to_numpy(dtype=float), 4),
        )
        assert np.array_equal(
            uncovered["trace_emitted"].to_numpy(dtype=float),
            uncovered["weight"].to_numpy(dtype=float),
        )

    def test_covered_positions_are_actually_modified_by_the_structure(self, partial):
        """Otherwise the previous test would pass on a structure that did nothing."""
        components, _, _ = partial
        covered_rows = components[components["has_structure"]]
        differs = ~np.isclose(
            covered_rows["weight"].to_numpy(dtype=float),
            covered_rows["trace_jet"].to_numpy(dtype=float),
        )
        assert differs.sum() > 0.5 * len(covered_rows)

    def test_uncovered_positions_are_written_as_zero_pc_and_cv(self, partial, tmp_path):
        components, _, covered = partial
        out = tmp_path / "partial.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        missing = [p for p in range(1, C.QUERY_LENGTH + 1) if p not in covered]
        rows = parsed[parsed["pos"].isin(missing)]
        assert len(rows) == len(missing)
        assert (rows["pc"] == 0.0).all()
        assert (rows["cv"] == 0.0).all()
        assert np.all(np.isfinite(rows["trace"].to_numpy(dtype=float)))

    def test_residues_beyond_the_query_length_are_ignored_not_appended(
        self, msa_path, query_numbered_pdb_factory, stub_dssp
    ):
        stub_dssp()
        longer = C.QUERY_PROTEIN + "AAAAA"
        pdb = query_numbered_pdb_factory(protein=longer)
        components, meta = build(msa_path, pdb)
        assert len(components) == C.QUERY_LENGTH
        assert meta["structure"]["covered"] == C.QUERY_LENGTH

    def test_structure_in_the_wrong_frame_is_refused(
        self, msa_path, cv_ladder_pdb, stub_dssp
    ):
        """All-glycine ladder numbered 1..8 vs a query starting MKTIIALS: 0% identity."""
        stub_dssp()
        with pytest.raises(ValueError, match="residue numbering"):
            build(msa_path, cv_ladder_pdb["path"])

    def test_structure_query_identity_is_recorded(
        self, msa_path, query_numbered_pdb_factory, stub_dssp
    ):
        stub_dssp()
        pdb = query_numbered_pdb_factory(covered=range(1, 41))
        _, meta = build(msa_path, pdb)
        assert meta["structure"]["structure_query_identity"] == 1.0


# =========================================================================== #
# 5.  All-gap columns
# =========================================================================== #

@pytest.mark.unit
class TestAllGapColumns:
    """A column nobody occupies must read as unconstrained, not as conserved."""

    def test_fully_absent_column_has_zero_occupancy_and_zero_kl(self):
        encoded = js.encode_msa([("q", "M-A"), ("h1", "M-C")])
        kl, occ = js.column_conservation(encoded, np.ones(2), 0.5)
        assert occ[1] == 0.0
        assert kl[1] == 0.0
        assert occ[0] == 1.0 and kl[0] > 0.0

    def test_all_gap_except_query_has_occupancy_exactly_one_over_n(
        self, msa_path, tiny_msa
    ):
        """Unweighted, so the expected occupancy is an exact rational: 1/12.

        The components frame rounds occupancy to 6 dp, hence the 1e-6 tolerance.
        """
        components, _ = build(msa_path, sequence_weighting="none")
        for pos in tiny_msa["all_gap_positions"]:
            assert components.loc[pos - 1, "occupancy"] == pytest.approx(
                1.0 / tiny_msa["n_rows"], abs=1e-6
            )
        for pos in tiny_msa["conserved_positions"]:
            assert components.loc[pos - 1, "occupancy"] == 1.0

    def test_all_gap_column_is_penalised_relative_to_a_conserved_one(
        self, msa_path, tiny_msa
    ):
        """The whole point of multiplying KL by occupancy: an empty column looks
        perfectly conserved to a naive score and must not score like one."""
        components, _ = build(msa_path, sequence_weighting="none")
        gap_raw = components.loc[
            [p - 1 for p in tiny_msa["all_gap_positions"]], "trace_raw"]
        conserved_raw = components.loc[
            [p - 1 for p in tiny_msa["conserved_positions"]], "trace_raw"]
        assert gap_raw.max() < conserved_raw.min()

    def test_the_raw_signal_is_kl_times_occupancy(self, msa_path, tiny_msa):
        """The documented composition, asserted directly.

        Comparing an all-gap column with a conserved one is not enough on its own:
        the pseudocount already drags a one-residue column down, so dropping the
        occupancy factor entirely still leaves the ordering intact.  This pins the
        factor itself.
        """
        components, _ = build(msa_path, sequence_weighting="none")
        product = (components["kl_bits"].to_numpy(dtype=float)
                   * components["occupancy"].to_numpy(dtype=float))
        expected = np.clip(product / float(np.quantile(product, 0.99)), 0.0, 1.0)
        assert components["trace_raw"].to_numpy(dtype=float) == pytest.approx(
            expected, abs=1e-5)
        # ... and the factor is not vacuously 1 everywhere.
        assert components["occupancy"].min() < 0.1
        assert components["occupancy"].max() == 1.0

    def test_occupancy_dominates_a_shallow_but_perfectly_conserved_column(self, tmp_path):
        """Two rows of W and ten gaps must score far below twelve rows of W."""
        rows = ["W" + "W", "W" + "W"] + ["W" + "-"] * 10
        path = C.write_fasta(tmp_path / "msa_depth.fasta",
                             [(f"r{i}", row) for i, row in enumerate(rows)])
        components, _ = build(path, sequence_weighting="none")
        deep, shallow = components.loc[0], components.loc[1]
        assert deep["occupancy"] == 1.0
        assert shallow["occupancy"] == pytest.approx(2 / 12, abs=1e-6)
        assert shallow["trace_raw"] < 0.5 * deep["trace_raw"]

    def test_all_gap_column_still_produces_a_row(self, msa_path, tiny_msa, tmp_path):
        components, _ = build(msa_path)
        out = tmp_path / "g.res"
        js.write_jet_res(components, out)
        parsed = escott_parser(out)
        for pos in tiny_msa["all_gap_positions"]:
            row = parsed[parsed["pos"] == pos]
            assert len(row) == 1
            assert row["AA"].iloc[0] == C.THREE_LETTER[C.QUERY_PROTEIN[pos - 1]]

    def test_a_column_absent_in_every_row_scores_exactly_zero_trace(self, tmp_path):
        """The extreme case: even the query has no residue there (an ``X``)."""
        protein = C.QUERY_PROTEIN[:5] + "X" + C.QUERY_PROTEIN[6:]
        path = C.write_fasta(tmp_path / "msa_allgap.fasta",
                             [("q", protein), ("h1", protein), ("h2", protein)])
        components, meta = build(path)
        assert components.loc[5, "occupancy"] == 0.0
        assert components.loc[5, "kl_bits"] == 0.0
        assert components.loc[5, "trace_raw"] == 0.0
        assert components.loc[5, "trace_emitted"] == 0.0
        assert meta["n_zero_trace_columns"] >= 1

    def test_henikoff_weights_fall_back_to_ones_when_nothing_is_present(self):
        encoded = js.encode_msa([("a", "---"), ("b", "---")])
        assert np.array_equal(js.henikoff_weights(encoded), np.ones(2))


# =========================================================================== #
# 6.  A fully conserved column (entropy exactly 0)
# =========================================================================== #

@pytest.mark.unit
class TestFullyConservedColumns:
    """Zero Shannon entropy, but NOT zero KL: that difference is deliberate."""

    def test_uniform_alignment_has_exactly_one_residue_type_per_column(self, uniform_msa):
        encoded = js.encode_msa([("r", row) for row in uniform_msa["rows"]])
        for col in range(encoded.shape[1]):
            assert len(set(encoded[:, col].tolist())) == 1

    def test_conserved_column_kl_matches_the_closed_form(self, uniform_msa):
        """Independently recomputed from the published KL formula, per residue."""
        rows = uniform_msa["rows"]
        encoded = js.encode_msa([("r", row) for row in rows])
        weights = np.ones(len(rows))
        kl, occ = js.column_conservation(encoded, weights, 0.5)
        n = float(len(rows))
        for col, aa in enumerate(rows[0]):
            expected = kl_bits_reference({aa: n}, n, 0.5)
            assert kl[col] == pytest.approx(expected, rel=1e-12, abs=1e-12), (col, aa)
            assert occ[col] == 1.0

    def test_conserved_rare_residue_scores_above_conserved_common_residue(self):
        """Shannon entropy is 0 for both; KL-to-background is not.  That is the
        documented reason the module uses KL rather than entropy."""
        n = 12
        rows_w = [("r%d" % i, "W") for i in range(n)]
        rows_l = [("r%d" % i, "L") for i in range(n)]
        kl_w, _ = js.column_conservation(js.encode_msa(rows_w), np.ones(n), 0.5)
        kl_l, _ = js.column_conservation(js.encode_msa(rows_l), np.ones(n), 0.5)
        assert kl_w[0] > kl_l[0]
        # W background 0.0133, L background 0.09019 -- the gap is ~2.7 bits.
        assert kl_w[0] - kl_l[0] == pytest.approx(
            kl_bits_reference({"W": 12.0}, 12.0) - kl_bits_reference({"L": 12.0}, 12.0),
            rel=1e-12,
        )

    def test_conserved_columns_outrank_hypervariable_ones(self, msa_path, tiny_msa):
        components, _ = build(msa_path, sequence_weighting="none")
        conserved = components.loc[
            [p - 1 for p in tiny_msa["conserved_positions"]], "kl_bits"]
        hypervariable = components.loc[
            [p - 1 for p in tiny_msa["hypervariable_positions"]], "kl_bits"]
        assert conserved.min() > hypervariable.max()

    def test_hypervariable_column_kl_matches_the_closed_form(self, msa_path, tiny_msa):
        components, _ = build(msa_path, sequence_weighting="none")
        rows = tiny_msa["rows"]
        for pos in tiny_msa["hypervariable_positions"]:
            counts: Dict[str, float] = {}
            for row in rows:
                counts[row[pos - 1]] = counts.get(row[pos - 1], 0.0) + 1.0
            assert len(counts) == len(rows)          # genuinely 12 distinct residues
            expected = kl_bits_reference(counts, float(len(rows)), 0.5)
            assert components.loc[pos - 1, "kl_bits"] == pytest.approx(
                round(expected, 6), abs=1e-6
            )


# =========================================================================== #
# 7.  Henikoff weights
# =========================================================================== #

@pytest.mark.unit
class TestHenikoffWeights:

    def test_handworked_alignment_matches_the_by_hand_values(self, handworked_msa):
        encoded = js.encode_msa([("r", row) for row in handworked_msa["rows"]])
        weights = js.henikoff_weights(encoded)
        assert weights == pytest.approx(handworked_msa["expected_weights"], abs=1e-12)
        assert weights[0] == pytest.approx(weights[3], abs=1e-12)
        assert weights[1] == pytest.approx(weights[2], abs=1e-12)

    def test_identical_rows_all_weigh_exactly_one(self, uniform_msa):
        encoded = js.encode_msa([("r", row) for row in uniform_msa["rows"]])
        assert js.henikoff_weights(encoded) == pytest.approx(
            uniform_msa["expected_weights"], abs=1e-12
        )

    def test_weights_always_have_mean_one(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        weights = js.henikoff_weights(encoded)
        assert float(weights.mean()) == pytest.approx(1.0, abs=1e-12)

    def test_matches_the_independent_reimplementation_on_the_tiny_msa(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        assert js.henikoff_weights(encoded) == pytest.approx(
            henikoff_reference(tiny_msa["rows"]), abs=1e-12
        )

    def test_a_redundant_cluster_is_down_weighted(self):
        """Ten identical rows plus one outlier: the outlier must weigh most."""
        rows = ["AAAA"] * 10 + ["CCCC"]
        weights = js.henikoff_weights(js.encode_msa([("r", r) for r in rows]))
        assert weights[10] > weights[0]
        assert weights == pytest.approx(henikoff_reference(rows), abs=1e-12)

    def test_gap_only_rows_get_zero_raw_weight(self):
        rows = ["AAAA", "AAAA", "----"]
        weights = js.henikoff_weights(js.encode_msa([("r", r) for r in rows]))
        assert weights[2] == 0.0
        assert weights[0] == weights[1] > 0.0


# =========================================================================== #
# 8.  encode_msa
# =========================================================================== #

@pytest.mark.unit
class TestEncodeMsa:

    def test_maps_residues_to_alphabetical_indices(self):
        encoded = js.encode_msa([("q", "ACDEFGHIKLMNPQRSTVWY")])
        assert encoded.shape == (1, 20)
        assert encoded[0].tolist() == list(range(20))

    @pytest.mark.parametrize("char", ["-", ".", "~", "*", "X", "x", "B", "Z", "J", "U"])
    def test_non_standard_characters_collapse_to_absent(self, char):
        encoded = js.encode_msa([("q", f"A{char}C")])
        assert encoded[0].tolist() == [0, -1, 1]

    def test_lowercase_is_uppercased(self):
        assert js.encode_msa([("q", "acdw")])[0].tolist() == \
               js.encode_msa([("q", "ACDW")])[0].tolist()

    def test_ragged_alignment_is_refused(self):
        with pytest.raises(ValueError, match="ragged"):
            js.encode_msa([("q", "ACDE"), ("h", "ACD")])

    def test_dtype_is_int8_and_absent_is_minus_one(self):
        encoded = js.encode_msa([("q", "A-")])
        assert encoded.dtype == np.int8
        assert encoded[0, 1] == -1


# =========================================================================== #
# 9.  column_conservation / scale_to_unit / compute_trace
# =========================================================================== #

@pytest.mark.unit
class TestColumnConservation:

    def test_occupancy_is_the_weighted_share_of_present_rows(self):
        encoded = js.encode_msa([("a", "A"), ("b", "A"), ("c", "-"), ("d", "-")])
        weights = np.array([3.0, 1.0, 5.0, 11.0])
        _, occ = js.column_conservation(encoded, weights, 0.5)
        assert occ[0] == pytest.approx(4.0 / 20.0, abs=1e-12)

    def test_weighted_counts_are_used_not_raw_counts(self):
        encoded = js.encode_msa([("a", "A"), ("b", "C")])
        weights = np.array([3.0, 1.0])
        kl, _ = js.column_conservation(encoded, weights, 0.5)
        assert kl[0] == pytest.approx(kl_bits_reference({"A": 3.0, "C": 1.0}, 4.0), abs=1e-12)

    def test_pseudocount_moves_the_answer_in_the_documented_direction(self):
        encoded = js.encode_msa([("a", "W")] * 4)
        weights = np.ones(4)
        small, _ = js.column_conservation(encoded, weights, 0.01)
        large, _ = js.column_conservation(encoded, weights, 5.0)
        assert small[0] > large[0]           # more smoothing -> closer to background
        assert small[0] == pytest.approx(kl_bits_reference({"W": 4.0}, 4.0, 0.01), abs=1e-12)
        assert large[0] == pytest.approx(kl_bits_reference({"W": 4.0}, 4.0, 5.0), abs=1e-12)

    def test_zero_total_weight_gives_zero_occupancy_everywhere(self):
        encoded = js.encode_msa([("a", "AC"), ("b", "AC")])
        kl, occ = js.column_conservation(encoded, np.zeros(2), 0.5)
        assert occ.tolist() == [0.0, 0.0]
        assert kl.tolist() == [0.0, 0.0]

    def test_background_table_matches_the_published_values(self):
        assert js.ROBINSON_BACKGROUND == ROBINSON_BACKGROUND_LITERAL
        assert sum(js.ROBINSON_BACKGROUND.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.unit
class TestScaleToUnit:

    def test_divides_by_the_requested_quantile_and_clips(self):
        raw = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        scaled = js.scale_to_unit(raw, 0.75)
        denom = float(np.quantile(raw, 0.75))
        assert scaled[4] == 1.0                              # the outlier is clipped
        assert scaled[1] == pytest.approx(1.0 / denom, abs=1e-12)
        assert scaled.min() >= 0.0 and scaled.max() <= 1.0

    def test_a_single_freak_column_cannot_compress_the_rest(self):
        """The documented reason for a quantile rather than the max."""
        raw = np.array([1.0] * 20 + [1000.0])       # the 0.90 quantile is 1.0
        by_quantile = js.scale_to_unit(raw, 0.90)
        by_max = raw / raw.max()
        assert by_quantile[0] == 1.0
        assert by_max[0] < 0.01

    def test_all_zero_input_returns_zeros(self):
        assert js.scale_to_unit(np.zeros(4), 0.99).tolist() == [0.0] * 4

    def test_all_negative_input_returns_zeros(self):
        assert js.scale_to_unit(np.array([-1.0, -5.0]), 0.99).tolist() == [0.0, 0.0]

    def test_empty_input_returns_empty(self):
        assert js.scale_to_unit(np.array([]), 0.99).size == 0

    def test_nonpositive_quantile_falls_back_to_the_max(self):
        raw = np.array([0.0, 0.0, 0.0, 4.0])
        scaled = js.scale_to_unit(raw, 0.5)      # median is 0 -> denom becomes max
        assert scaled[3] == 1.0
        assert scaled[0] == 0.0

    def test_non_finite_entries_do_not_set_the_denominator(self):
        raw = np.array([1.0, np.nan, 2.0])
        scaled = js.scale_to_unit(raw, 1.0)
        assert scaled[2] == 1.0
        assert math.isnan(scaled[1])


@pytest.mark.unit
class TestComputeTrace:

    def test_direct_definition_sets_freq_to_one_and_tr_to_the_scaled_score(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        parts = js.compute_trace(encoded, "direct", 0, 0.9, 0.5, 0.99, "henikoff", 1)
        assert parts["freq"].tolist() == [1.0] * encoded.shape[1]
        assert np.array_equal(parts["tr"], parts["trace_raw"])
        assert np.allclose(parts["trace_jet"], np.round(parts["trace_raw"], 4))

    def test_direct_definition_ignores_the_bootstrap_arguments(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        a = js.compute_trace(encoded, "direct", 50, 0.1, 0.5, 0.99, "henikoff", 1)
        b = js.compute_trace(encoded, "direct", 3, 0.9, 0.5, 0.99, "henikoff", 999)
        assert np.array_equal(a["trace_jet"], b["trace_jet"])

    def test_bootstrap_is_deterministic_for_a_fixed_seed(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        a = js.compute_trace(encoded, "bootstrap", 6, 0.9, 0.5, 0.99, "henikoff", 42)
        b = js.compute_trace(encoded, "bootstrap", 6, 0.9, 0.5, 0.99, "henikoff", 42)
        assert np.array_equal(a["trace_jet"], b["trace_jet"])
        assert np.array_equal(a["freq"], b["freq"])

    def test_a_different_seed_gives_a_different_answer(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        a = js.compute_trace(encoded, "bootstrap", 6, 0.5, 0.5, 0.99, "henikoff", 1)
        b = js.compute_trace(encoded, "bootstrap", 6, 0.5, 0.5, 0.99, "henikoff", 2)
        assert not np.array_equal(a["freq"], b["freq"])

    @pytest.mark.parametrize("bootstraps", [1, 4, 8])
    def test_freq_is_quantised_to_one_over_B(self, tiny_msa, bootstraps):
        """JET2's 0.02 steps at B=50 are the shape this reproduces."""
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        parts = js.compute_trace(encoded, "bootstrap", bootstraps, 0.6, 0.5, 0.99,
                                 "henikoff", 7)
        steps = parts["freq"] * bootstraps
        assert np.allclose(steps, np.round(steps), atol=1e-9)
        assert parts["freq"].min() >= 0.0 and parts["freq"].max() <= 1.0

    def test_top_fraction_one_selects_every_column(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        parts = js.compute_trace(encoded, "bootstrap", 4, 1.0, 0.5, 0.99, "henikoff", 7)
        assert parts["freq"].tolist() == [1.0] * encoded.shape[1]
        assert (parts["trace_jet"] > 0).sum() >= encoded.shape[1] - 3

    def test_lower_top_fraction_zeroes_more_columns(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        n_zero = {}
        for top in (0.2, 0.5, 0.9):
            parts = js.compute_trace(encoded, "bootstrap", 6, top, 0.5, 0.99,
                                     "henikoff", 7)
            n_zero[top] = int((parts["trace_jet"] <= 1e-9).sum())
        assert n_zero[0.2] > n_zero[0.5] > n_zero[0.9]

    def test_unselected_columns_have_zero_tr_and_zero_trace(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        parts = js.compute_trace(encoded, "bootstrap", 6, 0.3, 0.5, 0.99, "henikoff", 7)
        never = parts["freq"] == 0.0
        assert never.any()
        assert np.all(parts["tr"][never] == 0.0)
        assert np.all(parts["trace_jet"][never] == 0.0)

    def test_uniform_weighting_bypasses_henikoff(self, tiny_msa):
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        parts = js.compute_trace(encoded, "direct", 0, 0.9, 0.5, 0.99, "none", 1)
        assert parts["weights"].tolist() == [1.0] * encoded.shape[0]
        henikoff = js.compute_trace(encoded, "direct", 0, 0.9, 0.5, 0.99, "henikoff", 1)
        assert not np.array_equal(parts["weights"], henikoff["weights"])

    def test_single_row_alignment_does_not_crash_the_bootstrap(self):
        encoded = js.encode_msa([("q", "ACDEFGHIKL")])
        parts = js.compute_trace(encoded, "bootstrap", 3, 0.5, 0.5, 0.99, "henikoff", 1)
        assert parts["trace_jet"].shape == (10,)
        assert np.all(np.isfinite(parts["trace_jet"]))

    def test_row_zero_is_always_retained_in_every_resample(self, monkeypatch, tiny_msa):
        """GEMME treats row 0 as the reference; dropping it changes the alphabet."""
        encoded = js.encode_msa(list(zip(tiny_msa["headers"], tiny_msa["rows"])))
        seen: List[np.ndarray] = []
        original = js.column_conservation

        def _spy(sub, weights, pseudocount):
            seen.append(sub[0].copy())
            return original(sub, weights, pseudocount)

        monkeypatch.setattr(js, "column_conservation", _spy)
        js.compute_trace(encoded, "bootstrap", 5, 0.9, 0.5, 0.99, "none", 1)
        assert len(seen) == 6                    # 1 full pass + 5 resamples
        for row in seen[1:]:
            assert np.array_equal(row, encoded[0])


# =========================================================================== #
# 10.  Structure-derived columns
# =========================================================================== #

@pytest.mark.unit
@pytest.mark.requires_prody
@pytest.mark.requires_scipy
class TestCircularVariance:
    """JET2's own definition, Rc = 7.0 A (default.conf ``>CV max_dist 7.0``)."""

    def test_ladder_monomer_values_are_exactly_zero_and_one(self, cv_ladder_pdb):
        struct = js.load_structure(cv_ladder_pdb["path"])
        cv = js.circular_variance(struct, "A", range(1, cv_ladder_pdb["n_residues"] + 1))
        assert cv == cv_ladder_pdb["expected_cv"]

    def test_context_chain_buries_the_terminal_residue(self, cv_context_pdb):
        """Monomer vs trimer in miniature: residue 1 goes 0.0 -> 1.0, nothing else moves."""
        struct = js.load_structure(cv_context_pdb["path"])
        cv = js.circular_variance(struct, "A", range(1, C.CV_LADDER_N_RESIDUES + 1))
        assert cv == cv_context_pdb["expected_cv"]
        assert cv[cv_context_pdb["changed_position"]] == 1.0

    def test_matches_the_independent_formula(self, cv_ladder_pdb):
        struct = js.load_structure(cv_ladder_pdb["path"])
        cv = js.circular_variance(struct, "A", [4])
        neighbours = [(3 * C.CV_LADDER_SPACING, 0.0, 0.0),
                      (1 * C.CV_LADDER_SPACING, 0.0, 0.0)]
        expected = circular_variance_reference(neighbours,
                                               (2 * C.CV_LADDER_SPACING, 0.0, 0.0))
        assert cv[4] == pytest.approx(expected, abs=1e-12)

    def test_position_absent_from_the_structure_is_nan(self, cv_ladder_pdb):
        struct = js.load_structure(cv_ladder_pdb["path"])
        cv = js.circular_variance(struct, "A", [999])
        assert math.isnan(cv[999])

    def test_no_neighbour_within_the_radius_gives_zero(self, cv_ladder_pdb):
        struct = js.load_structure(cv_ladder_pdb["path"])
        cv = js.circular_variance(struct, "A", [4], radius=1.0)
        assert cv[4] == 0.0

    def test_the_residues_own_atoms_are_excluded(self, cv_ladder_pdb):
        """A residue that sees only itself has an empty neighbour set, hence 0.0."""
        struct = js.load_structure(cv_ladder_pdb["path"])
        assert js.circular_variance(struct, "A", [4], radius=0.5)[4] == 0.0

    def test_radius_controls_the_neighbourhood(self, cv_ladder_pdb):
        """Residue 2 sees {1, 3} at 7.0 A and {1, 3, 4} at 8.0 A.

        Two opposite unit vectors give cv exactly 1.0; adding a third at +x makes
        the mean (1/3, 0, 0), i.e. cv exactly 2/3.  Both are exact, so a radius
        that silently stopped being applied would change the value.
        """
        struct = js.load_structure(cv_ladder_pdb["path"])
        assert js.circular_variance(struct, "A", [2], radius=7.0)[2] == 1.0
        assert js.circular_variance(struct, "A", [2], radius=8.0)[2] == \
            pytest.approx(2.0 / 3.0, abs=1e-12)

    def test_ca_and_heavy_modes_agree_on_a_ca_only_structure(self, cv_ladder_pdb):
        struct = js.load_structure(cv_ladder_pdb["path"])
        heavy = js.circular_variance(struct, "A", range(1, 9), atom_mode="heavy")
        ca = js.circular_variance(struct, "A", range(1, 9), atom_mode="ca")
        assert heavy == ca

    def test_context_without_protein_heavy_atoms_is_refused(self):
        class _Struct:
            def select(self, _spec): return None

        with pytest.raises(ValueError, match="no protein heavy atoms"):
            js.circular_variance(_Struct(), "A", [1])

    def test_an_empty_neighbourhood_around_the_centroid_gives_zero(self, tmp_path):
        """Two atoms 100 A apart put the residue's centroid in empty space."""
        atoms = [("CA", "GLY", "A", 1, -50.0, 0.0, 0.0),
                 ("CB", "GLY", "A", 1, 50.0, 0.0, 0.0)]
        path = tmp_path / "split.pdb"
        path.write_text(C.build_pdb(atoms))
        cv = js.circular_variance(js.load_structure(path), "A", [1], radius=1.0)
        assert cv[1] == 0.0

    def test_a_neighbour_exactly_on_the_origin_is_ignored(self, tmp_path):
        """Zero-length vectors cannot be normalised; the module must not divide
        by them and must not return NaN into the ``cv`` column."""
        atoms = [("CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
                 ("CA", "GLY", "B", 1, 0.0, 0.0, 0.0)]
        path = tmp_path / "coincident.pdb"
        path.write_text(C.build_pdb(atoms))
        cv = js.circular_variance(js.load_structure(path), "A", [1], radius=7.0)
        assert cv[1] == 0.0
        assert not math.isnan(cv[1])


@pytest.mark.unit
@pytest.mark.requires_freesasa
class TestRelativeSasa:

    def test_isolated_residue_clips_to_exactly_one(self, sasa_monomer_pdb):
        rsa = js.relative_sasa(sasa_monomer_pdb["path"], "A")
        assert rsa == sasa_monomer_pdb["expected_rsa"]

    def test_shelled_residue_is_buried(self, sasa_context_pdb):
        rsa = js.relative_sasa(sasa_context_pdb["path"], "A")
        assert rsa[sasa_context_pdb["buried_resnum"]] <= \
               sasa_context_pdb["expected_buried_max_rsa"]
        assert rsa[sasa_context_pdb["isolated_resnum"]] == \
               sasa_context_pdb["expected_isolated_rsa"]

    def test_context_changes_the_answer(self, sasa_monomer_pdb, sasa_context_pdb):
        """The reason freesasa is run on the trimer and not on the monomer."""
        alone = js.relative_sasa(sasa_monomer_pdb["path"], "A")
        in_context = js.relative_sasa(sasa_context_pdb["path"], "A")
        assert alone[C.SASA_BURIED_RESNUM] == 1.0
        assert in_context[C.SASA_BURIED_RESNUM] < 0.05

    def test_unknown_chain_yields_an_empty_map(self, sasa_monomer_pdb):
        assert js.relative_sasa(sasa_monomer_pdb["path"], "Z") == {}

    def test_all_values_are_within_the_unit_interval(self, sasa_context_pdb):
        for chain in ("A", "B"):
            for value in js.relative_sasa(sasa_context_pdb["path"], chain).values():
                assert 0.0 <= value <= 1.0

    def test_insertion_codes_and_nonstandard_residues_are_skipped(self, tmp_path):
        """freesasa keys residues by string; ``2A`` is not an int and ``MSE`` has
        no Tien maximum, so both are dropped rather than crashing the run."""
        lines = [
            C.pdb_atom_line(1, "CA", "GLY", "A", 1, 0.0, 0.0, 0.0),
            C.pdb_atom_line(2, "CA", "GLY", "A", 2, 50.0, 0.0, 0.0),
            C.pdb_atom_line(3, "CA", "MSE", "A", 3, 100.0, 0.0, 0.0),
        ]
        with_icode = list(lines[1])
        with_icode[26] = "A"                     # PDB column 27 = insertion code
        lines[1] = "".join(with_icode)
        path = tmp_path / "icode.pdb"
        path.write_text("\n".join(lines) + "\nTER\nEND\n")
        assert js.relative_sasa(path, "A") == {1: 1.0}

    def test_tien_max_asa_table_is_the_published_one(self):
        assert js.TIEN_MAX_ASA["G"] == 104.0
        assert js.TIEN_MAX_ASA["W"] == 285.0
        assert set(js.TIEN_MAX_ASA) == set(AA20)


@pytest.mark.unit
class TestInterfacePropensity:

    def test_extremes_of_the_table_rescale_to_exactly_one_and_zero(self):
        pc = js.interface_propensity("WD", [1, 2], {1: 1.0, 2: 1.0},
                                     "interface_propensity")
        assert pc[1] == 1.0          # W is the table maximum, RSA 1.0
        assert pc[2] == 0.5          # D is the table minimum, RSA 1.0

    def test_value_is_the_documented_half_sum(self):
        pc = js.interface_propensity("A", [1], {1: 0.4}, "interface_propensity")
        scaled_a = (JT_A - JT_D) / (JT_W - JT_D)
        assert pc[1] == pytest.approx(0.5 * (scaled_a + 0.4), abs=1e-12)

    def test_missing_rsa_gives_nan(self):
        pc = js.interface_propensity("WD", [1, 2], {1: 1.0}, "interface_propensity")
        assert pc[1] == 1.0
        assert math.isnan(pc[2])

    def test_unknown_residue_gives_nan(self):
        pc = js.interface_propensity("X", [1], {1: 1.0}, "interface_propensity")
        assert math.isnan(pc[1])

    def test_zero_mode_zeroes_everything(self):
        pc = js.interface_propensity("WD", [1, 2], {1: 1.0, 2: 0.2}, "zero")
        assert pc == {1: 0.0, 2: 0.0}

    def test_constant_mode_uses_the_median_of_the_finite_values(self):
        pc = js.interface_propensity("WDA", [1, 2, 3], {1: 1.0, 2: 1.0, 3: 1.0},
                                     "constant")
        scaled_a = (JT_A - JT_D) / (JT_W - JT_D)
        expected = float(np.median([1.0, 0.5, 0.5 * (scaled_a + 1.0)]))
        assert len(set(pc.values())) == 1
        assert all(v == pytest.approx(expected, abs=1e-12) for v in pc.values())

    def test_constant_mode_with_no_finite_values_falls_back_to_zero(self):
        pc = js.interface_propensity("WD", [1, 2], {}, "constant")
        assert pc == {1: 0.0, 2: 0.0}

    def test_all_values_are_within_the_unit_interval(self):
        rsa = {i + 1: (i % 11) / 10.0 for i in range(20)}
        pc = js.interface_propensity(AA20, list(range(1, 21)), rsa,
                                     "interface_propensity")
        assert all(0.0 <= v <= 1.0 for v in pc.values())


@pytest.mark.unit
@pytest.mark.requires_prody
class TestResidueIndex:

    def test_indexes_by_residue_number_with_coordinates_and_ca(self, cv_ladder_pdb):
        index = js.residue_index(js.load_structure(cv_ladder_pdb["path"]), "A")
        assert sorted(index) == list(range(1, 9))
        assert index[1]["resname"] == "GLY"
        assert index[1]["ca"] is not None
        assert np.allclose(index[1]["centroid"], [0.0, 0.0, 0.0])
        assert np.allclose(index[3]["ca"], [2 * C.CV_LADDER_SPACING, 0.0, 0.0])

    def test_centroid_substitutes_for_a_missing_ca(self):
        """Driven through a stub selection, because prody's ``protein`` flag
        already drops a CA-less residue (see the test below) -- so this branch is
        only reachable if the selection ever changes."""
        class _Selection:
            def getResnums(self): return np.array([1, 1, 2])
            def getResnames(self): return np.array(["ALA", "ALA", "GLY"])
            def getCoords(self): return np.array([[0.0, 0.0, 0.0],
                                                  [2.0, 0.0, 0.0],
                                                  [9.0, 0.0, 0.0]])
            def getNames(self): return np.array(["CB", "CG", "CA"])

        class _Struct:
            def select(self, _spec): return _Selection()

        index = js.residue_index(_Struct(), None)
        assert np.allclose(index[1]["ca"], index[1]["centroid"])
        assert np.allclose(index[1]["centroid"], [1.0, 0.0, 0.0])
        assert np.allclose(index[2]["ca"], [9.0, 0.0, 0.0])

    def test_a_residue_without_a_ca_is_not_indexed_at_all(self, tmp_path):
        atoms = [("CB", "ALA", "A", 1, 0.0, 0.0, 0.0),
                 ("CA", "ALA", "A", 2, 9.0, 0.0, 0.0)]
        path = tmp_path / "no_ca.pdb"
        path.write_text(C.build_pdb(atoms))
        index = js.residue_index(js.load_structure(path), "A")
        assert sorted(index) == [2]

    def test_absent_chain_is_an_error_not_an_empty_dict(self, cv_ladder_pdb):
        with pytest.raises(ValueError, match="matched no atoms"):
            js.residue_index(js.load_structure(cv_ladder_pdb["path"]), "Z")

    def test_chain_none_takes_every_chain(self, cv_context_pdb):
        index = js.residue_index(js.load_structure(cv_context_pdb["path"]), None)
        assert len(index) == C.CV_LADDER_N_RESIDUES     # chain B reuses resnum 1

    def test_unparseable_structure_is_refused(self, tmp_path):
        path = tmp_path / "junk.pdb"
        path.write_text("this is not a pdb file\n")
        with pytest.raises(Exception):
            js.load_structure(path)

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif"])
    def test_mmcif_suffixes_go_through_parseMMCIF(self, monkeypatch, tmp_path, suffix):
        import prody
        seen = {}
        def _fake_mmcif(path):
            seen["cif"] = path
            return "S"

        monkeypatch.setattr(prody, "parseMMCIF", _fake_mmcif)
        monkeypatch.setattr(prody, "parsePDB",
                            lambda p: pytest.fail("parsePDB must not be used for mmCIF"))
        path = tmp_path / f"model{suffix}"
        path.write_text("")
        assert js.load_structure(path) == "S"
        assert seen["cif"] == str(path)

    def test_a_parser_returning_none_is_an_explicit_error(self, monkeypatch, tmp_path):
        import prody
        monkeypatch.setattr(prody, "parsePDB", lambda p: None)
        path = tmp_path / "nothing.pdb"
        path.write_text("")
        with pytest.raises(ValueError, match="could not parse structure"):
            js.load_structure(path)


# =========================================================================== #
# 11.  combine_weight -- escott's sstjetormaxtwocomponent formula
# =========================================================================== #

@pytest.mark.unit
class TestCombineWeight:

    @staticmethod
    def _call(trace, pc, cv, ss, runlength, has_structure, max_coil=5):
        return js.combine_weight(
            np.asarray(trace, dtype=float), np.asarray(pc, dtype=float),
            np.asarray(cv, dtype=float), ss, runlength,
            np.asarray(has_structure, dtype=bool), max_coil,
        )

    def test_no_structure_falls_back_to_pure_trace(self):
        out = self._call([0.4], [0.9], [0.9], [None], [None], [False])
        assert out.tolist() == [0.4]

    def test_long_coil_falls_back_to_pure_trace(self):
        out = self._call([0.4], [0.9], [0.9], ["C"], [6], [True], max_coil=5)
        assert out.tolist() == [0.4]

    def test_coil_exactly_at_the_limit_does_not_fall_back(self):
        """escott's test is ``>`` not ``>=``; an off-by-one here changes every
        boundary residue's weight."""
        out = self._call([0.4], [0.9], [0.9], ["C"], [5], [True], max_coil=5)
        assert out.tolist() == [0.65]

    def test_short_coil_uses_the_normal_formula(self):
        out = self._call([0.4], [0.8], [0.2], ["C"], [2], [True])
        assert out.tolist() == [0.6]

    def test_non_coil_long_run_uses_the_normal_formula(self):
        """Only ``ss == 'C'`` triggers the fallback -- mkdssp 4's 'P' must not."""
        out = self._call([0.4], [0.8], [0.2], ["P"], [99], [True])
        assert out.tolist() == [0.6]
        out_h = self._call([0.4], [0.8], [0.2], ["H"], [99], [True])
        assert out_h.tolist() == [0.6]

    def test_takes_the_maximum_of_the_two_components(self):
        assert self._call([0.2], [0.9], [0.1], ["H"], [1], [True]).tolist() == [0.55]
        assert self._call([0.2], [0.1], [0.9], ["H"], [1], [True]).tolist() == [0.55]

    def test_nan_pc_is_replaced_by_trace(self):
        out = self._call([0.4], [np.nan], [0.2], ["H"], [1], [True])
        assert out.tolist() == [0.4]             # max((0.4+0.4)/2, (0.4+0.2)/2)

    def test_nan_cv_is_replaced_by_trace(self):
        out = self._call([0.4], [0.2], [np.nan], ["H"], [1], [True])
        assert out.tolist() == [0.4]

    def test_both_nan_gives_exactly_trace(self):
        out = self._call([0.37], [np.nan], [np.nan], ["H"], [1], [True])
        assert out.tolist() == [0.37]

    def test_missing_runlength_is_treated_as_zero(self):
        out = self._call([0.4], [0.8], [0.2], ["C"], [None], [True])
        assert out.tolist() == [0.6]

    def test_output_is_rounded_to_four_decimals(self):
        out = self._call([1.0 / 3.0], [1.0 / 7.0], [np.nan], ["H"], [1], [True])
        assert out.tolist() == [round((1 / 3 + 1 / 3) / 2, 4)]
        assert all(v == round(v, 4) for v in out.tolist())

    def test_never_exceeds_one(self):
        out = self._call([1.0], [1.0], [1.0], ["H"], [1], [True])
        assert out.tolist() == [1.0]

    def test_vectorised_over_a_mixed_batch(self):
        out = self._call(
            [0.4, 0.4, 0.4, 0.4],
            [0.8, 0.8, 0.8, np.nan],
            [0.2, 0.2, 0.2, 0.2],
            [None, "C", "H", "H"],
            [None, 9, 9, 9],
            [False, True, True, True],
        )
        assert out.tolist() == [0.4, 0.4, 0.6, 0.4]


# =========================================================================== #
# 12.  build_jet_table input validation and metadata
# =========================================================================== #

@pytest.mark.unit
class TestBuildJetTableValidation:

    def test_empty_msa_is_refused(self, tmp_path):
        path = tmp_path / "empty.fasta"
        path.write_text("")
        with pytest.raises(ValueError, match="empty MSA"):
            build(path)

    def test_gapped_query_row_is_refused(self, gapped_query_msa):
        with pytest.raises(ValueError, match="gap-free first row"):
            build(gapped_query_msa)

    def test_ragged_msa_is_refused(self, tmp_path):
        path = C.write_fasta(tmp_path / "ragged.fasta",
                             [("q", "ACDEF"), ("h", "ACD")])
        with pytest.raises(ValueError, match="ragged"):
            build(path)

    def test_query_fasta_mismatch_is_refused(self, msa_path, tmp_path):
        wrong = C.write_fasta(tmp_path / "wrong.fasta", [("q", "A" * C.QUERY_LENGTH)])
        with pytest.raises(ValueError, match="does not match the query FASTA"):
            build(msa_path, query=wrong)

    def test_query_fasta_with_two_records_is_refused(self, msa_path, tmp_path):
        two = C.write_fasta(tmp_path / "two.fasta",
                            [("a", C.QUERY_PROTEIN), ("b", C.QUERY_PROTEIN)])
        with pytest.raises(ValueError, match="exactly one record"):
            build(msa_path, query=two)

    def test_matching_query_fasta_is_accepted(self, msa_path, query_protein_fasta):
        components, _ = build(msa_path, query=query_protein_fasta)
        assert len(components) == C.QUERY_LENGTH

    def test_gapped_query_fasta_is_stripped_before_comparison(self, msa_path, tmp_path):
        gapped = C.write_fasta(tmp_path / "gapped_query.fasta",
                               [("q", "--" + C.QUERY_PROTEIN + "--")])
        components, _ = build(msa_path, query=gapped)
        assert len(components) == C.QUERY_LENGTH

    def test_metadata_records_every_knob(self, msa_path):
        _, meta = build(msa_path, weight_mode="tjet", trace_pseudocount=0.25,
                        trace_scale_quantile=0.95, sequence_weighting="none",
                        pc_mode="zero", structure_chain="B", cv_radius=6.0,
                        cv_atom="ca", max_coil_length=9, seed=1234)
        assert meta["weight_mode"] == "tjet"
        assert meta["trace_pseudocount"] == 0.25
        assert meta["trace_scale_quantile"] == 0.95
        assert meta["sequence_weighting"] == "none"
        assert meta["pc_mode"] == "zero"
        assert meta["cv_radius"] == 6.0
        assert meta["cv_atom"] == "ca"
        assert meta["max_coil_length"] == 9
        assert meta["seed"] == 1234
        assert meta["msa_md5"] == C.md5_file(msa_path)
        assert meta["msa_n_sequences"] == C.MSA_N_ROWS
        assert meta["msa_n_columns"] == meta["query_length"] == C.QUERY_LENGTH
        assert meta["mkdssp_version"] is None      # no --pdb, so DSSP is never run

    def test_bootstraps_recorded_as_zero_under_direct(self, msa_path):
        _, direct = build(msa_path, trace_definition="direct", trace_bootstraps=50)
        assert direct["trace_bootstraps"] == 0
        _, boot = build(msa_path, trace_definition="bootstrap", trace_bootstraps=4)
        assert boot["trace_bootstraps"] == 4

    def test_structure_metadata_is_none_without_a_pdb(self, msa_path):
        _, meta = build(msa_path)
        assert meta["structure"] == {"pdb": None, "context_pdb": None, "covered": 0}
        assert meta["n_positions_without_structure"] == C.QUERY_LENGTH

    def test_weight_mode_selects_which_column_is_emitted(
        self, msa_path, query_numbered_pdb_factory, stub_dssp
    ):
        pytest.importorskip("prody")
        pytest.importorskip("freesasa")
        stub_dssp()
        pdb = query_numbered_pdb_factory()
        structural, _ = build(msa_path, pdb, weight_mode="structural")
        tjet, _ = build(msa_path, pdb, weight_mode="tjet")
        assert np.array_equal(structural["trace_emitted"], structural["weight"])
        assert np.array_equal(tjet["trace_emitted"], tjet["trace_jet"])
        assert not np.array_equal(structural["trace_emitted"], tjet["trace_emitted"])

    def test_components_frame_carries_every_audit_column(self, msa_path):
        components, _ = build(msa_path)
        expected = {"pos", "AA", "aa1", "chain", "has_structure", "occupancy",
                    "kl_bits", "trace_raw", "tr", "freq", "trace_jet", "pc", "cv",
                    "ss", "runlength", "weight", "trace_emitted"}
        assert set(components.columns) == expected
        assert components["aa1"].tolist() == list(C.QUERY_PROTEIN)

    @pytest.mark.requires_prody
    @pytest.mark.requires_freesasa
    @pytest.mark.requires_scipy
    def test_dssp_states_outside_the_query_range_are_discarded(
        self, msa_path, query_numbered_pdb_factory, stub_dssp
    ):
        """A structure numbered past the query end must not write off the end of
        the ``ss``/``runlength`` lists."""
        stub_dssp(mapping={**{p: ("H", 1) for p in range(1, C.QUERY_LENGTH + 1)},
                           0: ("C", 1), 999: ("C", 1)})
        pdb = query_numbered_pdb_factory()
        components, _ = build(msa_path, pdb)
        assert len(components) == C.QUERY_LENGTH
        assert set(components["ss"]) == {"H"}

    @pytest.mark.requires_prody
    @pytest.mark.requires_freesasa
    @pytest.mark.requires_scipy
    def test_context_pdb_is_used_for_cv_and_sasa_but_not_for_dssp(
        self, msa_path, query_numbered_pdb_factory, stub_dssp
    ):
        calls = stub_dssp()
        mono = query_numbered_pdb_factory(chains=("A",), name="mono.pdb")
        multi = query_numbered_pdb_factory(chains=("A", "B"), name="multi.pdb")
        _, meta = build(msa_path, mono, context=multi)
        assert meta["structure"]["pdb"] == str(mono)
        assert meta["structure"]["context_pdb"] == str(multi)
        assert calls == [mono]                   # DSSP ran on --pdb, not --context-pdb


# =========================================================================== #
# 13.  validate_jet_table
# =========================================================================== #

@pytest.mark.unit
class TestValidateJetTable:

    @staticmethod
    def _good(n=4):
        return pd.DataFrame({
            "AA": ["ALA"] * n,
            "pos": list(range(1, n + 1)),
            "chain": ["A"] * n,
            "pc": [0.1] * n, "tr": [0.2] * n, "freq": [0.3] * n,
            "trace": [0.4] * n, "cv": [0.5] * n,
        })[js.JET_COLUMNS]

    def test_a_correct_table_passes(self):
        js.validate_jet_table(self._good(), expected_rows=4)

    def test_wrong_column_order_is_refused(self):
        table = self._good()[["pos", "AA", "chain", "pc", "tr", "freq", "trace", "cv"]]
        with pytest.raises(AssertionError, match="column layout"):
            js.validate_jet_table(table, expected_rows=4)

    def test_missing_column_is_refused(self):
        with pytest.raises(AssertionError, match="column layout"):
            js.validate_jet_table(self._good().drop(columns=["cv"]), expected_rows=4)

    def test_extra_traceMax_column_is_refused(self):
        """computePred.R:61 would silently prefer it over ``trace``."""
        table = self._good()
        table["traceMax"] = 0.9
        with pytest.raises(AssertionError):
            js.validate_jet_table(table, expected_rows=4)

    def test_wrong_row_count_is_refused(self):
        with pytest.raises(AssertionError, match="computePred.R:140"):
            js.validate_jet_table(self._good(4), expected_rows=5)

    def test_non_ascending_pos_is_refused(self):
        table = self._good()
        table["pos"] = [1, 3, 2, 4]
        with pytest.raises(AssertionError, match="ascending"):
            js.validate_jet_table(table, expected_rows=4)

    def test_pos_not_starting_at_one_is_refused(self):
        table = self._good()
        table["pos"] = [0, 1, 2, 3]
        with pytest.raises(AssertionError, match="ascending"):
            js.validate_jet_table(table, expected_rows=4)

    @pytest.mark.parametrize("col", ["pc", "tr", "freq", "trace", "cv"])
    def test_nan_in_any_numeric_column_is_refused(self, col):
        table = self._good()
        table.loc[2, col] = np.nan
        with pytest.raises(AssertionError, match=f"non-finite values in `{col}`"):
            js.validate_jet_table(table, expected_rows=4)

    @pytest.mark.parametrize("col", ["pc", "tr", "freq", "trace", "cv"])
    def test_out_of_range_values_are_refused(self, col):
        table = self._good()
        table.loc[1, col] = 1.5
        with pytest.raises(AssertionError, match=r"out of \[0,1\]"):
            js.validate_jet_table(table, expected_rows=4)
        table = self._good()
        table.loc[1, col] = -0.5
        with pytest.raises(AssertionError, match=r"out of \[0,1\]"):
            js.validate_jet_table(table, expected_rows=4)

    def test_write_jet_res_refuses_to_write_a_bad_table(self, msa_path, tmp_path):
        components, _ = build(msa_path)
        components.loc[0, "trace_emitted"] = np.nan
        out = tmp_path / "bad.res"
        with pytest.raises(AssertionError, match="non-finite"):
            js.write_jet_res(components, out)
        assert not out.exists()


# =========================================================================== #
# 14.  read_jet_res / compare_to_reference
# =========================================================================== #

@pytest.mark.unit
class TestCompareToReference:

    @staticmethod
    def _frame(trace, tr=None, freq=None, pc=None):
        n = len(trace)
        return pd.DataFrame({
            "AA": ["ALA"] * n, "pos": list(range(1, n + 1)), "chain": ["A"] * n,
            "pc": pc if pc is not None else list(trace),
            "tr": tr if tr is not None else list(trace),
            "freq": freq if freq is not None else [1.0] * n,
            "trace": list(trace),
        })

    def test_identical_tables_correlate_perfectly(self, tmp_path):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        frame = self._frame(values)
        ref = tmp_path / "ref.res"
        frame.to_csv(ref, sep="\t", index=False)
        report = js.compare_to_reference(frame, ref)
        assert report["column"].tolist() == ["pc", "tr", "freq", "trace"]
        assert report.loc[report["column"] == "trace", "spearman_r"].iloc[0] == 1.0
        assert report.loc[report["column"] == "trace", "n"].iloc[0] == len(values)
        assert report.loc[report["column"] == "trace", "n_total"].iloc[0] == len(values)

    def test_reversed_ranking_gives_minus_one(self, tmp_path):
        ours = self._frame([0.1, 0.2, 0.3, 0.4, 0.5])
        theirs = self._frame([0.5, 0.4, 0.3, 0.2, 0.1])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        report = js.compare_to_reference(ours, ref)
        assert report.loc[report["column"] == "trace", "spearman_r"].iloc[0] == -1.0

    def test_row_count_mismatch_is_refused(self, tmp_path):
        ref = tmp_path / "ref.res"
        self._frame([0.1, 0.2, 0.3]).to_csv(ref, sep="\t", index=False)
        with pytest.raises(ValueError, match="not comparable"):
            js.compare_to_reference(self._frame([0.1, 0.2]), ref)

    def test_zero_fractions_are_reported_for_both_sides(self, tmp_path):
        ours = self._frame([0.0, 0.0, 0.3, 0.4, 0.5, 0.6])
        theirs = self._frame([0.0, 0.2, 0.3, 0.4, 0.5, 0.6])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        row = js.compare_to_reference(ours, ref).set_index("column").loc["trace"]
        assert row["ours_n_zero"] == 2
        assert row["ours_frac_zero"] == pytest.approx(2 / 6, abs=1e-4)
        assert row["ref_n_zero"] == 1
        assert row["ref_frac_zero"] == pytest.approx(1 / 6, abs=1e-4)

    def test_nan_rows_are_excluded_and_counted(self, tmp_path):
        ours = self._frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ours.loc[0, "pc"] = np.nan
        ours.loc[1, "pc"] = np.nan
        theirs = self._frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        row = js.compare_to_reference(ours, ref).set_index("column").loc["pc"]
        assert row["n"] == 4
        assert row["n_total"] == 6
        assert row["spearman_r"] == 1.0

    def test_too_few_usable_rows_gives_nan_not_a_crash(self, tmp_path):
        ours = self._frame([0.1, 0.2, 0.3])
        ours.loc[0, "pc"] = np.nan
        theirs = self._frame([0.1, 0.2, 0.3])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        row = js.compare_to_reference(ours, ref).set_index("column").loc["pc"]
        assert math.isnan(row["spearman_r"])
        assert row["n"] == 2

    def test_extra_pairs_are_appended_with_a_combined_name(self, tmp_path):
        ours = self._frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ours["trace_emitted"] = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        theirs = self._frame([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        report = js.compare_to_reference(ours, ref,
                                         extra_pairs=(("trace_emitted", "trace"),))
        assert report["column"].tolist()[-1] == "trace_emitted_vs_trace"
        assert report["spearman_r"].iloc[-1] == -1.0

    def test_columns_absent_on_either_side_are_skipped(self, tmp_path):
        ours = self._frame([0.1, 0.2, 0.3, 0.4]).drop(columns=["freq"])
        theirs = self._frame([0.1, 0.2, 0.3, 0.4])
        ref = tmp_path / "ref.res"
        theirs.to_csv(ref, sep="\t", index=False)
        report = js.compare_to_reference(ours, ref,
                                         extra_pairs=(("nope", "trace"),))
        assert report["column"].tolist() == ["pc", "tr", "trace"]

    def test_range_statistics_are_the_real_min_max_mean(self, tmp_path):
        values = [0.0, 0.25, 0.5, 1.0]
        frame = self._frame(values)
        ref = tmp_path / "ref.res"
        frame.to_csv(ref, sep="\t", index=False)
        row = js.compare_to_reference(frame, ref).set_index("column").loc["trace"]
        assert row["ours_min"] == 0.0
        assert row["ours_max"] == 1.0
        assert row["ours_mean"] == pytest.approx(0.4375, abs=1e-9)
        assert row["ref_mean"] == pytest.approx(0.4375, abs=1e-9)

    def test_read_jet_res_is_whitespace_agnostic(self, tmp_path):
        text = ("AA  pos chain pc     tr     freq trace  \n"
                "MET 1   A     0.6606 0.0    0.0  0.0    \n"
                "SER 2   A     0.3574 0.1    0.2  0.02   \n")
        path = tmp_path / "padded.res"
        path.write_text(text)
        frame = js.read_jet_res(path)
        assert list(frame.columns) == BLAT_REFERENCE_COLUMNS
        assert frame["pos"].tolist() == [1, 2]
        assert frame["trace"].tolist() == [0.0, 0.02]


# =========================================================================== #
# 15.  mkdssp discovery and DSSP parsing
# =========================================================================== #

@pytest.mark.unit
class TestMkdsspBin:

    def test_prefers_whatever_is_on_PATH(self, monkeypatch):
        monkeypatch.setattr(js.shutil, "which", lambda name: "/somewhere/mkdssp")
        assert js.mkdssp_bin() == "/somewhere/mkdssp"

    def test_falls_back_to_the_interpreters_own_bin_directory(self, monkeypatch, tmp_path):
        """The module is run as ``<env>/bin/python jet_surrogate.py`` without
        ``conda activate``, so the env's bin is often NOT on PATH."""
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        (fake_bin / "mkdssp").write_text("#!/bin/sh\n")
        (fake_bin / "python").write_text("")
        monkeypatch.setattr(js.shutil, "which", lambda name: None)
        monkeypatch.setattr(js.sys, "executable", str(fake_bin / "python"))
        assert js.mkdssp_bin() == str(fake_bin / "mkdssp")

    def test_raises_a_named_error_when_nowhere_to_be_found(self, monkeypatch, tmp_path):
        empty = tmp_path / "emptybin"
        empty.mkdir()
        (empty / "python").write_text("")
        monkeypatch.setattr(js.shutil, "which", lambda name: None)
        monkeypatch.setattr(js.sys, "executable", str(empty / "python"))
        with pytest.raises(FileNotFoundError, match="mkdssp not found"):
            js.mkdssp_bin()


@pytest.mark.unit
class TestDsspVersion:

    def test_reports_the_first_line_of_the_version_output(self, monkeypatch):
        class _Proc:
            stdout = "mkdssp 4.6.1\nextra\n"
            stderr = ""
        monkeypatch.setattr(js, "mkdssp_bin", lambda: "/bin/mkdssp")
        monkeypatch.setattr(js.subprocess, "run", lambda *a, **k: _Proc())
        assert js.dssp_version() == "mkdssp 4.6.1"

    def test_falls_back_to_stderr(self, monkeypatch):
        class _Proc:
            stdout = ""
            stderr = "mkdssp version 4.6.1\n"
        monkeypatch.setattr(js, "mkdssp_bin", lambda: "/bin/mkdssp")
        monkeypatch.setattr(js.subprocess, "run", lambda *a, **k: _Proc())
        assert js.dssp_version() == "mkdssp version 4.6.1"

    def test_a_failure_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(js, "mkdssp_bin",
                            lambda: (_ for _ in ()).throw(FileNotFoundError("nope")))
        assert js.dssp_version().startswith("unknown (")


@pytest.mark.requires_dssp
@pytest.mark.requires_blat_reference
@pytest.mark.integration
class TestDsspRuns:
    """The one path that needs a real backbone; every synthetic PDB here is CA-only."""

    def test_run_lengths_are_the_groupby_of_the_state_string(self):
        runs = js.dssp_runs(Path("/home3/oml4h/PRESCOTT/data/blat-af2.pdb"))
        assert runs, "DSSP returned nothing for the shipped BLAT structure"
        positions = sorted(runs)
        letters = [runs[p][0] for p in positions]
        # Recompute the run lengths independently, without itertools.groupby.
        expected: List[int] = []
        i = 0
        while i < len(letters):
            j = i
            while j < len(letters) and letters[j] == letters[i]:
                j += 1
            expected.extend([j - i] * (j - i))
            i = j
        assert [runs[p][1] for p in positions] == expected

    def test_ca_only_structure_is_a_loud_failure_not_a_silent_empty_table(self, tmp_path):
        path = tmp_path / "ca_only.pdb"
        path.write_text(C.build_pdb(C.cv_ladder_atoms("A")))
        with pytest.raises(ValueError, match="DSSP returned"):
            js.dssp_runs(path)


# =========================================================================== #
# 16.  CLI: argument handling, cache, side outputs
# =========================================================================== #

@pytest.mark.unit
class TestCliArguments:

    def test_msa_and_out_jet_are_required_unless_validate_only(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            js.main([])
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "--msa" in err and "--out-jet" in err
        assert "--validate-only" in err

    def test_missing_out_jet_alone_is_reported(self, msa_path, capsys):
        with pytest.raises(SystemExit):
            js.main(["--msa", str(msa_path)])
        err = capsys.readouterr().err
        assert "--out-jet" in err
        assert "--msa," not in err

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            js.main(["--help"])
        assert excinfo.value.code == 0
        assert "--trace-top-fraction" in capsys.readouterr().out

    @pytest.mark.parametrize("flag,value", [
        ("--weight-mode", "nonsense"),
        ("--trace-definition", "nonsense"),
        ("--sequence-weighting", "nonsense"),
        ("--pc-mode", "nonsense"),
        ("--cv-atom", "nonsense"),
    ])
    def test_choice_flags_reject_unknown_values(self, flag, value):
        with pytest.raises(SystemExit):
            js.build_parser().parse_args(["--msa", "m", "--out-jet", "o", flag, value])

    def test_parser_defaults(self):
        args = js.build_parser().parse_args([])
        assert args.msa is None and args.out_jet is None
        assert args.weight_mode == "structural"
        assert args.trace_definition == "bootstrap"
        assert args.trace_bootstraps == 50
        assert args.trace_pseudocount == 0.5
        assert args.trace_scale_quantile == 0.99
        assert args.sequence_weighting == "henikoff"
        assert args.pc_mode == "interface_propensity"
        assert args.structure_chain == "A"
        assert args.cv_radius == 7.0
        assert args.cv_atom == "heavy"
        assert args.max_coil_length == 5
        assert args.seed == 20260805
        assert args.min_validation_spearman == 0.4
        assert args.force is False
        assert args.validate_only is False
        assert args.validation_msa == Path(constants.BLAT_REFERENCE_MSA)
        assert args.validation_pdb == Path(constants.BLAT_REFERENCE_PDB)


@pytest.mark.unit
class TestCliOutputs:

    def test_writes_the_res_the_manifest_and_the_side_tables(
        self, msa_path, tmp_path, capsys
    ):
        out = tmp_path / "lin_jet.res"
        components = tmp_path / "sub" / "components.tsv"
        dssp = tmp_path / "sub" / "dssp.csv"
        manifest = tmp_path / "sub" / "manifest.json"
        rc = js.main(["--msa", str(msa_path), "--out-jet", str(out),
                      "--trace-definition", "direct",
                      "--out-components", str(components),
                      "--out-dssp", str(dssp),
                      "--out-manifest", str(manifest)])
        assert rc == 0
        assert out.exists() and components.exists() and dssp.exists() and manifest.exists()

        meta = json.loads(manifest.read_text())
        assert meta["jet_res_path"] == str(out)
        assert meta["jet_res_md5"] == C.md5_file(out)
        assert meta["escott_parser_roundtrip_ok"] is True
        assert meta["components_path"] == str(components)
        assert meta["dssp_path"] == str(dssp)
        assert meta["msa_md5"] == C.md5_file(msa_path)

        frame = pd.read_csv(components, sep="\t")
        assert len(frame) == C.QUERY_LENGTH
        dssp_frame = pd.read_csv(dssp)
        assert list(dssp_frame.columns) == ["pos", "ss", "runlength"]
        assert len(dssp_frame) == C.QUERY_LENGTH

        stdout = capsys.readouterr().out
        assert f"wrote {out}" in stdout
        assert "72 rows" in stdout

    def test_default_manifest_path_is_derived_from_out_jet(self, msa_path, tmp_path):
        out = tmp_path / "lin_jet.res"
        js.main(["--msa", str(msa_path), "--out-jet", str(out),
                 "--trace-definition", "direct"])
        assert out.with_suffix(".manifest.json").exists()

    def test_output_directories_are_created(self, msa_path, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "lin_jet.res"
        assert js.main(["--msa", str(msa_path), "--out-jet", str(out),
                        "--trace-definition", "direct"]) == 0
        assert out.exists()

    def test_validate_against_a_matching_reference_records_the_spearman(
        self, msa_path, tmp_path, capsys
    ):
        components, _ = build(msa_path)
        reference = tmp_path / "ref.res"
        js.write_jet_res(components, reference)

        out = tmp_path / "again_jet.res"
        validation = tmp_path / "val.tsv"
        rc = js.main(["--msa", str(msa_path), "--out-jet", str(out),
                      "--trace-definition", "direct",
                      "--validate-against", str(reference),
                      "--out-validation", str(validation)])
        assert rc == 0
        assert validation.exists()
        meta = json.loads(out.with_suffix(".manifest.json").read_text())
        assert meta["validation_trace_spearman"] == 1.0
        assert meta["validation_path"] == str(validation)
        assert "RED FLAG" not in capsys.readouterr().err

    def test_red_flag_when_the_surrogate_does_not_track_the_reference(
        self, msa_path, tmp_path, capsys
    ):
        components, _ = build(msa_path)
        reversed_ranks = components.copy()
        # Rank-reverse `trace`: Spearman against the real table is then exactly -1.
        order = np.argsort(np.argsort(components["trace_emitted"].to_numpy(dtype=float)))
        reversed_ranks["trace_emitted"] = np.round(
            np.linspace(1.0, 0.0, len(components))[order], 4
        )
        reference = tmp_path / "ref.res"
        js.write_jet_res(reversed_ranks, reference)

        out = tmp_path / "flagged_jet.res"
        js.main(["--msa", str(msa_path), "--out-jet", str(out),
                 "--trace-definition", "direct",
                 "--validate-against", str(reference),
                 "--min-validation-spearman", "0.99"])
        captured = capsys.readouterr()
        assert "RED FLAG" in captured.err
        meta = json.loads(out.with_suffix(".manifest.json").read_text())
        assert meta["validation_trace_spearman"] < 0.99

    def test_a_parser_that_alters_trace_is_caught_before_success(
        self, msa_path, tmp_path, monkeypatch
    ):
        """The final round-trip assertion is the last line of defence: if the
        file on disk no longer says what the frame said, the run must fail."""
        real = js.read_jet_res

        def _lying(path):
            frame = real(path)
            frame["trace"] = np.round(frame["trace"].to_numpy() * 0.5, 4)
            return frame

        monkeypatch.setattr(js, "read_jet_res", _lying)
        out = tmp_path / "lied_jet.res"
        with pytest.raises(AssertionError, match="round-trip"):
            js.main(["--msa", str(msa_path), "--out-jet", str(out),
                     "--trace-definition", "direct"])

    @pytest.mark.cli
    @pytest.mark.requires_prescott_python
    def test_module_runs_as_a_script(self, msa_path, tmp_path, run_module_cli):
        out = tmp_path / "script_jet.res"
        proc = run_module_cli("jet_surrogate", [
            "--msa", str(msa_path), "--out-jet", str(out),
            "--trace-definition", "direct",
        ], timeout=300)
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        parsed = escott_parser(out)
        assert list(parsed.columns) == EXPECTED_JET_COLUMNS
        assert len(parsed) == C.QUERY_LENGTH


@pytest.mark.unit
class TestCacheShortCircuit:
    """Every input that changes the output must invalidate the cached table."""

    BASE = ["--trace-definition", "direct", "--trace-top-fraction", "0.95"]

    def _run(self, msa, out, extra=()):
        return js.main(["--msa", str(msa), "--out-jet", str(out),
                        *self.BASE, *[str(a) for a in extra]])

    def test_second_identical_run_is_a_cache_hit(self, msa_path, tmp_path, capsys):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        capsys.readouterr()
        assert self._run(msa_path, out) == 0
        assert "cache hit" in capsys.readouterr().out

    def test_force_rebuilds(self, msa_path, tmp_path, capsys):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        capsys.readouterr()
        self._run(msa_path, out, ["--force"])
        stdout = capsys.readouterr().out
        assert "cache hit" not in stdout
        assert "wrote" in stdout

    def test_missing_manifest_is_a_cache_miss(self, msa_path, tmp_path, capsys):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        out.with_suffix(".manifest.json").unlink()
        capsys.readouterr()
        self._run(msa_path, out)
        assert "cache hit" not in capsys.readouterr().out

    def test_corrupt_manifest_is_a_cache_miss_not_a_crash(self, msa_path, tmp_path, capsys):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        out.with_suffix(".manifest.json").write_text("{not json at all")
        capsys.readouterr()
        assert self._run(msa_path, out) == 0
        assert "cache hit" not in capsys.readouterr().out

    @pytest.mark.parametrize("extra", [
        ["--weight-mode", "tjet"],
        ["--trace-definition", "bootstrap", "--trace-bootstraps", "4"],
        ["--trace-top-fraction", "0.80"],
        ["--trace-pseudocount", "0.25"],
        ["--trace-scale-quantile", "0.95"],
        ["--sequence-weighting", "none"],
        ["--seed", "1"],
        ["--pc-mode", "zero"],
        ["--cv-radius", "6.0"],
        ["--cv-atom", "ca"],
        ["--max-coil-length", "9"],
    ], ids=lambda e: e[0].lstrip("-"))
    def test_every_knob_invalidates_the_cache(self, msa_path, tmp_path, capsys, extra):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        capsys.readouterr()
        self._run(msa_path, out, extra)
        assert "cache hit" not in capsys.readouterr().out, extra

    def test_bootstrap_count_is_ignored_under_direct(self, msa_path, tmp_path, capsys):
        """The manifest records ``trace_bootstraps: 0`` under ``direct``, so
        comparing it there would make every rerun a miss."""
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out, ["--trace-bootstraps", "7"])
        capsys.readouterr()
        self._run(msa_path, out, ["--trace-bootstraps", "99"])
        assert "cache hit" in capsys.readouterr().out

    def test_a_changed_msa_invalidates_the_cache(self, msa_path, tmp_path, capsys):
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        capsys.readouterr()
        rows = list(C.TINY_MSA_ROWS)
        rows[1] = "A" * C.QUERY_LENGTH
        other = C.write_fasta(tmp_path / "msa_other.fasta",
                              list(zip(C.TINY_MSA_HEADERS, rows)))
        self._run(other, out)
        assert "cache hit" not in capsys.readouterr().out

    @pytest.mark.requires_prody
    @pytest.mark.requires_freesasa
    @pytest.mark.requires_scipy
    def test_adding_or_changing_a_structure_invalidates_the_cache(
        self, msa_path, tmp_path, capsys, query_numbered_pdb_factory, stub_dssp
    ):
        """6WXB covers 485/566 residues and a contemporary model covers 566/566:
        serving one for the other is silent and large."""
        stub_dssp()
        out = tmp_path / "c_jet.res"
        pdb_a = query_numbered_pdb_factory(name="a.pdb")
        pdb_b = query_numbered_pdb_factory(covered=range(1, 61), name="b.pdb")

        self._run(msa_path, out)
        capsys.readouterr()
        self._run(msa_path, out, ["--pdb", pdb_a])
        assert "cache hit" not in capsys.readouterr().out

        self._run(msa_path, out, ["--pdb", pdb_a])
        assert "cache hit" in capsys.readouterr().out

        self._run(msa_path, out, ["--pdb", pdb_b])
        assert "cache hit" not in capsys.readouterr().out

    @pytest.mark.requires_prody
    @pytest.mark.requires_freesasa
    @pytest.mark.requires_scipy
    def test_changing_only_the_context_pdb_invalidates_the_cache(
        self, msa_path, tmp_path, capsys, query_numbered_pdb_factory, stub_dssp
    ):
        stub_dssp()
        out = tmp_path / "c_jet.res"
        mono = query_numbered_pdb_factory(chains=("A",), name="mono.pdb")
        multi = query_numbered_pdb_factory(chains=("A", "B"), name="multi.pdb")
        self._run(msa_path, out, ["--pdb", mono])
        capsys.readouterr()
        self._run(msa_path, out, ["--pdb", mono, "--context-pdb", multi])
        assert "cache hit" not in capsys.readouterr().out

    def test_structure_chain_invalidates_the_cache(self, msa_path, tmp_path, capsys):
        """Regression test for a fixed cache bug.

        ``--structure-chain`` used to be neither stored in the manifest nor
        compared by the short-circuit, so a rerun on a different chain was served
        the previous chain's structural columns -- and the wrong ``chain`` value --
        with exit code 0.  On a real trimer chain B has different ``cv``, ``pc``
        and DSSP states, so the structural half of the table was silently the
        wrong subunit.
        """
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out, ["--structure-chain", "A"])
        capsys.readouterr()
        self._run(msa_path, out, ["--structure-chain", "B"])
        assert "cache hit" not in capsys.readouterr().out
        assert set(escott_parser(out)["chain"].unique()) == {"B"}

    def test_cache_hit_still_writes_requested_side_outputs(
        self, msa_path, tmp_path, capsys
    ):
        """Regression test for a fixed cache bug.

        The cache short-circuit returns before the ``--out-components`` /
        ``--out-dssp`` blocks, so a first run without them followed by one WITH
        them used to exit 0 having produced nothing -- and the caller (the driver
        asks for the components TSV on reruns) saw success with no file.  The
        cache is now only taken when every requested side output already exists.
        """
        out = tmp_path / "c_jet.res"
        self._run(msa_path, out)
        capsys.readouterr()
        components = tmp_path / "components.tsv"
        assert self._run(msa_path, out, ["--out-components", components]) == 0
        assert components.exists()


# =========================================================================== #
# 17.  --validate-only against the shipped JET2 reference
# =========================================================================== #

@pytest.mark.unit
class TestValidationMode:

    def test_validate_only_does_not_need_msa_or_out_jet(self, monkeypatch, tmp_path):
        called = {}

        def _fake(args):
            called["validation_msa"] = args.validation_msa
            return 0

        monkeypatch.setattr(js, "run_validation", _fake)
        assert js.main(["--validate-only"]) == 0
        assert called["validation_msa"] == Path(constants.BLAT_REFERENCE_MSA)

    def test_missing_validation_inputs_are_named(self, tmp_path):
        args = js.build_parser().parse_args([
            "--validate-only", "--validation-msa", str(tmp_path / "nope.fasta")
        ])
        with pytest.raises(FileNotFoundError, match="--validation-msa"):
            js.run_validation(args)

    def test_missing_reference_is_named(self, tmp_path):
        args = js.build_parser().parse_args([
            "--validate-only", "--validate-against", str(tmp_path / "nope.res")
        ])
        with pytest.raises(FileNotFoundError, match="--validate-against"):
            js.run_validation(args)

    def test_missing_validation_pdb_is_named(self, tmp_path):
        args = js.build_parser().parse_args([
            "--validate-only", "--validation-pdb", str(tmp_path / "nope.pdb")
        ])
        with pytest.raises(FileNotFoundError, match="--validation-pdb"):
            js.run_validation(args)

    @pytest.fixture()
    def stubbed_validation(self, monkeypatch, msa_path, tmp_path):
        """``run_validation`` on the tiny inputs, with the BLAT build stubbed out.

        The real path is exercised (slowly) in
        :class:`TestValidationAgainstRealJet2`; this one is here so the
        ``--out-jet`` / ``--out-manifest`` / RED FLAG branches are covered in
        milliseconds and offline.
        """
        components, meta = build(msa_path)
        monkeypatch.setattr(js, "build_jet_table",
                            lambda *a, **k: (components.copy(), dict(meta)))
        # A rank-reversed reference: Spearman(trace) is exactly -1, so the RED
        # FLAG branch fires at any sane threshold.
        reference = tmp_path / "ref_jet.res"
        reversed_ranks = components.copy()
        order = np.argsort(np.argsort(components["trace_jet"].to_numpy(dtype=float)))
        reversed_ranks["trace_emitted"] = np.round(
            np.linspace(1.0, 0.0, len(components))[order], 4)
        js.write_jet_res(reversed_ranks, reference)
        return components, reference

    def test_validation_writes_every_requested_output(
        self, stubbed_validation, msa_path, tmp_path, capsys
    ):
        components, reference = stubbed_validation
        out_validation = tmp_path / "diag" / "jet_surrogate_vs_blat_reference.tsv"
        out_jet = tmp_path / "diag" / "surrogate_jet.res"
        out_manifest = tmp_path / "diag" / "manifest.json"
        args = js.build_parser().parse_args([
            "--validate-only",
            "--validation-msa", str(msa_path),
            "--validation-pdb", str(msa_path),      # only existence is checked here
            "--validate-against", str(reference),
            "--out-validation", str(out_validation),
            "--out-jet", str(out_jet),
            "--out-manifest", str(out_manifest),
        ])
        assert js.run_validation(args) == 0

        assert out_validation.exists() and out_jet.exists() and out_manifest.exists()
        report = pd.read_csv(out_validation, sep="\t")
        assert report["column"].tolist() == [
            "pc", "tr", "freq", "trace", "trace_emitted_vs_trace"]
        # Rank-reversed, but both sides are rounded to 4 dp so a few ties survive.
        assert report.loc[report["column"] == "trace", "spearman_r"].iloc[0] < -0.99
        assert report["reference"].iloc[0] == str(reference)
        assert report["trace_top_fraction"].iloc[0] == 0.90
        assert report["weight_mode"].iloc[0] == "structural"

        assert len(escott_parser(out_jet)) == len(components)
        meta = json.loads(out_manifest.read_text())
        assert meta["validation_reference"] == str(reference)
        assert meta["validation_path"] == str(out_validation)
        assert meta["validation_spearman"]["trace"] < -0.99

        captured = capsys.readouterr()
        assert "RED FLAG" in captured.err
        assert "zero-trace: ours" in captured.out


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_blat_reference
@pytest.mark.requires_prody
@pytest.mark.requires_freesasa
@pytest.mark.requires_scipy
@pytest.mark.requires_dssp
class TestValidationAgainstRealJet2:
    """The one honest check available: BLAT is the only protein shipping a real .res."""

    @pytest.fixture(scope="class")
    def report(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("validation") / "jet_surrogate_vs_blat.tsv"
        args = js.build_parser().parse_args([
            "--validate-only", "--trace-definition", "direct",
            "--out-validation", str(out),
        ])
        assert js.run_validation(args) == 0
        return pd.read_csv(out, sep="\t"), out

    def test_writes_the_table_caveats_points_at(self, report):
        frame, path = report
        assert path.exists()
        assert frame["column"].tolist() == [
            "pc", "tr", "freq", "trace", "trace_emitted_vs_trace"]
        for column in ("spearman_r", "n", "n_total", "ours_n_zero", "ours_frac_zero",
                       "ref_n_zero", "ref_frac_zero", "weight_mode",
                       "trace_top_fraction", "reference"):
            assert column in frame.columns
        assert (frame["n_total"] == 286).all()

    def test_the_surrogate_tracks_real_jet2_trace(self, report):
        frame, _ = report
        rho = float(frame.set_index("column").loc["trace", "spearman_r"])
        assert rho > 0.4, "below the module's own RED FLAG threshold"

    def test_real_jet2_leaves_three_of_286_columns_at_zero_trace(self, report):
        frame, _ = report
        row = frame.set_index("column").loc["trace"]
        assert int(row["ref_n_zero"]) == 3
        assert float(row["ref_frac_zero"]) == pytest.approx(3 / 286, abs=1e-4)

    def test_the_ceiling_is_disabled_in_the_validation_path(self, tmp_path):
        """Measuring what a bad top fraction does is the point of this mode."""
        out = tmp_path / "bad.tsv"
        args = js.build_parser().parse_args([
            "--validate-only", "--weight-mode", "tjet",
            "--trace-bootstraps", "5", "--trace-top-fraction", "0.30",
            "--out-validation", str(out),
        ])
        assert js.run_validation(args) == 0        # would be a ZeroTraceError normally
        frame = pd.read_csv(out, sep="\t")
        row = frame.set_index("column").loc["trace"]
        assert float(row["ours_frac_zero"]) > js.MAX_ZERO_TRACE_FRACTION

    @pytest.mark.requires_r
    def test_a_full_surrogate_table_for_blat_survives_R(self, tmp_path, subprocess_env):
        out = tmp_path / "BLAT_surrogate_jet.res"
        assert js.main([
            "--msa", str(constants.BLAT_REFERENCE_MSA),
            "--pdb", str(constants.BLAT_REFERENCE_PDB),
            "--out-jet", str(out), "--trace-definition", "direct",
        ]) == 0
        parsed = escott_parser(out)
        assert list(parsed.columns) == EXPECTED_JET_COLUMNS
        assert len(parsed) == 286
        assert np.all(np.isfinite(parsed["trace"].to_numpy(dtype=float)))

        script = tmp_path / "r.R"
        script.write_text(
            'jet <- read.table(commandArgs(TRUE)[1], head=TRUE)\n'
            'cat("NROW", nrow(jet), "ANY_NA", any(is.na(jet$trace)), "\\n")\n'
        )
        proc = subprocess.run(["Rscript", str(script), str(out)],
                              capture_output=True, text=True, env=subprocess_env,
                              timeout=300)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "NROW 286 ANY_NA FALSE"


# =========================================================================== #
# 18.  Small IO helpers
# =========================================================================== #

@pytest.mark.unit
class TestIoHelpers:

    def test_read_fasta_joins_wrapped_lines_and_strips_headers(self, tmp_path):
        path = tmp_path / "wrapped.fasta"
        path.write_text(">  first  desc\nACDE\nFGHI\n\n>second\nKLMN\n")
        records = js.read_fasta(path)
        assert records == [("first  desc", "ACDEFGHI"), ("second", "KLMN")]

    def test_read_fasta_on_an_empty_file(self, tmp_path):
        path = tmp_path / "empty.fasta"
        path.write_text("")
        assert js.read_fasta(path) == []

    def test_read_fasta_preserves_case_and_gaps(self, tmp_path):
        path = tmp_path / "mixed.fasta"
        path.write_text(">q\nac-DE\n")
        assert js.read_fasta(path) == [("q", "ac-DE")]

    def test_md5_of_matches_an_independent_hash(self, tmp_path):
        path = tmp_path / "blob.bin"
        path.write_bytes(b"jet surrogate" * 1000)
        assert js.md5_of(path) == C.md5_file(path)

    def test_ensure_dir_is_idempotent(self, tmp_path):
        target = tmp_path / "a" / "b"
        assert js.ensure_dir(target) == target
        assert js.ensure_dir(target) == target
        assert target.is_dir()

    def test_three_letter_and_one_letter_are_inverse(self):
        assert set(js.THREE_LETTER) == set(AA20)
        for one, three in js.THREE_LETTER.items():
            assert js.ONE_LETTER[three] == one

    def test_gap_chars_cover_the_characters_the_encoder_rejects(self):
        for char in "-.~ *Xx":
            assert char in js.GAP_CHARS
            assert js.encode_msa([("q", char)])[0, 0] == -1
