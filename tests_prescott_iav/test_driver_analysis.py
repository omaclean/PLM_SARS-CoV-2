#!/usr/bin/env python3
r"""Scoring and evaluation maths of ``scripts/run_prescott_diversity.py``.

WHAT THIS FILE IS FOR
=====================
The driver's job is to turn a raw ESCOTT matrix into ``plm_prob`` and then hand the
result to the *shared* analysis half (``run_mutational_accessibility`` +
``Functions_HuggingFace``).  Almost every way that can go wrong is silent: the tables
still fill in, the figures still render, and the only symptom is a number with the
wrong sign or a position shifted by one.  So the tests here are built around three
questions that have exact answers.

1.  **Does the alpha sweep find the alpha that is actually there?**
    :class:`TestAlphaSweepAnalytic` plants a known optimum -- ``obs_freq`` is made a
    strictly increasing function of ``log(plm) + a*log(mut)`` for a chosen ``a`` on
    the grid -- so the sweep's Spearman is *exactly* 1.0 at ``a`` and strictly less
    everywhere else.  Not "roughly right": ``rho == 1.0`` and ``argmax == a``.

2.  **Does a tolerated mutation score HIGH?**
    :class:`TestScoreOrientation` runs the real chain --
    ``write.table``-format ``_normPred_evolCombi.txt`` -> ``run_escott.read_escott_matrix``
    -> ``run_escott.escott_to_probability`` -> ``run_escott.write_score_matrix``
    -> ``driver.ensure_score_matrix`` -> ``rma.build_combined_rows`` -- and asserts
    ``spearman(plm_prob, obs_freq) == +1.0`` when the observed frequency mirrors
    tolerance.  The sign has to survive three conventions in a row:

        ESCOTT raw            negative = deleterious   (pred.R emits -normPred)
        prescott.py output    HIGH     = deleterious   (1 - rankSortData flips it)
        per-column softmax    monotone increasing      (so orientation is preserved)

    A flip anywhere inverts every reported correlation without changing a single
    table's shape, so the same test is run against a deliberately flipped
    (``1 - ranksort``) matrix and required to come out at exactly ``-1.0``.  That is
    what proves the assertion can fail.

3.  **Do the five coordinate systems line up?**
    :class:`TestCoordinateConsistency` pins query protein <-> ``jet.res`` row <-> PDB
    residue number <-> ESCOTT column label ``"<WT><pos>"`` <-> score-matrix
    ``sequence`` row <-> ``combined_long_table.position``.  All are 1-based except the
    matrix column index and ``coord_map``, which are 0-based; an off-by-one is
    invisible in every output file.

Everything else in the module -- the plan/reconcile machinery, the cache guards, the
design keys, the manifest and CAVEATS renderers -- is covered branch by branch, because
those are what decide *which* numbers get reported under *which* label.

GROUND TRUTH
============
The ESCOTT matrix used by the orientation tests is built by :func:`orientation_cells`
rather than taken from ``conftest.escott_matrix_values``.  The reason is arithmetic:
``fill_wildtype`` replaces each column's NA with that column's max, so a column's
softmax denominator depends on which residue is wild type.  :func:`orientation_cells`
gives **every** column the identical multiset of values, which makes ``Z`` identical
across columns and therefore makes the softmax a single globally monotone map from raw
ESCOTT value to probability -- with a closed form (:func:`orientation_partition`) that
a reader can check by hand.

RUNNING
=======
    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_driver_analysis.py -q
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from prescott_iav import constants, run_escott
from tests_prescott_iav import conftest as C

pytestmark = [pytest.mark.unit, pytest.mark.requires_rma]


AA20 = "ACDEFGHIKLMNPQRSTVWY"
"""Alphabetical, matching ``rma.STANDARD_AMINO_ACIDS`` -- the row order
``normalise_plm_matrix`` reindexes to.  Spelled out here so a change to that constant
is caught rather than followed."""


# --------------------------------------------------------------------------- #
# Argument / lineage-data builders.
# --------------------------------------------------------------------------- #

def make_args(driver, out_dir: Path, extra: Sequence[str] = (), **overrides):
    """A real parsed Namespace (so every default is the CLI's), then overrides.

    Never hand-build the Namespace: half the branches under test read a flag this file
    does not mention, and a hand-built one would silently take the ``getattr(..., default)``
    path instead of the real one.  The parameter is ``out_dir``, not ``output_dir``, so
    ``make_args(..., output_dir=None)`` stays available as an override.
    """
    argv = ["--output-dir", str(out_dir), "--analysis-mode", "MONTHLY_GUIDE",
            "--mutation-model", "H3N2", *[str(item) for item in extra]]
    args = driver.build_parser().parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_lineage_data(
    protein: str,
    *,
    obs: Optional[Mapping[Tuple[str, int], float]] = None,
    mut: Optional[Mapping[Tuple[str, int], float]] = None,
    depth: object = 100,
    coord_map: Optional[Dict[int, int]] = None,
    positions: Optional[Sequence[int]] = None,
    lineage_key: str = "K",
) -> Dict[str, object]:
    """The subset of ``rma.build_lineage_cache``'s output that scoring actually reads.

    ``obs``/``mut`` are ``{(aa, 1-based position): value}``; anything unnamed defaults to
    a constant.  ``coord_map`` maps a 0-based *matrix column* to a 0-based *reference*
    position, which is the one place the pipeline changes coordinate origin.
    """
    length = len(protein)
    columns = list(positions) if positions is not None else list(range(1, length + 1))
    rows = list(AA20)
    mut_frame = pd.DataFrame(0.01, index=rows, columns=columns, dtype=float)
    obs_frame = pd.DataFrame(0.0, index=rows, columns=columns, dtype=float)
    for (aa, pos), value in (mut or {}).items():
        mut_frame.loc[aa, pos] = float(value)
    for (aa, pos), value in (obs or {}).items():
        obs_frame.loc[aa, pos] = float(value)
    depth_map = dict(depth) if isinstance(depth, dict) else {pos: int(depth) for pos in columns}
    return {
        "lineage_key": lineage_key,
        "full_ref_protein": protein,
        "plm_ref_protein": protein,
        "coord_map": coord_map if coord_map is not None else {j: j for j in range(length)},
        "mut_profile": mut_frame,
        "obs_freq": obs_frame,
        "obs_depth": depth_map,
        "records": [],
        "diversity_path": "",
        "reference_path": "",
        "any_nucleotide_diversity": False,
        "alignment_diff_stats": {"mapped_sites": length, "compared_sites": length,
                                 "differing_sites": 0, "fixed_differing_sites": 0},
    }


def make_spec(model_tag: str = "ESCOTT", **overrides) -> Dict[str, object]:
    """A model spec shaped exactly like ``build_score_specs`` emits one."""
    spec: Dict[str, object] = {
        "model_tag": model_tag,
        "model_display_label": model_tag,
        "base_model": "ESCOTT",
        "checkpoint_label": None,
        "epoch_label": "escott",
        "epoch_value": 0.0,
        "precomputed_plm_path": None,
        "checkpoint_dir": None,
        "lineages": ["K"],
        "parent_by_lineage": {"K": None},
        "source_variant_by_lineage": {"K": model_tag},
        "matrix_path_by_lineage": {"K": None},
        "equation": None,
        "coefficient": None,
        "frequency_cutoff_k": None,
    }
    spec.update(overrides)
    return spec


# --------------------------------------------------------------------------- #
# The orientation matrix and its closed form.
#
# Every column carries the SAME 19 non-wild-type values, so after fill_wildtype
# (column max) every column carries the same 20-value multiset and therefore the
# same softmax denominator Z.  P is then a single globally monotone function of
# the raw value, with the closed form below.
# --------------------------------------------------------------------------- #

ORIENTATION_STEP = 0.5
ORIENTATION_LEVELS: Tuple[float, ...] = tuple(-ORIENTATION_STEP * k for k in range(19))
"""The 19 raw ESCOTT levels handed to the 19 non-wild-type residues of every column:
0.0 (most tolerated) down to -9.0 (most deleterious), in steps of 0.5.  Negative is
deleterious, matching ``pred.R``'s ``-normPred``."""


def orientation_cells(protein: str) -> Dict[Tuple[str, int], Optional[float]]:
    """``{(AA, 1-based pos): raw ESCOTT value or None on the wild-type cell}``.

    The 19 levels are rotated by position, so no residue keeps the same rank down the
    protein (a matrix whose rows were constant would pass an orientation test that a
    transposed matrix also passed).
    """
    cells: Dict[Tuple[str, int], Optional[float]] = {}
    for pos in range(1, len(protein) + 1):
        wt = protein[pos - 1]
        others = [aa for aa in AA20 if aa != wt]
        assert len(others) == 19
        for rank, aa in enumerate(others):
            cells[(aa, pos)] = ORIENTATION_LEVELS[(rank + pos) % 19]
        cells[(wt, pos)] = None
    return cells


def orientation_partition(temperature: float = 1.0) -> float:
    """``Z``: the softmax denominator every column shares.

    The wild-type NA is filled with the column max, which is ``0.0``; the 19 non-wild-type
    values are :data:`ORIENTATION_LEVELS`.  So

        Z = exp(0/T)  +  sum_{k=0}^{18} exp(-0.5k / T)

    -- one term for the filled wild type and 19 for the real values.
    """
    return math.exp(0.0) + sum(math.exp(level / temperature) for level in ORIENTATION_LEVELS)


def orientation_probability(raw_value: float, temperature: float = 1.0) -> float:
    """The probability :func:`orientation_cells` must produce for one raw value."""
    return math.exp(raw_value / temperature) / orientation_partition(temperature)


def build_orientation_chain(
    tmp_path: Path,
    protein: str,
    *,
    temperature: float = 1.0,
    flip: bool = False,
    name: str = "chain",
) -> Dict[str, object]:
    """Run the REAL stage-1 -> stage-2 handoff and return every intermediate.

    ``flip=True`` replaces the raw matrix with ``prescott.py``'s own orientation
    (``1 - rankSortData``, i.e. HIGH = deleterious).  Nothing else changes, so a test
    that passes on both cannot be testing the sign at all.
    """
    cells = orientation_cells(protein)
    if flip:
        # prescott.py:  1.0 - rankSortData(raw).  Rank ascending within the column, scale
        # to (0, 1], subtract from 1: the most deleterious raw value gets the LARGEST
        # number.  Reproduced by hand rather than imported, so the fixture is independent.
        flipped: Dict[Tuple[str, int], Optional[float]] = {}
        for pos in range(1, len(protein) + 1):
            wt = protein[pos - 1]
            values = {aa: cells[(aa, pos)] for aa in AA20 if aa != wt}
            order = sorted(values, key=lambda aa: values[aa])
            for rank, aa in enumerate(order):
                flipped[(aa, pos)] = 1.0 - (rank + 1) / 19.0
            flipped[(wt, pos)] = None
        cells = flipped

    escott_path = C.write_escott_normpred(
        tmp_path / f"{name}_normPred_evolCombi.txt", protein, (), cells
    )
    raw = run_escott.read_escott_matrix(escott_path, expect_protein=protein)
    probabilities = run_escott.escott_to_probability(raw, temperature=temperature)
    scores_dir = tmp_path / f"{name}_scores"
    matrix_path = run_escott.write_score_matrix(
        probabilities, protein, scores_dir / "K_ESCOTT_score_matrix.csv"
    )
    return {
        "cells": cells,
        "escott_path": escott_path,
        "raw": raw,
        "probabilities": probabilities,
        "scores_dir": scores_dir,
        "matrix_path": matrix_path,
        "protein": protein,
    }


def combined_frame_from_chain(driver, chain: Dict[str, object], args, lineage_data) -> pd.DataFrame:
    """``ensure_score_matrix`` -> ``build_combined_rows``, i.e. what run_analysis does."""
    spec = make_spec("ESCOTT")
    matrix, _path, _sequence = driver.ensure_score_matrix(
        args, spec, "K", lineage_data, chain["scores_dir"]
    )
    rows = driver.rma.build_combined_rows(
        args, spec, "K", lineage_data, matrix, coord_map=lineage_data["coord_map"]
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic combined tables with a planted alpha.
# --------------------------------------------------------------------------- #

def planted_alpha_frame(alpha_true: float, *, n_positions: int = 12, seed: int = 0,
                        lineage: str = "L") -> pd.DataFrame:
    r"""A combined table whose optimal alpha is exactly ``alpha_true``.

    ``obs_freq`` is the rank of ``log(plm) + alpha_true*log(mut)``, scaled into ``(0, 1)``.
    A strictly monotone transform leaves Spearman alone, so at ``alpha == alpha_true``

        rho( log(plm) + alpha*log(mut) , obs_freq ) == 1.0   exactly

    and at any other alpha the two orderings differ (``log plm`` and ``log mut`` are drawn
    independently, so no other linear combination reproduces the ranking).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for position in range(1, n_positions + 1):
        ref_aa = AA20[position % 20]
        for aa in AA20:
            if aa == ref_aa:
                continue
            log_plm = float(rng.uniform(-4.0, -0.5))
            log_mut = float(rng.uniform(-9.0, -2.0))
            rows.append({
                "model": "M", "model_display_label": "M", "base_model": "ESCOTT",
                "checkpoint_label": None, "epoch_label": "e", "epoch_value": 0.0,
                "lineage": lineage, "position": position, "ref_aa": ref_aa, "aa": aa,
                "plm_prob": math.exp(log_plm), "mut_prob": math.exp(log_mut),
                "_target": log_plm + alpha_true * log_mut, "depth": 100.0,
            })
    frame = pd.DataFrame(rows)
    frame["obs_freq"] = frame["_target"].rank(method="first") / (len(frame) + 1.0)
    frame["obs_present"] = 1
    return frame.drop(columns=["_target"])


def run_sweep(driver, frame: pd.DataFrame, alpha_grid: np.ndarray,
              model_label: str = "M") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The exact call ``run_analysis`` makes, including the pseudocount."""
    return driver.rma._build_alpha_tables_from_combined(
        frame, alpha_grid,
        model_label=model_label,
        model_spec={"model_tag": model_label, "epoch_label": "e", "epoch_value": 0.0},
        parallel=False, max_workers=None, alpha_sweep_min_grid=8, pseudocount=1e-16,
    )


def sweep_rows_only(alpha_df: pd.DataFrame) -> pd.DataFrame:
    """Drop the mutation-only baseline row, which carries ``alpha = NaN``."""
    return alpha_df.loc[~alpha_df["is_mutation_only_baseline"].astype(bool)]


ALPHA_GRID = np.round(np.arange(-1.0, 1.0001, 0.25), 6)

TIE_DECIMALS = 12
"""Rounding used before the exact ``rho == +/-1`` orientation assertions.

Two cells carrying the *same* raw ESCOTT value must get the same probability, and they
do -- to within about ``1e-20`` on values of order ``1e-5``.  The residual is pure
floating-point summation order: ``escott_to_probability`` sums each column's 20
exponentials in row order, and :func:`orientation_cells` rotates which residue holds
which level, so two columns add the same 20 numbers in a different sequence.  Left
unrounded that last ulp splits an exact tie into 2-4 distinct ranks and drags a
Spearman of 1.0 down to 0.9998, which would say nothing about orientation and
everything about IEEE 754.  Rounding to 12 decimals (about 8 significant figures on the
smallest probability here) restores the intended ties and lets the assertion stay exact.
"""


# =========================================================================== #
# Small helpers
# =========================================================================== #

class TestSmallHelpers:
    def test_variant_token_strips_dots_and_underscores(self, driver_module):
        assert driver_module.variant_token("G.1") == "G1"
        assert driver_module.variant_token("J_int") == "Jint"
        assert driver_module.variant_token("J.2_int") == "J2int"
        assert driver_module.variant_token("J.2.4") == "J24"
        assert driver_module.variant_token("K") == "K"

    def test_variant_token_is_idempotent_and_non_alnum_free(self, driver_module):
        for label in ("G.1", "J.2_int", "a-b c/d"):
            once = driver_module.variant_token(label)
            assert driver_module.variant_token(once) == once
            assert once.isalnum() or once == ""

    def test_safe_key_matches_functions_huggingface(self, driver_module):
        from Functions_HuggingFace import _safe_label

        for label in ("J.2_int", "A/England/1/2025", " K "):
            assert driver_module.safe_key(label) == _safe_label(label)
        assert driver_module.safe_key("A/England 1") == "A-England_1"

    def test_file_md5_matches_hashlib(self, driver_module, tmp_path):
        import hashlib

        path = tmp_path / "payload.bin"
        payload = b"prescott" * 1000
        path.write_bytes(payload)
        assert driver_module.file_md5(path) == hashlib.md5(payload).hexdigest()

    def test_file_md5_of_missing_file_is_none(self, driver_module, tmp_path):
        assert driver_module.file_md5(tmp_path / "absent") is None

    def test_parse_float_grid(self, driver_module):
        assert driver_module.parse_float_grid("0.25,0.5,1.0") == [0.25, 0.5, 1.0]
        assert driver_module.parse_float_grid(" 1 , , 2 ") == [1.0, 2.0]
        assert driver_module.parse_float_grid("") == []

    def test_parse_int_grid_accepts_float_text(self, driver_module):
        assert driver_module.parse_int_grid("1,2,3") == [1, 2, 3]
        # via float() first, so "2.0" from a TSV round trip is not a ValueError
        assert driver_module.parse_int_grid("2.0, 5.0") == [2, 5]

    def test_iqr_is_the_75_25_percentile_gap(self, driver_module):
        values = np.arange(0.0, 101.0)  # p75 = 75, p25 = 25
        assert driver_module._iqr(values) == pytest.approx(50.0)

    def test_iqr_needs_two_finite_values(self, driver_module):
        assert math.isnan(driver_module._iqr(np.array([1.0])))
        assert math.isnan(driver_module._iqr(np.array([np.nan, np.inf])))

    def test_iqr_ignores_non_finite_entries(self, driver_module):
        values = np.array([0.0, 25.0, 50.0, 75.0, 100.0, np.nan, np.inf, -np.inf])
        assert driver_module._iqr(values) == pytest.approx(50.0)

    @pytest.mark.parametrize("value,expected", [
        (None, None), (np.nan, None), (float("nan"), None), ("", None),
        ("abc", None), (pd.NA, None), (2, 2.0), ("2.5", 2.5), (np.float64(3.0), 3.0),
    ])
    def test_optional_number(self, driver_module, value, expected):
        assert driver_module._optional_number(value) == expected

    def test_optional_number_survives_an_array_like(self, driver_module):
        # pd.isna([1, 2]) returns an array, so `if pd.isna(value)` raises ValueError;
        # the helper has to swallow that rather than take down a cache check.
        assert driver_module._optional_number([1.0, 2.0]) is None
        assert driver_module._normalised_label([1.0, 2.0]) == "[1.0, 2.0]"

    @pytest.mark.parametrize("value,expected", [
        (None, None), (np.nan, None), ("", None), ("   ", None), (pd.NA, None),
        ("K", "K"), (" J.2.4 ", "J.2.4"), (3, "3"),
    ])
    def test_normalised_label(self, driver_module, value, expected):
        assert driver_module._normalised_label(value) == expected


# =========================================================================== #
# The shared constants module is the authority
# =========================================================================== #

class TestSharedConstantsAuthority:
    def test_parent_map_presets_is_the_shared_module_object(self, driver_module):
        presets = driver_module.parent_map_presets()
        assert presets is constants.DEFAULT_PARENT_MAPS
        # The contested edge, spelled out as a literal: K descends from J.2.4.
        assert presets["clade_evidence"] == C.EXPECTED_PARENT_MAP
        assert presets["clade_evidence"]["K"] == "J.2.4"
        assert presets["brief_as_stated"]["K"] == "J.2_int"

    def test_driver_fallback_copy_agrees_with_the_shared_module(self, driver_module):
        # The fallback only matters when prescott_iav is absent, but a stale copy is how
        # the two halves drift; the module warns about it, so it must not be stale here.
        assert driver_module.DEFAULT_PARENT_MAPS == dict(constants.DEFAULT_PARENT_MAPS)

    def test_default_trace_top_fraction_is_the_measured_090(self, driver_module):
        assert driver_module.default_trace_top_fraction() == C.EXPECTED_TRACE_TOP_FRACTION
        assert driver_module.default_trace_top_fraction() == constants.DEFAULT_TRACE_TOP_FRACTION

    def test_presets_fall_back_when_the_shared_module_is_absent(self, driver_module, monkeypatch):
        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: None)
        assert driver_module.parent_map_presets() is driver_module.DEFAULT_PARENT_MAPS

    def test_presets_fall_back_when_the_shared_table_is_empty(self, driver_module, monkeypatch):
        stub = argparse.Namespace(DEFAULT_PARENT_MAPS={})
        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: stub)
        assert driver_module.parent_map_presets() is driver_module.DEFAULT_PARENT_MAPS

    def test_drift_between_the_two_copies_is_reported_once(self, driver_module, monkeypatch, capsys):
        drifted = {"clade_evidence": {"K": "SOMETHING_ELSE"}}
        stub = argparse.Namespace(DEFAULT_PARENT_MAPS=drifted)
        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: stub)
        monkeypatch.setattr(driver_module, "_PRESET_DRIFT_REPORTED", [])
        assert driver_module.parent_map_presets() is drifted
        first = capsys.readouterr().out
        assert "differs from this driver's fallback copy" in first
        driver_module.parent_map_presets()
        assert capsys.readouterr().out == ""  # reported once, not per call

    def test_trace_top_fraction_falls_back_to_090_without_the_module(self, driver_module, monkeypatch):
        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: None)
        assert driver_module.default_trace_top_fraction() == 0.90

    def test_load_constants_returns_none_when_the_file_is_absent(self, driver_module, monkeypatch, tmp_path):
        monkeypatch.setattr(driver_module, "PRESCOTT_IAV_DIR", tmp_path / "nowhere")
        assert driver_module.load_prescott_iav_constants() is None

    def test_load_constants_reraises_a_broken_module(self, driver_module, monkeypatch, tmp_path):
        (tmp_path / "constants.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(driver_module, "PRESCOTT_IAV_DIR", tmp_path)

        def boom(name):
            raise ImportError("no such module")

        monkeypatch.setattr(driver_module.importlib, "import_module", boom)
        with pytest.raises(RuntimeError, match="half-written stage-1 package"):
            driver_module.load_prescott_iav_constants()

    def test_input_only_lineages_is_G1(self, driver_module):
        assert driver_module.input_only_lineages() == frozenset({"G.1"})

    def test_leakage_defaults_quote_the_stage1_numbers(self, driver_module):
        from prescott_iav import leakage_check

        assert driver_module.leakage_default("min_identity") == leakage_check.DEFAULT_MIN_IDENTITY
        assert driver_module.leakage_default("max_hamming") == leakage_check.DEFAULT_MAX_HAMMING
        assert driver_module.leakage_default("coverage_basis") == "both"
        assert driver_module.leakage_default("not_a_threshold") is None

    def test_require_stage1_script_resolves_and_complains(self, driver_module, monkeypatch, tmp_path):
        assert driver_module.require_stage1_script("escott").exists()
        monkeypatch.setitem(driver_module.STAGE1_SCRIPTS, "escott", tmp_path / "gone.py")
        with pytest.raises(RuntimeError, match="Stage-1 script missing"):
            driver_module.require_stage1_script("escott")


# =========================================================================== #
# Parent map resolution
# =========================================================================== #

class TestParentMapResolution:
    def test_default_preset_is_the_corrected_ladder(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path)
        assert args.parent_map_preset == "clade_evidence"
        assert driver_module.resolve_parent_map(args) == C.EXPECTED_PARENT_MAP

    def test_brief_as_stated_preset_restores_the_contested_edge(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map_preset="brief_as_stated")
        assert driver_module.resolve_parent_map(args) == C.EXPECTED_SENSITIVITY_PARENT_MAP

    def test_unknown_preset_is_rejected(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map_preset="made_up")
        with pytest.raises(ValueError, match="Unknown --parent-map-preset"):
            driver_module.resolve_parent_map(args)

    def test_explicit_edges_override_the_preset(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map="K=J.2_int, NEW=K")
        resolved = driver_module.resolve_parent_map(args)
        assert resolved["K"] == "J.2_int"
        assert resolved["NEW"] == "K"
        assert resolved["J.2.4"] == "J.2_int"  # untouched preset edge

    def test_blank_chunks_are_ignored(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map="K=J.2_int,, ,")
        assert driver_module.resolve_parent_map(args)["K"] == "J.2_int"

    @pytest.mark.parametrize("spec,pattern", [
        ("KJ.2_int", "must look like child=parent"),
        ("=J.2_int", "empty side"),
        ("K=", "empty side"),
    ])
    def test_malformed_edges_are_rejected(self, driver_module, tmp_path, spec, pattern):
        args = make_args(driver_module, tmp_path, parent_map=spec)
        with pytest.raises(ValueError, match=pattern):
            driver_module.resolve_parent_map(args)

    def test_a_two_node_cycle_is_rejected(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map="G.1=K")
        with pytest.raises(ValueError, match="cycle"):
            driver_module.resolve_parent_map(args)

    def test_a_self_edge_is_rejected(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_map="K=K")
        with pytest.raises(ValueError, match="cycle"):
            driver_module.resolve_parent_map(args)

    def test_sensitivity_edges_are_only_where_the_presets_disagree(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_sensitivity=True)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {"K": "J.2_int"}

    def test_sensitivity_edges_mirror_when_the_preset_is_swapped(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path,
                         parent_map_preset="brief_as_stated", parent_sensitivity=True)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {"K": "J.2.4"}

    def test_sensitivity_edges_are_empty_when_disabled(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_sensitivity=False)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {}

    def test_sensitivity_edges_are_empty_with_a_single_preset(self, driver_module, tmp_path, monkeypatch):
        monkeypatch.setattr(driver_module, "parent_map_presets",
                            lambda: {"clade_evidence": dict(C.EXPECTED_PARENT_MAP)})
        args = make_args(driver_module, tmp_path, parent_sensitivity=True)
        assert driver_module.sensitivity_edges(args, dict(C.EXPECTED_PARENT_MAP)) == {}

    def test_effective_edges_drop_lineages_that_are_not_evaluated(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_sensitivity=True)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.effective_sensitivity_edges(args, parent_map, ["J_int"]) == {}
        assert driver_module.effective_sensitivity_edges(
            args, parent_map, ["J_int", "K"]) == {"K": "J.2_int"}

    def test_edge_spec_is_what_the_shared_parser_reads_back(self, driver_module):
        spec = driver_module.sensitivity_edge_spec({"K": "J.2_int", "J.2.4": "J_int"})
        assert spec == "J.2.4=J_int,K=J.2_int"  # sorted, so two runs produce one string
        assert constants.parse_edge_spec(spec) == {"K": "J.2_int", "J.2.4": "J_int"}


# =========================================================================== #
# CLI validation
# =========================================================================== #

class TestValidateArgs:
    def test_valid_arguments_pass(self, driver_module, tmp_path):
        driver_module.validate_args(make_args(driver_module, tmp_path,
                                              guide_path=C.REAL_GUIDE))

    def test_regen_figures_only_short_circuits_every_other_check(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, regen_figures_only=True,
                         analysis_mode=None, mutation_model=None, alpha_step=-1.0)
        driver_module.validate_args(args)  # must not raise

    def test_regen_figures_only_still_needs_an_output_dir(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, regen_figures_only=True, output_dir=None)
        with pytest.raises(ValueError, match="--output-dir is required"):
            driver_module.validate_args(args)

    def test_analysis_mode_is_required(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, analysis_mode=None)
        with pytest.raises(ValueError, match="--analysis-mode is required"):
            driver_module.validate_args(args)

    def test_mutation_model_is_required(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, mutation_model=None)
        with pytest.raises(ValueError, match="--mutation-model is required"):
            driver_module.validate_args(args)

    def test_single_fasta_is_refused_with_the_monthly_guide_equivalent(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, analysis_mode="SINGLE_FASTA")
        with pytest.raises(ValueError) as excinfo:
            driver_module.validate_args(args)
        message = str(excinfo.value)
        assert "SINGLE_FASTA is not supported" in message
        assert "MONTHLY_GUIDE" in message and "--parent-map" in message

    def test_monthly_guide_requires_a_guide_path(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=None)
        with pytest.raises(ValueError, match="--guide-path is required"):
            driver_module.validate_args(args)

    def test_monthly_guide_guide_must_exist(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=tmp_path / "absent.csv")
        with pytest.raises(FileNotFoundError, match="Guide file not found"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("step", [0.0, -0.1])
    def test_alpha_step_must_be_positive(self, driver_module, tmp_path, step):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE, alpha_step=step)
        with pytest.raises(ValueError, match="--alpha-step must be > 0"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("temperature", [0.0, -1.0])
    def test_temperature_must_be_positive(self, driver_module, tmp_path, temperature):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE,
                         escott_temperature=temperature)
        with pytest.raises(ValueError, match="--escott-temperature must be > 0"):
            driver_module.validate_args(args)

    def test_match_plm_requires_a_reference_table(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE,
                         escott_temperature_mode="match-plm", plm_reference_table=None)
        with pytest.raises(ValueError, match="requires --plm-reference-table"):
            driver_module.validate_args(args)

    def test_equation_4_is_refused_because_prescott_exits_there(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE, equation_grid="2,4")
        with pytest.raises(ValueError, match="equation 4 is not implemented"):
            driver_module.validate_args(args)

    def test_unknown_equations_are_refused(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE, equation_grid="6")
        with pytest.raises(ValueError, match=r"subset of 1,2,3,5"):
            driver_module.validate_args(args)

    def test_supported_equations_match_the_stage1_module(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE, equation_grid="1,2,3,5")
        driver_module.validate_args(args)
        assert set(run_escott.SUPPORTED_PRESCOTT_EQUATIONS) == {1, 2, 3, 5}

    def test_negative_coefficients_are_refused(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE,
                         coefficient_grid="0.5,-0.1")
        with pytest.raises(ValueError, match="must be >= 0"):
            driver_module.validate_args(args)

    def test_zero_coefficient_is_allowed(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE, coefficient_grid="0")
        driver_module.validate_args(args)


class TestApplyPrescottDefaults:
    def test_derived_directories_hang_off_the_output_dir(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(make_args(driver_module, tmp_path))
        assert args.scores_dir == tmp_path / "scores"
        assert args.inputs_dir == tmp_path / "inputs"
        assert args.escott_workdir == tmp_path / "escott"
        assert args.prescott_ref_dir == tmp_path / "prescott_ref"

    def test_explicit_directories_are_preserved(self, driver_module, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, extra=["--scores-dir", str(elsewhere)])
        )
        assert args.scores_dir == elsewhere
        assert args.inputs_dir == tmp_path / "inputs"

    def test_plm_only_knobs_are_neutralised(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(make_args(driver_module, tmp_path))
        assert args.plm_max_aa_length is None
        assert args.plm_max_nt_length is None
        assert args.use_global_plm_reference is False

    def test_test_mode_zero_records_means_no_truncation(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, test_mode=True, test_max_records=0)
        )
        assert args.test_max_records == 10 ** 9

    def test_test_mode_respects_an_explicit_record_cap(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, test_mode=True, test_max_records=400)
        )
        assert args.test_max_records == 400

    def test_test_mode_does_not_touch_the_trace_definition(self, driver_module, tmp_path):
        # The regression this pins: --test-mode used to force trace_definition='direct',
        # so the smoke test never exercised the production trace path.
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, test_mode=True)
        )
        assert args.trace_definition == "bootstrap"
        assert args.trace_top_fraction is None
        assert args.alpha_step == 0.1

    def test_record_cap_is_untouched_outside_test_mode(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, test_mode=False, test_max_records=0)
        )
        assert args.test_max_records == 0


# =========================================================================== #
# Variant naming and the requested design
# =========================================================================== #

class TestVariantNaming:
    def test_stage1_variant_name_matches_run_escotts_builder(self, driver_module):
        assert driver_module.stage1_variant_name(2, 0.5, 1, "J.2.4") == "PRESCOTT_eq2_c0p50_k1_parentJ24"
        assert driver_module.stage1_variant_name(3, 1.0, 2, "G.1") == "PRESCOTT_eq3_c1p00_k2_parentG1"
        assert driver_module.stage1_variant_name(1, 0.25, 1, "J.2_int") == "PRESCOTT_eq1_c0p25_k1_parentJ2int"

    def test_stage1_variant_name_agrees_with_stage1_itself(self, driver_module):
        # If the two ever disagree the driver predicts a filename stage 1 never wrote and
        # every rerun looks uncached.
        assert driver_module.stage1_variant_name(2, 0.5, 1, "J.2.4") == run_escott.build_variant_name(
            equation=2, coefficient=0.5, frequency_cutoff_k=1, parent_lineage="J.2.4"
        )

    @pytest.mark.parametrize("equation", [None, np.nan, float("nan")])
    def test_no_equation_means_the_escott_baseline(self, driver_module, equation):
        assert driver_module.canonical_model_tag(equation, None, None, None, None) == "ESCOTT"

    def test_primary_parent_gets_a_parent_free_model_tag(self, driver_module):
        assert driver_module.canonical_model_tag(2, 0.5, 1, "J.2.4", "J.2.4") == "PRESCOTT_eq2_c0p50_k1"

    def test_alternate_parent_keeps_the_suffix(self, driver_module):
        assert driver_module.canonical_model_tag(
            2, 0.5, 1, "J.2_int", "J.2.4") == "PRESCOTT_eq2_c0p50_k1_parentJ2int"

    def test_unmapped_lineage_gets_no_suffix(self, driver_module):
        assert driver_module.canonical_model_tag(2, 0.5, 1, "J.2.4", None) == "PRESCOTT_eq2_c0p50_k1"

    def test_plan_entry_key_collapses_a_tsv_round_trip(self, driver_module):
        native = {"lineage": "K", "equation": 2, "coefficient": 0.5,
                  "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"}
        from_tsv = {"lineage": "K", "equation": 2.0, "coefficient": "0.5",
                    "frequency_cutoff_k": np.float64(1.0), "parent_lineage": " J.2.4 "}
        assert driver_module.plan_entry_key(native) == driver_module.plan_entry_key(from_tsv)
        assert driver_module.plan_entry_key(native) == ("K", 2, 0.5, 1, "J.2.4")

    def test_plan_entry_key_for_escott_ignores_everything_else(self, driver_module):
        key = driver_module.plan_entry_key({"lineage": "K", "equation": None,
                                            "coefficient": 9.0, "parent_lineage": "X"})
        assert key == ("K", "ESCOTT")

    def test_plan_entry_key_separates_the_two_parents(self, driver_module):
        base = {"lineage": "K", "equation": 2, "coefficient": 0.5, "frequency_cutoff_k": 1}
        assert driver_module.plan_entry_key({**base, "parent_lineage": "J.2.4"}) != \
            driver_module.plan_entry_key({**base, "parent_lineage": "J.2_int"})

    def test_describe_plan_entry(self, driver_module):
        assert driver_module.describe_plan_entry(
            {"lineage": "K", "equation": None}) == "ESCOTT / K"
        text = driver_module.describe_plan_entry(
            {"lineage": "K", "equation": 2, "coefficient": 0.5,
             "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"})
        assert text == "eq2 c=0.5 k=1 parent=J.2.4 / K"


class TestExpectedVariantPlan:
    @pytest.fixture()
    def plan_args(self, driver_module, tmp_path):
        return make_args(driver_module, tmp_path, coefficient_grid="0.25,0.5",
                         equation_grid="2", frequency_cutoff_k="1")

    def test_plan_size_is_exactly_the_cli_grid(self, driver_module, plan_args):
        evaluable = ["J_int", "K"]
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan_args.parent_sensitivity = False
        plan = driver_module.expected_variant_plan(plan_args, parent_map, evaluable)
        # 2 ESCOTT rows + (1 equation x 2 coefficients x 1 k) x 2 lineages
        assert len(plan) == 2 + 4

    def test_escott_rows_carry_no_parent_and_no_grid_point(self, driver_module, plan_args):
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan = driver_module.expected_variant_plan(plan_args, parent_map, ["K"])
        escott = [entry for entry in plan if entry["source_variant"] == "ESCOTT"]
        assert len(escott) == 1
        assert escott[0]["parent_lineage"] is None
        assert escott[0]["equation"] is None
        assert escott[0]["lineage_key"] == driver_module.safe_key("K")

    def test_sensitivity_edge_adds_a_second_entry_per_grid_point(self, driver_module, plan_args):
        plan_args.parent_sensitivity = True
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan = driver_module.expected_variant_plan(plan_args, parent_map, ["K"])
        parents = sorted({entry["parent_lineage"] for entry in plan
                          if entry["parent_lineage"] is not None})
        assert parents == ["J.2.4", "J.2_int"]
        assert len(plan) == 1 + 2 * 2  # ESCOTT + 2 coefficients x 2 parents

    def test_no_sensitivity_entries_when_disabled(self, driver_module, plan_args):
        plan_args.parent_sensitivity = False
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan = driver_module.expected_variant_plan(plan_args, parent_map, ["K"])
        assert {entry["parent_lineage"] for entry in plan} == {None, "J.2.4"}

    def test_sensitivity_edge_for_an_unevaluated_lineage_is_dropped(self, driver_module, plan_args):
        plan_args.parent_sensitivity = True
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan = driver_module.expected_variant_plan(plan_args, parent_map, ["J_int"])
        assert {entry["parent_lineage"] for entry in plan} == {None, "G.1"}

    def test_every_predicted_name_is_the_stage1_spelling(self, driver_module, plan_args):
        parent_map = driver_module.resolve_parent_map(plan_args)
        plan = driver_module.expected_variant_plan(plan_args, parent_map, ["K"])
        for entry in plan:
            if entry["source_variant"] == "ESCOTT":
                continue
            assert entry["source_variant"] == run_escott.build_variant_name(
                equation=entry["equation"], coefficient=entry["coefficient"],
                frequency_cutoff_k=entry["frequency_cutoff_k"],
                parent_lineage=entry["parent_lineage"],
            )


# =========================================================================== #
# Reconciling the cached table against the requested design
# =========================================================================== #

def write_variants_table(scores_dir: Path, rows: Sequence[Dict[str, object]]) -> Path:
    scores_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / "score_variants.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def touch_matrix(scores_dir: Path, lineage_key: str, variant: str) -> Path:
    scores_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / f"{lineage_key}_{variant}_score_matrix.csv"
    path.write_text("sequence,M\nA,0.5\n", encoding="utf-8")
    return path


class TestScoreMatrixPath:
    def test_local_convention_wins_over_a_stale_recorded_path(self, driver_module, tmp_path):
        scores_dir = tmp_path / "scores"
        local = touch_matrix(scores_dir, "K", "ESCOTT")
        stale = tmp_path / "copied_from" / "K_ESCOTT_score_matrix.csv"
        stale.parent.mkdir(parents=True)
        stale.write_text("x", encoding="utf-8")
        assert driver_module.score_matrix_path(scores_dir, "K", "ESCOTT", str(stale)) == local

    def test_recorded_path_is_used_when_the_local_one_is_absent(self, driver_module, tmp_path):
        recorded = tmp_path / "elsewhere" / "K_ESCOTT_score_matrix.csv"
        recorded.parent.mkdir(parents=True)
        recorded.write_text("x", encoding="utf-8")
        assert driver_module.score_matrix_path(
            tmp_path / "scores", "K", "ESCOTT", str(recorded)) == recorded

    def test_conventional_path_is_the_last_resort(self, driver_module, tmp_path):
        expected = tmp_path / "scores" / "K_ESCOTT_score_matrix.csv"
        assert driver_module.score_matrix_path(tmp_path / "scores", "K", "ESCOTT", None) == expected
        assert driver_module.score_matrix_path(
            tmp_path / "scores", "K", "ESCOTT", str(tmp_path / "gone.csv")) == expected


class TestLoadScoreVariantsTable:
    def test_missing_file_gives_an_empty_frame(self, driver_module, tmp_path):
        assert driver_module.load_score_variants_table(tmp_path).empty

    def test_a_table_without_the_required_columns_is_a_hard_error(self, driver_module, tmp_path):
        write_variants_table(tmp_path, [{"something": 1}])
        with pytest.raises(RuntimeError, match="lacks the required column"):
            driver_module.load_score_variants_table(tmp_path)

    def test_variant_plan_from_table_filters_to_the_evaluable_set(self, driver_module, tmp_path):
        write_variants_table(tmp_path, [
            {"variant": "ESCOTT", "lineage": "K", "lineage_key": "K"},
            {"variant": "ESCOTT", "lineage": "J_int", "lineage_key": "J_int"},
        ])
        table = driver_module.load_score_variants_table(tmp_path)
        plan = driver_module.variant_plan_from_table(table, ["K"])
        assert [entry["lineage"] for entry in plan] == ["K"]

    def test_variant_plan_from_table_normalises_missing_cells(self, driver_module, tmp_path):
        write_variants_table(tmp_path, [{"variant": "ESCOTT", "lineage": "K"}])
        table = driver_module.load_score_variants_table(tmp_path)
        entry = driver_module.variant_plan_from_table(table, ["K"])[0]
        assert entry["parent_lineage"] is None
        assert entry["equation"] is None
        assert entry["score_matrix_path"] is None
        assert entry["lineage_key"] == "K"


class TestReconcileVariantPlan:
    @pytest.fixture()
    def requested(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, coefficient_grid="0.25,0.5",
                         equation_grid="2", frequency_cutoff_k="1", parent_sensitivity=False)
        parent_map = driver_module.resolve_parent_map(args)
        return driver_module.expected_variant_plan(args, parent_map, ["K"])

    def test_an_empty_cache_makes_everything_missing(self, driver_module, tmp_path, requested):
        plan, missing, ignored = driver_module.reconcile_variant_plan(
            requested, pd.DataFrame(), tmp_path / "scores", ["K"])
        assert len(plan) == len(requested)
        assert len(missing) == len(requested)
        assert all("not in score_variants.tsv" in item for item in missing)
        assert ignored == []

    def test_a_complete_cache_leaves_nothing_missing(self, driver_module, tmp_path, requested):
        scores_dir = tmp_path / "scores"
        rows = []
        for entry in requested:
            touch_matrix(scores_dir, "K", str(entry["source_variant"]))
            rows.append({"variant": entry["source_variant"], "lineage": "K", "lineage_key": "K",
                         "parent_lineage": entry["parent_lineage"], "equation": entry["equation"],
                         "coefficient": entry["coefficient"],
                         "frequency_cutoff_k": entry["frequency_cutoff_k"]})
        table = pd.DataFrame(rows)
        plan, missing, ignored = driver_module.reconcile_variant_plan(
            requested, table, scores_dir, ["K"])
        assert missing == []
        assert ignored == []
        assert len(plan) == len(requested)
        assert all(Path(str(entry["score_matrix_path"])).exists() for entry in plan)

    def test_a_cached_row_whose_matrix_is_gone_is_missing_by_path(self, driver_module, tmp_path, requested):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        table = pd.DataFrame([{"variant": "ESCOTT", "lineage": "K", "lineage_key": "K"}])
        _plan, missing, _ignored = driver_module.reconcile_variant_plan(
            requested, table, scores_dir, ["K"])
        assert str(scores_dir / "K_ESCOTT_score_matrix.csv") in missing

    def test_a_shrunk_grid_reports_the_dropped_coefficients_as_ignored(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, coefficient_grid="0.5",
                         equation_grid="2", frequency_cutoff_k="1", parent_sensitivity=False)
        parent_map = driver_module.resolve_parent_map(args)
        requested = driver_module.expected_variant_plan(args, parent_map, ["K"])
        scores_dir = tmp_path / "scores"
        rows = []
        for coefficient in (0.25, 0.5, 1.0):
            variant = driver_module.stage1_variant_name(2, coefficient, 1, "J.2.4")
            touch_matrix(scores_dir, "K", variant)
            rows.append({"variant": variant, "lineage": "K", "lineage_key": "K",
                         "parent_lineage": "J.2.4", "equation": 2,
                         "coefficient": coefficient, "frequency_cutoff_k": 1})
        touch_matrix(scores_dir, "K", "ESCOTT")
        rows.append({"variant": "ESCOTT", "lineage": "K", "lineage_key": "K",
                     "parent_lineage": None, "equation": None,
                     "coefficient": None, "frequency_cutoff_k": None})
        plan, missing, ignored = driver_module.reconcile_variant_plan(
            requested, pd.DataFrame(rows), scores_dir, ["K"])
        assert missing == []
        assert len(plan) == len(requested) == 2  # ESCOTT + c=0.5, never 4
        assert len(ignored) == 2
        assert all("c=0.5 " not in item for item in ignored)

    def test_the_plan_adopts_stage1s_own_variant_name(self, driver_module, tmp_path, requested):
        scores_dir = tmp_path / "scores"
        touch_matrix(scores_dir, "K", "SOME_OTHER_NAME")
        table = pd.DataFrame([{"variant": "SOME_OTHER_NAME", "lineage": "K", "lineage_key": "K",
                               "parent_lineage": "J.2.4", "equation": 2, "coefficient": 0.5,
                               "frequency_cutoff_k": 1}])
        plan, _missing, _ignored = driver_module.reconcile_variant_plan(
            requested, table, scores_dir, ["K"])
        matched = [entry for entry in plan
                   if driver_module.plan_entry_key(entry) == ("K", 2, 0.5, 1, "J.2.4")]
        assert len(matched) == 1
        assert matched[0]["source_variant"] == "SOME_OTHER_NAME"

    def test_the_cache_can_never_lengthen_the_plan(self, driver_module, tmp_path, requested):
        scores_dir = tmp_path / "scores"
        rows = [{"variant": driver_module.stage1_variant_name(2, c, 1, "J.2.4"), "lineage": "K",
                 "lineage_key": "K", "parent_lineage": "J.2.4", "equation": 2,
                 "coefficient": c, "frequency_cutoff_k": 1} for c in (0.25, 0.5, 0.75, 1.0, 2.0)]
        plan, _missing, ignored = driver_module.reconcile_variant_plan(
            requested, pd.DataFrame(rows), scores_dir, ["K"])
        assert len(plan) == len(requested)
        assert len(ignored) == 3


# =========================================================================== #
# Model specs
# =========================================================================== #

class TestBuildScoreSpecs:
    @pytest.fixture()
    def context(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, coefficient_grid="0.5", equation_grid="2",
                         frequency_cutoff_k="1", parent_sensitivity=True)
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, ["J.2.4", "K"])
        return args, parent_map, plan

    def test_lineages_collapse_into_one_model_per_grid_point(self, driver_module, context):
        args, parent_map, plan = context
        specs = driver_module.build_score_specs(args, plan, parent_map)
        tags = [str(spec["model_tag"]) for spec in specs]
        assert tags == ["ESCOTT", "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int"]
        escott = specs[0]
        assert sorted(escott["lineages"]) == ["J.2.4", "K"]

    def test_epoch_value_is_the_prescott_coefficient(self, driver_module, context):
        args, parent_map, plan = context
        specs = driver_module.build_score_specs(args, plan, parent_map)
        by_tag = {str(spec["model_tag"]): spec for spec in specs}
        assert by_tag["ESCOTT"]["epoch_value"] == 0.0
        assert by_tag["ESCOTT"]["epoch_label"] == "escott"
        assert by_tag["PRESCOTT_eq2_c0p50_k1"]["epoch_value"] == 0.5
        assert by_tag["PRESCOTT_eq2_c0p50_k1"]["epoch_label"] == "prescott_c0.50"

    def test_the_sensitivity_model_is_labelled_by_its_parent(self, driver_module, context):
        args, parent_map, plan = context
        specs = driver_module.build_score_specs(args, plan, parent_map)
        sensitivity = [spec for spec in specs
                       if str(spec["model_tag"]).endswith("_parentJ2int")][0]
        assert sensitivity["lineages"] == ["K"]
        assert sensitivity["parent_by_lineage"] == {"K": "J.2_int"}
        assert sensitivity["epoch_label"] == "prescott_c0.50_parentJ2int"
        assert "parent J.2_int" in str(sensitivity["model_display_label"])

    def test_score_variant_accepts_the_model_tag(self, driver_module, context):
        args, parent_map, plan = context
        args.score_variants = ["PRESCOTT_eq2_c0p50_k1"]
        specs = driver_module.build_score_specs(args, plan, parent_map)
        assert [str(spec["model_tag"]) for spec in specs] == ["PRESCOTT_eq2_c0p50_k1"]

    def test_score_variant_also_accepts_the_stage1_variant_name(self, driver_module, context):
        args, parent_map, plan = context
        args.score_variants = ["PRESCOTT_eq2_c0p50_k1_parentJ24"]
        specs = driver_module.build_score_specs(args, plan, parent_map)
        assert [str(spec["model_tag"]) for spec in specs] == ["PRESCOTT_eq2_c0p50_k1"]

    def test_an_unknown_score_variant_lists_both_vocabularies(self, driver_module, context):
        args, parent_map, plan = context
        args.score_variants = ["PRESCOTT_eq9_c9p99_k9"]
        with pytest.raises(ValueError) as excinfo:
            driver_module.build_score_specs(args, plan, parent_map)
        message = str(excinfo.value)
        assert "Model tags" in message and "Stage-1 variants" in message
        assert "PRESCOTT_eq2_c0p50_k1_parentJ24" in message

    def test_an_empty_plan_is_a_hard_error(self, driver_module, context):
        args, parent_map, _plan = context
        with pytest.raises(RuntimeError, match="No score variants resolved"):
            driver_module.build_score_specs(args, [], parent_map)


# =========================================================================== #
# Reading a score matrix back
# =========================================================================== #

class TestEnsureScoreMatrix:
    def test_reads_a_20xL_matrix_and_recovers_the_sequence(self, driver_module, tmp_path,
                                                           score_matrix_factory):
        protein = C.QUERY_PROTEIN
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        written = score_matrix_factory(protein)
        target = scores_dir / "K_ESCOTT_score_matrix.csv"
        target.write_bytes(written.read_bytes())

        args = make_args(driver_module, tmp_path)
        matrix, path, sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", make_lineage_data(protein), scores_dir)
        assert Path(path) == target
        assert sequence == protein
        assert list(matrix.index) == list(AA20)
        assert matrix.shape == (20, len(protein))
        assert matrix.to_numpy().min() == pytest.approx(0.05)

    def test_a_missing_matrix_names_the_command_that_makes_it(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            driver_module.ensure_score_matrix(
                args, make_spec("ESCOTT"), "K", make_lineage_data(C.QUERY_PROTEIN),
                tmp_path / "scores")
        message = str(excinfo.value)
        assert "run_escott.py" in message and "--auto-prepare" in message

    def test_falls_back_to_the_lineage_reference_without_a_sequence_row(self, driver_module, tmp_path):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        frame = pd.DataFrame(0.05, index=list(AA20), columns=range(1, 6))
        frame.to_csv(scores_dir / "K_ESCOTT_score_matrix.csv", header=False)
        args = make_args(driver_module, tmp_path)
        _matrix, _path, sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", make_lineage_data("MKTII"), scores_dir)
        assert sequence == "MKTII"

    def test_the_recorded_path_is_used_when_it_is_the_only_copy(self, driver_module, tmp_path,
                                                                score_matrix_factory):
        recorded = score_matrix_factory(C.QUERY_PROTEIN, name="recorded_matrix.csv")
        spec = make_spec("ESCOTT", matrix_path_by_lineage={"K": str(recorded)})
        args = make_args(driver_module, tmp_path)
        _matrix, path, _sequence = driver_module.ensure_score_matrix(
            args, spec, "K", make_lineage_data(C.QUERY_PROTEIN), tmp_path / "empty_scores")
        assert Path(path) == recorded


# =========================================================================== #
# Stage-1 orchestration: what the driver actually asks stage 1 for
# =========================================================================== #

def flag_value(command: Sequence[str], flag: str) -> Optional[str]:
    """The argument following ``flag``, or None if the flag is absent."""
    parts = [str(part) for part in command]
    return parts[parts.index(flag) + 1] if flag in parts else None


def flag_values(command: Sequence[str], flag: str) -> List[str]:
    """Every argument following a repeated ``flag`` (``--only-lineage``, ``--lineage``)."""
    parts = [str(part) for part in command]
    return [parts[i + 1] for i, part in enumerate(parts) if part == flag]


class TestStage1Orchestration:
    @pytest.fixture()
    def captured(self, driver_module, tmp_path, monkeypatch, prepared_inputs_tree):
        """Run ``run_stage1`` with the subprocess runner stubbed out."""
        calls: List[Tuple[str, List[str]]] = []

        def fake_step(command, env, label):
            calls.append((label, [str(part) for part in command]))

        monkeypatch.setattr(driver_module, "run_stage1_step", fake_step)

        def _run(**overrides):
            calls.clear()
            deep = tmp_path / "deep.fasta"
            deep.write_text(">a\nMK\n", encoding="utf-8")
            args = make_args(driver_module, tmp_path / "out", guide_path=C.REAL_GUIDE,
                             deep_fasta=deep,
                             inputs_dir=prepared_inputs_tree["inputs_dir"])
            args = driver_module.apply_prescott_defaults(args)
            args.inputs_dir = prepared_inputs_tree["inputs_dir"]
            args.prescott_python = C.PRESCOTT_PYTHON
            for key, value in overrides.items():
                setattr(args, key, value)
            parent_map = driver_module.resolve_parent_map(args)
            evaluable = overrides.pop("_evaluable", ["K"])
            driver_module.run_stage1(args, parent_map, evaluable,
                                     tmp_path / "out" / "tables" / "diagnostics")
            return dict(calls), args

        return _run

    def test_the_three_stages_run_in_order(self, captured):
        calls, _args = captured()
        labels = list(calls)
        assert labels[0] == "prepare_inputs"
        assert "jet_surrogate:K" in labels
        assert labels[-1] == "run_escott"

    def test_prepared_lineages_are_targets_plus_parents_plus_alternates(self, captured):
        calls, _args = captured()
        assert flag_values(calls["prepare_inputs"], "--only-lineage") == \
            ["J.2.4", "J.2_int", "K"]

    def test_trace_top_fraction_is_not_forwarded_unless_set(self, captured):
        calls, _args = captured()
        assert "--trace-top-fraction" not in calls["jet_surrogate:K"]
        assert "--max-zero-trace-fraction" not in calls["jet_surrogate:K"]

    def test_an_explicit_trace_top_fraction_is_forwarded(self, captured):
        calls, _args = captured(trace_top_fraction=0.30, max_zero_trace_fraction=1.0)
        assert flag_value(calls["jet_surrogate:K"], "--trace-top-fraction") == "0.3"
        assert flag_value(calls["jet_surrogate:K"], "--max-zero-trace-fraction") == "1.0"

    def test_drop_parent_reversions_is_pinned_in_both_directions(self, captured):
        calls, _args = captured()
        assert "--drop-parent-reversions" in calls["prepare_inputs"]
        calls, _args = captured(drop_parent_reversions=False)
        assert "--no-drop-parent-reversions" in calls["prepare_inputs"]

    def test_the_leakage_booleans_are_always_explicit(self, captured):
        calls, _args = captured()
        assert "--leakage-check" in calls["prepare_inputs"]
        assert "--purge-leakage" in calls["prepare_inputs"]
        assert "--fail-on-leakage" not in calls["prepare_inputs"]
        calls, _args = captured(leakage_check=False, purge_leakage=False, fail_on_leakage=True)
        assert "--no-leakage-check" in calls["prepare_inputs"]
        assert "--no-purge-leakage" in calls["prepare_inputs"]
        assert "--fail-on-leakage" in calls["prepare_inputs"]

    def test_leakage_thresholds_are_forwarded_only_when_set(self, captured):
        calls, _args = captured()
        for flag in ("--leakage-min-identity", "--leakage-max-hamming",
                     "--leakage-min-coverage", "--leakage-coverage-basis",
                     "--leakage-max-removed-fraction", "--leakage-min-depth-after",
                     "--leakage-threads", "--blast-task"):
            assert flag not in calls["prepare_inputs"], flag
        calls, _args = captured(leakage_min_identity="98.0", leakage_max_hamming="5",
                                blast_task="blastp")
        assert flag_value(calls["prepare_inputs"], "--leakage-min-identity") == "98.0"
        assert flag_value(calls["prepare_inputs"], "--leakage-max-hamming") == "5"
        assert flag_value(calls["prepare_inputs"], "--blast-task") == "blastp"

    def test_the_sensitivity_edge_reaches_both_stage1_scripts(self, captured):
        calls, _args = captured()
        assert flag_value(calls["prepare_inputs"], "--sensitivity-parent-map") == "K=J.2_int"
        assert flag_value(calls["run_escott"], "--sensitivity-parent-map") == "K=J.2_int"
        assert "--no-parent-sensitivity" not in calls["run_escott"]

    def test_disabling_sensitivity_is_forwarded_so_a_cached_edge_cannot_return(self, captured):
        calls, _args = captured(parent_sensitivity=False)
        assert "--sensitivity-parent-map" not in calls["prepare_inputs"]
        assert "--no-parent-sensitivity" in calls["run_escott"]

    def test_the_diagnostics_dir_is_pinned_to_this_output_tree(self, captured, tmp_path):
        calls, _args = captured()
        expected = str(tmp_path / "out" / "tables" / "diagnostics")
        assert flag_value(calls["run_escott"], "--diagnostics-dir") == expected

    def test_the_jet_validation_table_lands_in_the_diagnostics_dir(self, captured, tmp_path):
        calls, _args = captured()
        expected = str(tmp_path / "out" / "tables" / "diagnostics"
                       / driver_validation_basename())
        assert "--validate-only" in calls["jet_surrogate:validate"]
        assert flag_value(calls["jet_surrogate:validate"], "--out-validation") == expected

    def test_validation_can_be_switched_off(self, captured):
        calls, _args = captured(jet_validation=False)
        assert "jet_surrogate:validate" not in calls

    def test_the_context_pdb_selects_trimer_or_monomer(self, captured, prepared_inputs_tree):
        calls, _args = captured(sasa_context="trimer")
        assert flag_value(calls["jet_surrogate:K"], "--context-pdb") == \
            str(prepared_inputs_tree["trimer_pdb"])
        calls, _args = captured(sasa_context="monomer")
        assert flag_value(calls["jet_surrogate:K"], "--context-pdb") == \
            str(prepared_inputs_tree["monomer_pdb"])

    def test_the_parity_run_is_skipped_in_test_mode(self, captured):
        calls, _args = captured()
        assert "--prescott-ref-dir" in calls["run_escott"]
        calls, _args = captured(test_mode=True)
        assert "--prescott-ref-dir" not in calls["run_escott"]
        # ...but run_escott's own --test-mode is NOT forwarded: that would drop every
        # PRESCOTT variant and leave the smoke test exercising half the score path.
        assert "--test-mode" not in calls["run_escott"]

    def test_force_recompute_reaches_every_step(self, captured):
        calls, _args = captured(force_recompute_scores=True)
        assert "--force" in calls["prepare_inputs"]
        assert "--force" in calls["jet_surrogate:K"]
        assert "--force" in calls["run_escott"]

    def test_extra_args_are_appended_verbatim(self, captured):
        calls, _args = captured(prepare_args="--foo 1", jet_args="--bar", escott_args="--baz 2")
        assert calls["prepare_inputs"][-2:] == ["--foo", "1"]
        assert calls["jet_surrogate:K"][-1] == "--bar"
        assert calls["run_escott"][-2:] == ["--baz", "2"]

    def test_run_escott_is_told_exactly_which_lineages_to_score(self, captured):
        calls, _args = captured()
        assert flag_values(calls["run_escott"], "--lineage") == ["K"]
        assert flag_value(calls["run_escott"], "--escott-temperature") == "1.0"

    def test_a_missing_interpreter_is_refused_before_anything_runs(
            self, driver_module, tmp_path, monkeypatch):
        monkeypatch.setattr(driver_module, "run_stage1_step",
                            lambda *a, **k: pytest.fail("stage 1 must not start"))
        args = driver_module.apply_prescott_defaults(
            make_args(driver_module, tmp_path, prescott_python=tmp_path / "no-python"))
        with pytest.raises(RuntimeError, match="does not exist"):
            driver_module.run_stage1(args, dict(C.EXPECTED_PARENT_MAP), ["K"], tmp_path)


def driver_validation_basename() -> str:
    return "jet_surrogate_vs_blat_reference.tsv"


class TestStage1Subprocess:
    def test_the_environment_puts_the_prescott_bin_first(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, prescott_python=C.PRESCOTT_PYTHON)
        env = driver_module.stage1_environment(args)
        assert env["PATH"].split(":")[0] == str(C.PRESCOTT_ENV_BIN)
        assert env["MPLBACKEND"] == "Agg"
        assert env["R_LIBS_USER"] == ""

    def test_a_successful_step_prints_the_command_and_returns(self, driver_module, capsys):
        import os as _os

        driver_module.run_stage1_step(
            [sys.executable, "-c", "pass"], dict(_os.environ), "noop")
        out = capsys.readouterr().out
        assert "[stage1:noop]" in out and "ok in" in out

    def test_a_failing_step_raises_with_the_exit_code_and_command(self, driver_module):
        import os as _os

        with pytest.raises(RuntimeError, match=r"failed with exit code 3"):
            driver_module.run_stage1_step(
                [sys.executable, "-c", "import sys; sys.exit(3)"], dict(_os.environ), "boom")


@pytest.mark.requires_real_data
class TestResolveTargets:
    def test_the_shipped_guide_resolves_the_five_lineages(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, guide_path=C.REAL_GUIDE)
        labels = [str(target["label"]) for target in driver_module.resolve_targets(args)]
        assert labels[:5] == list(C.LINEAGE_ORDER)


# =========================================================================== #
# SCORE ORIENTATION -- the sign convention, end to end
# =========================================================================== #

@pytest.mark.integration
class TestScoreOrientation:
    PROTEIN = C.QUERY_PROTEIN[:24]

    def test_softmax_probability_has_the_hand_derived_closed_form(self, tmp_path):
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="closed_form")
        probabilities = chain["probabilities"]
        for position in (1, 7, 24):
            wt = self.PROTEIN[position - 1]
            for aa in AA20:
                if aa == wt:
                    continue
                raw = chain["cells"][(aa, position)]
                assert probabilities.at[aa, position] == pytest.approx(
                    orientation_probability(raw), rel=0, abs=1e-12)

    def test_every_column_sums_to_one(self, tmp_path):
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="sums")
        sums = chain["probabilities"].sum(axis=0).to_numpy()
        assert np.allclose(sums, 1.0, atol=1e-12)

    def test_a_more_tolerated_mutation_gets_a_higher_probability(self, tmp_path):
        """The core of the sign convention: ESCOTT raw is negative-is-deleterious."""
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="within_column")
        for position in range(1, len(self.PROTEIN) + 1):
            wt = self.PROTEIN[position - 1]
            raw = {aa: chain["cells"][(aa, position)] for aa in AA20 if aa != wt}
            probability = {aa: chain["probabilities"].at[aa, position] for aa in raw}
            by_raw = sorted(raw, key=lambda aa: raw[aa])
            by_probability = sorted(probability, key=lambda aa: probability[aa])
            assert by_raw == by_probability

    def test_a_zero_trace_column_softmaxes_to_exactly_one_twentieth(self, tmp_path):
        """pred.R multiplies each column by trace[i]; trace == 0 is a dead site."""
        cells = orientation_cells(self.PROTEIN)
        flat_position = 3
        for aa in AA20:
            if cells[(aa, flat_position)] is not None:
                cells[(aa, flat_position)] = 0.0
        path = C.write_escott_normpred(tmp_path / "flat_normPred_evolCombi.txt",
                                       self.PROTEIN, (), cells)
        matrix = run_escott.escott_to_probability(
            run_escott.read_escott_matrix(path, expect_protein=self.PROTEIN))
        column = matrix[flat_position].to_numpy()
        assert np.allclose(column, 0.05, atol=1e-15)
        assert run_escott.count_flat_columns(
            run_escott.read_escott_matrix(path, expect_protein=self.PROTEIN)) == 1

    def test_end_to_end_a_tolerated_mutation_scores_higher_than_a_deleterious_one(
            self, driver_module, tmp_path):
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="e2e")
        lineage_data = make_lineage_data(self.PROTEIN)
        args = make_args(driver_module, tmp_path)
        frame = combined_frame_from_chain(driver_module, chain, args, lineage_data)

        assert len(frame) == 19 * len(self.PROTEIN)
        # For every row, plm_prob must be the monotone image of the raw ESCOTT value.
        raw_values = np.array([chain["cells"][(row.aa, row.position)]
                               for row in frame.itertuples()])
        expected = np.array([orientation_probability(value) for value in raw_values])
        assert np.allclose(frame["plm_prob"].to_numpy(), expected, atol=1e-12)
        # And the single best-scoring row is a raw value of 0.0, never -9.0.
        assert raw_values[frame["plm_prob"].to_numpy().argmax()] == pytest.approx(0.0)
        assert raw_values[frame["plm_prob"].to_numpy().argmin()] == pytest.approx(
            min(ORIENTATION_LEVELS))

    def test_end_to_end_correlation_with_observed_diversity_is_exactly_plus_one(
            self, driver_module, tmp_path):
        """A tolerated mutation must end up HIGHER against observed frequency."""
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="rho_plus")
        obs = {}
        for (aa, position), raw in chain["cells"].items():
            if raw is None:
                continue
            # obs_freq strictly increasing in tolerance, scaled into (0, 1).
            obs[(aa, position)] = (raw - min(ORIENTATION_LEVELS) + 1.0) / 100.0
        lineage_data = make_lineage_data(self.PROTEIN, obs=obs)
        args = make_args(driver_module, tmp_path)
        frame = combined_frame_from_chain(driver_module, chain, args, lineage_data)
        rho = driver_module.rma.safe_spearman(
            frame["plm_prob"].round(TIE_DECIMALS), frame["obs_freq"])
        assert rho == pytest.approx(1.0, abs=1e-12)
        # Unrounded it is still overwhelmingly positive; see TIE_DECIMALS for the gap.
        assert driver_module.rma.safe_spearman(frame["plm_prob"], frame["obs_freq"]) > 0.999

    def test_the_flipped_prescott_orientation_gives_exactly_minus_one(
            self, driver_module, tmp_path):
        """Proof that the previous test can fail.

        ``prescott.py`` emits ``1 - rankSortData(raw)``, i.e. HIGH = deleterious.  Feeding
        that orientation through the same softmax inverts the reported correlation without
        changing one row count, which is exactly the silent failure mode.
        """
        reference = build_orientation_chain(tmp_path, self.PROTEIN, name="ref_for_flip")
        flipped = build_orientation_chain(tmp_path, self.PROTEIN, flip=True, name="rho_minus")
        obs = {}
        for (aa, position), raw in reference["cells"].items():
            if raw is None:
                continue
            obs[(aa, position)] = (raw - min(ORIENTATION_LEVELS) + 1.0) / 100.0
        lineage_data = make_lineage_data(self.PROTEIN, obs=obs)
        args = make_args(driver_module, tmp_path)
        frame = combined_frame_from_chain(driver_module, flipped, args, lineage_data)
        rho = driver_module.rma.safe_spearman(
            frame["plm_prob"].round(TIE_DECIMALS), frame["obs_freq"])
        assert rho == pytest.approx(-1.0, abs=1e-12)
        assert driver_module.rma.safe_spearman(frame["plm_prob"], frame["obs_freq"]) < -0.999

    def test_the_headline_epoch_metric_carries_the_same_sign(self, driver_module, tmp_path):
        """``epoch_lineage_metrics.tsv``'s spearman_obs_freq_vs_plm is what gets reported."""
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="epoch")
        obs = {(aa, position): (raw - min(ORIENTATION_LEVELS) + 1.0) / 100.0
               for (aa, position), raw in chain["cells"].items() if raw is not None}
        lineage_data = make_lineage_data(self.PROTEIN, obs=obs)
        args = make_args(driver_module, tmp_path)
        frame = combined_frame_from_chain(driver_module, chain, args, lineage_data)
        frame["lineage"] = "K"
        frame["plm_prob"] = frame["plm_prob"].round(TIE_DECIMALS)
        metrics = driver_module.rma.compute_epoch_lineage_metrics(frame)
        assert len(metrics) == 1
        assert float(metrics.loc[0, "spearman_obs_freq_vs_plm"]) == pytest.approx(1.0, abs=1e-12)

    def test_alpha_zero_ranks_by_the_model_alone(self, driver_module, tmp_path):
        """At alpha = 0 the combined score is log(plm_prob), so the sweep sees the model."""
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="alpha0")
        obs = {(aa, position): (raw - min(ORIENTATION_LEVELS) + 1.0) / 100.0
               for (aa, position), raw in chain["cells"].items() if raw is not None}
        lineage_data = make_lineage_data(self.PROTEIN, obs=obs)
        args = make_args(driver_module, tmp_path)
        frame = combined_frame_from_chain(driver_module, chain, args, lineage_data)
        frame["lineage"] = "K"
        frame["plm_prob"] = frame["plm_prob"].round(TIE_DECIMALS)
        alpha_df, _by_lineage = run_sweep(driver_module, frame, np.array([0.0]))
        sweep = sweep_rows_only(alpha_df)
        assert len(sweep) == 1
        assert float(sweep.iloc[0]["mut_flat_global_spearman_r"]) == pytest.approx(1.0, abs=1e-12)

    def test_temperature_rescales_the_spread_without_reordering(self, driver_module, tmp_path):
        r"""``log P = E/T - log Z``, so ``sd(log P) == sd(E)/T`` exactly and the order is fixed."""
        hot = build_orientation_chain(tmp_path, self.PROTEIN, temperature=2.0, name="T2")
        cold = build_orientation_chain(tmp_path, self.PROTEIN, temperature=1.0, name="T1")
        hot_values = np.log(hot["probabilities"].to_numpy().ravel())
        cold_values = np.log(cold["probabilities"].to_numpy().ravel())
        assert np.std(cold_values) == pytest.approx(2.0 * np.std(hot_values), rel=1e-12)
        # Ranks identical: softmax at any positive T is strictly monotone in E.
        from scipy.stats import rankdata

        assert np.array_equal(rankdata(np.round(hot_values, TIE_DECIMALS)),
                              rankdata(np.round(cold_values, TIE_DECIMALS)))

    def test_a_negative_temperature_is_refused_upstream(self, tmp_path):
        chain = build_orientation_chain(tmp_path, self.PROTEIN, name="negT")
        with pytest.raises(ValueError, match="temperature must be positive"):
            run_escott.escott_to_probability(chain["raw"], temperature=-1.0)


# =========================================================================== #
# ALPHA SWEEP -- a planted optimum, found exactly
# =========================================================================== #

@pytest.mark.integration
class TestAlphaSweepAnalytic:
    @pytest.mark.parametrize("alpha_true", [-0.75, -0.25, 0.0, 0.5, 1.0])
    def test_the_sweep_recovers_the_planted_alpha_exactly(self, driver_module, alpha_true):
        frame = planted_alpha_frame(alpha_true, seed=int(abs(alpha_true) * 100) + 1)
        alpha_df, _by_lineage = run_sweep(driver_module, frame, ALPHA_GRID)
        sweep = sweep_rows_only(alpha_df)
        best = sweep.loc[sweep["mut_flat_global_spearman_r"].idxmax()]
        assert float(best["alpha"]) == pytest.approx(alpha_true)
        assert float(best["mut_flat_global_spearman_r"]) == pytest.approx(1.0, abs=1e-12)

    def test_the_sweep_is_unimodal_around_the_planted_alpha(self, driver_module):
        alpha_true = 0.5
        frame = planted_alpha_frame(alpha_true, seed=11)
        alpha_df, _ = run_sweep(driver_module, frame, ALPHA_GRID)
        sweep = sweep_rows_only(alpha_df).sort_values("alpha")
        alphas = sweep["alpha"].to_numpy(dtype=float)
        rho = sweep["mut_flat_global_spearman_r"].to_numpy(dtype=float)
        peak = int(np.argmin(np.abs(alphas - alpha_true)))
        assert np.all(np.diff(rho[: peak + 1]) > 0), "rho must rise up to the planted alpha"
        assert np.all(np.diff(rho[peak:]) < 0), "rho must fall away from the planted alpha"

    def test_the_grid_is_inclusive_of_both_endpoints(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, alpha_start=-1.0, alpha_stop=1.0, alpha_step=0.5)
        grid = driver_module.rma.parse_alpha_grid(args)
        assert grid.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_a_planted_alpha_off_the_grid_lands_on_the_nearest_grid_point(self, driver_module):
        frame = planted_alpha_frame(0.30, seed=5)
        alpha_df, _ = run_sweep(driver_module, frame, ALPHA_GRID)
        sweep = sweep_rows_only(alpha_df)
        best = float(sweep.loc[sweep["mut_flat_global_spearman_r"].idxmax(), "alpha"])
        assert best == pytest.approx(0.25)
        assert float(sweep["mut_flat_global_spearman_r"].max()) < 1.0

    def test_a_by_lineage_sweep_recovers_a_different_alpha_per_lineage(self, driver_module):
        left = planted_alpha_frame(-0.5, seed=2, lineage="A")
        right = planted_alpha_frame(0.75, seed=3, lineage="B")
        frame = pd.concat([left, right], ignore_index=True)
        _alpha_df, by_lineage = run_sweep(driver_module, frame, ALPHA_GRID)
        best_per_lineage = {}
        for lineage, rows in sweep_rows_only(by_lineage).groupby("lineage"):
            best_per_lineage[lineage] = float(
                rows.loc[rows["mut_flat_global_spearman_r"].idxmax(), "alpha"])
        assert best_per_lineage == {"A": -0.5, "B": 0.75}

    def test_the_pooled_row_is_the_mean_across_lineages(self, driver_module):
        left = planted_alpha_frame(0.5, seed=2, lineage="A")
        right = planted_alpha_frame(0.5, seed=3, lineage="B")
        frame = pd.concat([left, right], ignore_index=True)
        alpha_df, by_lineage = run_sweep(driver_module, frame, np.array([0.0, 0.5]))
        pooled = sweep_rows_only(alpha_df).set_index("alpha")["mut_flat_global_spearman_r"]
        per = sweep_rows_only(by_lineage).groupby("alpha")["mut_flat_global_spearman_r"].mean()
        assert pooled.loc[0.5] == pytest.approx(per.loc[0.5])
        assert pooled.loc[0.5] == pytest.approx(1.0, abs=1e-12)
        assert int(sweep_rows_only(alpha_df)["n_lineages_averaged"].iloc[0]) == 2

    def test_the_score_formula_is_relabelled_for_the_sweep_rows_only(self, driver_module):
        frame = planted_alpha_frame(0.5, seed=9)
        alpha_df, by_lineage = run_sweep(driver_module, frame, np.array([0.0, 0.5]))
        stamped = driver_module._stamp_score_formula(alpha_df)
        sweep = stamped.loc[stamped["model_variant"] == "plm_alpha_sweep"]
        baseline = stamped.loc[stamped["model_variant"] == "mutation_accessibility_only"]
        assert set(sweep["input_score_formula"]) == {driver_module.INPUT_SCORE_FORMULA}
        assert set(baseline["input_score_formula"]) == {"mut_prob"}
        assert driver_module.INPUT_SCORE_FORMULA == "escott_prob * mut_prob^alpha"
        assert not driver_module._stamp_score_formula(by_lineage).empty

    def test_stamp_is_a_no_op_on_an_empty_or_column_less_frame(self, driver_module):
        assert driver_module._stamp_score_formula(pd.DataFrame()).empty
        frame = pd.DataFrame({"alpha": [0.0]})
        assert driver_module._stamp_score_formula(frame).equals(frame)

    def test_stamp_falls_back_to_the_formula_column_without_model_variant(self, driver_module):
        frame = pd.DataFrame({"input_score_formula": ["plm_prob * mut_prob^alpha", "mut_prob"]})
        stamped = driver_module._stamp_score_formula(frame)
        assert stamped["input_score_formula"].tolist() == [
            driver_module.INPUT_SCORE_FORMULA, "mut_prob"]

    @staticmethod
    def _baseline_wins_frame():
        """A sweep frame in which the mutation-only baseline beats every alpha.

        ``obs_freq`` is made monotone in ``mut_prob`` ALONE, so the codon model by
        itself scores rho = 1.0 while every alpha on the grid -- each of which mixes
        in an independent, uninformative ``plm_prob`` -- scores less.  This is the
        regime that used to produce ``best_alpha = NaN``.
        """
        rng = np.random.default_rng(7)
        rows: List[Dict[str, object]] = []
        for position in range(1, 13):
            ref_aa = AA20[position % 20]
            for aa in AA20:
                if aa == ref_aa:
                    continue
                log_plm = float(rng.uniform(-4.0, -0.5))
                log_mut = float(rng.uniform(-9.0, -2.0))
                rows.append({"model": "M", "epoch_label": "e", "epoch_value": 0.0,
                             "lineage": "L", "position": position, "ref_aa": ref_aa, "aa": aa,
                             "plm_prob": math.exp(log_plm), "mut_prob": math.exp(log_mut),
                             "_mut": log_mut, "depth": 100.0})
        frame = pd.DataFrame(rows)
        frame["obs_freq"] = frame["_mut"].rank(method="first") / (len(frame) + 1.0)
        frame["obs_present"] = 1
        return frame.drop(columns=["_mut"])

    def test_best_alpha_is_never_nan(self, driver_module):
        """Regression test for a fixed real bug in ``run_analysis``.

        The best-alpha block used to do
        ``alpha_df['mut_flat_global_spearman_r'].idxmax()`` over the WHOLE table,
        which includes the mutation-only baseline row -- and that row carries
        ``alpha = NaN``.  Whenever the codon model alone out-ranked every alpha on
        the grid (entirely reachable: the baseline is the ``alpha -> +inf`` limit
        with ``plm_prob`` pinned at 1.0), ``best_alpha_two_methods.tsv`` got
        ``best_alpha = NaN``, which reads as a failed fit rather than "the baseline
        won".  Fixed by routing both selections through ``best_alpha_index``, which
        excludes the baseline row.

        This calls the driver's own helper, not a copy of it, so it fails if the
        selection ever regresses.
        """
        alpha_df, _ = run_sweep(driver_module, self._baseline_wins_frame(), ALPHA_GRID)

        # Precondition: the bug's trigger really is present in this frame.
        raw_idx = alpha_df["mut_flat_global_spearman_r"].idxmax()
        assert bool(alpha_df.loc[raw_idx, "is_mutation_only_baseline"]), (
            "this frame is supposed to be one where the baseline out-ranks the grid"
        )
        assert not np.isfinite(float(alpha_df.loc[raw_idx, "alpha"]))

        idx = driver_module.best_alpha_index(alpha_df, "mut_flat_global_spearman_r")
        best_alpha = float(alpha_df.loc[idx, "alpha"])
        assert np.isfinite(best_alpha), (
            "best_alpha must be a real alpha from the grid, not the baseline row's NaN"
        )
        assert best_alpha in ALPHA_GRID

    def test_best_alpha_index_ignores_the_baseline_for_both_criteria(self, driver_module):
        """Both selection columns, not just the one the bug was found through."""
        alpha_df, _ = run_sweep(driver_module, self._baseline_wins_frame(), ALPHA_GRID)
        for column in ("site_top10pct_mutated_enrichment", "mut_flat_global_spearman_r"):
            idx = driver_module.best_alpha_index(alpha_df, column)
            assert idx is not None
            assert not bool(alpha_df.loc[idx, "is_mutation_only_baseline"])
            assert np.isfinite(float(alpha_df.loc[idx, "alpha"]))

    def test_best_alpha_index_still_picks_the_true_maximum_of_the_grid(self, driver_module):
        """Excluding the baseline must not change which GRID row wins."""
        alpha_df, _ = run_sweep(driver_module, planted_alpha_frame(0.5), ALPHA_GRID)
        grid = alpha_df[~alpha_df["is_mutation_only_baseline"].astype(bool)]
        idx = driver_module.best_alpha_index(alpha_df, "mut_flat_global_spearman_r")
        assert idx == grid["mut_flat_global_spearman_r"].idxmax()
        assert float(alpha_df.loc[idx, "alpha"]) == pytest.approx(0.5)

    def test_a_frame_of_nothing_but_the_baseline_yields_no_best_alpha_row(self, driver_module):
        """The caller skips the row entirely rather than emitting a NaN alpha."""
        frame = pd.DataFrame({
            "alpha": [np.nan],
            "is_mutation_only_baseline": [True],
            "mut_flat_global_spearman_r": [0.9],
        })
        assert driver_module.best_alpha_index(frame, "mut_flat_global_spearman_r") is None
        assert driver_module.alpha_sweep_grid_rows(frame).empty

    def test_a_frame_without_the_marker_column_is_left_alone(self, driver_module):
        """Back-compatible on hand-made or older tables that lack the marker."""
        frame = pd.DataFrame({"alpha": [0.0, 0.5], "mut_flat_global_spearman_r": [0.1, 0.7]})
        assert len(driver_module.alpha_sweep_grid_rows(frame)) == 2
        assert driver_module.best_alpha_index(frame, "mut_flat_global_spearman_r") == 1

    def test_an_all_nan_metric_column_yields_no_best_alpha_row(self, driver_module):
        """idxmax on an all-NaN column raises; the helper returns None instead."""
        frame = pd.DataFrame({
            "alpha": [0.0, 0.5],
            "is_mutation_only_baseline": [False, False],
            "mut_flat_global_spearman_r": [np.nan, np.nan],
        })
        assert driver_module.best_alpha_index(frame, "mut_flat_global_spearman_r") is None
        assert driver_module.best_alpha_index(frame, "no_such_column") is None


# =========================================================================== #
# COORDINATE SYSTEMS -- five artefacts, one numbering
# =========================================================================== #

@pytest.mark.integration
class TestCoordinateConsistency:
    PROTEIN = C.QUERY_PROTEIN

    def test_escott_column_labels_are_one_based_with_no_offset(self, tmp_path, fake_escott_matrix):
        matrix = run_escott.read_escott_matrix(fake_escott_matrix["path"],
                                               expect_protein=self.PROTEIN)
        assert list(matrix.columns) == list(range(1, len(self.PROTEIN) + 1))
        assert run_escott.escott_wt_sequence(matrix) == self.PROTEIN
        for position in (1, len(self.PROTEIN)):
            assert np.isnan(matrix.at[self.PROTEIN[position - 1], position])

    def test_jet_res_rows_are_one_based_and_name_the_same_residues(self, fake_jet_res):
        table = pd.read_table(fake_jet_res["path"], sep=r"\s+")
        assert table["pos"].tolist() == list(range(1, len(self.PROTEIN) + 1))
        three_letter = {v: k for k, v in C.THREE_LETTER.items()}
        recovered = "".join(three_letter[name] for name in table["AA"])
        assert recovered == self.PROTEIN

    def test_pdb_residue_numbers_match_the_jet_rows(self, prepared_inputs_tree, fake_jet_res):
        text = Path(prepared_inputs_tree["monomer_pdb"]).read_text(encoding="utf-8")
        resnums = sorted({int(line[22:26]) for line in text.splitlines()
                          if line.startswith("ATOM") and line[21] == "A"})
        table = pd.read_table(fake_jet_res["path"], sep=r"\s+")
        assert resnums == table["pos"].tolist()

    def test_the_score_matrix_sequence_row_is_the_query_protein(self, score_matrix_factory):
        from Functions_HuggingFace import load_plm_probability_matrix

        raw = load_plm_probability_matrix(str(score_matrix_factory(self.PROTEIN)))
        assert raw.index[0] == "sequence"
        recovered = "".join(str(value) for value in raw.iloc[0, :].tolist())
        assert recovered == self.PROTEIN

    def test_combined_rows_are_one_based_and_carry_the_right_wild_type(
            self, driver_module, tmp_path, score_matrix_factory):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(self.PROTEIN).read_bytes())
        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(self.PROTEIN)
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert frame["position"].min() == 1
        assert frame["position"].max() == len(self.PROTEIN)
        for row in frame.itertuples():
            assert row.ref_aa == self.PROTEIN[row.position - 1]
            assert row.aa != row.ref_aa
        assert len(frame) == 19 * len(self.PROTEIN)

    def test_a_shifted_coord_map_shifts_the_reported_positions_by_exactly_that_much(
            self, driver_module, tmp_path, score_matrix_factory):
        """The 16-aa signal peptide case: the matrix starts at the mature N-terminus."""
        offset = 16
        mature = self.PROTEIN[offset:]
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(mature).read_bytes())
        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(
            self.PROTEIN,
            coord_map={j: j + offset for j in range(len(mature))},
        )
        matrix, _path, sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        assert sequence == mature
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert frame["position"].min() == offset + 1
        assert frame["position"].max() == len(self.PROTEIN)
        for row in frame.itertuples():
            assert row.ref_aa == self.PROTEIN[row.position - 1]

    def test_positions_absent_from_the_mutation_profile_are_skipped(
            self, driver_module, tmp_path, score_matrix_factory):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(self.PROTEIN).read_bytes())
        args = make_args(driver_module, tmp_path)
        kept = list(range(17, len(self.PROTEIN) + 1))
        lineage_data = make_lineage_data(self.PROTEIN, positions=kept)
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert sorted(frame["position"].unique()) == kept

    def test_matrix_columns_outside_the_coord_map_are_skipped(
            self, driver_module, tmp_path, score_matrix_factory):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(self.PROTEIN).read_bytes())
        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(self.PROTEIN, coord_map={0: 0, 1: 1, 2: 2})
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert sorted(frame["position"].unique()) == [1, 2, 3]

    def test_all_five_artefacts_agree_at_every_position(
            self, driver_module, tmp_path, fake_escott_matrix, fake_jet_res):
        """query protein <-> jet.res <-> ESCOTT columns <-> score matrix <-> combined rows."""
        protein = self.PROTEIN
        escott = run_escott.read_escott_matrix(fake_escott_matrix["path"],
                                               expect_protein=protein)
        probabilities = run_escott.escott_to_probability(escott)
        scores_dir = tmp_path / "scores"
        run_escott.write_score_matrix(probabilities, protein,
                                      scores_dir / "K_ESCOTT_score_matrix.csv")
        jet = pd.read_table(fake_jet_res["path"], sep=r"\s+")
        three_letter = {v: k for k, v in C.THREE_LETTER.items()}

        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(protein)
        matrix, _path, sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        wt_by_position = dict(zip(frame["position"], frame["ref_aa"]))

        assert sequence == protein
        for position in range(1, len(protein) + 1):
            wt = protein[position - 1]
            assert three_letter[jet.loc[position - 1, "AA"]] == wt, f"jet.res row {position}"
            assert int(jet.loc[position - 1, "pos"]) == position
            assert np.isnan(escott.at[wt, position]), f"ESCOTT NA row at {position}"
            assert wt_by_position[position] == wt, f"combined row at {position}"

    def test_the_wild_type_residue_never_appears_as_a_mutant(
            self, driver_module, tmp_path, score_matrix_factory):
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(self.PROTEIN).read_bytes())
        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(self.PROTEIN)
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert not (frame["aa"] == frame["ref_aa"]).any()


# =========================================================================== #
# Degenerate inputs
# =========================================================================== #

class TestDegenerateMetrics:
    def test_spearman_of_a_constant_vector_is_nan_not_a_crash(self, driver_module):
        constant = pd.Series([3.0] * 10)
        varying = pd.Series(np.arange(10.0))
        assert math.isnan(driver_module.rma.safe_spearman(constant, varying))
        assert math.isnan(driver_module.rma.safe_spearman(varying, constant))

    def test_spearman_needs_two_finite_pairs(self, driver_module):
        assert math.isnan(driver_module.rma.safe_spearman(
            pd.Series([1.0, np.nan]), pd.Series([np.nan, 2.0])))

    def test_spearman_of_a_monotone_pair_is_exactly_one(self, driver_module):
        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert driver_module.rma.safe_spearman(x, x ** 3) == pytest.approx(1.0)

    def test_pearson_of_a_constant_vector_is_nan(self, driver_module):
        assert math.isnan(driver_module.rma.safe_pearson(
            pd.Series([1.0] * 5), pd.Series(np.arange(5.0))))

    def test_auroc_with_a_single_class_is_nan(self, driver_module):
        scores = pd.Series(np.arange(10.0))
        assert math.isnan(driver_module.rma.safe_auroc(scores, pd.Series([1] * 10)))
        assert math.isnan(driver_module.rma.safe_auroc(scores, pd.Series([0] * 10)))

    def test_auroc_of_a_perfect_and_an_inverted_ranking(self, driver_module):
        scores = pd.Series([0.0, 1.0, 2.0, 3.0])
        outcome = pd.Series([0, 0, 1, 1])
        assert driver_module.rma.safe_auroc(scores, outcome) == pytest.approx(1.0)
        assert driver_module.rma.safe_auroc(-scores, outcome) == pytest.approx(0.0)

    def test_pr_auc_with_a_single_class_is_nan(self, driver_module):
        assert math.isnan(driver_module.rma.safe_pr_auc(
            pd.Series(np.arange(4.0)), pd.Series([1, 1, 1, 1])))

    def test_the_sweeps_correlation_guard_returns_a_nan_pair(self):
        from scipy.stats import spearmanr

        from Functions_HuggingFace import _safe_corr_pair

        r, p = _safe_corr_pair(spearmanr, [1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert math.isnan(r) and math.isnan(p)
        r, p = _safe_corr_pair(spearmanr, [1.0], [2.0])
        assert math.isnan(r) and math.isnan(p)

    def test_a_completely_flat_score_matrix_yields_nan_not_a_crash(self, driver_module):
        """Every column zero-trace: the softmax is uniform, so there is no rank signal."""
        rows: List[Dict[str, object]] = []
        for position in range(1, 6):
            ref_aa = AA20[position]
            for aa in AA20:
                if aa == ref_aa:
                    continue
                rows.append({"model": "M", "epoch_label": "e", "epoch_value": 0.0,
                             "lineage": "L", "position": position, "ref_aa": ref_aa, "aa": aa,
                             "plm_prob": 0.05, "mut_prob": 0.01,
                             "obs_freq": 0.0, "obs_present": 0, "depth": 100.0})
        frame = pd.DataFrame(rows)
        alpha_df, _ = run_sweep(driver_module, frame, np.array([0.0, 1.0]))
        sweep = sweep_rows_only(alpha_df)
        assert len(sweep) == 2
        assert sweep["mut_flat_global_spearman_r"].isna().all()

    def test_a_zero_depth_site_is_reported_with_depth_zero(
            self, driver_module, tmp_path, score_matrix_factory):
        protein = "MKTII"
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(protein).read_bytes())
        args = make_args(driver_module, tmp_path)
        lineage_data = make_lineage_data(protein, depth={1: 0, 2: 0, 3: 50, 4: 50, 5: 50})
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert set(frame.loc[frame["position"] <= 2, "depth"]) == {0.0}
        assert set(frame.loc[frame["position"] >= 3, "depth"]) == {50.0}
        assert len(frame) == 19 * len(protein)

    def test_singleton_filtering_zeroes_a_zero_depth_site(
            self, driver_module, tmp_path, score_matrix_factory):
        protein = "MKTII"
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(protein).read_bytes())
        args = make_args(driver_module, tmp_path, filter_singleton_mutations=True,
                         min_obs_count=2, skip_low_count_sites=False)
        lineage_data = make_lineage_data(protein, depth=0, obs={("A", 1): 0.5})
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        assert len(frame) == 19 * len(protein)
        assert (frame["obs_freq"] == 0.0).all()
        assert (frame["obs_present"] == 0).all()

    def test_skip_low_count_sites_drops_the_rows_entirely(
            self, driver_module, tmp_path, score_matrix_factory):
        protein = "MKTII"
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(protein).read_bytes())
        args = make_args(driver_module, tmp_path, filter_singleton_mutations=True,
                         min_obs_count=2, skip_low_count_sites=True)
        lineage_data = make_lineage_data(protein, depth=0)
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        rows = driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"])
        assert rows == []

    def test_fixed_mutations_are_filtered_by_default(
            self, driver_module, tmp_path, score_matrix_factory):
        protein = "MKTII"
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        (scores_dir / "K_ESCOTT_score_matrix.csv").write_bytes(
            score_matrix_factory(protein).read_bytes())
        args = make_args(driver_module, tmp_path)
        assert args.filter_fixed_mutations is True
        lineage_data = make_lineage_data(protein, obs={("A", 1): 1.0, ("C", 1): 0.4})
        matrix, _path, _sequence = driver_module.ensure_score_matrix(
            args, make_spec("ESCOTT"), "K", lineage_data, scores_dir)
        frame = pd.DataFrame(driver_module.rma.build_combined_rows(
            args, make_spec("ESCOTT"), "K", lineage_data, matrix,
            coord_map=lineage_data["coord_map"]))
        at_one = frame.loc[(frame["position"] == 1) & (frame["aa"] == "A")]
        assert at_one.empty
        assert len(frame) == 19 * len(protein) - 1

    def test_constant_scores_give_nan_epoch_metrics_without_raising(self, driver_module):
        frame = pd.DataFrame({
            "model": "M", "epoch_label": "e", "epoch_value": 0.0, "lineage": "L",
            "position": [1, 1, 2, 2], "ref_aa": ["M", "M", "K", "K"], "aa": ["A", "C", "A", "C"],
            "plm_prob": [0.05] * 4, "mut_prob": [0.01] * 4,
            "obs_freq": [0.0, 0.1, 0.2, 0.3], "obs_present": [0, 1, 1, 1], "depth": [10.0] * 4,
        })
        metrics = driver_module.rma.compute_epoch_lineage_metrics(frame)
        assert len(metrics) == 1
        assert math.isnan(float(metrics.loc[0, "spearman_obs_freq_vs_plm"]))
        assert float(metrics.loc[0, "spearman_mut_vs_mut_baseline"]) == 1.0

    def test_empty_combined_frames_produce_empty_tables(self, driver_module):
        assert driver_module.rma.compute_epoch_lineage_metrics(pd.DataFrame()).empty
        assert driver_module.rma.summarize_epoch_metrics(pd.DataFrame()).empty


# =========================================================================== #
# Score-scale diagnostics
# =========================================================================== #

class TestScoreScaleReport:
    def test_an_empty_frame_writes_nothing(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(pd.DataFrame(), tmp_path, args)
        assert not (tmp_path / "diagnostics" / "score_scale_report.tsv").exists()

    def test_the_spreads_are_the_standard_deviations_of_the_logs(self, driver_module, tmp_path):
        plm = np.array([0.01, 0.1, 0.2, 0.5])
        mut = np.array([1e-4, 1e-3, 1e-2, 1e-1])
        frame = pd.DataFrame({
            "model": "M", "lineage": "K", "position": [1, 1, 2, 2],
            "plm_prob": plm, "mut_prob": mut,
        })
        args = make_args(driver_module, tmp_path, escott_temperature=1.0)
        driver_module.write_score_scale_report(frame, tmp_path, args)
        report = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert len(report) == 1
        row = report.iloc[0]
        assert row["n_rows"] == 4
        assert row["sd_log_score"] == pytest.approx(float(np.std(np.log(plm))))
        assert row["sd_log_mut"] == pytest.approx(float(np.std(np.log(mut))))
        assert row["ratio_sd"] == pytest.approx(
            float(np.std(np.log(plm)) / np.std(np.log(mut))))
        assert math.isnan(row["sd_log_plm_reference"])
        assert math.isnan(row["alpha_rescale"])

    def test_uniform_columns_are_counted_as_flat_sites(self, driver_module, tmp_path):
        """A zero-trace ESCOTT column softmaxes to a constant 1/20 -- a dead site."""
        rows = []
        for position in (1, 2, 3):
            for index, aa in enumerate("ACDE"):
                probability = 0.05 if position != 2 else 0.05
                if position == 3:
                    probability = 0.05 + 0.001 * index
                rows.append({"model": "M", "lineage": "K", "position": position, "aa": aa,
                             "plm_prob": probability, "mut_prob": 0.01})
        args = make_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(pd.DataFrame(rows), tmp_path, args)
        report = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert int(report.iloc[0]["n_flat_sites"]) == 2  # positions 1 and 2, never 3

    def test_no_position_column_means_no_flat_site_count(self, driver_module, tmp_path):
        frame = pd.DataFrame({"model": "M", "lineage": "K",
                              "plm_prob": [0.1, 0.2], "mut_prob": [0.01, 0.02]})
        args = make_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(frame, tmp_path, args)
        report = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert int(report.iloc[0]["n_flat_sites"]) == 0

    def test_the_reference_table_supplies_the_alpha_rescale(self, driver_module, tmp_path):
        reference_values = np.array([0.001, 0.01, 0.1, 0.5])
        reference = tmp_path / "plm_reference.csv"
        pd.DataFrame({"plm_prob": reference_values, "other": 1}).to_csv(reference, index=False)
        plm = np.array([0.01, 0.1, 0.2, 0.5])
        frame = pd.DataFrame({"model": "M", "lineage": "K", "position": [1, 1, 2, 2],
                              "plm_prob": plm, "mut_prob": [1e-4, 1e-3, 1e-2, 1e-1]})
        args = make_args(driver_module, tmp_path, plm_reference_table=reference)
        driver_module.write_score_scale_report(frame, tmp_path, args)
        report = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        reference_sd = float(np.std(np.log(reference_values)))
        assert report.iloc[0]["sd_log_plm_reference"] == pytest.approx(reference_sd)
        assert report.iloc[0]["alpha_rescale"] == pytest.approx(
            reference_sd / float(np.std(np.log(plm))))

    def test_zero_probabilities_are_clipped_rather_than_producing_minus_inf(
            self, driver_module, tmp_path):
        frame = pd.DataFrame({"model": "M", "lineage": "K", "position": [1, 1],
                              "plm_prob": [0.0, 0.5], "mut_prob": [0.0, 0.5]})
        args = make_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(frame, tmp_path, args)
        report = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert np.isfinite(report.iloc[0]["sd_log_score"])
        assert report.iloc[0]["sd_log_score"] == pytest.approx(
            float(np.std([math.log(1e-32), math.log(0.5)])))


# =========================================================================== #
# JET-surrogate diagnostics
# =========================================================================== #

class TestJetSurrogateSummary:
    def _manifest(self, **overrides):
        base = {
            "msa_n_sequences": 6434, "msa_n_columns": 566, "weight_mode": "structural",
            "trace_definition": "bootstrap", "trace_bootstraps": 50,
            "trace_top_fraction": 0.90, "n_zero_trace_columns": 18,
            "frac_zero_trace_columns": 18 / 566,
            "n_positions_without_structure": 0,
            "structure": {"pdb": "/x/mono.pdb", "context_pdb": "/x/tri.pdb",
                          "covered": 566, "structure_query_identity": 0.85},
            "jet_res_path": "/x/K_surrogate_jet.res", "jet_res_md5": "deadbeef",
        }
        base.update(overrides)
        return base

    def test_manifests_are_read_from_the_stage1_layout(self, driver_module, prepared_inputs_tree):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        payload = self._manifest()
        (inputs_dir / "jet" / "K_jet_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8")
        found = driver_module.read_jet_manifests(
            inputs_dir, prepared_inputs_tree["manifest"], ["K", "J.2.4"], "primary")
        assert list(found) == ["K"]
        assert found["K"]["n_zero_trace_columns"] == 18

    def test_a_corrupt_manifest_is_skipped_not_fatal(self, driver_module, prepared_inputs_tree):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        (inputs_dir / "jet" / "K_jet_manifest.json").write_text("{not json", encoding="utf-8")
        assert driver_module.read_jet_manifests(
            inputs_dir, prepared_inputs_tree["manifest"], ["K"], "primary") == {}

    def test_no_manifests_means_no_summary_file(self, driver_module, tmp_path):
        assert driver_module.write_jet_surrogate_summary({}, tmp_path) is None
        assert not (tmp_path / "jet_surrogate_summary.tsv").exists()

    def test_the_summary_row_carries_the_zero_trace_count(self, driver_module, tmp_path):
        path = driver_module.write_jet_surrogate_summary({"K": self._manifest()}, tmp_path)
        table = pd.read_csv(path, sep="\t")
        assert len(table) == 1
        row = table.iloc[0]
        assert row["lineage"] == "K"
        assert int(row["n_zero_trace_columns"]) == 18
        assert row["trace_top_fraction"] == pytest.approx(0.90)
        assert row["structure_pdb"] == "/x/mono.pdb"
        assert row["structure_context_pdb"] == "/x/tri.pdb"

    def test_a_zero_trace_fraction_above_five_percent_warns_loudly(
            self, driver_module, tmp_path, capsys):
        manifest = self._manifest(n_zero_trace_columns=60, frac_zero_trace_columns=60 / 566)
        driver_module.write_jet_surrogate_summary({"K": manifest}, tmp_path)
        out = capsys.readouterr().out
        assert "WARNING" in out and "trace == 0" in out
        assert "10.6%" in out

    def test_a_low_zero_trace_fraction_is_silent(self, driver_module, tmp_path, capsys):
        manifest = self._manifest(n_zero_trace_columns=5, frac_zero_trace_columns=5 / 566)
        driver_module.write_jet_surrogate_summary({"K": manifest}, tmp_path)
        assert "WARNING" not in capsys.readouterr().out

    def test_the_warning_threshold_is_the_shared_constant(self):
        assert constants.WARN_ZERO_TRACE_FRACTION == 0.05
        assert constants.MAX_ZERO_TRACE_FRACTION == 0.10


# =========================================================================== #
# Temperature resolution
# =========================================================================== #

class TestEscottTemperature:
    def test_fixed_mode_returns_the_flag_value(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, escott_temperature=2.5,
                         escott_temperature_mode="fixed")
        assert driver_module.resolve_escott_temperature(args, tmp_path, ["K"]) == 2.5

    def test_match_plm_makes_sd_log_plm_prob_equal_the_reference(self, driver_module, tmp_path):
        """The DEFINITION of the mode, not the formula that used to implement it.

        This test previously asserted ``T == sd(E) / sd(log plm_ref)``, which is the
        closed form the driver used -- and it was wrong: ``plm_prob`` is a per-column
        softmax, so ``log P = E/T - logsumexp_col(E/T)`` keeps only the within-column
        variance while ``sd(E)`` is the total.  Asserting the formula could therefore
        never have caught the defect.  See
        ``test_regressions_numerics.TestMatchPlmTemperatureCalibration``.
        """
        reference_values = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
        reference = tmp_path / "reference.csv"
        pd.DataFrame({"plm_prob": reference_values}).to_csv(reference, index=False)
        raw_values = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        pd.DataFrame(raw_values, index=["A", "C"], columns=[1, 2]).to_csv(
            scores_dir / "K_ESCOTT_raw.tsv", sep="\t")
        args = make_args(driver_module, tmp_path, escott_temperature_mode="match-plm",
                         plm_reference_table=reference)
        target = float(np.nanstd(np.log(reference_values)))

        temperature = driver_module.resolve_escott_temperature(args, scores_dir, ["K"])
        assert driver_module.sd_log_softmax(raw_values, temperature) == pytest.approx(
            target, rel=1e-9)
        # And the discarded closed form does NOT satisfy that, which is the whole point.
        assert driver_module.sd_log_softmax(
            raw_values, float(np.nanstd(raw_values)) / target) < 0.95 * target

    def test_match_plm_takes_the_median_over_lineages(self, driver_module, tmp_path):
        """One scalar T per run: the median of the per-lineage solutions."""
        reference_values = np.array([0.01, 0.1, 0.5])
        reference = tmp_path / "reference.csv"
        pd.DataFrame({"plm_prob": reference_values}).to_csv(reference, index=False)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        target = float(np.nanstd(np.log(reference_values)))
        per_lineage = []
        for key, scale in (("K", 1.0), ("J_int", 2.0), ("J.2.4", 3.0)):
            values = np.array([[-1.0, -2.0], [-3.0, -4.0]]) * scale
            pd.DataFrame(values, index=["A", "C"], columns=[1, 2]).to_csv(
                scores_dir / f"{driver_module.safe_key(key)}_ESCOTT_raw.tsv", sep="\t")
            per_lineage.append(driver_module.solve_softmax_temperature(values, target))
        args = make_args(driver_module, tmp_path, escott_temperature_mode="match-plm",
                         plm_reference_table=reference)
        assert driver_module.resolve_escott_temperature(
            args, scores_dir, ["K", "J_int", "J.2.4"]) == pytest.approx(
                float(np.median(per_lineage)))

    def test_a_missing_reference_table_is_a_hard_error(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, escott_temperature_mode="match-plm",
                         plm_reference_table=tmp_path / "absent.csv")
        with pytest.raises(FileNotFoundError, match="--plm-reference-table not found"):
            driver_module.resolve_escott_temperature(args, tmp_path, ["K"])

    def test_a_constant_reference_cannot_be_matched(self, driver_module, tmp_path):
        reference = tmp_path / "reference.csv"
        pd.DataFrame({"plm_prob": [0.1, 0.1, 0.1]}).to_csv(reference, index=False)
        args = make_args(driver_module, tmp_path, escott_temperature_mode="match-plm",
                         plm_reference_table=reference)
        with pytest.raises(ValueError, match="non-positive sd"):
            driver_module.resolve_escott_temperature(args, tmp_path, ["K"])

    def test_match_plm_explains_that_it_needs_a_first_pass(self, driver_module, tmp_path):
        reference = tmp_path / "reference.csv"
        pd.DataFrame({"plm_prob": [0.01, 0.1, 0.5]}).to_csv(reference, index=False)
        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        args = make_args(driver_module, tmp_path, escott_temperature_mode="match-plm",
                         plm_reference_table=reference)
        with pytest.raises(FileNotFoundError, match="Run once with"):
            driver_module.resolve_escott_temperature(args, scores_dir, ["K"])


# =========================================================================== #
# Structure and stage-1 path resolution
# =========================================================================== #

class TestStructureResolution:
    def test_an_empty_manifest_yields_an_empty_entry(self, driver_module):
        assert driver_module.resolve_structure_entry({}, "primary") == {}
        assert driver_module.resolve_structure_entry({"structures": {}}, "primary") == {}

    def test_a_missing_role_is_refused_and_lists_what_exists(self, driver_module):
        manifest = {"structures": {"primary": {"monomer": {"path": "/x.pdb"}}}}
        with pytest.raises(RuntimeError) as excinfo:
            driver_module.resolve_structure_entry(manifest, "extra")
        message = str(excinfo.value)
        assert "'extra' is not present" in message
        assert "available: ['primary']" in message
        assert "--no-extra-structure" in message

    def test_a_present_role_is_returned_verbatim(self, driver_module, prepared_inputs_tree):
        entry = driver_module.resolve_structure_entry(
            prepared_inputs_tree["manifest"], "primary")
        assert entry["coverage_fraction"] == 1.0
        assert Path(entry["monomer"]["path"]).exists()

    def test_stage1_paths_prefer_the_manifest(self, driver_module, prepared_inputs_tree):
        paths = driver_module.stage1_paths(
            prepared_inputs_tree["inputs_dir"], "K",
            prepared_inputs_tree["manifest"], "primary")
        assert paths["msa"].exists() and paths["msa"].name == "msa_K.fasta"
        assert paths["query"].exists()
        assert paths["jet"].name == "K_surrogate_jet.res"
        assert paths["jet"].exists()
        assert paths["chain_pdb"] == Path(prepared_inputs_tree["monomer_pdb"])
        assert paths["trimer_pdb"] == Path(prepared_inputs_tree["trimer_pdb"])

    def test_stage1_paths_fall_back_before_prepare_inputs_has_run(self, driver_module, tmp_path):
        paths = driver_module.stage1_paths(tmp_path, "K", {}, "primary")
        assert paths["msa"] == tmp_path / "msa" / "msa_K.fasta"
        assert paths["query"] == tmp_path / "query" / "K_query.fasta"
        assert paths["chain_pdb"] == tmp_path / "structure" / "6WXB_chainA_qnum.pdb"
        assert paths["jet_manifest"] == tmp_path / "jet" / "K_jet_manifest.json"

    def test_read_inputs_manifest_tolerates_absence(self, driver_module, tmp_path):
        assert driver_module.read_inputs_manifest(tmp_path) == {}

    def test_the_structure_record_names_what_the_surrogate_read(
            self, driver_module, prepared_inputs_tree, tmp_path):
        args = make_args(driver_module, tmp_path, structure_role="primary",
                         inputs_dir=prepared_inputs_tree["inputs_dir"])
        record = driver_module.resolve_structure_record(args, prepared_inputs_tree["manifest"])
        assert record["structure_resolved_from_inputs_manifest"] is True
        assert record["structure_monomer_path"] == str(prepared_inputs_tree["monomer_pdb"])
        assert record["structure_monomer_md5"] == C.md5_file(prepared_inputs_tree["monomer_pdb"])
        assert record["structure_n_covered"] == C.QUERY_LENGTH
        assert record["structure_coverage_fraction"] == 1.0

    def test_without_a_manifest_the_record_says_so(self, driver_module, tmp_path):
        structure = tmp_path / "6WXB.cif"
        structure.write_text("data_x\n", encoding="utf-8")
        args = make_args(driver_module, tmp_path, structure=structure)
        record = driver_module.resolve_structure_record(args, {})
        assert record["structure_resolved_from_inputs_manifest"] is False
        assert record["structure_source_path"] == str(structure)
        assert record["structure_source_md5"] == C.md5_file(structure)
        assert record["structure_n_covered"] is None


# =========================================================================== #
# Leakage record
# =========================================================================== #

class TestLeakageManifestRecord:
    def test_no_stage1_block_says_unaudited(self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path)
        record = driver_module.leakage_manifest_record(args, {})
        assert record["leakage_stage_ran"] is False
        assert record["leakage_status"] is None
        assert "UNAUDITED" in str(record["leakage_note"])
        assert record["leakage_check_requested"] is True  # the CLI default is on

    def test_the_requested_flags_are_recorded_even_when_stage1_did_not_run(
            self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, leakage_check=False,
                         purge_leakage=False, fail_on_leakage=True)
        record = driver_module.leakage_manifest_record(args, {})
        assert record["leakage_check_requested"] is False
        assert record["purge_leakage_requested"] is False
        assert record["fail_on_leakage"] is True

    def test_a_stage1_block_is_summarised_per_target(self, driver_module, tmp_path):
        block = {
            "status": "PASS",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True,
            "report_dir": str(tmp_path),
            "purges": {"K": {"depth_before": 6434, "n_removed": 0, "depth_after": 6434,
                             "removed_fraction": 0.0, "n_removed_exact_full_length": 0,
                             "removed_identity_distribution": {"max": 94.0},
                             "removed_hamming_min": None, "query_exempted": True,
                             "drop_manifest_path": "/x/drop.tsv",
                             "prepurge_path": "/x/pre.fasta"}},
            "checks": {"B_parent_vs_target": {"K": {"parent": "J.2.4", "n_flagged": 8,
                                                    "accessions": {"n_shared_accessions": 0,
                                                                   "n_shared_exact_sequences": 0}}}},
        }
        args = make_args(driver_module, tmp_path)
        record = driver_module.leakage_manifest_record(args, {"leakage": block})
        assert record["leakage_stage_ran"] is True
        assert record["leakage_status"] == "PASS"
        assert record["leakage_purge_applied"] is True
        assert record["leakage_per_target"]["K"]["depth_before"] == 6434
        assert record["leakage_per_target"]["K"]["query_would_have_been_purged"] is True
        assert record["leakage_parent_vs_target"]["K"]["parent"] == "J.2.4"
        assert record["leakage_parent_vs_target"]["K"]["n_flagged"] == 8


# =========================================================================== #
# Design signatures and keys
# =========================================================================== #

class TestDesignKeys:
    @pytest.fixture()
    def spec_and_map(self, driver_module):
        parent_map = dict(C.EXPECTED_PARENT_MAP)
        spec = make_spec("PRESCOTT_eq2_c0p50_k1", lineages=["K"],
                         parent_by_lineage={"K": "J.2.4"},
                         source_variant_by_lineage={"K": "PRESCOTT_eq2_c0p50_k1_parentJ24"},
                         equation=2, coefficient=0.5, frequency_cutoff_k=1,
                         epoch_label="prescott_c0.50", epoch_value=0.5)
        return spec, parent_map

    def test_the_key_is_a_deterministic_16_hex_digest(self, driver_module, tmp_path, spec_and_map):
        spec, parent_map = spec_and_map
        args = make_args(driver_module, tmp_path)
        first = driver_module.model_design_key(args, spec, parent_map)
        second = driver_module.model_design_key(args, spec, parent_map)
        assert first == second
        assert len(first) == 16 and set(first) <= set("0123456789abcdef")

    def test_the_coefficient_grid_does_not_invalidate_a_models_key(
            self, driver_module, tmp_path, spec_and_map):
        """Changing --coefficient-grid must not recompute ESCOTT's alpha sweep."""
        spec, parent_map = spec_and_map
        narrow = make_args(driver_module, tmp_path, coefficient_grid="0.5")
        wide = make_args(driver_module, tmp_path, coefficient_grid="0.25,0.5,1.0")
        assert driver_module.model_design_key(narrow, spec, parent_map) == \
            driver_module.model_design_key(wide, spec, parent_map)

    @pytest.mark.parametrize("field,value", [
        ("escott_temperature", 2.0),
        ("mutation_model", "SC2"),
        ("frequency_cutoff_mode", "fixed"),
        ("parent_freq_max", 0.5),
        ("drop_parent_reversions", False),
        ("filter_fixed_mutations", False),
        ("min_obs_count", 5),
        ("alpha_step", 0.5),
        ("test_max_records", 400),
    ])
    def test_terms_that_reach_the_numbers_do_invalidate_the_key(
            self, driver_module, tmp_path, spec_and_map, field, value):
        spec, parent_map = spec_and_map
        base = make_args(driver_module, tmp_path)
        changed = make_args(driver_module, tmp_path, **{field: value})
        assert driver_module.model_design_key(base, spec, parent_map) != \
            driver_module.model_design_key(changed, spec, parent_map)

    def test_a_different_parent_changes_the_key(self, driver_module, tmp_path, spec_and_map):
        spec, parent_map = spec_and_map
        args = make_args(driver_module, tmp_path)
        alternate = dict(spec, parent_by_lineage={"K": "J.2_int"})
        assert driver_module.model_design_key(args, spec, parent_map) != \
            driver_module.model_design_key(args, alternate, parent_map)

    def test_the_run_signature_is_a_superset_of_the_shared_one(
            self, driver_module, tmp_path, spec_and_map):
        _spec, parent_map = spec_and_map
        args = make_args(driver_module, tmp_path)
        shared = driver_module.shared_design_signature(args)
        whole = driver_module.design_signature(args, parent_map, ["K"])
        assert set(shared).issubset(set(whole))
        assert whole["parent_map"] == {"K": "J.2.4"}
        assert whole["prescott_equations"] == [2]
        assert shared["cache_version"] == driver_module.PRESCOTT_CACHE_VERSION

    def test_the_run_signature_records_the_applicable_sensitivity_edges(
            self, driver_module, tmp_path):
        args = make_args(driver_module, tmp_path, parent_sensitivity=True)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.design_signature(
            args, parent_map, ["K"])["parent_sensitivity_edges"] == {"K": "J.2_int"}
        assert driver_module.design_signature(
            args, parent_map, ["J_int"])["parent_sensitivity_edges"] == {}

    def test_design_key_is_stable_under_dict_ordering(self, driver_module):
        left = {"a": 1, "b": {"x": 1, "y": 2}}
        right = {"b": {"y": 2, "x": 1}, "a": 1}
        assert driver_module.design_key(left) == driver_module.design_key(right)


# =========================================================================== #
# Cache guards
# =========================================================================== #

def write_cached_model_tables(model_tables_dir: Path, model_label: str) -> None:
    model_tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"model": [model_label], "lineage": ["K"], "plm_prob": [0.1],
                  "mut_prob": [0.01], "obs_freq": [0.0]}).to_csv(
        model_tables_dir / f"{model_label}_combined_long_table.csv", index=False)
    pd.DataFrame({"alpha": [0.0], "mut_flat_global_spearman_r": [0.1]}).to_csv(
        model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
    pd.DataFrame({"alpha": [0.0], "lineage": ["K"]}).to_csv(
        model_tables_dir / f"{model_label}_alpha_sweep_fit_metrics_BY_LINEAGE.tsv",
        sep="\t", index=False)


class TestCacheValidity:
    @pytest.fixture()
    def scenario(self, driver_module, tmp_path):
        parent_map = dict(C.EXPECTED_PARENT_MAP)
        spec = make_spec("PRESCOTT_eq2_c0p50_k1", lineages=["K"],
                         parent_by_lineage={"K": "J.2.4"},
                         source_variant_by_lineage={"K": "PRESCOTT_eq2_c0p50_k1_parentJ24"},
                         equation=2, coefficient=0.5, frequency_cutoff_k=1)
        args = make_args(driver_module, tmp_path)
        model_tables_dir = tmp_path / "per_model"
        write_cached_model_tables(model_tables_dir, "PRESCOTT_eq2_c0p50_k1")
        metadata = pd.DataFrame([{
            "model": "PRESCOTT_eq2_c0p50_k1", "lineage": "K",
            "cache_version": driver_module.PRESCOTT_CACHE_VERSION,
            "design_key": driver_module.model_design_key(args, spec, parent_map),
            "mutation_model": "H3N2",
            "escott_temperature": 1.0, "parent_lineage": "J.2.4",
        }])
        return args, spec, parent_map, model_tables_dir, metadata

    def test_a_matching_cache_is_valid(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is True

    def test_an_empty_metadata_table_is_never_valid(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, _metadata = scenario
        assert driver_module.model_cache_is_valid(
            pd.DataFrame(), args, spec, parent_map, model_tables_dir) is False

    @pytest.mark.parametrize("field", ["force_recompute_scores", "diagnostic_exports"])
    def test_explicit_recompute_flags_defeat_the_cache(self, driver_module, scenario, field):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        setattr(args, field, True)
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_a_missing_model_column_is_not_valid(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        assert driver_module.model_cache_is_valid(
            metadata.drop(columns=["model"]), args, spec, parent_map, model_tables_dir) is False

    def test_a_model_absent_from_the_table_is_not_valid(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        other = dict(spec, model_tag="PRESCOTT_eq2_c1p00_k1")
        assert driver_module.model_cache_is_valid(
            metadata, args, other, parent_map, model_tables_dir) is False

    @pytest.mark.parametrize("column", ["cache_version", "design_key", "parent_lineage"])
    def test_a_table_written_before_a_guard_existed_is_stale(self, driver_module, scenario, column):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        assert driver_module.model_cache_is_valid(
            metadata.drop(columns=[column]), args, spec, parent_map, model_tables_dir) is False

    def test_a_bumped_cache_version_invalidates(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        metadata = metadata.assign(cache_version=driver_module.PRESCOTT_CACHE_VERSION + 1)
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_a_different_design_key_invalidates(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        metadata = metadata.assign(design_key="0000000000000000")
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_a_different_mutation_model_invalidates(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        metadata = metadata.assign(mutation_model="SC2")
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_a_different_temperature_invalidates(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        metadata = metadata.assign(escott_temperature=2.0)
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_a_different_parent_invalidates(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        metadata = metadata.assign(parent_lineage="J.2_int")
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is False

    def test_the_escott_baselines_empty_parent_cell_still_matches(
            self, driver_module, tmp_path):
        """The NaN-vs-'None' bug: ESCOTT has no parent, so its cell reads back as NaN."""
        parent_map = dict(C.EXPECTED_PARENT_MAP)
        spec = make_spec("ESCOTT", lineages=["K"], parent_by_lineage={"K": None})
        args = make_args(driver_module, tmp_path)
        model_tables_dir = tmp_path / "per_model"
        write_cached_model_tables(model_tables_dir, "ESCOTT")
        metadata_path = tmp_path / "panel_metadata.tsv"
        pd.DataFrame([{
            "model": "ESCOTT", "lineage": "K",
            "cache_version": driver_module.PRESCOTT_CACHE_VERSION,
            "design_key": driver_module.model_design_key(args, spec, parent_map),
            "mutation_model": "H3N2", "escott_temperature": 1.0, "parent_lineage": None,
        }]).to_csv(metadata_path, sep="\t", index=False)
        # Round-tripped through the TSV, exactly as run_analysis reads it back.
        metadata = pd.read_csv(metadata_path, sep="\t")
        assert metadata["parent_lineage"].isna().all()
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, model_tables_dir) is True

    def test_missing_cached_tables_defeat_the_cache(self, driver_module, scenario, tmp_path):
        args, spec, parent_map, _model_tables_dir, metadata = scenario
        empty = tmp_path / "empty_per_model"
        empty.mkdir()
        assert driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, empty) is False

    def test_the_whole_run_gate_is_the_conjunction(self, driver_module, scenario):
        args, spec, parent_map, model_tables_dir, metadata = scenario
        other = dict(spec, model_tag="PRESCOTT_eq2_c1p00_k1")
        assert driver_module.prescott_cache_is_valid(
            metadata, args, [spec], parent_map, model_tables_dir) is True
        assert driver_module.prescott_cache_is_valid(
            metadata, args, [spec, other], parent_map, model_tables_dir) is False

    def test_no_specs_is_never_a_valid_whole_run_cache(self, driver_module, scenario):
        args, _spec, parent_map, model_tables_dir, metadata = scenario
        assert driver_module.prescott_cache_is_valid(
            metadata, args, [], parent_map, model_tables_dir) is False


# =========================================================================== #
# Target resolution
# =========================================================================== #

class TestResolveTestTargetCount:
    TARGETS = [{"label": label} for label in C.LINEAGE_ORDER]

    def test_the_slice_grows_past_the_input_only_leading_rows(self, driver_module):
        # Guide row 1 is G.1, which is input-only; asking for 1 must yield 2.
        assert driver_module.resolve_test_target_count(self.TARGETS, 1, {"G.1"}) == 2

    def test_a_slice_that_already_contains_an_evaluable_lineage_is_kept(self, driver_module):
        assert driver_module.resolve_test_target_count(self.TARGETS, 3, {"G.1"}) == 3

    def test_a_non_positive_request_is_floored_at_one(self, driver_module):
        assert driver_module.resolve_test_target_count(self.TARGETS, 0, set()) == 1
        assert driver_module.resolve_test_target_count(self.TARGETS, -5, set()) == 1

    def test_an_all_input_only_guide_falls_back_to_the_whole_thing(self, driver_module):
        skip = set(C.LINEAGE_ORDER)
        assert driver_module.resolve_test_target_count(self.TARGETS, 1, skip) == len(self.TARGETS)


# =========================================================================== #
# CAVEATS and the manifest
# =========================================================================== #

@pytest.mark.integration
class TestCaveatsAndManifest:
    @pytest.fixture()
    def caveats_context(self, driver_module, tmp_path, prepared_inputs_tree):
        output_dir = tmp_path / "run"
        (output_dir / "tables" / "diagnostics").mkdir(parents=True)
        deep = tmp_path / "deep.fasta"
        deep.write_text(">a\nMK\n", encoding="utf-8")
        args = make_args(driver_module, output_dir,
                         inputs_dir=prepared_inputs_tree["inputs_dir"], deep_fasta=deep)
        args = driver_module.apply_prescott_defaults(args)
        args.inputs_dir = prepared_inputs_tree["inputs_dir"]
        parent_map = driver_module.resolve_parent_map(args)
        specs = [make_spec("ESCOTT", lineages=["K"])]
        return args, output_dir, parent_map, specs

    def test_caveats_names_the_corrected_parent_map(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert (output_dir / "CAVEATS.md").read_text(encoding="utf-8") == text
        assert '"K": "J.2.4"' in text
        assert "K is J.2.4.1, a child of J.2.4" in text

    def test_caveats_reports_the_jet_default_when_the_flag_is_unset(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        assert args.trace_top_fraction is None
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "0.9 (jet_surrogate.py default)" in text

    def test_caveats_reports_an_explicit_trace_top_fraction(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.trace_top_fraction = 0.30
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "top=0.3)" in text

    def test_caveats_only_cites_the_validation_table_when_it_exists(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "NOT PRESENT" in text
        validation = output_dir / "tables" / "diagnostics" / driver_module.JET_VALIDATION_BASENAME
        validation.write_text("metric\tvalue\n", encoding="utf-8")
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert f"tables/diagnostics/{driver_module.JET_VALIDATION_BASENAME}" in text
        assert "NOT PRESENT" not in text

    def test_caveats_says_so_when_validation_was_switched_off(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.jet_validation = False
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "NOT RUN (--no-jet-validation)" in text

    def test_caveats_explains_a_skipped_parity_check_in_test_mode(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.test_mode = True
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "SKIPPED in --test-mode" in text

    def test_caveats_sensitivity_status_off(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.parent_sensitivity = False
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "OFF (--no-parent-sensitivity)" in text

    def test_caveats_sensitivity_status_not_applicable(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.parent_sensitivity = True
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["J_int"])
        assert "the presets agree on every lineage evaluated here" in text

    def test_caveats_sensitivity_status_untested_when_no_model_carries_the_suffix(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        args.parent_sensitivity = True
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "treat the parent choice as UNTESTED" in text

    def test_caveats_sensitivity_status_names_the_scored_models(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, _specs = caveats_context
        args.parent_sensitivity = True
        specs = [make_spec("ESCOTT", lineages=["K"]),
                 make_spec("PRESCOTT_eq2_c0p50_k1_parentJ2int", lineages=["K"])]
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "were scored as separate model rows" in text
        assert "PRESCOTT_eq2_c0p50_k1_parentJ2int" in text

    def test_caveats_reports_the_zero_trace_counts(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        manifests = {"K": {"n_zero_trace_columns": 18, "frac_zero_trace_columns": 18 / 566}}
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"], manifests)
        assert "K 18 (3.2%)" in text

    def test_caveats_reports_an_unaudited_alignment(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "NOT AUDITED" in text

    # ---- caveat 8: the leakage audit, rendered from what stage 1 actually did ------

    @staticmethod
    def _install_leakage_block(prepared_inputs_tree, block):
        path = Path(prepared_inputs_tree["manifest_path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["leakage"] = block
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_caveats_renders_a_clean_purge(self, driver_module, caveats_context,
                                           prepared_inputs_tree, tmp_path):
        args, output_dir, parent_map, specs = caveats_context
        report_dir = tmp_path / "leakage_report"
        report_dir.mkdir()
        self._install_leakage_block(prepared_inputs_tree, {
            "status": "PASS",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True,
            "report_dir": str(report_dir),
            "purges": {"K": {"depth_before": 6434, "n_removed": 0, "depth_after": 6434,
                             "removed_fraction": 0.0}},
            "checks": {"B_parent_vs_target": {"K": {
                "parent": "J.2.4", "n_flagged": 8,
                "accessions": {"n_shared_accessions": 0, "n_shared_exact_sequences": 0}}}},
        })
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "PASS -- purge ON, 0 deep-set sequences removed across 1 target(s)" in text
        assert "identity >= 99.0 OR hamming <= 10" in text
        assert "K: 6434 -> 6434 (0 removed, 0.00%)" in text
        assert "J.2.4 -> K: 0 shared accessions, 0 shared exact sequences" in text
        assert str(report_dir) in text
        assert "no residual leakage above the configured gates" in text

    def test_caveats_reports_removed_sequences_with_their_identity(
            self, driver_module, caveats_context, prepared_inputs_tree):
        args, output_dir, parent_map, specs = caveats_context
        self._install_leakage_block(prepared_inputs_tree, {
            "status": "FAIL", "failures": ["removed_fraction above ceiling"],
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True, "report_dir": "/does/not/exist",
            "purges": {"K": {"depth_before": 6434, "n_removed": 12, "depth_after": 6422,
                             "removed_fraction": 0.0019,
                             "removed_identity_distribution": {"max": 99.8}}},
            "checks": {},
        })
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "12 removed, 0.19%, max removed identity 99.8%" in text
        assert "1 gate(s) failed: removed_fraction above ceiling" in text
        assert "not present in this output tree" in text
        assert "not run (--no-leakage-check)" in text  # no B_parent_vs_target block

    def test_caveats_says_detection_only_when_the_purge_is_off(
            self, driver_module, caveats_context, prepared_inputs_tree):
        args, output_dir, parent_map, specs = caveats_context
        self._install_leakage_block(prepared_inputs_tree, {
            "status": "PASS",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": False, "purges": {}, "checks": {},
        })
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "PURGE OFF (--no-purge-leakage): detection only" in text
        assert "DETECTION ONLY (purge off)" in text

    def test_caveats_says_nothing_was_purged_when_the_purge_ran_empty(
            self, driver_module, caveats_context, prepared_inputs_tree):
        args, output_dir, parent_map, specs = caveats_context
        self._install_leakage_block(prepared_inputs_tree, {
            "status": "PASS",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True, "purges": {}, "checks": {},
        })
        text = driver_module.write_caveats(args, output_dir, parent_map, specs, ["K"])
        assert "no evaluation target was purged in this pass" in text

    def test_the_manifest_carries_the_same_leakage_record(
            self, driver_module, caveats_context, prepared_inputs_tree):
        args, output_dir, parent_map, specs = caveats_context
        self._install_leakage_block(prepared_inputs_tree, {
            "status": "PASS",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True,
            "purges": {"K": {"depth_before": 6434, "n_removed": 0, "depth_after": 6434}},
            "checks": {}, "blast": {"task": "blastp-fast"},
        })
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["K"], None)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["leakage_stage_ran"] is True
        assert manifest["leakage_status"] == "PASS"
        assert manifest["leakage_blast"] == {"task": "blastp-fast"}
        assert manifest["leakage_per_target"]["K"]["depth_after"] == 6434

    def test_the_manifest_is_a_superset_of_the_plm_runs_keys(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        variants = pd.DataFrame([{"variant": "ESCOTT", "lineage": "K"}])
        driver_module.save_run_manifest(
            args, output_dir, [{"label": "K", "diversity_path": "", "reference_path": ""}],
            parent_map, specs, variants, ["K"], None)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        for key in ("alignment_verify_max_cols", "rolling_identity_window",
                    "observed_mutation_fasta", "observed_mutation_sequence_id",
                    "observed_mutation_selection", "alpha_grid", "scatter_alphas",
                    "mutation_model", "analysis_mode", "output_dir", "targets"):
            assert key in manifest, key

    def test_the_manifest_records_the_design_key_each_model_will_be_cached_under(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["K"], None)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["model_specs"][0]["design_key"] == \
            driver_module.model_design_key(args, specs[0], parent_map)
        assert manifest["design_key"] == driver_module.design_key(
            driver_module.design_signature(args, parent_map, ["K"]))

    def test_the_manifest_records_both_the_requested_and_effective_trace_fraction(
            self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["K"], None)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["trace_top_fraction_requested"] is None
        assert manifest["trace_top_fraction"] == 0.90
        assert manifest["escott_transform"] == "per_position_softmax"

    def test_the_manifest_reads_drop_parent_reversions_back_from_stage1(
            self, driver_module, caveats_context, prepared_inputs_tree):
        args, output_dir, parent_map, specs = caveats_context
        manifest_path = Path(prepared_inputs_tree["manifest_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["args"] = {"drop_parent_reversions": False}
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        assert args.drop_parent_reversions is True  # what the CLI asked for
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["K"], None)
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["drop_parent_reversions"] is False  # what stage 1 actually did

    def test_the_manifest_labels_a_pure_escott_run(self, driver_module, caveats_context):
        args, output_dir, parent_map, specs = caveats_context
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["K"], None)
        assert json.loads((output_dir / "run_manifest.json").read_text(
            encoding="utf-8"))["score_source"] == "escott"

        driver_module.save_run_manifest(
            args, output_dir, [], parent_map,
            [*specs, make_spec("PRESCOTT_eq2_c0p50_k1")], pd.DataFrame(), ["K"], None)
        assert json.loads((output_dir / "run_manifest.json").read_text(
            encoding="utf-8"))["score_source"] == "prescott"


# =========================================================================== #
# run_analysis end to end, on synthetic panels
# =========================================================================== #

@pytest.mark.integration
class TestRunAnalysis:
    """A complete pass with the score matrices pre-built and --no-auto-prepare.

    ``rma.export_plots`` is stubbed out in the fast variant: it is ~50 s of matplotlib
    belonging to the shared half, and every line it would exercise here is the PLM
    run's, not this driver's.  :meth:`test_the_real_figure_export_also_completes` runs
    it for real behind ``--run-slow``.
    """

    @pytest.fixture()
    def run_tree(self, driver_module, tmp_path, guide_factory):
        guide = guide_factory(["G.1", "J_int"], n_records=12)
        output_dir = tmp_path / "run"
        scores_dir = output_dir / "scores"
        protein = C.QUERY_PROTEIN
        escott_path = C.write_escott_normpred(
            tmp_path / "HAJ_normPred_evolCombi.txt", protein, ())
        probabilities = run_escott.escott_to_probability(
            run_escott.read_escott_matrix(escott_path, expect_protein=protein))
        rows = []
        for variant, meta in (
            ("ESCOTT", {"parent_lineage": None, "equation": None,
                        "coefficient": None, "frequency_cutoff_k": None}),
            ("PRESCOTT_eq2_c0p50_k1_parentG1", {"parent_lineage": "G.1", "equation": 2,
                                                "coefficient": 0.5, "frequency_cutoff_k": 1}),
        ):
            path = run_escott.write_score_matrix(
                probabilities, protein, scores_dir / f"J_int_{variant}_score_matrix.csv")
            rows.append({"variant": variant, "lineage": "J_int", "lineage_key": "J_int",
                         "score_matrix_path": str(path), **meta})
        pd.DataFrame(rows).to_csv(scores_dir / "score_variants.tsv", sep="\t", index=False)

        deep = tmp_path / "deep.fasta"
        deep.write_text(">a\nMK\n", encoding="utf-8")
        structure = tmp_path / "structure.cif"
        structure.write_text("data_x\n", encoding="utf-8")

        def _args(*extra):
            return driver_module.build_parser().parse_args([
                "--output-dir", str(output_dir),
                "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide["path"]),
                "--mutation-model", "H3N2", "--expect-protein-diversity",
                "--no-auto-prepare", "--no-parent-sensitivity",
                "--coefficient-grid", "0.5", "--equation-grid", "2",
                "--alpha-start", "-1", "--alpha-stop", "1", "--alpha-step", "0.5",
                "--scatter-alphas", "0",
                "--deep-fasta", str(deep), "--structure", str(structure),
                *extra,
            ])

        return {"guide": guide, "output_dir": output_dir, "scores_dir": scores_dir,
                "args": _args, "protein": protein}

    @pytest.fixture()
    def no_plots(self, driver_module, monkeypatch):
        monkeypatch.setattr(driver_module.rma, "export_plots", lambda **kwargs: None)

    def test_a_complete_pass_writes_every_pooled_table(self, driver_module, run_tree, no_plots):
        assert driver_module.run_analysis(run_tree["args"]()) == 0
        tables = run_tree["output_dir"] / "tables"
        for name in ("combined_long_table.csv", "panel_metadata.tsv",
                     "alpha_sweep_fit_metrics.tsv", "alpha_sweep_fit_metrics_BY_LINEAGE.tsv",
                     "best_alpha_two_methods.tsv", "best_alpha_per_group_two_methods.tsv",
                     "epoch_lineage_metrics.tsv", "epoch_metric_summary.tsv",
                     "model_run_status.tsv"):
            assert (tables / name).exists(), name
        assert (tables / "diagnostics" / "score_scale_report.tsv").exists()
        assert (run_tree["output_dir"] / "CAVEATS.md").exists()
        assert (run_tree["output_dir"] / "run_manifest.json").exists()

    def test_both_models_are_scored_over_every_non_reference_residue(
            self, driver_module, run_tree, no_plots):
        driver_module.run_analysis(run_tree["args"]())
        combined = pd.read_csv(run_tree["output_dir"] / "tables" / "combined_long_table.csv")
        assert sorted(combined["model"].unique()) == ["ESCOTT", "PRESCOTT_eq2_c0p50_k1"]
        assert combined["lineage"].unique().tolist() == ["J_int"]  # G.1 is input-only
        assert len(combined) == 2 * 19 * len(run_tree["protein"])
        assert combined["plm_prob"].between(0.0, 1.0).all()

    def test_the_sweep_rows_carry_the_escott_score_formula(
            self, driver_module, run_tree, no_plots):
        driver_module.run_analysis(run_tree["args"]())
        alpha = pd.read_csv(run_tree["output_dir"] / "tables" / "alpha_sweep_fit_metrics.tsv",
                            sep="\t")
        sweep = alpha.loc[alpha["model_variant"] == "plm_alpha_sweep"]
        assert set(sweep["input_score_formula"]) == {driver_module.INPUT_SCORE_FORMULA}
        assert sorted(sweep["alpha"].unique()) == [-1.0, -0.5, 0.0, 0.5, 1.0]

    def test_panel_metadata_records_the_cache_key_and_the_parent(
            self, driver_module, run_tree, no_plots):
        args = run_tree["args"]()
        driver_module.run_analysis(args)
        metadata = pd.read_csv(run_tree["output_dir"] / "tables" / "panel_metadata.tsv", sep="\t")
        assert set(metadata["model"]) == {"ESCOTT", "PRESCOTT_eq2_c0p50_k1"}
        assert (metadata["cache_version"] == driver_module.PRESCOTT_CACHE_VERSION).all()
        assert metadata["design_key"].notna().all()
        prescott_row = metadata.loc[metadata["model"] != "ESCOTT"].iloc[0]
        assert prescott_row["parent_lineage"] == "G.1"
        assert pd.isna(metadata.loc[metadata["model"] == "ESCOTT"].iloc[0]["parent_lineage"])
        manifest = json.loads(
            (run_tree["output_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
        keys = {entry["model_tag"]: entry["design_key"] for entry in manifest["model_specs"]}
        for _, row in metadata.iterrows():
            assert row["design_key"] == keys[row["model"]]

    def test_a_rerun_reuses_every_cached_model(self, driver_module, run_tree, no_plots, capsys):
        driver_module.run_analysis(run_tree["args"]())
        capsys.readouterr()
        assert driver_module.run_analysis(run_tree["args"]()) == 0
        out = capsys.readouterr().out
        assert "Reusing cached tables for ESCOTT" in out
        assert "Reusing cached tables for PRESCOTT_eq2_c0p50_k1" in out
        assert "re-parsing the diversity panels" not in out or "cached" in out

    def test_changing_the_grid_forces_stage_1_rather_than_analysing_the_old_design(
            self, driver_module, run_tree, no_plots, capsys):
        driver_module.run_analysis(run_tree["args"]())
        capsys.readouterr()
        with pytest.raises(FileNotFoundError, match="Stage 1 is needed"):
            driver_module.run_analysis(run_tree["args"]("--coefficient-grid", "0.75"))

    def test_a_shrunk_grid_is_reported_as_ignored_not_analysed(
            self, driver_module, run_tree, no_plots, capsys):
        # Ask for the ESCOTT baseline only; the cached PRESCOTT variant must be ignored.
        assert driver_module.run_analysis(
            run_tree["args"]("--score-variant", "ESCOTT")) == 0
        combined = pd.read_csv(run_tree["output_dir"] / "tables" / "combined_long_table.csv")
        assert combined["model"].unique().tolist() == ["ESCOTT"]

    def test_dry_run_stops_before_scoring_but_writes_the_profiles(
            self, driver_module, run_tree, no_plots):
        assert driver_module.run_analysis(run_tree["args"]("--dry-run")) == 0
        groups = run_tree["output_dir"] / "groups"
        assert (groups / "J_int_observed_diversity_profile.csv").exists()
        assert (groups / "J_int_mutation_accessibility_profile.csv").exists()
        assert (groups / "J_int_observed_depth_profile.csv").exists()
        assert (run_tree["output_dir"] / "run_manifest.json").exists()
        assert not (run_tree["output_dir"] / "tables" / "combined_long_table.csv").exists()

    def test_an_unmapped_lineage_is_refused_before_any_work(
            self, driver_module, run_tree, no_plots):
        args = run_tree["args"]("--parent-map-preset", "clade_evidence")
        args.parent_map = None
        args.parent_map_preset = "clade_evidence"
        # Strip J_int from the map so it has no basal panel.
        import copy

        presets = copy.deepcopy(dict(constants.DEFAULT_PARENT_MAPS))
        presets["clade_evidence"].pop("J_int")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(driver_module, "parent_map_presets", lambda: presets)
            with pytest.raises(ValueError, match="No basal lineage defined"):
                driver_module.run_analysis(args)

    def test_a_guide_of_only_input_only_lineages_has_nothing_to_score(
            self, driver_module, tmp_path, guide_factory, no_plots):
        guide = guide_factory(["G.1"], n_records=6)
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "empty_run"),
            "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide["path"]),
            "--mutation-model", "H3N2", "--expect-protein-diversity", "--no-auto-prepare",
        ])
        with pytest.raises(RuntimeError, match="Every resolved target is input-only"):
            driver_module.run_analysis(args)

    def test_missing_score_matrices_with_no_auto_prepare_name_the_first_gap(
            self, driver_module, tmp_path, guide_factory, no_plots):
        guide = guide_factory(["G.1", "J_int"], n_records=6)
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "bare_run"),
            "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide["path"]),
            "--mutation-model", "H3N2", "--expect-protein-diversity", "--no-auto-prepare",
        ])
        with pytest.raises(FileNotFoundError) as excinfo:
            driver_module.run_analysis(args)
        assert "Stage 1 is needed" in str(excinfo.value)
        assert "First missing: ESCOTT / J_int" in str(excinfo.value)

    def test_test_mode_grows_the_guide_slice_past_the_input_only_row(
            self, driver_module, run_tree, no_plots, capsys):
        assert driver_module.run_analysis(
            run_tree["args"]("--test-mode", "--test-max-targets", "1")) == 0
        out = capsys.readouterr().out
        assert "using the first 2 guide row(s)" in out
        assert "Evaluable lineages: ['J_int']" in out

    def test_a_score_matrix_built_on_a_different_sequence_is_refused(
            self, driver_module, run_tree, no_plots, capsys):
        # Overwrite the ESCOTT matrix with one built on a truncated protein.
        truncated = run_tree["protein"][:-5]
        probabilities = pd.DataFrame(
            1.0 / 20.0, index=list(run_escott.PLM_CACHE_AA_ORDER),
            columns=range(1, len(truncated) + 1))
        run_escott.write_score_matrix(
            probabilities, truncated,
            run_tree["scores_dir"] / "J_int_ESCOTT_score_matrix.csv")
        assert driver_module.run_analysis(run_tree["args"]()) == 0
        status = pd.read_csv(run_tree["output_dir"] / "tables" / "model_run_status.tsv", sep="\t")
        failed = status.loc[status["status"] == "failed"]
        assert not failed.empty
        assert "refusing to combine them" in " ".join(failed["reason"].astype(str))
        combined = pd.read_csv(run_tree["output_dir"] / "tables" / "combined_long_table.csv")
        assert "ESCOTT" not in combined["model"].unique()

    def test_regen_figures_only_delegates_to_the_shared_replotter(
            self, driver_module, run_tree, monkeypatch):
        seen = {}

        def fake_regen(args, **kwargs):
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(driver_module.rma,
                            "_regenerate_figures_from_existing_tables", fake_regen)
        assert driver_module.run_analysis(run_tree["args"]("--regen-figures-only")) == 0
        assert seen["tables_dir"] == run_tree["output_dir"] / "tables"
        assert seen["plots_dir"] == run_tree["output_dir"] / "plots"

    def test_cached_variants_outside_the_design_are_named_and_left_alone(
            self, driver_module, run_tree, no_plots, capsys):
        # Add a c=1.00 variant to the cache that the CLI does not ask for.
        scores_dir = run_tree["scores_dir"]
        extra = "PRESCOTT_eq2_c1p00_k1_parentG1"
        (scores_dir / f"J_int_{extra}_score_matrix.csv").write_bytes(
            (scores_dir / "J_int_PRESCOTT_eq2_c0p50_k1_parentG1_score_matrix.csv").read_bytes())
        table = pd.read_csv(scores_dir / "score_variants.tsv", sep="\t")
        table = pd.concat([table, pd.DataFrame([{
            "variant": extra, "lineage": "J_int", "lineage_key": "J_int",
            "score_matrix_path": str(scores_dir / f"J_int_{extra}_score_matrix.csv"),
            "parent_lineage": "G.1", "equation": 2, "coefficient": 1.0,
            "frequency_cutoff_k": 1}])], ignore_index=True)
        table.to_csv(scores_dir / "score_variants.tsv", sep="\t", index=False)

        assert driver_module.run_analysis(run_tree["args"]()) == 0
        out = capsys.readouterr().out
        assert "Ignoring 1 cached score variant(s) outside the requested design" in out
        combined = pd.read_csv(run_tree["output_dir"] / "tables" / "combined_long_table.csv")
        assert "PRESCOTT_eq2_c1p00_k1" not in set(combined["model"])
        assert (scores_dir / f"J_int_{extra}_score_matrix.csv").exists()  # left on disk

    def test_auto_prepare_runs_stage_1_and_then_reconciles_again(
            self, driver_module, run_tree, no_plots, monkeypatch, capsys):
        scores_dir = run_tree["scores_dir"]
        stashed = {path.name: path.read_bytes() for path in scores_dir.glob("*")}
        for path in scores_dir.glob("*"):
            path.unlink()
        calls = []

        def fake_stage1(args, parent_map, evaluable, diagnostics_dir):
            calls.append((sorted(evaluable), Path(diagnostics_dir)))
            for name, payload in stashed.items():
                (scores_dir / name).write_bytes(payload)

        monkeypatch.setattr(driver_module, "run_stage1", fake_stage1)
        assert driver_module.run_analysis(run_tree["args"]("--auto-prepare")) == 0
        assert calls == [(["J_int"], run_tree["output_dir"] / "tables" / "diagnostics")]
        out = capsys.readouterr().out
        assert "Stage 1 needed (2 requested score matrix/matrices not available)" in out
        assert "needs: ESCOTT / J_int" in out

    def test_a_stage_1_that_does_not_deliver_is_refused_rather_than_reported(
            self, driver_module, run_tree, no_plots, monkeypatch):
        for path in run_tree["scores_dir"].glob("*"):
            path.unlink()
        monkeypatch.setattr(driver_module, "run_stage1",
                            lambda *a, **k: None)  # produces nothing
        with pytest.raises(RuntimeError) as excinfo:
            driver_module.run_analysis(run_tree["args"]("--auto-prepare"))
        message = str(excinfo.value)
        assert "the requested design is still incomplete" in message
        assert "refusing to report a run whose manifest would not describe its outputs" in message

    def test_force_recompute_reruns_stage_1_even_with_a_complete_cache(
            self, driver_module, run_tree, no_plots, monkeypatch, capsys):
        monkeypatch.setattr(driver_module, "run_stage1", lambda *a, **k: None)
        assert driver_module.run_analysis(
            run_tree["args"]("--auto-prepare", "--force-recompute-scores")) == 0
        assert "Stage 1 needed (--force-recompute-scores)" in capsys.readouterr().out

    def test_a_temperature_change_under_match_plm_invalidates_the_cached_matrices(
            self, driver_module, run_tree, no_plots, tmp_path):
        reference = tmp_path / "plm_reference.csv"
        pd.DataFrame({"plm_prob": [0.001, 0.01, 0.1, 0.5]}).to_csv(reference, index=False)
        pd.DataFrame([[-1.0, -2.0], [-3.0, -4.0]], index=["A", "C"], columns=[1, 2]).to_csv(
            run_tree["scores_dir"] / "J_int_ESCOTT_raw.tsv", sep="\t")
        table = pd.read_csv(run_tree["scores_dir"] / "score_variants.tsv", sep="\t")
        table["temperature"] = 1.0  # what the cached matrices were built at
        table.to_csv(run_tree["scores_dir"] / "score_variants.tsv", sep="\t", index=False)
        args = run_tree["args"]("--escott-temperature-mode", "match-plm",
                                "--plm-reference-table", str(reference))
        with pytest.raises(FileNotFoundError, match="temperature changed under"):
            driver_module.run_analysis(args)

    def test_a_guide_that_resolves_nothing_is_a_hard_error(
            self, driver_module, run_tree, no_plots, monkeypatch):
        monkeypatch.setattr(driver_module, "resolve_targets", lambda args: [])
        with pytest.raises(RuntimeError, match="No targets resolved"):
            driver_module.run_analysis(run_tree["args"]())

    @pytest.mark.slow
    def test_the_real_figure_export_also_completes(self, driver_module, run_tree):
        assert driver_module.run_analysis(run_tree["args"]()) == 0
        plots = run_tree["output_dir"] / "plots"
        assert plots.is_dir()
        assert any(plots.rglob("*.png")) or any(plots.rglob("*.pdf"))


# =========================================================================== #
# CLI surface
# =========================================================================== #

@pytest.mark.cli
class TestCli:
    def test_the_defaults_that_matter_are_what_the_measurements_chose(self, driver_module):
        args = driver_module.build_parser().parse_args(["--output-dir", "/tmp/x"])
        assert args.trace_top_fraction is None, "must not override jet_surrogate's 0.90"
        assert args.max_zero_trace_fraction is None
        assert args.parent_map_preset == "clade_evidence"
        assert args.parent_sensitivity is True
        assert args.drop_parent_reversions is True
        assert args.parent_freq_max == 0.95
        assert args.trace_definition == "bootstrap"
        assert args.escott_temperature == 1.0
        assert args.frequency_cutoff_mode == "depth_scaled"
        assert args.leakage_check is True and args.purge_leakage is True
        assert args.jet_validation is True
        assert args.test_max_records == 0

    def test_preset_choices_come_from_the_shared_module(self, driver_module):
        parser = driver_module.build_parser()
        action = [a for a in parser._actions if a.dest == "parent_map_preset"][0]
        assert set(action.choices) == set(constants.DEFAULT_PARENT_MAPS)

    def test_single_fasta_exits_2_from_main(self, driver_module, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "run_prescott_diversity.py", "--output-dir", str(tmp_path),
            "--analysis-mode", "SINGLE_FASTA",
        ])
        with pytest.raises(SystemExit) as excinfo:
            driver_module.main()
        assert excinfo.value.code == 2
        assert "SINGLE_FASTA is not supported" in capsys.readouterr().err

    def test_equation_4_exits_2_from_main(self, driver_module, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "run_prescott_diversity.py", "--output-dir", str(tmp_path),
            "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(C.REAL_GUIDE),
            "--equation-grid", "4",
        ])
        with pytest.raises(SystemExit) as excinfo:
            driver_module.main()
        assert excinfo.value.code == 2
        assert "equation 4 is not implemented" in capsys.readouterr().err

    def test_an_unknown_analysis_mode_is_an_argparse_error(self, driver_module):
        with pytest.raises(SystemExit):
            driver_module.build_parser().parse_args(
                ["--output-dir", "/tmp/x", "--analysis-mode", "WEEKLY"])

    def test_output_dir_is_mandatory(self, driver_module):
        with pytest.raises(SystemExit):
            driver_module.build_parser().parse_args(["--analysis-mode", "MONTHLY_GUIDE"])

    def test_the_input_score_formula_is_the_escott_one(self, driver_module):
        assert driver_module.INPUT_SCORE_FORMULA == "escott_prob * mut_prob^alpha"
        assert "plm_prob" not in driver_module.INPUT_SCORE_FORMULA
