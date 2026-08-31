#!/usr/bin/env python3
"""CLI, validation, planning and caching tests for ``scripts/run_prescott_diversity.py``.

SCOPE
=====
Everything here is *cheap*: argparse, ``validate_args``, parent-map resolution, the
requested-design plan, the reconcile against ``scores/score_variants.tsv``, the model
specs, the design keys, the per-model cache guard, the stage-1 command lines and the
manifest/CAVEATS renderers.  Nothing in this file runs ESCOTT, builds a lineage cache,
parses a real panel or draws a figure -- the two places where ``run_analysis`` would
tip over into real work (``resolve_targets`` and
``Functions_HuggingFace.build_codon_aa_mutation_tables``) are replaced, and the second
one is replaced with a *raise* so the planning half can be exercised end to end and
then stopped dead.

WHAT THIS FILE IS MOSTLY ABOUT
==============================
The fixed blocker: **the CLI is the authority over ``scores/score_variants.tsv``.**
A rerun into a populated ``--output-dir`` must honour ``--coefficient-grid``,
``--equation-grid``, ``--frequency-cutoff-k``, ``--parent-map`` and
``--parent-map-preset``; a combination the CLI asked for and the cache does not hold
must force stage 1 or fail loudly; a combination the cache holds and the CLI did not
ask for must be dropped and reported, never analysed.  ``TestReconcileVariantPlan``
and ``TestRunAnalysisPlanning`` cover that from both ends, and
``TestManifestDescribesTheModelsProduced`` checks the manifest cannot describe a
design the run did not produce.

GROUND TRUTH
============
Expected parent maps, variant names and model tags are written here as **literals**.
A test that asked the module for the map and then compared it with itself would pass
happily after a regression to the old ``K <- J.2_int`` edge.  The one map imported
from anywhere is the conftest's ``EXPECTED_PARENT_MAP``, which is itself a literal for
the same reason.

RUN IT
======
``/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
      /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_driver_cli.py -q``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from tests_prescott_iav.conftest import (
    CONTESTED_EDGE,
    EXPECTED_PARENT_MAP,
    EXPECTED_SENSITIVITY_PARENT_MAP,
    LINEAGE_ORDER,
)

# Every test in this module is a fast, offline unit/CLI test.  ``requires_rma`` is the
# only capability it needs: the driver imports run_mutational_accessibility, which
# imports torch, at module scope.
pytestmark = [pytest.mark.unit, pytest.mark.requires_rma]


# --------------------------------------------------------------------------- #
# Literal ground truth (never imported from the module under test)
# --------------------------------------------------------------------------- #

CLADE_EVIDENCE_MAP: Dict[str, str] = {
    "J_int": "G.1",
    "J.2_int": "J_int",
    "J.2.4": "J.2_int",
    "K": "J.2.4",
}
BRIEF_AS_STATED_MAP: Dict[str, str] = {
    "J_int": "G.1",
    "J.2_int": "J_int",
    "J.2.4": "J.2_int",
    "K": "J.2_int",
}
EVALUABLE = ["J_int", "J.2_int", "J.2.4", "K"]

# The stage-1 variant names the driver must predict for the default grid, spelled out.
EXPECTED_PRIMARY_VARIANTS_C050_K1 = {
    "J_int": "PRESCOTT_eq2_c0p50_k1_parentG1",
    "J.2_int": "PRESCOTT_eq2_c0p50_k1_parentJint",
    "J.2.4": "PRESCOTT_eq2_c0p50_k1_parentJ2int",
    "K": "PRESCOTT_eq2_c0p50_k1_parentJ24",
}
EXPECTED_SENSITIVITY_VARIANT_C050_K1 = "PRESCOTT_eq2_c0p50_k1_parentJ2int"  # K under J.2_int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _StopAfterPlanning(Exception):
    """Raised in place of the first genuinely expensive call in ``run_analysis``."""


def _tiny_file(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def parse_cli(driver, tmp_path: Path, *extra: str, guide: Optional[Path] = None,
              output_dir: Optional[Path] = None) -> argparse.Namespace:
    """``build_parser().parse_args`` with the two mandatory-ish flags filled in.

    Deliberately goes through the real parser rather than hand-building a Namespace:
    a hand-built one cannot notice a default changing, and defaults are exactly what
    several of the fixes here are about (``--trace-top-fraction`` is None, the preset
    is ``clade_evidence``, ``--test-max-records`` is 0).
    """
    guide = guide if guide is not None else _tiny_file(tmp_path / "guide.csv",
                                                       "month,fasta,reference\n")
    output_dir = output_dir if output_dir is not None else tmp_path / "out"
    argv = ["--analysis-mode", "MONTHLY_GUIDE",
            "--guide-path", str(guide),
            "--output-dir", str(output_dir),
            "--deep-fasta", str(_tiny_file(tmp_path / "deep.fasta", ">a\nMKT\n")),
            "--structure", str(_tiny_file(tmp_path / "structure.cif", "data_x\n")),
            *extra]
    return driver.build_parser().parse_args(argv)


def prepared_args(driver, tmp_path: Path, *extra: str, **kwargs) -> argparse.Namespace:
    """``parse_cli`` followed by ``apply_prescott_defaults`` (derived dirs filled in)."""
    return driver.apply_prescott_defaults(parse_cli(driver, tmp_path, *extra, **kwargs))


# The subset of run_escott.write_variants_table's columns this driver consumes, plus
# the two the sibling agent added (is_primary_parent / frequency_path).  Kept faithful
# so a schema drift in stage 1 shows up here.
VARIANT_TABLE_COLUMNS = [
    "variant", "lineage", "lineage_key", "parent_lineage", "is_primary_parent",
    "frequency_path", "equation", "coefficient", "frequency_cutoff",
    "frequency_cutoff_k", "temperature", "score_matrix_path", "md5",
]


def write_variants_table(
    scores_dir: Path,
    entries: Sequence[Dict[str, object]],
    *,
    temperature: float = 1.0,
    create_matrices: bool = True,
    matrix_dir: Optional[Path] = None,
) -> Path:
    """Write a stage-1-shaped ``score_variants.tsv`` (and, by default, its matrices).

    ``entries`` are plan-shaped dicts (``source_variant``/``lineage``/``lineage_key``/
    ``parent_lineage``/``equation``/``coefficient``/``frequency_cutoff_k``), i.e. what
    ``expected_variant_plan`` returns, so a test can say "the cache holds exactly the
    plan for grid X".

    The matrix files are one-byte stubs: nothing in the planning or caching code reads
    them, it only asks whether they exist, and a real 20xL matrix here would only
    slow the suite down.  ``matrix_dir`` writes them somewhere other than
    ``scores_dir`` so the "recorded absolute path in a copied tree" case is testable.
    """
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for entry in entries:
        variant = str(entry["source_variant"])
        key = str(entry["lineage_key"])
        target_dir = Path(matrix_dir) if matrix_dir is not None else scores_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        matrix = target_dir / f"{key}_{variant}_score_matrix.csv"
        if create_matrices:
            matrix.write_text("stub\n", encoding="utf-8")
        rows.append({
            "variant": variant,
            "lineage": entry["lineage"],
            "lineage_key": key,
            "parent_lineage": entry.get("parent_lineage"),
            "is_primary_parent": None if entry.get("equation") is None else True,
            "frequency_path": None if entry.get("equation") is None else "freq.txt",
            "equation": entry.get("equation"),
            "coefficient": entry.get("coefficient"),
            "frequency_cutoff": None if entry.get("equation") is None else -2.9,
            "frequency_cutoff_k": entry.get("frequency_cutoff_k"),
            "temperature": float(temperature),
            "score_matrix_path": str(matrix),
            "md5": "0" * 32,
        })
    path = scores_dir / "score_variants.tsv"
    pd.DataFrame(rows, columns=VARIANT_TABLE_COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def flag_value(command: Sequence[str], flag: str) -> str:
    """The argument immediately after ``flag`` in a built command line."""
    parts = [str(part) for part in command]
    return parts[parts.index(flag) + 1]


def flag_values(command: Sequence[str], flag: str) -> List[str]:
    """Every argument following every occurrence of ``flag`` (for repeatable flags)."""
    parts = [str(part) for part in command]
    return [parts[i + 1] for i, part in enumerate(parts) if part == flag]


def capture_stage1_steps(driver, monkeypatch) -> List[Dict[str, object]]:
    """Replace ``run_stage1_step`` with a recorder.  Returns the (growing) call list."""
    calls: List[Dict[str, object]] = []

    def _record(command, env, label):
        calls.append({"label": label, "command": [str(part) for part in command], "env": env})

    monkeypatch.setattr(driver, "run_stage1_step", _record)
    return calls


def run_planning(
    driver,
    args: argparse.Namespace,
    monkeypatch,
    *,
    targets: Sequence[str],
    stage1: Optional[object] = None,
) -> Dict[str, object]:
    """Drive ``run_analysis`` through its planning half and stop before real work.

    ``run_analysis`` resolves the parent map, resolves targets, builds the requested
    plan, reconciles it against the cache, decides on stage 1 and builds the model
    specs BEFORE it touches a codon table -- that ordering is deliberate (it is what
    lets a fully cached rerun skip parsing the 27452-sequence panels).  Replacing
    ``build_codon_aa_mutation_tables`` with a raise therefore stops the run at exactly
    the boundary this file cares about.

    Returns the recorded ``plan``/``specs``, the stage-1 calls and the exception (if
    the planner refused).
    """
    import Functions_HuggingFace as fhf

    resolved = [{"label": label, "diversity_path": f"/panels/{label}.fa",
                 "reference_path": f"/refs/{label}.fa"} for label in targets]
    monkeypatch.setattr(driver, "resolve_targets", lambda _a: list(resolved))

    recorded: Dict[str, object] = {"stage1_calls": [], "specs": None, "plan": None,
                                   "error": None}

    real_build_specs = driver.build_score_specs

    def _spy_specs(a, plan, parent_map):
        recorded["plan"] = [dict(entry) for entry in plan]
        specs = real_build_specs(a, plan, parent_map)
        recorded["specs"] = specs
        return specs

    monkeypatch.setattr(driver, "build_score_specs", _spy_specs)

    def _stage1(a, parent_map, evaluable, diagnostics_dir):
        recorded["stage1_calls"].append({
            "parent_map": dict(parent_map),
            "evaluable": list(evaluable),
            "diagnostics_dir": Path(diagnostics_dir),
        })
        if stage1 is not None:
            stage1(a, parent_map, evaluable, diagnostics_dir)

    monkeypatch.setattr(driver, "run_stage1", _stage1)

    def _boom(*_a, **_k):
        raise _StopAfterPlanning("planning finished")

    monkeypatch.setattr(fhf, "build_codon_aa_mutation_tables", _boom)

    try:
        driver.run_analysis(args)
    except _StopAfterPlanning:
        recorded["reached_compute"] = True
    except Exception as exc:  # the planner refused; that is a result, not a failure
        recorded["reached_compute"] = False
        recorded["error"] = exc
    else:  # pragma: no cover - run_analysis cannot return without hitting the boom
        recorded["reached_compute"] = False
    return recorded


def model_tags(specs) -> List[str]:
    return [str(spec["model_tag"]) for spec in specs]


# =========================================================================== #
# 1. Parser surface
# =========================================================================== #

class TestParserDefaults:
    """Defaults that other halves of the pipeline depend on being exactly these."""

    def test_output_dir_is_required(self, driver_module):
        with pytest.raises(SystemExit) as excinfo:
            driver_module.build_parser().parse_args(["--analysis-mode", "MONTHLY_GUIDE"])
        assert excinfo.value.code == 2

    def test_trace_top_fraction_defaults_to_none_so_jet_surrogate_wins(
        self, driver_module, tmp_path
    ):
        # The blocker fix: a non-None default here silently overrode jet_surrogate's
        # measured 0.90 with 0.30 and left 77/566 HA positions at trace == 0.
        args = parse_cli(driver_module, tmp_path)
        assert args.trace_top_fraction is None
        assert args.max_zero_trace_fraction is None

    def test_shared_constants_supply_the_documented_trace_default(self, driver_module):
        assert driver_module.default_trace_top_fraction() == 0.90

    def test_design_grid_defaults_are_exact(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        assert args.coefficient_grid == "0.25,0.5,1.0"
        assert args.equation_grid == "2"
        assert args.frequency_cutoff_k == "1"
        assert args.frequency_cutoff == -4.0
        assert args.frequency_cutoff_mode == "depth_scaled"

    def test_parent_design_defaults_are_exact(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        assert args.parent_map is None
        assert args.parent_map_preset == "clade_evidence"
        assert args.parent_sensitivity is True
        assert args.drop_parent_reversions is True
        assert args.parent_freq_max == 0.95
        assert args.parent_min_count == 1
        assert args.parent_min_depth == 50

    def test_test_max_records_defaults_to_zero_not_five(self, driver_module, tmp_path):
        # Deliberately unlike the PLM driver's 5: with 5 sequences the observed
        # frequency profile is noise.  0 is this driver's "do not truncate" sentinel.
        args = parse_cli(driver_module, tmp_path)
        assert args.test_max_records == 0
        assert args.test_max_targets == 1
        assert args.test_mode is False

    def test_leakage_thresholds_default_to_none_so_stage1_stays_authoritative(
        self, driver_module, tmp_path
    ):
        args = parse_cli(driver_module, tmp_path)
        for name in ("leakage_min_identity", "leakage_max_hamming", "leakage_min_coverage",
                     "leakage_coverage_basis", "leakage_max_removed_fraction",
                     "leakage_min_depth_after", "leakage_threads", "blast_task"):
            assert getattr(args, name) is None, name
        assert args.leakage_check is True
        assert args.purge_leakage is True
        assert args.fail_on_leakage is False

    def test_derived_directories_default_under_output_dir(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        assert args.scores_dir is None and args.inputs_dir is None
        args = driver_module.apply_prescott_defaults(args)
        out = Path(args.output_dir)
        assert Path(args.scores_dir) == out / "scores"
        assert Path(args.inputs_dir) == out / "inputs"
        assert Path(args.escott_workdir) == out / "escott"
        assert Path(args.prescott_ref_dir) == out / "prescott_ref"

    def test_parent_map_preset_choices_come_from_the_shared_constants(self, driver_module):
        parser = driver_module.build_parser()
        action = next(a for a in parser._actions if a.dest == "parent_map_preset")
        assert set(action.choices) == {"clade_evidence", "brief_as_stated"}

    def test_jet_validation_is_on_by_default(self, driver_module, tmp_path):
        assert parse_cli(driver_module, tmp_path).jet_validation is True


class TestParserPlumbing:
    """Flags whose *shape* matters: repeatable, boolean-optional, explicit off."""

    def test_score_variant_is_repeatable_and_accumulates(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path,
                         "--score-variant", "ESCOTT",
                         "--score-variant", "PRESCOTT_eq2_c0p50_k1")
        assert args.score_variants == ["ESCOTT", "PRESCOTT_eq2_c0p50_k1"]

    def test_score_variant_default_is_none_not_empty_list(self, driver_module, tmp_path):
        # build_score_specs branches on `is not None`; an empty list would silently
        # mean "restrict to nothing" and raise "No score variants resolved".
        assert parse_cli(driver_module, tmp_path).score_variants is None

    @pytest.mark.parametrize("flag,dest,expected", [
        ("--no-parent-sensitivity", "parent_sensitivity", False),
        ("--no-drop-parent-reversions", "drop_parent_reversions", False),
        ("--no-leakage-check", "leakage_check", False),
        ("--no-purge-leakage", "purge_leakage", False),
        ("--no-jet-validation", "jet_validation", False),
        ("--no-auto-prepare", "auto_prepare", False),
        ("--no-filter-fixed-mutations", "filter_fixed_mutations", False),
        ("--filter-singleton-mutations", "filter_singleton_mutations", True),
    ])
    def test_boolean_optional_actions(self, driver_module, tmp_path, flag, dest, expected):
        assert getattr(parse_cli(driver_module, tmp_path, flag), dest) is expected

    def test_single_fasta_is_still_an_argparse_choice(self, driver_module):
        # It must reach validate_args, so the user gets the explanation rather than
        # argparse's bare "invalid choice".
        parser = driver_module.build_parser()
        action = next(a for a in parser._actions if a.dest == "analysis_mode")
        assert set(action.choices) == {"SINGLE_FASTA", "MONTHLY_GUIDE"}

    def test_test_mode_help_promises_only_that_it_limits_reading(self, driver_module):
        parser = driver_module.build_parser()
        action = next(a for a in parser._actions if a.dest == "test_mode")
        help_text = action.help
        assert "limits how much data is READ" in help_text
        # The documented promise the fix had to make true.
        assert "--trace-definition" in help_text and "NOTHING else" in help_text


# =========================================================================== #
# 2. validate_args
# =========================================================================== #

class TestValidateArgs:
    def test_single_fasta_is_rejected_at_parse_time_with_the_equivalent_recipe(
        self, driver_module, tmp_path
    ):
        args = parse_cli(driver_module, tmp_path, "--analysis-mode", "SINGLE_FASTA")
        with pytest.raises(ValueError) as excinfo:
            driver_module.validate_args(args)
        message = str(excinfo.value)
        assert "SINGLE_FASTA is not supported" in message
        # The error must hand the user the working alternative, not just refuse.
        assert "--analysis-mode MONTHLY_GUIDE" in message
        assert "--parent-map" in message
        assert "BASAL (parent) lineage panel" in message

    def test_single_fasta_is_rejected_even_when_both_fastas_are_supplied(
        self, driver_module, tmp_path
    ):
        # There is no combination of flags that makes SINGLE_FASTA work: the missing
        # thing is a parent PANEL, which no single FASTA can carry.
        args = parse_cli(
            driver_module, tmp_path,
            "--analysis-mode", "SINGLE_FASTA",
            "--diversity-fasta", str(_tiny_file(tmp_path / "div.fa", ">a\nMK\n")),
            "--reference-fasta", str(_tiny_file(tmp_path / "ref.fa", ">a\nATG\n")),
            "--parent-map", "population=K",
        )
        with pytest.raises(ValueError, match="SINGLE_FASTA is not supported"):
            driver_module.validate_args(args)

    def test_monthly_guide_accepts_an_existing_guide(self, driver_module, tmp_path):
        driver_module.validate_args(parse_cli(driver_module, tmp_path))

    def test_missing_guide_raises_file_not_found(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, guide=tmp_path / "absent.csv")
        with pytest.raises(FileNotFoundError, match="Guide file not found"):
            driver_module.validate_args(args)

    def test_analysis_mode_required_unless_regen_figures_only(self, driver_module, tmp_path):
        parser = driver_module.build_parser()
        args = parser.parse_args(["--output-dir", str(tmp_path / "o")])
        with pytest.raises(ValueError, match="--analysis-mode is required"):
            driver_module.validate_args(args)

    def test_regen_figures_only_skips_every_other_check(self, driver_module, tmp_path):
        parser = driver_module.build_parser()
        args = parser.parse_args([
            "--output-dir", str(tmp_path / "o"),
            "--regen-figures-only",
            "--alpha-step", "-5",              # would fail every other validation
            "--equation-grid", "4",
        ])
        driver_module.validate_args(args)      # must not raise

    def test_regen_figures_only_still_needs_output_dir(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--regen-figures-only")
        args.output_dir = None
        with pytest.raises(ValueError, match="--output-dir is required"):
            driver_module.validate_args(args)

    def test_missing_mutation_model_is_rejected(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        args.mutation_model = None
        with pytest.raises(ValueError, match="--mutation-model is required"):
            driver_module.validate_args(args)

    def test_missing_guide_path_is_rejected(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        args.guide_path = None
        with pytest.raises(ValueError, match="--guide-path is required"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("step", ["0", "-0.1"])
    def test_non_positive_alpha_step_is_rejected(self, driver_module, tmp_path, step):
        args = parse_cli(driver_module, tmp_path, "--alpha-step", step)
        with pytest.raises(ValueError, match=r"--alpha-step must be > 0"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("temperature", ["0", "-1"])
    def test_non_positive_temperature_is_rejected(self, driver_module, tmp_path, temperature):
        args = parse_cli(driver_module, tmp_path, "--escott-temperature", temperature)
        with pytest.raises(ValueError, match=r"--escott-temperature must be > 0"):
            driver_module.validate_args(args)

    def test_match_plm_requires_a_reference_table(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--escott-temperature-mode", "match-plm")
        with pytest.raises(ValueError, match="requires --plm-reference-table"):
            driver_module.validate_args(args)

    def test_match_plm_with_a_reference_table_is_accepted(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path,
                         "--escott-temperature-mode", "match-plm",
                         "--plm-reference-table", str(tmp_path / "plm.csv"))
        driver_module.validate_args(args)  # existence is checked later, not here

    def test_equation_four_is_rejected_by_name(self, driver_module, tmp_path):
        # prescott.py's equation-4 branch is an unconditional sys.exit(-1); offering it
        # produces a mid-run death inside a subprocess.
        args = parse_cli(driver_module, tmp_path, "--equation-grid", "2,4")
        with pytest.raises(ValueError, match="equation 4 is not implemented upstream"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("grid", ["0", "6", "2,7"])
    def test_unsupported_equations_are_rejected(self, driver_module, tmp_path, grid):
        args = parse_cli(driver_module, tmp_path, "--equation-grid", grid)
        with pytest.raises(ValueError, match=r"--equation-grid must be a subset of 1,2,3,5"):
            driver_module.validate_args(args)

    @pytest.mark.parametrize("grid", ["1", "2", "3", "5", "1,2,3,5"])
    def test_supported_equations_are_accepted(self, driver_module, tmp_path, grid):
        driver_module.validate_args(parse_cli(driver_module, tmp_path, "--equation-grid", grid))

    def test_negative_coefficients_are_rejected(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5,-0.25")
        with pytest.raises(ValueError, match=r"--coefficient-grid values must be >= 0"):
            driver_module.validate_args(args)

    def test_zero_coefficient_is_allowed(self, driver_module, tmp_path):
        driver_module.validate_args(parse_cli(driver_module, tmp_path,
                                              "--coefficient-grid", "0,0.5"))


# =========================================================================== #
# 3. apply_prescott_defaults -- --test-mode's documented behaviour
# =========================================================================== #

class TestTestModeSemantics:
    """``--test-mode`` must limit how much data is READ, and nothing else."""

    def test_zero_max_records_becomes_an_unreachable_bound(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            parse_cli(driver_module, tmp_path, "--test-mode")
        )
        assert args.test_max_records == 10 ** 9

    def test_an_explicit_record_cap_is_preserved(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            parse_cli(driver_module, tmp_path, "--test-mode", "--test-max-records", "400")
        )
        assert args.test_max_records == 400

    def test_record_cap_is_untouched_when_test_mode_is_off(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(parse_cli(driver_module, tmp_path))
        assert args.test_max_records == 0

    @pytest.mark.parametrize("definition", ["bootstrap", "direct"])
    def test_test_mode_does_not_touch_the_trace_definition(
        self, driver_module, tmp_path, definition
    ):
        # The regression this pins: --test-mode used to force trace_definition='direct',
        # so the smoke test never exercised the production trace path and a wrong
        # --trace-top-fraction survived a full end-to-end run.
        args = driver_module.apply_prescott_defaults(
            parse_cli(driver_module, tmp_path, "--test-mode",
                      "--trace-definition", definition)
        )
        assert args.trace_definition == definition

    def test_test_mode_changes_nothing_except_the_record_cap(self, driver_module, tmp_path):
        """Whole-namespace diff: exactly one modelling-irrelevant key may move."""
        base = parse_cli(driver_module, tmp_path, "--test-mode", "--test-max-records", "400")
        before = dict(vars(base))
        after = dict(vars(driver_module.apply_prescott_defaults(base)))

        # apply_arg_defaults adds PLM-only keys and caps the worker count; those are
        # not modelling parameters and are added, not changed.
        added = set(after) - set(before)
        assert added <= {
            "plm_max_aa_length", "plm_max_nt_length", "use_global_plm_reference",
            "alignment_verify_max_cols", "rolling_identity_window",
            "observed_mutation_fasta", "observed_mutation_sequence_id",
            "observed_mutation_selection", "diagnostic_exports",
        }
        changed = {
            key for key in before
            if key not in ("scores_dir", "inputs_dir", "escott_workdir",
                           "prescott_ref_dir", "alpha_sweep_max_workers")
            and before[key] != after[key]
        }
        assert changed == set(), f"--test-mode moved {sorted(changed)}"

    def test_alpha_grid_is_not_coarsened_by_test_mode(self, driver_module, tmp_path):
        args = driver_module.apply_prescott_defaults(
            parse_cli(driver_module, tmp_path, "--test-mode",
                      "--alpha-start", "-1", "--alpha-stop", "1", "--alpha-step", "0.5")
        )
        np.testing.assert_allclose(
            driver_module.rma.parse_alpha_grid(args), [-1.0, -0.5, 0.0, 0.5, 1.0]
        )

    def test_resolve_test_target_count_grows_past_input_only_rows(self, driver_module):
        targets = [{"label": label} for label in LINEAGE_ORDER]
        # Row 1 is G.1, which is input-only: asking for 1 target must yield 2.
        assert driver_module.resolve_test_target_count(targets, 1, {"G.1"}) == 2
        assert driver_module.resolve_test_target_count(targets, 3, {"G.1"}) == 3

    def test_resolve_test_target_count_stops_at_the_guide_length(self, driver_module):
        targets = [{"label": "G.1"}, {"label": "G.1"}]
        assert driver_module.resolve_test_target_count(targets, 1, {"G.1"}) == 2

    def test_resolve_test_target_count_floors_at_one(self, driver_module):
        targets = [{"label": "K"}, {"label": "J.2.4"}]
        assert driver_module.resolve_test_target_count(targets, 0, set()) == 1


# =========================================================================== #
# 4. Small helpers
# =========================================================================== #

class TestSmallHelpers:
    @pytest.mark.parametrize("label,expected", [
        ("G.1", "G1"), ("J_int", "Jint"), ("J.2_int", "J2int"), ("J.2.4", "J24"), ("K", "K"),
    ])
    def test_variant_token_strips_dots_and_underscores(self, driver_module, label, expected):
        assert driver_module.variant_token(label) == expected

    def test_safe_key_matches_stage_ones_key(self, driver_module):
        assert driver_module.safe_key("J.2_int") == "J.2_int"
        assert driver_module.safe_key("A/B C") == "A-B_C"

    def test_parse_float_grid_tolerates_spaces_and_empty_chunks(self, driver_module):
        assert driver_module.parse_float_grid(" 0.25 , 0.5 ,, 1.0 ") == [0.25, 0.5, 1.0]
        assert driver_module.parse_float_grid("") == []

    def test_parse_int_grid_accepts_float_spellings(self, driver_module):
        assert driver_module.parse_int_grid("1, 2.0 ,3") == [1, 2, 3]
        assert driver_module.parse_int_grid("") == []

    def test_file_md5_is_the_real_digest_and_none_when_absent(self, driver_module, tmp_path):
        path = tmp_path / "abc.txt"
        path.write_bytes(b"abc")
        assert driver_module.file_md5(path) == hashlib.md5(b"abc").hexdigest()
        assert driver_module.file_md5(tmp_path / "nope.txt") is None

    def test_iqr_is_the_75_25_gap_and_nan_below_two_points(self, driver_module):
        assert driver_module._iqr(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.5)
        assert np.isnan(driver_module._iqr(np.array([np.nan, 1.0])))
        assert np.isnan(driver_module._iqr(np.array([])))

    @pytest.mark.parametrize("value,expected", [
        (None, None), (float("nan"), None), (np.nan, None), ("", None), ("abc", None),
        (2, 2.0), ("2", 2.0), (2.0, 2.0), (0, 0.0),
    ])
    def test_optional_number(self, driver_module, value, expected):
        assert driver_module._optional_number(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (None, None), (float("nan"), None), ("", None), ("   ", None),
        ("J.2.4", "J.2.4"), ("  K  ", "K"),
    ])
    def test_normalised_label(self, driver_module, value, expected):
        assert driver_module._normalised_label(value) == expected

    def test_require_stage1_script_names_the_missing_file(self, driver_module, monkeypatch, tmp_path):
        monkeypatch.setitem(driver_module.STAGE1_SCRIPTS, "prepare", tmp_path / "gone.py")
        with pytest.raises(RuntimeError, match="Stage-1 script missing"):
            driver_module.require_stage1_script("prepare")

    def test_require_stage1_script_returns_the_real_paths(self, driver_module):
        for kind in ("prepare", "jet", "escott"):
            assert driver_module.require_stage1_script(kind).exists()

    def test_stage1_environment_puts_the_interpreters_bin_first(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path,
                             "--prescott-python", str(Path(sys.executable)))
        env = driver_module.stage1_environment(args)
        assert env["PATH"].split(":")[0] == str(Path(sys.executable).resolve().parent)
        assert env["MPLBACKEND"] == "Agg"
        assert env["R_LIBS_USER"] == ""     # a stale user library must not shadow seqinr


# =========================================================================== #
# 5. Parent map resolution
# =========================================================================== #

class TestResolveParentMap:
    def test_default_preset_is_the_corrected_clade_evidence_ladder(self, driver_module, tmp_path):
        # Literal on the right-hand side: K descends from J.2.4, never J.2_int.
        assert driver_module.resolve_parent_map(parse_cli(driver_module, tmp_path)) == \
            CLADE_EVIDENCE_MAP
        assert CLADE_EVIDENCE_MAP == EXPECTED_PARENT_MAP  # agrees with the conftest pin

    def test_brief_as_stated_preset_restores_the_contested_edge(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--parent-map-preset", "brief_as_stated")
        resolved = driver_module.resolve_parent_map(args)
        assert resolved == BRIEF_AS_STATED_MAP
        assert resolved["K"] == "J.2_int"
        assert BRIEF_AS_STATED_MAP == EXPECTED_SENSITIVITY_PARENT_MAP

    def test_the_two_presets_disagree_on_exactly_one_edge(self, driver_module):
        presets = driver_module.parent_map_presets()
        differing = {child for child in presets["clade_evidence"]
                     if presets["brief_as_stated"].get(child) != presets["clade_evidence"][child]}
        assert differing == {"K"}
        assert CONTESTED_EDGE == ("K", "J.2.4", "J.2_int")

    def test_unknown_preset_raises(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        args.parent_map_preset = "made_up"
        with pytest.raises(ValueError, match="Unknown --parent-map-preset"):
            driver_module.resolve_parent_map(args)

    def test_explicit_edges_override_the_preset(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--parent-map", "K=J.2_int,J.2.4=J_int")
        resolved = driver_module.resolve_parent_map(args)
        assert resolved["K"] == "J.2_int"
        assert resolved["J.2.4"] == "J_int"
        assert resolved["J_int"] == "G.1"  # untouched edges survive

    def test_explicit_edges_may_add_a_new_child(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--parent-map", "L.9=K")
        assert driver_module.resolve_parent_map(args)["L.9"] == "K"

    def test_whitespace_and_empty_chunks_are_tolerated(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--parent-map", " K = J.2_int , ,")
        assert driver_module.resolve_parent_map(args)["K"] == "J.2_int"

    @pytest.mark.parametrize("spec", ["K", "K:J.2.4", "justtext"])
    def test_entries_without_an_equals_sign_are_rejected(self, driver_module, tmp_path, spec):
        args = parse_cli(driver_module, tmp_path, "--parent-map", spec)
        with pytest.raises(ValueError, match="must look like child=parent"):
            driver_module.resolve_parent_map(args)

    @pytest.mark.parametrize("spec", ["=J.2.4", "K=", " = "])
    def test_entries_with_an_empty_side_are_rejected(self, driver_module, tmp_path, spec):
        args = parse_cli(driver_module, tmp_path, "--parent-map", spec)
        with pytest.raises(ValueError, match="has an empty side"):
            driver_module.resolve_parent_map(args)

    @pytest.mark.parametrize("spec,offender", [
        ("K=K", "K"),                                   # self loop
        ("K=J.2.4,J.2.4=K", "K"),                       # two-cycle
        ("K=J.2.4,J.2.4=J.2_int,J.2_int=K", "K"),       # three-cycle
        ("J_int=K", "J_int"),                           # closes the whole ladder
    ])
    def test_cycles_are_rejected(self, driver_module, tmp_path, spec, offender):
        args = parse_cli(driver_module, tmp_path, "--parent-map", spec)
        with pytest.raises(ValueError, match="contains a cycle"):
            driver_module.resolve_parent_map(args)

    def test_a_linear_ladder_is_not_a_cycle(self, driver_module, tmp_path):
        # The production map is a chain of length 5; the walk must terminate.
        driver_module.resolve_parent_map(parse_cli(driver_module, tmp_path))

    def test_input_only_lineages_is_exactly_g1(self, driver_module):
        assert driver_module.input_only_lineages() == frozenset({"G.1"})


class TestSensitivityEdges:
    def test_default_run_contests_only_ks_parent(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {"K": "J.2_int"}

    def test_the_edge_flips_with_the_preset(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--parent-map-preset", "brief_as_stated")
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {"K": "J.2.4"}

    def test_disabled_sensitivity_yields_no_edges(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--no-parent-sensitivity")
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {}

    def test_an_explicit_override_that_matches_the_other_preset_removes_the_edge(
        self, driver_module, tmp_path
    ):
        # --parent-map K=J.2_int under clade_evidence *is* the other preset's answer,
        # so a sensitivity variant would be a byte-identical duplicate.
        args = parse_cli(driver_module, tmp_path, "--parent-map", "K=J.2_int")
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.sensitivity_edges(args, parent_map) == {}

    def test_effective_edges_are_restricted_to_evaluated_lineages(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        assert driver_module.effective_sensitivity_edges(args, parent_map, ["J_int"]) == {}
        assert driver_module.effective_sensitivity_edges(args, parent_map, EVALUABLE) == \
            {"K": "J.2_int"}

    def test_edge_spec_is_the_sorted_wire_format(self, driver_module):
        assert driver_module.sensitivity_edge_spec({"K": "J.2_int"}) == "K=J.2_int"
        assert driver_module.sensitivity_edge_spec({"K": "J.2_int", "J.2.4": "J_int"}) == \
            "J.2.4=J_int,K=J.2_int"
        assert driver_module.sensitivity_edge_spec({}) == ""


# =========================================================================== #
# 6. Variant naming
# =========================================================================== #

class TestVariantNaming:
    @pytest.mark.parametrize("equation,coefficient,k,parent,expected", [
        (2, 0.5, 1, "J.2.4", "PRESCOTT_eq2_c0p50_k1_parentJ24"),
        (2, 0.25, 1, "G.1", "PRESCOTT_eq2_c0p25_k1_parentG1"),
        (3, 1.0, 2, "J_int", "PRESCOTT_eq3_c1p00_k2_parentJint"),
        (5, 0.0, 10, "J.2_int", "PRESCOTT_eq5_c0p00_k10_parentJ2int"),
    ])
    def test_stage1_variant_name_is_byte_exact(
        self, driver_module, equation, coefficient, k, parent, expected
    ):
        assert driver_module.stage1_variant_name(equation, coefficient, k, parent) == expected

    def test_escott_tag_for_a_missing_equation(self, driver_module):
        assert driver_module.canonical_model_tag(None, None, None, None, None) == "ESCOTT"
        assert driver_module.canonical_model_tag(np.nan, 0.5, 1, "K", "K") == "ESCOTT"

    def test_primary_parent_gets_a_parent_free_model_tag(self, driver_module):
        # Otherwise one grid point splits into four single-lineage models and the
        # pooled alpha table averages one lineage per row while ESCOTT averages four.
        assert driver_module.canonical_model_tag(2, 0.5, 1, "J.2.4", "J.2.4") == \
            "PRESCOTT_eq2_c0p50_k1"

    def test_sensitivity_parent_keeps_the_suffix(self, driver_module):
        assert driver_module.canonical_model_tag(2, 0.5, 1, "J.2_int", "J.2.4") == \
            "PRESCOTT_eq2_c0p50_k1_parentJ2int"

    def test_tag_survives_the_tsv_round_trip_types(self, driver_module):
        assert driver_module.canonical_model_tag(2.0, 0.5, 1.0, "J.2.4", "J.2.4") == \
            "PRESCOTT_eq2_c0p50_k1"

    def test_unknown_resolved_parent_does_not_create_a_suffix(self, driver_module):
        assert driver_module.canonical_model_tag(2, 0.5, 1, "G.1", None) == \
            "PRESCOTT_eq2_c0p50_k1"


# =========================================================================== #
# 7. The requested design
# =========================================================================== #

class TestExpectedVariantPlan:
    def test_default_grid_shape(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        # 4 ESCOTT + 3 coefficients x (4 primary edges + 1 sensitivity edge) = 19
        assert len(plan) == 19
        assert sum(1 for e in plan if e["source_variant"] == "ESCOTT") == 4

    def test_escott_rows_carry_no_design_terms(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        escott = [e for e in driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
                  if e["source_variant"] == "ESCOTT"]
        assert [e["lineage"] for e in escott] == EVALUABLE
        for entry in escott:
            assert entry["parent_lineage"] is None
            assert entry["equation"] is None
            assert entry["coefficient"] is None
            assert entry["frequency_cutoff_k"] is None

    def test_primary_variant_names_are_exactly_predicted(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        primary = {e["lineage"]: e["source_variant"] for e in plan
                   if e["equation"] is not None
                   and e["parent_lineage"] == parent_map[e["lineage"]]}
        assert primary == EXPECTED_PRIMARY_VARIANTS_C050_K1

    def test_the_sensitivity_row_is_present_and_named_for_the_other_parent(
        self, driver_module, tmp_path
    ):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        sensitivity = [e for e in plan
                       if e["lineage"] == "K" and e["parent_lineage"] == "J.2_int"]
        assert len(sensitivity) == 1
        assert sensitivity[0]["source_variant"] == EXPECTED_SENSITIVITY_VARIANT_C050_K1

    def test_sensitivity_rows_vanish_for_a_lineage_that_is_not_evaluated(
        self, driver_module, tmp_path
    ):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, ["J_int"])
        assert len(plan) == 2                      # ESCOTT + one PRESCOTT edge
        assert {e["lineage"] for e in plan} == {"J_int"}

    @pytest.mark.parametrize("flags,expected_prescott_rows", [
        (["--coefficient-grid", "0.5"], 5),                       # 4 primary + 1 sens
        (["--coefficient-grid", "0.25,0.5"], 10),
        (["--coefficient-grid", "0.5", "--equation-grid", "2,3"], 10),
        (["--coefficient-grid", "0.5", "--frequency-cutoff-k", "1,2,3"], 15),
        (["--coefficient-grid", "0.5", "--no-parent-sensitivity"], 4),
    ])
    def test_every_grid_flag_multiplies_the_plan(
        self, driver_module, tmp_path, flags, expected_prescott_rows
    ):
        args = parse_cli(driver_module, tmp_path, *flags)
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        assert sum(1 for e in plan if e["equation"] is not None) == expected_prescott_rows

    def test_parent_map_override_reaches_the_plan(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path,
                         "--coefficient-grid", "0.5", "--parent-map", "K=J_int")
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        k_rows = [e for e in plan if e["lineage"] == "K" and e["equation"] is not None]
        assert {e["parent_lineage"] for e in k_rows} == {"J_int", "J.2_int"}
        assert any(e["source_variant"] == "PRESCOTT_eq2_c0p50_k1_parentJint" for e in k_rows)

    def test_lineage_key_uses_stage_ones_safe_key(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        assert {e["lineage_key"] for e in plan} == set(EVALUABLE)


class TestPlanEntryKey:
    def test_escott_key_ignores_everything_but_the_lineage(self, driver_module):
        assert driver_module.plan_entry_key({"lineage": "K", "equation": None}) == ("K", "ESCOTT")

    def test_prescott_key_carries_the_whole_design(self, driver_module):
        entry = {"lineage": "K", "equation": 2, "coefficient": 0.5,
                 "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"}
        assert driver_module.plan_entry_key(entry) == ("K", 2, 0.5, 1, "J.2.4")

    def test_key_is_stable_across_the_tsv_round_trip(self, driver_module):
        native = {"lineage": "K", "equation": 2, "coefficient": 0.5,
                  "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"}
        from_tsv = {"lineage": "K", "equation": 2.0, "coefficient": 0.5,
                    "frequency_cutoff_k": 1.0, "parent_lineage": "J.2.4"}
        assert driver_module.plan_entry_key(native) == driver_module.plan_entry_key(from_tsv)

    def test_nan_parent_and_none_parent_agree(self, driver_module):
        # The ESCOTT baseline's empty parent cell reads back as NaN.
        a = {"lineage": "K", "equation": 2, "coefficient": 0.5,
             "frequency_cutoff_k": 1, "parent_lineage": None}
        b = dict(a, parent_lineage=np.nan)
        assert driver_module.plan_entry_key(a) == driver_module.plan_entry_key(b)

    @pytest.mark.parametrize("changed", [
        {"equation": 3}, {"coefficient": 0.25}, {"frequency_cutoff_k": 2},
        {"parent_lineage": "J.2_int"}, {"lineage": "J.2.4"},
    ])
    def test_every_design_term_changes_the_key(self, driver_module, changed):
        base = {"lineage": "K", "equation": 2, "coefficient": 0.5,
                "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"}
        assert driver_module.plan_entry_key(base) != \
            driver_module.plan_entry_key({**base, **changed})

    def test_describe_plan_entry_is_readable(self, driver_module):
        entry = {"lineage": "K", "equation": 2, "coefficient": 0.5,
                 "frequency_cutoff_k": 1, "parent_lineage": "J.2.4"}
        assert driver_module.describe_plan_entry(entry) == \
            "eq2 c=0.5 k=1 parent=J.2.4 / K"
        assert driver_module.describe_plan_entry({"lineage": "K", "equation": None}) == \
            "ESCOTT / K"


# =========================================================================== #
# 8. The cached table
# =========================================================================== #

class TestScoreVariantsTable:
    def test_missing_table_is_an_empty_frame_not_an_error(self, driver_module, tmp_path):
        assert driver_module.load_score_variants_table(tmp_path).empty

    def test_a_table_without_the_required_columns_is_refused(self, driver_module, tmp_path):
        path = tmp_path / "score_variants.tsv"
        pd.DataFrame({"variant": ["ESCOTT"]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(RuntimeError, match=r"lacks the required column\(s\) \['lineage'\]"):
            driver_module.load_score_variants_table(tmp_path)

    def test_variant_plan_from_table_drops_lineages_outside_the_run(
        self, driver_module, tmp_path
    ):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        parent_map = driver_module.resolve_parent_map(args)
        entries = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        write_variants_table(tmp_path / "scores", entries)
        table = driver_module.load_score_variants_table(tmp_path / "scores")
        plan = driver_module.variant_plan_from_table(table, ["K"])
        assert {e["lineage"] for e in plan} == {"K"}
        assert len(plan) == 3      # ESCOTT + primary + sensitivity

    def test_variant_plan_from_table_normalises_nan_cells(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        parent_map = driver_module.resolve_parent_map(args)
        entries = driver_module.expected_variant_plan(args, parent_map, ["J_int"])
        write_variants_table(tmp_path / "scores", entries)
        table = driver_module.load_score_variants_table(tmp_path / "scores")
        escott = next(e for e in driver_module.variant_plan_from_table(table, ["J_int"])
                      if e["source_variant"] == "ESCOTT")
        assert escott["parent_lineage"] is None
        assert escott["equation"] is None
        assert escott["coefficient"] is None
        assert escott["frequency_cutoff_k"] is None
        assert escott["score_matrix_path"] is not None


class TestScoreMatrixPath:
    def test_conventional_path_wins_when_it_exists(self, driver_module, tmp_path):
        scores = tmp_path / "scores"
        scores.mkdir()
        conventional = scores / "K_ESCOTT_score_matrix.csv"
        conventional.write_text("x", encoding="utf-8")
        elsewhere = _tiny_file(tmp_path / "other" / "K_ESCOTT_score_matrix.csv")
        assert driver_module.score_matrix_path(scores, "K", "ESCOTT", str(elsewhere)) == \
            conventional

    def test_recorded_path_is_used_when_the_conventional_one_is_absent(
        self, driver_module, tmp_path
    ):
        scores = tmp_path / "scores"
        scores.mkdir()
        elsewhere = _tiny_file(tmp_path / "other" / "K_ESCOTT_score_matrix.csv")
        assert driver_module.score_matrix_path(scores, "K", "ESCOTT", str(elsewhere)) == \
            elsewhere

    def test_a_stale_recorded_path_falls_back_to_the_conventional_name(
        self, driver_module, tmp_path
    ):
        # A copied output tree records absolute paths into the tree it came from.
        scores = tmp_path / "scores"
        scores.mkdir()
        stale = "/nonexistent/elsewhere/K_ESCOTT_score_matrix.csv"
        assert driver_module.score_matrix_path(scores, "K", "ESCOTT", stale) == \
            scores / "K_ESCOTT_score_matrix.csv"

    def test_no_recorded_path_gives_the_conventional_name(self, driver_module, tmp_path):
        assert driver_module.score_matrix_path(tmp_path, "K", "ESCOTT", None) == \
            tmp_path / "K_ESCOTT_score_matrix.csv"


# =========================================================================== #
# 9. THE BLOCKER: reconcile_variant_plan
# =========================================================================== #

class TestReconcileVariantPlan:
    """The CLI is the authority; ``score_variants.tsv`` may only supply filenames.

    Each test here populates a scores dir from one design and then reconciles a
    DIFFERENT design against it, one CLI flag at a time.
    """

    @staticmethod
    def _cache(driver, tmp_path, *flags, evaluable=EVALUABLE, **kwargs):
        """Populate ``<tmp>/scores`` with the plan implied by ``flags``."""
        args = parse_cli(driver, tmp_path, *flags)
        parent_map = driver.resolve_parent_map(args)
        entries = driver.expected_variant_plan(args, parent_map, evaluable)
        write_variants_table(tmp_path / "scores", entries, **kwargs)
        return entries

    @staticmethod
    def _request(driver, tmp_path, *flags, evaluable=EVALUABLE):
        args = parse_cli(driver, tmp_path, *flags)
        parent_map = driver.resolve_parent_map(args)
        requested = driver.expected_variant_plan(args, parent_map, evaluable)
        table = driver.load_score_variants_table(tmp_path / "scores")
        return driver.reconcile_variant_plan(requested, table, tmp_path / "scores", evaluable)

    # ---- the fully-cached happy path ------------------------------------------------
    def test_an_exactly_matching_cache_needs_nothing(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5")
        assert missing == [] and ignored == []
        assert len(plan) == 9          # 4 ESCOTT + 4 primary + 1 sensitivity
        assert all(Path(entry["score_matrix_path"]).exists() for entry in plan)

    def test_plan_length_always_equals_the_requested_length(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5,1.0")
        for flags in (["--coefficient-grid", "0.5"],
                      ["--coefficient-grid", "0.25,0.5,1.0,2.0"],
                      ["--coefficient-grid", "0.5", "--equation-grid", "2,3"]):
            args = parse_cli(driver_module, tmp_path, *flags)
            parent_map = driver_module.resolve_parent_map(args)
            requested = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
            plan, _, _ = driver_module.reconcile_variant_plan(
                requested,
                driver_module.load_score_variants_table(tmp_path / "scores"),
                tmp_path / "scores", EVALUABLE,
            )
            assert len(plan) == len(requested), flags

    # ---- one flag at a time ----------------------------------------------------------
    def test_shrinking_the_coefficient_grid_drops_the_other_coefficients(
        self, driver_module, tmp_path
    ):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5,1.0")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5")
        assert missing == []
        coefficients = {driver_module._optional_number(e["coefficient"]) for e in plan}
        assert coefficients == {None, 0.5}
        # 2 dropped coefficients x (4 primary + 1 sensitivity) = 10 ignored rows
        assert len(ignored) == 10
        assert all("c=0.5 " not in text for text in ignored)

    def test_growing_the_coefficient_grid_forces_stage_one(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5,0.75")
        assert ignored == []
        assert len(missing) == 5
        assert all("c=0.75" in text for text in missing)
        assert len(plan) == 14

    def test_changing_the_equation_grid_forces_stage_one(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5", "--equation-grid", "2")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5",
                                               "--equation-grid", "3")
        assert len(missing) == 5 and all("eq3" in text for text in missing)
        assert len(ignored) == 5 and all("eq2" in text for text in ignored)
        assert {e["source_variant"] for e in plan if e["equation"] == 3} == {
            "PRESCOTT_eq3_c0p50_k1_parentG1", "PRESCOTT_eq3_c0p50_k1_parentJint",
            "PRESCOTT_eq3_c0p50_k1_parentJ2int", "PRESCOTT_eq3_c0p50_k1_parentJ24",
        }

    def test_changing_the_frequency_cutoff_k_forces_stage_one(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5",
                    "--frequency-cutoff-k", "1")
        _, missing, ignored = self._request(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5",
                                            "--frequency-cutoff-k", "2")
        assert len(missing) == 5 and all("k=2" in text for text in missing)
        assert len(ignored) == 5 and all("k=1" in text for text in ignored)

    def test_changing_the_parent_map_forces_stage_one(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5",
                                               "--parent-map", "J.2.4=J_int")
        # J.2.4 under J_int is a new combination, so stage 1 is required.
        assert missing == [
            "eq2 c=0.5 k=1 parent=J_int / J.2.4 [not in score_variants.tsv]"]
        # J.2.4 under J.2_int is NOT ignored: the override made the presets disagree
        # about J.2.4 too, so the old edge is now its sensitivity variant and is still
        # requested -- under a _parent-suffixed model tag rather than the primary one.
        assert ignored == []
        j24 = {e["parent_lineage"] for e in plan
               if e["lineage"] == "J.2.4" and e["equation"] is not None}
        assert j24 == {"J_int", "J.2_int"}

    def test_changing_the_parent_map_with_sensitivity_off_ignores_the_old_edge(
        self, driver_module, tmp_path
    ):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5",
                    "--no-parent-sensitivity")
        _, missing, ignored = self._request(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5",
                                            "--no-parent-sensitivity",
                                            "--parent-map", "J.2.4=J_int")
        assert missing == [
            "eq2 c=0.5 k=1 parent=J_int / J.2.4 [not in score_variants.tsv]"]
        assert ignored == ["eq2 c=0.5 k=1 parent=J.2_int / J.2.4"]

    def test_changing_the_parent_map_preset_forces_stage_one(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5",
                    "--parent-map-preset", "clade_evidence", "--no-parent-sensitivity")
        _, missing, ignored = self._request(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5",
                                            "--parent-map-preset", "brief_as_stated")
        # brief_as_stated wants K under J.2_int; the cache only has K under J.2.4, and
        # K under J.2.4 is now the *sensitivity* edge, so it is requested too.
        assert missing == [
            "eq2 c=0.5 k=1 parent=J.2_int / K [not in score_variants.tsv]"]
        assert ignored == []

    def test_turning_sensitivity_off_ignores_the_alternate_parent_rows(
        self, driver_module, tmp_path
    ):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5",
                                               "--no-parent-sensitivity")
        assert missing == []
        assert ignored == ["eq2 c=0.5 k=1 parent=J.2_int / K"]
        assert len(plan) == 8
        assert not any(e["lineage"] == "K" and e["parent_lineage"] == "J.2_int"
                       for e in plan)

    def test_turning_sensitivity_on_requests_the_alternate_parent(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5",
                    "--no-parent-sensitivity")
        _, missing, ignored = self._request(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5")
        assert missing == ["eq2 c=0.5 k=1 parent=J.2_int / K [not in score_variants.tsv]"]
        assert ignored == []

    # ---- disk state ------------------------------------------------------------------
    def test_a_cached_row_whose_matrix_is_gone_counts_as_missing(self, driver_module, tmp_path):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        victim = tmp_path / "scores" / "K_PRESCOTT_eq2_c0p50_k1_parentJ24_score_matrix.csv"
        assert victim.exists()
        victim.unlink()
        _, missing, ignored = self._request(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5")
        assert missing == [str(victim)]
        assert ignored == []

    def test_an_empty_scores_dir_makes_every_combination_missing(self, driver_module, tmp_path):
        (tmp_path / "scores").mkdir()
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5")
        assert len(missing) == len(plan) == 9
        assert ignored == []
        assert all(text.endswith("[not in score_variants.tsv]") for text in missing)

    def test_the_plan_adopts_stage_ones_recorded_matrix_paths(self, driver_module, tmp_path):
        # Stage 1 wrote the matrices somewhere else and recorded absolute paths; the
        # conventional name does not exist, so the recorded one must be used.
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5",
                    matrix_dir=tmp_path / "elsewhere")
        plan, missing, _ = self._request(driver_module, tmp_path,
                                         "--coefficient-grid", "0.5")
        assert missing == []
        assert all(str(entry["score_matrix_path"]).startswith(str(tmp_path / "elsewhere"))
                   for entry in plan)

    def test_a_missing_entry_still_carries_the_predicted_name_and_path(
        self, driver_module, tmp_path
    ):
        (tmp_path / "scores").mkdir()
        plan, _, _ = self._request(driver_module, tmp_path, "--coefficient-grid", "0.5")
        entry = next(e for e in plan if e["lineage"] == "K" and e["equation"] is not None
                     and e["parent_lineage"] == "J.2.4")
        assert entry["source_variant"] == "PRESCOTT_eq2_c0p50_k1_parentJ24"
        assert entry["score_matrix_path"] == str(
            tmp_path / "scores" / "K_PRESCOTT_eq2_c0p50_k1_parentJ24_score_matrix.csv"
        )

    def test_cached_rows_for_lineages_outside_the_run_are_neither_used_nor_reported(
        self, driver_module, tmp_path
    ):
        self._cache(driver_module, tmp_path, "--coefficient-grid", "0.5")
        plan, missing, ignored = self._request(driver_module, tmp_path,
                                               "--coefficient-grid", "0.5",
                                               evaluable=["J_int"])
        assert {e["lineage"] for e in plan} == {"J_int"}
        assert missing == [] and ignored == []


# =========================================================================== #
# 10. build_score_specs
# =========================================================================== #

def _specs_for(driver, tmp_path, *flags, evaluable=EVALUABLE):
    args = parse_cli(driver, tmp_path, *flags)
    parent_map = driver.resolve_parent_map(args)
    plan = driver.expected_variant_plan(args, parent_map, evaluable)
    return args, parent_map, driver.build_score_specs(args, plan, parent_map)


class TestBuildScoreSpecs:
    def test_one_model_per_grid_point_plus_escott_plus_the_sensitivity_model(
        self, driver_module, tmp_path
    ):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5")
        assert model_tags(specs) == [
            "ESCOTT",
            "PRESCOTT_eq2_c0p25_k1",
            "PRESCOTT_eq2_c0p25_k1_parentJ2int",
            "PRESCOTT_eq2_c0p50_k1",
            "PRESCOTT_eq2_c0p50_k1_parentJ2int",
        ]

    def test_a_primary_model_pools_every_lineage(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5")
        primary = next(s for s in specs if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1")
        assert primary["lineages"] == EVALUABLE
        assert primary["parent_by_lineage"] == CLADE_EVIDENCE_MAP
        assert primary["source_variant_by_lineage"] == EXPECTED_PRIMARY_VARIANTS_C050_K1

    def test_the_sensitivity_model_holds_only_its_own_lineage(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5")
        sensitivity = next(s for s in specs
                           if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1_parentJ2int")
        assert sensitivity["lineages"] == ["K"]
        assert sensitivity["parent_by_lineage"] == {"K": "J.2_int"}

    def test_epoch_fields_make_the_penalty_the_trajectory_axis(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5")
        by_tag = {str(s["model_tag"]): s for s in specs}
        assert by_tag["ESCOTT"]["epoch_label"] == "escott"
        assert by_tag["ESCOTT"]["epoch_value"] == 0.0
        assert by_tag["ESCOTT"]["model_display_label"] == "ESCOTT (no frequency term)"
        assert by_tag["PRESCOTT_eq2_c0p25_k1"]["epoch_label"] == "prescott_c0.25"
        assert by_tag["PRESCOTT_eq2_c0p25_k1"]["epoch_value"] == 0.25
        assert by_tag["PRESCOTT_eq2_c0p25_k1"]["model_display_label"] == "PRESCOTT c=0.25"
        sens = by_tag["PRESCOTT_eq2_c0p50_k1_parentJ2int"]
        assert sens["epoch_label"] == "prescott_c0.50_parentJ2int"
        assert sens["model_display_label"] == "PRESCOTT c=0.50 (parent J.2_int)"

    def test_specs_use_the_key_names_the_shared_code_expects(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5")
        for spec in specs:
            assert {"model_tag", "model_display_label", "base_model", "checkpoint_label",
                    "epoch_label", "epoch_value", "precomputed_plm_path",
                    "checkpoint_dir"} <= set(spec)
            assert spec["base_model"] == "ESCOTT"
            assert spec["precomputed_plm_path"] is None

    # ---- --score-variant: both spellings ---------------------------------------------
    def test_score_variant_accepts_the_model_tag(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5",
                                 "--score-variant", "PRESCOTT_eq2_c0p50_k1")
        assert model_tags(specs) == ["PRESCOTT_eq2_c0p50_k1"]

    def test_score_variant_accepts_the_stage1_variant_name(self, driver_module, tmp_path):
        # The fix: a user copies the `variant` column out of score_variants.tsv, which
        # always carries the _parent<TOK> suffix.
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5",
                                 "--score-variant", "PRESCOTT_eq2_c0p50_k1_parentG1")
        assert model_tags(specs) == ["PRESCOTT_eq2_c0p50_k1"]

    def test_score_variant_accepts_escott(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--score-variant", "ESCOTT")
        assert model_tags(specs) == ["ESCOTT"]

    def test_score_variant_accepts_a_mixture_of_both_spellings(self, driver_module, tmp_path):
        _, _, specs = _specs_for(
            driver_module, tmp_path, "--coefficient-grid", "0.25,0.5",
            "--score-variant", "ESCOTT",
            "--score-variant", "PRESCOTT_eq2_c0p25_k1_parentJ24",
        )
        assert model_tags(specs) == ["ESCOTT", "PRESCOTT_eq2_c0p25_k1"]

    def test_unknown_score_variant_lists_both_vocabularies(self, driver_module, tmp_path):
        with pytest.raises(ValueError) as excinfo:
            _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5",
                       "--score-variant", "PRESCOTT_eq9_c9p99_k9")

        message = str(excinfo.value)
        assert "match neither a model tag nor a stage-1 variant name" in message
        assert "Model tags     : ['ESCOTT'," in message
        assert "PRESCOTT_eq2_c0p50_k1_parentG1" in message   # the stage-1 vocabulary

    def test_a_stage1_variant_name_shared_by_two_models_selects_both(
        self, driver_module, tmp_path
    ):
        """DOCUMENTED SHARP EDGE, pinned so a change is deliberate.

        ``PRESCOTT_eq2_c0p50_k1_parentJ2int`` is J.2.4's PRIMARY variant name (its
        parent really is J.2_int) *and* K's SENSITIVITY variant name.  The alias
        table therefore matches two different models and both are kept.  That is a
        genuine ambiguity in the stage-1 vocabulary, not in the alias lookup: the
        same string names two different (lineage, parent) rows.
        """
        _, _, specs = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5",
                                 "--score-variant", "PRESCOTT_eq2_c0p50_k1_parentJ2int")
        assert model_tags(specs) == [
            "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int",
        ]

    def test_no_resolvable_variants_is_a_runtime_error(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        with pytest.raises(RuntimeError, match="No score variants resolved"):
            driver_module.build_score_specs(args, [], parent_map)

    def test_an_empty_equation_grid_leaves_only_escott(self, driver_module, tmp_path):
        _, _, specs = _specs_for(driver_module, tmp_path, "--equation-grid", "")
        assert model_tags(specs) == ["ESCOTT"]


# =========================================================================== #
# 11. Design signatures and keys
# =========================================================================== #

class TestDesignSignatures:
    def test_shared_signature_excludes_the_prescott_grids(self, driver_module, tmp_path):
        # Otherwise ESCOTT's cached alpha sweep is invalidated every time the
        # coefficient grid moves, for no reason at all.
        a = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5")
        b = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5,1.0",
                      "--equation-grid", "3", "--frequency-cutoff-k", "2")
        assert driver_module.shared_design_signature(a) == \
            driver_module.shared_design_signature(b)

    @pytest.mark.parametrize("flags", [
        ["--frequency-cutoff-mode", "fixed"],
        ["--frequency-cutoff", "-3.0"],
        ["--no-drop-parent-reversions"],
        ["--parent-min-count", "3"],
        ["--parent-min-depth", "80"],
        ["--parent-freq-max", "0.9"],
        ["--escott-temperature", "2.0"],
        ["--mutation-model", "H1N1"],
        ["--alpha-step", "0.5"],
        ["--no-filter-fixed-mutations"],
        ["--filter-singleton-mutations"],
        ["--skip-low-count-sites"],
        ["--min-obs-count", "5"],
        ["--expect-protein-diversity"],
    ])
    def test_every_shared_term_moves_the_shared_signature(self, driver_module, tmp_path, flags):
        base = driver_module.shared_design_signature(parse_cli(driver_module, tmp_path))
        changed = driver_module.shared_design_signature(parse_cli(driver_module, tmp_path, *flags))
        assert base != changed, flags

    def test_run_signature_adds_the_grids_and_the_parent_map(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--equation-grid", "3", "--frequency-cutoff-k", "2")
        parent_map = driver_module.resolve_parent_map(args)
        signature = driver_module.design_signature(args, parent_map, EVALUABLE)
        assert signature["prescott_equations"] == [3]
        assert signature["prescott_coefficients"] == [0.5]
        assert signature["frequency_cutoff_k"] == [2]
        assert signature["parent_map"] == CLADE_EVIDENCE_MAP
        assert signature["parent_sensitivity_edges"] == {"K": "J.2_int"}

    def test_design_key_is_a_stable_16_hex_digest(self, driver_module, tmp_path):
        args = parse_cli(driver_module, tmp_path)
        parent_map = driver_module.resolve_parent_map(args)
        signature = driver_module.design_signature(args, parent_map, EVALUABLE)
        key = driver_module.design_key(signature)
        assert re.fullmatch(r"[0-9a-f]{16}", key)
        assert key == driver_module.design_key(dict(reversed(list(signature.items()))))

    def test_model_key_separates_two_coefficients(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.25,0.5")
        keys = {str(s["model_tag"]): driver_module.model_design_key(args, s, parent_map)
                for s in specs}
        assert len(set(keys.values())) == len(keys)

    def test_model_key_ignores_the_grid_the_model_is_not_in(self, driver_module, tmp_path):
        # Shrinking --coefficient-grid must leave the surviving models' keys alone,
        # which is what makes per-model resumption work.
        args_a, map_a, specs_a = _specs_for(driver_module, tmp_path,
                                            "--coefficient-grid", "0.25,0.5")
        args_b, map_b, specs_b = _specs_for(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5")
        a = {str(s["model_tag"]): driver_module.model_design_key(args_a, s, map_a)
             for s in specs_a}
        b = {str(s["model_tag"]): driver_module.model_design_key(args_b, s, map_b)
             for s in specs_b}
        for tag in b:
            assert a[tag] == b[tag], tag

    def test_model_key_moves_when_the_parent_moves(self, driver_module, tmp_path):
        args_a, map_a, specs_a = _specs_for(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5")
        args_b, map_b, specs_b = _specs_for(driver_module, tmp_path,
                                            "--coefficient-grid", "0.5",
                                            "--parent-map", "J.2.4=J_int")
        key_a = driver_module.model_design_key(
            args_a, next(s for s in specs_a if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1"), map_a)
        key_b = driver_module.model_design_key(
            args_b, next(s for s in specs_b if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1"), map_b)
        assert key_a != key_b

    def test_model_signature_normalises_a_nan_parent(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        escott = next(s for s in specs if s["model_tag"] == "ESCOTT")
        with_nan = dict(escott, parent_by_lineage={k: np.nan for k in escott["parent_by_lineage"]})
        assert driver_module.model_design_key(args, escott, parent_map) == \
            driver_module.model_design_key(args, with_nan, parent_map)


# =========================================================================== #
# 12. Caching
# =========================================================================== #

def _write_model_tables(model_tables_dir: Path, model_tag: str) -> None:
    """The three files ``rma._load_cached_model_outputs`` insists on."""
    model_tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lineage": ["K"], "plm_prob": [0.05], "mut_prob": [0.01],
                  "depth": [10]}).to_csv(
        model_tables_dir / f"{model_tag}_combined_long_table.csv", index=False)
    for suffix in ("_alpha_sweep_fit_metrics.tsv",
                   "_alpha_sweep_fit_metrics_BY_LINEAGE.tsv"):
        pd.DataFrame({"alpha": [0.0], "site_top10pct_mutated_enrichment": [1.0],
                      "mut_flat_global_spearman_r": [0.5], "lineage": ["K"]}).to_csv(
            model_tables_dir / f"{model_tag}{suffix}", sep="\t", index=False)


def _panel_metadata_for(driver, args, specs, parent_map, *, parent_cell="real"):
    """A ``panel_metadata.tsv`` frame that a valid cache would have written."""
    rows = []
    for spec in specs:
        key = driver.model_design_key(args, spec, parent_map)
        for lineage in spec["lineages"]:
            parent = spec["parent_by_lineage"].get(lineage)
            if parent_cell == "nan":
                parent = np.nan
            rows.append({
                "model": str(spec["model_tag"]),
                "lineage": lineage,
                "parent_lineage": parent,
                "cache_version": driver.PRESCOTT_CACHE_VERSION,
                "design_key": key,
                "mutation_model": args.mutation_model,
                "escott_temperature": float(args.escott_temperature),
                "n_sequences": 10,
            })
    return pd.DataFrame(rows)


class TestModelCacheIsValid:
    @staticmethod
    def _setup(driver, tmp_path, *flags):
        args, parent_map, specs = _specs_for(driver, tmp_path, *flags)
        model_tables = tmp_path / "tables" / "per_model"
        for spec in specs:
            _write_model_tables(model_tables, str(spec["model_tag"]))
        metadata = _panel_metadata_for(driver, args, specs, parent_map)
        return args, parent_map, specs, metadata, model_tables

    def test_a_matching_cache_is_valid(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        for spec in specs:
            assert driver_module.model_cache_is_valid(metadata, args, spec, parent_map, tables)
        assert driver_module.prescott_cache_is_valid(metadata, args, specs, parent_map, tables)

    def test_escotts_empty_parent_cell_does_not_invalidate_the_cache(
        self, driver_module, tmp_path
    ):
        """The regression the driver agent caught: NaN vs None must compare equal.

        The ESCOTT baseline is conditioned on no parent, so its ``parent_lineage``
        cell is blank and pandas reads it back as NaN.  A naive ``str()`` comparison
        made ``'nan' != 'None'`` and recomputed ESCOTT's alpha sweep on every rerun.
        """
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        model_tables = tmp_path / "tables" / "per_model"
        escott = next(s for s in specs if s["model_tag"] == "ESCOTT")
        _write_model_tables(model_tables, "ESCOTT")
        metadata = _panel_metadata_for(driver_module, args, [escott], parent_map,
                                       parent_cell="nan")
        # Round-tripping through TSV is what actually produces the NaN.
        path = tmp_path / "panel_metadata.tsv"
        metadata.to_csv(path, sep="\t", index=False)
        metadata = pd.read_csv(path, sep="\t")
        assert metadata["parent_lineage"].isna().all()
        assert driver_module.model_cache_is_valid(
            metadata, args, escott, parent_map, model_tables)

    def test_force_recompute_invalidates_everything(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        args.force_recompute_scores = True
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_diagnostic_exports_invalidate_everything(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        args.diagnostic_exports = True
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_empty_metadata_is_not_a_cache(self, driver_module, tmp_path):
        args, parent_map, specs, _, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert not driver_module.model_cache_is_valid(
            pd.DataFrame(), args, specs[0], parent_map, tables)

    def test_metadata_without_a_model_column_is_not_a_cache(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert not driver_module.model_cache_is_valid(
            metadata.drop(columns=["model"]), args, specs[0], parent_map, tables)

    def test_a_model_with_no_rows_is_not_cached(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        target = str(specs[1]["model_tag"])
        assert not driver_module.model_cache_is_valid(
            metadata[metadata["model"] != target], args, specs[1], parent_map, tables)

    @pytest.mark.parametrize("column", ["cache_version", "design_key", "parent_lineage"])
    def test_a_missing_guard_column_invalidates(self, driver_module, tmp_path, column):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert not driver_module.model_cache_is_valid(
            metadata.drop(columns=[column]), args, specs[0], parent_map, tables)

    def test_a_stale_cache_version_invalidates(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        metadata["cache_version"] = driver_module.PRESCOTT_CACHE_VERSION - 1
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_a_stale_design_key_invalidates(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        metadata["design_key"] = "deadbeefdeadbeef"
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_a_changed_mutation_model_invalidates(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        args.mutation_model = "H1N1"
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_a_changed_temperature_invalidates(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        args.escott_temperature = 2.0
        assert not driver_module.model_cache_is_valid(
            metadata, args, specs[0], parent_map, tables)

    def test_a_changed_parent_invalidates(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        spec = next(s for s in specs if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1")
        metadata.loc[metadata["lineage"] == "K", "parent_lineage"] = "J.2_int"
        assert not driver_module.model_cache_is_valid(
            metadata, args, spec, parent_map, tables)

    def test_missing_per_model_tables_invalidate(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        (tables / "ESCOTT_alpha_sweep_fit_metrics.tsv").unlink()
        escott = next(s for s in specs if s["model_tag"] == "ESCOTT")
        assert not driver_module.model_cache_is_valid(
            metadata, args, escott, parent_map, tables)

    def test_whole_run_gate_is_the_conjunction(self, driver_module, tmp_path):
        args, parent_map, specs, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert driver_module.prescott_cache_is_valid(metadata, args, specs, parent_map, tables)
        # Break exactly one model; the whole-run gate must close.
        (tables / "PRESCOTT_eq2_c0p50_k1_combined_long_table.csv").unlink()
        assert not driver_module.prescott_cache_is_valid(metadata, args, specs, parent_map, tables)
        assert driver_module.model_cache_is_valid(
            metadata, args, next(s for s in specs if s["model_tag"] == "ESCOTT"),
            parent_map, tables)

    def test_no_specs_is_never_cached(self, driver_module, tmp_path):
        args, parent_map, _, metadata, tables = self._setup(
            driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert not driver_module.prescott_cache_is_valid(metadata, args, [], parent_map, tables)


# =========================================================================== #
# 13. Stage-1 command construction
# =========================================================================== #

class TestStage1Commands:
    @staticmethod
    def _run(driver, tmp_path, monkeypatch, *flags, evaluable=EVALUABLE):
        args = prepared_args(driver, tmp_path, "--prescott-python", sys.executable, *flags)
        parent_map = driver.resolve_parent_map(args)
        calls = capture_stage1_steps(driver, monkeypatch)
        driver.run_stage1(args, parent_map, evaluable, tmp_path / "diag")
        return args, {call["label"]: call["command"] for call in calls}, calls

    def test_a_missing_interpreter_is_refused_with_an_explanation(
        self, driver_module, tmp_path, monkeypatch
    ):
        args = prepared_args(driver_module, tmp_path,
                             "--prescott-python", str(tmp_path / "no-such-python"))
        parent_map = driver_module.resolve_parent_map(args)
        capture_stage1_steps(driver_module, monkeypatch)
        with pytest.raises(RuntimeError, match="does not exist"):
            driver_module.run_stage1(args, parent_map, ["J_int"], tmp_path / "diag")

    def test_the_step_order_is_prepare_jet_validate_escott(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, _, calls = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int", "K"])
        assert [call["label"] for call in calls] == [
            "prepare_inputs", "jet_surrogate:J_int", "jet_surrogate:K",
            "jet_surrogate:validate", "run_escott",
        ]

    def test_prepared_lineages_are_targets_plus_parents_plus_alternates(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["K"])
        # K's target panel, its primary parent J.2.4 and the alternate parent J.2_int.
        assert sorted(flag_values(commands["prepare_inputs"], "--only-lineage")) == \
            ["J.2.4", "J.2_int", "K"]

    def test_sensitivity_spec_goes_to_both_stage1_scripts(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["K"])
        assert flag_value(commands["prepare_inputs"], "--sensitivity-parent-map") == "K=J.2_int"
        assert flag_value(commands["run_escott"], "--sensitivity-parent-map") == "K=J.2_int"
        assert "--no-parent-sensitivity" not in commands["run_escott"]

    def test_sensitivity_off_is_forwarded_explicitly_to_run_escott(
        self, driver_module, tmp_path, monkeypatch
    ):
        # Otherwise stage C picks alternate edges up from inputs_manifest.json and an
        # earlier sensitivity run keeps emitting them after the flag is turned off.
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--no-parent-sensitivity", evaluable=["K"])
        assert "--sensitivity-parent-map" not in commands["prepare_inputs"]
        assert "--no-parent-sensitivity" in commands["run_escott"]

    def test_an_edge_whose_child_is_not_evaluated_is_not_forwarded(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        assert "--sensitivity-parent-map" not in commands["prepare_inputs"]
        assert "--no-parent-sensitivity" in commands["run_escott"]

    def test_trace_top_fraction_is_not_forwarded_unless_set(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        assert "--trace-top-fraction" not in commands["jet_surrogate:J_int"]
        assert "--trace-top-fraction" not in commands["jet_surrogate:validate"]
        assert "--max-zero-trace-fraction" not in commands["jet_surrogate:J_int"]

    def test_trace_top_fraction_is_forwarded_when_set(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--trace-top-fraction", "0.75",
                                   "--max-zero-trace-fraction", "0.2",
                                   evaluable=["J_int"])
        assert flag_value(commands["jet_surrogate:J_int"], "--trace-top-fraction") == "0.75"
        assert flag_value(commands["jet_surrogate:J_int"], "--max-zero-trace-fraction") == "0.2"
        assert flag_value(commands["jet_surrogate:validate"], "--trace-top-fraction") == "0.75"

    def test_the_validation_table_lands_in_the_runs_diagnostics_dir(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        validate = commands["jet_surrogate:validate"]
        assert "--validate-only" in validate
        assert flag_value(validate, "--out-validation") == str(
            tmp_path / "diag" / driver_module.JET_VALIDATION_BASENAME)
        assert (tmp_path / "diag").is_dir()

    def test_validation_can_be_switched_off(self, driver_module, tmp_path, monkeypatch):
        _, _, calls = self._run(driver_module, tmp_path, monkeypatch,
                                "--no-jet-validation", evaluable=["J_int"])
        assert "jet_surrogate:validate" not in [call["label"] for call in calls]

    def test_diagnostics_dir_is_passed_to_run_escott(self, driver_module, tmp_path, monkeypatch):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        assert flag_value(commands["run_escott"], "--diagnostics-dir") == str(tmp_path / "diag")

    @pytest.mark.parametrize("flag,expected", [
        (None, "--drop-parent-reversions"),
        ("--no-drop-parent-reversions", "--no-drop-parent-reversions"),
    ])
    def test_drop_parent_reversions_is_pinned_in_both_directions(
        self, driver_module, tmp_path, monkeypatch, flag, expected
    ):
        extra = [flag] if flag else []
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, *extra,
                                   evaluable=["J_int"])
        assert expected in commands["prepare_inputs"]
        opposite = ("--no-drop-parent-reversions" if expected == "--drop-parent-reversions"
                    else "--drop-parent-reversions")
        assert opposite not in commands["prepare_inputs"]

    @pytest.mark.parametrize("flags,expected", [
        ([], ["--leakage-check", "--purge-leakage"]),
        (["--no-leakage-check"], ["--no-leakage-check", "--purge-leakage"]),
        (["--no-purge-leakage"], ["--leakage-check", "--no-purge-leakage"]),
    ])
    def test_leakage_booleans_are_forwarded_explicitly_both_ways(
        self, driver_module, tmp_path, monkeypatch, flags, expected
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, *flags,
                                   evaluable=["J_int"])
        for token in expected:
            assert token in commands["prepare_inputs"]

    def test_unset_leakage_thresholds_are_not_forwarded(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        for flag in ("--leakage-min-identity", "--leakage-max-hamming",
                     "--leakage-min-coverage", "--leakage-coverage-basis",
                     "--leakage-max-removed-fraction", "--leakage-min-depth-after",
                     "--leakage-threads", "--blast-task", "--fail-on-leakage"):
            assert flag not in commands["prepare_inputs"], flag

    def test_set_leakage_thresholds_are_forwarded_verbatim(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(
            driver_module, tmp_path, monkeypatch,
            "--leakage-min-identity", "none", "--leakage-max-hamming", "5",
            "--leakage-min-coverage", "90", "--blast-task", "blastp",
            "--fail-on-leakage", evaluable=["J_int"])
        prepare = commands["prepare_inputs"]
        assert flag_value(prepare, "--leakage-min-identity") == "none"
        assert flag_value(prepare, "--leakage-max-hamming") == "5"
        assert flag_value(prepare, "--leakage-min-coverage") == "90.0"
        assert flag_value(prepare, "--blast-task") == "blastp"
        assert "--fail-on-leakage" in prepare

    def test_test_mode_skips_only_the_prescott_parity_run(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, plain, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        assert "--prescott-ref-dir" in plain["run_escott"]
        _, test, _ = self._run(driver_module, tmp_path, monkeypatch, "--test-mode",
                               evaluable=["J_int"])
        assert "--prescott-ref-dir" not in test["run_escott"]
        # and nothing else about the escott command changed
        assert "--coefficient-grid" in test["run_escott"]
        assert flag_value(test["run_escott"], "--coefficient-grid") == "0.25,0.5,1.0"

    def test_the_grids_reach_run_escott_verbatim(self, driver_module, tmp_path, monkeypatch):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--coefficient-grid", "0.5,0.75",
                                   "--equation-grid", "3",
                                   "--frequency-cutoff-k", "2,4",
                                   evaluable=["J_int", "K"])
        escott = commands["run_escott"]
        assert flag_value(escott, "--coefficient-grid") == "0.5,0.75"
        assert flag_value(escott, "--equation-grid") == "3"
        assert flag_value(escott, "--frequency-cutoff-k") == "2,4"
        assert flag_values(escott, "--lineage") == ["J_int", "K"]

    def test_explicit_parent_map_reaches_both_scripts(self, driver_module, tmp_path, monkeypatch):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--parent-map", "K=J_int", evaluable=["K"])
        assert flag_value(commands["prepare_inputs"], "--parent-map") == "K=J_int"
        assert flag_value(commands["run_escott"], "--parent-map") == "K=J_int"

    def test_force_recompute_is_forwarded_to_all_three(
        self, driver_module, tmp_path, monkeypatch
    ):
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--force-recompute-scores", evaluable=["J_int"])
        assert "--force" in commands["prepare_inputs"]
        assert "--force" in commands["jet_surrogate:J_int"]
        assert "--force" in commands["run_escott"]

    def test_extra_args_are_appended(self, driver_module, tmp_path, monkeypatch):
        # `=` rather than a space: argparse would otherwise read the leading `--` as a
        # flag of its own, which is exactly how a user gets bitten too.
        _, commands, _ = self._run(driver_module, tmp_path, monkeypatch,
                                   "--prepare-args=--no-extra-structure",
                                   "--jet-args=--verbose",
                                   "--escott-args=--alphabet protein",
                                   evaluable=["J_int"])
        assert commands["prepare_inputs"][-1] == "--no-extra-structure"
        assert commands["jet_surrogate:J_int"][-1] == "--verbose"
        assert commands["run_escott"][-2:] == ["--alphabet", "protein"]

    def test_sasa_context_selects_the_context_pdb(self, driver_module, tmp_path, monkeypatch):
        _, trimer, _ = self._run(driver_module, tmp_path, monkeypatch, evaluable=["J_int"])
        _, monomer, _ = self._run(driver_module, tmp_path, monkeypatch,
                                  "--sasa-context", "monomer", evaluable=["J_int"])
        jet_t, jet_m = trimer["jet_surrogate:J_int"], monomer["jet_surrogate:J_int"]
        assert flag_value(jet_t, "--context-pdb").endswith("6WXB_trimer_qnum.pdb")
        assert flag_value(jet_m, "--context-pdb") == flag_value(jet_m, "--pdb")
        assert flag_value(jet_m, "--pdb").endswith("6WXB_chainA_qnum.pdb")


# =========================================================================== #
# 14. Structure / manifest resolution
# =========================================================================== #

class TestStructureResolution:
    def test_an_empty_manifest_yields_no_entry_rather_than_an_error(self, driver_module):
        assert driver_module.resolve_structure_entry({}, "primary") == {}
        assert driver_module.resolve_structure_entry({"structures": {}}, "extra") == {}

    def test_a_missing_role_is_a_hard_error_naming_what_is_available(self, driver_module):
        manifest = {"structures": {"primary": {"monomer": {"path": "/tmp/a.pdb"}}}}
        with pytest.raises(RuntimeError) as excinfo:
            driver_module.resolve_structure_entry(manifest, "extra")
        message = str(excinfo.value)
        assert "--structure-role 'extra' is not present" in message
        assert "available: ['primary']" in message
        assert "--no-extra-structure" in message

    def test_the_named_role_is_returned(self, driver_module):
        entry = {"monomer": {"path": "/tmp/a.pdb"}, "coverage_fraction": 1.0}
        manifest = {"structures": {"primary": {}, "extra": entry}}
        with pytest.raises(RuntimeError):
            driver_module.resolve_structure_entry(manifest, "primary")  # empty dict role
        assert driver_module.resolve_structure_entry(manifest, "extra") == entry

    def test_stage1_paths_prefers_the_manifest_over_the_literal_fallbacks(
        self, driver_module, tmp_path
    ):
        manifest = {
            "lineage_msas": {"K": {"path": "/from/manifest/msa.fasta"}},
            "queries": {"K": {"path": "/from/manifest/query.fasta"}},
            "structures": {"primary": {"monomer": {"path": "/from/manifest/mono.pdb"},
                                       "trimer": {"path": "/from/manifest/tri.pdb"}}},
        }
        paths = driver_module.stage1_paths(tmp_path / "inputs", "K", manifest, "primary")
        assert paths["msa"] == Path("/from/manifest/msa.fasta")
        assert paths["query"] == Path("/from/manifest/query.fasta")
        assert paths["chain_pdb"] == Path("/from/manifest/mono.pdb")
        assert paths["trimer_pdb"] == Path("/from/manifest/tri.pdb")
        # These are always conventional; stage B names them, not the manifest.
        assert paths["jet"] == tmp_path / "inputs" / "jet" / "K_surrogate_jet.res"
        assert paths["jet_manifest"] == tmp_path / "inputs" / "jet" / "K_jet_manifest.json"

    def test_stage1_paths_falls_back_before_prepare_inputs_has_run(
        self, driver_module, tmp_path
    ):
        paths = driver_module.stage1_paths(tmp_path / "inputs", "K", {}, "primary")
        assert paths["msa"] == tmp_path / "inputs" / "msa" / "msa_K.fasta"
        assert paths["chain_pdb"].name == "6WXB_chainA_qnum.pdb"

    def test_read_inputs_manifest_is_empty_when_absent(self, driver_module, tmp_path):
        assert driver_module.read_inputs_manifest(tmp_path) == {}

    def test_structure_record_before_stage_one_says_so(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path)
        record = driver_module.resolve_structure_record(args, {})
        assert record["structure_resolved_from_inputs_manifest"] is False
        assert record["structure_source_path"] == str(args.structure)
        assert record["structure_source_md5"] == driver_module.file_md5(args.structure)
        assert record["structure_monomer_path"] is None

    def test_structure_record_reports_what_the_surrogate_actually_read(
        self, driver_module, tmp_path
    ):
        monomer = _tiny_file(tmp_path / "mono.pdb", "ATOM\n")
        manifest = {"structures": {"extra": {
            "source_path": "/src/contemporary.pdb", "source_md5": "abc",
            "monomer": {"path": str(monomer)}, "trimer": {"path": None},
            "coverage_fraction": 1.0, "n_covered": 566, "offset_identity": 0.99,
        }}}
        args = prepared_args(driver_module, tmp_path, "--structure-role", "extra")
        record = driver_module.resolve_structure_record(args, manifest)
        assert record["structure_role"] == "extra"
        assert record["structure_source_path"] == "/src/contemporary.pdb"
        assert record["structure_monomer_md5"] == driver_module.file_md5(monomer)
        assert record["structure_n_covered"] == 566
        assert record["structure_resolved_from_inputs_manifest"] is True


class TestLeakageManifestRecord:
    def test_no_leakage_block_is_reported_as_unaudited(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path)
        record = driver_module.leakage_manifest_record(args, {})
        assert record["leakage_stage_ran"] is False
        assert record["leakage_status"] is None
        assert "UNAUDITED" in record["leakage_note"]
        # The request is still recorded, so the two are distinguishable.
        assert record["leakage_check_requested"] is True
        assert record["purge_leakage_requested"] is True

    def test_the_stage1_record_wins_over_the_requested_flags(self, driver_module, tmp_path):
        # A cached inputs tree may have been purged at thresholds this run did not ask
        # for; args describes a purge that did not happen.
        args = prepared_args(driver_module, tmp_path, "--no-purge-leakage")
        manifest = {"leakage": {
            "status": "clean",
            "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                           "min_coverage": 95.0, "coverage_basis": "both"},
            "purge": True,
            "report_dir": "/inputs/leakage",
            "purges": {"K": {"depth_before": 6434, "n_removed": 0, "depth_after": 6434,
                             "removed_fraction": 0.0, "query_exempted": True}},
            "checks": {"B_parent_vs_target": {"K": {
                "parent": "J.2.4",
                "accessions": {"n_shared_accessions": 0, "n_shared_exact_sequences": 0},
                "n_flagged": 8}}},
        }}
        record = driver_module.leakage_manifest_record(args, manifest)
        assert record["purge_leakage_requested"] is False    # what this run asked for
        assert record["leakage_purge_applied"] is True       # what stage 1 actually did
        assert record["leakage_stage_ran"] is True
        assert record["leakage_per_target"]["K"]["depth_after"] == 6434
        assert record["leakage_per_target"]["K"]["query_would_have_been_purged"] is True
        assert record["leakage_parent_vs_target"]["K"]["parent"] == "J.2.4"
        assert record["leakage_parent_vs_target"]["K"]["n_flagged"] == 8


# =========================================================================== #
# 15. The manifest must describe the models actually produced
# =========================================================================== #

class TestManifestDescribesTheModelsProduced:
    @staticmethod
    def _manifest(driver, tmp_path, args, parent_map, specs, variants_table):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        driver.save_run_manifest(args, output_dir, [], parent_map, specs,
                                 variants_table, EVALUABLE, {})
        return json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

    def test_model_specs_carry_the_same_design_key_the_cache_guard_compares(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.25,0.5")
        args = driver_module.apply_prescott_defaults(args)
        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs,
                                  pd.DataFrame())
        recorded = {entry["model_tag"]: entry["design_key"] for entry in manifest["model_specs"]}
        assert recorded == {str(s["model_tag"]): driver_module.model_design_key(args, s, parent_map)
                            for s in specs}

    def test_a_shrunken_grid_is_not_described_by_the_old_cache(self, driver_module, tmp_path):
        """The blocker, at the manifest end.

        The cache holds three coefficients; the CLI asks for one.  The manifest must
        describe one, not three -- it used to record the requested grid while the run
        analysed the cached one.
        """
        # populate the cache from the wide grid
        wide = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5,1.0")
        wide_map = driver_module.resolve_parent_map(wide)
        write_variants_table(
            tmp_path / "scores",
            driver_module.expected_variant_plan(wide, wide_map, EVALUABLE),
        )
        # ask for one coefficient
        args = driver_module.apply_prescott_defaults(
            parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                      output_dir=tmp_path / "out2"))
        parent_map = driver_module.resolve_parent_map(args)
        requested = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        table = driver_module.load_score_variants_table(tmp_path / "scores")
        plan, missing, ignored = driver_module.reconcile_variant_plan(
            requested, table, tmp_path / "scores", EVALUABLE)
        assert missing == [] and len(ignored) == 10
        specs = driver_module.build_score_specs(args, plan, parent_map)

        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs, table)
        assert manifest["prescott_coefficients"] == [0.5]
        assert [entry["model_tag"] for entry in manifest["model_specs"]] == [
            "ESCOTT", "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int"]
        assert not any("c0p25" in entry["model_tag"] or "c1p00" in entry["model_tag"]
                       for entry in manifest["model_specs"])

    def test_the_manifest_is_a_superset_of_the_plm_runs_keys(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs,
                                  pd.DataFrame())
        for key in ("analysis_mode", "mutation_model", "output_dir", "alpha_start",
                    "alpha_stop", "alpha_step", "alpha_grid", "scatter_alphas",
                    "test_mode", "test_max_targets", "test_max_records",
                    "filter_fixed_mutations", "filter_singleton_mutations",
                    "skip_low_count_sites", "min_obs_count", "diagnostic_exports",
                    "alignment_verify_max_cols", "rolling_identity_window",
                    "observed_mutation_fasta", "observed_mutation_sequence_id",
                    "observed_mutation_selection", "targets"):
            assert key in manifest, key

    def test_the_manifest_records_both_the_requested_and_effective_trace_fraction(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs,
                                  pd.DataFrame())
        assert manifest["trace_top_fraction_requested"] is None
        assert manifest["trace_top_fraction"] == 0.90
        assert manifest["max_zero_trace_fraction"] is None

    def test_drop_parent_reversions_is_read_back_from_stage_one(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        inputs_dir = Path(args.inputs_dir)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        # stage 1 says False; the CLI default says True.  Stage 1 built the files.
        (inputs_dir / "inputs_manifest.json").write_text(json.dumps({
            "args": {"drop_parent_reversions": False},
            "frequency": {"K": {"n_parent_reversion_mutants": 7}},
        }), encoding="utf-8")
        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs,
                                  pd.DataFrame())
        assert manifest["drop_parent_reversions"] is False
        assert manifest["n_parent_reversion_mutants_dropped"] == {"K": 7}

    def test_score_source_is_escott_only_when_escott_is_the_only_model(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path, "--equation-grid", "")
        args = driver_module.apply_prescott_defaults(args)
        manifest = self._manifest(driver_module, tmp_path, args, parent_map, specs,
                                  pd.DataFrame())
        assert manifest["score_source"] == "escott"

        args2, map2, specs2 = _specs_for(driver_module, tmp_path, "--coefficient-grid", "0.5")
        args2 = driver_module.apply_prescott_defaults(args2)
        args2.output_dir = tmp_path / "out_prescott"
        manifest2 = self._manifest(driver_module, tmp_path, args2, map2, specs2,
                                   pd.DataFrame())
        assert manifest2["score_source"] == "prescott"

    def test_sensitivity_edges_are_recorded_both_as_declared_and_as_applied(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5",
                                             evaluable=["J_int"])
        args = driver_module.apply_prescott_defaults(args)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), ["J_int"], {})
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent_sensitivity_edges"] == {"K": "J.2_int"}   # preset property
        assert manifest["parent_sensitivity_edges_applied"] == {}         # run property


# =========================================================================== #
# 16. CAVEATS.md
# =========================================================================== #

class TestCaveats:
    @staticmethod
    def _write(driver, tmp_path, *flags, evaluable=EVALUABLE, jet_manifests=None):
        args, parent_map, specs = _specs_for(driver, tmp_path, *flags, evaluable=evaluable)
        args = driver.apply_prescott_defaults(args)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return args, driver.write_caveats(args, output_dir, parent_map, specs, evaluable,
                                          jet_manifests)

    def test_the_file_is_written_and_names_the_resolved_parent_map(
        self, driver_module, tmp_path
    ):
        args, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert (Path(args.output_dir) / "CAVEATS.md").read_text(encoding="utf-8") == text
        assert '"K": "J.2.4"' in text
        assert "preset: clade_evidence" in text

    def test_sensitivity_off_is_stated_plainly(self, driver_module, tmp_path):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                              "--no-parent-sensitivity")
        assert "OFF (--no-parent-sensitivity)" in text

    def test_sensitivity_with_no_applicable_edge_says_which_edge_is_untested(
        self, driver_module, tmp_path
    ):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                              evaluable=["J_int"])
        assert "the presets agree on every lineage evaluated here" in text
        assert '{"K": "J.2_int"}' in text

    def test_a_scored_sensitivity_model_is_named(self, driver_module, tmp_path):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert "were scored as separate model rows" in text
        assert "PRESCOTT_eq2_c0p50_k1_parentJ2int" in text

    def test_a_requested_but_unproduced_sensitivity_is_called_untested(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        primary_only = [s for s in specs
                        if not str(s["model_tag"]).rsplit("_", 1)[-1].startswith("parent")]
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        text = driver_module.write_caveats(args, output_dir, parent_map, primary_only,
                                           EVALUABLE, {})
        assert "treat the parent choice as UNTESTED" in text

    def test_jet_validation_status_tracks_the_file_and_the_flag(self, driver_module, tmp_path):
        _, absent = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert "NOT PRESENT" in absent

        args, _ = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                              "--no-jet-validation")
        _, off = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                             "--no-jet-validation")
        assert "NOT RUN (--no-jet-validation)" in off

        args, _ = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        diagnostics = Path(args.output_dir) / "tables" / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        _tiny_file(diagnostics / driver_module.JET_VALIDATION_BASENAME, "x\n")
        text = driver_module.write_caveats(
            args, Path(args.output_dir),
            driver_module.resolve_parent_map(args),
            [{"model_tag": "ESCOTT"}], EVALUABLE, {})
        assert f"tables/diagnostics/{driver_module.JET_VALIDATION_BASENAME}" in text
        assert "NOT PRESENT" not in text

    def test_parity_status_explains_the_test_mode_skip(self, driver_module, tmp_path):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                              "--test-mode")
        assert "SKIPPED in --test-mode" in text

    def test_zero_trace_counts_are_quoted_from_the_jet_manifests(self, driver_module, tmp_path):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5",
                              jet_manifests={"K": {"n_zero_trace_columns": 18,
                                                   "frac_zero_trace_columns": 0.0318}})
        assert "K 18 (3.2%)" in text

    def test_absent_jet_manifests_say_so_rather_than_printing_zero(
        self, driver_module, tmp_path
    ):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert "not recorded (stage 1 not run this pass)" in text

    def test_unaudited_leakage_is_stated_in_the_headline(self, driver_module, tmp_path):
        _, text = self._write(driver_module, tmp_path, "--coefficient-grid", "0.5")
        assert "NOT AUDITED -- no leakage record in inputs_manifest.json" in text


# =========================================================================== #
# 17. run_analysis's planning half
# =========================================================================== #

class TestRunAnalysisPlanning:
    """End-to-end planning: the CLI beats the cache, or the run stops."""

    def test_a_fully_cached_matching_design_does_not_run_stage_one(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, EVALUABLE))
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert result["stage1_calls"] == []
        assert model_tags(result["specs"]) == [
            "ESCOTT", "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int"]
        assert "Stage 1 needed" not in capsys.readouterr().out

    def test_input_only_lineages_are_never_evaluated(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, EVALUABLE))
        run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        printed = capsys.readouterr().out
        assert "Evaluable lineages: ['J_int', 'J.2_int', 'J.2.4', 'K']" in printed

    def test_a_target_with_no_parent_edge_is_refused(
        self, driver_module, tmp_path, monkeypatch
    ):
        args = parse_cli(driver_module, tmp_path, output_dir=tmp_path / "out")
        result = run_planning(driver_module, args, monkeypatch, targets=["Z.9"])
        assert isinstance(result["error"], ValueError)
        assert "No basal lineage defined for ['Z.9']" in str(result["error"])

    def test_an_all_input_only_run_is_refused(self, driver_module, tmp_path, monkeypatch):
        args = parse_cli(driver_module, tmp_path, output_dir=tmp_path / "out")
        result = run_planning(driver_module, args, monkeypatch, targets=["G.1"])
        assert isinstance(result["error"], RuntimeError)
        assert "nothing to score" in str(result["error"])

    def test_no_targets_at_all_is_refused(self, driver_module, tmp_path, monkeypatch):
        args = parse_cli(driver_module, tmp_path, output_dir=tmp_path / "out")
        result = run_planning(driver_module, args, monkeypatch, targets=[])
        assert isinstance(result["error"], RuntimeError)
        assert "No targets resolved" in str(result["error"])

    def test_a_changed_grid_on_a_rerun_forces_stage_one_for_the_new_points_only(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(cached, cached_map, EVALUABLE))

        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5,0.75",
                         output_dir=out)

        def _stage1(a, parent_map, evaluable, diagnostics_dir):
            # stage 1 does its job: it appends the new grid point
            new = [e for e in driver_module.expected_variant_plan(a, parent_map, evaluable)
                   if driver_module._optional_number(e["coefficient"]) == 0.75]
            existing = pd.read_csv(Path(a.scores_dir) / "score_variants.tsv", sep="\t")
            write_variants_table(Path(a.scores_dir), new)
            merged = pd.concat(
                [existing, pd.read_csv(Path(a.scores_dir) / "score_variants.tsv", sep="\t")],
                ignore_index=True)
            merged.to_csv(Path(a.scores_dir) / "score_variants.tsv", sep="\t", index=False)

        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER,
                              stage1=_stage1)
        assert result["error"] is None
        assert len(result["stage1_calls"]) == 1
        assert model_tags(result["specs"]) == [
            "ESCOTT",
            "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int",
            "PRESCOTT_eq2_c0p75_k1", "PRESCOTT_eq2_c0p75_k1_parentJ2int",
        ]
        printed = capsys.readouterr().out
        assert "Stage 1 needed (5 requested score matrix/matrices not available)" in printed

    def test_a_shrunken_grid_is_reported_as_ignored_and_never_analysed(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        wide = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5,1.0",
                         output_dir=out)
        wide_map = driver_module.resolve_parent_map(wide)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(wide, wide_map, EVALUABLE))

        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["stage1_calls"] == []
        assert model_tags(result["specs"]) == [
            "ESCOTT", "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ2int"]
        printed = capsys.readouterr().out
        assert "Ignoring 10 cached score variant(s) outside the requested design" in printed

    @pytest.mark.parametrize("flags,expected_tag", [
        (["--equation-grid", "3"], "PRESCOTT_eq3_c0p50_k1"),
        (["--frequency-cutoff-k", "4"], "PRESCOTT_eq2_c0p50_k4"),
        (["--coefficient-grid", "0.5,0.75"], "PRESCOTT_eq2_c0p75_k1"),
    ])
    def test_every_design_flag_beats_the_cached_table(
        self, driver_module, tmp_path, monkeypatch, flags, expected_tag
    ):
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(cached, cached_map, EVALUABLE))

        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         *flags, output_dir=out)

        def _stage1(a, parent_map, evaluable, diagnostics_dir):
            write_variants_table(
                Path(a.scores_dir),
                driver_module.expected_variant_plan(a, parent_map, evaluable))

        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER,
                              stage1=_stage1)
        assert result["error"] is None
        assert len(result["stage1_calls"]) == 1
        assert expected_tag in model_tags(result["specs"])

    def test_flipping_the_preset_swaps_primary_and_sensitivity_without_recomputing(
        self, driver_module, tmp_path, monkeypatch
    ):
        """``--parent-map-preset`` is a relabelling, and the cache survives it.

        Under ``clade_evidence`` K's primary parent is J.2.4 and its sensitivity
        parent is J.2_int; under ``brief_as_stated`` the two swap.  Both (lineage,
        parent) pairs are already in the cache, so nothing needs recomputing -- but
        the MODEL TAGS must swap, because the parent-free tag means "the parent the
        map specifies" and the ``_parent<TOK>`` tag means "the other one".  A driver
        that let the cached table dictate the design would report the old labelling.
        """
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(cached, cached_map, EVALUABLE))

        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--parent-map-preset", "brief_as_stated", output_dir=out)
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert result["stage1_calls"] == []
        assert model_tags(result["specs"]) == [
            "ESCOTT", "PRESCOTT_eq2_c0p50_k1", "PRESCOTT_eq2_c0p50_k1_parentJ24"]
        primary = next(s for s in result["specs"]
                       if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1")
        assert primary["parent_by_lineage"]["K"] == "J.2_int"
        sensitivity = next(s for s in result["specs"]
                           if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1_parentJ24")
        assert sensitivity["parent_by_lineage"] == {"K": "J.2.4"}

    def test_parent_map_override_beats_the_cached_table(
        self, driver_module, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(cached, cached_map, EVALUABLE))

        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--parent-map", "J.2.4=J_int", output_dir=out)
        calls: List[Dict[str, object]] = []

        def _stage1(a, parent_map, evaluable, diagnostics_dir):
            calls.append(dict(parent_map))
            write_variants_table(
                Path(a.scores_dir),
                driver_module.expected_variant_plan(a, parent_map, evaluable))

        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER,
                              stage1=_stage1)
        assert result["error"] is None
        assert calls == [{**CLADE_EVIDENCE_MAP, "J.2.4": "J_int"}]
        primary = next(s for s in result["specs"]
                       if s["model_tag"] == "PRESCOTT_eq2_c0p50_k1")
        assert primary["parent_by_lineage"]["J.2.4"] == "J_int"
        assert primary["source_variant_by_lineage"]["J.2.4"] == \
            "PRESCOTT_eq2_c0p50_k1_parentJint"

    def test_no_auto_prepare_refuses_instead_of_substituting(
        self, driver_module, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(cached, cached_map, EVALUABLE))
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.75",
                         "--no-auto-prepare", output_dir=out)
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert isinstance(result["error"], FileNotFoundError)
        message = str(result["error"])
        assert "Stage 1 is needed" in message and "--no-auto-prepare" in message
        assert result["stage1_calls"] == []

    def test_a_stage_one_that_does_not_deliver_is_a_hard_error(
        self, driver_module, tmp_path, monkeypatch
    ):
        """The design must never be silently substituted after stage 1 'succeeds'."""
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert isinstance(result["error"], RuntimeError)
        message = str(result["error"])
        assert "Stage 1 ran but the requested design is still incomplete" in message
        assert "refusing to report a run whose manifest would not describe its outputs" \
            in message
        assert len(result["stage1_calls"]) == 1

    def test_force_recompute_reruns_stage_one_even_when_everything_is_cached(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--force-recompute-scores", output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, EVALUABLE))
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert len(result["stage1_calls"]) == 1
        assert "Stage 1 needed (--force-recompute-scores)" in capsys.readouterr().out

    def test_dry_run_never_calls_stage_one_even_when_matrices_are_missing(
        self, driver_module, tmp_path, monkeypatch
    ):
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--dry-run", output_dir=tmp_path / "out")
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        # --dry-run stops before ESCOTT scoring but still builds the lineage cache,
        # which is where our sentinel fires.
        assert result["reached_compute"] is True
        assert result["stage1_calls"] == []

    def test_test_mode_narrows_the_target_list_past_the_input_only_row(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--test-mode", output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, ["J_int"]))
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert result["stage1_calls"] == []
        printed = capsys.readouterr().out
        assert "[test-mode] using the first 2 guide row(s)" in printed
        assert "Evaluable lineages: ['J_int']" in printed
        # No K, so no sensitivity model.
        assert model_tags(result["specs"]) == ["ESCOTT", "PRESCOTT_eq2_c0p50_k1"]

    def test_score_variant_restricts_the_models_but_not_the_requested_design(
        self, driver_module, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.25,0.5",
                         "--score-variant", "ESCOTT", output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, EVALUABLE))
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert model_tags(result["specs"]) == ["ESCOTT"]
        # The plan itself is still the full requested design -- --score-variant is a
        # reporting filter, not a design change.
        assert len(result["plan"]) == 4 + 2 * 5

    def test_regen_figures_only_short_circuits_before_any_planning(
        self, driver_module, tmp_path, monkeypatch
    ):
        out = tmp_path / "out"
        (out / "tables").mkdir(parents=True)
        args = driver_module.build_parser().parse_args(
            ["--output-dir", str(out), "--regen-figures-only"])
        seen: Dict[str, object] = {}

        def _regen(a, *, tables_dir, plots_dir, existing_panel_metadata_df):
            seen["tables_dir"] = tables_dir
            seen["plots_dir"] = plots_dir
            seen["metadata_empty"] = existing_panel_metadata_df.empty
            return 0

        monkeypatch.setattr(driver_module.rma, "_regenerate_figures_from_existing_tables", _regen)
        monkeypatch.setattr(driver_module, "resolve_targets",
                            lambda _a: pytest.fail("resolve_targets must not be reached"))
        assert driver_module.run_analysis(args) == 0
        assert seen["tables_dir"] == out / "tables"
        assert seen["plots_dir"] == out / "plots"
        assert seen["metadata_empty"] is True


# =========================================================================== #
# 18. main()
# =========================================================================== #

@pytest.mark.cli
class TestMain:
    @staticmethod
    def _main(driver, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", ["run_prescott_diversity.py", *argv])
        return driver.main()

    def test_single_fasta_exits_two_with_the_explanation(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        with pytest.raises(SystemExit) as excinfo:
            self._main(driver_module, monkeypatch, [
                "--analysis-mode", "SINGLE_FASTA",
                "--output-dir", str(tmp_path / "o"),
            ])
        assert excinfo.value.code == 2
        stderr = capsys.readouterr().err
        assert "SINGLE_FASTA is not supported by this pipeline" in stderr
        assert "--analysis-mode MONTHLY_GUIDE" in stderr

    def test_equation_four_exits_two(self, driver_module, tmp_path, monkeypatch, capsys):
        guide = _tiny_file(tmp_path / "guide.csv", "month,fasta,reference\n")
        with pytest.raises(SystemExit) as excinfo:
            self._main(driver_module, monkeypatch, [
                "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide),
                "--output-dir", str(tmp_path / "o"), "--equation-grid", "4",
            ])
        assert excinfo.value.code == 2
        assert "equation 4 is not implemented upstream" in capsys.readouterr().err

    def test_a_bad_parent_map_exits_two(self, driver_module, tmp_path, monkeypatch, capsys):
        guide = _tiny_file(tmp_path / "guide.csv", "month,fasta,reference\n")
        with pytest.raises(SystemExit) as excinfo:
            self._main(driver_module, monkeypatch, [
                "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide),
                "--output-dir", str(tmp_path / "o"), "--parent-map", "K=J.2.4,J.2.4=K",
            ])
        assert excinfo.value.code == 2
        assert "contains a cycle" in capsys.readouterr().err

    def test_argparse_rejects_an_unknown_preset_before_validate_args(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        with pytest.raises(SystemExit) as excinfo:
            self._main(driver_module, monkeypatch, [
                "--analysis-mode", "MONTHLY_GUIDE",
                "--output-dir", str(tmp_path / "o"),
                "--parent-map-preset", "made_up",
            ])
        assert excinfo.value.code == 2
        assert "invalid choice: 'made_up'" in capsys.readouterr().err

    def test_a_successful_run_returns_the_inner_status(
        self, driver_module, tmp_path, monkeypatch
    ):
        guide = _tiny_file(tmp_path / "guide.csv", "month,fasta,reference\n")
        monkeypatch.setattr(driver_module, "run_analysis", lambda _a: 0)
        assert self._main(driver_module, monkeypatch, [
            "--analysis-mode", "MONTHLY_GUIDE", "--guide-path", str(guide),
            "--output-dir", str(tmp_path / "o"),
        ]) == 0


# =========================================================================== #
# 19. The shared-constants bridge (the "two halves cannot disagree" machinery)
# =========================================================================== #

class TestSharedConstantsBridge:
    def test_the_shared_module_is_importable_and_wins(self, driver_module):
        shared = driver_module.load_prescott_iav_constants()
        assert shared is not None
        assert driver_module.parent_map_presets() is shared.DEFAULT_PARENT_MAPS

    def test_an_absent_constants_module_falls_back_silently(
        self, driver_module, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(driver_module, "PRESCOTT_IAV_DIR", tmp_path / "no_such_dir")
        assert driver_module.load_prescott_iav_constants() is None
        assert driver_module.parent_map_presets() == driver_module.DEFAULT_PARENT_MAPS
        assert driver_module.default_trace_top_fraction() == 0.90
        assert driver_module.input_only_lineages() == frozenset({"G.1"})

    def test_a_present_but_broken_constants_module_is_a_hard_error(
        self, driver_module, tmp_path, monkeypatch
    ):
        # "Fix it or delete it" -- a half-written stage-1 package is worse than none.
        _tiny_file(tmp_path / "constants.py", "raise ImportError('boom')\n")
        monkeypatch.setattr(driver_module, "PRESCOTT_IAV_DIR", tmp_path)

        class _Stub:
            @staticmethod
            def import_module(_name):
                raise ImportError("boom")

        monkeypatch.setattr(driver_module, "importlib", _Stub)
        with pytest.raises(RuntimeError, match="could not be imported"):
            driver_module.load_prescott_iav_constants()

    def test_drift_between_the_two_copies_is_reported_once_and_the_shared_copy_wins(
        self, driver_module, monkeypatch, capsys
    ):
        drifted = {"clade_evidence": {"K": "SOMETHING_ELSE"}}

        class _Shared:
            DEFAULT_PARENT_MAPS = drifted

        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: _Shared)
        monkeypatch.setattr(driver_module, "_PRESET_DRIFT_REPORTED", [])
        assert driver_module.parent_map_presets() is drifted
        first = capsys.readouterr().out
        assert "WARNING: scripts/prescott_iav/constants.DEFAULT_PARENT_MAPS differs" in first
        # ...and not on every subsequent call.
        driver_module.parent_map_presets()
        assert capsys.readouterr().out == ""

    def test_an_empty_shared_preset_table_is_ignored(self, driver_module, monkeypatch):
        class _Shared:
            DEFAULT_PARENT_MAPS: Dict[str, Dict[str, str]] = {}

        monkeypatch.setattr(driver_module, "load_prescott_iav_constants", lambda: _Shared)
        assert driver_module.parent_map_presets() == driver_module.DEFAULT_PARENT_MAPS

    def test_the_drivers_fallback_copy_agrees_with_the_shared_module(self, driver_module):
        # If this ever fails the drift warning above is firing on every real run.
        assert driver_module.DEFAULT_PARENT_MAPS == driver_module.parent_map_presets()

    def test_leakage_defaults_are_quoted_from_leakage_check(self, driver_module):
        assert driver_module.leakage_default("min_identity") == 99.0
        assert driver_module.leakage_default("max_hamming") == 10
        assert driver_module.leakage_default("min_coverage") == 95.0
        assert driver_module.leakage_default("coverage_basis") == "both"
        assert driver_module.leakage_default("not_a_threshold") is None

    def test_explicit_directories_are_not_overwritten_by_the_defaults(
        self, driver_module, tmp_path
    ):
        args = driver_module.apply_prescott_defaults(parse_cli(
            driver_module, tmp_path,
            "--scores-dir", str(tmp_path / "S"),
            "--inputs-dir", str(tmp_path / "I"),
            "--escott-workdir", str(tmp_path / "W"),
            "--prescott-ref-dir", str(tmp_path / "R"),
        ))
        assert Path(args.scores_dir) == tmp_path / "S"
        assert Path(args.inputs_dir) == tmp_path / "I"
        assert Path(args.escott_workdir) == tmp_path / "W"
        assert Path(args.prescott_ref_dir) == tmp_path / "R"

    @pytest.mark.parametrize("value", [[1, 2], np.array([1.0, 2.0])])
    def test_optional_number_survives_a_value_pandas_cannot_test_for_nan(
        self, driver_module, value
    ):
        # pd.isna on an array returns an array, and `if <array>` raises.
        assert driver_module._optional_number(value) is None

    def test_normalised_label_survives_the_same(self, driver_module):
        assert driver_module._normalised_label([1, 2]) == "[1, 2]"


# =========================================================================== #
# 20. resolve_escott_temperature (the match-plm two-pass rule)
# =========================================================================== #

def _write_plm_reference(path: Path, log_values: Sequence[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"plm_prob": np.exp(np.asarray(log_values, dtype=float))}).to_csv(
        path, index=False)
    return path


def _write_raw_escott(path: Path, values: Sequence[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"col": list(values)}, index=[f"r{i}" for i in range(len(values))])
    frame.to_csv(path, sep="\t")
    return path


class TestResolveEscottTemperature:
    def test_fixed_mode_returns_the_flag_untouched(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path, "--escott-temperature", "2.5")
        assert driver_module.resolve_escott_temperature(args, tmp_path, ["K"]) == 2.5

    def test_match_plm_needs_the_reference_table_on_disk(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path,
                             "--escott-temperature-mode", "match-plm",
                             "--plm-reference-table", str(tmp_path / "absent.csv"))
        with pytest.raises(FileNotFoundError, match="--plm-reference-table not found"):
            driver_module.resolve_escott_temperature(args, tmp_path, ["K"])

    def test_a_flat_reference_cannot_be_matched(self, driver_module, tmp_path):
        reference = _write_plm_reference(tmp_path / "plm.csv", [0.5, 0.5, 0.5, 0.5])
        args = prepared_args(driver_module, tmp_path,
                             "--escott-temperature-mode", "match-plm",
                             "--plm-reference-table", str(reference))
        with pytest.raises(ValueError, match="non-positive sd"):
            driver_module.resolve_escott_temperature(args, tmp_path, ["K"])

    def test_match_plm_explains_the_two_pass_requirement(self, driver_module, tmp_path):
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        args = prepared_args(driver_module, tmp_path,
                             "--escott-temperature-mode", "match-plm",
                             "--plm-reference-table", str(reference))
        with pytest.raises(FileNotFoundError) as excinfo:
            driver_module.resolve_escott_temperature(args, tmp_path / "scores", ["K"])
        assert "Run once with --escott-temperature-mode fixed" in str(excinfo.value)

    def test_the_temperature_is_the_ratio_of_the_two_spreads(self, driver_module, tmp_path):
        # sd(log plm_prob) = 1.0 exactly; sd(E) = 3.0 exactly => T = 3.0.
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        scores = tmp_path / "scores"
        _write_raw_escott(scores / "K_ESCOTT_raw.tsv", [-3.0, -3.0, 3.0, 3.0])
        args = prepared_args(driver_module, tmp_path,
                             "--escott-temperature-mode", "match-plm",
                             "--plm-reference-table", str(reference))
        assert driver_module.resolve_escott_temperature(args, scores, ["K"]) == \
            pytest.approx(3.0)

    def test_the_median_across_lineages_is_used(self, driver_module, tmp_path):
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        scores = tmp_path / "scores"
        _write_raw_escott(scores / "K_ESCOTT_raw.tsv", [-1.0, -1.0, 1.0, 1.0])
        _write_raw_escott(scores / "J_int_ESCOTT_raw.tsv", [-2.0, -2.0, 2.0, 2.0])
        _write_raw_escott(scores / "J.2.4_ESCOTT_raw.tsv", [-9.0, -9.0, 9.0, 9.0])
        args = prepared_args(driver_module, tmp_path,
                             "--escott-temperature-mode", "match-plm",
                             "--plm-reference-table", str(reference))
        assert driver_module.resolve_escott_temperature(
            args, scores, ["K", "J_int", "J.2.4"]) == pytest.approx(2.0)

    def test_a_changed_temperature_forces_a_score_rebuild(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        """match-plm resolving to a new T must invalidate matrices built at the old one."""
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(
            out / "scores",
            driver_module.expected_variant_plan(cached, cached_map, EVALUABLE),
            temperature=1.0,
        )
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        for lineage in EVALUABLE:
            _write_raw_escott(out / "scores" / f"{lineage}_ESCOTT_raw.tsv",
                              [-3.0, -3.0, 3.0, 3.0])
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--escott-temperature-mode", "match-plm",
                         "--plm-reference-table", str(reference), output_dir=out)

        def _stage1(a, parent_map, evaluable, diagnostics_dir):
            write_variants_table(
                Path(a.scores_dir),
                driver_module.expected_variant_plan(a, parent_map, evaluable),
                temperature=a.escott_temperature,
            )

        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER,
                              stage1=_stage1)
        assert result["error"] is None
        assert len(result["stage1_calls"]) == 1
        printed = capsys.readouterr().out
        assert "=> T=3.0000" in printed
        assert "Stage 1 needed (temperature changed under --escott-temperature-mode " \
               "match-plm)" in printed


# =========================================================================== #
# 21. Jet-surrogate diagnostics
# =========================================================================== #

class TestJetSurrogateDiagnostics:
    @staticmethod
    def _write_jet_manifest(inputs_dir: Path, lineage_key: str, payload: Dict[str, object]) -> Path:
        path = Path(inputs_dir) / "jet" / f"{lineage_key}_jet_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_manifests_are_read_per_lineage_and_keyed_by_label(self, driver_module, tmp_path):
        inputs = tmp_path / "inputs"
        self._write_jet_manifest(inputs, "K", {"n_zero_trace_columns": 18})
        found = driver_module.read_jet_manifests(inputs, {}, ["K", "J_int"], "primary")
        assert set(found) == {"K"}
        assert found["K"]["n_zero_trace_columns"] == 18

    def test_a_corrupt_manifest_is_skipped_rather_than_fatal(self, driver_module, tmp_path):
        inputs = tmp_path / "inputs"
        path = self._write_jet_manifest(inputs, "K", {"n_zero_trace_columns": 1})
        path.write_text("{not json", encoding="utf-8")
        assert driver_module.read_jet_manifests(inputs, {}, ["K"], "primary") == {}

    def test_no_manifests_means_no_summary_file(self, driver_module, tmp_path):
        assert driver_module.write_jet_surrogate_summary({}, tmp_path) is None
        assert not (tmp_path / "jet_surrogate_summary.tsv").exists()

    def test_the_summary_surfaces_the_zero_trace_counts(self, driver_module, tmp_path, capsys):
        manifests = {
            "J_int": {"msa_n_sequences": 6434, "msa_n_columns": 566,
                      "weight_mode": "structural", "trace_definition": "bootstrap",
                      "trace_bootstraps": 50, "trace_top_fraction": 0.90,
                      "n_zero_trace_columns": 18, "frac_zero_trace_columns": 0.0318,
                      "structure": {"pdb": "/x/mono.pdb", "covered": 485}},
            "K": {"n_zero_trace_columns": 60, "frac_zero_trace_columns": 0.106,
                  "structure": {}},
        }
        out_path = driver_module.write_jet_surrogate_summary(manifests, tmp_path / "diag")
        assert out_path == tmp_path / "diag" / "jet_surrogate_summary.tsv"
        frame = pd.read_csv(out_path, sep="\t")
        assert list(frame["lineage"]) == ["J_int", "K"]
        assert list(frame["n_zero_trace_columns"]) == [18, 60]
        assert frame.loc[0, "trace_top_fraction"] == 0.90
        assert frame.loc[0, "structure_pdb"] == "/x/mono.pdb"
        # The worst lineage is 10.6% > 5%, so the operator must be told.
        printed = capsys.readouterr().out
        assert "WARNING: up to 10.6% of positions have trace == 0" in printed

    def test_no_warning_below_the_five_percent_band(self, driver_module, tmp_path, capsys):
        driver_module.write_jet_surrogate_summary(
            {"K": {"n_zero_trace_columns": 18, "frac_zero_trace_columns": 0.032}},
            tmp_path / "diag")
        assert "WARNING" not in capsys.readouterr().out

    def test_jetfile_paths_reach_the_manifest_with_their_md5(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        jet_file = _tiny_file(Path(args.inputs_dir) / "jet" / "K_surrogate_jet.res",
                              "1 A 0.5\n")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        driver_module.save_run_manifest(args, output_dir, [], parent_map, specs,
                                        pd.DataFrame(), EVALUABLE, {})
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["jetfile_paths"]["K"]["path"] == str(jet_file)
        assert manifest["jetfile_paths"]["K"]["md5"] == driver_module.file_md5(jet_file)
        assert set(manifest["jetfile_paths"]) == {"K"}   # only the files that exist


# =========================================================================== #
# 22. Small output helpers
# =========================================================================== #

class TestOutputHelpers:
    def test_stamp_score_formula_relabels_only_the_sweep_rows(self, driver_module):
        frame = pd.DataFrame({
            "model_variant": ["plm_alpha_sweep", "mutation_only_baseline"],
            "input_score_formula": ["plm_prob * mut_prob^alpha", "mut_prob"],
        })
        stamped = driver_module._stamp_score_formula(frame)
        assert list(stamped["input_score_formula"]) == [
            "escott_prob * mut_prob^alpha", "mut_prob"]
        # the input frame is not mutated
        assert frame.loc[0, "input_score_formula"] == "plm_prob * mut_prob^alpha"

    def test_stamp_score_formula_falls_back_to_the_formula_column(self, driver_module):
        frame = pd.DataFrame({"input_score_formula": ["plm_prob * mut_prob^alpha", "mut_prob"]})
        stamped = driver_module._stamp_score_formula(frame)
        assert list(stamped["input_score_formula"]) == [
            "escott_prob * mut_prob^alpha", "mut_prob"]

    def test_stamp_score_formula_is_a_noop_without_the_column(self, driver_module):
        frame = pd.DataFrame({"alpha": [0.0]})
        assert driver_module._stamp_score_formula(frame) is frame
        assert driver_module._stamp_score_formula(pd.DataFrame()).empty

    def test_score_scale_report_writes_nothing_for_an_empty_table(self, driver_module, tmp_path):
        args = prepared_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(pd.DataFrame(), tmp_path, args)
        assert not (tmp_path / "diagnostics" / "score_scale_report.tsv").exists()

    def test_score_scale_report_counts_the_flat_sites(self, driver_module, tmp_path):
        # Position 1 is uniform (a zero-trace ESCOTT column); position 2 is not.
        combined = pd.DataFrame({
            "model": ["ESCOTT"] * 4,
            "lineage": ["K"] * 4,
            "position": [1, 1, 2, 2],
            "plm_prob": [0.05, 0.05, 0.10, 0.20],
            "mut_prob": [0.01, 0.02, 0.03, 0.04],
        })
        args = prepared_args(driver_module, tmp_path)
        driver_module.write_score_scale_report(combined, tmp_path, args)
        frame = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert len(frame) == 1
        assert frame.loc[0, "n_rows"] == 4
        assert frame.loc[0, "n_flat_sites"] == 1
        assert frame.loc[0, "escott_temperature"] == 1.0
        assert np.isnan(frame.loc[0, "sd_log_plm_reference"])

    def test_score_scale_report_reports_the_plm_rescale_when_a_reference_exists(
        self, driver_module, tmp_path
    ):
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        args = prepared_args(driver_module, tmp_path,
                             "--plm-reference-table", str(reference))
        combined = pd.DataFrame({
            "model": ["ESCOTT"] * 4,
            "lineage": ["K"] * 4,
            "position": [1, 2, 3, 4],
            "plm_prob": np.exp([-2.0, -2.0, 2.0, 2.0]),
            "mut_prob": [0.01, 0.02, 0.03, 0.04],
        })
        driver_module.write_score_scale_report(combined, tmp_path, args)
        frame = pd.read_csv(tmp_path / "diagnostics" / "score_scale_report.tsv", sep="\t")
        assert frame.loc[0, "sd_log_score"] == pytest.approx(2.0)
        assert frame.loc[0, "sd_log_plm_reference"] == pytest.approx(1.0)
        assert frame.loc[0, "alpha_rescale"] == pytest.approx(0.5)


class TestRunStage1Step:
    def test_a_successful_step_prints_the_command_and_its_duration(
        self, driver_module, capsys
    ):
        driver_module.run_stage1_step([sys.executable, "-c", "pass"], dict(), "probe")
        printed = capsys.readouterr().out
        assert "[stage1:probe]" in printed
        assert "[stage1:probe] ok in" in printed

    def test_a_failing_step_raises_with_the_exit_code_and_the_command(
        self, driver_module
    ):
        with pytest.raises(RuntimeError) as excinfo:
            driver_module.run_stage1_step(
                [sys.executable, "-c", "import sys; sys.exit(3)"], dict(), "probe")
        message = str(excinfo.value)
        assert "step 'probe' failed with exit code 3" in message
        assert "-c" in message


class TestResolveTargets:
    def test_the_driver_does_its_own_test_mode_truncation(
        self, driver_module, tmp_path, monkeypatch
    ):
        import Functions_HuggingFace as fhf

        seen: Dict[str, object] = {}

        def _load(**kwargs):
            seen.update(kwargs)
            return [{"label": "K"}]

        monkeypatch.setattr(fhf, "load_analysis_targets", _load)
        args = prepared_args(driver_module, tmp_path, "--test-mode",
                             "--test-max-targets", "3")
        assert driver_module.resolve_targets(args) == [{"label": "K"}]
        # test_mode is passed as False on purpose: resolve_test_target_count has to see
        # the WHOLE guide to know how far past the input-only rows it must reach.
        assert seen["test_mode"] is False
        assert seen["test_max_targets"] == 1
        assert seen["analysis_mode"] == "MONTHLY_GUIDE"
        assert seen["guide_path"] == str(args.guide_path)


class TestModelCacheOptionalColumns:
    def test_a_metadata_table_without_a_temperature_column_is_still_usable(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        tables = tmp_path / "tables" / "per_model"
        _write_model_tables(tables, "ESCOTT")
        escott = next(s for s in specs if s["model_tag"] == "ESCOTT")
        metadata = _panel_metadata_for(driver_module, args, [escott], parent_map)
        assert driver_module.model_cache_is_valid(
            metadata.drop(columns=["escott_temperature", "mutation_model"]),
            args, escott, parent_map, tables)


# =========================================================================== #
# 23. CAVEATS: the leakage block, rendered from what stage 1 actually did
# =========================================================================== #

def _leakage_block(report_dir: Optional[Path] = None, *, purge: bool = True,
                   n_removed: int = 3, failures=None) -> Dict[str, object]:
    return {"leakage": {
        "status": "fail" if failures else "clean",
        "failures": failures,
        "thresholds": {"min_identity": 99.0, "max_hamming": 10,
                       "min_coverage": 95.0, "coverage_basis": "both"},
        "purge": purge,
        "report_dir": str(report_dir) if report_dir else None,
        "purges": {"K": {"depth_before": 6434, "n_removed": n_removed,
                         "depth_after": 6434 - n_removed,
                         "removed_fraction": n_removed / 6434,
                         "removed_identity_distribution": {"max": 99.5},
                         "removed_hamming_min": 1}} if purge else {},
        "checks": {"B_parent_vs_target": {"K": {
            "parent": "J.2.4",
            "accessions": {"n_shared_accessions": 0, "n_shared_exact_sequences": 0},
            "n_flagged": 8}}},
    }}


class TestCaveatsLeakageBlock:
    @staticmethod
    def _render(driver, tmp_path, block, *flags):
        args, parent_map, specs = _specs_for(driver, tmp_path, "--coefficient-grid", "0.5",
                                             *flags)
        args = driver.apply_prescott_defaults(args)
        inputs_dir = Path(args.inputs_dir)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        (inputs_dir / "inputs_manifest.json").write_text(json.dumps(block), encoding="utf-8")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return driver.write_caveats(args, output_dir, parent_map, specs, EVALUABLE, {})

    def test_a_clean_purge_is_quantified_in_the_headline(self, driver_module, tmp_path):
        text = self._render(driver_module, tmp_path, _leakage_block())
        assert "clean -- purge ON, 3 deep-set sequences removed across 1 target(s)" in text
        assert "K: 6434 -> 6431 (3 removed, 0.05%, max removed identity 99.5%)" in text
        assert "drop when coverage >= 95.0% (both basis) AND " \
               "(identity >= 99.0 OR hamming <= 10)" in text
        assert "J.2.4 -> K: 0 shared accessions, 0 shared exact sequences" in text

    def test_detection_only_says_the_hits_are_still_in_the_alignment(
        self, driver_module, tmp_path
    ):
        text = self._render(driver_module, tmp_path,
                            _leakage_block(purge=False), "--no-purge-leakage")
        assert "DETECTION ONLY (purge off)" in text
        assert "PURGE OFF (--no-purge-leakage): detection only, nothing was removed." in text
        assert "still in the alignment ESCOTT scored" in text

    def test_failed_gates_are_listed(self, driver_module, tmp_path):
        text = self._render(driver_module, tmp_path,
                            _leakage_block(failures=["K: 12 residual hits >= 99%"]))
        assert "1 gate(s) failed: K: 12 residual hits >= 99%" in text

    def test_a_missing_audit_trail_is_named_as_missing(self, driver_module, tmp_path):
        text = self._render(driver_module, tmp_path,
                            _leakage_block(report_dir=tmp_path / "no_such_dir"))
        assert "Audit trail     : not present in this output tree" in text

    def test_an_existing_audit_trail_is_cited(self, driver_module, tmp_path):
        report_dir = tmp_path / "leakage_reports"
        report_dir.mkdir()
        text = self._render(driver_module, tmp_path, _leakage_block(report_dir=report_dir))
        assert f"Audit trail     : {report_dir}" in text

    def test_no_parent_check_is_reported_as_not_run(self, driver_module, tmp_path):
        block = _leakage_block()
        block["leakage"]["checks"] = {}
        text = self._render(driver_module, tmp_path, block, "--no-leakage-check")
        assert "Parent vs target: not run (--no-leakage-check)" in text

    def test_a_purge_that_removed_nothing_from_any_target(self, driver_module, tmp_path):
        block = _leakage_block()
        block["leakage"]["purges"] = {}
        text = self._render(driver_module, tmp_path, block)
        assert "no evaluation target was purged in this pass" in text

    def test_the_parity_table_is_cited_when_it_exists(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        diagnostics = Path(args.output_dir) / "tables" / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        _tiny_file(diagnostics / "prescott_parity_check.tsv", "x\n")
        text = driver_module.write_caveats(args, Path(args.output_dir), parent_map, specs,
                                           EVALUABLE, {})
        assert "tables/diagnostics/prescott_parity_check.tsv" in text
        assert "SKIPPED in --test-mode" not in text

    def test_an_absent_parity_table_outside_test_mode_says_why(self, driver_module, tmp_path):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        text = driver_module.write_caveats(args, Path(args.output_dir), parent_map, specs,
                                           EVALUABLE, {})
        assert "stage 1 did not run in this pass" in text


# =========================================================================== #
# 24. ensure_score_matrix -- the ESCOTT analogue of rma.ensure_plm_matrix
# =========================================================================== #

class TestEnsureScoreMatrix:
    @staticmethod
    def _spec(variant: str, recorded=None):
        return {
            "model_tag": "ESCOTT",
            "source_variant_by_lineage": {"K": variant},
            "matrix_path_by_lineage": {"K": recorded},
        }

    def test_a_matrix_is_read_and_its_source_sequence_recovered(
        self, driver_module, tmp_path, score_matrix_factory, query_protein
    ):
        scores = tmp_path / "scores"
        scores.mkdir(parents=True, exist_ok=True)
        built = score_matrix_factory(name="tmp_matrix.csv")
        target = scores / "K_ESCOTT_score_matrix.csv"
        target.write_bytes(built.read_bytes())

        args = prepared_args(driver_module, tmp_path)
        matrix, path, sequence = driver_module.ensure_score_matrix(
            args, self._spec("ESCOTT"), "K",
            {"lineage_key": "K", "plm_ref_protein": query_protein}, scores,
        )
        assert path == str(target)
        assert sequence == query_protein
        assert matrix.shape == (20, len(query_protein))
        # The conftest matrix is a uniform 1/20 everywhere -- a fully zero-trace protein.
        assert np.allclose(matrix.to_numpy(dtype=float), 1.0 / 20.0)

    def test_a_missing_matrix_names_the_command_that_would_build_it(
        self, driver_module, tmp_path, query_protein
    ):
        scores = tmp_path / "scores"
        scores.mkdir(parents=True, exist_ok=True)
        args = prepared_args(driver_module, tmp_path)
        with pytest.raises(FileNotFoundError) as excinfo:
            driver_module.ensure_score_matrix(
                args, self._spec("PRESCOTT_eq2_c0p50_k1_parentJ24"), "K",
                {"lineage_key": "K", "plm_ref_protein": query_protein}, scores,
            )
        message = str(excinfo.value)
        assert "Score matrix not found" in message
        assert "K_PRESCOTT_eq2_c0p50_k1_parentJ24_score_matrix.csv" in message
        assert "--lineage K" in message
        assert "--auto-prepare" in message

    def test_a_recorded_path_outside_the_scores_dir_is_honoured(
        self, driver_module, tmp_path, score_matrix_factory, query_protein
    ):
        scores = tmp_path / "scores"
        scores.mkdir(parents=True, exist_ok=True)
        elsewhere = tmp_path / "elsewhere" / "K_ESCOTT_score_matrix.csv"
        elsewhere.parent.mkdir(parents=True, exist_ok=True)
        elsewhere.write_bytes(score_matrix_factory(name="src.csv").read_bytes())

        args = prepared_args(driver_module, tmp_path)
        _, path, _ = driver_module.ensure_score_matrix(
            args, self._spec("ESCOTT", recorded=str(elsewhere)), "K",
            {"lineage_key": "K", "plm_ref_protein": query_protein}, scores,
        )
        assert path == str(elsewhere)


# =========================================================================== #
# 25. Remaining planning branches
# =========================================================================== #

class TestRemainingPlanningBranches:
    def test_a_fully_cached_run_never_parses_the_diversity_panels(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        """The SC2 cache-parity fix: the whole-run gate is decided BEFORE the panels.

        ``build_lineage_cache`` is what parses and aligns the 27452- and
        17898-sequence GISAID panels.  When every per-model table is cached and the
        design key matches, ``rma._build_lightweight_lineage_cache_from_metadata``
        must be used instead and the codon tables must never be built at all.
        """
        import Functions_HuggingFace as fhf

        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        args = driver_module.apply_prescott_defaults(args)
        parent_map = driver_module.resolve_parent_map(args)
        plan = driver_module.expected_variant_plan(args, parent_map, EVALUABLE)
        write_variants_table(out / "scores", plan)
        specs = driver_module.build_score_specs(args, plan, parent_map)

        model_tables = out / "tables" / "per_model"
        for spec in specs:
            _write_model_tables(model_tables, str(spec["model_tag"]))
        metadata = _panel_metadata_for(driver_module, args, specs, parent_map)
        metadata["diversity_fasta"] = "/panels/x.fa"
        metadata["reference_fasta"] = "/refs/x.fa"
        (out / "tables").mkdir(parents=True, exist_ok=True)
        metadata.to_csv(out / "tables" / "panel_metadata.tsv", sep="\t", index=False)

        monkeypatch.setattr(driver_module, "resolve_targets",
                            lambda _a: [{"label": label} for label in LINEAGE_ORDER])

        def _never(*_a, **_k):
            pytest.fail("the diversity panels must not be parsed on a fully cached rerun")

        monkeypatch.setattr(fhf, "build_codon_aa_mutation_tables", _never)
        monkeypatch.setattr(driver_module.rma, "build_lineage_cache", _never)

        class _Reached(Exception):
            pass

        def _stop(*_a, **_k):
            raise _Reached()

        monkeypatch.setattr(driver_module, "write_jet_surrogate_summary", _stop)
        with pytest.raises(_Reached):
            driver_module.run_analysis(args)
        assert "All per-variant tables are cached and the design key matches" in \
            capsys.readouterr().out

    def test_only_the_first_ten_missing_combinations_are_listed(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path,
                         "--coefficient-grid", "0.1,0.2,0.3,0.4,0.5",
                         output_dir=out)

        def _stage1(a, parent_map, evaluable, diagnostics_dir):
            write_variants_table(
                Path(a.scores_dir),
                driver_module.expected_variant_plan(a, parent_map, evaluable))

        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER,
                              stage1=_stage1)
        assert result["error"] is None
        printed = capsys.readouterr().out
        assert printed.count("  needs: ") == 10
        # 4 ESCOTT + 5 coefficients x 5 edges = 29
        assert "  ... and 19 more" in printed

    def test_match_plm_at_an_unchanged_temperature_does_not_force_a_rebuild(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        out = tmp_path / "out"
        cached = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                           output_dir=out)
        cached_map = driver_module.resolve_parent_map(cached)
        write_variants_table(
            out / "scores",
            driver_module.expected_variant_plan(cached, cached_map, EVALUABLE),
            temperature=3.0,
        )
        reference = _write_plm_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        for lineage in EVALUABLE:
            _write_raw_escott(out / "scores" / f"{lineage}_ESCOTT_raw.tsv",
                              [-3.0, -3.0, 3.0, 3.0])
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         "--escott-temperature-mode", "match-plm",
                         "--plm-reference-table", str(reference), output_dir=out)
        result = run_planning(driver_module, args, monkeypatch, targets=LINEAGE_ORDER)
        assert result["error"] is None
        assert result["stage1_calls"] == []
        assert "temperature changed" not in capsys.readouterr().out

    def test_a_single_preset_leaves_nothing_to_contest(self, driver_module, tmp_path, monkeypatch):
        monkeypatch.setattr(driver_module, "parent_map_presets",
                            lambda: {"clade_evidence": dict(CLADE_EVIDENCE_MAP)})
        args = parse_cli(driver_module, tmp_path)
        assert driver_module.sensitivity_edges(args, CLADE_EVIDENCE_MAP) == {}

    def test_a_jet_manifest_without_a_zero_trace_count_is_skipped_in_caveats(
        self, driver_module, tmp_path
    ):
        args, parent_map, specs = _specs_for(driver_module, tmp_path,
                                             "--coefficient-grid", "0.5")
        args = driver_module.apply_prescott_defaults(args)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        text = driver_module.write_caveats(
            args, Path(args.output_dir), parent_map, specs, EVALUABLE,
            {"K": {"msa_n_sequences": 6434}},          # no n_zero_trace_columns
        )
        assert "not recorded (stage 1 not run this pass)" in text


class TestModelCacheRedundantGuards:
    """The two guards ``design_key`` already subsumes.

    ``mutation_model`` and ``escott_temperature`` are both inside
    ``shared_design_signature``, so no CLI change can reach these branches -- the
    ``design_key`` comparison fires first.  They are still live for a hand-edited or
    partially-migrated ``panel_metadata.tsv``, which is what these tests exercise.
    """

    @staticmethod
    def _cached(driver, tmp_path):
        args, parent_map, specs = _specs_for(driver, tmp_path, "--coefficient-grid", "0.5")
        tables = tmp_path / "tables" / "per_model"
        _write_model_tables(tables, "ESCOTT")
        escott = next(s for s in specs if s["model_tag"] == "ESCOTT")
        return args, parent_map, escott, tables, _panel_metadata_for(
            driver, args, [escott], parent_map)

    def test_a_hand_edited_mutation_model_still_invalidates(self, driver_module, tmp_path):
        args, parent_map, spec, tables, metadata = self._cached(driver_module, tmp_path)
        assert driver_module.model_cache_is_valid(metadata, args, spec, parent_map, tables)
        metadata["mutation_model"] = "SC2"
        assert not driver_module.model_cache_is_valid(metadata, args, spec, parent_map, tables)

    def test_a_hand_edited_temperature_still_invalidates(self, driver_module, tmp_path):
        args, parent_map, spec, tables, metadata = self._cached(driver_module, tmp_path)
        metadata["escott_temperature"] = 7.0
        assert not driver_module.model_cache_is_valid(metadata, args, spec, parent_map, tables)


class TestLineageCacheRefusal:
    def test_a_lineage_cache_with_no_evaluable_panel_is_refused(
        self, driver_module, tmp_path, monkeypatch, capsys
    ):
        import Functions_HuggingFace as fhf

        out = tmp_path / "out"
        args = parse_cli(driver_module, tmp_path, "--coefficient-grid", "0.5",
                         output_dir=out)
        parent_map = driver_module.resolve_parent_map(args)
        write_variants_table(out / "scores",
                             driver_module.expected_variant_plan(args, parent_map, EVALUABLE))
        monkeypatch.setattr(driver_module, "resolve_targets",
                            lambda _a: [{"label": label} for label in LINEAGE_ORDER])
        monkeypatch.setattr(fhf, "build_codon_aa_mutation_tables", lambda _m: {})
        # Only the input-only lineage came back with a panel.
        monkeypatch.setattr(driver_module.rma, "build_lineage_cache", lambda *_a, **_k: {
            "G.1": {"records": [1, 2], "full_ref_protein": "MK",
                    "alignment_diff_stats": {"mapped_sites": 2, "differing_sites": 0},
                    "diversity_path": "d", "reference_path": "r"},
        })
        with pytest.raises(RuntimeError) as excinfo:
            driver_module.run_analysis(driver_module.apply_prescott_defaults(args))
        assert "produced a usable panel" in str(excinfo.value)
        assert "G.1        [input-only] n_seq=     2 ref_len=2" in capsys.readouterr().out
