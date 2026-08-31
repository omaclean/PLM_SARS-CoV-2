#!/usr/bin/env python3
r"""Numerical and statistical regressions of the ESCOTT/PRESCOTT diversity pipeline.

WHAT THIS FILE IS FOR
=====================
Every test here pins a defect that was **found by running the real arithmetic on real
ESCOTT output and observing a wrong number**, not by reading the code and imagining one.
Each class names the defect, states the observed magnitude, and asserts the property the
defect violated -- so a revert of the fix fails on the *number*, not on a message.

THE FIVE DEFECTS
================
1.  :class:`TestMatchPlmTemperatureCalibration` --
    ``--escott-temperature-mode match-plm`` solved ``T = sd(E) / sd(log plm_ref)``.
    But ``plm_prob`` is a *per-column* softmax, so ``log P = E/T - logsumexp_col(E/T)``
    subtracts a per-column constant and keeps only the WITHIN-column variance.  ``sd(E)``
    is the total, which on real data is 2.0x larger.  Measured on the PRESCOTT
    distribution's own ``data/MLH1_normPred_evolCombi.txt`` (20 x 756, genuine ESCOTT
    output): total sd 1.8155, within-column sd 0.9005, and the chosen T came out
    1.36x-1.75x too large, so the achieved ``sd(log plm_prob)`` was only 0.56-0.71 of the
    reference it was supposed to equal.  alpha is not scale-free, so best_alpha from this
    run was not comparable with the PLM run match-plm exists to make it comparable with.

2.  :class:`TestUniformColumnsAreCountedAsDead` --
    ``count_flat_columns`` tests ``|value| <= tol``, which sees only the ``trace == 0``
    route to a dead site.  A fully-conserved position gets a CONSTANT NON-ZERO column,
    which softmaxes to the same uniform 1/20 and carries exactly as little rank
    information.  In the real MLH1 matrix that is **8 of 756 columns, and 0 all-zero**:
    the pipeline reported zero dead sites while 1.06% of the protein was noise.

3.  :class:`TestNonFiniteTemperature` --
    ``nan <= 0`` is ``False``, so ``--escott-temperature nan`` passed the driver's
    validation, passed ``escott_to_probability``'s own guard, and produced an all-NaN
    matrix that resurfaced as a misleading "softmax produced a non-positive probability".
    ``inf`` did not raise at all: it returned an exactly uniform 1/20 matrix -- every
    site dead -- and the run completed with meaningless metrics.

4.  :class:`TestUnderflowTemperatureDiagnostic` --
    below ``T ~ 0.008`` on real ESCOTT ranges ``exp()`` underflows to exactly zero and
    the assertion fired with no mention of the temperature, the matrix spread, or the
    smallest usable T.

5.  :class:`TestFrequencyFileValidation` --
    ``load_frequency_file``'s ``(frequency <= 0)`` guard is transparent to NaN and inf,
    and dies with a bare ``TypeError`` on a non-numeric token.  A NaN frequency was the
    dangerous one: ``log10(nan)`` is ``nan``, and ``nan > Fc`` and ``nan <= Fc`` are
    *both* False, so the mutant counted towards ``n_mutants_with_frequency`` and then
    received no penalty from any equation -- silent, zero-diagnostic data loss.

GROUND TRUTH
============
Wherever an exact answer exists it is used instead of a tolerance:

* a column of identical values softmaxes to exactly ``1/20`` at every temperature;
* adding a constant to a whole column is a **no-op for the softmax**, therefore it must
  be a no-op for the matched temperature -- the closed-form invariant that the old
  ``sd(E)``-based formula breaks, and it needs no reference to any measured number;
* with a single column, between-column variance is zero, so the corrected solver and the
  old closed form agree exactly -- which is why the pre-existing single-column tests in
  ``test_driver_cli.py`` still pass unchanged;
* ``solve_softmax_temperature`` is checked by feeding its answer back through the real
  ``run_escott.escott_to_probability`` and measuring ``sd(log P)``.

The real-ESCOTT tests read ``/home3/oml4h/PRESCOTT/data/MLH1_normPred_evolCombi.txt``
(the installed PRESCOTT distribution -- the same tree ``conftest`` already reads
``BLAT_jet.res`` from) and skip when it is absent.  Every assertion they make is also
made by a synthetic sibling that always runs, so the suite never depends on that file.

RUNNING
=======
    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_regressions_numerics.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from prescott_iav import run_escott as R
from tests_prescott_iav import conftest as C

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Real ESCOTT output.
# --------------------------------------------------------------------------- #

#: The PRESCOTT distribution ships one genuine ESCOTT ``normPred`` matrix.  Its column
#: labels are R's default ``V1..VN`` rather than ``"<WT><pos>"`` (it was written without
#: ``col.names=myColumnNames``), so :func:`real_escott_values` takes the *numbers* only
#: and :func:`write_escott_from_values` relabels them into the format our parser expects.
REAL_ESCOTT_TXT = Path("/home3/oml4h/PRESCOTT/data/MLH1_normPred_evolCombi.txt")

requires_real_escott = pytest.mark.skipif(
    not REAL_ESCOTT_TXT.exists(),
    reason=f"the installed PRESCOTT distribution's {REAL_ESCOTT_TXT.name} is not present",
)

#: Independently measured off the file, by hand, before any fix was written.  Literals,
#: so a failure says the *data* changed rather than that the code disagrees with itself.
REAL_ESCOTT_SHAPE = (20, 756)
REAL_ESCOTT_N_ALL_ZERO_COLUMNS = 0
REAL_ESCOTT_N_CONSTANT_COLUMNS = 8
REAL_ESCOTT_TOTAL_SD = 1.8155
REAL_ESCOTT_WITHIN_COLUMN_SD = 0.9005


def real_escott_values() -> np.ndarray:
    """The 20 x 756 array of genuine ESCOTT scores, NaN on the wild-type cell."""
    raw = pd.read_table(REAL_ESCOTT_TXT, sep=r"\s+")
    return raw.to_numpy(dtype=float)


def write_escott_from_values(path: Path, values: np.ndarray) -> Tuple[Path, str]:
    """Write ``values`` as a ``write.table``-format ``_normPred_evolCombi.txt``.

    Each column's wild-type letter is read back off its single NaN, so the file the
    parser sees is self-consistent by construction.  Returns ``(path, wt_sequence)``.
    """
    rows = list(C.ESCOTT_ROW_ORDER)
    wt: List[str] = []
    for j in range(values.shape[1]):
        nan_rows = np.flatnonzero(np.isnan(values[:, j]))
        assert nan_rows.size == 1, f"column {j} has {nan_rows.size} NaNs, need exactly 1"
        wt.append(rows[int(nan_rows[0])].upper())
    columns = [f"{wt[j]}{j + 1}" for j in range(values.shape[1])]
    lines = [" ".join(f'"{name}"' for name in columns)]
    for i, aa in enumerate(rows):
        cells = ["NA" if np.isnan(x) else repr(float(x)) for x in values[i]]
        lines.append(f'"{aa}" ' + " ".join(cells))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path), "".join(wt)


@pytest.fixture()
def parsed_matrix(tmp_path: Path) -> pd.DataFrame:
    """The conftest ESCOTT product, parsed by the module under test.

    A local copy of ``test_run_escott.parsed_matrix`` -- pytest fixtures do not cross
    test modules, and this suite deliberately keeps each module self-contained.
    """
    path = C.write_escott_normpred(tmp_path / f"{C.QUERY_HEADER}_normPred_evolCombi.txt")
    return R.read_escott_matrix(path)


@pytest.fixture(scope="session")
def real_escott_matrix(tmp_path_factory) -> pd.DataFrame:
    """Genuine ESCOTT output, relabelled into ``"<WT><pos>"`` and parsed by our reader."""
    if not REAL_ESCOTT_TXT.exists():
        pytest.skip(f"{REAL_ESCOTT_TXT} is not present")
    values = real_escott_values()
    # Only the column *labels* are rewritten; every number is the distribution's own.
    path, _ = write_escott_from_values(
        tmp_path_factory.mktemp("real_escott") / "MLH1_normPred_evolCombi.txt", values
    )
    return R.read_escott_matrix(path)


def synthetic_offset_values(
    n_positions: int = 24,
    within_spread: float = 1.0,
    between_spread: float = 4.0,
    seed: int = 20240805,
) -> np.ndarray:
    """A 20 x L raw ESCOTT array with a deliberately large BETWEEN-column component.

    Every column has the same within-column shape plus a per-column offset.  Those
    offsets are pure between-column variance: they move ``sd(E)`` a long way and move
    ``sd(log softmax(E/T))`` not at all, because the softmax subtracts a per-column
    constant.  That gap is the whole defect, so this is the fixture that exposes it.
    """
    rng = np.random.default_rng(seed)
    shape = rng.normal(0.0, within_spread, size=(20, 1))
    offsets = rng.normal(0.0, between_spread, size=(1, n_positions))
    values = shape + offsets
    for j in range(n_positions):
        values[j % 20, j] = np.nan  # exactly one NA per column, on a rotating row
    return values


def make_match_plm_args(driver_module, tmp_path: Path, reference: Path):
    """A real parsed Namespace in match-plm mode (never a hand-built one)."""
    return driver_module.build_parser().parse_args([
        "--output-dir", str(tmp_path / "out"),
        "--analysis-mode", "MONTHLY_GUIDE",
        "--mutation-model", "H3N2",
        "--escott-temperature-mode", "match-plm",
        "--plm-reference-table", str(reference),
    ])


def write_minimal_guide(tmp_path: Path) -> Path:
    """The smallest MONTHLY_GUIDE file ``validate_args`` will accept."""
    path = tmp_path / "guide.csv"
    path.write_text("month,fasta,reference\nK,/dev/null,/dev/null\n", encoding="utf-8")
    return path


# =========================================================================== #
# 1. match-plm was calibrating against the wrong spread.
# =========================================================================== #

@pytest.mark.requires_rma
class TestMatchPlmTemperatureCalibration:
    """``match-plm`` must actually make ``sd(log plm_prob)`` equal the reference.

    That is the *definition* of the mode and the only reason it exists: alpha is not
    scale-free, so a best_alpha from this run and one from a PLM run mean the same
    trade-off only when the two log-score spreads match.  The old implementation solved
    ``T = sd(E) / sd(log plm_ref)``, which is the answer to a different question.
    """

    @staticmethod
    def _write_reference(path: Path, log_values: Sequence[float]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"plm_prob": np.exp(np.asarray(log_values, dtype=float))}).to_csv(
            path, index=False
        )
        return path

    @staticmethod
    def _write_raw(path: Path, values: np.ndarray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            values,
            index=[aa.upper() for aa in C.ESCOTT_ROW_ORDER],
            columns=list(range(1, values.shape[1] + 1)),
        ).to_csv(path, sep="\t")
        return path

    @staticmethod
    def _achieved_sd(values: np.ndarray, temperature: float) -> float:
        """``sd(log plm_prob)`` measured on the matrix stage 1 would actually write."""
        frame = pd.DataFrame(
            values,
            index=list(C.PLM_CACHE_ROW_ORDER),
            columns=list(range(1, values.shape[1] + 1)),
        )
        probabilities = R.escott_to_probability(frame, temperature=temperature)
        return float(np.std(np.log(probabilities.to_numpy(dtype=float))))

    # --- the closed-form invariant, no measured numbers involved ------------------

    def test_a_per_column_offset_cannot_change_the_matched_temperature(
        self, driver_module, tmp_path
    ):
        """Adding a constant to a whole column is a no-op for the softmax.

        ``softmax(x + c) == softmax(x)`` exactly, so ``log plm_prob`` is bit-identical
        before and after, so the T that matches a given reference spread must be
        identical too.  ``sd(E)``, in contrast, moves a long way -- which is precisely
        why the old ``T = sd(E) / sd(log plm_ref)`` was wrong.  This assertion needs no
        knowledge of ESCOTT at all; it is a property of the softmax.
        """
        reference = self._write_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        base = synthetic_offset_values(between_spread=0.0)
        shifted = base + np.linspace(-6.0, 6.0, base.shape[1])[None, :]

        temperatures = []
        for name, values in (("flat", base), ("offset", shifted)):
            scores = tmp_path / name
            self._write_raw(scores / "K_ESCOTT_raw.tsv", values)
            args = make_match_plm_args(driver_module, tmp_path, reference)
            temperatures.append(driver_module.resolve_escott_temperature(args, scores, ["K"]))

        # sd(E) really did move, so a formula built on it really would have differed.
        assert float(np.nanstd(shifted)) > 2.0 * float(np.nanstd(base))
        assert temperatures[0] == pytest.approx(temperatures[1], rel=1e-6)

    # --- the definition, measured end to end --------------------------------------

    @pytest.mark.parametrize("target_sd", [0.5, 1.0, 2.0, 3.0])
    def test_the_resolved_temperature_hits_the_reference_spread(
        self, driver_module, tmp_path, target_sd
    ):
        """Feed the answer back through the real softmax and measure.

        Before the fix this ratio was 0.56-0.71 on real ESCOTT output; the tolerance here
        is 1e-6, so a revert fails by five orders of magnitude.
        """
        reference = self._write_reference(
            tmp_path / "plm.csv", [-target_sd, -target_sd, target_sd, target_sd]
        )
        values = synthetic_offset_values()
        scores = tmp_path / "scores"
        self._write_raw(scores / "K_ESCOTT_raw.tsv", values)
        args = make_match_plm_args(driver_module, tmp_path, reference)

        temperature = driver_module.resolve_escott_temperature(args, scores, ["K"])
        assert self._achieved_sd(values, temperature) == pytest.approx(target_sd, rel=1e-6)

    def test_the_old_closed_form_would_have_missed_by_a_third_or_more(
        self, driver_module, tmp_path
    ):
        """The defect is quantified here so a revert cannot look like a rounding change.

        ``T_old = sd(E) / target`` is recomputed explicitly and shown to overshoot, and
        the spread it delivers is shown to fall short of the target by more than 20%.
        """
        target_sd = 1.0
        reference = self._write_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        values = synthetic_offset_values()
        scores = tmp_path / "scores"
        self._write_raw(scores / "K_ESCOTT_raw.tsv", values)
        args = make_match_plm_args(driver_module, tmp_path, reference)

        resolved = driver_module.resolve_escott_temperature(args, scores, ["K"])
        old_formula = float(np.nanstd(values)) / target_sd

        assert old_formula > 1.3 * resolved
        assert self._achieved_sd(values, old_formula) < 0.8 * target_sd
        assert self._achieved_sd(values, resolved) == pytest.approx(target_sd, rel=1e-6)

    def test_a_single_column_matrix_still_agrees_with_the_closed_form(
        self, driver_module, tmp_path
    ):
        """With one column there is no between-column variance, so both agree exactly.

        This is why the pre-existing single-column tests in ``test_driver_cli.py`` were
        not pinning the bug and did not have to change: they could not see it.
        """
        reference = self._write_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        column = np.linspace(-4.5, 4.5, 20).reshape(20, 1)
        column[7, 0] = np.nan
        scores = tmp_path / "scores"
        self._write_raw(scores / "K_ESCOTT_raw.tsv", column)
        args = make_match_plm_args(driver_module, tmp_path, reference)

        filled = column.copy()
        filled[7, 0] = np.nanmax(column)
        assert driver_module.resolve_escott_temperature(args, scores, ["K"]) == pytest.approx(
            float(np.std(filled)) / 1.0, rel=1e-6
        )

    def test_the_median_is_taken_over_solved_temperatures(self, driver_module, tmp_path):
        """One scalar T for the run, chosen as the median of the per-lineage solutions.

        The median must be of the *temperatures*, not of the spreads: those are different
        statistics unless the map from spread to T is linear, which it is not.
        """
        reference = self._write_reference(tmp_path / "plm.csv", [-1.0, -1.0, 1.0, 1.0])
        scores = tmp_path / "scores"
        per_lineage = []
        for key, scale in (("K", 1.0), ("J_int", 2.0), ("J.2.4", 5.0)):
            values = synthetic_offset_values() * scale
            self._write_raw(scores / f"{driver_module.safe_key(key)}_ESCOTT_raw.tsv", values)
            per_lineage.append(driver_module.solve_softmax_temperature(values, 1.0))
        args = make_match_plm_args(driver_module, tmp_path, reference)
        assert driver_module.resolve_escott_temperature(
            args, scores, ["K", "J_int", "J.2.4"]
        ) == pytest.approx(float(np.median(per_lineage)), rel=1e-9)

    # --- the solver itself ---------------------------------------------------------

    def test_sd_log_softmax_equals_the_real_softmax_it_models(self, driver_module):
        """``sd_log_softmax`` must be the number ``escott_to_probability`` produces."""
        values = synthetic_offset_values()
        frame = pd.DataFrame(
            values, index=list(C.PLM_CACHE_ROW_ORDER),
            columns=list(range(1, values.shape[1] + 1)),
        )
        for temperature in (0.05, 0.5, 1.0, 7.0):
            direct = float(np.std(np.log(
                R.escott_to_probability(frame, temperature=temperature).to_numpy(dtype=float)
            )))
            assert driver_module.sd_log_softmax(values, temperature) == pytest.approx(
                direct, rel=1e-12
            )

    def test_sd_log_softmax_survives_a_temperature_that_underflows(self, driver_module):
        """It works below the temperature at which the probabilities themselves die.

        ``escott_to_probability`` cannot run at T = 1e-6 (exp underflows to exactly 0),
        but the solver must still be able to evaluate there or the bisection bracket is
        unusable.  Computing ``shifted - logsumexp`` instead of ``log(softmax(...))`` is
        what buys that.
        """
        values = synthetic_offset_values()
        result = driver_module.sd_log_softmax(values, 1e-6)
        assert np.isfinite(result) and result > 1e5

    def test_sd_log_softmax_is_strictly_decreasing_in_temperature(self, driver_module):
        """Monotonicity is what makes the bisection well posed."""
        values = synthetic_offset_values()
        spreads = [driver_module.sd_log_softmax(values, t) for t in np.logspace(-6, 6, 60)]
        assert all(later < earlier for earlier, later in zip(spreads, spreads[1:]))

    def test_a_column_constant_matrix_cannot_be_matched_and_says_so(self, driver_module):
        """Every column uniform => sd(log P) is 0 at every T => no solution exists."""
        values = np.tile(np.arange(1.0, 5.0)[None, :], (20, 1))
        for j in range(values.shape[1]):
            values[j % 20, j] = np.nan
        with pytest.raises(ValueError, match="almost no within-position spread"):
            driver_module.solve_softmax_temperature(values, 1.0)

    @pytest.mark.parametrize("target", [0.0, -1.0, float("nan")])
    def test_a_non_positive_target_is_refused(self, driver_module, target):
        with pytest.raises(ValueError, match="non-positive target"):
            driver_module.solve_softmax_temperature(synthetic_offset_values(), target)

    def test_the_solver_never_returns_a_non_finite_temperature(self, driver_module):
        """The invariant behind the ``median(temperatures)`` guard in the driver.

        ``solve_softmax_temperature`` either raises or returns a strictly positive finite
        T inside its bracket, so the median of several can never be NaN or inf.  Pinned
        rather than faked, per this suite's convention for a branch that is dead by
        construction.
        """
        lo, hi = driver_module.SOFTMAX_TEMPERATURE_BRACKET
        for scale in (0.05, 1.0, 20.0):
            for target in (0.25, 1.0, 6.0):
                temperature = driver_module.solve_softmax_temperature(
                    synthetic_offset_values() * scale, target
                )
                assert np.isfinite(temperature)
                assert lo < temperature < hi

    # --- on genuine ESCOTT output --------------------------------------------------

    @requires_real_escott
    def test_the_two_spreads_differ_by_two_fold_on_real_escott_output(self):
        """The measurement the whole finding rests on, made on the distribution file."""
        values = real_escott_values()
        assert values.shape == REAL_ESCOTT_SHAPE
        total = float(np.nanstd(values))
        within = float(np.nanstd(values - np.nanmean(values, axis=0, keepdims=True)))
        assert total == pytest.approx(REAL_ESCOTT_TOTAL_SD, abs=5e-4)
        assert within == pytest.approx(REAL_ESCOTT_WITHIN_COLUMN_SD, abs=5e-4)
        assert total / within > 1.9

    @requires_real_escott
    @pytest.mark.parametrize("target_sd", [1.0, 3.0])
    def test_real_escott_output_is_matched_exactly(
        self, driver_module, real_escott_matrix, target_sd
    ):
        values = real_escott_matrix.to_numpy(dtype=float)
        temperature = driver_module.solve_softmax_temperature(values, target_sd)
        achieved = float(np.std(np.log(
            R.escott_to_probability(real_escott_matrix, temperature=temperature).to_numpy(float)
        )))
        assert achieved == pytest.approx(target_sd, rel=1e-9)
        # And the old formula demonstrably did not.
        assert float(np.nanstd(values)) / target_sd > 1.3 * temperature


# =========================================================================== #
# 2. Constant-but-non-zero columns are dead sites and were never counted.
# =========================================================================== #

class TestUniformColumnsAreCountedAsDead:
    """A position whose 20 scores are equal is noise, whatever that equal value is.

    ``pred.R:487`` scales each column by ``trace[i]``, so ``trace == 0`` gives an
    identically-zero column -- the case ``count_flat_columns`` was written for.  But full
    conservation gives a constant NON-ZERO column, and after the per-column softmax the
    two are indistinguishable: both are exactly ``1/20`` everywhere, both contribute
    nothing but noise to every site-level metric.  Only the first was counted.
    """

    @staticmethod
    def constant_column_cells(value: float, protein: str = "MKT") -> Dict[object, object]:
        """Cells for :func:`conftest.escott_matrix_values` with column 2 held constant."""
        cells = dict(C.escott_matrix_values(protein, ()))
        for aa in C.PLM_CACHE_ROW_ORDER:
            if cells[(aa, 2)] is not None:
                cells[(aa, 2)] = value
        return cells

    def test_a_constant_non_zero_column_softmaxes_to_exactly_one_twentieth(
        self, escott_matrix_factory
    ):
        """The closed form: 20 equal values give ``1/20`` at every temperature."""
        path, _ = escott_matrix_factory(
            protein="MKT", values=self.constant_column_cells(-5.5)
        )
        matrix = R.read_escott_matrix(path)
        for temperature in (0.25, 1.0, 40.0):
            column = R.escott_to_probability(matrix, temperature=temperature)[2]
            assert column.to_numpy(dtype=float) == pytest.approx(0.05, abs=1e-15)

    def test_count_uniform_columns_sees_it_and_count_flat_columns_does_not(
        self, escott_matrix_factory
    ):
        """The regression itself: a dead site, reported as zero dead sites."""
        path, _ = escott_matrix_factory(
            protein="MKT", values=self.constant_column_cells(-5.5)
        )
        matrix = R.read_escott_matrix(path)
        assert R.count_flat_columns(matrix) == 0        # documented trace == 0 semantics
        assert R.count_uniform_columns(matrix) == 1     # the honest dead-site count

    def test_the_uniform_count_is_a_superset_of_the_zero_trace_count(
        self, parsed_matrix, fake_escott_matrix
    ):
        """Every all-zero column is also a uniform one, so the counts must nest."""
        flat = R.count_flat_columns(parsed_matrix)
        uniform = R.count_uniform_columns(parsed_matrix)
        assert flat == len(fake_escott_matrix["flat_positions"])
        assert uniform >= flat

    def test_a_matrix_with_no_constant_column_counts_zero(self, escott_matrix_factory):
        path, _ = escott_matrix_factory(flat_positions=())
        assert R.count_uniform_columns(R.read_escott_matrix(path)) == 0

    def test_the_wild_type_na_does_not_fake_a_constant_column(self, escott_matrix_factory):
        """The NA must be excluded, not zero-filled.

        ``count_flat_columns`` zero-fills the NA, which is right for its question.  Doing
        the same here would call a constant column of value ``v != 0`` non-constant
        (because the filled cell would be 0), i.e. it would reintroduce the bug.
        ``fill_wildtype`` puts the column *maximum* there, which for a constant column is
        that same constant -- so excluding the NA is the faithful test.
        """
        path, _ = escott_matrix_factory(
            protein="MKT", values=self.constant_column_cells(-5.5)
        )
        matrix = R.read_escott_matrix(path)
        filled = R.fill_wildtype(matrix)
        assert filled[2].nunique() == 1
        assert R.count_uniform_columns(filled) == R.count_uniform_columns(matrix)

    @pytest.mark.parametrize("value", [0.0, -1e-13, 1e-13])
    def test_near_zero_constant_columns_are_counted_by_both(self, escott_matrix_factory, value):
        path, _ = escott_matrix_factory(
            protein="MKT", values=self.constant_column_cells(value)
        )
        matrix = R.read_escott_matrix(path)
        assert R.count_flat_columns(matrix) == 1
        assert R.count_uniform_columns(matrix) == 1

    def test_the_stage_one_report_names_both_kinds_of_dead_site(
        self, escott_matrix_factory, capsys
    ):
        """``report_dead_columns`` is what stage 1 prints and what the manifest records.

        A constant non-zero column used to produce no line at all; the run looked clean.
        """
        path, _ = escott_matrix_factory(
            protein="MKT", values=self.constant_column_cells(-5.5)
        )
        matrix = R.read_escott_matrix(path)
        assert R.report_dead_columns("K", matrix) == (0, 1)
        out = capsys.readouterr().out
        assert "uniform 1/20" in out and "constant non-zero" in out
        assert "trace == 0)" not in out          # there are none of that kind here

    def test_the_report_is_silent_when_nothing_is_dead(self, escott_matrix_factory, capsys):
        path, _ = escott_matrix_factory(flat_positions=())
        assert R.report_dead_columns("K", R.read_escott_matrix(path)) == (0, 0)
        assert capsys.readouterr().out == ""

    def test_the_report_does_not_double_count_a_zero_trace_column(
        self, parsed_matrix, fake_escott_matrix, capsys
    ):
        """An all-zero column is uniform too, so it must be reported once, as trace == 0."""
        n_flat, n_uniform = R.report_dead_columns("K", parsed_matrix)
        assert n_flat == n_uniform == len(fake_escott_matrix["flat_positions"])
        out = capsys.readouterr().out
        assert "trace == 0" in out
        assert "constant non-zero" not in out

    @requires_real_escott
    def test_real_escott_output_has_eight_dead_sites_and_zero_all_zero_columns(
        self, real_escott_matrix
    ):
        """The observation that turned this from a code smell into a bug.

        The numbers were measured directly off the distribution file, independently of
        the code under test (see the module docstring); ``count_flat_columns`` returning 0
        while 8 positions are dead is exactly the mis-report the fix removes.
        """
        assert R.count_flat_columns(real_escott_matrix) == REAL_ESCOTT_N_ALL_ZERO_COLUMNS
        assert R.count_uniform_columns(real_escott_matrix) == REAL_ESCOTT_N_CONSTANT_COLUMNS

        values = R.escott_to_probability(real_escott_matrix).to_numpy(dtype=float)
        uniform = np.flatnonzero(values.max(axis=0) - values.min(axis=0) <= 1e-15)
        assert uniform.size == REAL_ESCOTT_N_CONSTANT_COLUMNS
        assert values[:, uniform] == pytest.approx(0.05, abs=1e-15)


# =========================================================================== #
# 3. Non-finite temperatures walked through every guard.
# =========================================================================== #

class TestNonFiniteTemperature:
    """``nan <= 0`` is ``False``; every ``<= 0`` guard was therefore NaN-transparent."""

    @pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
    def test_escott_to_probability_refuses_them(self, parsed_matrix, temperature):
        with pytest.raises(ValueError, match="temperature must be positive"):
            R.escott_to_probability(parsed_matrix, temperature=temperature)

    def test_nan_used_to_reach_the_arithmetic_and_poison_the_whole_matrix(
        self, parsed_matrix
    ):
        """What the old code did, reproduced directly, so the severity is on record.

        Every one of the 20 x L probabilities came out NaN, and the failure surfaced as
        "softmax produced a non-positive probability" -- a message that points at the
        matrix, not at the temperature the user typed.
        """
        filled = R.fill_wildtype(parsed_matrix).to_numpy(dtype=float)
        assert np.isnan(filled / float("nan")).all()

    def test_inf_used_to_return_a_silently_uniform_matrix(self, parsed_matrix):
        """The worse half: no exception at all, just every site dead.

        ``E / inf`` is 0 everywhere, so the softmax is exactly ``1/20`` at every position
        and the run completes with a score matrix that carries zero information.
        """
        scaled = R.fill_wildtype(parsed_matrix).to_numpy(dtype=float) / float("inf")
        exponentiated = np.exp(scaled - scaled.max(axis=0, keepdims=True))
        probabilities = exponentiated / exponentiated.sum(axis=0, keepdims=True)
        assert probabilities == pytest.approx(0.05, abs=1e-15)

    @pytest.mark.requires_rma
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_the_driver_rejects_them_at_parse_time(self, driver_module, tmp_path, value):
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "out"),
            "--analysis-mode", "MONTHLY_GUIDE",
            "--mutation-model", "H3N2",
            "--guide-path", str(write_minimal_guide(tmp_path)),
            f"--escott-temperature={value}",
        ])
        with pytest.raises(ValueError, match=r"--escott-temperature must be > 0"):
            driver_module.validate_args(args)

    @pytest.mark.requires_rma
    @pytest.mark.parametrize("value", ["nan", "inf"])
    def test_the_driver_rejects_a_non_finite_alpha_step(self, driver_module, tmp_path, value):
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "out"),
            "--analysis-mode", "MONTHLY_GUIDE",
            "--mutation-model", "H3N2",
            "--guide-path", str(write_minimal_guide(tmp_path)),
            f"--alpha-step={value}",
        ])
        with pytest.raises(ValueError, match=r"--alpha-step must be > 0"):
            driver_module.validate_args(args)

    @pytest.mark.requires_rma
    def test_the_driver_rejects_a_non_finite_coefficient(self, driver_module, tmp_path):
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "out"),
            "--analysis-mode", "MONTHLY_GUIDE",
            "--mutation-model", "H3N2",
            "--guide-path", str(write_minimal_guide(tmp_path)),
            "--coefficient-grid", "0.5,nan",
        ])
        with pytest.raises(ValueError, match=r"--coefficient-grid values must be >= 0"):
            driver_module.validate_args(args)

    @pytest.mark.requires_rma
    def test_a_finite_temperature_still_passes(self, driver_module, tmp_path):
        args = driver_module.build_parser().parse_args([
            "--output-dir", str(tmp_path / "out"),
            "--analysis-mode", "MONTHLY_GUIDE",
            "--mutation-model", "H3N2",
            "--guide-path", str(write_minimal_guide(tmp_path)),
            "--escott-temperature", "0.75",
        ])
        driver_module.validate_args(args)  # must not raise


# =========================================================================== #
# 4. The underflow diagnostic named neither the temperature nor a way out.
# =========================================================================== #

class TestUnderflowTemperatureDiagnostic:
    """Below ``T ~ spread/745`` the softmax underflows to exactly zero.

    On real ESCOTT output the widest column spans 6.03 score units, so anything under
    ~0.008 lands here -- and ``--escott-temperature 0.001`` is a perfectly ordinary thing
    to try when sharpening the distribution.  The guard is right to fire; what it said
    was not actionable.
    """

    @staticmethod
    def wide_matrix(escott_matrix_factory, spread: float = 6.0):
        protein = "MKT"
        cells = dict(C.escott_matrix_values(protein, ()))
        for position in (1, 2, 3):
            for index, aa in enumerate(C.PLM_CACHE_ROW_ORDER):
                if cells[(aa, position)] is not None:
                    cells[(aa, position)] = -spread * index / 19.0
        path, _ = escott_matrix_factory(protein=protein, values=cells)
        return R.read_escott_matrix(path)

    def test_the_message_names_the_temperature_and_the_smallest_usable_one(
        self, escott_matrix_factory
    ):
        matrix = self.wide_matrix(escott_matrix_factory)
        with pytest.raises(AssertionError) as excinfo:
            R.escott_to_probability(matrix, temperature=1e-4)
        message = str(excinfo.value)
        assert "non-positive probability" in message      # the old, pinned substring
        assert "0.0001" in message or "1e-04" in message  # the temperature the user gave
        assert "underflows" in message and "T = " in message

    def test_the_reported_floor_is_the_real_boundary(self, escott_matrix_factory):
        """Just above the quoted minimum must work; well below it must not."""
        spread = 6.0
        matrix = self.wide_matrix(escott_matrix_factory, spread=spread)
        floor = spread / 700.0
        probabilities = R.escott_to_probability(matrix, temperature=floor * 1.05)
        assert (probabilities.to_numpy(dtype=float) > 0.0).all()
        with pytest.raises(AssertionError, match="non-positive probability"):
            R.escott_to_probability(matrix, temperature=floor / 10.0)

    @requires_real_escott
    def test_the_real_escott_boundary_is_where_it_was_measured(self, real_escott_matrix):
        """~0.008 on the real matrix: reachable with an ordinary --escott-temperature."""
        values = R.fill_wildtype(real_escott_matrix).to_numpy(dtype=float)
        spread = float(np.max(values.max(axis=0) - values.min(axis=0)))
        assert spread == pytest.approx(6.03, abs=0.05)
        R.escott_to_probability(real_escott_matrix, temperature=0.01)  # must not raise
        with pytest.raises(AssertionError, match="non-positive probability"):
            R.escott_to_probability(real_escott_matrix, temperature=0.005)


# =========================================================================== #
# 5. Frequency-file validation was transparent to NaN and inf.
# =========================================================================== #

class TestFrequencyFileValidation:
    """``nan <= 0`` is False, so the "non-positive frequencies" guard never saw a NaN."""

    @staticmethod
    def write(tmp_path: Path, body: str, name: str = "f.txt") -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    @pytest.mark.parametrize("token", ["nan", "NaN", "NA"])
    def test_a_nan_frequency_is_refused(self, tmp_path, token):
        path = self.write(tmp_path, f"I10K {token}\nF15D 0.5\n")
        with pytest.raises(ValueError, match="non-finite frequencies"):
            R.load_frequency_file(path)

    @pytest.mark.parametrize("token", ["inf", "-inf", "Infinity"])
    def test_an_infinite_frequency_is_refused(self, tmp_path, token):
        path = self.write(tmp_path, f"I10K {token}\nF15D 0.5\n")
        with pytest.raises(ValueError, match="non-finite frequencies"):
            R.load_frequency_file(path)

    def test_a_non_numeric_frequency_gets_a_message_not_a_typeerror(self, tmp_path):
        """It used to die inside pandas with ``'<=' not supported between str and int``."""
        path = self.write(tmp_path, "I10K abc\nF15D 0.5\n")
        with pytest.raises(ValueError, match="non-numeric frequencies"):
            R.load_frequency_file(path)

    def test_the_message_names_the_offending_mutants(self, tmp_path):
        path = self.write(tmp_path, "I10K nan\nF15D 0.5\nQ20E inf\n")
        with pytest.raises(ValueError) as excinfo:
            R.load_frequency_file(path)
        assert "I10K" in str(excinfo.value) or "Q20E" in str(excinfo.value)

    def test_valid_frequencies_still_load_including_exactly_one(self, tmp_path):
        """A variant fixed in the parent panel is legal: ``log10(1.0)`` is ``0.0``.

        It must NOT collide with :data:`run_escott.NO_FREQUENCY_SENTINEL` (999.0), and it
        must still be recognised as a frequency-carrying cell.
        """
        path = self.write(tmp_path, "I10K 1.0\nF15D 0.5\n")
        assert R.load_frequency_file(path) == pytest.approx({"I10K": 1.0, "F15D": 0.5})

    @pytest.mark.parametrize("value", ["0.0", "-0.1"])
    def test_non_positive_frequencies_are_still_refused(self, tmp_path, value):
        path = self.write(tmp_path, f"I10K {value}\n")
        with pytest.raises(ValueError, match="non-positive frequencies"):
            R.load_frequency_file(path)

    def test_a_nan_frequency_used_to_vanish_from_every_equation(self, parsed_matrix):
        """Why it mattered: silent, and counted in the denominator anyway.

        Driving ``build_log10_frequency_matrix`` with a NaN directly (bypassing the now-
        fixed reader) reproduces the old behaviour: the mutant is 'matched', it is counted
        in ``n_mutants_with_frequency``, and every equation leaves its score untouched,
        because ``nan > Fc`` and ``nan <= Fc`` are both False.
        """
        wt = R.escott_wt_sequence(parsed_matrix)
        mutant = f"{wt[9]}10{'A' if wt[9] != 'A' else 'C'}"
        frequency, report = R.build_log10_frequency_matrix(
            {mutant: float("nan")}, parsed_matrix
        )
        assert report["n_matched"] == 1

        ranked = R.escott_rank_scores(parsed_matrix)
        penalised = R.apply_prescott_equation(ranked, frequency, 1.0, -2.5, equation=2)
        assert penalised.to_numpy(dtype=float) == pytest.approx(ranked.to_numpy(dtype=float))
        clipping = R.count_clipped_to_zero(ranked, penalised, frequency)
        assert clipping["n_mutants_with_frequency"] == 1
        assert clipping["n_clipped_to_zero"] == 0


# =========================================================================== #
# Properties that were checked and found SOUND -- pinned so they stay that way.
# =========================================================================== #

class TestNumericPropertiesThatHold:
    """Hazards from the brief that were investigated empirically and did NOT fail.

    They are pinned rather than dropped because each is one edit away from breaking, and
    because a reader deserves to know they were tested rather than assumed.
    """

    def test_a_frequency_of_exactly_one_does_not_collide_with_the_sentinel(self):
        """``log10(1.0) == 0.0`` and the sentinel is 999.0, so 'fixed' != 'absent'."""
        assert R.NO_FREQUENCY_SENTINEL == 999.0
        assert float(np.log10(1.0)) != R.NO_FREQUENCY_SENTINEL

    def test_a_fixed_variant_is_penalised_the_maximum_under_equation_2(self, parsed_matrix):
        """Frequency 1.0 must attract the full coefficient, not zero."""
        wt = R.escott_wt_sequence(parsed_matrix)
        mutant = f"{wt[9]}10{'A' if wt[9] != 'A' else 'C'}"
        frequency, _ = R.build_log10_frequency_matrix({mutant: 1.0}, parsed_matrix)
        ranked = R.escott_rank_scores(parsed_matrix)
        cutoff, coefficient = -2.5, 0.4
        penalised = R.apply_prescott_equation(
            ranked, frequency, coefficient, cutoff, equation=2
        )
        delta = ranked.to_numpy(dtype=float) - penalised.to_numpy(dtype=float)
        # penalty = c*(Fc - 0)/Fc = c exactly, unless the raw score clipped at 0 first.
        assert float(np.nanmax(delta)) == pytest.approx(coefficient, abs=1e-12)

    def test_the_zero_coefficient_ablation_is_the_identity(
        self, parsed_matrix, frequency_file_factory
    ):
        """``c = 0`` must reproduce ESCOTT exactly, or the ablation is not an ablation."""
        frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        remapped = R.prescott_v2_scores(parsed_matrix, frequency, 0.0, -2.5, equation=2)
        assert remapped.to_numpy(dtype=float) == pytest.approx(
            R.fill_wildtype(parsed_matrix).to_numpy(dtype=float), abs=1e-12
        )

    def test_the_softmax_survives_the_csv_round_trip_at_every_usable_temperature(
        self, parsed_matrix, tmp_path
    ):
        """Probabilities must come back positive, and columns must still sum to 1.

        ``write_score_matrix`` uses pandas' default float formatting, which round-trips a
        float64 exactly; a ``float_format`` added here would silently create zeros and
        ``log(0) = -inf`` in the alpha sweep.
        """
        protein = R.escott_wt_sequence(parsed_matrix)
        for temperature in (1.0, 0.1, 0.05):
            probabilities = R.escott_to_probability(parsed_matrix, temperature=temperature)
            out = R.write_score_matrix(
                probabilities, protein, tmp_path / f"m_{temperature}.csv"
            )
            back = pd.read_csv(out, index_col=0, header=None).iloc[1:].astype(float)
            assert (back.to_numpy() > 0.0).all()
            assert back.to_numpy().sum(axis=0) == pytest.approx(1.0, abs=1e-12)

    def test_the_depth_scaled_cutoff_makes_a_singleton_free_at_median_depth(self):
        """``Fc = log10(1/N)`` => the equation-2 penalty is ``c * log_N(count)``.

        Zero for a singleton, exactly ``c`` for a fixed variant, at any panel depth.  The
        real panels span 229 (G.1) to 27452 (J.2_int) sequences, so this is what makes
        their PRESCOTT scores comparable at all; checked at both extremes.
        """
        for depth in (229, 27452):
            cutoff = float(np.log10(1.0 / depth))
            for count, expected in ((1, 0.0), (depth, 1.0)):
                frequency = float(np.log10(count / depth))
                penalty = (cutoff - frequency) / cutoff if frequency > cutoff else 0.0
                assert penalty == pytest.approx(expected, abs=1e-12)

    def test_the_same_count_is_not_equally_penalised_across_panel_depths(self):
        """The honest limit of that design, recorded rather than claimed away.

        Only the two anchors (singleton, fixed) are depth-free.  In between the penalty is
        ``log_N(count)``, so a doubleton costs 0.128c in the 229-sequence G.1 panel and
        0.068c in the 27452-sequence J.2_int panel -- a 1.9x difference.  Anyone comparing
        mid-frequency variants across lineages needs to know that.
        """
        def penalty(count: int, depth: int) -> float:
            return float(np.log(count) / np.log(depth))

        shallow, deep = penalty(2, 229), penalty(2, 27452)
        assert shallow == pytest.approx(0.1275, abs=1e-3)
        assert deep == pytest.approx(0.0679, abs=1e-3)
        assert shallow / deep == pytest.approx(1.878, abs=1e-2)

    def test_the_column_depth_spread_of_the_real_panels_is_negligible(self):
        """The other half of that check: ``median_depth`` is a fair stand-in for ``N``.

        The penalty uses each column's own depth, not the median, so a shallow column's
        singleton would be penalised as if it were frequent.  Measured on the real aligned
        panels the worst case is 0.001c, i.e. nothing -- which is why no fix was made for
        it.  Pinned so that a future panel with real coverage gaps is noticed.
        """
        panel = Path(
            "/home3/oml4h/PLM_SARS-CoV-2/Sequences/gisaid_data/alignment_based_19feb26"
            "/hard/H3N2_G.1_hard_nextle2_max5.fasta"
        )
        if not panel.exists():
            pytest.skip(f"{panel} is not present")
        sequences: List[str] = []
        current: List[str] = []
        for line in panel.read_text().splitlines():
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line.strip())
        if current:
            sequences.append("".join(current))
        grid = np.frombuffer("".join(sequences).encode(), dtype=np.uint8).reshape(
            len(sequences), -1
        )
        residues = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
        depths = np.isin(grid, residues).sum(axis=0)
        depths = depths[depths > 0]
        median = float(np.median(depths))
        cutoff = float(np.log10(1.0 / median))
        frequencies = np.log10(1.0 / depths)
        penalties = np.where(frequencies > cutoff, (cutoff - frequencies) / cutoff, 0.0)
        assert float(penalties.max()) < 0.01
