#!/usr/bin/env python3
"""The immune-landscape weighting: recency, period normalisation, diversity.

The synthetic landscape (see ``conftest``) is four 2020 sequences and one 2022
sequence, scored as of 2023.0 with a 1-year half-life, so the raw recency
weights are 0.125 and 0.5 and every expected number below is an exact decimal
worked out in the docstrings rather than read off the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plant_population_escape as pop
from conftest import AS_OF_2023

pytestmark = pytest.mark.unit


def weights_for(backgrounds, **overrides):
    parameters = {
        "as_of": AS_OF_2023,
        "half_life": 1.0,
        "normalise_by": "none",
        "max_age": None,
    }
    parameters.update(overrides)
    return pop.immune_weights(backgrounds, **parameters)


class TestNormalisationAndBasics:
    def test_weights_sum_to_one(self, loaded_backgrounds):
        _, weights, _ = weights_for(loaded_backgrounds)
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_are_all_positive(self, loaded_backgrounds):
        _, weights, _ = weights_for(loaded_backgrounds)
        assert (weights > 0).all()

    def test_coordinates_line_up_with_weights(self, loaded_backgrounds):
        coordinates, weights, _ = weights_for(loaded_backgrounds)
        assert len(coordinates) == len(weights) == 5
        assert coordinates.shape[1] == 3


class TestRecency:
    def test_one_half_life_apart_is_a_factor_of_two(self, loaded_backgrounds):
        """2020 raw 0.5**3 = 0.125, 2022 raw 0.5**1 = 0.5 -> a factor of 4 for 2 years."""
        _, weights, _ = weights_for(loaded_backgrounds)
        twenty_twenty = weights[:4]
        twenty_twenty_two = weights[4]
        assert twenty_twenty_two / twenty_twenty[0] == pytest.approx(4.0)

    def test_per_sequence_mode_gives_the_documented_50_50_split(self, loaded_backgrounds):
        """4 x 0.125 = 0.5 against 1 x 0.5 = 0.5."""
        _, weights, _ = weights_for(loaded_backgrounds)
        assert weights[:4].sum() == pytest.approx(0.5)
        assert weights[4] == pytest.approx(0.5)

    def test_longer_half_life_flattens_the_landscape(self, loaded_backgrounds):
        _, sharp, _ = weights_for(loaded_backgrounds, half_life=0.25)
        _, flat, _ = weights_for(loaded_backgrounds, half_life=50.0)
        assert sharp[4] > flat[4]
        assert flat.max() - flat.min() < sharp.max() - sharp.min()

    def test_half_life_must_be_positive(self, loaded_backgrounds):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="half-life"):
                weights_for(loaded_backgrounds, half_life=bad)


class TestStrictlyBefore:
    def test_a_sequence_dated_exactly_as_of_is_excluded(self, loaded_backgrounds):
        """`before a date` must not include the date itself."""
        coordinates, _, diagnostics = weights_for(loaded_backgrounds, as_of=2022.0)
        assert diagnostics["n_sequences"] == 4
        assert coordinates.shape[0] == 4

    def test_nothing_before_the_landscape_raises(self, loaded_backgrounds):
        with pytest.raises(ValueError, match="No background sequences"):
            weights_for(loaded_backgrounds, as_of=2019.0)

    def test_max_age_trims_the_tail(self, loaded_backgrounds):
        _, _, diagnostics = weights_for(loaded_backgrounds, max_age=2.0)
        assert diagnostics["n_sequences"] == 1  # only the 2022 sequence is <= 2 y old

    def test_max_age_that_excludes_everything_raises(self, loaded_backgrounds):
        with pytest.raises(ValueError, match="within 0.1"):
            weights_for(loaded_backgrounds, max_age=0.1)

    def test_max_age_renormalises(self, loaded_backgrounds):
        _, weights, _ = weights_for(loaded_backgrounds, max_age=2.0)
        assert weights.sum() == pytest.approx(1.0)


class TestPeriodNormalisation:
    def test_year_normalisation_gives_the_documented_20_80_split(self, loaded_backgrounds):
        """Each year contributes its recency weight, not its sequence count.

        2020 -> 0.125, 2022 -> 0.5, so 0.125/0.625 = 0.2 and 0.5/0.625 = 0.8.
        """
        _, weights, _ = weights_for(loaded_backgrounds, normalise_by="year")
        assert weights[:4].sum() == pytest.approx(0.2)
        assert weights[4] == pytest.approx(0.8)

    def test_year_normalisation_is_immune_to_duplicating_a_year(self, loaded_backgrounds):
        """Sequencing 2020 ten times harder must not buy 2020 more immunity."""
        inflated = pd.concat(
            [loaded_backgrounds] + [loaded_backgrounds[loaded_backgrounds["decimal_year"] < 2021]] * 9,
            ignore_index=True,
        )
        _, weights, _ = weights_for(inflated, normalise_by="year")
        recent = weights[inflated["decimal_year"].to_numpy() > 2021].sum()
        assert recent == pytest.approx(0.8)

    def test_per_sequence_mode_is_not_immune_to_it(self, loaded_backgrounds):
        """The contrast that makes the flag worth having."""
        inflated = pd.concat(
            [loaded_backgrounds] + [loaded_backgrounds[loaded_backgrounds["decimal_year"] < 2021]] * 9,
            ignore_index=True,
        )
        _, weights, _ = weights_for(inflated, normalise_by="none")
        recent = weights[inflated["decimal_year"].to_numpy() > 2021].sum()
        assert recent < 0.15

    def test_month_normalisation_splits_finer(self, loaded_backgrounds):
        _, weights, _ = weights_for(loaded_backgrounds, normalise_by="month")
        assert weights.sum() == pytest.approx(1.0)
        # All four 2020 rows share one month, so month and year agree here.
        assert weights[:4].sum() == pytest.approx(0.2)


class TestWithinPeriodDiversity:
    """The question 'does it weight a year's sequences by diversity?'.

    In the 2020 block, rows 2 and 3 sit on *identical* coordinates and rows 1
    and 4 are distinct, so 2020 holds 3 distinct antigenic positions among 4
    sequences.
    """

    def test_abundance_splits_a_year_evenly_per_sequence(self, loaded_backgrounds):
        """Default: each of the four 2020 rows gets 0.2/4 = 0.05."""
        _, weights, _ = weights_for(
            loaded_backgrounds, normalise_by="year", within_period="abundance"
        )
        assert weights[:4] == pytest.approx([0.05, 0.05, 0.05, 0.05])

    def test_unique_splits_a_year_evenly_per_distinct_position(self, loaded_backgrounds):
        """Diversity: 3 distinct positions, so each gets 0.2/3; the tied pair
        shares its third, i.e. 0.2/6 each."""
        _, weights, _ = weights_for(
            loaded_backgrounds, normalise_by="year", within_period="unique"
        )
        third = 0.2 / 3
        assert weights[0] == pytest.approx(third)
        assert weights[1] == pytest.approx(third / 2)
        assert weights[2] == pytest.approx(third / 2)
        assert weights[3] == pytest.approx(third)
        assert weights[:4].sum() == pytest.approx(0.2)

    def test_unique_does_not_change_the_between_year_split(self, loaded_backgrounds):
        _, weights, _ = weights_for(
            loaded_backgrounds, normalise_by="year", within_period="unique"
        )
        assert weights[:4].sum() == pytest.approx(0.2)
        assert weights[4] == pytest.approx(0.8)

    def test_duplicating_one_strain_does_not_change_unique_weights(self, loaded_backgrounds):
        """The whole point: 100 deposits of one phenotype != 100x the immunity."""
        duplicated = pd.concat(
            [loaded_backgrounds, pd.concat([loaded_backgrounds.iloc[[1]]] * 99)],
            ignore_index=True,
        )
        _, base, _ = weights_for(
            loaded_backgrounds, normalise_by="year", within_period="unique"
        )
        _, swollen, _ = weights_for(
            duplicated, normalise_by="year", within_period="unique"
        )
        # The isolated 2020 strains keep exactly their old weight.
        assert swollen[0] == pytest.approx(base[0])
        assert swollen[3] == pytest.approx(base[3])
        # The duplicated position still holds one third of 2020 in total.
        tied = swollen[1] + swollen[2] + swollen[5:].sum()
        assert tied == pytest.approx(0.2 / 3)

    def test_abundance_is_badly_affected_by_the_same_duplication(self, loaded_backgrounds):
        duplicated = pd.concat(
            [loaded_backgrounds, pd.concat([loaded_backgrounds.iloc[[1]]] * 99)],
            ignore_index=True,
        )
        _, weights, _ = weights_for(
            duplicated, normalise_by="year", within_period="abundance"
        )
        tied = weights[1] + weights[2] + weights[5:].sum()
        assert tied > 0.19  # nearly all of 2020's weight, on one phenotype

    def test_density_collapses_near_identical_positions(self, loaded_backgrounds):
        """A radius wide enough to swallow the whole 2020 block leaves 1 cell."""
        nudged = loaded_backgrounds.copy()
        nudged.loc[0, "X"] = 0.01  # distinct coordinates, same neighbourhood
        _, weights, _ = weights_for(
            nudged, normalise_by="year", within_period="density", density_radius=100.0
        )
        assert weights[:4].sum() == pytest.approx(0.2)
        assert weights[:4] == pytest.approx([0.05, 0.05, 0.05, 0.05])

    def test_tiny_density_radius_matches_unique(self, loaded_backgrounds):
        _, by_density, _ = weights_for(
            loaded_backgrounds, normalise_by="year",
            within_period="density", density_radius=1e-9,
        )
        _, by_unique, _ = weights_for(
            loaded_backgrounds, normalise_by="year", within_period="unique"
        )
        assert by_density == pytest.approx(by_unique)

    def test_within_period_without_a_period_is_refused(self, loaded_backgrounds):
        """Silently ignoring the flag would be worse than refusing it."""
        with pytest.raises(ValueError, match="only has meaning inside a period"):
            weights_for(loaded_backgrounds, normalise_by="none", within_period="unique")

    def test_unknown_mode_raises(self, loaded_backgrounds):
        with pytest.raises(ValueError, match="within-period"):
            weights_for(loaded_backgrounds, normalise_by="year", within_period="nonsense")

    def test_negative_density_radius_raises(self, loaded_backgrounds):
        with pytest.raises(ValueError, match="density-radius"):
            weights_for(
                loaded_backgrounds, normalise_by="year",
                within_period="density", density_radius=-1.0,
            )


class TestWithinPeriodShareDirectly:
    def test_shares_sum_to_one_per_period(self):
        coordinates = np.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [5, 0, 0]])
        periods = np.array([2020.0, 2020.0, 2020.0, 2021.0])
        for mode in ("abundance", "unique", "density"):
            share = pop.within_period_share(coordinates, periods, mode, 0.25)
            for period in np.unique(periods):
                assert share[periods == period].sum() == pytest.approx(1.0)

    def test_abundance_is_uniform(self):
        coordinates = np.zeros((4, 3))
        periods = np.zeros(4)
        share = pop.within_period_share(coordinates, periods, "abundance", 0.25)
        assert share == pytest.approx([0.25] * 4)

    def test_unique_of_all_identical_is_also_uniform(self):
        """One position shared four ways is still 1/4 each -- but for a different
        reason, and it stops being 1/4 as soon as a second position appears."""
        coordinates = np.zeros((4, 3))
        periods = np.zeros(4)
        assert pop.within_period_share(coordinates, periods, "unique", 0.25) == pytest.approx(
            [0.25] * 4
        )
        coordinates = np.array([[0.0, 0, 0], [0, 0, 0], [0, 0, 0], [9, 0, 0]])
        share = pop.within_period_share(coordinates, periods, "unique", 0.25)
        assert share == pytest.approx([1 / 6, 1 / 6, 1 / 6, 0.5])


class TestDiagnostics:
    def test_effective_sample_size_of_uniform_weights_is_n(self):
        """Kish ESS: 1 / sum(w^2) with w = 1/N is exactly N."""
        frame = pd.DataFrame(
            {
                "name": [f"s{i}" for i in range(8)],
                "collection_date": ["2022-01-01"] * 8,
                "decimal_year": [2022.0] * 8,
                "X": np.arange(8.0), "Y": np.zeros(8), "Z": np.zeros(8),
            }
        )
        _, _, diagnostics = weights_for(frame, as_of=2023.0)
        assert diagnostics["effective_sample_size"] == pytest.approx(8.0)

    def test_effective_sample_size_falls_when_one_sequence_dominates(self, loaded_backgrounds):
        _, _, flat = weights_for(loaded_backgrounds, half_life=100.0)
        _, _, peaked = weights_for(loaded_backgrounds, half_life=0.05)
        assert peaked["effective_sample_size"] < flat["effective_sample_size"]
        assert peaked["effective_sample_size"] == pytest.approx(1.0, abs=0.01)

    def test_weighted_mean_age(self, loaded_backgrounds):
        """0.5 * 3 y + 0.5 * 1 y = 2 y, from the 50/50 per-sequence split."""
        _, _, diagnostics = weights_for(loaded_backgrounds)
        assert diagnostics["weighted_mean_age_years"] == pytest.approx(2.0)

    def test_weighted_centroid(self, loaded_backgrounds):
        """Per-sequence: 2020 block is (0,0,0),(6,0,0),(6,0,0),(0,6,0) each at
        0.125, and 2022 is (0,0,6) at 0.5 -> (1.5, 0.75, 3.0)."""
        _, _, diagnostics = weights_for(loaded_backgrounds)
        assert diagnostics["weighted_centroid"] == pytest.approx([1.5, 0.75, 3.0])

    def test_collection_range_is_reported(self, loaded_backgrounds):
        _, _, diagnostics = weights_for(loaded_backgrounds)
        assert diagnostics["earliest_collection"] == "2020-01-01"
        assert diagnostics["latest_collection"] == "2022-01-01"

    def test_weight_by_year_table(self, loaded_backgrounds):
        _, _, diagnostics = weights_for(loaded_backgrounds)
        by_year = diagnostics["weight_by_year"]
        assert list(by_year.index) == [2020, 2022]
        assert by_year.loc[2020, "n_sequences"] == 4
        assert by_year.loc[2022, "n_sequences"] == 1
        assert by_year["weight"].sum() == pytest.approx(1.0)

    def test_diagnostics_are_json_safe_once_the_table_is_removed(self, loaded_backgrounds):
        """The metadata JSON dumps **diagnostics, so nothing else may be a frame."""
        import json

        _, _, diagnostics = weights_for(loaded_backgrounds)
        diagnostics.pop("weight_by_year")
        json.dumps(diagnostics)  # must not raise
