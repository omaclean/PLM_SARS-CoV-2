#!/usr/bin/env python3
"""The stationary-point escape geometry in ``plant_order_scan``.

Every expectation here comes from the closed forms documented in ``conftest``,
never from re-running the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plant_order_scan as scan
from conftest import (
    EXPECTED_PAIR_EPISTASIS,
    EXPECTED_SINGLES,
    PLANTED_EPISTASIS_VECTOR,
    SPAN,
)

pytestmark = pytest.mark.unit


class TestSplitGenotypeLabel:
    def test_root_is_the_empty_set(self):
        assert scan.split_genotype_label("root") == ()

    def test_single_mutation(self):
        assert scan.split_genotype_label("N122D") == ("N122D",)

    def test_order_is_preserved(self):
        assert scan.split_genotype_label("N122D+T135K+K189R") == (
            "N122D", "T135K", "K189R",
        )


class TestEscapeBasis:
    def test_span_and_unit_vector(self, synthetic_genotypes):
        root, unit, span = scan.escape_basis(synthetic_genotypes)
        assert root == pytest.approx([0.0, 0.0, 0.0])
        assert span == pytest.approx(SPAN)
        assert unit == pytest.approx([1 / 3, 2 / 3, 2 / 3])
        assert np.linalg.norm(unit) == pytest.approx(1.0)

    def test_row_order_does_not_matter(self, synthetic_genotypes):
        shuffled = synthetic_genotypes.sample(frac=1.0, random_state=0).reset_index(drop=True)
        for before, after in zip(scan.escape_basis(synthetic_genotypes),
                                 scan.escape_basis(shuffled)):
            assert np.allclose(before, after)

    def test_degenerate_axis_raises(self, synthetic_genotypes):
        """A full mutant on top of the root has no axis to project onto."""
        collapsed = synthetic_genotypes.copy()
        top = collapsed["n_fixed"].idxmax()
        collapsed.loc[top, ["X", "Y", "Z"]] = 0.0
        with pytest.raises(ValueError, match="no"):
            scan.escape_basis(collapsed)


class TestSingleMutantEscape:
    @pytest.fixture(autouse=True)
    def _tables(self, synthetic_genotypes):
        self.singles, self.pairs = scan.build_escape_tables(synthetic_genotypes)
        self.by_name = self.singles.set_index("mutation_h3")

    def test_every_mutation_present_exactly_once(self):
        assert sorted(self.singles["mutation_h3"]) == sorted(EXPECTED_SINGLES)
        assert len(self.singles) == len(EXPECTED_SINGLES)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SINGLES))
    def test_closed_form_components(self, name):
        total, along, off = EXPECTED_SINGLES[name]
        row = self.by_name.loc[name]
        assert row["escape_total"] == pytest.approx(total)
        assert row["escape_along_axis"] == pytest.approx(along)
        assert row["escape_off_axis"] == pytest.approx(off)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SINGLES))
    def test_components_are_orthogonal(self, name):
        """total^2 == along^2 + off^2, or the decomposition is not a projection."""
        row = self.by_name.loc[name]
        assert row["escape_total"] ** 2 == pytest.approx(
            row["escape_along_axis"] ** 2 + row["escape_off_axis"] ** 2
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_SINGLES))
    def test_fraction_of_endpoint(self, name):
        _, along, _ = EXPECTED_SINGLES[name]
        assert self.by_name.loc[name, "frac_of_endpoint"] == pytest.approx(along / SPAN)

    def test_fractions_sum_to_one_for_this_additive_endpoint(self):
        """1/9 + 4/9 + 4/9 = 1: the three singles reach the endpoint on-axis."""
        assert self.singles["frac_of_endpoint"].sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SINGLES))
    def test_off_axis_fraction_is_a_fraction(self, name):
        total, _, off = EXPECTED_SINGLES[name]
        row = self.by_name.loc[name]
        assert row["off_axis_fraction"] == pytest.approx(off / total)
        assert 0.0 <= row["off_axis_fraction"] <= 1.0


class TestPairwiseEscape:
    @pytest.fixture(autouse=True)
    def _tables(self, synthetic_genotypes):
        self.singles, self.pairs = scan.build_escape_tables(synthetic_genotypes)
        self.by_pair = self.pairs.set_index("pair_h3")

    def test_all_three_pairs_present(self):
        assert sorted(self.pairs["pair_h3"]) == sorted(EXPECTED_PAIR_EPISTASIS)

    @pytest.mark.parametrize("pair", sorted(EXPECTED_PAIR_EPISTASIS))
    def test_epistasis_matches_the_planted_value(self, pair):
        assert self.by_pair.loc[pair, "epistasis_along_axis"] == pytest.approx(
            EXPECTED_PAIR_EPISTASIS[pair], abs=1e-12
        )

    def test_additive_pairs_are_exactly_zero(self):
        """Not 'small' -- exactly zero, because the geometry is exactly additive."""
        additive = self.by_pair.loc[["N122D+K189R", "T135K+K189R"]]
        assert additive["epistasis_along_axis"].abs().max() < 1e-12
        assert additive["epistasis_magnitude"].max() < 1e-12

    def test_epistasis_magnitude_is_the_planted_vector_length(self):
        assert self.by_pair.loc["N122D+T135K", "epistasis_magnitude"] == pytest.approx(
            np.linalg.norm(PLANTED_EPISTASIS_VECTOR)
        )

    def test_additive_expectation_is_the_sum_of_the_singles(self):
        by_name = self.singles.set_index("mutation_h3")["escape_along_axis"]
        for row in self.pairs.itertuples():
            assert row.additive_along_axis == pytest.approx(
                by_name[row.mutation_a] + by_name[row.mutation_b]
            )

    def test_epistasis_is_observed_minus_additive(self):
        for row in self.pairs.itertuples():
            assert row.epistasis_along_axis == pytest.approx(
                row.escape_along_axis - row.additive_along_axis
            )

    def test_pairwise_epistasis_is_not_inferred_from_the_endpoint(self):
        """The triple is exactly additive while one pair is not.

        Anything that derives pairwise terms from the full mutant rather than
        measuring each double would report zero here.
        """
        endpoint_residual = (
            self.singles["escape_along_axis"].sum() - SPAN
        )
        assert endpoint_residual == pytest.approx(0.0, abs=1e-12)
        assert self.by_pair["epistasis_along_axis"].abs().max() > 0.1

    def test_relative_epistasis(self):
        for row in self.pairs.itertuples():
            if row.additive_along_axis:
                assert row.relative_epistasis == pytest.approx(
                    row.epistasis_along_axis / row.additive_along_axis
                )

    def test_pairs_are_listed_once_in_canonical_order(self):
        names = list(self.singles["mutation_h3"])
        for row in self.pairs.itertuples():
            assert names.index(row.mutation_a) < names.index(row.mutation_b)


class TestDegradedInputs:
    def test_no_two_mutation_backgrounds_returns_none(self, pairless_genotypes):
        singles, pairs = scan.build_escape_tables(pairless_genotypes)
        assert pairs is None
        assert len(singles) == 3

    def test_extra_columns_are_ignored(self, synthetic_genotypes):
        """The real CSV carries dist_to_start etc.; they must not be required."""
        padded = synthetic_genotypes.assign(dist_to_start=-1.0, axis_progress=99.0)
        singles, pairs = scan.build_escape_tables(padded)
        assert singles["escape_total"].to_list() == pytest.approx(
            scan.build_escape_tables(synthetic_genotypes)[0]["escape_total"].to_list()
        )
        assert pairs is not None

    def test_missing_root_raises(self, synthetic_genotypes):
        rootless = synthetic_genotypes[synthetic_genotypes["n_fixed"] > 0]
        # With no root the lowest n_fixed is a single mutant, so the basis is
        # built from the wrong origin -- catch it as a wrong answer, not a crash.
        singles, _ = scan.build_escape_tables(rootless)
        assert singles["escape_total"].min() >= 0.0

    def test_single_mutation_set_has_no_pairs(self):
        frame = pd.DataFrame(
            {
                "genotype_id": ["root", "N122D"],
                "genotype_h3": ["root", "N122D"],
                "n_fixed": [0, 1],
                "X": [0.0, 1.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0],
            }
        )
        singles, pairs = scan.build_escape_tables(frame)
        assert pairs is None
        assert singles["escape_along_axis"].iloc[0] == pytest.approx(1.0)

    def test_empty_singles_raises(self):
        frame = pd.DataFrame(
            {
                "genotype_id": ["root", "A+B"],
                "genotype_h3": ["root", "A+B"],
                "n_fixed": [0, 2],
                "X": [0.0, 1.0], "Y": [0.0, 0.0], "Z": [0.0, 0.0],
            }
        )
        with pytest.raises(ValueError, match="single-mutant"):
            scan.build_escape_tables(frame)


class TestInvariants:
    def test_escape_is_translation_invariant(self, synthetic_genotypes):
        """Shifting the whole map cannot change any escape number."""
        shifted = synthetic_genotypes.copy()
        shifted[["X", "Y", "Z"]] += np.array([100.0, -50.0, 7.5])
        base_singles, base_pairs = scan.build_escape_tables(synthetic_genotypes)
        moved_singles, moved_pairs = scan.build_escape_tables(shifted)
        for column in ("escape_total", "escape_along_axis", "escape_off_axis"):
            assert moved_singles[column].to_numpy() == pytest.approx(
                base_singles[column].to_numpy()
            )
        assert moved_pairs["epistasis_along_axis"].to_numpy() == pytest.approx(
            base_pairs["epistasis_along_axis"].to_numpy()
        )

    def test_escape_is_rotation_invariant(self, synthetic_genotypes):
        """Distances and projections live in the geometry, not in PLANT's axes."""
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated = synthetic_genotypes.copy()
        rotated[["X", "Y", "Z"]] = synthetic_genotypes[["X", "Y", "Z"]].to_numpy() @ rotation.T
        base, _ = scan.build_escape_tables(synthetic_genotypes)
        turned, _ = scan.build_escape_tables(rotated)
        assert turned["escape_total"].to_numpy() == pytest.approx(
            base["escape_total"].to_numpy()
        )
        assert turned["escape_along_axis"].to_numpy() == pytest.approx(
            base["escape_along_axis"].to_numpy()
        )

    def test_uniform_scaling_scales_escape_linearly(self, synthetic_genotypes):
        scaled = synthetic_genotypes.copy()
        scaled[["X", "Y", "Z"]] *= 3.0
        base, _ = scan.build_escape_tables(synthetic_genotypes)
        bigger, _ = scan.build_escape_tables(scaled)
        assert bigger["escape_total"].to_numpy() == pytest.approx(
            3.0 * base["escape_total"].to_numpy()
        )
        # ...but the fraction of the endpoint is scale-free.
        assert bigger["frac_of_endpoint"].to_numpy() == pytest.approx(
            base["frac_of_endpoint"].to_numpy()
        )
