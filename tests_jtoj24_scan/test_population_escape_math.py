#!/usr/bin/env python3
"""Population-escape scoring and the single/pair decomposition built on it.

``population_escape`` computes its distances with the expanded ``|a-b|^2``
identity and a chunk loop, both of which are easy to get subtly wrong, so it is
checked against a naive reference implementation written here rather than
against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plant_population_escape as pop

pytestmark = pytest.mark.unit

SCALE = 2.0


def reference_escape(points, background, weights, scale=SCALE, kernel="exponential"):
    """The definition, written the slow obvious way."""
    out = []
    for point in points:
        total = 0.0
        for coordinate, weight in zip(background, weights):
            distance = float(np.sqrt(((point - coordinate) ** 2).sum()))
            total += weight * (1.0 - float(pop.cross_immunity(np.array([distance]),
                                                              scale, kernel)[0]))
        out.append(total)
    return np.array(out)


@pytest.fixture
def landscape():
    background = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 9.0]]
    )
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    return background, weights


class TestAgainstAReferenceImplementation:
    def test_matches_the_naive_loop(self, landscape, synthetic_genotypes):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        assert pop.population_escape(points, background, weights, SCALE, "exponential") == \
            pytest.approx(reference_escape(points, background, weights))

    @pytest.mark.parametrize("kernel", ["exponential", "sigmoid", "linear"])
    def test_matches_for_every_kernel(self, landscape, synthetic_genotypes, kernel):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        assert pop.population_escape(points, background, weights, SCALE, kernel) == \
            pytest.approx(reference_escape(points, background, weights, kernel=kernel))

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 1000])
    def test_chunking_does_not_change_the_answer(self, landscape, synthetic_genotypes, chunk):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        whole = pop.population_escape(points, background, weights, SCALE, "exponential", 8)
        chunked = pop.population_escape(points, background, weights, SCALE,
                                        "exponential", chunk)
        assert chunked == pytest.approx(whole)

    def test_zero_chunk_size_does_not_hang(self, landscape, synthetic_genotypes):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        assert len(pop.population_escape(points, background, weights, SCALE,
                                         "exponential", 0)) == len(points)


class TestClosedForms:
    def test_sitting_on_the_only_strain_means_zero_escape(self):
        background = np.array([[1.0, 2.0, 3.0]])
        weights = np.array([1.0])
        escape = pop.population_escape(np.array([[1.0, 2.0, 3.0]]), background,
                                       weights, SCALE, "exponential")
        assert escape[0] == pytest.approx(0.0)

    def test_one_strain_landscape_is_the_kernel_itself(self):
        background = np.array([[0.0, 0.0, 0.0]])
        weights = np.array([1.0])
        point = np.array([[3.0, 0.0, 0.0]])
        escape = pop.population_escape(point, background, weights, SCALE, "exponential")
        assert escape[0] == pytest.approx(1.0 - np.exp(-3.0 / SCALE))

    def test_two_identical_strains_equal_one(self):
        point = np.array([[3.0, 0.0, 0.0]])
        single = pop.population_escape(point, np.array([[0.0, 0, 0]]), np.array([1.0]),
                                       SCALE, "exponential")
        doubled = pop.population_escape(point, np.array([[0.0, 0, 0], [0.0, 0, 0]]),
                                        np.array([0.5, 0.5]), SCALE, "exponential")
        assert doubled == pytest.approx(single)

    def test_escape_is_bounded_in_zero_one(self, landscape, synthetic_genotypes):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        assert escape.min() >= 0.0
        assert escape.max() <= 1.0

    def test_moving_away_from_a_one_point_landscape_increases_escape(self):
        background = np.array([[0.0, 0.0, 0.0]])
        weights = np.array([1.0])
        points = np.array([[d, 0.0, 0.0] for d in np.linspace(0.0, 10.0, 25)])
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        assert np.all(np.diff(escape) > 0)

    def test_translation_invariance(self, landscape):
        background, weights = landscape
        point = np.array([[1.0, 1.0, 1.0]])
        shift = np.array([100.0, -7.0, 3.0])
        base = pop.population_escape(point, background, weights, SCALE, "exponential")
        moved = pop.population_escape(point + shift, background + shift, weights,
                                      SCALE, "exponential")
        assert moved == pytest.approx(base)


class TestCheckSharedFrame:
    def test_finds_the_true_nearest(self, loaded_backgrounds):
        report = pop.check_shared_frame(np.array([0.0, 0.0, 0.1]), loaded_backgrounds)
        assert report["nearest_background"] == "A/Test/1/2020"
        assert report["nearest_subclade"] == "X.1"
        assert report["distance"] == pytest.approx(0.1)

    def test_warns_when_the_frames_look_wrong(self, loaded_backgrounds, capsys):
        pop.check_shared_frame(np.array([500.0, 500.0, 500.0]), loaded_backgrounds)
        assert "same PLANT checkpoint" in capsys.readouterr().out

    def test_stays_quiet_when_close(self, loaded_backgrounds, capsys):
        pop.check_shared_frame(np.array([0.0, 0.0, 0.0]), loaded_backgrounds)
        assert capsys.readouterr().out == ""

    def test_reports_the_collection_date(self, loaded_backgrounds):
        report = pop.check_shared_frame(np.array([0.0, 0.0, 6.0]), loaded_backgrounds)
        assert report["nearest_collection_date"] == "2022-01-01"


class TestBuildPopulationTables:
    @pytest.fixture
    def tables(self, synthetic_genotypes, landscape):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        return pop.build_population_tables(synthetic_genotypes, escape)

    def test_root_gain_is_exactly_zero(self, tables):
        table, _, _ = tables
        root = table[table["n_fixed"] == 0]
        assert root["escape_gain"].iloc[0] == pytest.approx(0.0, abs=1e-15)

    def test_gain_is_escape_minus_root_escape(self, tables):
        table, singles, _ = tables
        root_escape = float(singles["root_escape"].iloc[0])
        assert table["escape_gain"].to_numpy() == pytest.approx(
            table["population_escape"].to_numpy() - root_escape
        )

    def test_every_single_mutation_appears(self, tables):
        _, singles, _ = tables
        assert sorted(singles["mutation_h3"]) == ["K189R", "N122D", "T135K"]

    def test_all_three_pairs_appear(self, tables):
        _, _, pairs = tables
        assert sorted(pairs["pair_h3"]) == [
            "N122D+K189R", "N122D+T135K", "T135K+K189R",
        ]

    def test_epistasis_is_observed_minus_additive(self, tables):
        _, _, pairs = tables
        assert pairs["epistasis"].to_numpy() == pytest.approx(
            (pairs["escape_gain"] - pairs["additive_gain"]).to_numpy()
        )

    def test_additive_gain_is_the_sum_of_the_single_gains(self, tables):
        _, singles, pairs = tables
        by_name = singles.set_index("mutation_h3")["escape_gain"]
        for row in pairs.itertuples():
            assert row.additive_gain == pytest.approx(
                by_name[row.mutation_a] + by_name[row.mutation_b]
            )

    def test_share_of_remaining_immunity(self, tables):
        _, singles, _ = tables
        root_escape = float(singles["root_escape"].iloc[0])
        assert singles["share_of_remaining_immunity"].to_numpy() == pytest.approx(
            (singles["escape_gain"] / (1.0 - root_escape)).to_numpy()
        )

    def test_a_saturated_landscape_leaves_nothing_to_gain(self, synthetic_genotypes):
        """Immunity so distant that everything already escapes it: gains ~ 0."""
        background = np.array([[1000.0, 0.0, 0.0]])
        weights = np.array([1.0])
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        _, singles, _ = pop.build_population_tables(synthetic_genotypes, escape)
        assert singles["escape_gain"].abs().max() < 1e-9
        assert float(singles["root_escape"].iloc[0]) == pytest.approx(1.0)

    def test_a_landscape_on_the_root_makes_every_gain_positive(self, synthetic_genotypes):
        background = np.array([[0.0, 0.0, 0.0]])
        weights = np.array([1.0])
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        _, singles, _ = pop.build_population_tables(synthetic_genotypes, escape)
        assert (singles["escape_gain"] > 0).all()

    def test_a_landscape_beyond_the_endpoint_can_make_gains_negative(self, synthetic_genotypes):
        """Moving towards the immunity must cost escape, not merely gain less."""
        background = np.array([[2.0, 4.0, 4.0]])  # past the endpoint, along the axis
        weights = np.array([1.0])
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        _, singles, _ = pop.build_population_tables(synthetic_genotypes, escape)
        assert (singles["escape_gain"] < 0).all()

    def test_no_pairs_returns_none(self, pairless_genotypes, landscape):
        background, weights = landscape
        points = pairless_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        _, singles, pairs = pop.build_population_tables(pairless_genotypes, escape)
        assert pairs is None
        assert len(singles) == 3

    def test_missing_root_raises(self, synthetic_genotypes, landscape):
        background, weights = landscape
        rootless = synthetic_genotypes[synthetic_genotypes["n_fixed"] > 0].reset_index(drop=True)
        points = rootless[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        with pytest.raises(ValueError, match="No root genotype"):
            pop.build_population_tables(rootless, escape)


class TestCurvatureDecomposition:
    """Splitting ε into the saturating kernel and the actual interaction.

    ``epistasis = observed - (gain_a + gain_b)`` is additivity in *escape*, and
    escape saturates, so a pair that is perfectly additive in coordinates still
    reports a non-zero ε. These tests pin down that the split isolates it.
    """

    @staticmethod
    def scorer(background, weights, kernel="exponential"):
        return lambda points: pop.population_escape(points, background, weights,
                                                    SCALE, kernel)

    @pytest.fixture
    def decomposed(self, synthetic_genotypes, landscape):
        background, weights = landscape
        score = self.scorer(background, weights)
        escape = score(synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float))
        return pop.build_population_tables(synthetic_genotypes, escape, score)

    def test_columns_absent_without_a_scorer(self, synthetic_genotypes, landscape):
        background, weights = landscape
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, background, weights, SCALE, "exponential")
        _, _, pairs = pop.build_population_tables(synthetic_genotypes, escape)
        assert "epistasis_vs_additive_genotype" not in pairs

    def test_the_split_is_exact(self, decomposed):
        """epistasis == kernel_curvature + interaction, to machine precision."""
        _, _, pairs = decomposed
        assert pairs["epistasis"].to_numpy() == pytest.approx(
            (pairs["kernel_curvature"] + pairs["epistasis_vs_additive_genotype"]).to_numpy(),
            abs=1e-15,
        )

    def test_a_coordinate_additive_pair_has_zero_interaction(self, decomposed):
        """N122D+K189R and T135K+K189R are exactly additive in coordinates, so
        their entire ε must land in the curvature term."""
        _, _, pairs = decomposed
        additive = pairs.set_index("pair_h3").loc[["N122D+K189R", "T135K+K189R"]]
        assert additive["epistasis_vs_additive_genotype"].abs().max() < 1e-12
        assert additive["kernel_curvature"].to_numpy() == pytest.approx(
            additive["epistasis"].to_numpy()
        )

    def test_the_non_additive_pair_keeps_a_real_interaction(self, decomposed):
        """N122D+T135K carries a planted (0.5, 0, 0), which no kernel can explain."""
        _, _, pairs = decomposed
        row = pairs.set_index("pair_h3").loc["N122D+T135K"]
        assert abs(row["epistasis_vs_additive_genotype"]) > 1e-6

    def test_curvature_is_nonzero_for_a_saturating_kernel(self, decomposed):
        """If it were zero the decomposition would be pointless."""
        _, _, pairs = decomposed
        assert pairs["kernel_curvature"].abs().max() > 1e-6

    def test_a_linear_kernel_inside_its_range_has_no_curvature(self, synthetic_genotypes):
        """The one kernel that is locally straight in distance -- but only
        because distance itself is still non-linear, so this checks the
        machinery, not that curvature vanishes everywhere."""
        background = np.array([[-50.0, 0.0, 0.0]])
        weights = np.array([1.0])
        score = self.scorer(background, weights, "linear")
        # Everything is far beyond the scale, so escape is flat at 1.0 and every
        # term collapses to zero.
        escape = score(synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float))
        _, _, pairs = pop.build_population_tables(synthetic_genotypes, escape, score)
        assert pairs["kernel_curvature"].abs().max() < 1e-12
        assert pairs["epistasis"].abs().max() < 1e-12

    def test_curvature_grows_as_the_immunity_closes_in(self, synthetic_genotypes):
        """The reported failure mode: a landscape sitting on the start lineage
        makes the kernel term dominate."""
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        weights = np.array([1.0])
        shares = []
        for distance in (0.2, 20.0):
            background = np.array([[-distance, 0.0, 0.0]])
            score = self.scorer(background, weights)
            _, _, pairs = pop.build_population_tables(
                synthetic_genotypes, score(points), score
            )
            shares.append(
                pairs["kernel_curvature"].abs().sum() / pairs["epistasis"].abs().sum()
            )
        assert shares[0] > shares[1]

    def test_curvature_share_column(self, decomposed):
        _, _, pairs = decomposed
        finite = pairs[pairs["epistasis"].abs() > 0]
        assert finite["curvature_share_of_epistasis"].to_numpy() == pytest.approx(
            (finite["kernel_curvature"] / finite["epistasis"]).to_numpy()
        )

    def test_the_scorer_is_called_on_the_predicted_genotypes(self, synthetic_genotypes,
                                                             landscape):
        """The additive prediction is root + Δa + Δb, not the observed double."""
        background, weights = landscape
        seen = []

        def spy(points):
            seen.append(np.asarray(points, dtype=float).copy())
            return pop.population_escape(points, background, weights, SCALE, "exponential")

        escape = pop.population_escape(
            synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float),
            background, weights, SCALE, "exponential",
        )
        pop.build_population_tables(synthetic_genotypes, escape, spy)
        predicted = seen[-1]
        # N122D (1,0,0) + T135K (0,2,0) predicts (1,2,0); the observed double is
        # (1.5, 2, 0), so the two must not coincide.
        assert any(np.allclose(row, [1.0, 2.0, 0.0]) for row in predicted)
        assert not any(np.allclose(row, [1.5, 2.0, 0.0]) for row in predicted)


class TestPopulationVsStationaryDisagree:
    """The reason this module exists at all.

    Stationary escape is a displacement from the root and cannot depend on where
    the immunity is. Population escape must. If the two ever ranked mutations
    identically for every landscape, the second measure would be redundant.
    """

    def test_the_ranking_changes_with_the_landscape(self, synthetic_genotypes):
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        weights = np.array([1.0])

        # Immunity sitting on the X axis, which only N122D moves along.
        near_x = pop.population_escape(points, np.array([[1.0, 0.0, 0.0]]), weights,
                                       SCALE, "exponential")
        # Immunity sitting on the Y axis, which only T135K moves along.
        near_y = pop.population_escape(points, np.array([[0.0, 2.0, 0.0]]), weights,
                                       SCALE, "exponential")

        _, singles_x, _ = pop.build_population_tables(synthetic_genotypes, near_x)
        _, singles_y, _ = pop.build_population_tables(synthetic_genotypes, near_y)

        best_x = singles_x.loc[singles_x["escape_gain"].idxmax(), "mutation_h3"]
        best_y = singles_y.loc[singles_y["escape_gain"].idxmax(), "mutation_h3"]
        assert best_x != best_y

    def test_stationary_escape_ignores_the_landscape_entirely(self, synthetic_genotypes):
        import plant_order_scan as scan

        singles, _ = scan.build_escape_tables(synthetic_genotypes)
        # T135K and K189R are equidistant from the root and equally on-axis, so
        # the stationary measure cannot tell them apart at all...
        by_name = singles.set_index("mutation_h3")
        assert by_name.loc["T135K", "escape_total"] == pytest.approx(
            by_name.loc["K189R", "escape_total"]
        )
        # ...while immunity placed beside one of them separates them immediately.
        points = synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float)
        escape = pop.population_escape(points, np.array([[0.0, 2.0, 0.0]]),
                                       np.array([1.0]), SCALE, "exponential")
        _, population_singles, _ = pop.build_population_tables(synthetic_genotypes, escape)
        gains = population_singles.set_index("mutation_h3")["escape_gain"]
        assert gains["T135K"] != pytest.approx(gains["K189R"])
