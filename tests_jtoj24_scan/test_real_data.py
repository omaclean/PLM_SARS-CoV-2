#!/usr/bin/env python3
"""Opt-in checks against the real J -> J.2.4 run and the real background cloud.

Skipped unless ``--run-slow`` is passed AND the inputs are present, so the
default suite stays fast and offline.

The golden numbers below were read off the committed
``Results/JtoJ.2.4_scan/plant/genotype_embeddings.csv``. They are a regression
guard, not a claim that they are *right*: if the PLANT checkpoint or the scan
changes they should be updated deliberately, in a commit that says so.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import plant_order_scan as scan
import plant_population_escape as pop
import plot_plant_escape as replot

pytestmark = [pytest.mark.slow, pytest.mark.integration]

#: mutation -> (total displacement, along-axis component, fraction of the endpoint)
GOLDEN_SINGLES = {
    "N122D": (0.20400, 0.16274, 0.14133),
    "T135K": (0.59156, 0.58606, 0.50895),
    "K189R": (0.19274, 0.12236, 0.10626),
    "K276E": (0.24974, 0.22482, 0.19524),
}
#: The one pair with a materially non-additive term.
GOLDEN_EPISTASIS = {"N122D+T135K": 0.05187}
GOLDEN_SPAN = 1.15152


@pytest.mark.requires_real_run
class TestRealRunGeometry:
    @pytest.fixture
    def genotypes(self, real_run_dir):
        return pd.read_csv(real_run_dir / "genotype_embeddings.csv")

    def test_the_hypercube_is_complete(self, genotypes):
        assert len(genotypes) == 16
        assert sorted(genotypes["n_fixed"].unique()) == [0, 1, 2, 3, 4]

    def test_axis_length(self, genotypes):
        _, _, span = scan.escape_basis(genotypes)
        assert span == pytest.approx(GOLDEN_SPAN, abs=1e-4)

    @pytest.mark.parametrize("name", sorted(GOLDEN_SINGLES))
    def test_single_mutant_golden_values(self, genotypes, name):
        singles, _ = scan.build_escape_tables(genotypes)
        row = singles.set_index("mutation_h3").loc[name]
        total, along, fraction = GOLDEN_SINGLES[name]
        assert row["escape_total"] == pytest.approx(total, abs=1e-4)
        assert row["escape_along_axis"] == pytest.approx(along, abs=1e-4)
        assert row["frac_of_endpoint"] == pytest.approx(fraction, abs=1e-4)

    def test_t135k_carries_about_half_the_move(self, genotypes):
        singles, _ = scan.build_escape_tables(genotypes)
        by_name = singles.set_index("mutation_h3")["frac_of_endpoint"]
        assert by_name["T135K"] > 0.45
        assert by_name["T135K"] > 2 * by_name.drop("T135K").max()

    def test_the_one_real_pairwise_interaction(self, genotypes):
        _, pairs = scan.build_escape_tables(genotypes)
        by_pair = pairs.set_index("pair_h3")["epistasis_along_axis"]
        for label, expected in GOLDEN_EPISTASIS.items():
            assert by_pair[label] == pytest.approx(expected, abs=1e-4)
        others = by_pair.drop(list(GOLDEN_EPISTASIS))
        assert others.abs().max() < 0.015

    def test_off_axis_mutations_are_flagged_as_such(self, genotypes):
        """K189R and N122D move mostly sideways; T135K does not."""
        singles, _ = scan.build_escape_tables(genotypes)
        by_name = singles.set_index("mutation_h3")["off_axis_fraction"]
        assert by_name["K189R"] > 0.7
        assert by_name["N122D"] > 0.5
        assert by_name["T135K"] < 0.2

    def test_replotting_the_real_run_into_a_scratch_dir(self, real_run_dir, tmp_path):
        assert replot.main([str(real_run_dir), "--output-dir", str(tmp_path)]) == 0
        for name in ("single_mutation_escape.csv", "pairwise_escape.csv",
                     "plant_escape_singles_pairs.png", "plant_escape_map.png",
                     "plant_escape_epistasis_matrix.png"):
            assert (tmp_path / name).exists(), name


@pytest.mark.requires_real_run
@pytest.mark.requires_real_backgrounds
class TestRealBackgrounds:
    @pytest.fixture(scope="class")
    def backgrounds(self, real_background_csv):
        return pop.load_backgrounds(real_background_csv)

    def test_the_cloud_is_the_size_and_span_expected(self, backgrounds):
        assert len(backgrounds) > 140_000
        assert backgrounds["decimal_year"].min() < 1970
        assert backgrounds["decimal_year"].max() > 2023

    def test_every_row_parsed_a_date(self, backgrounds):
        assert backgrounds["decimal_year"].notna().all()

    def test_both_date_formats_are_present(self, real_background_csv):
        """Bare years early, ISO dates later -- the reason the parser is tolerant."""
        raw = pd.read_csv(real_background_csv, usecols=["collection date"])
        text = raw["collection date"].astype(str)
        assert (text.str.len() == 4).any()
        assert (text.str.len() == 10).any()

    def test_the_frames_really_are_shared(self, real_run_dir, backgrounds):
        """The start lineage must sit essentially on top of a real sequence."""
        genotypes = pd.read_csv(real_run_dir / "genotype_embeddings.csv")
        root = genotypes.loc[genotypes["n_fixed"] == 0, ["X", "Y", "Z"]].to_numpy(float)[0]
        report = pop.check_shared_frame(root, backgrounds)
        assert report["distance"] < 0.05, (
            "start lineage is far from every background sequence; the genotype "
            "embeddings and backgrounds.csv may be from different checkpoints"
        )

    def test_observed_lineages_land_on_their_own_subclade(self, real_run_dir, backgrounds):
        """The J.2 embedding should be nearest a background sequence labelled J.2."""
        observed_path = real_run_dir / "observed_sequence_embeddings.csv"
        if not observed_path.exists():
            pytest.skip("no observed_sequence_embeddings.csv in the run")
        observed = pd.read_csv(observed_path)
        row = observed[observed["lineage"] == "J.2"].iloc[0]
        report = pop.check_shared_frame(
            np.array([row["X"], row["Y"], row["Z"]], dtype=float), backgrounds
        )
        assert report["nearest_subclade"] == "J.2"
        assert report["distance"] < 0.05

    def test_full_run_at_one_date(self, real_run_dir, real_background_csv, tmp_path):
        exit_code = pop.main([
            str(real_run_dir),
            "--background-csv", str(real_background_csv),
            "--as-of", "2024-01-01",
            "--output-dir", str(tmp_path),
        ])
        assert exit_code == 0
        metadata = json.loads((tmp_path / "run_metadata_2024-01-01.json").read_text())
        assert metadata["n_sequences"] > 100_000
        # A 1-year half-life over a decade of sequencing must still leave a
        # landscape made of many sequences, or the score is noise.
        assert metadata["effective_sample_size"] > 100
        singles = pd.read_csv(
            tmp_path / "single_mutation_population_escape_2024-01-01.csv"
        )
        assert len(singles) == 4
        assert singles["population_escape"].between(0.0, 1.0).all()

    def test_year_normalisation_changes_the_real_answer(self, real_run_dir,
                                                        real_background_csv, tmp_path):
        """If it did not, the flag would not be worth having on real data."""
        gains = {}
        for mode in ("year", "none"):
            destination = tmp_path / mode
            pop.main([
                str(real_run_dir),
                "--background-csv", str(real_background_csv),
                "--as-of", "2024-01-01", "--normalise-by", mode,
                "--no-plots", "--output-dir", str(destination),
            ])
            gains[mode] = pd.read_csv(
                destination / "single_mutation_population_escape_2024-01-01.csv"
            ).set_index("mutation_h3")["escape_gain"]
        assert not np.allclose(gains["year"].to_numpy(),
                               gains["none"].reindex(gains["year"].index).to_numpy())

    def test_unique_within_period_changes_the_real_answer(self, real_run_dir,
                                                          real_background_csv, tmp_path):
        """The real CSV is full of exact coordinate ties, so this must bite."""
        gains = {}
        for mode in ("abundance", "unique"):
            destination = tmp_path / mode
            pop.main([
                str(real_run_dir),
                "--background-csv", str(real_background_csv),
                "--as-of", "2024-01-01", "--within-period", mode,
                "--no-plots", "--output-dir", str(destination),
            ])
            gains[mode] = pd.read_csv(
                destination / "single_mutation_population_escape_2024-01-01.csv"
            ).set_index("mutation_h3")["escape_gain"]
        assert not np.allclose(
            gains["abundance"].to_numpy(),
            gains["unique"].reindex(gains["abundance"].index).to_numpy(),
        )

    def test_a_date_sweep_runs(self, real_run_dir, real_background_csv, tmp_path):
        exit_code = pop.main([
            str(real_run_dir),
            "--background-csv", str(real_background_csv),
            "--as-of", "2022-01-01", "--as-of", "2023-01-01", "--as-of", "2024-01-01",
            "--output-dir", str(tmp_path),
        ])
        assert exit_code == 0
        assert (tmp_path / "population_escape_vs_date.png").exists()
        sweep = pd.read_csv(tmp_path / "single_mutation_escape_by_date.csv")
        assert len(sweep) == 12  # 4 mutations x 3 dates
