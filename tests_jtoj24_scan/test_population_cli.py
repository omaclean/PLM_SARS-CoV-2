#!/usr/bin/env python3
"""The ``plant_population_escape.py`` command line, end to end on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import plant_population_escape as pop

pytestmark = pytest.mark.cli


def run(run_dir, background_csv, *extra):
    return pop.main([str(run_dir), "--background-csv", str(background_csv), *extra])


def per_date_outputs(stem):
    return (
        f"genotype_population_escape_{stem}.csv",
        f"single_mutation_population_escape_{stem}.csv",
        f"pairwise_population_escape_{stem}.csv",
        f"immune_landscape_by_year_{stem}.csv",
        f"run_metadata_{stem}.json",
        f"population_escape_singles_pairs_{stem}.png",
        f"immune_landscape_{stem}.png",
        f"population_epistasis_matrix_{stem}.png",
    )


class TestHappyPath:
    def test_writes_the_subfolder_and_every_file(self, run_dir, background_csv):
        assert run(run_dir, background_csv, "--as-of", "2023-01-01") == 0
        out = run_dir / "population_escape"
        assert out.is_dir()
        for name in per_date_outputs("2023-01-01"):
            assert (out / name).exists(), name

    def test_default_output_is_a_subfolder_of_the_run(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01")
        assert (run_dir / "population_escape").is_dir()

    def test_output_dir_can_be_redirected(self, run_dir, background_csv, tmp_path):
        destination = tmp_path / "somewhere"
        run(run_dir, background_csv, "--as-of", "2023-01-01",
            "--output-dir", str(destination))
        assert (destination / "run_metadata_2023-01-01.json").exists()
        assert not (run_dir / "population_escape").exists()

    def test_default_date_is_the_latest_background(self, run_dir, background_csv, capsys):
        assert run(run_dir, background_csv) == 0
        out = run_dir / "population_escape"
        assert (out / "run_metadata_2022-01-01.json").exists()
        assert "2022-01-01" in capsys.readouterr().out

    def test_no_plots_writes_csvs_only(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        out = run_dir / "population_escape"
        assert (out / "single_mutation_population_escape_2023-01-01.csv").exists()
        assert not list(out.glob("*.png"))

    def test_is_idempotent(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01")
        path = run_dir / "population_escape" / "single_mutation_population_escape_2023-01-01.csv"
        first = path.read_text()
        run(run_dir, background_csv, "--as-of", "2023-01-01")
        assert path.read_text() == first


class TestMultipleDates:
    def test_one_set_of_files_per_date(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2021-01-01", "--as-of", "2023-01-01")
        out = run_dir / "population_escape"
        for stem in ("2021-01-01", "2023-01-01"):
            for name in per_date_outputs(stem):
                assert (out / name).exists(), name

    def test_trend_figure_only_appears_with_more_than_one_date(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01")
        out = run_dir / "population_escape"
        assert not (out / "population_escape_vs_date.png").exists()
        run(run_dir, background_csv, "--as-of", "2021-01-01", "--as-of", "2023-01-01")
        assert (out / "population_escape_vs_date.png").exists()

    def test_by_date_csv_holds_every_date(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2021-01-01", "--as-of", "2023-01-01")
        sweep = pd.read_csv(run_dir / "population_escape" / "single_mutation_escape_by_date.csv")
        assert set(sweep["as_of"]) == {"2021-01-01", "2023-01-01"}
        assert len(sweep) == 6  # 3 mutations x 2 dates

    def test_the_landscape_actually_differs_between_dates(self, run_dir, background_csv):
        """2021 sees only the 2020 sequences; 2023 also sees the 2022 one."""
        run(run_dir, background_csv, "--as-of", "2021-01-01", "--as-of", "2023-01-01")
        out = run_dir / "population_escape"
        early = json.loads((out / "run_metadata_2021-01-01.json").read_text())
        late = json.loads((out / "run_metadata_2023-01-01.json").read_text())
        assert early["n_sequences"] == 4
        assert late["n_sequences"] == 5
        assert early["root_population_escape"] != pytest.approx(
            late["root_population_escape"]
        )


class TestMetadata:
    def test_is_valid_json_and_records_every_knob(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01",
            "--half-life", "0.5", "--kernel", "sigmoid",
            "--cross-immunity-scale", "3.0", "--normalise-by", "month",
            "--within-period", "unique", "--no-plots")
        metadata = json.loads(
            (run_dir / "population_escape" / "run_metadata_2023-01-01.json").read_text()
        )
        assert metadata["half_life_years"] == 0.5
        assert metadata["kernel"] == "sigmoid"
        assert metadata["cross_immunity_scale"] == 3.0
        assert metadata["normalise_by"] == "month"
        assert metadata["within_period"] == "unique"
        assert metadata["as_of"] == "2023-01-01"

    def test_records_the_diagnostics(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        metadata = json.loads(
            (run_dir / "population_escape" / "run_metadata_2023-01-01.json").read_text()
        )
        assert metadata["n_sequences"] == 5
        assert metadata["effective_sample_size"] > 0
        assert len(metadata["weighted_centroid"]) == 3
        assert "weight_by_year" not in metadata  # a DataFrame would not serialise

    def test_records_the_frame_check(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        metadata = json.loads(
            (run_dir / "population_escape" / "run_metadata_2023-01-01.json").read_text()
        )
        assert metadata["frame_check"]["nearest_background"] == "A/Test/1/2020"
        assert metadata["frame_check"]["distance"] == pytest.approx(0.0)


class TestFlagsChangeTheAnswer:
    def _gains(self, run_dir, background_csv, *extra):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots", *extra)
        frame = pd.read_csv(
            run_dir / "population_escape"
            / "single_mutation_population_escape_2023-01-01.csv"
        )
        return frame.set_index("mutation_h3")["escape_gain"]

    def test_normalise_by_changes_the_gains(self, run_dir, background_csv, tmp_path):
        by_year = self._gains(run_dir, background_csv, "--normalise-by", "year")
        per_sequence = self._gains(run_dir, background_csv, "--normalise-by", "none")
        assert not np.allclose(by_year.to_numpy(), per_sequence.to_numpy())

    def test_within_period_unique_changes_the_gains(self, run_dir, background_csv):
        abundance = self._gains(run_dir, background_csv, "--within-period", "abundance")
        unique = self._gains(run_dir, background_csv, "--within-period", "unique")
        assert not np.allclose(abundance.to_numpy(), unique.to_numpy())

    def test_kernel_changes_the_gains(self, run_dir, background_csv):
        exponential = self._gains(run_dir, background_csv, "--kernel", "exponential")
        sigmoid = self._gains(run_dir, background_csv, "--kernel", "sigmoid")
        assert not np.allclose(exponential.to_numpy(), sigmoid.to_numpy())

    def test_a_huge_scale_drives_every_gain_towards_zero(self, run_dir, background_csv):
        """Infinite cross-immunity means nothing escapes anything."""
        gains = self._gains(run_dir, background_csv, "--cross-immunity-scale", "10000")
        assert gains.abs().max() < 1e-3

    def test_half_life_changes_which_sequences_matter(self, run_dir, background_csv):
        sharp = self._gains(run_dir, background_csv, "--half-life", "0.1")
        flat = self._gains(run_dir, background_csv, "--half-life", "100")
        assert not np.allclose(sharp.to_numpy(), flat.to_numpy())


class TestReportingAndWarnings:
    def test_prints_the_background_range_and_frame_check(self, run_dir, background_csv, capsys):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        captured = capsys.readouterr().out
        assert "Background cloud" in captured
        assert "Frame check" in captured
        assert "2020-01-01" in captured

    def test_warns_when_the_date_is_past_the_last_background(self, run_dir, background_csv, capsys):
        run(run_dir, background_csv, "--as-of", "2030-01-01", "--no-plots")
        captured = capsys.readouterr().out
        assert "after the last background collection date" in captured
        assert "staler than requested" in captured

    def test_no_warning_when_the_date_is_inside_the_range(self, run_dir, background_csv, capsys):
        run(run_dir, background_csv, "--as-of", "2021-06-01", "--no-plots")
        assert "staler than requested" not in capsys.readouterr().out

    def test_prints_a_per_mutation_summary(self, run_dir, background_csv, capsys):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        captured = capsys.readouterr().out
        for name in ("N122D", "T135K", "K189R"):
            assert name in captured
        assert "already escapes" in captured

    def test_reports_when_there_are_no_pairs(self, pairless_run_dir, background_csv, capsys):
        run(pairless_run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        out = pairless_run_dir / "population_escape"
        assert not (out / "pairwise_population_escape_2023-01-01.csv").exists()
        assert "Largest departures" not in capsys.readouterr().out


class TestCurvatureDecompositionReachesTheCsv:
    def test_the_columns_are_written(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        pairs = pd.read_csv(
            run_dir / "population_escape" / "pairwise_population_escape_2023-01-01.csv"
        )
        for column in ("additive_genotype_gain", "kernel_curvature",
                       "epistasis_vs_additive_genotype", "curvature_share_of_epistasis"):
            assert column in pairs, column

    def test_the_split_is_exact_in_the_csv(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        pairs = pd.read_csv(
            run_dir / "population_escape" / "pairwise_population_escape_2023-01-01.csv"
        )
        assert pairs["epistasis"].to_numpy() == pytest.approx(
            (pairs["kernel_curvature"] + pairs["epistasis_vs_additive_genotype"]).to_numpy()
        )

    def test_the_curvature_share_is_reported(self, run_dir, background_csv, capsys):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        captured = capsys.readouterr().out
        assert "kernel curvature" in captured.lower()
        assert "interaction" in captured


class TestFailureModes:
    def test_missing_run_directory(self, tmp_path, background_csv):
        with pytest.raises(FileNotFoundError, match="genotype_embeddings.csv"):
            run(tmp_path / "nope", background_csv)

    def test_missing_background_csv(self, run_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            run(run_dir, tmp_path / "no_such.csv")

    def test_unparseable_date_names_the_argument(self, run_dir, background_csv):
        with pytest.raises(ValueError, match="Could not parse"):
            run(run_dir, background_csv, "--as-of", "last tuesday")

    def test_date_before_all_backgrounds(self, run_dir, background_csv):
        with pytest.raises(ValueError, match="No background sequences"):
            run(run_dir, background_csv, "--as-of", "1990-01-01")

    def test_within_period_without_a_period_is_refused(self, run_dir, background_csv):
        with pytest.raises(ValueError, match="only has meaning inside a period"):
            run(run_dir, background_csv, "--as-of", "2023-01-01",
                "--normalise-by", "none", "--within-period", "unique")

    @pytest.mark.parametrize("flag,value", [("--kernel", "gaussian"),
                                            ("--normalise-by", "decade"),
                                            ("--within-period", "vibes")])
    def test_invalid_choices_are_rejected_by_argparse(self, run_dir, background_csv,
                                                      flag, value):
        with pytest.raises(SystemExit):
            run(run_dir, background_csv, flag, value)


class TestPairlessRun:
    def test_still_produces_singles_and_a_landscape(self, pairless_run_dir, background_csv):
        assert run(pairless_run_dir, background_csv, "--as-of", "2023-01-01") == 0
        out = pairless_run_dir / "population_escape"
        assert (out / "single_mutation_population_escape_2023-01-01.csv").exists()
        assert (out / "population_escape_singles_pairs_2023-01-01.png").exists()
        assert (out / "immune_landscape_2023-01-01.png").exists()
        assert not (out / "population_epistasis_matrix_2023-01-01.png").exists()


class TestValuesAreConsistentAcrossTheCli:
    def test_csv_gains_match_a_direct_computation(self, run_dir, background_csv,
                                                  synthetic_genotypes, loaded_backgrounds):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots",
            "--normalise-by", "none")
        from_csv = pd.read_csv(
            run_dir / "population_escape"
            / "single_mutation_population_escape_2023-01-01.csv"
        ).set_index("mutation_h3")["escape_gain"]

        coordinates, weights, _ = pop.immune_weights(
            loaded_backgrounds, 2023.0, 1.0, "none", None
        )
        escape = pop.population_escape(
            synthetic_genotypes[["X", "Y", "Z"]].to_numpy(float),
            coordinates, weights, 2.0, "exponential",
        )
        _, singles, _ = pop.build_population_tables(synthetic_genotypes, escape)
        direct = singles.set_index("mutation_h3")["escape_gain"]
        assert from_csv.to_numpy() == pytest.approx(direct.reindex(from_csv.index).to_numpy())

    def test_genotype_table_and_singles_agree(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        out = run_dir / "population_escape"
        table = pd.read_csv(out / "genotype_population_escape_2023-01-01.csv")
        singles = pd.read_csv(out / "single_mutation_population_escape_2023-01-01.csv")
        by_label = table.set_index("genotype_h3")["escape_gain"]
        for row in singles.itertuples():
            assert by_label[row.mutation_h3] == pytest.approx(row.escape_gain)

    def test_immune_landscape_weights_sum_to_one(self, run_dir, background_csv):
        run(run_dir, background_csv, "--as-of", "2023-01-01", "--no-plots")
        by_year = pd.read_csv(
            run_dir / "population_escape" / "immune_landscape_by_year_2023-01-01.csv"
        )
        assert by_year["weight"].sum() == pytest.approx(1.0)
