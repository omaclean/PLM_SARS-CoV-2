#!/usr/bin/env python3
"""The ``plot_plant_escape.py`` command-line surface, end to end.

``main()`` is called in-process rather than through a subprocess: it is faster,
and an exception surfaces as a traceback instead of an exit code.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import plot_plant_escape as replot

pytestmark = pytest.mark.cli

EXPECTED_OUTPUTS = (
    "single_mutation_escape.csv",
    "pairwise_escape.csv",
    "plant_escape_singles_pairs.png",
    "plant_escape_map.png",
    "plant_escape_epistasis_matrix.png",
)


class TestHappyPath:
    def test_writes_every_output_next_to_the_run(self, run_dir, capsys):
        assert replot.main([str(run_dir)]) == 0
        for name in EXPECTED_OUTPUTS:
            assert (run_dir / name).exists(), name
        captured = capsys.readouterr().out
        assert "3 single mutant(s)" in captured
        assert "3 pair(s)" in captured

    def test_is_idempotent(self, run_dir):
        replot.main([str(run_dir)])
        first = (run_dir / "single_mutation_escape.csv").read_text()
        replot.main([str(run_dir)])
        assert (run_dir / "single_mutation_escape.csv").read_text() == first

    def test_output_dir_redirects_everything(self, run_dir, tmp_path):
        destination = tmp_path / "elsewhere"
        assert replot.main([str(run_dir), "--output-dir", str(destination)]) == 0
        for name in EXPECTED_OUTPUTS:
            assert (destination / name).exists(), name
        assert not (run_dir / "single_mutation_escape.csv").exists()

    def test_creates_a_missing_output_dir(self, run_dir, tmp_path):
        destination = tmp_path / "deep" / "nested" / "out"
        assert replot.main([str(run_dir), "--output-dir", str(destination)]) == 0
        assert destination.is_dir()

    def test_several_run_dirs_in_one_call(self, run_dir, pairless_run_dir):
        assert replot.main([str(run_dir), str(pairless_run_dir)]) == 0
        assert (run_dir / "pairwise_escape.csv").exists()
        assert not (pairless_run_dir / "pairwise_escape.csv").exists()


class TestLabels:
    def test_labels_come_from_run_metadata(self, run_dir):
        assert replot.resolve_labels(run_dir, _args()) == ("J", "J.2.4")

    def test_missing_metadata_falls_back(self, bare_run_dir):
        assert replot.resolve_labels(bare_run_dir, _args()) == ("start", "end")

    def test_cli_labels_win(self, run_dir):
        args = _args(start_label="ANCESTOR", end_label="DERIVED")
        assert replot.resolve_labels(run_dir, args) == ("ANCESTOR", "DERIVED")

    def test_one_cli_label_still_reads_the_other_from_metadata(self, run_dir):
        args = _args(start_label="ANCESTOR")
        assert replot.resolve_labels(run_dir, args) == ("ANCESTOR", "J.2.4")

    def test_metadata_without_headers_falls_back(self, run_dir):
        (run_dir / "run_metadata.json").write_text(json.dumps({"mutations": []}))
        assert replot.resolve_labels(run_dir, _args()) == ("start", "end")


class TestObserved:
    def test_observed_csv_is_used_when_present(self, run_dir):
        assert replot.main([str(run_dir)]) == 0
        assert (run_dir / "plant_escape_map.png").exists()

    def test_no_observed_flag_still_draws(self, run_dir):
        assert replot.main([str(run_dir), "--no-observed"]) == 0
        assert (run_dir / "plant_escape_map.png").exists()

    def test_missing_observed_csv_is_not_an_error(self, bare_run_dir):
        assert replot.main([str(bare_run_dir)]) == 0
        assert (bare_run_dir / "plant_escape_map.png").exists()


class TestDegradedRuns:
    def test_pairless_run_reports_and_skips(self, pairless_run_dir, capsys):
        assert replot.main([str(pairless_run_dir)]) == 0
        captured = capsys.readouterr().out
        assert "No two-mutation backgrounds" in captured
        assert "--max-background-size 2" in captured
        assert not (pairless_run_dir / "pairwise_escape.csv").exists()
        assert not (pairless_run_dir / "plant_escape_epistasis_matrix.png").exists()
        assert (pairless_run_dir / "plant_escape_singles_pairs.png").exists()


class TestFailureModes:
    def test_missing_table_names_the_file(self, tmp_path):
        empty = tmp_path / "not_a_run"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="genotype_embeddings.csv"):
            replot.main([str(empty)])

    def test_nonexistent_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            replot.main([str(tmp_path / "nope")])

    def test_a_later_run_dir_failing_does_not_hide_the_first(self, run_dir, tmp_path):
        broken = tmp_path / "broken"
        broken.mkdir()
        with pytest.raises(FileNotFoundError):
            replot.main([str(run_dir), str(broken)])
        assert (run_dir / "single_mutation_escape.csv").exists()


class TestValuesSurviveTheCli:
    def test_csv_matches_the_closed_form(self, run_dir):
        from conftest import EXPECTED_SINGLES

        replot.main([str(run_dir)])
        singles = pd.read_csv(run_dir / "single_mutation_escape.csv").set_index("mutation_h3")
        for name, (total, along, off) in EXPECTED_SINGLES.items():
            assert singles.loc[name, "escape_total"] == pytest.approx(total)
            assert singles.loc[name, "escape_along_axis"] == pytest.approx(along)
            assert singles.loc[name, "escape_off_axis"] == pytest.approx(off)

    def test_pair_csv_matches_the_planted_epistasis(self, run_dir):
        from conftest import EXPECTED_PAIR_EPISTASIS

        replot.main([str(run_dir)])
        pairs = pd.read_csv(run_dir / "pairwise_escape.csv").set_index("pair_h3")
        for label, expected in EXPECTED_PAIR_EPISTASIS.items():
            assert pairs.loc[label, "epistasis_along_axis"] == pytest.approx(
                expected, abs=1e-12
            )


def _args(**overrides):
    """A namespace shaped like the parsed CLI, for resolve_labels."""
    import argparse

    defaults = {"start_label": None, "end_label": None}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
