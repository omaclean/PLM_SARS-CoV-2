#!/usr/bin/env python3
"""Figure helpers and the three stationary-point escape figures.

A plotting test cannot assert that a chart is *good*, but it can assert that it
was written, that it is a real PNG, that the degraded paths still produce one,
and that the pure helpers behind it (label placement, the colour ramps) obey the
properties the figures rely on.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import plant_order_scan as scan

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def assert_is_png(path, minimum_bytes: int = 5_000) -> None:
    assert path.exists(), f"{path} was not written"
    payload = path.read_bytes()
    assert payload.startswith(PNG_MAGIC), f"{path} is not a PNG"
    assert len(payload) > minimum_bytes, f"{path} is suspiciously small ({len(payload)} B)"


@pytest.mark.unit
class TestRampColour:
    def test_endpoints_are_the_ramp_ends(self):
        assert scan.ramp_colour(0, 4).lower() == scan.BLUE_RAMP[0].lower()
        assert scan.ramp_colour(4, 4).lower() == scan.BLUE_RAMP[-1].lower()

    def test_four_levels_reproduce_the_literal_ramp(self):
        """Sampling a 5-anchor ramp at k/4 must hit anchor k exactly."""
        for level, expected in enumerate(scan.BLUE_RAMP):
            assert scan.ramp_colour(level, 4).lower() == expected.lower()

    def test_large_sets_still_span_the_ramp(self):
        """11 mutations must not pile every level past the fourth on one step."""
        colours = [scan.ramp_colour(level, 11) for level in range(12)]
        assert len(set(colours)) > 8
        assert colours[0].lower() == scan.BLUE_RAMP[0].lower()
        assert colours[-1].lower() == scan.BLUE_RAMP[-1].lower()

    def test_monotone_darkening(self):
        from matplotlib.colors import to_rgb

        luminance = [sum(to_rgb(scan.ramp_colour(level, 6))) for level in range(7)]
        assert luminance == sorted(luminance, reverse=True)

    def test_zero_top_level_does_not_divide_by_zero(self):
        assert scan.ramp_colour(0, 0).lower() == scan.BLUE_RAMP[0].lower()


@pytest.mark.unit
class TestDivergingCmap:
    def test_poles_and_midpoint(self):
        from matplotlib.colors import to_hex, to_rgb

        cmap = scan.diverging_cmap()
        assert to_hex(cmap(0.0)).lower() == scan.DIVERGING_LOW.lower()
        assert to_hex(cmap(1.0)).lower() == scan.DIVERGING_HIGH.lower()
        # The midpoint is only approximate: a LinearSegmentedColormap quantises
        # to 256 levels, so 0.5 lands next to the anchor rather than on it.
        # One level is ~1/255 per channel, so half a percent is ample.
        assert cmap(0.5)[:3] == pytest.approx(to_rgb(scan.DIVERGING_MID), abs=0.005)

    def test_midpoint_is_the_least_saturated(self):
        from matplotlib.colors import to_rgb

        def chroma(value):
            red, green, blue = to_rgb(scan.diverging_cmap()(value))
            return max(red, green, blue) - min(red, green, blue)

        assert chroma(0.5) < chroma(0.0)
        assert chroma(0.5) < chroma(1.0)


@pytest.mark.unit
class TestPlaceLabels:
    def _drawn_axis(self):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 10)
        figure.canvas.draw()
        return figure, axis

    def test_every_label_is_placed(self):
        figure, axis = self._drawn_axis()
        entries = [(float(i), 5.0, f"label{i}", {}) for i in range(6)]
        scan.place_labels(axis, entries)
        texts = [text.get_text() for text in axis.texts]
        assert texts == [f"label{i}" for i in range(6)]
        plt.close(figure)

    def test_colliding_points_get_different_offsets(self):
        """Two labels on the same coordinate must not be stacked on each other."""
        figure, axis = self._drawn_axis()
        scan.place_labels(axis, [(5.0, 5.0, "first", {}), (5.0, 5.0, "second", {})])
        offsets = [text.get_position() for text in axis.texts]
        assert offsets[0] != offsets[1]
        plt.close(figure)

    def test_distant_points_both_take_the_first_candidate(self):
        figure, axis = self._drawn_axis()
        scan.place_labels(axis, [(1.0, 1.0, "a", {}), (9.0, 9.0, "b", {})])
        assert {text.get_position() for text in axis.texts} == {(0, 11)}
        plt.close(figure)

    def test_never_drops_a_label_even_when_saturated(self):
        figure, axis = self._drawn_axis()
        entries = [(5.0, 5.0, f"crowded{i}", {}) for i in range(20)]
        scan.place_labels(axis, entries)
        assert len(axis.texts) == 20
        plt.close(figure)

    def test_style_overrides_are_applied(self):
        figure, axis = self._drawn_axis()
        scan.place_labels(
            axis, [(5.0, 5.0, "styled", {"fontsize": 14, "color": "#123456"})]
        )
        text = axis.texts[0]
        assert text.get_fontsize() == pytest.approx(14)
        assert text.get_color() == "#123456"
        plt.close(figure)

    def test_empty_entry_list_is_a_no_op(self):
        figure, axis = self._drawn_axis()
        scan.place_labels(axis, [])
        # axis.texts is an ArtistList, not a list, so compare its length.
        assert len(axis.texts) == 0
        plt.close(figure)


@pytest.mark.figure
class TestEscapeFigures:
    def test_singles_and_pairs_figure(self, tmp_path, synthetic_genotypes):
        singles, pairs = scan.build_escape_tables(synthetic_genotypes)
        path = tmp_path / "singles_pairs.png"
        scan.plot_escape_singles_pairs(singles, pairs, path, "J", "J.2.4")
        assert_is_png(path)

    def test_singles_only_figure_when_there_are_no_pairs(self, tmp_path, pairless_genotypes):
        singles, pairs = scan.build_escape_tables(pairless_genotypes)
        assert pairs is None
        path = tmp_path / "singles_only.png"
        scan.plot_escape_singles_pairs(singles, pairs, path, "J", "J.2.4")
        assert_is_png(path)

    def test_epistasis_matrix(self, tmp_path, synthetic_genotypes):
        singles, pairs = scan.build_escape_tables(synthetic_genotypes)
        path = tmp_path / "matrix.png"
        scan.plot_escape_epistasis_matrix(singles, pairs, path, "J", "J.2.4")
        assert_is_png(path)

    def test_map_with_observed(self, tmp_path, synthetic_genotypes, observed_frame):
        _, pairs = scan.build_escape_tables(synthetic_genotypes)
        path = tmp_path / "map.png"
        scan.plot_escape_map(synthetic_genotypes, pairs, observed_frame, path, "J", "J.2.4")
        assert_is_png(path)

    def test_map_without_observed_or_pairs(self, tmp_path, pairless_genotypes):
        path = tmp_path / "map_bare.png"
        scan.plot_escape_map(pairless_genotypes, None, None, path, "J", "J.2.4")
        assert_is_png(path)

    def test_map_survives_a_degenerate_off_axis_direction(self, tmp_path):
        """All genotypes collinear: the SVD residual is zero everywhere."""
        import pandas as pd

        collinear = pd.DataFrame(
            {
                "genotype_id": ["root", "A", "B", "A+B"],
                "genotype_h3": ["root", "A", "B", "A+B"],
                "n_fixed": [0, 1, 1, 2],
                "X": [0.0, 1.0, 2.0, 3.0],
                "Y": [0.0, 0.0, 0.0, 0.0],
                "Z": [0.0, 0.0, 0.0, 0.0],
            }
        )
        _, pairs = scan.build_escape_tables(collinear)
        path = tmp_path / "collinear.png"
        scan.plot_escape_map(collinear, pairs, None, path, "J", "J.2.4")
        assert_is_png(path)

    def test_no_figures_leak_between_tests(self, tmp_path, synthetic_genotypes):
        """Every plot function must close its figure or a long run exhausts memory."""
        singles, pairs = scan.build_escape_tables(synthetic_genotypes)
        plt.close("all")
        scan.plot_escape_singles_pairs(singles, pairs, tmp_path / "a.png", "J", "E")
        scan.plot_escape_epistasis_matrix(singles, pairs, tmp_path / "b.png", "J", "E")
        scan.plot_escape_map(synthetic_genotypes, pairs, None, tmp_path / "c.png", "J", "E")
        assert plt.get_fignums() == []


@pytest.mark.integration
class TestWriteEscapeOutputs:
    def test_writes_tables_and_figures(self, tmp_path, synthetic_genotypes, observed_frame):
        singles, pairs = scan.write_escape_outputs(
            synthetic_genotypes, observed_frame, tmp_path, "J", "J.2.4"
        )
        assert pairs is not None
        for name in (
            "single_mutation_escape.csv",
            "pairwise_escape.csv",
            "plant_escape_singles_pairs.png",
            "plant_escape_map.png",
            "plant_escape_epistasis_matrix.png",
        ):
            assert (tmp_path / name).exists(), name

    def test_draw_false_writes_csvs_only(self, tmp_path, synthetic_genotypes):
        scan.write_escape_outputs(
            synthetic_genotypes, None, tmp_path, "J", "J.2.4", draw=False
        )
        assert (tmp_path / "single_mutation_escape.csv").exists()
        assert (tmp_path / "pairwise_escape.csv").exists()
        assert not list(tmp_path.glob("*.png"))

    def test_pairless_run_skips_the_matrix(self, tmp_path, pairless_genotypes):
        singles, pairs = scan.write_escape_outputs(
            pairless_genotypes, None, tmp_path, "J", "J.2.4"
        )
        assert pairs is None
        assert not (tmp_path / "pairwise_escape.csv").exists()
        assert not (tmp_path / "plant_escape_epistasis_matrix.png").exists()
        assert (tmp_path / "plant_escape_singles_pairs.png").exists()
        assert (tmp_path / "plant_escape_map.png").exists()

    def test_written_csv_round_trips(self, tmp_path, synthetic_genotypes):
        import pandas as pd

        singles, _ = scan.write_escape_outputs(
            synthetic_genotypes, None, tmp_path, "J", "J.2.4", draw=False
        )
        reloaded = pd.read_csv(tmp_path / "single_mutation_escape.csv")
        assert reloaded["escape_total"].to_numpy() == pytest.approx(
            singles["escape_total"].to_numpy()
        )


@pytest.mark.unit
class TestLineageLabel:
    @pytest.mark.parametrize(
        "header,expected",
        [
            ("EPI2178977|HA|A/Thailand/8/2022|EPI_ISL_14991375|J", "J"),
            ("EPI4551140|HA|A/England/415/2024|EPI_ISL_20080368|J.2.4", "J.2.4"),
            ("no_pipes_here", "no_pipes_here"),
            ("trailing|", ""),
        ],
    )
    def test_takes_the_last_field(self, header, expected):
        assert scan.lineage_label(header) == expected
