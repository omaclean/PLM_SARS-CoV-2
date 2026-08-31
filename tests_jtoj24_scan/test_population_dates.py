#!/usr/bin/env python3
"""Date handling in ``plant_population_escape``.

``backgrounds.csv`` mixes bare years (everything before ~2005) with full ISO
dates, so a parser that only understands one of them silently mis-ages half the
immune landscape. These are the cases that actually appear in the file, plus the
ones that would corrupt a weight if mishandled.
"""

from __future__ import annotations

import numpy as np
import pytest

import plant_population_escape as pop

pytestmark = pytest.mark.unit


class TestToDecimalYear:
    def test_full_iso_date(self):
        # 2006-09-17 is day 260 of a 365-day year (index 259 from Jan 1).
        assert pop.to_decimal_year("2006-09-17") == pytest.approx(2006 + 259 / 365)

    def test_january_first_is_the_year_exactly(self):
        assert pop.to_decimal_year("2024-01-01") == pytest.approx(2024.0)

    def test_new_years_eve_is_just_short_of_the_next_year(self):
        value = pop.to_decimal_year("2023-12-31")
        assert 2023.99 < value < 2024.0

    def test_leap_year_uses_366_days(self):
        # 2020-12-31 is day 366; 2021-12-31 is day 365.
        assert pop.to_decimal_year("2020-12-31") == pytest.approx(2020 + 365 / 366)
        assert pop.to_decimal_year("2021-12-31") == pytest.approx(2021 + 364 / 365)

    def test_bare_year_lands_mid_year(self):
        """Snapping a bare year to 1 January would systematically age it."""
        assert pop.to_decimal_year("1968") == pytest.approx(1968.5)

    def test_bare_month_lands_mid_month(self):
        assert pop.to_decimal_year("2006-09") == pytest.approx(2006 + 8.5 / 12)
        assert pop.to_decimal_year("2006-01") == pytest.approx(2006 + 0.5 / 12)

    def test_slashes_are_accepted(self):
        assert pop.to_decimal_year("2006/09/17") == pop.to_decimal_year("2006-09-17")

    def test_surrounding_whitespace(self):
        assert pop.to_decimal_year("  2010-05-05  ") == pop.to_decimal_year("2010-05-05")

    @pytest.mark.parametrize("value", ["", "   ", "nan", "NaN", "None", "nat", None, np.nan])
    def test_missing_values_become_none(self, value):
        assert pop.to_decimal_year(value) is None

    @pytest.mark.parametrize("value", ["not-a-date", "unknown", "XXXX-01-01"])
    def test_unparseable_becomes_none(self, value):
        assert pop.to_decimal_year(value) is None

    @pytest.mark.parametrize("value,expected", [("2010-13-01", 2010.5), ("2010-00-01", 2010.5)])
    def test_impossible_month_falls_back_to_mid_year(self, value, expected):
        assert pop.to_decimal_year(value) == pytest.approx(expected)

    def test_impossible_day_falls_back_to_mid_month(self):
        assert pop.to_decimal_year("2010-02-30") == pytest.approx(2010 + 1.5 / 12)

    def test_ordering_is_preserved(self):
        dates = ["1968", "2000-06-15", "2006-09-17", "2020", "2023-12-31", "2024-01-01"]
        values = [pop.to_decimal_year(text) for text in dates]
        assert values == sorted(values)


class TestFormatDecimalYear:
    def test_round_trips_a_full_date(self):
        for text in ("2006-09-17", "2020-02-29", "1999-01-01", "2023-12-31"):
            assert pop.format_decimal_year(pop.to_decimal_year(text)) == text

    def test_year_boundary(self):
        assert pop.format_decimal_year(2024.0) == "2024-01-01"

    def test_mid_year_of_a_bare_year(self):
        # 0.5 * 365 = 182.5 -> rounds to day index 182 -> 2 July.
        assert pop.format_decimal_year(1968.5).startswith("1968-07")


class TestLoadBackgrounds:
    def test_parses_dates_and_keeps_coordinates(self, background_csv):
        frame = pop.load_backgrounds(background_csv)
        assert len(frame) == 5
        assert "decimal_year" in frame
        assert frame["decimal_year"].min() == pytest.approx(2020.0)
        assert frame["decimal_year"].max() == pytest.approx(2022.0)

    def test_renames_the_date_column(self, background_csv):
        frame = pop.load_backgrounds(background_csv)
        assert "collection_date" in frame
        assert "collection date" not in frame

    def test_accepts_a_plain_date_column(self, tmp_path, background_frame):
        renamed = background_frame.rename(columns={"collection date": "date"})
        path = tmp_path / "alt.csv"
        renamed.to_csv(path, index=False)
        assert len(pop.load_backgrounds(path)) == 5

    def test_missing_date_column_raises_helpfully(self, tmp_path, background_frame):
        path = tmp_path / "undated.csv"
        background_frame.drop(columns=["collection date"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="collection date"):
            pop.load_backgrounds(path)

    def test_missing_coordinates_raise(self, tmp_path, background_frame):
        path = tmp_path / "flat.csv"
        background_frame.drop(columns=["Z"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="coordinate"):
            pop.load_backgrounds(path)

    def test_undated_rows_are_dropped_and_reported(self, tmp_path, background_frame, capsys):
        polluted = background_frame.copy()
        polluted.loc[0, "collection date"] = "unknown"
        path = tmp_path / "polluted.csv"
        polluted.to_csv(path, index=False)
        frame = pop.load_backgrounds(path)
        assert len(frame) == 4
        assert "unparseable collection" in capsys.readouterr().out

    def test_all_rows_undated_raises(self, tmp_path, background_frame):
        broken = background_frame.copy()
        broken["collection date"] = "unknown"
        path = tmp_path / "broken.csv"
        broken.to_csv(path, index=False)
        with pytest.raises(ValueError, match="no dated background"):
            pop.load_backgrounds(path)

    def test_rows_with_missing_coordinates_are_dropped(self, tmp_path, background_frame):
        holed = background_frame.copy()
        holed.loc[0, "X"] = np.nan
        path = tmp_path / "holed.csv"
        holed.to_csv(path, index=False)
        assert len(pop.load_backgrounds(path)) == 4
