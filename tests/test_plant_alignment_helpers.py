import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions_HuggingFace import (
    plant_trim_to_target_length,
    plant_sequence_identity,
    plant_alignment_coverage_metrics,
    plant_extract_year,
)


def test_plant_trim_to_target_length_exact_length_no_change():
    ref = "ACDEFGHIK"
    seq = "ACDEFGHIK"
    trimmed = plant_trim_to_target_length(seq, ref, target_len=len(ref), return_start_pos=False)
    assert trimmed == ref


def test_plant_trim_to_target_length_returns_start_pos():
    ref = "ACDEFGHIK"
    seq = "XXACDEFGHIKYY"
    trimmed, start_pos = plant_trim_to_target_length(seq, ref, target_len=len(ref), return_start_pos=True)
    assert trimmed == ref
    assert isinstance(start_pos, int)
    assert start_pos >= 0


def test_plant_trim_to_target_length_handles_gaps_and_dots():
    ref = "ACDEFGHIK"
    seq = "ACD.EF-GHIK"
    trimmed = plant_trim_to_target_length(seq, ref, target_len=len(ref), return_start_pos=False)
    assert isinstance(trimmed, str)
    assert len(trimmed) == len(ref)


def test_plant_trim_to_target_length_invalid_input():
    ref = "ACDEFGHIK"
    assert plant_trim_to_target_length(None, ref, target_len=len(ref), return_start_pos=False) is None
    trimmed, start_pos = plant_trim_to_target_length(None, ref, target_len=len(ref), return_start_pos=True)
    assert trimmed is None
    assert start_pos is None


def test_plant_sequence_identity_basic_cases():
    assert plant_sequence_identity("AAAA", "AAAA") == 1.0
    assert plant_sequence_identity("AAAA", "AAAT") == 0.75
    assert np.isnan(plant_sequence_identity("AAAA", "AAA"))


def test_plant_alignment_coverage_metrics_perfect_match():
    ref = "ACDEFGHIK"
    seq = "ACDEFGHIK"
    metrics = plant_alignment_coverage_metrics(seq, ref)
    assert pytest.approx(metrics["aligned_ref_coverage"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["aligned_query_coverage"], rel=1e-6) == 1.0
    assert metrics["alignment_score"] > 0


def test_plant_alignment_coverage_metrics_partial_match():
    ref = "ACDEFGHIK"
    seq = "XXACDEFG"
    metrics = plant_alignment_coverage_metrics(seq, ref)
    assert 0 < metrics["aligned_ref_coverage"] <= 1
    assert 0 < metrics["aligned_query_coverage"] <= 1


def test_plant_extract_year_parsing():
    assert plant_extract_year("2024") == 2024
    assert plant_extract_year("2024/03/01") == 2024
    assert plant_extract_year("2024-03-01") == 2024
    assert plant_extract_year("not-a-date") is None
