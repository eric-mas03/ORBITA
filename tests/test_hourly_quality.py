import json

import numpy as np
import pytest

from astronaut_twin.hourly_quality import apply_quality, load_quality_policy


def test_policy_covers_all_dynamic_variables_without_clinical_cutoffs():
    policy = load_quality_policy()
    assert len(policy["variables"]) == 34
    assert policy["variables"]["BaseExcess"]["minimum_inclusive"] is None
    assert policy["variables"]["HR"]["maximum_inclusive"] is None
    for name in ["FiO2", "Lactate", "Magnesium"]:
        assert policy["variables"][name]["unit_status"].startswith("review_required")


def test_invalid_cell_does_not_mask_other_variables_or_drop_row():
    raw = np.array([[4000.0, 94.0, 2.2], [0.5, 88.0, 0.0]])
    original = raw.copy()
    cleaned, flags = apply_quality(raw, ["FiO2", "HR", "Creatinine"])
    assert cleaned.dtype == np.float32
    assert cleaned.shape == raw.shape
    assert np.isnan(cleaned[0, 0])
    assert np.array_equal(np.isfinite(cleaned), [[False, True, True], [True, True, True]])
    np.testing.assert_array_equal(raw, original)
    assert flags == [{"row": 0, "variable": "FiO2", "reason": "percentage_above_100", "original": 4000.0}]


def test_signed_excess_and_celsius_are_not_blanket_nonnegative():
    raw = np.array([[-20.0, -5.0, -1.0], [0.0, -273.15, 0.0], [-1.0, -273.16, 10.0]])
    cleaned, flags = apply_quality(raw, ["BaseExcess", "Temp", "Glucose"])
    assert cleaned[0, 0] == -20
    assert cleaned[0, 1] == -5
    assert np.isnan(cleaned[0, 2])
    assert np.isfinite(cleaned[1]).all()
    assert np.isnan(cleaned[2, 1])
    assert {(f["row"], f["variable"]) for f in flags} == {(0, "Glucose"), (2, "Temp")}


def test_support_bounds_inclusive_and_extreme_valid_values_retained():
    names = ["O2Sat", "SaO2", "FiO2", "pH", "HR", "Platelets"]
    raw = np.array([[0, 100, 0.21, 0, 0, 0], [100, 0, 100, 14, 400, 2_000_000],
                    [101, -1, 101, 14.01, -1, -1]], dtype=float)
    cleaned, flags = apply_quality(raw, names)
    assert np.isfinite(cleaned[:2]).all()
    assert np.isnan(cleaned[2]).all()
    assert cleaned[1, 4] == 400
    assert cleaned[1, 5] == 2_000_000
    assert len(flags) == 6


def test_unit_ambiguity_does_not_convert_or_discard_native_values():
    raw = np.array([[0.5, 2.0, 1.9], [50.0, 9.7, 14.1]])
    cleaned, flags = apply_quality(raw, ["FiO2", "Magnesium", "Lactate"])
    np.testing.assert_allclose(cleaned, raw, rtol=1e-6)
    assert flags == []


def test_hematocrit_is_a_bounded_volume_percentage_not_a_normal_range():
    cleaned, flags = apply_quality(np.array([[100.0, 99.0], [101.0, 150.0]]), ["Hct", "Glucose"])
    assert np.isfinite(cleaned[0]).all()
    assert np.isnan(cleaned[1, 0])
    assert cleaned[1, 1] == 150
    assert flags[0]["variable"] == "Hct"
    assert flags[0]["reason"] == "percentage_above_100"


def test_missingness_is_distinct_from_infinite_or_overflow_value():
    raw = np.array([[np.nan, np.inf], [-np.inf, 1e100]])
    cleaned, flags = apply_quality(raw, ["HR", "BaseExcess"])
    assert np.isnan(cleaned).all()
    assert len(flags) == 3
    assert {f["reason"] for f in flags} == {"non_finite_value", "float32_overflow"}
    assert {(f["row"], f["variable"]) for f in flags} == {(0, "BaseExcess"), (1, "HR"), (1, "BaseExcess")}
    json.dumps(flags, allow_nan=False)


def test_prefixed_names_empty_rows_and_contract_failures():
    cleaned, flags = apply_quality(np.empty((0, 2)), ["SEPSIS2019::HR", "SEPSIS2019::Glucose"])
    assert cleaned.shape == (0, 2)
    assert flags == []
    with pytest.raises(ValueError, match="2-D"):
        apply_quality(np.ones((3, 2)), ["HR"])
    with pytest.raises(ValueError, match="Duplicate"):
        apply_quality(np.ones((1, 2)), ["HR", "SEPSIS2019::HR"])
    with pytest.raises(ValueError, match="No reviewed"):
        apply_quality(np.ones((1, 1)), ["SepsisLabel"])
