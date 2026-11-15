"""Checks on acquired public tables, conversion, identity and censor masks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external/NHANES_2017_2018"


@pytest.fixture(scope="module")
def nhanes():
    if not (RAW / "CBC_J.xpt").exists():
        pytest.skip("Acquire NHANES with scripts/download_nhanes.py first")
    data = pd.read_csv(ROOT / "data/processed/external/nhanes_adult_labs.csv")
    mask = pd.read_csv(ROOT / "data/processed/external/nhanes_adult_labs_observed_mask.csv")
    cbc = pd.read_sas(RAW / "CBC_J.xpt", format="xport").set_index("SEQN")
    biopro = pd.read_sas(RAW / "BIOPRO_J.xpt", format="xport").set_index("SEQN")
    mapping = json.loads((ROOT / "config/nhanes_candidate_mapping.json").read_text())
    return data, mask, cbc, biopro, mapping


def test_adult_identity_is_cross_sectional_not_artificial_history(nhanes):
    data, mask, _, _, mapping = nhanes
    assert data.subject_id.is_unique and len(data) == 5274
    assert data.subject_id.tolist() == mask.subject_id.tolist()
    assert data.age_years.ge(18).all()
    assert set(data.domain) == {"terrestrial_cross_sectional"}
    assert not {"timepoint", "visit_index", "timestamp", "mission_id"}.intersection(data.columns)
    assert len(mapping) == 29
    assert not any(row["forecasting_enabled"] or row["input_enabled"] for row in mapping)


def test_count_unit_conversion_and_values_match_sources(nhanes):
    data, _, cbc, biopro, mapping = nhanes
    ids = data.subject_id.str.removeprefix("NHANES_J_").astype(float)
    for row in mapping:
        source = cbc if row["source_variable"] in cbc else biopro
        expected = source[row["source_variable"]].reindex(ids).to_numpy() * row["factor"]
        actual = data[row["candidate_common_key"]].to_numpy()
        np.testing.assert_allclose(actual, expected, equal_nan=True)
    absolute = [r for r in mapping if "ABSOLUTE" in r["candidate_common_key"]]
    assert len(absolute) == 5 and all(r["factor"] == 1000 for r in absolute)


def test_missing_and_below_detection_labels_are_not_loss_targets(nhanes):
    data, mask, _, biopro, mapping = nhanes
    ids = data.subject_id.str.removeprefix("NHANES_J_").astype(float)
    for row in mapping:
        key = row["candidate_common_key"]
        assert not (mask[key] & data[key].isna()).any()
        if row["censor_flag"]:
            flags = biopro[row["censor_flag"]].reindex(ids).round().to_numpy()
            expected = data[key].notna().to_numpy() & (flags == 0)
            np.testing.assert_array_equal(mask[key].to_numpy(), expected)
            assert mask[key].sum() > 5000  # SAS categorical zero must not mask all rows.
