"""Audit new analog data and strict preservation of clinical measurement flags."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


utmb = module("prepare_lsda_utmb")
sleep = module("prepare_utmb_sleep")


def test_clinical_flags_keep_abnormal_values_but_not_ranges_as_point_targets():
    number, flag, suffix, exact, _, _ = utmb.parse_measurement("4.4 (H)")
    assert number == 4.4 and exact and suffix == "(H)" and flag == "numeric_with_reference_flag"
    number, flag, suffix, exact, _, _ = utmb.parse_measurement("72 A (L)")
    assert number == 72 and not exact and suffix == "A (L)" and flag == "numeric_with_footnote_review"
    for raw in ["3-5", "0-2", "1+", "10-20 (A)", "Negative", "N/A I"]:
        parsed = utmb.parse_measurement(raw)
        assert not parsed[3] and np.isnan(parsed[0])
    parsed = utmb.parse_measurement("<=1.005")
    assert not parsed[3] and np.isnan(parsed[0]) and parsed[4:] == ("<=", 1.005)
    assert utmb.clean_label(" Alanine\\n aminotransferase (ALT) ") == "Alanine aminotransferase (ALT)"


def test_source_identity_coverage_and_nonblood_separation():
    path = ROOT / "data/processed/utmb_c3/observations.csv"
    if not path.exists():
        pytest.skip("Download LSDA and run scripts/prepare_lsda_utmb.py first")
    observations = pd.read_csv(path)
    common = pd.read_csv(path.parent / "candidate_common_observations.csv")
    visits = pd.read_csv(path.parent / "visits.csv")
    assert len(observations) == 9534 and observations.subject_id.nunique() == 31
    assert observations.ParameterName.notna().all()  # Both source header spellings preserved.
    assert len(visits) == 118 and visits.groupby("subject_id").size().ge(2).sum() == 29
    assert not observations.duplicated(["subject_id", "observed_at", "variable"]).any()
    assert not observations.available_time_known.any()
    assert common.candidate_common_key.nunique() == 23
    assert not common.Group.isin(["Urinalysis", "i-STAT Profile", "SPE Panel"]).any()
    assert set(common.mapping_status) == {"confirmed"}
    assert not common.loc[common.quality_flag.isin(["censored_inequality", "numeric_with_footnote_review"]), "observed_exact_numeric"].any()
    assert common.loc[common.quality_flag.eq("numeric_with_reference_flag"), "observed_exact_numeric"].all()
    # PRE magnitudes are not used as positive global elapsed mission days.
    assert visits.loc[visits.Test_Phase.eq("PRE_TEST"), "BR_Day"].gt(0).all()
    for _, sub in visits.groupby("subject_id"):
        dates = pd.to_datetime(sub.observed_at)
        np.testing.assert_array_equal((dates - dates.min()).dt.days, sub.day_from_first_sample)


def test_sleep_duration_units_do_not_treat_minutes_as_decimal_hours():
    assert sleep.duration_hours("0:14") == pytest.approx(14 / 60)
    assert sleep.duration_hours("7:44:00") == pytest.approx(7 + 44 / 60)
    assert sleep.duration_hours("0") == 0
    assert np.isnan(sleep.duration_hours(np.nan))
    for value in ["7:90", "7.44", "unknown"]:
        with pytest.raises(ValueError):
            sleep.duration_hours(value)


def test_sleep_missing_days_stay_missing_without_duplicate_query_copy():
    path = ROOT / "data/processed/utmb_c1/daily_sleep.csv"
    if not path.exists():
        pytest.skip("Prepare UTMB C1 actigraphy first")
    data = pd.read_csv(path)
    assert len(data) == 207 and data.subject_id.nunique() == 3
    assert not data.duplicated(["subject_id", "observed_at"]).any()
    for variable in ["sleep_duration_hours", "sleep_latency_hours", "sleep_efficiency_percent"]:
        assert data[variable].count() == 190
    assert not data.available_time_known.any()
