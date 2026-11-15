"""Data-contract checks against the public NASA files; no synthetic physiology."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_inspiration4", ROOT / "scripts/analyze_osd575.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


@pytest.fixture(scope="module")
def observations():
    return eda.load_observations(ROOT / "data/raw/OSD-575", ROOT / "data/raw/OSD-569")


def test_temporal_parser_preserves_launch_and_return_references():
    assert eda.parse_time(" L-92 ") == ("launch", -92)
    assert eda.parse_time("R+194") == ("return", 194)
    for unsupported in ("FD1", "2021-09-15", "L92", "R+1trailing", ""):
        with pytest.raises(ValueError):
            eda.parse_time(unsupported)


def test_subject_visit_identity_and_no_duplicated_measures(observations):
    data = observations
    assert set(data.subject_id) == {"C001", "C002", "C003", "C004"}
    times = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
    assert data.sort_values("visit_index").timepoint.drop_duplicates().tolist() == times
    assert len(data[["subject_id", "timepoint"]].drop_duplicates()) == 28
    assert not data.duplicated(["subject_id", "timepoint", "assay", "analyte"]).any()
    assert not data.available_time_known.any()
    assert set(data.loc[data.timepoint.str.startswith("L"), "time_reference"]) == {"launch"}
    assert set(data.loc[data.timepoint.str.startswith("R"), "time_reference"]) == {"return"}


def test_panel_counts_and_missing_alamar_visit(observations):
    data = observations
    expected = {"CMP": 19, "EVE_CARDIO": 9, "EVE_IMMUNE": 71, "ALAMAR": 203, "CBC": 20}
    assert data.groupby("assay").variable.nunique().to_dict() == expected
    alamar = data[data.assay.eq("ALAMAR")]
    assert len(alamar) == 27 * 203
    pairs = set(zip(alamar.subject_id, alamar.timepoint))
    assert len(pairs) == 27
    assert ("C003", "L-44") not in pairs
    assert set(alamar.unit) == {"NPQ (log2)"}


def test_percent_derivatives_and_reference_ranges_are_not_measures(observations):
    data = observations
    # A CBC differential percentage is an observed measure; exclude derived
    # percentage-normalized assay copies, not genuine percentage units.
    assert not data.analyte.str.contains("percent_normalized|range_min|range_max", case=False).any()
    eve = data[data.assay.str.startswith("EVE_")]
    assert not eve.unit.str.contains("%|percent", case=False).any()
    assert not eve.analyte.str.contains("percent", case=False).any()
    assert data[data.assay.eq("CBC")].unit.eq("%").any()
    assert not data.analyte.str.contains(r"\(FEMALE\)", case=False).any()


def test_cbc_hemoglobin_preserves_original_units_and_sex_alias_values(observations):
    original = pd.read_csv(ROOT / "data/raw/OSD-569/LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv")
    original = original[original.ANALYTE.isin(["HEMOGLOBIN", "HEMOGLOBIN (FEMALE)"])]
    actual = observations[observations.assay.eq("CBC") & observations.analyte.eq("HEMOGLOBIN")]
    assert len(actual) == len(original) == 28
    assert set(actual.unit) == set(original.UNITS) == {"g/dL"}
    expected_values = original.set_index(["SUBJECT_ID", "TEST_DATE"]).VALUE.astype(float).sort_index()
    actual_values = actual.set_index(["subject_id", "timepoint"]).value.sort_index()
    assert actual_values.index.tolist() == expected_values.index.tolist()
    np.testing.assert_array_equal(actual_values.to_numpy(), expected_values.to_numpy())


def test_censored_parser_value_is_not_silently_numeric(monkeypatch):
    # The downloaded measurement cells have no inequality strings. Inject a
    # parser sentinel into an in-memory copy to verify future files safely.
    read_csv = pd.read_csv

    def read_with_sentinel(path, *args, **kwargs):
        frame = read_csv(path, *args, **kwargs)
        if Path(path).name == "LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv":
            frame["VALUE"] = frame["VALUE"].astype(object)
            frame.loc[0, "VALUE"] = "<5"
        return frame

    monkeypatch.setattr(eda.pd, "read_csv", read_with_sentinel)
    parsed = eda.load_observations(ROOT / "data/raw/OSD-575")
    censored = parsed[parsed.raw_value.eq("<5")]
    assert len(censored) == 1
    assert censored.value.isna().all()
    assert not censored.quality_flag.eq("numeric_as_published").any()


def test_vectorized_correlations_match_pairwise_reference_with_missing(observations):
    data = observations
    columns = ["CBC::MCHC", "ALAMAR::activin_a"]
    wide = data.pivot(index=["subject_id", "visit_index", "timepoint"], columns="variable", values="value")[columns]
    assert wide.isna().any().all()
    result = eda.pair_correlations(wide, {"CBC::MCHC": "CBC", "ALAMAR::activin_a": "ALAMAR"}).iloc[0]
    paired = wide.dropna()
    assert result.n_pairs == len(paired)
    assert result.n_subjects == paired.index.get_level_values("subject_id").nunique()
    np.testing.assert_allclose(result.spearman_pooled, spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic, atol=1e-12)
    # Match the documented marginal ranks, then center only complete pairs
    # within each subject. Independently compute the scalar Pearson formula.
    marginal = wide.rank().dropna()
    residuals = []
    subject_rhos = []
    for subject in marginal.index.get_level_values("subject_id").unique():
        values = marginal.xs(subject, level="subject_id").to_numpy()
        residuals.append(values - values.mean(axis=0))
        person = paired.xs(subject, level="subject_id")
        if len(person) >= 4:
            subject_rhos.append(spearmanr(person.iloc[:, 0], person.iloc[:, 1]).statistic)
    centered = np.concatenate(residuals)
    expected = np.dot(centered[:, 0], centered[:, 1]) / np.sqrt(np.square(centered[:, 0]).sum() * np.square(centered[:, 1]).sum())
    np.testing.assert_allclose(result.within_person_rank_centered, expected, atol=1e-12)
    np.testing.assert_allclose([result.subject_rho_min, result.subject_rho_max], [min(subject_rhos), max(subject_rhos)], atol=1e-12)
