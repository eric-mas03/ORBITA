"""Normalize public UTMB Campaign 3 clinical observations without mixing domains."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_measurement(value):
    """Retain reference-range flags; do not turn ranges into point labels."""
    if pd.isna(value) or not str(value).strip():
        return np.nan, "missing", "", False, "", np.nan
    raw = str(value).strip()
    censored = re.fullmatch(rf"([<>]=?)\s*({NUMBER})", raw)
    if censored:
        return np.nan, "censored_inequality", "", False, censored[1], float(censored[2])
    numeric = re.fullmatch(rf"({NUMBER})\s*((?:(?:\((?:H|L|CH|CL)\)|[A-Z])\s*)*)", raw)
    if numeric:
        suffix = numeric[2].strip()
        tokens = re.findall(r"\((?:H|L|CH|CL)\)|[A-Z]", suffix)
        footnote = any(not token.startswith("(") for token in tokens)
        quality = ("numeric_with_footnote_review" if footnote else
                   "numeric_with_reference_flag" if suffix else "numeric_as_published")
        # Local footnotes may describe assay changes/interference. Preserve
        # their number, but wait for a per-file note review before point loss.
        return float(numeric[1]), quality, suffix, not footnote, "", np.nan
    return np.nan, "text_or_flag_requires_review", "", False, "", np.nan


def clean_label(value):
    if pd.isna(value):
        return "unknown"
    return re.sub(r"\s+", " ", str(value).replace("\\n", " ").replace("\\r", " ")).strip()


def main():
    raw = ROOT / "data/raw/LSDA-UTMB-C3"
    out = ROOT / "data/processed/utmb_c3"
    report = ROOT / "outputs/data_expansion"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in sorted(raw.glob("*.csv")):
        frame = pd.read_csv(path, dtype={"Measured_Value": str, "Subject": str})
        # C3A/B use a different header; harmonize each table before concat.
        frame = frame.rename(columns={"Parameter Name": "ParameterName"})
        expected = {"Subject", "Campaign", "Date", "Group", "ParameterName", "Units", "Measured_Value", "Test_Phase", "BR_Day"}
        if not expected.issubset(frame):
            raise ValueError(f"Unexpected schema: {path.name}")
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(len(frame)) + 2
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for col in ("Group", "ParameterName", "Units", "Test", "Campaign", "Subject"):
        data[col] = data[col].map(clean_label)
    data["subject_id"] = "UTMB_C3_" + data.Subject
    data["domain"] = "space_analog_bedrest"
    data["cohort_id"] = "UTMB_Campaign3"
    data["observed_at"] = pd.to_datetime(data.Date, format="%m/%d/%Y", errors="raise")
    data["raw_value"] = data.Measured_Value
    parsed = data.raw_value.map(parse_measurement)
    fields = ["value", "quality_flag", "measurement_suffix", "observed_exact_numeric", "censor_direction", "censor_bound"]
    for i, field in enumerate(fields):
        data[field] = parsed.map(lambda result: result[i])
    data["available_time_known"] = False
    # Matrix/group disambiguates serum glucose, urine glucose and i-STAT glucose.
    data["variable"] = "UTMB::" + data.Group + "::" + data.ParameterName + "::" + data.Units
    keys = ["subject_id", "observed_at", "variable"]
    if data.duplicated(keys).any():
        raise ValueError("Repeated measurements collide; do not average without review")
    data = data.sort_values(["subject_id", "observed_at", "variable"])
    data.to_csv(out / "observations.csv", index=False)
    visits = data[["subject_id", "observed_at", "Campaign", "Test_Phase", "BR_Day", "Date"]].drop_duplicates()
    if visits.duplicated(["subject_id", "observed_at"]).any():
        raise ValueError("Multiple phase/day labels on the same subject/date")
    visits = visits.sort_values(["subject_id", "observed_at"])
    visits["day_from_first_sample"] = (visits.observed_at - visits.groupby("subject_id").observed_at.transform("min")).dt.days
    visits["visit_index"] = visits.groupby("subject_id").cumcount()
    visits.to_csv(out / "visits.csv", index=False)
    eligible = data[data.observed_exact_numeric]
    wide = eligible.pivot(index=["subject_id", "observed_at"], columns="variable", values="value")
    wide = wide.reindex(pd.MultiIndex.from_frame(visits[["subject_id", "observed_at"]]))
    wide.to_csv(out / "numeric_wide.csv")
    dictionary = data.groupby("variable").agg(
        group=("Group", "first"), parameter=("ParameterName", "first"), unit=("Units", "first"),
        n_rows=("value", "size"), n_numeric=("observed_exact_numeric", "sum"), n_subjects=("subject_id", "nunique"))
    repeated = eligible.groupby(["variable", "subject_id"]).size().ge(2).groupby(level="variable").sum()
    dictionary["subjects_with_at_least_2_numeric_visits"] = repeated.reindex(dictionary.index, fill_value=0)
    dictionary.to_csv(report / "utmb_c3_variable_dictionary.csv")
    summary = {"dataset": "NASA LSDA UTMB Campaign3 clinical laboratory assessment", "source_files": len(frames),
               "domain": "space_analog_bedrest", "subject_ids_within_cohort": data.subject_id.nunique(),
               "subject_visits": len(visits), "rows": len(data), "source_group_parameter_unit_keys": data.variable.nunique(),
               "exact_numeric_rows": int(data.observed_exact_numeric.sum()),
               "quality_counts": data.quality_flag.value_counts().to_dict(),
               "visits_per_subject": visits.groupby("subject_id").size().to_dict(),
               "subjects_with_repeated_visits": int(visits.groupby("subject_id").size().ge(2).sum()),
               "time_policy": "Use observed Date for elapsed days; preserve Test_Phase and original BR_Day separately.",
               "real_astronauts_added": 0,
               "cross_campaign_identity": "No subject crosswalk available; do not claim globally unique added people across campaigns.",
               "available_time_known": False, "training_status": "not_trained_not_merged_with_astronauts"}
    mapping_path = ROOT / "config/utmb_candidate_mapping.json"
    if mapping_path.exists():
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = raw_mapping if isinstance(raw_mapping, list) else raw_mapping["mappings"]
        mapped = []
        for item in mapping:
            if not item.get("auto_harmonize", False):
                continue
            group, parameter, unit = (clean_label(item[k]) for k in ("group", "parameter", "source_unit"))
            subset = data[data.Group.eq(group) & data.ParameterName.eq(parameter) & data.Units.eq(unit)].copy()
            if subset.empty:
                raise ValueError(f"Mapping has no source match: {item}")
            subset["candidate_common_key"] = item["target"]
            subset["candidate_common_unit"] = item["unit"]
            subset["candidate_common_value"] = subset.value * item["factor"]
            subset["mapping_status"] = item["status"]
            mapped.append(subset)
        common = pd.concat(mapped, ignore_index=True)
        if common.duplicated(["subject_id", "observed_at", "candidate_common_key"]).any():
            raise ValueError("Multiple source assays map onto the same candidate target; review before pivot")
        common.to_csv(out / "candidate_common_observations.csv", index=False)
        exact = common[common.observed_exact_numeric]
        common_wide = exact.pivot(index=["subject_id", "observed_at"], columns="candidate_common_key", values="candidate_common_value")
        common_wide.to_csv(out / "candidate_common_wide.csv")
        counts = exact.groupby(["candidate_common_key", "subject_id"]).size().ge(2).groupby(level="candidate_common_key").sum()
        summary["candidate_common_variables"] = common.candidate_common_key.nunique()
        summary["common_subjects_with_at_least_two_numeric_visits"] = counts.to_dict()
        summary["harmonization_enabled_for_training"] = False
    (report / "utmb_c3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
