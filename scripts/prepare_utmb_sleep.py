"""Prepare the daily sleep modality for the same three UTMB C1 lab subjects."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def duration_hours(value):
    if pd.isna(value) or not str(value).strip():
        return float("nan")
    if str(value).strip() == "0":
        return 0.  # Three published latency cells use literal zero.
    match = re.fullmatch(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", str(value).strip())
    if not match:
        raise ValueError(f"Not a documented HH:mm[:ss] duration: {value!r}")
    hours, minutes, seconds = int(match[1]), int(match[2]), int(match[3] or 0)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid duration: {value!r}")
    return hours + minutes / 60 + seconds / 3600


def main():
    raw = ROOT / "data/raw/LSDA-UTMB-C1-ACTIGRAPHY"
    source = raw / "BRSMACT_Campagin_1_Actigraphy.csv"
    data = pd.read_csv(source)
    data["subject_id"] = "UTMB_C1_" + data.Subject
    data["observed_at"] = pd.to_datetime(data.Date, format="%m/%d/%Y", errors="raise")
    data["domain"] = "space_analog_bedrest"
    data["cohort_id"] = "UTMB_Campaign1"
    data["sleep_latency_hours"] = data["Sleep Latency (hours)"].map(duration_hours)
    data["sleep_duration_hours"] = data["Sleep Duration (hours)"].map(duration_hours)
    data["sleep_efficiency_percent"] = data["Sleep Efficiency (%)"]
    data["available_time_known"] = False
    data["source_file"] = source.name
    data["source_row"] = range(2, len(data) + 2)
    if data.duplicated(["subject_id", "observed_at"]).any():
        raise ValueError("Duplicate daily sample; do not append overlapping Query_Result copy")
    data = data.sort_values(["subject_id", "observed_at"])
    data["day_from_first_sample"] = (data.observed_at - data.groupby("subject_id").observed_at.transform("min")).dt.days
    if not data.sleep_efficiency_percent.dropna().between(0, 100).all():
        raise ValueError("Efficiency outside percent support; review source")
    out = ROOT / "data/processed/utmb_c1"
    report = ROOT / "outputs/data_expansion"
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "daily_sleep.csv", index=False)
    summary = {"dataset": "UTMB Campaign1 actigraphy", "subjects": data.subject_id.nunique(),
               "subject_days": len(data), "calendar_dates": data.observed_at.nunique(),
               "days_per_subject": data.groupby("subject_id").size().to_dict(),
               "numeric_observed": {key: int(data[key].notna().sum()) for key in ["sleep_latency_hours", "sleep_duration_hours", "sleep_efficiency_percent"]},
               "sleep_duration_hours_range": [float(data.sleep_duration_hours.min()), float(data.sleep_duration_hours.max())],
               "cohort_overlap": "Same three participants as UTMB C1 clinical labs; not three additional people.",
               "training_status": "not_trained", "available_time_known": False,
               "source_copy_policy": "Use complete 207-row table, not overlapping 191-row Query_Result"}
    (report / "utmb_c1_sleep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
