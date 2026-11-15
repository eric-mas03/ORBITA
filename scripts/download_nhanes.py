"""Acquire public cross-sectional lab data for future encoder pretraining.

These participants are terrestrial survey participants, NOT astronauts and
NOT repeated visits. Never pass this table to a temporal forecasting dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
# Codebook units, not name matching. Values from separate assays/domains must
# keep their source identities even when a common biological label is useful.
# (source, common target, source unit, target unit, conversion, comment flag)
MAPPING = [
    ("LBXSNASI", "CMP::SODIUM", "mmol/L", "mmol/L", 1, None),
    ("LBXSKSI", "CMP::POTASSIUM", "mmol/L", "mmol/L", 1, None),
    ("LBXSCLSI", "CMP::CHLORIDE", "mmol/L", "mmol/L", 1, None),
    ("LBXSC3SI", "CMP::CARBON DIOXIDE", "mmol/L bicarbonate", "mmol/L", 1, None),
    ("LBXSCA", "CMP::CALCIUM", "mg/dL", "mg/dL", 1, None),
    ("LBXSBU", "CMP::UREA NITROGEN (BUN)", "mg/dL", "mg/dL", 1, None),
    ("LBXSCR", "CMP::CREATININE", "mg/dL", "mg/dL", 1, None),
    ("LBXSGL", "CMP::GLUCOSE", "mg/dL", "mg/dL", 1, None),
    ("LBXSAPSI", "CMP::ALKALINE PHOSPHATASE", "IU/L", "U/L", 1, None),
    ("LBXSATSI", "CMP::ALT", "U/L", "U/L", 1, "LBDSATLC"),
    ("LBXSASSI", "CMP::AST", "U/L", "U/L", 1, None),
    ("LBXSTB", "CMP::BILIRUBIN; TOTAL", "mg/dL", "mg/dL", 1, "LBDSTBLC"),
    ("LBXSAL", "CMP::ALBUMIN", "g/dL", "g/dL", 1, None),
    ("LBXSTP", "CMP::PROTEIN; TOTAL", "g/dL", "g/dL", 1, None),
    ("LBXRBCSI", "CBC::RED BLOOD CELL COUNT", "million cells/uL", "Million/uL", 1, None),
    ("LBXHGB", "CBC::HEMOGLOBIN", "g/dL", "g/dL", 1, None),
    ("LBXHCT", "CBC::HEMATOCRIT", "%", "%", 1, None),
    ("LBXMCVSI", "CBC::MCV", "fL", "fL", 1, None),
    ("LBXMCHSI", "CBC::MCH", "pg", "pg", 1, None),
    ("LBXMC", "CBC::MCHC", "g/dL", "g/dL", 1, None),
    ("LBXRDW", "CBC::RDW", "%", "%", 1, None),
    ("LBXPLTSI", "CBC::PLATELET COUNT", "1000 cells/uL", "Thousand/uL", 1, None),
    ("LBXMPSI", "CBC::MPV", "fL", "fL", 1, None),
    ("LBXWBCSI", "CBC::WHITE BLOOD CELL COUNT", "1000 cells/uL", "Thousand/uL", 1, None),
    ("LBDNENO", "CBC::ABSOLUTE NEUTROPHILS", "1000 cells/uL (derived)", "cells/uL", 1000, None),
    ("LBDLYMNO", "CBC::ABSOLUTE LYMPHOCYTES", "1000 cells/uL (derived)", "cells/uL", 1000, None),
    ("LBDMONO", "CBC::ABSOLUTE MONOCYTES", "1000 cells/uL (derived)", "cells/uL", 1000, None),
    ("LBDEONO", "CBC::ABSOLUTE EOSINOPHILS", "1000 cells/uL (derived)", "cells/uL", 1000, None),
    ("LBDBANO", "CBC::ABSOLUTE BASOPHILS", "1000 cells/uL (derived)", "cells/uL", 1000, None),
]


def acquire(directory: Path, filename: str, refresh: bool) -> dict:
    target = directory / filename
    url = f"{BASE}/{filename}"
    if refresh or not target.exists():
        with urlopen(Request(url, headers={"User-Agent": "AstronautTwinResearch/0.2"}), timeout=30) as response:
            body = response.read(10_000_001)
        if len(body) > 10_000_000:
            raise ValueError("Unexpectedly large public table")
        if filename.endswith(".xpt") and not body.startswith(b"HEADER RECORD"):
            raise ValueError("Not a SAS transport table; do not cache an error page")
        target.write_bytes(body)
    body = target.read_bytes()
    return {"dataset": "NHANES_2017_2018", "domain": "terrestrial_cross_sectional",
            "url": url, "path": str(target.relative_to(ROOT)), "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(), "public_no_credentials": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "data/raw/external/NHANES_2017_2018"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    tables = {}
    for name in ("CBC_J", "BIOPRO_J", "DEMO_J"):
        for suffix in (".xpt", ".htm"):
            entry = acquire(raw, name + suffix, args.refresh)
            manifest.append(entry)
            print(f"Verified {name + suffix}: {entry['bytes']} bytes", flush=True)
        tables[name] = pd.read_sas(raw / (name + ".xpt"), format="xport")
        if tables[name].SEQN.duplicated().any():
            raise ValueError("Participant ID is not unique in source")
    manifest_path = ROOT / "data/manifests/nhanes_downloads.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    demographics = tables["DEMO_J"][["SEQN", "RIDAGEYR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]]
    labs = tables["CBC_J"].merge(tables["BIOPRO_J"], on="SEQN", how="outer", validate="one_to_one")
    joined = demographics.merge(labs, on="SEQN", how="inner", validate="one_to_one")
    adults = joined[joined.RIDAGEYR.ge(18)].copy()
    data = pd.DataFrame({"subject_id": "NHANES_J_" + adults.SEQN.astype(int).astype(str),
                         "domain": "terrestrial_cross_sectional", "age_years": adults.RIDAGEYR,
                         "survey_weight_mec_2yr": adults.WTMEC2YR,
                         "survey_psu": adults.SDMVPSU, "survey_stratum": adults.SDMVSTRA})
    observed = data[["subject_id"]].copy()
    mapping = []
    for source, target, source_unit, target_unit, factor, flag in MAPPING:
        if source not in adults:
            raise ValueError(f"Missing expected codebook variable {source}")
        data[target] = adults[source] * factor
        observed[target] = adults[source].notna()
        if flag:
            if flag not in adults:
                raise ValueError(f"Expected censoring flag missing: {flag}")
            # Flag 1 is an LLOD/sqrt(2) substitution, not an exact measurement.
            # pandas' IBM/SAS conversion can decode categorical zero as
            # 5.3976e-79. Validate the categorical domain before rounding.
            flag_values = adults[flag].dropna()
            if not ((flag_values - flag_values.round()).abs().le(1e-8).all()
                    and flag_values.round().isin([0., 1.]).all()):
                raise ValueError(f"Unrecognized detection flag values: {flag}")
            observed[target] &= adults[flag].round().eq(0)
        mapping.append({"source_variable": source, "candidate_common_key": target,
                        "source_unit": source_unit, "common_unit": target_unit, "factor": factor,
                        "censor_flag": flag, "censoring_policy": "flag1_not_exact_target" if flag else "published_values_censoring_not_fully_resolved",
                        "biomedical_review_status": "candidate_mapping_not_assay_equivalence",
                        "input_enabled": False, "forecasting_enabled": False})
    keys = [row[1] for row in MAPPING]
    active = observed[keys].any(axis=1)
    data, observed = data[active], observed[active]
    if data.subject_id.duplicated().any():
        raise ValueError("Duplicate exported participant")
    processed = ROOT / "data/processed/external"
    processed.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed / "nhanes_adult_labs.csv", index=False)
    observed.to_csv(processed / "nhanes_adult_labs_observed_mask.csv", index=False)
    output = ROOT / "outputs/data_expansion"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": "NHANES 2017-2018", "domain": "terrestrial_cross_sectional",
               "subjects_adult_with_labs": len(data), "variables_candidate_overlap": len(keys),
               "raw_rows": {name: len(frame) for name, frame in tables.items()},
               "observed_eligible_values": int(observed[keys].sum().sum()),
               "present_per_variable": observed[keys].sum().to_dict(),
               "use": "candidate encoder masked reconstruction pretraining; not temporal forecasting",
               "longitudinal": False, "real_astronauts_added": 0,
               "notes": ["No synthetic visit order or astronaut identity is assigned.",
                         "Common labels are candidates; methods, fasting state and handling differ.",
                         "Bicarbonate versus CMP total CO2 is a related measurement requiring review.",
                         "CBC absolute differential counts are calculated and rounded in NHANES.",
                         "Survey weights preserved; no population-representative inference performed.",
                         "No models are trained or combined with astronaut outcomes by this script."]}
    (output / "nhanes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "config/nhanes_candidate_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
