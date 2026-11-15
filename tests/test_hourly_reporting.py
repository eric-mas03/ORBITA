"""Evaluation contracts: identical targets and explicit missing metric cells."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hourly_reporting_audit", ROOT / "scripts/report_hourly_baselines.py")
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def test_comparison_rejects_different_targets_or_easier_subsets():
    rows = []
    for method in ("baseline", "neural"):
        for episode in ("A/one", "B/two"):
            rows.append({"split": "test", "method": method, "episode_id": episode,
                         "origin_hour": 10, "target_hour": 14, "horizon_hours": 4,
                         "variable": "X", "head": "group", "site": episode[0],
                         "observed": 2., "target_scale": .5})
    pred = pd.DataFrame(rows)
    assert reporting.validate_shared_predictions(pred) == {"baseline": 2, "neural": 2}
    changed = pred.copy()
    changed.loc[changed.method.eq("neural"), "observed"] += 1
    with pytest.raises(ValueError, match="different observed"):
        reporting.validate_shared_predictions(changed)
    with pytest.raises(ValueError, match="same target identities"):
        reporting.validate_shared_predictions(pred.iloc[:-1])


def test_empty_target_horizon_is_not_a_zero_error():
    scored = pd.DataFrame([{"split": "test", "method": "reference", "variable": "X", "horizon_hours": 1,
                            "n_labels": 2, "n_episodes": 2, "mae": 1., "coverage": .5}])
    config = {"variables": [{"source_key": key, "group": "group", "source_unit_as_documented": "native"} for key in ("X", "Y")]}
    result = reporting.complete_metrics(scored, config, ["reference"], ["test"])
    assert len(result) == 8
    missing = result[result.n_labels.eq(0)]
    assert len(missing) == 7 and missing.mae.isna().all() and missing.coverage.isna().all()
    assert missing.support_status.eq("not_evaluable").all()
