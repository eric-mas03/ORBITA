import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("common_scale_report", Path(__file__).resolve().parents[1] / "scripts/report_v1_v2_common_scale.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_changed_training_scale_cannot_create_a_reporting_improvement(tmp_path):
    rows = [dict(split="dev",method=method,episode_id="fixture",site="A",origin_hour=1,
        target_hour=2,horizon_hours=1,head="vitals",variable="HR",unit="fixture-unit",
        unit_review_required=False,observed=10.,q10=8.,q50=10.,q90=12.,history_hours=1,
        own_history_observed=1,used_train_median_fallback=False,target_scale=scale)
        for method,scale in [("a",2.),("b",20.)]]
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path,index=False)
    scores,cells,identity = module.common_scale_scores([path],{"HR":4.})
    assert scores.loc["a","macro_pinball"] == pytest.approx(scores.loc["b","macro_pinball"])
    assert scores.loc["a","macro_pinball"] == pytest.approx((.2+.2)/3/4)
    assert len(identity) == 1
