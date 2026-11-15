"""Guard scoring integrity: units, quantile order and matched target identity."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location("campaign_report", Path(__file__).resolve().parents[1] / "scripts/report_model_campaign.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def example(**changes):
    row = dict(observed=10., q10=8., q50=10., q90=12., target_scale=2.,
               origin_hour=1, target_hour=5, horizon_hours=4, variable="HR")
    row.update(changes)
    return pd.DataFrame([row])


def test_pinball_uses_all_three_quantiles_and_training_scale():
    scored = report.annotate(example())
    assert scored.loc[0, "pinball"] == pytest.approx((.2 + 0 + .2) / 3)
    assert scored.loc[0, "normalized_pinball"] == pytest.approx(.4 / 6)
    assert scored.loc[0, "covered"]


@pytest.mark.parametrize("changes", [dict(q10=11.), dict(target_scale=0.), dict(target_hour=6), dict(q90=float("nan"))])
def test_rejects_invalid_forecasts(changes):
    with pytest.raises(ValueError):
        report.annotate(example(**changes))


def test_support_audit_does_not_silently_clip_forecasts():
    scored = report.annotate(example(q10=-1.))
    assert scored.loc[0, "support_violation"]
    assert scored.loc[0, "q10"] == -1.
    assert not scored.loc[0, "median_support_violation"]


def test_missing_method_targets_cannot_improve_ranking_by_omission():
    rows = pd.DataFrame([
        dict(split="dev", method=m, episode_id="A/p1", site="A", head="vitals", origin_hour=1,
             horizon_hours=h, variable="HR", target_hour=1+h, observed=80., target_scale=10.)
        for m in ("a", "b") for h in (1, 4)])
    with pytest.raises(ValueError, match="same target identities"):
        report.utils.validate_shared_predictions(rows.iloc[:-1])
