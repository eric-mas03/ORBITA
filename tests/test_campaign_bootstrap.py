import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("bootstrap_report", Path(__file__).resolve().parents[1] / "scripts/bootstrap_campaign_confirmation.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_paired_stratified_bootstrap_preserves_known_constant_improvement():
    rows = []
    for method, gap in [("linear_quantile", 2.), ("candidate", 1.)]:
        for site in ("A", "B"):
            for episode in range(3):
                rows.append(dict(split="confirmation", method=method, episode_id=f"{site}/{episode}",
                    site=site, head="vitals", variable="HR", origin_hour=1, target_hour=5,
                    horizon_hours=4, observed=10., q10=10-gap, q50=10., q90=10+gap, target_scale=2.))
    frame = pd.DataFrame(rows)
    result = module.paired_intervals(frame, replicates=30, minimum_episodes=2).iloc[0]
    assert result.relative_improvement == pytest.approx(.5)
    assert result.relative_improvement_ci95_lower == pytest.approx(.5)
    assert result.relative_improvement_ci95_upper == pytest.approx(.5)
    assert result.valid_replicates == 30
    assert result.supported_cells == 1
    with pytest.raises(ValueError, match="adequately supported"):
        module.paired_intervals(frame, minimum_episodes=7)
