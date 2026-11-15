"""Guard/frozen-preparation tests. No reserved confirmation file is opened."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "confirmation_evaluator", Path(__file__).resolve().parents[1] / "scripts/evaluate_confirmation_campaign.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def frozen_selection(root):
    directory = root / "outputs/model_campaign"
    directory.mkdir(parents=True)
    evidence = root / "candidate_metadata.json"
    evidence.write_text("metadata only")
    selection = {"selection_split": "dev", "confirmation_used_for_selection": False,
                 "calibration_used_for_selection": False, "ensemble_trained": False,
                 "selected_for_confirmation": ["linear_quantile", "hist_boost_quantile"],
                 "evidence_sha256": {str(evidence): evaluator.file_sha256(evidence)}}
    path = directory / "selection_frozen.json"
    path.write_text(json.dumps(selection))
    return selection, path, evidence


def test_canonical_freeze_is_mandatory_even_if_smoke_selection_exists(tmp_path):
    smoke = tmp_path / "outputs/model_campaign_smoke"
    smoke.mkdir(parents=True)
    (smoke / "selection_frozen.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Canonical"):
        evaluator.guard_selection(tmp_path)
    assert not (tmp_path / "outputs/model_campaign/confirmation_run").exists()


def test_frozen_evidence_and_shortlist_are_checked_before_data_access(tmp_path):
    selection, path, evidence = frozen_selection(tmp_path)
    assert evaluator.guard_selection(tmp_path)[0] == selection
    evidence.write_text("changed metadata")
    with pytest.raises(ValueError, match="evidence changed"):
        evaluator.guard_selection(tmp_path)
    evidence.write_text("metadata only")
    selection["selected_for_confirmation"].append("not_selected_model")
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="supported new"):
        evaluator.guard_selection(tmp_path)


def test_existing_confirmation_cannot_be_overwritten(tmp_path):
    _, path, _ = frozen_selection(tmp_path)
    (path.parent / "confirmation_predictions.csv").write_text("previous result")
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluator.guard_selection(tmp_path)
    assert (path.parent / "confirmation_predictions.csv").read_text() == "previous result"


def test_selection_using_confirmation_or_calibration_is_rejected(tmp_path):
    selection, path, _ = frozen_selection(tmp_path)
    selection["confirmation_used_for_selection"] = True
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="development-only"):
        evaluator.guard_selection(tmp_path)


def synthetic_packed():
    # Generic tensor fixture for software invariants, never physiological data.
    from astronaut_twin.hourly_data import DYNAMIC
    values = np.tile(np.arange(1, 35, dtype=np.float32), (30, 1))
    return {"values": values, "times": np.arange(1, 31, dtype=np.int64),
            "episode_offsets": np.array([0, 30]), "episode_ids": np.array(["fixture-episode"]),
            "sites": np.array(["fixture-site"]), "episode_split": np.array(["train"]),
            "origins": np.array([23]), "history_start": np.array([0]),
            "target_rows": np.array([[24, 27, -1, -1]]), "sample_episode": np.array([0]),
            "variables": np.array(DYNAMIC), "horizons": np.array([1, 4, 8, 24])}


def test_confirmation_override_precedes_frozen_preparation_without_refit():
    from astronaut_twin.hourly_architecture_trials import prepare_frozen_trial_data
    original = synthetic_packed()
    packed = evaluator.mark_confirmation(original)
    assert original["episode_split"].tolist() == ["train"]
    assert packed["episode_split"].tolist() == ["confirmation"]
    scalers = {"variables": packed["variables"], "mean": np.full(34, 10., dtype=np.float32),
               "scale": np.full(34, 2., dtype=np.float32), "median": np.full(34, 9., dtype=np.float32)}
    evaluator.verify_frozen_scalers(packed, scalers)
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    assert neural.split.tolist() == baseline.split.tolist() == ["confirmation"]
    np.testing.assert_array_equal(neural.mean, scalers["mean"])
    np.testing.assert_array_equal(baseline.scale, scalers["scale"])
    assert np.isnan(baseline.labels[0, 2:]).all()
    changed = dict(packed)
    changed["values"] = packed["values"] + 100
    other, _ = prepare_frozen_trial_data(changed, scalers)
    np.testing.assert_array_equal(other.mean, neural.mean)
    np.testing.assert_array_equal(other.scale, neural.scale)


def test_packing_uses_isolated_directory_and_marks_all_episodes_confirmation(tmp_path):
    from astronaut_twin.hourly_data import DYNAMIC
    source = tmp_path / "isolated_source/training_setA"
    source.mkdir(parents=True)
    header = "|".join(DYNAMIC + ["ICULOS"])
    # Pure software fixture, unrelated to the unopened real confirmation files.
    lines = ["|".join(["1"] * len(DYNAMIC) + [str(t)]) for t in range(1, 31)]
    (source / "fixture.psv").write_text(header + "\n" + "\n".join(lines) + "\n")
    run = tmp_path / "isolated_run"
    run.mkdir()
    packed, _ = evaluator.pack_confirmation(source.parent, run)
    assert set(packed["episode_split"]) == {"confirmation"}
    assert len(packed["episode_ids"]) == 1
    assert (run / "confirmation_packed.npz").exists()
    assert len(packed["origins"]) > 0
    # The original source is never rewritten by the packing override.
    assert (source / "fixture.psv").read_text().startswith(header)


def test_wrong_split_or_variable_order_cannot_reach_frozen_inference():
    packed = synthetic_packed()
    scalers = {"variables": packed["variables"], "mean": np.zeros(34), "scale": np.ones(34), "median": np.zeros(34)}
    with pytest.raises(ValueError, match="confirmation-only"):
        evaluator.verify_frozen_scalers(packed, scalers)
    packed = evaluator.mark_confirmation(packed)
    scalers["variables"] = scalers["variables"][::-1]
    with pytest.raises(ValueError, match="variable order"):
        evaluator.verify_frozen_scalers(packed, scalers)


def test_export_is_sparse_terminal_bin_and_never_repairs_crossed_quantiles():
    from astronaut_twin.hourly_architecture_trials import prepare_frozen_trial_data
    packed = evaluator.mark_confirmation(synthetic_packed())
    scalers = {"variables": packed["variables"], "mean": np.zeros(34), "scale": np.ones(34), "median": np.zeros(34)}
    _, baseline = prepare_frozen_trial_data(packed, scalers)
    prediction = np.broadcast_to(np.array([1., 2., 3.]), (1, 4, 34, 3)).copy()
    metadata = {"variables": [{"source_key": str(v), "group": "fixture-group", "source_unit_as_documented": "fixture-unit"} for v in packed["variables"]]}
    table = evaluator.prediction_table("linear_quantile", prediction, packed, baseline, np.ones((4, 34)), metadata)
    assert len(table) == 68
    assert set(table.split) == {"confirmation"}
    assert set(table.horizon_hours) == {1, 4}
    assert np.array_equal(table.target_hour - table.origin_hour, table.horizon_hours)
    prediction[0, 0, 0] = [3., 2., 1.]
    with pytest.raises(ValueError, match="do not repair"):
        evaluator.prediction_table("linear_quantile", prediction, packed, baseline, np.ones((4, 34)), metadata)
