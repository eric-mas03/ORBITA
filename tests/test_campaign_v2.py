"""Isolation, update budgets, role filters and optimizer guards for V2."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import torch

from astronaut_twin.campaign_v2 import (ARCH_WORKERS, JOBS, NEURAL_METHODS, TRIAL_METHODS,
    _new_model, expanded_epoch_budget, freeze_components, prepare_campaign, read_json,
    sha, skip_unlabelled_heads, view_roles, write_json)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_tmp():
    base = ROOT / "outputs/model_campaign_v2/test_scratch"
    base.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=base) as name:
        resolved = Path(name).resolve()
        assert resolved.is_relative_to(base.resolve())
        yield resolved


def toy_packed():
    offsets = np.arange(0, 121, 30)
    origins = np.array([start + i for start in offsets[:-1] for i in (1, 4)])
    targets = origins[:, None] + np.array([1, 4, 8, 24])[None]
    episode = np.repeat(np.arange(4), 2)
    targets[targets >= offsets[episode + 1, None]] = -1
    return {"values": np.arange(120 * 34, dtype=np.float32).reshape(120, 34),
        "times": np.tile(np.arange(1, 31, dtype=np.float32), 4), "episode_offsets": offsets,
        "episode_ids": np.array(["a", "b", "c", "d"]), "sites": np.array(["A"] * 4),
        "episode_split": np.array(["train", "dev", "meta", "calibration"]), "origins": origins,
        "history_start": offsets[episode], "target_rows": targets, "sample_episode": episode,
        "variables": np.array([f"x{i}" for i in range(34)]), "horizons": np.array([1, 4, 8, 24])}


def test_budget_preserves_approximate_maximum_optimizer_updates():
    assert expanded_epoch_budget(20, 21661, 56669) == 8
    assert expanded_epoch_budget(15, 21661, 56669) == 6
    assert expanded_epoch_budget(30, 21661, 56669) == 12
    with pytest.raises(ValueError):
        expanded_epoch_budget(20, 1, 0)


def test_role_view_excludes_meta_calibration_before_windows_without_mutating_source():
    original = toy_packed()
    view = view_roles(original, ("train", "dev"))
    assert len(view["origins"]) == 4
    assert len(original["origins"]) == 8
    assert view["values"] is original["values"]
    assert (view["sample_episode"] < 2).all()
    np.testing.assert_array_equal(view["target_rows"], original["target_rows"][:4])


def test_prepare_isolated_snapshots_and_matching_audit_required(local_tmp):
    root = local_tmp / "project"
    for folder in ("src", "scripts", "config"):
        (root / folder).mkdir(parents=True)
    source = root / "src/model.py"
    source.write_text("# immutable V1 source\n")
    write_json(root / "config/hourly_task_candidates.json", {
        "evidence": {"old_cohort_counts": True}, "primary_hourly_domain": {"dataset": "old"}})
    packed = toy_packed()
    path = root / "new.npz"
    np.savez_compressed(path, **packed)
    audit = root / "audit.json"
    acquisition = root / "data/manifests/hourly_expansion_v2.json"
    roles = root / "data/processed/hourly_v2/roles.json"
    write_json(acquisition, {"status": "toy structural audit"})
    write_json(roles, {"a": "train", "b": "dev", "c": "meta", "d": "calibration"})
    write_json(audit, {"status": "audited_nonfinal_v2", "dataset_sha256": sha(path),
                       "final_not_packed": True, "final_episode_ids": ["unseen"],
                       "source_manifest_sha256": sha(acquisition)})
    config = read_json(ROOT / "config/model_campaign_v2.json")
    config.update(dataset_path=str(path), audit_path=str(audit))
    before = sha(source)
    workspace = local_tmp / "campaign"
    prepared = prepare_campaign(root, workspace, config)
    assert prepared["status"] == "prepared_without_training"
    assert sha(source) == before
    with np.load(workspace / "data/processed/hourly/hourly_dataset.npz") as data:
        assert len(data["origins"]) == 4
    derived = read_json(workspace / "config/hourly_task_candidates.json")
    assert "evidence" not in derived
    assert read_json(root / "config/hourly_task_candidates.json")["evidence"]
    assert prepare_campaign(root, workspace, config)["status"] == "already_prepared"
    bad = read_json(audit)
    bad["final_episode_ids"] = ["a"]
    write_json(audit, bad)
    with pytest.raises(ValueError, match="Final IDs"):
        prepare_campaign(root, local_tmp / "bad", config)


@pytest.mark.parametrize("method", ["gru_shared_readout", "causal_residual", "group_temporal_residual"])
def test_v2_inactive_head_is_not_updated_by_adamw_but_shared_encoder_is(method):
    torch.set_num_threads(2)
    model = _new_model(method, {"a": ["x"], "b": ["y"]}, ("x", "y"), 42)
    core = model if method in NEURAL_METHODS else model.core
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    inactive = list(core.decoders["b"].parameters()) + list(core.output_layers["b"].parameters())
    if method == "group_temporal_residual":
        inactive += list(model.group_memories["b"].parameters())
    before = [parameter.detach().clone() for parameter in inactive]
    shared = core.fusion.weight.detach().clone()
    mask = torch.tensor([[[True, False]]])
    skip_unlabelled_heads(model, method, mask, {"a": [0], "b": [1]})
    optimizer.step()
    for a, b in zip(before, inactive):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert b.grad is None
    assert not torch.equal(shared, core.fusion.weight)


def test_resume_payload_with_optimizer_and_rng_is_weights_only_readable():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    state = {"optimizer": optimizer.state_dict(), "numpy_rng": np.random.default_rng(42).bit_generator.state,
             "torch_rng": torch.get_rng_state(), "best_state": {"x": parameter.detach()}}
    stream = BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    assert loaded["numpy_rng"] == state["numpy_rng"]


def test_component_freeze_uses_only_complete_dev_and_diverse_families(local_tmp):
    workspace = local_tmp / "freeze"
    workspace.mkdir()
    config = read_json(ROOT / "config/model_campaign_v2.json")
    write_json(workspace / "campaign_config.json", config)
    write_json(workspace / "prepared.json", {"dataset_sha256": "data"})
    write_json(workspace / "individuals_complete.json", {"status": "done"})
    by_directory = {"hourly_baselines": ["persistence", "linear_quantile"], "hourly_neural": list(NEURAL_METHODS),
        "hourly_trees": ["hist_boost_quantile", "quantile_forest", "hist_boost_group_features"],
        "hourly_architecture_trials": list(TRIAL_METHODS), "hourly_residual_control": ["own_summary_mlp_residual"]}
    for directory, methods in by_directory.items():
        out = workspace / "outputs" / directory
        out.mkdir(parents=True)
        rows = []
        for method in methods:
            error = config["families"].index(method) / 10
            rows.append({"split": "dev", "method": method, "episode_id": "e", "origin_hour": 1,
                "horizon_hours": 1, "variable": "HR", "observed": 1., "target_scale": 1.,
                "q10": 1 + error - .2, "q50": 1 + error, "q90": 1 + error + .2})
            (out / f"{method}.pt").write_bytes(b"dummy artifact for hash test")
        pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
        manifest = {"selection": {method: {"model_directory": method} for method in methods}}
        write_json(out / "fit_manifest.json", manifest)
        if directory == "hourly_trees":
            for method in methods:
                (out / method).mkdir()
                (out / method / "cell.joblib").write_bytes(b"dummy")
    result = freeze_components(workspace)
    assert result["methods"][0] == "linear_quantile"
    assert len(result["methods"]) == 4
    families = [config["individual_selection"]["diversity_families"][m] for m in result["methods"]]
    assert len(set(families)) == len(families)
    assert not result["meta_used"] and not result["calibration_used"] and not result["final_used"]
    assert result["best_individual_dev"] == "persistence"
