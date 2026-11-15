"""One-shot inference after the canonical development selection is frozen.

Never call a fitting routine. Imports and tests do not open confirmation values.
Actual invocation opens the reserved data, reuses frozen scalers/models, exports
only the frozen shortlist, and records source/model hashes before inference.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TREE_METHODS = {"hist_boost_quantile", "quantile_forest", "hist_boost_group_features"}
TRIAL_METHODS = {"causal_residual", "window_bidirectional_residual", "group_temporal_residual"}
CONTROL_METHODS = {"own_summary_mlp_residual"}
ALLOWED_METHODS = {"linear_quantile"} | TREE_METHODS | TRIAL_METHODS | CONTROL_METHODS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def guard_selection(root: Path = ROOT) -> tuple[dict, Path]:
    """Require the canonical real freeze; reject reuse before touching raw data."""
    campaign = root / "outputs/model_campaign"
    selection_path = campaign / "selection_frozen.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Canonical outputs/model_campaign/selection_frozen.json is required")
    if (campaign / "confirmation_run").exists() or list(campaign.glob("confirmation_*")):
        raise FileExistsError("Confirmation artifacts already exist; no automatic overwrite or repeat")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    methods = selection.get("selected_for_confirmation", [])
    if (selection.get("selection_split") != "dev" or
            selection.get("confirmation_used_for_selection") is not False or
            selection.get("calibration_used_for_selection") is not False or
            selection.get("ensemble_trained") is not False):
        raise ValueError("Selection is not an untouched-confirmation development-only freeze")
    if (not isinstance(methods, list) or not 2 <= len(methods) <= 3 or
            len(set(methods)) != len(methods) or "linear_quantile" not in methods or
            not set(methods).issubset(ALLOWED_METHODS)):
        raise ValueError("Freeze must contain linear reference and one/two supported new methods")
    evidence = selection.get("evidence_sha256", {})
    if not evidence:
        raise ValueError("Frozen selection must bind candidate evidence hashes")
    for name, expected in evidence.items():
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path) != expected:
            raise ValueError("Frozen selection evidence changed")
    return selection, selection_path


def artifact_inventory(root: Path, methods: list[str]) -> list[Path]:
    baseline = root / "outputs/hourly_baselines"
    paths = [baseline / name for name in ("fit_manifest.json", "linear_quantile.npz", "persistence.npz", "scalers.npz")]
    for method in methods:
        if method in TREE_METHODS:
            base = root / "outputs/hourly_trees"
            manifest = json.loads((base / "fit_manifest.json").read_text(encoding="utf-8"))
            directory = (base / manifest["selection"][method]["model_directory"]).resolve()
            if not directory.is_relative_to(base.resolve()):
                raise ValueError("Unsafe tree model directory")
            paths.extend([base / "fit_manifest.json", base / "scalers.npz"])
            paths.extend(directory / f"h{h}_v{v}.joblib" for h in range(4) for v in range(34))
        elif method in TRIAL_METHODS:
            base = root / "outputs/hourly_architecture_trials"
            manifest = json.loads((base / "fit_manifest.json").read_text(encoding="utf-8"))
            if method not in manifest.get("models", {}) or not manifest.get("models_frozen_before_export"):
                raise ValueError("Selected architecture is not frozen")
            paths.extend([base / "fit_manifest.json", base / "scalers.npz", base / f"{method}.pt"])
        elif method in CONTROL_METHODS:
            base = root / "outputs/hourly_residual_control"
            paths.extend([base / "fit_manifest.json", base / f"{method}.pt"])
    result = sorted(set(paths))
    if any(not path.is_file() for path in result):
        raise FileNotFoundError("Missing frozen model artifact")
    return result


def verify_acquisition(root: Path) -> tuple[dict, str]:
    """Hash raw bytes after selection guard; do not fit/filter by their outcomes."""
    path = root / "data/manifests/confirmation_holdout.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_structural_validation_only" or manifest.get("selected_episodes") != 300:
        raise ValueError("Confirmation acquisition is incomplete or changed")
    if (manifest.get("inspection", {}).get("models_evaluated") is not False or
            manifest.get("inspection", {}).get("outcomes_or_values_inspected") is not False):
        raise ValueError("Confirmation acquisition does not record an untouched model holdout")
    raw = (root / "data/raw/external/PHYSIONET_CONFIRMATION").resolve()
    expected = set()
    episodes = []
    for record in manifest["files"]:
        file = (root / record["path"]).resolve()
        if not file.is_relative_to(raw):
            raise ValueError("Confirmation file outside isolated raw directory")
        expected.add(file)
        if file.stat().st_size != record["bytes"] or file_sha256(file) != record["sha256"]:
            raise ValueError("Confirmation raw file differs from acquisition hash")
        if record["kind"] == "episode":
            episodes.append(file)
    if {p.resolve() for p in raw.rglob("*") if p.is_file()} != expected:
        raise ValueError("Unexpected files in isolated confirmation inventory")
    existing = {p.name for p in (root / "data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0").rglob("*.psv")}
    names = [p.name for p in episodes]
    if (len(names) != 300 or len(names) != len(set(names)) or set(names) & existing or
            any(sum(p.parent.name == site for p in episodes) != 150 for site in ("training_setA", "training_setB"))):
        raise ValueError("Confirmation identities overlap original development inventory")
    frozen_names = {site: sorted(p.name for p in episodes if p.parent.name == site)
                    for site in ("training_setA", "training_setB")}
    if hashlib.sha256(json.dumps(frozen_names, sort_keys=True).encode()).hexdigest() != manifest["selection_sha256"]:
        raise ValueError("Acquisition selection identities differ from frozen hash")
    for index in manifest["source_indexes"].values():
        if file_sha256(root / index["path"]) != index["sha256"]:
            raise ValueError("Acquisition source index differs from frozen hash")
    return manifest, file_sha256(path)


def mark_confirmation(packed: dict) -> dict:
    """Override every episode split before any preparation; no train set exists."""
    result = dict(packed)
    result["episode_split"] = np.full(len(packed["episode_ids"]), "confirmation", dtype="U12")
    return result


def pack_confirmation(raw: Path, run: Path) -> tuple[dict, dict]:
    from astronaut_twin.hourly_data import build_packed
    summary = build_packed(raw, run / "builder_temporary.npz", run / "builder_temporary_splits.json")
    packed = mark_confirmation(load_npz(run / "builder_temporary.npz"))
    # The builder's temporary balancing labels are never passed to preparation.
    np.savez_compressed(run / "confirmation_packed.npz", **packed)
    (run / "confirmation_splits.json").write_text(json.dumps(
        {str(name): "confirmation" for name in packed["episode_ids"]}, indent=2), encoding="utf-8")
    return packed, summary


def verify_frozen_scalers(packed: dict, scalers: dict) -> None:
    variables = packed["variables"].astype(str)
    if not np.array_equal(variables, scalers["variables"].astype(str)):
        raise ValueError("Frozen scaler variable order differs")
    for name in ("mean", "scale", "median"):
        if np.asarray(scalers[name]).shape != (len(variables),) or not np.isfinite(scalers[name]).all():
            raise ValueError("Invalid frozen scaler metadata")
    if not np.all(scalers["scale"] > 0) or not np.all(packed["episode_split"] == "confirmation"):
        raise ValueError("Need positive frozen scales and confirmation-only episodes")


def prediction_table(method, prediction, packed, baseline, train_counts, metadata):
    import pandas as pd
    if prediction.shape != baseline.labels.shape + (3,):
        raise ValueError("Incorrect confirmation prediction shape")
    local, h, v = np.nonzero(np.isfinite(baseline.labels))
    selected = prediction[local, h, v]
    if not np.isfinite(selected).all() or not np.all(np.diff(selected, axis=-1) >= 0):
        raise ValueError("Frozen predictions are nonfinite or crossed; do not repair on confirmation")
    episode = baseline.episode[local]
    origins = packed["origins"][local]
    variables = packed["variables"].astype(str)
    vm = {entry["source_key"]: entry for entry in metadata["variables"]}
    support = np.asarray(train_counts)[h, v]
    return pd.DataFrame({"split": "confirmation", "method": method,
        "episode_id": packed["episode_ids"][episode], "site": packed["sites"][episode],
        "origin_hour": packed["times"][origins], "target_hour": packed["times"][origins] + packed["horizons"][h],
        "horizon_hours": packed["horizons"][h], "head": [vm[name]["group"] for name in variables[v]],
        "variable": variables[v], "unit": [vm[name]["source_unit_as_documented"] for name in variables[v]],
        "unit_review_required": [vm[name].get("unit_review_required", False) for name in variables[v]],
        "observed": baseline.labels[local, h, v], "q10": selected[:, 0], "q50": selected[:, 1], "q90": selected[:, 2],
        "history_hours": baseline.history_hours[local], "history_length": baseline.history_length[local],
        "own_history_observed": baseline.counts[local, v],
        "used_train_median_fallback": baseline.counts[local, v] == 0,
        "target_scale": baseline.scale[v], "train_target_count": support,
        "prediction_status": np.where(support == 0, "no_train_labels_fallback_or_untrained",
                                       np.where(support < 20, "low_train_support", "fitted"))})


def evaluate_confirmation(root: Path = ROOT, batch_size: int = 256) -> dict:
    selection, selection_path = guard_selection(root)
    methods = selection["selected_for_confirmation"]
    artifacts = artifact_inventory(root, methods)
    artifact_hashes = {str(p.relative_to(root)): file_sha256(p) for p in artifacts}
    selection_hash = file_sha256(selection_path)
    config_path = root / "config/hourly_task_candidates.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_manifest = json.loads((root / "outputs/hourly_baselines/fit_manifest.json").read_text(encoding="utf-8"))
    if baseline_manifest["task_config_sha256"] != file_sha256(config_path):
        raise ValueError("Task contract differs from frozen reference")
    quality_hash = file_sha256(root / "config/hourly_quality_policy.json")
    if baseline_manifest.get("quality_policy_sha256_at_export", quality_hash) != quality_hash:
        raise ValueError("Quality policy differs from frozen reference")
    for parent in {p.parent for p in artifacts if p.name == "fit_manifest.json"}:
        manifest = json.loads((parent / "fit_manifest.json").read_text(encoding="utf-8"))
        dataset_hash = manifest.get("dataset_sha256", manifest.get("data_sha256", manifest.get("source_sha256", {}).get("dataset")))
        if dataset_hash != baseline_manifest["data_sha256"]:
            raise ValueError("Frozen candidates were not trained on the reference dataset")
        if manifest.get("task_config_sha256", file_sha256(config_path)) != file_sha256(config_path):
            raise ValueError("Candidate task differs from reference")
    acquisition, acquisition_hash = verify_acquisition(root)
    campaign = root / "outputs/model_campaign"
    run = campaign / "confirmation_run"
    run.mkdir(exist_ok=False)
    provenance_path = campaign / "confirmation_run_manifest.json"
    provenance = {"status": "frozen_confirmation_started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_methods": methods, "selection_sha256": selection_hash, "artifact_sha256_before_inference": artifact_hashes,
        "acquisition_manifest_sha256": acquisition_hash, "acquisition_selection_sha256": acquisition["selection_sha256"],
        "task_config_sha256": file_sha256(config_path), "quality_policy_sha256": file_sha256(root / "config/hourly_quality_policy.json"),
        "evaluator_sha256": file_sha256(Path(__file__)), "no_refit": True, "no_selection_or_tuning_on_confirmation": True,
        "calibration_applied": False, "ensemble_trained": False, "holdout_values_opened_after_selection_freeze": True}
    inference_sources = [root / name for name in (
        "src/astronaut_twin/hourly_data.py", "src/astronaut_twin/hourly_quality.py",
        "src/astronaut_twin/hourly_baselines.py", "src/astronaut_twin/hourly_architecture_trials.py",
        "src/astronaut_twin/hourly_neural.py", "src/astronaut_twin/hourly_tree_models.py",
        "src/astronaut_twin/hourly_residual_control.py", "src/astronaut_twin/models/twin.py",
        "scripts/report_model_campaign.py", "config/hourly_quality_policy.json", "config/hourly_task_candidates.json",
        "config/model_campaign.json")]
    source_hashes = {str(p.relative_to(root)): file_sha256(p) for p in inference_sources if p.exists()}
    provenance["inference_source_sha256"] = source_hashes
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    try:
        import torch
        from astronaut_twin.hourly_architecture_trials import prepare_frozen_trial_data, load_trial_checkpoint
        from astronaut_twin.hourly_baselines import predict_linear
        torch.set_num_threads(2)
        packed, packing = pack_confirmation(root / "data/raw/external/PHYSIONET_CONFIRMATION", run)
        base = root / "outputs/hourly_baselines"
        scalers = load_npz(base / "scalers.npz")
        verify_frozen_scalers(packed, scalers)
        neural, baseline = prepare_frozen_trial_data(packed, scalers)
        if not np.all(neural.split == "confirmation") or not np.all(baseline.split == "confirmation"):
            raise ValueError("Frozen preparation introduced a nonconfirmation split")
        linear = load_npz(base / "linear_quantile.npz")
        bands = load_npz(base / "persistence.npz")
        reference_prediction = predict_linear(baseline, linear, bands)
        base_z = torch.from_numpy((reference_prediction - baseline.mean[None, None, :, None]) / baseline.scale[None, None, :, None])
        own = torch.from_numpy(baseline.features)
        destination = campaign / "confirmation_predictions.csv"
        counts = {}
        for method in methods:
            train_counts = linear["train_target_counts"]
            if method == "linear_quantile":
                prediction = reference_prediction
            elif method in TREE_METHODS:
                from astronaut_twin.hourly_tree_models import load_tree_family, predict_tree_family
                artifact = load_tree_family(root / "outputs/hourly_trees", method)
                if (list(map(str, artifact["manifest"]["variables"])) != list(neural.variables) or
                        list(artifact["manifest"]["horizons_hours"]) != packed["horizons"].tolist()):
                    raise ValueError("Tree variable/horizon contract differs from confirmation")
                for name in ("mean", "scale", "median"):
                    if not np.array_equal(artifact["scalers"][name], scalers[name]):
                        raise ValueError("Tree scalers differ from frozen reference")
                prediction = predict_tree_family(artifact, baseline.features)
                train_counts = np.asarray(artifact["manifest"]["train_target_counts"])
            else:
                if method in CONTROL_METHODS:
                    from astronaut_twin.hourly_residual_control import load_summary_checkpoint
                    model, state = load_summary_checkpoint(root / "outputs/hourly_residual_control" / f"{method}.pt")
                else:
                    model, state = load_trial_checkpoint(root / "outputs/hourly_architecture_trials" / f"{method}.pt")
                if list(state["variables"]) != list(neural.variables):
                    raise ValueError("Architecture variable order differs")
                if "horizons" in state and list(state["horizons"]) != packed["horizons"].tolist():
                    raise ValueError("Control horizon contract differs")
                for name in ("mean", "scale"):
                    if not np.array_equal(np.asarray(state[f"scaler_{name}"]), scalers[name]):
                        raise ValueError("Architecture scalers differ from frozen reference")
                parts = []
                with torch.no_grad():
                    for start in range(0, len(baseline.features), batch_size):
                        rows = np.arange(start, min(start + batch_size, len(baseline.features)))
                        args, _, _ = neural.batch(rows, state.get("groups", metadata["output_groups"]))
                        value = model(args, own[rows], base_z[rows]).numpy()
                        parts.append(value * baseline.scale[None, None, :, None] + baseline.mean[None, None, :, None])
                prediction = np.concatenate(parts)
            table = prediction_table(method, prediction, packed, baseline, train_counts, metadata)
            if not counts:
                with destination.open("x", encoding="utf-8", newline="") as target:
                    table.to_csv(target, index=False)
            else:
                table.to_csv(destination, mode="a", header=False, index=False)
            counts[method] = len(table)
        if (file_sha256(Path(__file__)) != provenance["evaluator_sha256"] or
                file_sha256(selection_path) != selection_hash or
                any(file_sha256(root / name) != digest for name, digest in artifact_hashes.items())):
            raise ValueError("Selection or model artifacts changed during inference")
        if any(file_sha256(root / name) != digest for name, digest in source_hashes.items()):
            raise ValueError("Inference sources changed during confirmation")
        _, final_acquisition_hash = verify_acquisition(root)
        if final_acquisition_hash != acquisition_hash:
            raise ValueError("Confirmation acquisition changed during inference")
        spec = importlib.util.spec_from_file_location("confirmation_campaign_report", root / "scripts/report_model_campaign.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        report.summarize_campaign([destination], "confirmation", campaign)
        provenance.update(status="frozen_confirmation_complete", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                          prediction_rows_by_method=counts, predictions_sha256=file_sha256(destination),
                          packed_sha256=file_sha256(run / "confirmation_packed.npz"),
                          episodes=len(packed["episode_ids"]), origins=len(packed["origins"]),
                          artifact_hashes_unchanged=True, acquisition_hash_unchanged=True)
    except Exception as exc:
        provenance.update(status="failed_no_automatic_retry", error_type=type(exc).__name__)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        raise
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    result = evaluate_confirmation()
    print(json.dumps({key: result[key] for key in ("status", "selected_methods", "prediction_rows_by_method", "selection_sha256")}))
