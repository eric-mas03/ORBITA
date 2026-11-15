"""Isolated V2 training, held-out stacking and one frozen final evaluation.

V1 files are imported from a campaign snapshot, never rewritten. Development
training views exclude meta/calibration; final episodes are not packed initially.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS = {"baselines": "hourly_baselines", "neural": "hourly_neural", "trees": "hourly_trees",
        "architectures": "hourly_architecture_trials", "summary": "hourly_residual_control"}
NEURAL_METHODS = ("gru_shared_readout", "transformer_grouped")
TRIAL_METHODS = ("causal_residual", "window_bidirectional_residual", "group_temporal_residual")
TREE_METHODS = ("hist_boost_quantile", "quantile_forest", "hist_boost_group_features")
ARCH_WORKERS = {"architecture_causal": "causal_residual", "architecture_bidir": "window_bidirectional_residual",
                "architecture_group": "group_temporal_residual"}
KEYS = ["episode_id", "origin_hour", "horizon_hours", "variable"]
TRAINING_SOURCES = ("src/astronaut_twin/campaign_v2.py", "src/astronaut_twin/hourly_baselines.py",
    "src/astronaut_twin/hourly_neural.py", "src/astronaut_twin/hourly_architecture_trials.py",
    "src/astronaut_twin/hourly_residual_control.py", "src/astronaut_twin/hourly_tree_models.py",
    "src/astronaut_twin/hourly_quality.py", "src/astronaut_twin/models/twin.py",
    "scripts/run_model_campaign_v2.py", "scripts/train_hourly_trees.py")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def view_roles(packed, roles):
    result = dict(packed)
    episode = np.searchsorted(packed["episode_offsets"], packed["origins"], side="right") - 1
    keep = np.isin(packed["episode_split"][episode], roles)
    for name in ("origins", "history_start", "target_rows", "sample_episode"):
        if name in result:
            result[name] = result[name][keep]
    return result


def expanded_epoch_budget(v1_epochs, v1_origins, current_origins):
    if current_origins < 1:
        raise ValueError("Training origins required")
    return max(1, math.ceil(v1_epochs * v1_origins / current_origins))


def prepare_campaign(root, workspace, config):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    dataset = Path(config["dataset_path"])
    audit_path = Path(config["audit_path"])
    dataset = dataset if dataset.is_absolute() else root / dataset
    audit_path = audit_path if audit_path.is_absolute() else root / audit_path
    audit = read_json(audit_path)
    if audit.get("status") != "audited_nonfinal_v2" or audit.get("dataset_sha256") != sha(dataset):
        raise ValueError("An audited matching nonfinal V2 dataset is required")
    if audit.get("final_not_packed") is not True:
        raise ValueError("The final reserve must remain outside the nonfinal package")
    expansion = root / "data/manifests/hourly_expansion_v2.json"
    roles_path = root / "data/processed/hourly_v2/roles.json"
    if not expansion.exists() or not roles_path.exists() or audit.get("source_manifest_sha256") != sha(expansion):
        raise ValueError("Audited acquisition manifest and fixed role mapping must match before preparing models")
    packed = load_npz(dataset)
    roles = set(map(str, packed["episode_split"]))
    if roles != set(config["required_nonfinal_roles"]):
        raise ValueError(f"Unexpected nonfinal roles: {roles}")
    if len(set(packed["episode_ids"])) != len(packed["episode_ids"]):
        raise ValueError("Episode IDs must be unique across roles")
    if set(map(str, audit.get("final_episode_ids", []))) & set(map(str, packed["episode_ids"])):
        raise ValueError("Final IDs leaked into the nonfinal package")
    if config["source_resolution_hours"] != 1 or config["lookback_hours"] != 24:
        raise ValueError("This implementation supports 24 real hourly slots; true 15-minute data require a new preparer")
    if len(packed["variables"]) != 34 or list(packed["horizons"]) != config["horizons_hours"]:
        raise ValueError("Frozen family interfaces require 34 variables and the four declared horizons")
    training_view = view_roles(packed, ("train", "dev"))
    sample_ep = np.searchsorted(packed["episode_offsets"], training_view["origins"], side="right") - 1
    n_train = int((packed["episode_split"][sample_ep] == "train").sum())
    config = copy.deepcopy(config)
    config["dataset_path"], config["audit_path"] = str(dataset.resolve()), str(audit_path.resolve())
    config["source_project_root"] = str(root)
    config["project_root"] = str(root)
    config["expansion_manifest_path"] = str(root / "data/manifests/hourly_expansion_v2.json")
    config["roles_path"] = str(root / "data/processed/hourly_v2/roles.json")
    config["original_tree_dependency_path"] = str(root / ".deps/tree_models")
    cfg = config["training"]
    config["effective_epochs"] = {key: expanded_epoch_budget(cfg[f"{key}_v1_max_epochs"], cfg["v1_train_origins"], n_train)
                                  for key in ("neural", "residual", "mlp")}
    signature = {"dataset_sha256": sha(dataset), "audit_sha256": sha(audit_path), "train_origins": n_train,
                 "source_root": str(root),
                 "expansion_manifest_sha256": sha(expansion), "roles_json_sha256": sha(roles_path),
                 "training_source_sha256": {name: sha(root / name) for name in TRAINING_SOURCES if (root / name).exists()},
                 "config": config}
    if (workspace / "prepared.json").exists():
        if read_json(workspace / "prepared.json") != signature:
            raise ValueError("Existing campaign differs; use a new workspace instead of overwriting models")
        return {"status": "already_prepared", "workspace": str(workspace)}
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("New campaign workspace must be empty")
    workspace.mkdir(parents=True, exist_ok=True)
    for folder in ("src", "scripts", "config"):
        shutil.copytree(root / folder, workspace / folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(workspace / "campaign_config.json", config)
    # Keep the V1 task schema as provenance; remove its old-cohort count evidence.
    task_path = workspace / "config/hourly_task_candidates.json"
    task = read_json(task_path)
    shutil.copyfile(task_path, task_path.with_name("hourly_task_candidates.source_v1.json"))
    task.pop("evidence", None)
    for variable in task.get("variables", []):
        variable.pop("observed_support", None)
        variable.pop("eligibility_status", None)
    task["status"] = "v2_expanded_dataset_same_variable_group_semantics"
    task["primary_hourly_domain"]["dataset"] = "PhysioNet Challenge2019 audited V2 expansion"
    write_json(task_path, task)
    directory = workspace / "data/processed/hourly"
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "hourly_dataset.npz", **training_view)
    shutil.copyfile(dataset, workspace / "nonfinal_dataset.npz")
    shutil.copyfile(audit_path, workspace / "dataset_audit.json")
    write_json(workspace / "prepared.json", signature)
    return {"status": "prepared_without_training", "workspace": str(workspace),
            "train_origins": n_train, "effective_epochs": config["effective_epochs"]}


def _snapshot_module(workspace, name):
    path = Path(workspace) / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"v2_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_data(workspace):
    from .hourly_baselines import prepare_baseline_data
    from .hourly_architecture_trials import prepare_frozen_trial_data
    packed = load_npz(Path(workspace) / "data/processed/hourly/hourly_dataset.npz")
    base = Path(workspace) / "outputs/hourly_baselines"
    if (base / "scalers.npz").exists():
        neural, baseline = prepare_frozen_trial_data(packed, load_npz(base / "scalers.npz"))
    else:
        baseline = prepare_baseline_data(packed)
        neural = None
    return packed, baseline, neural


def export_predictions(workspace, output, method, predictions, packed, baseline,
                       rows=None, split="dev", median_fallback=True):
    if rows is None:
        rows = np.flatnonzero(baseline.split == split)
    metadata = read_json(Path(workspace) / "config/hourly_task_candidates.json")
    vm = {v["source_key"]: v for v in metadata["variables"]}
    variables = np.asarray(packed["variables"]).astype(str)
    local, h, v = np.nonzero(np.isfinite(baseline.labels[rows]))
    selected = rows[local]
    ep, origin = baseline.episode[selected], packed["origins"][selected]
    q = predictions[local, h, v]
    if not np.isfinite(q).all() or (np.diff(q, axis=-1) < -1e-5).any():
        raise ValueError("Export requires finite ordered quantiles for actual labels")
    table = pd.DataFrame({"split": split, "method": method, "episode_id": packed["episode_ids"][ep],
        "site": packed["sites"][ep], "origin_hour": packed["times"][origin],
        "target_hour": packed["times"][origin] + packed["horizons"][h], "horizon_hours": packed["horizons"][h],
        "head": [vm[x]["group"] for x in variables[v]], "variable": variables[v],
        "unit": [vm[x]["source_unit_as_documented"] for x in variables[v]],
        "unit_review_required": np.isin(variables[v], ["FiO2", "Magnesium", "Lactate"]),
        "observed": baseline.labels[selected, h, v], "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
        "target_scale": baseline.scale[v], "history_hours": baseline.history_hours[selected],
        "history_length": baseline.history_length[selected], "own_history_observed": baseline.counts[selected, v],
        "used_train_median_fallback": (baseline.counts[selected, v] == 0) if median_fallback else False})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, mode="a" if output.exists() else "w", header=not output.exists(), index=False)
    return table


def _fit_baselines(workspace, config, out):
    from .hourly_baselines import (FEATURE_NAMES, fit_linear_quantiles, fit_persistence_bands,
                                   predict_linear, predict_persistence)
    packed, baseline, _ = _training_data(workspace)
    np.savez_compressed(out / "scalers.npz", mean=baseline.mean, scale=baseline.scale, median=baseline.median,
        observed_counts=baseline.scaler_counts, train_row_indices=baseline.train_row_indices,
        feature_names=np.asarray(FEATURE_NAMES), variables=packed["variables"])
    bands = fit_persistence_bands(baseline)
    np.savez_compressed(out / "persistence.npz", **bands)
    linear = fit_linear_quantiles(baseline, epochs=config["training"]["linear_epochs"], seed=config["training"]["seed"])
    np.savez_compressed(out / "linear_quantile.npz", weight=linear["weight"], bias=linear["bias"],
                        train_target_counts=linear["train_target_counts"], feature_names=np.asarray(FEATURE_NAMES))
    dev = np.flatnonzero(baseline.split == "dev")
    for method, prediction in (("persistence", predict_persistence(baseline, bands, dev)),
                               ("linear_quantile", predict_linear(baseline, linear, bands, dev))):
        export_predictions(workspace, out / "predictions.csv", method, prediction, packed, baseline, dev)
    pd.DataFrame(linear["curves"]).to_csv(out / "learning_curves.csv", index=False)
    return {"selection": linear["selection"], "train_target_counts": linear["train_target_counts"].tolist(),
            "calibration_test_meta_used": False, "export_splits": ["dev"]}


def _new_model(method, groups, variables, seed):
    from .hourly_neural import make_hourly_twin
    from .hourly_architecture_trials import ResidualHourlyTrial, TrialConfig
    from .hourly_residual_control import SummaryResidualMLP
    if method in NEURAL_METHODS:
        return make_hourly_twin(groups, method, seed)
    if method in TRIAL_METHODS:
        return ResidualHourlyTrial(groups, variables, TrialConfig(kind=method), seed)
    if method == "own_summary_mlp_residual":
        return SummaryResidualMLP(variables, seed=seed)
    raise ValueError("Unknown neural method")


def _fit_neural_family(workspace, config, out, methods):
    from .hourly_neural import flatten_predictions, global_cell_loss
    from .hourly_baselines import predict_linear
    packed, baseline, neural = _training_data(workspace)
    groups = read_json(Path(workspace) / "config/hourly_task_candidates.json")["output_groups"]
    train, dev = (np.flatnonzero(neural.split == role) for role in ("train", "dev"))
    base = Path(workspace) / "outputs/hourly_baselines"
    if not (out / "scalers.npz").exists():
        shutil.copyfile(base / "scalers.npz", out / "scalers.npz")
    frozen = predict_linear(baseline, load_npz(base / "linear_quantile.npz"), load_npz(base / "persistence.npz"))
    base_z = torch.from_numpy((frozen - neural.mean[None, None, :, None]) / neural.scale[None, None, :, None])
    own = torch.from_numpy(baseline.features)
    cfg, batch_size = config["training"], config["training"]["batch_size"]
    counts = torch.isfinite(neural.labels[train]).sum(0)
    group_columns = {g: [neural.variables.index(v.removeprefix("SEPSIS2019::")) for v in vs]
                     for g, vs in groups.items()}
    manifest = {"models": {}, "export_splits": ["dev"], "calibration_test_meta_used": False}
    for method in methods:
        residual = method not in NEURAL_METHODS
        category = "neural" if not residual else "mlp" if method == "own_summary_mlp_residual" else "residual"
        epochs = config["effective_epochs"][category]
        model = _new_model(method, groups, neural.variables, cfg["seed"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg[f"{category}_lr"], weight_decay=cfg["weight_decay"])
        rng = np.random.default_rng(cfg["seed"])
        best, epoch_best, stale, best_state, start_epoch, curves = float("inf"), 0, 0, None, 1, []
        signature = sha(Path(workspace) / "prepared.json")
        def forward(rows):
            args, target, mask = neural.batch(rows, groups)
            pred = model(args, own[rows], base_z[rows]) if residual else flatten_predictions(model(*args), groups, neural.variables)
            return pred, target, mask
        def evaluate(rows):
            sums, nums = torch.zeros_like(counts, dtype=torch.float32), torch.zeros_like(counts)
            parts = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(rows), batch_size):
                    subset = rows[i:i + batch_size]
                    pred, target, mask = forward(subset)
                    err = target[..., None] - pred
                    q = pred.new_tensor([.1, .5, .9])
                    sums += (torch.maximum(q * err, (q - 1) * err).mean(-1) * mask).sum(0)
                    nums += mask.sum(0)
                    parts.append(pred.numpy() * neural.scale[None, None, :, None] + neural.mean[None, None, :, None])
            active = nums > 0
            return float((sums[active] / nums[active]).mean()), np.concatenate(parts)
        resume = out / f"{method}_resume.pt"
        if resume.exists():
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved["signature"] != signature:
                raise ValueError("Resume data/configuration signature changed")
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            best, epoch_best, stale, best_state = saved["best"], saved["epoch_best"], saved["stale"], saved["best_state"]
            start_epoch, curves = saved["epoch"] + 1, saved["curves"]
            rng.bit_generator.state = saved["numpy_rng"]
            torch.set_rng_state(saved["torch_rng"])
        elif residual:
            best, _ = evaluate(dev)
            best_state = copy.deepcopy(model.state_dict())
            curves.append({"epoch": 0, "dev_macro_pinball": best})
        method_started = time.monotonic()
        for epoch in range(start_epoch, epochs + 1):
            if stale >= cfg[f"{category}_patience"]:
                break
            model.train()
            tick = time.monotonic()
            for i in range(0, len(train), batch_size):
                if i == 0:
                    order = rng.permutation(train)
                rows = order[i:i + batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction, target, mask = forward(rows)
                loss = global_cell_loss(prediction, target, mask, counts, len(train))
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training objective")
                loss.backward()
                skip_unlabelled_heads(model, method, mask, group_columns)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
                optimizer.step()
            score, _ = evaluate(dev)
            curves.append({"epoch": epoch, "dev_macro_pinball": score, "epoch_seconds": time.monotonic() - tick})
            print(f"{method} epoch={epoch}/{epochs} dev_macro={score:.9f} seconds={curves[-1]['epoch_seconds']:.1f}", flush=True)
            if score < best - 1e-6:
                best, epoch_best, stale, best_state = score, epoch, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            payload = {"signature": signature, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "best": best, "epoch_best": epoch_best, "stale": stale, "best_state": best_state,
                "epoch": epoch, "curves": curves, "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}
            temporary = resume.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(resume)
            pd.DataFrame(curves).to_csv(out / f"{method}_learning_curves.csv", index=False)
        model.load_state_dict(best_state)
        score, predictions = evaluate(dev)
        state = {"method": method, "state_dict": best_state, "groups": groups, "variables": neural.variables,
                 "seed": cfg["seed"], "selected_epoch": epoch_best, "dev_macro_pinball": best,
                 "signature": signature, "maximum_epochs": epochs,
                 "scaler_mean": torch.from_numpy(neural.mean), "scaler_scale": torch.from_numpy(neural.scale)}
        if method in TRIAL_METHODS:
            state["config"] = asdict(model.config)
        elif method in NEURAL_METHODS:
            state["config"] = model.config.__dict__
        else:
            state.update(horizons=tuple(packed["horizons"].tolist()), hidden=model.hidden)
        torch.save(state, out / f"{method}.pt")
        export_predictions(workspace, out / "predictions.csv", method, predictions, packed, baseline, dev,
                           median_fallback=residual)
        manifest["models"][method] = {"selected_epoch": epoch_best, "maximum_epochs": epochs,
            "dev_macro_pinball": best, "parameter_count": sum(p.numel() for p in model.parameters()),
            "run_elapsed_seconds": round(time.monotonic() - method_started, 2)}
        write_json(out / "fit_manifest.json", manifest)
    manifest["train_target_counts"] = counts.tolist()
    manifest["models_frozen_before_export"] = True
    return manifest


def skip_unlabelled_heads(model, method, mask, group_columns):
    """Preserve V1 AdamW behavior for exclusive heads absent in a minibatch."""
    if method == "own_summary_mlp_residual":
        return
    core = model if method in NEURAL_METHODS else model.core
    for group, columns in group_columns.items():
        if not mask[:, :, columns].any():
            for parameter in list(core.decoders[group].parameters()) + list(core.output_layers[group].parameters()):
                parameter.grad = None
            if method == "group_temporal_residual":
                for parameter in model.group_memories[group].parameters():
                    parameter.grad = None


def run_worker(workspace, worker):
    workspace = Path(workspace).resolve()
    config = read_json(workspace / "campaign_config.json")
    for name, expected in read_json(workspace / "prepared.json")["training_source_sha256"].items():
        if sha(workspace / name) != expected:
            raise ValueError(f"Frozen training source changed: {name}")
    torch.set_num_threads(config["training"]["threads_per_worker"])
    out = workspace / "outputs" / (f"_architecture_parts/{ARCH_WORKERS[worker]}" if worker in ARCH_WORKERS else JOBS[worker])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return read_json(out / "complete.json")
    # Resumed exports are rebuilt from each selected frozen state; never append duplicates.
    if worker != "trees" and (out / "predictions.csv").exists():
        (out / "predictions.csv").unlink()
    started = time.monotonic()
    if worker == "baselines":
        result = _fit_baselines(workspace, config, out)
    elif worker in {"neural", "architectures", "summary"} or worker in ARCH_WORKERS:
        methods = ((ARCH_WORKERS[worker],) if worker in ARCH_WORKERS else NEURAL_METHODS if worker == "neural"
                   else TRIAL_METHODS if worker == "architectures" else ("own_summary_mlp_residual",))
        result = _fit_neural_family(workspace, config, out, methods)
    elif worker == "trees":
        module = _snapshot_module(workspace, "train_hourly_trees")
        old_args = sys.argv
        sys.argv = [str(workspace / "scripts/train_hourly_trees.py"), "--out", str(out)]
        try:
            module.main()
        finally:
            sys.argv = old_args
        result = read_json(out / "fit_manifest.json")
    else:
        raise ValueError("Unknown V2 worker")
    result.update(v2=True, worker=worker, elapsed_seconds=round(time.monotonic() - started, 2),
                  prepared_signature=sha(workspace / "prepared.json"), predictions_sha256=sha(out / "predictions.csv"),
                  data_sha256=sha(workspace / "data/processed/hourly/hourly_dataset.npz"),
                  dataset_sha256=sha(workspace / "data/processed/hourly/hourly_dataset.npz"),
                  task_config_sha256=sha(workspace / "config/hourly_task_candidates.json"),
                  quality_policy_sha256_at_export=sha(workspace / "config/hourly_quality_policy.json"))
    write_json(out / "fit_manifest.json", result)
    write_json(out / "complete.json", result)
    return result


def run_individuals(workspace):
    workspace = Path(workspace).resolve()
    config = read_json(workspace / "campaign_config.json")
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    def launch(worker):
        out = workspace / "outputs" / (f"_architecture_parts/{ARCH_WORKERS[worker]}" if worker in ARCH_WORKERS else JOBS[worker])
        if (out / "complete.json").exists():
            return worker, "already_complete"
        env = dict(os.environ)
        deps = config["original_tree_dependency_path"]
        env["PYTHONPATH"] = os.pathsep.join([str(workspace / "src"), deps, env.get("PYTHONPATH", "")])
        env["OMP_NUM_THREADS"], env["MKL_NUM_THREADS"] = "2", "2"
        command = [sys.executable, str(workspace / "scripts/run_model_campaign_v2.py"), "--workspace", str(workspace), "--worker", worker]
        with (logs / f"{worker}.log").open("a", encoding="utf-8") as output:
            process = subprocess.run(command, cwd=workspace, env=env, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if process.returncode:
            raise RuntimeError(f"Worker {worker} failed; see {logs / (worker + '.log')}")
        return worker, "complete"
    launch("baselines")
    # Longest worker first; each process has two threads and independent outputs.
    results = {}
    with ThreadPoolExecutor(max_workers=config["training"]["parallel_workers"]) as pool:
        futures = [pool.submit(launch, worker) for worker in ("architecture_group", "neural", "trees",
                                                            "architecture_causal", "architecture_bidir", "summary")]
        for future in as_completed(futures):
            worker, status = future.result()
            results[worker] = status
            write_json(workspace / "individual_progress.json", results)
    _merge_architecture_parts(workspace)
    result = {"status": "all_11_individual_families_complete", "workers": results,
              "calibration_meta_final_used": False, "prepared_signature": sha(workspace / "prepared.json")}
    write_json(workspace / "individuals_complete.json", result)
    return result


def _merge_architecture_parts(workspace):
    out = Path(workspace) / "outputs/hourly_architecture_trials"
    out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        return
    models, manifests = {}, []
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as destination:
        for i, method in enumerate(TRIAL_METHODS):
            part = Path(workspace) / "outputs/_architecture_parts" / method
            manifest = read_json(part / "complete.json")
            manifests.append(manifest)
            models.update(manifest["models"])
            shutil.copyfile(part / f"{method}.pt", out / f"{method}.pt")
            with (part / "predictions.csv").open(encoding="utf-8") as source:
                if i:
                    next(source)
                shutil.copyfileobj(source, destination)
    manifest = {**manifests[0], "models": models, "models_frozen_before_export": True,
                "parts": [str(m) for m in TRIAL_METHODS], "predictions_sha256": sha(out / "predictions.csv")}
    shutil.copyfile(Path(workspace) / "outputs/hourly_baselines/scalers.npz", out / "scalers.npz")
    write_json(out / "fit_manifest.json", manifest)
    write_json(out / "complete.json", manifest)


def freeze_components(workspace):
    workspace = Path(workspace).resolve()
    if (workspace / "components_frozen.json").exists():
        return read_json(workspace / "components_frozen.json")
    if not (workspace / "individuals_complete.json").exists():
        raise ValueError("All 11 individual families must finish before component selection")
    config = read_json(workspace / "campaign_config.json")
    scores, artifacts, tables = {}, {}, []
    for directory in JOBS.values():
        out = workspace / "outputs" / directory
        table = pd.read_csv(out / "predictions.csv")
        if set(table["split"]) != {"dev"}:
            raise ValueError("Component selection may use DEV predictions only")
        numeric = table[["q10", "q50", "q90", "observed", "target_scale"]].to_numpy()
        if not np.isfinite(numeric).all() or (numeric[:, -1] <= 0).any():
            raise ValueError("Component evidence must contain finite values and positive training scales")
        if (np.diff(numeric[:, :3], axis=-1) < -1e-6).any():
            raise ValueError("Component evidence contains crossed quantiles")
        error = table["observed"].to_numpy()[:, None] - table[["q10", "q50", "q90"]].to_numpy()
        q = np.asarray([.1, .5, .9])
        table["standardized_pinball"] = np.maximum(q * error, (q - 1) * error).mean(-1) / table["target_scale"]
        macro = table.groupby(["method", "variable", "horizon_hours"])["standardized_pinball"].mean().groupby("method").mean()
        scores.update({method: float(value) for method, value in macro.items()})
        tables.append(table[["method"] + KEYS + ["observed", "target_scale"]])
    if set(scores) != set(config["families"]):
        raise ValueError("The completed evidence does not contain all 11 families")
    combined = pd.concat(tables)
    reference = combined[combined.method == "linear_quantile"].set_index(KEYS).sort_index()
    reference_keys = set(reference.index)
    for method, group in combined.groupby("method"):
        if group.duplicated(KEYS).any() or set(map(tuple, group[KEYS].to_numpy())) != reference_keys:
            raise ValueError(f"Development labels differ for {method}")
        candidate = group.set_index(KEYS).sort_index()
        if not np.allclose(candidate[["observed", "target_scale"]], reference[["observed", "target_scale"]], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Observed targets or frozen scales differ for {method}")
    diversity = config["individual_selection"]["diversity_families"]
    selected, seen = ["linear_quantile"], {diversity["linear_quantile"]}
    for method in sorted(scores, key=scores.get):
        if method in {"persistence", "linear_quantile"} or diversity[method] in seen:
            continue
        selected.append(method)
        seen.add(diversity[method])
        if len(selected) == 1 + config["individual_selection"]["ensemble_new_components"]:
            break
    for directory in JOBS.values():
        out = workspace / "outputs" / directory
        # Bind all completed evidence; selected model files are separately explicit.
        for path in (out / "fit_manifest.json", out / "predictions.csv", out / "scalers.npz"):
            if path.exists():
                artifacts[str(path.relative_to(workspace))] = sha(path)
    for path in (workspace / "outputs/hourly_baselines").glob("*.npz"):
        artifacts[str(path.relative_to(workspace))] = sha(path)
    for method in selected:
        if method in TREE_METHODS:
            out = workspace / "outputs/hourly_trees"
            manifest = read_json(out / "fit_manifest.json")
            folder = out / manifest["selection"][method]["model_directory"]
            for path in folder.glob("*.joblib"):
                artifacts[str(path.relative_to(workspace))] = sha(path)
        elif method not in {"linear_quantile", "persistence"}:
            family = "hourly_neural" if method in NEURAL_METHODS else "hourly_architecture_trials" if method in TRIAL_METHODS else "hourly_residual_control"
            path = workspace / "outputs" / family / f"{method}.pt"
            artifacts[str(path.relative_to(workspace))] = sha(path)
    result = {"methods": selected, "selection_split": "dev", "meta_used": False, "calibration_used": False,
        "final_used": False, "prepared_signature": sha(workspace / "prepared.json"), "artifact_sha256": artifacts,
        "dataset_sha256": read_json(workspace / "prepared.json")["dataset_sha256"],
        "best_individual_dev": min(scores, key=scores.get),
        "selected_dev_scores": {m: scores[m] for m in selected}, "all_dev_scores": scores,
        "selection_rule": config["individual_selection"], "quantiles": [.1, .5, .9],
        "performance_claim": "Ranked diverse candidates, not a claim that each beats the reference or is clinically validated."}
    write_json(workspace / "components_frozen.json", result)
    write_json(workspace.parent / "individual_selection.json", result)
    return result


def run_meta(workspace):
    from .ensemble_v2 import run_meta as operation
    return operation(workspace)


def fit_ensemble(workspace):
    from .ensemble_v2 import fit_ensemble as operation
    return operation(workspace)


def calibrate_ensemble(workspace):
    from .ensemble_v2 import calibrate_ensemble as operation
    return operation(workspace)


def evaluate_final(workspace, final_dataset):
    from .ensemble_v2 import evaluate_final as operation
    return operation(workspace, final_dataset)
