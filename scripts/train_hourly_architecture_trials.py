"""Development-only residual architecture campaign; calibration/test are unused."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.hourly_architecture_trials import (ResidualHourlyTrial, TrialConfig,
                                                     prepare_frozen_trial_data)
from astronaut_twin.hourly_baselines import predict_linear
from astronaut_twin.hourly_neural import global_cell_loss


def chunks(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def evaluate(model, neural, own, base_z, groups, rows, batch_size):
    model.eval()
    sums = torch.zeros(len(neural.horizons), len(neural.variables))
    counts = torch.zeros_like(sums)
    with torch.no_grad():
        for selected in chunks(rows, batch_size):
            args, target, mask = neural.batch(selected, groups)
            prediction = model(args, own[selected], base_z[selected])
            error = target[..., None] - prediction
            q = prediction.new_tensor([.1, .5, .9])
            loss = torch.maximum(q * error, (q - 1) * error).mean(-1)
            sums += (loss * mask).sum(0)
            counts += mask.sum(0)
    active = counts > 0
    return float((sums[active] / counts[active]).mean())


def fit_trial(kind, neural, own, base_z, groups, args):
    train = np.flatnonzero(neural.split == "train")
    dev = np.flatnonzero(neural.split == "dev")
    config = TrialConfig(kind=kind)
    model = ResidualHourlyTrial(groups, neural.variables, config, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    counts = torch.isfinite(neural.labels[train]).sum(0)
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    # Zero correction is a protected candidate, not merely an initialization.
    initial = evaluate(model, neural, own, base_z, groups, dev, args.batch_size)
    best, best_epoch, stale = initial, 0, 0
    state = copy.deepcopy(model.state_dict())
    curves = [{"method": kind, "epoch": 0, "online_train_macro_pinball": None,
               "dev_macro_pinball": initial, "epoch_seconds": time.monotonic() - started}]
    torch.save({"state_dict": state, "config": asdict(config), "groups": groups,
                "variables": neural.variables, "seed": args.seed, "selected_epoch": 0,
                "dev_macro_standardized_pinball": initial, "status": "initial_frozen_baseline"},
               args.out / f"{kind}_epoch0.pt")
    print(f"{kind} epoch=0 exact_frozen_baseline_dev={initial:.9f}", flush=True)
    group_columns = model.group_columns
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        online_loss = 0.
        for selected in chunks(rng.permutation(train), args.batch_size):
            batch_args, target, mask = neural.batch(selected, groups)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_args, own[selected], base_z[selected])
            loss = global_cell_loss(prediction, target, mask, counts, len(train))
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite residual objective")
            loss.backward()
            for group, columns in group_columns.items():
                if not mask[:, :, columns].any():
                    for parameter in list(model.core.decoders[group].parameters()) + list(model.core.output_layers[group].parameters()):
                        parameter.grad = None
                    if kind == "group_temporal_residual":
                        for parameter in model.group_memories[group].parameters():
                            parameter.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            online_loss += float(loss.detach()) * len(selected) / len(train)
        dev_loss = evaluate(model, neural, own, base_z, groups, dev, args.batch_size)
        elapsed = time.monotonic() - epoch_start
        curves.append({"method": kind, "epoch": epoch, "online_train_macro_pinball": online_loss,
                       "dev_macro_pinball": dev_loss, "epoch_seconds": elapsed})
        print(f"{kind} epoch={epoch} train={online_loss:.6f} dev={dev_loss:.9f} seconds={elapsed:.1f}", flush=True)
        if dev_loss < best - 1e-6:
            best, best_epoch, stale, state = dev_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(state)
    info = {"method": kind, "selected_epoch": best_epoch, "epochs_run": epoch,
            "initial_frozen_baseline_dev_macro": initial, "dev_macro_standardized_pinball": best,
            "relative_dev_improvement": 1 - best / initial,
            "status": "no_added_value" if best_epoch == 0 else "dev_improved_exploratory",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": round(time.monotonic() - started, 2), "config": asdict(config)}
    checkpoint = {"state_dict": state, "config": asdict(config), "groups": groups,
                  "variables": neural.variables, "seed": args.seed, "selection": info,
                  "scaler_mean": torch.from_numpy(neural.mean), "scaler_scale": torch.from_numpy(neural.scale),
                  "frozen_baseline_artifact": "outputs/hourly_baselines/linear_quantile.npz",
                  "frozen_band_artifact": "outputs/hourly_baselines/persistence.npz"}
    torch.save(checkpoint, args.out / f"{kind}.pt")
    return model, info, curves


def export_dev(model, method, neural, own, base_z, baseline, packed, groups, metadata, args, first):
    dev = np.flatnonzero(neural.split == "dev")
    predictions = []
    model.eval()
    with torch.no_grad():
        for rows in chunks(dev, args.batch_size):
            model_args, _, _ = neural.batch(rows, groups)
            standardized = model(model_args, own[rows], base_z[rows]).numpy()
            predictions.append(standardized * neural.scale[None, None, :, None] + neural.mean[None, None, :, None])
    predictions = np.concatenate(predictions)
    local, h, v = np.nonzero(np.isfinite(baseline.labels[dev]))
    selected = dev[local]
    episode = neural.episode[selected]
    origins = packed["origins"][selected]
    variables = np.asarray(neural.variables)
    vm = {item["source_key"]: item for item in metadata["variables"]}
    heads = np.asarray([vm[x]["group"] for x in variables])
    units = np.asarray([vm[x]["source_unit_as_documented"] for x in variables])
    unit_review = np.asarray([x in {"FiO2", "Magnesium", "Lactate"} for x in variables])
    train_counts = torch.isfinite(neural.labels[neural.split == "train"]).sum(0).numpy()
    q = predictions[local, h, v]
    table = pd.DataFrame({"split": "dev", "method": method, "episode_id": packed["episode_ids"][episode],
        "site": packed["sites"][episode], "origin_hour": packed["times"][origins],
        "target_hour": packed["times"][origins] + np.asarray(packed["horizons"])[h],
        "horizon_hours": np.asarray(packed["horizons"])[h], "head": heads[v], "variable": variables[v],
        "unit": units[v], "unit_review_required": unit_review[v], "observed": baseline.labels[selected, h, v],
        "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
        "history_hours": baseline.history_hours[selected], "history_length": baseline.history_length[selected],
        "own_history_observed": baseline.counts[selected, v],
        "used_train_median_fallback": baseline.counts[selected, v] == 0,
        "target_scale": neural.scale[v], "train_target_count": train_counts[h, v],
        "prediction_status": np.where(train_counts[h, v] == 0, "no_train_labels",
                                      np.where(train_counts[h, v] < 20, "low_train_support", "fitted"))})
    table.to_csv(args.out / "predictions.csv", mode="w" if first else "a", header=first, index=False)
    print(f"Exported dev {method}: {len(table)} actual labels; no calibration/test exports", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/hourly/hourly_dataset.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hourly_architecture_trials")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-seconds", type=int, default=600)
    parser.add_argument("--group-only", action="store_true",
                        help="Continue a completed first phase with only the group-specific temporal trial")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        parser.error("epochs must be between 1 and 20")
    torch.set_num_threads(2)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    def load_npz(path):
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    packed = load_npz(args.data)
    # Remove every calibration/test sample before creating windows, targets or
    # frozen-baseline forecasts. Raw source rows remain archived but not queried.
    sample_episode = np.searchsorted(packed["episode_offsets"], packed["origins"], side="right") - 1
    keep = np.isin(packed["episode_split"][sample_episode], ["train", "dev"])
    for name in ("origins", "history_start", "target_rows", "sample_episode"):
        if name in packed:
            packed[name] = packed[name][keep]
    baseline_path = ROOT / "outputs/hourly_baselines"
    scalers = load_npz(baseline_path / "scalers.npz")
    frozen_linear = load_npz(baseline_path / "linear_quantile.npz")
    frozen_bands = load_npz(baseline_path / "persistence.npz")
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    metadata = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    groups = metadata["output_groups"]
    allowed = np.flatnonzero(np.isin(neural.split, ["train", "dev"]))
    base_z = torch.full((len(neural.origins), len(neural.horizons), len(neural.variables), 3), float("nan"))
    frozen_predictions = predict_linear(baseline, frozen_linear, frozen_bands, allowed)
    base_z[allowed] = torch.from_numpy((frozen_predictions - neural.mean[None, None, :, None]) / neural.scale[None, None, :, None])
    own = torch.from_numpy(baseline.features)
    if not args.group_only:
        np.savez_compressed(args.out / "scalers.npz", **scalers)
    source_paths = {"dataset": args.data, "linear": baseline_path / "linear_quantile.npz",
        "bands": baseline_path / "persistence.npz", "scalers": baseline_path / "scalers.npz",
        "implementation": ROOT / "src/astronaut_twin/hourly_architecture_trials.py",
        "script": Path(__file__), "base_model": ROOT / "src/astronaut_twin/models/twin.py",
        "neural_preparation": ROOT / "src/astronaut_twin/hourly_neural.py",
        "quality_policy": ROOT / "config/hourly_quality_policy.json",
        "campaign": ROOT / "config/model_campaign.json"}
    manifest = {"status": "development_only_architecture_trials", "source_sha256":
        {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in source_paths.items() if path.exists()},
        "sources": {key: str(path.resolve()) for key, path in source_paths.items()},
        "train_origins": int((neural.split == "train").sum()), "dev_origins": int((neural.split == "dev").sum()),
        "calibration_test_used": False, "baseline_refitted": False, "scalers_refitted": False,
        "ensemble_trained": False, "augmentation": False, "pretraining": False, "seed": args.seed,
        "maximum_epochs": args.epochs, "patience": 4, "batch_size": args.batch_size,
        "optimizer": "AdamW", "learning_rate": .0003, "weight_decay": .001, "gradient_clip_norm": 1.,
        "initial_epoch0_is_selection_candidate": True, "temporal_axis": "Full [B,24,32] memory retained; no temporal pooling before attention.",
        "direct_state": "Per-target current z-value/mask/age plus frozen nine causal own summaries at origin.",
        "residual": "q50 + bounded z correction; baseline gaps times exp(bounded log multiplier); initially exactly baseline; zero gaps remain zero.",
        "loss": "Fixed global train cell counts, macro variable-horizon pinball; minibatch contribution scaled N/B.",
        "physical_projection": False, "selection": "Only development macro standardized pinball, including epoch0 baseline.",
        "gap_zero_fraction_train_dev": float((torch.diff(base_z[allowed], dim=-1) == 0).to(torch.float32).mean()),
        "budget_seconds": args.budget_seconds, "third_trial_gate": "Run only if elapsed plus 1.75*max observed epoch time*maxepochs +20s <=budget.",
        "models": {}, "deferred_trials": {}, "torch_version": str(torch.__version__), "threads": 2}
    models, curves, previous_elapsed = {}, [], 0.
    kinds = ("causal_residual", "window_bidirectional_residual", "group_temporal_residual")
    if args.group_only:
        initial_manifest = json.loads((args.out / "fit_manifest.json").read_text(encoding="utf-8"))
        required = {"causal_residual", "window_bidirectional_residual"}
        if not required.issubset(initial_manifest["models"]) or "group_temporal_residual" in initial_manifest["models"]:
            raise ValueError("Group-only continuation requires exactly the two completed first-phase models")
        if (args.epochs, args.batch_size, args.seed) != (initial_manifest["maximum_epochs"],
                initial_manifest["batch_size"], initial_manifest["seed"]):
            raise ValueError("Continuation cannot change the frozen optimization budget or seed")
        for source in ("dataset", "linear", "bands", "scalers", "implementation", "base_model", "neural_preparation", "quality_policy"):
            if manifest["source_sha256"][source] != initial_manifest["source_sha256"][source]:
                raise ValueError(f"Frozen source changed before continuation: {source}")
        snapshot = args.out / "source_initial_training_script.py"
        if not snapshot.exists() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != initial_manifest["source_sha256"]["script"]:
            raise ValueError("Historical first-phase script snapshot is required and must match its stored hash")
        previous_elapsed = initial_manifest["elapsed_seconds"]
        current_sources = manifest["source_sha256"]
        manifest = initial_manifest
        manifest["runtime_extension"] = {"rootauthorizedruntimeextension": True,
            "previous_budget_seconds": manifest["budget_seconds"], "new_budget_seconds": args.budget_seconds,
            "reason": "Explicit request to complete the group-specific temporal architecture without retraining first-phase models",
            "optimization_unchanged": True, "first_phase_elapsed_seconds": previous_elapsed,
            "initial_script_snapshot": str(snapshot.resolve()), "followup_source_sha256": current_sources}
        manifest["budget_seconds"] = args.budget_seconds
        curves = pd.read_csv(args.out / "learning_curves.csv").to_dict("records")
        kinds = ("group_temporal_residual",)
        (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for kind in kinds:
        if kind == "group_temporal_residual":
            durations = [row["epoch_seconds"] for row in curves if row["epoch"] > 0]
            estimate = (max(durations) if durations else 30.) * args.epochs * 1.75 + 20.
            if previous_elapsed + time.monotonic() - started + estimate > args.budget_seconds:
                manifest["deferred_trials"][kind] = {"reason": "Predeclared runtime budget gate", "estimated_additional_seconds": estimate}
                break
        model, info, history = fit_trial(kind, neural, own, base_z, groups, args)
        models[kind], manifest["models"][kind] = model, info
        manifest["deferred_trials"].pop(kind, None)
        curves.extend(history)
        (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.DataFrame(curves).to_csv(args.out / "learning_curves.csv", index=False)
    manifest["models_frozen_before_export"] = True
    (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for index, (method, model) in enumerate(models.items()):
        export_dev(model, method, neural, own, base_z, baseline, packed, groups, metadata, args,
                   first=(index == 0 and not args.group_only))
    manifest["elapsed_seconds"] = round(previous_elapsed + time.monotonic() - started, 2)
    (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"], "models": manifest["models"],
                      "deferred": manifest["deferred_trials"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
