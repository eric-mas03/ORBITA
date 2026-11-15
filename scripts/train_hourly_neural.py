"""Controlled GRU-memory vs causal-MHA-memory comparison on hourly ICU data."""
from __future__ import annotations

import argparse
import copy
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
from astronaut_twin.hourly_neural import (flatten_predictions, global_cell_loss,
                                        make_hourly_twin, prepare_hourly_neural)


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def evaluate(model, data, groups, indices, batch_size):
    model.eval()
    total = torch.zeros(len(data.horizons), len(data.variables))
    counts = torch.zeros_like(total)
    q = torch.tensor([.1, .5, .9])
    with torch.no_grad():
        for rows in batches(indices, batch_size):
            args, target, mask = data.batch(rows, groups)
            prediction = flatten_predictions(model(*args), groups, data.variables)
            error = target[..., None] - prediction
            pair_loss = torch.maximum(q * error, (q - 1) * error).mean(-1)
            total += (pair_loss * mask).sum(0)
            counts += mask.sum(0)
    active = counts > 0
    return float((total[active] / counts[active]).mean()) if active.any() else float("nan")


def train_one(method, data, groups, epochs, batch_size, seed, out):
    train = np.flatnonzero(data.split == "train")
    dev = np.flatnonzero(data.split == "dev")
    counts = torch.isfinite(data.labels[train]).sum(0)
    model = make_hourly_twin(groups, method, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(seed)
    best, best_epoch, stale, state, curves = float("inf"), 0, 0, None, []
    started = time.monotonic()
    group_columns = {g: [data.variables.index(v.removeprefix("SEPSIS2019::")) for v in vs]
                     for g, vs in groups.items()}
    for epoch in range(1, epochs + 1):
        model.train()
        objective_sum, supervised_batches = 0., 0
        epoch_started = time.monotonic()
        for rows in batches(rng.permutation(train), batch_size):
            args, target, mask = data.batch(rows, groups)
            optimizer.zero_grad(set_to_none=True)
            output = flatten_predictions(model(*args), groups, data.variables)
            loss = global_cell_loss(output, target, mask, counts, len(train))
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss in {method}")
            loss.backward()
            # A whole head without labels in this minibatch is not an AdamW
            # decay/momentum update. Shared encoders still learn from other heads.
            for group, columns in group_columns.items():
                if not mask[:, :, columns].any():
                    for parameter in list(model.decoders[group].parameters()) + list(model.output_layers[group].parameters()):
                        parameter.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            objective_sum += float(loss.detach()) * len(rows) / len(train)
            supervised_batches += 1
        dev_loss = evaluate(model, data, groups, dev, batch_size)
        if not np.isfinite(dev_loss):
            raise RuntimeError("No finite dev selection criterion")
        curves.append({"method": method, "epoch": epoch, "online_train_macro_pinball": objective_sum,
                       "dev_macro_pinball": dev_loss, "supervised_batches": supervised_batches,
                       "epoch_seconds": time.monotonic() - epoch_started})
        print(f"{method} epoch={epoch} train={objective_sum:.6f} dev={dev_loss:.6f} seconds={curves[-1]['epoch_seconds']:.1f}", flush=True)
        if dev_loss < best - 1e-6:
            best, best_epoch, stale, state = dev_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(state)
    info = {"method": method, "selected_epoch": best_epoch, "epochs_run": epoch,
            "dev_macro_standardized_pinball": best, "parameter_count": sum(p.numel() for p in model.parameters()),
            "temporal_parameter_count": sum(p.numel() for p in model.temporal.parameters()),
            "elapsed_seconds": round(time.monotonic() - started, 2)}
    # The selected state and settings exist before any calibration/test forecasts.
    torch.save({"state_dict": state, "method": method, "config": model.config.__dict__,
                "selection": info, "mean": data.mean, "scale": data.scale, "groups": groups},
               out / f"{method}.pt")
    return model, info, curves


def forecast(model, data, groups, indices, batch_size):
    model.eval()
    predictions, own_counts, durations, lengths = [], [], [], []
    grouped_order = [v.removeprefix("SEPSIS2019::") for variables in groups.values() for v in variables]
    reorder = [grouped_order.index(v) for v in data.variables]
    with torch.no_grad():
        for rows in batches(indices, batch_size):
            args, _, _ = data.batch(rows, groups)
            prediction = flatten_predictions(model(*args), groups, data.variables)
            prediction = prediction.numpy() * data.scale[None, None, :, None] + data.mean[None, None, :, None]
            predictions.append(prediction)
            counts = torch.cat([args[1][group].sum(1) for group in groups], 1)[:, reorder]
            own_counts.append(counts.numpy())
            lengths.append((~args[-1]).sum(1).numpy())
            last = args[3].gather(1, ((~args[-1]).sum(1) - 1)[:, None])[:, 0]
            durations.append((last - args[3][:, 0] + 1).numpy())
    return np.concatenate(predictions), np.concatenate(own_counts), np.concatenate(durations), np.concatenate(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/hourly/hourly_dataset.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hourly_neural")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 25 or args.batch_size < 1:
        parser.error("epochs must be 1..25 and batch-size positive")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        packed = {key: archive[key] for key in archive.files}
    data = prepare_hourly_neural(packed)
    config_path = ROOT / "config/hourly_task_candidates.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    groups = metadata["output_groups"]
    assert {v.removeprefix("SEPSIS2019::") for vs in groups.values() for v in vs} == set(data.variables)
    np.savez_compressed(args.out / "scalers.npz", mean=data.mean, scale=data.scale, median=data.median,
                        observed_counts=data.scaler_counts, train_row_indices=data.train_row_indices,
                        variables=np.asarray(data.variables))
    manifest = {
        "status": "terrestrial_hourly_from_scratch_comparison_not_clinical_or_space_validation",
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "data_path": str(args.data.resolve()), "task_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "quality_policy_sha256": hashlib.sha256((ROOT / "config/hourly_quality_policy.json").read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256((ROOT / "src/astronaut_twin/hourly_neural.py").read_bytes()).hexdigest(),
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_twin_sha256": hashlib.sha256((ROOT / "src/astronaut_twin/models/twin.py").read_bytes()).hexdigest(),
        "models": {}, "groups": groups, "variables": data.variables, "horizons_hours": data.horizons.tolist(),
        "seed": args.seed, "batch_size": args.batch_size, "maximum_epochs": args.epochs, "patience": 5,
        "learning_rate": .001, "weight_decay": .001, "optimizer": "AdamW", "gradient_clip_norm": 1.,
        "d_model": 16, "nhead": 2, "temporal_layers": 1, "decoder_layers": 1, "dropout": .1,
        "time_scale_hours": 24., "same_common_initialization": True,
        "difference": "Only temporal memory swapped: causal MHA vs causal GRU+dropout+LayerNorm; same category MLPs, fusion, horizon queries, grouped Transformer attention readout and ordered outputs.",
        "gru_is_pure_recurrent_model": False,
        "inputs": "Actual own measurements z-scaled on unique train rows, mask, causal age, real times; right pad to 24 slots; no carried/imputed observations.",
        "missing_numeric_input": "zero normalized value plus observed=false; not train median fallback",
        "loss": "Mean observations/quantiles within each globally counted train variable-horizon cell, then equal active-cell mean. Each minibatch sum uses fixed global cell counts times Ntrain/B; no rare-cell batch denominator.",
        "unlabeled_whole_batch": "skip backward/optimizer step",
        "unlabeled_head_in_batch": "exclusive decoder/output parameters grad=None before AdamW; shared encoders may update",
        "selection_metric": "dev macro standardized pinball averaged within actual variable-horizon cells then across nonempty cells",
        "selection_uses_test_or_calibration": False,
        "calibration_method": "none", "quantiles": [.1, .5, .9], "physical_projection": False,
        "scaler_unique_train_rows": len(data.train_row_indices),
        "origins_by_split": {s: int((data.split == s).sum()) for s in ("train", "dev", "calibration", "test")},
        "availability_assumption": "Retrospective row immediately available; laboratory latency not supplied.",
        "independence_limit": "Overlapping origins/targets correlated; episodes not guaranteed unique people across admissions.",
        "torch_version": torch.__version__, "threads": 2,
    }
    trained, curves = {}, []
    for method in ("gru_shared_readout", "transformer_grouped"):
        model, info, history = train_one(method, data, groups, args.epochs, args.batch_size, args.seed, args.out)
        trained[method], manifest["models"][method] = model, info
        curves.extend(history)
        (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.DataFrame(curves).to_csv(args.out / "learning_curves.csv", index=False)
    manifest["both_models_frozen_before_test_predictions"] = True
    (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    variable_metadata = {item["source_key"]: item for item in metadata["variables"]}
    variables = np.asarray(data.variables)
    head = np.asarray([variable_metadata[v]["group"] for v in variables])
    units = np.asarray([variable_metadata[v]["source_unit_as_documented"] for v in variables])
    unit_review = np.asarray([v in {"FiO2", "Magnesium", "Lactate"} for v in variables])
    train = np.flatnonzero(data.split == "train")
    train_counts = torch.isfinite(data.labels[train]).sum(0).numpy()
    first = True
    for split in ("dev", "calibration", "test"):
        selected = np.flatnonzero(data.split == split)
        labels = data.labels[selected].numpy()
        local, h, v = np.nonzero(np.isfinite(labels))
        global_indices = selected[local]
        episode = data.episode[global_indices]
        origins = packed["origins"][global_indices]
        for method, model in trained.items():
            prediction, counts, duration, length = forecast(model, data, groups, selected, args.batch_size)
            q = prediction[local, h, v]
            table = pd.DataFrame({
                "split": split, "method": method, "episode_id": packed["episode_ids"][episode],
                "site": packed["sites"][episode], "origin_hour": packed["times"][origins],
                "target_hour": packed["times"][origins] + np.asarray(packed["horizons"])[h],
                "horizon_hours": np.asarray(packed["horizons"])[h], "head": head[v], "variable": variables[v],
                "unit": units[v], "unit_review_required": unit_review[v], "observed": labels[local, h, v],
                "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2], "history_hours": duration[local],
                "history_length": length[local], "own_history_observed": counts[local, v],
                "used_train_median_fallback": False, "target_scale": data.scale[v],
                "train_target_count": train_counts[h, v],
                "prediction_status": np.where(train_counts[h, v] == 0, "untrained_output_no_train_labels",
                                              np.where(train_counts[h, v] < 20, "low_train_support", "fitted"))})
            table.to_csv(args.out / "predictions.csv", mode="w" if first else "a", header=first, index=False)
            first = False
            print(f"Exported {split} {method}: {len(table)} actual label pairs", flush=True)
    print("Neural comparison complete; test was not used for model selection.", flush=True)


if __name__ == "__main__":
    main()
