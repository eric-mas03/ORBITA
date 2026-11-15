"""Technical training experiment; four people cannot validate a health model.

Train: targets through R+45; validation R+82; exploratory test R+194.
Training uses future visits within one time reference; evaluation origins also
respect when fitting and validation-based model selection have completed.
No invented launch-to-return duration and no interpolated target observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.models import AstronautTwin, TwinConfig, grouped_masked_pinball_loss

TRAIN_VISITS = {"L-92", "L-44", "L-3", "R+1", "R+45"}


def prepare(wide: pd.DataFrame, groups: dict) -> dict:
    inputs, outputs = groups["input_groups"], groups["output_groups"]
    features = [v for vs in inputs.values() for v in vs]
    targets = [v for vs in outputs.values() for v in vs]
    if len(features) != len(set(features)) or not set(features).issubset(wide.columns):
        raise ValueError("Input keys missing or duplicated")
    if not set(targets).issubset(features):
        raise ValueError("Initial experiment expects targets to have observed histories")
    train_rows = wide.index.get_level_values("timepoint").isin(TRAIN_VISITS)
    mean = wide.loc[train_rows, features].mean()
    scale = wide.loc[train_rows, features].std(ddof=0)
    if mean.isna().any():
        raise ValueError("An enabled feature has no training observations")
    scale = scale.where(scale.gt(1e-8), 1.0)
    normalized = (wide[features] - mean) / scale
    samples = []
    for subject in wide.index.get_level_values("subject_id").unique():
        subject_frame = normalized.xs(subject, level="subject_id").sort_index()
        for reference in ("L", "R"):
            segment = subject_frame[subject_frame.index.get_level_values("timepoint").str.startswith(reference)]
            for origin in range(len(segment) - 1):
                history = segment.iloc[:origin + 1]
                times = np.asarray([int(t[1:]) for t in history.index.get_level_values("timepoint")], dtype=np.float32)
                for target_idx in range(origin + 1, len(segment)):
                    target_visit = segment.index[target_idx][1]
                    target_day = int(target_visit[1:])
                    split = "val" if target_visit == "R+82" else "test" if target_visit == "R+194" else "train"
                    # A model fitted through R+45 cannot be issued at R+1;
                    # a checkpoint selected on R+82 cannot be issued at R+45.
                    # Historical target pairs are still allowed in training.
                    if split == "val" and history.index[-1][1] != "R+45":
                        continue
                    if split == "test" and history.index[-1][1] != "R+82":
                        continue
                    samples.append({"subject_id": subject, "origin": history.index[-1][1],
                                    "target_visit": target_visit, "horizon": float(target_day - times[-1]),
                                    "history": history.to_numpy(dtype=np.float32), "times": times,
                                    "target": segment.iloc[target_idx].to_numpy(dtype=np.float32), "split": split})
    return {"samples": samples, "mean": mean, "scale": scale,
            "features": features, "targets": targets, "input_groups": inputs, "output_groups": outputs,
            "training_scaler_rows": int(train_rows.sum())}


def batch(samples: list[dict], setup: dict) -> tuple:
    features = setup["features"]
    b, t, f = len(samples), max(len(s["times"]) for s in samples), len(features)
    value = np.zeros((b, t, f), dtype=np.float32)
    mask = np.zeros((b, t, f), dtype=bool)
    age = np.zeros((b, t, f), dtype=np.float32)
    times = np.zeros((b, t), dtype=np.float32)
    padding = np.ones((b, t), dtype=bool)
    for i, sample in enumerate(samples):
        n = len(sample["times"])
        finite = np.isfinite(sample["history"])
        value[i, :n] = np.nan_to_num(sample["history"])
        mask[i, :n] = finite
        times[i, :n] = sample["times"]
        padding[i, :n] = False
        # No forward-filled value: missing values remain masked. Age records
        # time since last observed assay value, or since segment start.
        last = np.full(f, sample["times"][0], dtype=np.float32)
        for k, now in enumerate(sample["times"]):
            last = np.where(finite[k], now, last)
            age[i, k] = now - last
    tensor = torch.from_numpy
    indices = {g: [features.index(v) for v in variables] for g, variables in setup["input_groups"].items()}
    values = {g: tensor(value[:, :, j]) for g, j in indices.items()}
    observed = {g: tensor(mask[:, :, j]) for g, j in indices.items()}
    ages = {g: tensor(age[:, :, j]) for g, j in indices.items()}
    target_values = np.asarray([s["target"] for s in samples], dtype=np.float32)
    targets, target_masks = {}, {}
    for g, variables in setup["output_groups"].items():
        j = [features.index(v) for v in variables]
        target = target_values[:, None, j]
        targets[g] = tensor(np.nan_to_num(target))
        target_masks[g] = tensor(np.isfinite(target))
    arguments = (values, observed, ages, tensor(times), tensor(np.asarray([s["horizon"] for s in samples], dtype=np.float32)[:, None]), tensor(padding))
    return arguments, targets, target_masks


def loss_function(predictions, targets, masks, reduction="equal_head"):
    # Each biomedical block has equal loss weight; large multiplex panels do
    # not silently dominate the smaller chemistry/hematology blocks.
    return grouped_masked_pinball_loss(predictions, targets, masks, reduction=reduction).loss


def predictions_table(model, samples, setup, split, residual_bands):
    model.eval()
    arguments, _, _ = batch(samples, setup)
    with torch.no_grad():
        output = {g: x.numpy() for g, x in model(*arguments).items()}
    rows = []
    for i, sample in enumerate(samples):
        for g, variables in setup["output_groups"].items():
            for k, variable in enumerate(variables):
                j = setup["features"].index(variable)
                actual = sample["target"][j]
                if not np.isfinite(actual):
                    continue
                observed_history = sample["history"][:, j]
                observed_history = observed_history[np.isfinite(observed_history)]
                if not len(observed_history):
                    continue
                base = observed_history[-1] + residual_bands[j]
                for method, q in (("grouped_transformer", output[g][i, 0, k]), ("persistence", base)):
                    mean, scale = float(setup["mean"][variable]), float(setup["scale"][variable])
                    # All 109 enabled targets are nonnegative measurements.
                    # Enforce known support for both methods, without modifying
                    # observations or claiming this postprocessing calibrates.
                    raw_q = np.maximum(q * scale + mean, 0.)
                    q = (raw_q - mean) / scale
                    raw_y = float(actual * scale + mean)
                    error = actual - q
                    pinball = np.maximum(np.array([.1, .5, .9]) * error, (np.array([.1, .5, .9]) - 1) * error)
                    rows.append({"split": split, "method": method, "subject_id": sample["subject_id"],
                                 "origin": sample["origin"], "target_visit": sample["target_visit"],
                                 "horizon_days": sample["horizon"], "group": g, "variable": variable,
                                 "observed": raw_y, "q10": float(raw_q[0]), "q50": float(raw_q[1]), "q90": float(raw_q[2]),
                                 "normalized_mae": float(abs(error[1])), "normalized_pinball": float(pinball.mean()),
                                 "covered": bool(q[0] <= actual <= q[2]),
                                 "normalized_width": float(q[2] - q[0]), "mae_original_units": abs(raw_y - float(raw_q[1]))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-reduction", choices=["equal_head", "per_target"], default="equal_head")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = ROOT / "outputs/training"
    out.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "data/processed/inspiration4_wide.csv"
    config_path = ROOT / "config/biomedical_groups.json"
    wide = pd.read_csv(data_path).set_index(["subject_id", "visit_index", "timepoint"]).sort_index()
    groups = json.loads(config_path.read_text(encoding="utf-8"))
    setup = prepare(wide, groups)
    partitions = {split: [s for s in setup["samples"] if s["split"] == split] for split in ("train", "val", "test")}
    config = TwinConfig(input_groups={g: len(v) for g, v in setup["input_groups"].items()},
                        output_groups=setup["output_groups"], d_model=16, nhead=2, num_layers=1,
                        decoder_layers=1, dropout=.1, time_scale=30.)
    model = AstronautTwin(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    batches = {split: batch(samples, setup) for split, samples in partitions.items()}
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        arguments, target, mask = batches["train"]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(*arguments), target, mask, args.loss_reduction)
        if loss is None:
            # No synthetic zero loss and no AdamW decay/momentum update on a
            # batch without any observed supervised label.
            continue
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            arguments, target, mask = batches["val"]
            val_objective = loss_function(model(*arguments), target, mask, args.loss_reduction)
            if val_objective is None:
                raise ValueError("Validation has no observed labels; cannot select a model")
            val_loss = float(val_objective)
        history.append({"epoch": epoch, "train_loss": float(loss.detach()), "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, best_epoch, stale = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= 15:
            break
    if best_state is None:
        raise ValueError("No labeled training update was performed")
    model.load_state_dict(best_state)
    # Persistence bands fitted exclusively to training transitions, without
    # claims of calibration or horizon-conditional coverage at this sample size.
    residuals = [[] for _ in setup["features"]]
    for sample in partitions["train"]:
        for j, actual in enumerate(sample["target"]):
            previous = sample["history"][:, j]
            previous = previous[np.isfinite(previous)]
            if len(previous) and np.isfinite(actual):
                residuals[j].append(actual - previous[-1])
    bands = []
    for values in residuals:
        if not values:
            raise ValueError("No training residuals for an enabled target")
        low, high = np.quantile(values, [.1, .9])
        bands.append([min(low, 0.), 0., max(high, 0.)])
    tables = [predictions_table(model, samples, setup, split, np.asarray(bands, dtype=np.float32)) for split, samples in partitions.items()]
    predictions = pd.concat(tables, ignore_index=True)
    dictionary = pd.read_csv(ROOT / "outputs/eda/variable_dictionary.csv").set_index("variable")
    predictions["unit"] = predictions.variable.map(dictionary.unit)
    predictions.to_csv(out / "predictions.csv", index=False)
    metrics = predictions.groupby(["split", "method"]).agg(
        scored_pairs=("covered", "size"), normalized_mae=("normalized_mae", "mean"),
        normalized_pinball=("normalized_pinball", "mean"), coverage=("covered", "mean"),
        normalized_width=("normalized_width", "mean"))
    metrics.to_csv(out / "metrics.csv")
    predictions.groupby(["split", "method", "variable", "unit", "horizon_days"]).agg(
        n=("covered", "size"), mae_original_units=("mae_original_units", "mean"),
        normalized_pinball=("normalized_pinball", "mean"), coverage=("covered", "mean"),
        normalized_width=("normalized_width", "mean")).to_csv(out / "metrics_by_variable_horizon.csv")
    pd.DataFrame(history).to_csv(out / "loss_history.csv", index=False)
    manifest = []
    for split, samples in partitions.items():
        manifest.extend({"split": split, "subject_id": s["subject_id"], "origin": s["origin"],
                         "target_visit": s["target_visit"], "horizon_days": s["horizon"],
                         "history_length": len(s["times"])} for s in samples)
    pd.DataFrame(manifest).to_csv(out / "samples.csv", index=False)
    coverage = []
    for split, (_, _, masks) in batches.items():
        for head, mask in masks.items():
            coverage.append({"split": split, "head": head, "observed_labels": int(mask.sum()),
                             "possible_labels": mask.numel(), "fraction": float(mask.float().mean())})
    pd.DataFrame(coverage).to_csv(out / "label_coverage.csv", index=False)
    setup["mean"].rename("mean").to_frame().join(setup["scale"].rename("scale")).to_csv(out / "scaler.csv")
    metadata = {
        "status": "technical_exploratory_not_clinically_validated", "seed": args.seed,
        "epochs_run": len(history), "selected_epoch": best_epoch, "parameter_count": sum(p.numel() for p in model.parameters()),
        "d_model": 16, "nhead": 2, "layers": 1, "time_scale_days": 30.,
        "variables": len(setup["features"]), "groups": len(setup["input_groups"]), "subjects": 4,
        "loss_reduction": args.loss_reduction,
        "samples": {s: len(v) for s, v in partitions.items()}, "scaler_subject_visits": setup["training_scaler_rows"],
        "train_target_visits": ["L-44", "L-3", "R+45"], "val_target_visits": ["R+82"], "test_target_visits": ["R+194"],
        "horizon_days": {s: sorted(set(x["horizon"] for x in v)) for s, v in partitions.items()},
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "groups_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "pandas", "numpy", "scipy"]},
        "assumptions": ["Immediate availability at sample visit is assumed, not observed.",
                        "Separate launch and return segments: no fabricated bridge duration.",
                        "No prospective calibration claim; bands are uncalibrated.",
                        "Test horizons exceed training horizon range.",
                        "Training pairs share targets; biomarkers/visits are dependent, only four test subjects.",
                        "Validation issued at R+45 after fitting; test issued at R+82 after checkpoint selection.",
                        "Full-cohort EDA precedes this experiment: evaluation is exploratory.",
                        "Scalers train-only; no test-tuned parameters; early stopping uses val only.",
                        "No log transform; both methods project predicted quantiles to nonnegative support before scoring.",
                        "Global table averages weight variables equally; group-balanced loss differs."]}
    (out / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    checkpoint = ROOT / "outputs/checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__, "metadata": metadata,
                "mean": setup["mean"].to_dict(), "scale": setup["scale"].to_dict(),
                "input_groups": setup["input_groups"], "output_groups": setup["output_groups"]}, checkpoint / "exploratory_twin.pt")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metrics.to_string())


if __name__ == "__main__":
    main()
