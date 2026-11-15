"""Fit frozen hourly terrestrial baselines; no Transformer or transfer training."""
from __future__ import annotations

import argparse
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
from astronaut_twin.hourly_baselines import (FEATURE_NAMES, fit_linear_quantiles,
    fit_persistence_bands, predict_linear, predict_persistence, prepare_baseline_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/hourly/hourly_dataset.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hourly_baselines")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 80:
        parser.error("epochs must be between 1 and 80")
    torch.set_num_threads(2)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as archive:
        packed = {key: archive[key] for key in archive.files}
    data = prepare_baseline_data(packed)
    print(f"Prepared {len(data.features)} origins, {data.features.shape[1]} variables; train rows={len(data.train_row_indices)}", flush=True)
    bands = fit_persistence_bands(data, minimum_count=20)
    linear = fit_linear_quantiles(data, epochs=args.epochs, seed=args.seed)
    print(f"Frozen linear selection: {linear['selection']}", flush=True)
    np.savez_compressed(args.out / "scalers.npz", mean=data.mean, scale=data.scale, median=data.median,
                        observed_counts=data.scaler_counts, train_row_indices=data.train_row_indices,
                        feature_names=np.asarray(FEATURE_NAMES), variables=packed["variables"])
    np.savez_compressed(args.out / "persistence.npz", **bands)
    np.savez_compressed(args.out / "linear_quantile.npz", weight=linear["weight"], bias=linear["bias"],
                        train_target_counts=linear["train_target_counts"], feature_names=np.asarray(FEATURE_NAMES))
    pd.DataFrame(linear["curves"]).to_csv(args.out / "learning_curves.csv", index=False)
    config_path = ROOT / "config/hourly_task_candidates.json"
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    variable_metadata = {item["source_key"]: item for item in metadata["variables"]}
    variables = np.asarray(packed["variables"]).astype(str)
    heads = np.asarray([variable_metadata[v]["group"] for v in variables])
    units = np.asarray([variable_metadata[v]["source_unit_as_documented"] for v in variables])
    unit_review = np.asarray([v in {"FiO2", "Magnesium", "Lactate"} for v in variables])
    horizons = np.asarray(packed["horizons"])
    support_rows = []
    for split in ("train", "dev", "calibration", "test"):
        selected = np.flatnonzero(data.split == split)
        valid = np.isfinite(data.labels[selected])
        for h, horizon in enumerate(horizons):
            for v, variable in enumerate(variables):
                sample_indices = selected[valid[:, h, v]]
                support_rows.append({"split": split, "head": heads[v], "variable": variable,
                                     "horizon_hours": int(horizon), "observed_labels": len(sample_indices),
                                     "episodes": len(np.unique(data.episode[sample_indices])),
                                     "origins": len(np.unique(packed["origins"][sample_indices])),
                                     "origins_within_split": len(selected),
                                     "own_history_missing_labels": int((data.counts[sample_indices, v] == 0).sum())})
    pd.DataFrame(support_rows).to_csv(args.out / "label_availability.csv", index=False)
    manifest = {
        "status": "terrestrial_hourly_baselines_not_space_or_clinically_validated",
        "data_path": str(args.data.resolve()), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "task_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selection_frozen_before_test_predictions": True,
        "source": "PhysioNet_Challenge_2019_downloaded_1000_sequence_subset",
        "domain": "terrestrial_ICU", "feature_names": list(FEATURE_NAMES),
        "feature_statistics_scope": "observed own-variable values in causal context, not intrahour signal statistics",
        "context": "at most 24 real hourly slots, partial histories retained",
        "targets": "actual valid source cell at exact origin+horizon, terminal interval ending there",
        "horizons_hours": horizons.tolist(), "variables": variables.tolist(),
        "quantiles": [.1, .5, .9], "seed": args.seed,
        "linear_model": "independent own-variable linear quantile regressions; raw quantiles sorted at prediction",
        "linear_parameter_count": int(linear["weight"].size + linear["bias"].size),
        "learning_rate_candidates": linear["candidates"], "selected": linear["selection"],
        "selection_metric": "macro mean standardized pinball: mean observations within each variable-horizon then mean nonempty cells in dev",
        "training_loss": "macro mean observed pinball per variable-horizon on train-standardized targets; L2 mean(weight^2)*0.001",
        "test_used_for_selection": False, "calibration_used_for_selection": False,
        "calibration_method": "none; calibration split predictions retained for later independent interval calibration",
        "interval_claim": "uncalibrated marginal q10/q90 bands; no joint trajectory or finite-sample coverage guarantee",
        "persistence": "last observed own value within context; train-median fallback if absent; q50 unchanged",
        "persistence_bands": "train residual q10/q90 per horizon-variable if n>=20, otherwise pooled same-variable horizons; min(low,0),max(high,0) ensure q50 inside",
        "persistence_counts_file": "persistence.npz: horizon_counts,fitted_counts,source",
        "unsupported_linear_outputs": "zero train labels: persistence fallback with explicit prediction_status",
        "support": "quality-filtered labels only; no normal-range clipping of observations or predictions",
        "zero_training_input_observations": [str(v) for v, n in zip(variables, data.scaler_counts) if not n],
        "split_origins": {s: int((data.split == s).sum()) for s in ("train", "dev", "calibration", "test")},
        "split_episodes": {s: int((packed["episode_split"] == s).sum()) for s in ("train", "dev", "calibration", "test")},
        "scaler_unique_training_rows": len(data.train_row_indices),
        "unit_review_variables": [str(v) for v, flag in zip(variables, unit_review) if flag],
        "availability_assumption": "Each retrospective hourly row available at issuance; lab turnaround unknown.",
        "identity_limit": "Published sequences are not established unique people across admissions.",
        "quality_policy_sha256_at_export": hashlib.sha256((ROOT / "config/hourly_quality_policy.json").read_bytes()).hexdigest(),
        "quality_implementation_sha256_at_export": hashlib.sha256((ROOT / "src/astronaut_twin/hourly_quality.py").read_bytes()).hexdigest(),
        "baseline_implementation_sha256_at_export": hashlib.sha256((ROOT / "src/astronaut_twin/hourly_baselines.py").read_bytes()).hexdigest(),
        "training_script_sha256_at_export": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__, "torch_threads": 2,
    }
    # Persist the selected settings before generating any held-out test forecasts.
    (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output_path = args.out / "predictions.csv"
    first = True
    prediction_rows = 0
    for split in ("dev", "calibration", "test"):
        selected = np.flatnonzero(data.split == split)
        valid = np.isfinite(data.labels[selected])
        local, h, v = np.nonzero(valid)
        global_indices = selected[local]
        ep = data.episode[global_indices]
        original_rows = packed["origins"][global_indices]
        common = {
            "split": split, "episode_id": packed["episode_ids"][ep], "site": packed["sites"][ep],
            "origin_hour": packed["times"][original_rows],
            "target_hour": packed["times"][original_rows] + horizons[h], "horizon_hours": horizons[h],
            "head": heads[v], "variable": variables[v], "unit": units[v], "unit_review_required": unit_review[v],
            "observed": data.labels[global_indices, h, v], "history_hours": data.history_hours[global_indices],
            "history_length": data.history_length[global_indices],
            "own_history_observed": data.counts[global_indices, v],
            "used_train_median_fallback": data.counts[global_indices, v] == 0,
            "target_scale": data.scale[v], "train_target_count": linear["train_target_counts"][h, v],
            "band_train_horizon_count": bands["horizon_counts"][h, v],
            "band_fit_count": bands["fitted_counts"][h, v], "band_source": bands["source"][h, v],
        }
        for method in ("persistence", "linear_quantile"):
            predicted = predict_persistence(data, bands, selected) if method == "persistence" else predict_linear(data, linear, bands, selected)
            q = predicted[local, h, v]
            table = pd.DataFrame(common)
            table.insert(1, "method", method)
            table["q10"], table["q50"], table["q90"] = q[:, 0], q[:, 1], q[:, 2]
            table["prediction_status"] = np.where(linear["train_target_counts"][h, v] == 0,
                "no_train_labels_persistence_fallback", np.where(linear["train_target_counts"][h, v] < 20,
                "low_train_support", "fitted"))
            table.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
            first = False
            prediction_rows += len(table)
            print(f"Wrote {split} {method}: {len(table)} observed label pairs", flush=True)
    manifest["prediction_rows"] = prediction_rows
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 2)
    manifest["quality_policy_sha256_at_export"] = hashlib.sha256((ROOT / "config/hourly_quality_policy.json").read_bytes()).hexdigest()
    manifest["quality_implementation_sha256_at_export"] = hashlib.sha256((ROOT / "src/astronaut_twin/hourly_quality.py").read_bytes()).hexdigest()
    (args.out / "fit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"], "prediction_rows": prediction_rows,
                      "selected": linear["selection"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
