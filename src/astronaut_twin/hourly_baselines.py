"""Causal own-variable hourly baselines with train-only statistics.

Packed NPZ input is produced by hourly_data. No labels are interpolated and no
spatial adaptation or clinical claims are made by these terrestrial baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Mapping

import numpy as np
import torch


FEATURE_NAMES = ("last_z", "has_history", "age_over_24", "mean_z", "min_z", "max_z",
                 "count_over_24", "std_z", "history_hours_over_24")
QUANTILES = np.array([.1, .5, .9], dtype=np.float32)


@dataclass
class BaselineData:
    features: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    episode: np.ndarray
    last: np.ndarray
    counts: np.ndarray
    history_hours: np.ndarray
    history_length: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    median: np.ndarray
    scaler_counts: np.ndarray
    train_row_indices: np.ndarray


def prepare_baseline_data(packed: Mapping[str, np.ndarray]) -> BaselineData:
    values = np.asarray(packed["values"], dtype=np.float32)
    times = np.asarray(packed["times"], dtype=np.float32)
    offsets = np.asarray(packed["episode_offsets"], dtype=np.int64)
    origins = np.asarray(packed["origins"], dtype=np.int64)
    target_rows = np.asarray(packed["target_rows"], dtype=np.int64)
    starts = np.asarray(packed["history_start"], dtype=np.int64)
    episode_split = np.asarray(packed["episode_split"]).astype(str)
    episode = np.searchsorted(offsets, origins, side="right") - 1
    split = episode_split[episode]
    train_rows = np.concatenate([np.arange(offsets[i], offsets[i + 1])
                                 for i in range(len(episode_split)) if episode_split[i] == "train"])
    if not len(train_rows):
        raise ValueError("No training rows")
    feature_count = values.shape[1]
    mean, scale, median = np.zeros(feature_count), np.ones(feature_count), np.zeros(feature_count)
    scaler_counts = np.zeros(feature_count, dtype=np.int64)
    for j in range(feature_count):
        available = values[train_rows, j]
        available = available[np.isfinite(available)]
        scaler_counts[j] = len(available)
        if len(available):
            mean[j], median[j] = np.mean(available, dtype=np.float64), np.median(available)
            std = float(np.std(available, dtype=np.float64))
            scale[j] = std if std > 1e-8 else 1.
    mean, scale, median = (x.astype(np.float32) for x in (mean, scale, median))
    x = np.zeros((len(origins), feature_count, len(FEATURE_NAMES)), dtype=np.float32)
    last = np.zeros((len(origins), feature_count), dtype=np.float32)
    counts = np.zeros_like(last, dtype=np.int32)
    history_hours = np.zeros(len(origins), dtype=np.float32)
    history_length = np.zeros(len(origins), dtype=np.int32)
    for n, (start, origin, ep) in enumerate(zip(starts, origins, episode)):
        if not offsets[ep] <= start <= origin < offsets[ep + 1]:
            raise ValueError("History crosses episode boundary")
        history = values[start:origin + 1]
        observed = np.isfinite(history)
        count = observed.sum(0)
        exists = count > 0
        local_times = times[start:origin + 1]
        if (np.diff(local_times) < 0).any() or times[origin] - local_times[0] > 23.0001:
            raise ValueError("History must be causal, ordered and within 24 hourly slots")
        latest_indices = np.maximum(np.where(observed, np.arange(len(history))[:, None], -1).max(0), 0)
        last_value = np.where(exists, history[latest_indices, np.arange(feature_count)], median)
        age = np.where(exists, times[origin] - local_times[latest_indices], 24.)
        total = np.where(observed, history, 0).sum(0, dtype=np.float64)
        avg = np.where(exists, total / np.maximum(count, 1), median)
        low = np.where(exists, np.where(observed, history, np.inf).min(0), median)
        high = np.where(exists, np.where(observed, history, -np.inf).max(0), median)
        variance = np.where(observed, (history - avg) ** 2, 0).sum(0) / np.maximum(count, 1)
        duration = times[origin] - local_times[0] + 1.
        x[n] = np.stack(((last_value - mean) / scale, exists, age / 24., (avg - mean) / scale,
                         (low - mean) / scale, (high - mean) / scale, count / 24.,
                         np.sqrt(variance) / scale, np.full(feature_count, duration / 24.)), -1)
        last[n], counts[n], history_hours[n], history_length[n] = last_value, count, duration, len(history)
    labels = values[np.maximum(target_rows, 0)].copy()
    labels[target_rows < 0] = np.nan
    horizons = np.asarray(packed.get("horizons", [1, 4, 8, 24]))
    if target_rows.shape != (len(origins), len(horizons)):
        raise ValueError("Incorrect target row shape")
    for h, horizon in enumerate(horizons):
        valid = target_rows[:, h] >= 0
        rows = target_rows[valid, h]
        ep = episode[valid]
        if (rows < offsets[ep]).any() or (rows >= offsets[ep + 1]).any():
            raise ValueError("Target crosses episode boundary")
        if not np.allclose(times[rows] - times[origins[valid]], horizon):
            raise ValueError("Target is not at its exact positive horizon")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite engineered features")
    return BaselineData(x, labels, split, episode, last, counts, history_hours, history_length,
                        mean, scale, median, scaler_counts, train_rows)


def fit_persistence_bands(data: BaselineData, minimum_count: int = 20) -> dict[str, np.ndarray]:
    """Train-only residual bands; q50 remains literal persistence/median fallback."""
    train = data.split == "train"
    residual = data.labels[train] - data.last[train, None, :]
    _, horizons, variables = residual.shape
    bands = np.zeros((horizons, variables, 3), dtype=np.float32)
    counts = np.isfinite(residual).sum(0)
    fitted_counts = np.zeros_like(counts)
    source = np.full(counts.shape, "no_training_residuals", dtype="U32")
    for h in range(horizons):
        for v in range(variables):
            selected = residual[:, h, v]
            selected = selected[np.isfinite(selected)]
            if len(selected) >= minimum_count:
                source[h, v] = "horizon_variable"
            else:
                selected = residual[:, :, v].reshape(-1)
                selected = selected[np.isfinite(selected)]
                if len(selected):
                    source[h, v] = "pooled_variable_low_support" if len(selected) < minimum_count else "pooled_variable"
            fitted_counts[h, v] = len(selected)
            if len(selected):
                low, high = np.quantile(selected, [.1, .9])
                bands[h, v] = [min(float(low), 0.), 0., max(float(high), 0.)]
    return {"bands": bands, "horizon_counts": counts, "fitted_counts": fitted_counts, "source": source}


def predict_persistence(data: BaselineData, artifact: Mapping[str, np.ndarray], rows=None) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(data.features))
    return data.last[rows, None, :, None] + artifact["bands"][None]


class OwnVariableLinearQuantiles(torch.nn.Module):
    def __init__(self, variables: int, horizons: int, features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(variables, horizons, 3, features))
        self.bias = torch.nn.Parameter(torch.zeros(variables, horizons, 3))

    def forward(self, features):
        return torch.einsum("nvf,vhqf->nhvq", features, self.weight) + self.bias.permute(1, 0, 2)


def score_pinball(predictions: np.ndarray, labels: np.ndarray, scale: np.ndarray) -> float:
    valid = np.isfinite(labels)
    if not valid.any():
        return float("nan")
    error = (labels[..., None] - predictions) / scale[None, None, :, None]
    return float(np.maximum(QUANTILES * error, (QUANTILES - 1) * error)[valid].mean())


def fit_linear_quantiles(data: BaselineData, learning_rates=(.01, .03), epochs: int = 80,
                         weight_penalty: float = 1e-3, patience: int = 12, seed: int = 42) -> dict:
    """Choose learning rate/checkpoint using dev only; never score calibration/test."""
    if epochs < 1 or not learning_rates:
        raise ValueError("Need positive epochs and at least one candidate")
    train, dev = np.flatnonzero(data.split == "train"), np.flatnonzero(data.split == "dev")
    if not len(train) or not len(dev):
        raise ValueError("Training and development origins required")
    train_y = (data.labels[train] - data.mean[None, None]) / data.scale[None, None]
    valid = np.isfinite(train_y)
    counts = valid.sum(0)
    if not valid.any() or not np.isfinite(data.labels[dev]).any():
        raise ValueError("Training and development labels required")
    x_train = torch.from_numpy(data.features[train])
    y_train = torch.from_numpy(np.where(valid, train_y, 0))
    mask = torch.from_numpy(valid)
    cell_counts = mask.sum(0).clamp_min(1)
    active_cells = mask.any(0)
    x_dev = torch.from_numpy(data.features[dev])
    dev_y = (data.labels[dev] - data.mean[None, None]) / data.scale[None, None]
    dev_mask = torch.from_numpy(np.isfinite(dev_y))
    dev_counts = dev_mask.sum(0).clamp_min(1)
    dev_active = dev_mask.any(0)
    dev_target = torch.from_numpy(np.nan_to_num(dev_y))
    q = torch.from_numpy(QUANTILES)
    initial_bias = np.zeros((train_y.shape[2], train_y.shape[1], 3), dtype=np.float32)
    for v in range(train_y.shape[2]):
        for h in range(train_y.shape[1]):
            target = train_y[:, h, v]
            target = target[np.isfinite(target)]
            if len(target):
                initial_bias[v, h] = np.quantile(target, QUANTILES)
    candidates, curves, selected = [], [], None
    for rate in learning_rates:
        torch.manual_seed(seed)
        model = OwnVariableLinearQuantiles(train_y.shape[2], train_y.shape[1], data.features.shape[-1])
        with torch.no_grad():
            model.bias.copy_(torch.from_numpy(initial_bias))
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        best, state, best_epoch, stale = float("inf"), None, 0, 0
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            error = y_train[..., None] - model(x_train)
            per_pair = torch.maximum(q * error, (q - 1) * error).mean(-1)
            loss = ((per_pair * mask).sum(0) / cell_counts)[active_cells].mean()
            objective = loss + weight_penalty * model.weight.square().mean()
            if not torch.isfinite(objective):
                raise RuntimeError("Nonfinite linear training objective")
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                pred = model(x_dev).sort(-1).values
                error = dev_target[..., None] - pred
                per_pair = torch.maximum(q * error, (q - 1) * error).mean(-1)
                dev_loss = float(((per_pair * dev_mask).sum(0) / dev_counts)[dev_active].mean())
            curves.append({"learning_rate": rate, "epoch": epoch, "train_pinball": float(loss.detach()), "dev_pinball": dev_loss})
            if dev_loss < best - 1e-6:
                best, state, best_epoch, stale = dev_loss, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
        candidate = {"learning_rate": rate, "selected_epoch": best_epoch, "epochs_run": epoch,
                     "dev_standardized_pinball": best}
        candidates.append(candidate)
        if selected is None or best < selected["selection"]["dev_standardized_pinball"]:
            selected = {"weight": state["weight"].numpy(), "bias": state["bias"].numpy(),
                        "selection": candidate, "train_target_counts": counts}
    selected["candidates"], selected["curves"] = candidates, curves
    return selected


def predict_linear(data: BaselineData, artifact: Mapping, persistence_artifact: Mapping, rows=None) -> np.ndarray:
    if rows is None:
        rows = np.arange(len(data.features))
    model = OwnVariableLinearQuantiles(data.features.shape[1], data.labels.shape[1], data.features.shape[-1])
    with torch.no_grad():
        model.weight.copy_(torch.as_tensor(artifact["weight"]))
        model.bias.copy_(torch.as_tensor(artifact["bias"]))
        standardized = model(torch.from_numpy(data.features[rows])).numpy()
    prediction = np.sort(standardized, axis=-1) * data.scale[None, None, :, None] + data.mean[None, None, :, None]
    # No labels in a variable/horizon means no supervised linear fit there.
    unsupported = artifact["train_target_counts"] == 0
    baseline = predict_persistence(data, persistence_artifact, rows)
    return np.where(unsupported[None, :, :, None], baseline, prediction)
