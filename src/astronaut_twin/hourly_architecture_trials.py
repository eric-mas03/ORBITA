"""Full-window residual sequence trials over a frozen linear-quantile baseline."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .hourly_baselines import BaselineData
from .hourly_neural import NeuralData, flatten_predictions
from .models import AstronautTwin, TwinConfig


@dataclass
class TrialConfig:
    kind: str = "causal_residual"
    d_model: int = 32
    nhead: int = 4
    dropout: float = .1
    time_scale_hours: float = 24.
    median_correction_bound_z: float = 1.
    log_gap_multiplier_bound: float = 1.


class FullPastMemory(nn.Module):
    """Bidirectional attention is legal only over a pretruncated past window."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, src, mask=None, src_key_padding_mask=None):
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)


class IdentityMemory(nn.Module):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        return src


class OwnTargetCorrection(nn.Module):
    """Per-target horizon-context plus direct own summaries/current state."""

    def __init__(self, targets: int, context_width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(targets, 3, context_width + 12))
        self.bias = nn.Parameter(torch.zeros(targets, 3))

    def forward(self, context, summary, state):
        batch, horizons, width = context.shape
        own = torch.cat((summary, state), -1)
        features = torch.cat((context[:, :, None, :].expand(-1, -1, own.shape[1], -1),
                              own[:, None, :, :].expand(-1, horizons, -1, -1)), -1)
        return torch.einsum("bhkf,kqf->bhkq", features, self.weight) + self.bias


def ordered_residual_quantiles(baseline, correction, config: TrialConfig):
    """Identity at zero; baseline positive gaps get bounded multiplicative changes."""
    if baseline.shape != correction.shape or baseline.shape[-1] != 3:
        raise ValueError("Expected matching baseline/correction [...,3]")
    if not torch.isfinite(baseline).all() or (torch.diff(baseline, dim=-1) < 0).any():
        raise ValueError("Frozen baseline must be finite and ordered")
    delta = config.median_correction_bound_z * torch.tanh(correction[..., 1])
    log_low = config.log_gap_multiplier_bound * torch.tanh(correction[..., 0])
    log_high = config.log_gap_multiplier_bound * torch.tanh(correction[..., 2])
    median = baseline[..., 1] + delta
    # expm1(0)==0 preserves the exact baseline rather than reconstructing q10/90.
    low = baseline[..., 0] + delta - (baseline[..., 1] - baseline[..., 0]) * torch.expm1(log_low)
    high = baseline[..., 2] + delta + (baseline[..., 2] - baseline[..., 1]) * torch.expm1(log_high)
    return torch.stack((torch.minimum(low, median), median, torch.maximum(high, median)), -1)


class ResidualHourlyTrial(nn.Module):
    """Preserves [B,N,d] through encoding and reads it with horizon queries.

    All inputs, baseline and own summaries must already be truncated to their
    issuance origin. Intermediate cutoff requests are rejected for every trial:
    otherwise even causal memory would coexist with future-derived summaries.
    """

    def __init__(self, groups: Mapping[str, Sequence[str]], variables: Sequence[str],
                 config: TrialConfig | None = None, seed: int = 42):
        super().__init__()
        self.config = config or TrialConfig()
        if self.config.kind not in {"causal_residual", "window_bidirectional_residual", "group_temporal_residual"}:
            raise ValueError("Unknown architecture trial")
        torch.manual_seed(seed)
        self.groups = dict(groups)
        self.variables = tuple(variables)
        self.group_columns = {g: [self.variables.index(v.removeprefix("SEPSIS2019::")) for v in vs]
                              for g, vs in groups.items()}
        columns = [v for vs in self.group_columns.values() for v in vs]
        if sorted(columns) != list(range(len(self.variables))):
            raise ValueError("Groups must partition variables exactly once")
        model_config = TwinConfig({g: len(vs) for g, vs in groups.items()}, groups,
                                  d_model=self.config.d_model, nhead=self.config.nhead,
                                  num_layers=1, decoder_layers=1, dropout=self.config.dropout,
                                  time_scale=self.config.time_scale_hours)
        self.core = AstronautTwin(model_config)
        if self.config.kind == "window_bidirectional_residual":
            self.core.temporal = FullPastMemory(self.core.temporal)
        elif self.config.kind == "group_temporal_residual":
            self.group_memories = nn.ModuleDict({g: copy.deepcopy(self.core.temporal) for g in groups})
            self.core.temporal = IdentityMemory()
        self.core.output_layers = nn.ModuleDict({g: OwnTargetCorrection(len(vs), self.config.d_model)
                                                for g, vs in groups.items()})

    def forward(self, model_args, own_features, baseline_z, cutoff_index=None):
        values, observed, ages, timestamps, horizons, padding = model_args
        last = (~padding).sum(1) - 1
        if cutoff_index is not None:
            requested = torch.as_tensor(cutoff_index, device=last.device)
            if not torch.all(requested == last):
                raise ValueError("Pretruncate history, baseline and summaries; intermediate cutoff is unsafe")
        batch, steps = timestamps.shape
        if own_features.shape != (batch, len(self.variables), 9):
            raise ValueError("Own causal summary must have shape [B,V,9]")
        memory, padding = self.core.encode_history(values, observed, ages, timestamps, padding)
        rows = torch.arange(batch, device=timestamps.device)
        origin = timestamps[rows, last] - timestamps[:, 0]
        horizons = horizons.to(dtype=memory.dtype, device=memory.device)
        if horizons.ndim == 1:
            horizons = horizons[None].expand(batch, -1)
        if baseline_z.shape != (batch, horizons.shape[1], len(self.variables), 3):
            raise ValueError("Incorrect frozen baseline shape")
        queries = self.core.time_features((origin[:, None] + horizons) / self.config.time_scale_hours)
        queries += self.core.time_features(horizons / self.config.time_scale_hours)
        query_mask = ~torch.eye(horizons.shape[1], dtype=torch.bool, device=memory.device)
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=memory.device), 1)
        corrections = {}
        for group, columns in self.group_columns.items():
            group_memory = memory
            if self.config.kind == "group_temporal_residual":
                group_memory = self.group_memories[group](memory, mask=causal, src_key_padding_mask=padding)
            context = self.core.decoders[group](queries, group_memory,
                tgt_mask=query_mask, memory_key_padding_mask=padding)
            # Current measured slot plus explicit availability and age. A zero
            # value with mask=false remains missing, not a carried observation.
            state = torch.stack((values[group][rows, last], observed[group][rows, last].to(memory.dtype),
                                 ages[group][rows, last] / self.config.time_scale_hours), -1)
            corrections[group] = self.core.output_layers[group](context, own_features[:, columns], state)
        raw = flatten_predictions(corrections, self.groups, self.variables)
        return ordered_residual_quantiles(baseline_z, raw, self.config)


def prepare_frozen_trial_data(packed, scalers):
    """Apply frozen training scalers; works on held-out-only packed episodes.

    Returns both sequence inputs and exactly the original nine own-variable
    summaries. No statistics, models, thresholds or residual bands are fitted.
    """
    values = np.asarray(packed["values"], dtype=np.float32)
    mean, scale, median = (np.asarray(scalers[k], dtype=np.float32) for k in ("mean", "scale", "median"))
    variables = tuple(str(v) for v in packed["variables"])
    if "variables" in scalers and list(map(str, scalers["variables"])) != list(variables):
        raise ValueError("Variable order differs from frozen scalers")
    offsets = np.asarray(packed["episode_offsets"], dtype=np.int64)
    origins = np.asarray(packed["origins"], dtype=np.int64)
    starts = np.asarray(packed["history_start"], dtype=np.int64)
    times = np.asarray(packed["times"], dtype=np.float32)
    episode = np.searchsorted(offsets, origins, side="right") - 1
    split = np.asarray(packed["episode_split"]).astype(str)[episode]
    target_rows = np.asarray(packed["target_rows"], dtype=np.int64)
    if ((starts < offsets[episode]) | (starts > origins) | (origins >= offsets[episode + 1])).any():
        raise ValueError("History crosses episode")
    if (times[origins] - times[starts] > 23.0001).any():
        raise ValueError("History exceeds 24 hourly slots")
    labels = values[np.maximum(target_rows, 0)].copy()
    labels[target_rows < 0] = np.nan
    for h, horizon in enumerate(packed["horizons"]):
        valid = target_rows[:, h] >= 0
        rows, ep = target_rows[valid, h], episode[valid]
        if (rows < offsets[ep]).any() or (rows >= offsets[ep + 1]).any():
            raise ValueError("Target crosses episode")
        if not np.allclose(times[rows] - times[origins[valid]], horizon):
            raise ValueError("Target is not at exact horizon")
    observed = np.isfinite(values)
    normalized = np.where(observed, (values - mean) / scale, 0).astype(np.float32)
    scaler_counts = np.asarray(scalers.get("observed_counts", np.zeros(values.shape[1])), dtype=np.int64)
    train_rows = np.asarray(scalers.get("train_row_indices", []), dtype=np.int64)
    neural = NeuralData(torch.from_numpy(normalized), torch.from_numpy(observed), torch.from_numpy(times),
        torch.from_numpy(origins), torch.from_numpy(starts), torch.from_numpy(labels),
        torch.as_tensor(packed["horizons"], dtype=torch.float32), variables, split, episode,
        mean, scale, median, scaler_counts, train_rows)
    features = np.zeros((len(origins), len(variables), 9), dtype=np.float32)
    last = np.zeros((len(origins), len(variables)), dtype=np.float32)
    counts = np.zeros_like(last, dtype=np.int32)
    durations = np.zeros(len(origins), dtype=np.float32)
    lengths = np.zeros(len(origins), dtype=np.int32)
    for n, (start, origin) in enumerate(zip(starts, origins)):
        history = values[start:origin + 1]
        local_times = times[start:origin + 1]
        if (np.diff(local_times) < 0).any():
            raise ValueError("History times must be ordered")
        available = np.isfinite(history)
        count = available.sum(0)
        exists = count > 0
        latest = np.maximum(np.where(available, np.arange(len(history))[:, None], -1).max(0), 0)
        last_value = np.where(exists, history[latest, np.arange(len(variables))], median)
        age = np.where(exists, times[origin] - local_times[latest], 24.)
        avg = np.where(exists, np.where(available, history, 0).sum(0, dtype=np.float64) / np.maximum(count, 1), median)
        low = np.where(exists, np.where(available, history, np.inf).min(0), median)
        high = np.where(exists, np.where(available, history, -np.inf).max(0), median)
        variance = np.where(available, (history - avg) ** 2, 0).sum(0) / np.maximum(count, 1)
        duration = times[origin] - local_times[0] + 1.
        features[n] = np.stack(((last_value - mean) / scale, exists, age / 24., (avg - mean) / scale,
            (low - mean) / scale, (high - mean) / scale, count / 24., np.sqrt(variance) / scale,
            np.full(len(variables), duration / 24.)), -1)
        last[n], counts[n], durations[n], lengths[n] = last_value, count, duration, len(history)
    baseline = BaselineData(features, labels, split, episode, last, counts, durations, lengths,
                            mean, scale, median, scaler_counts, train_rows)
    return neural, baseline


def load_trial_checkpoint(path):
    """Load a locally produced tensor-only checkpoint and recreate its model."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualHourlyTrial(state["groups"], state["variables"], TrialConfig(**state["config"]), state["seed"])
    model.load_state_dict(state["state_dict"])
    return model.eval(), state
