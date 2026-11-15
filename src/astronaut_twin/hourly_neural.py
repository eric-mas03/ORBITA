"""Fair hourly temporal-memory comparison with shared grouped readout design."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .models import AstronautTwin, TwinConfig


class CausalGRUMemory(nn.Module):
    """Adapter with TransformerEncoder's call signature and causal GRU states."""

    def __init__(self, width: int, dropout: float = .1):
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        if src_key_padding_mask is None:
            lengths = torch.full((src.shape[0],), src.shape[1], dtype=torch.long)
        else:
            lengths = (~src_key_padding_mask).sum(1).cpu()
        packed = pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=src.shape[1])
        states = self.norm(self.dropout(states))
        if src_key_padding_mask is not None:
            states = states.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
        return states


def make_hourly_twin(groups: Mapping[str, Sequence[str]], memory_kind: str, seed: int = 42,
                     d_model: int = 16, dropout: float = .1) -> AstronautTwin:
    if memory_kind not in {"gru_shared_readout", "transformer_grouped"}:
        raise ValueError("Unknown temporal memory")
    torch.manual_seed(seed)
    config = TwinConfig({group: len(variables) for group, variables in groups.items()}, groups,
                        d_model=d_model, nhead=2, num_layers=1, decoder_layers=1,
                        dropout=dropout, time_scale=24.)
    model = AstronautTwin(config)
    if memory_kind == "gru_shared_readout":
        model.temporal = CausalGRUMemory(d_model, dropout)
    return model


@dataclass
class NeuralData:
    raw_values: torch.Tensor
    raw_observed: torch.Tensor
    raw_times: torch.Tensor
    origins: torch.Tensor
    history_start: torch.Tensor
    labels: torch.Tensor
    horizons: torch.Tensor
    variables: tuple[str, ...]
    split: np.ndarray
    episode: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    median: np.ndarray
    scaler_counts: np.ndarray
    train_row_indices: np.ndarray

    def batch(self, indices, groups: Mapping[str, Sequence[str]]):
        indices = torch.as_tensor(indices, dtype=torch.long)
        origins, starts = self.origins[indices], self.history_start[indices]
        positions = starts[:, None] + torch.arange(24)[None]
        valid = positions <= origins[:, None]
        rows = torch.minimum(positions, origins[:, None])
        times = self.raw_times[rows]
        observed = self.raw_observed[rows] & valid[..., None]
        values = self.raw_values[rows].masked_fill(~observed, 0)
        # Before the first actual measurement, age means elapsed since context
        # start, accompanied by the measurement mask. No future value is filled.
        last_time = torch.where(observed, times[..., None], times[:, :1, None])
        last_time = torch.cummax(last_time, dim=1).values
        ages = (times[..., None] - last_time).masked_fill(~valid[..., None], 0)
        timestamps = times.masked_fill(~valid, 0)
        group_indices = {group: [self.variables.index(v.removeprefix("SEPSIS2019::")) for v in variables]
                         for group, variables in groups.items()}
        grouped_values = {g: values[:, :, columns] for g, columns in group_indices.items()}
        grouped_observed = {g: observed[:, :, columns] for g, columns in group_indices.items()}
        grouped_ages = {g: ages[:, :, columns] for g, columns in group_indices.items()}
        target = self.labels[indices]
        target_mask = torch.isfinite(target)
        normalized = (target - torch.from_numpy(self.mean)) / torch.from_numpy(self.scale)
        target = normalized.masked_fill(~target_mask, 0)
        args = (grouped_values, grouped_observed, grouped_ages, timestamps, self.horizons, ~valid)
        return args, target, target_mask


def prepare_hourly_neural(packed: Mapping[str, np.ndarray]) -> NeuralData:
    values = np.asarray(packed["values"], dtype=np.float32)
    offsets = np.asarray(packed["episode_offsets"], dtype=np.int64)
    episode_split = np.asarray(packed["episode_split"]).astype(str)
    origins = np.asarray(packed["origins"], dtype=np.int64)
    starts = np.asarray(packed["history_start"], dtype=np.int64)
    episode = np.searchsorted(offsets, origins, side="right") - 1
    train_row_indices = np.concatenate([np.arange(offsets[e], offsets[e + 1])
                                         for e in range(len(episode_split)) if episode_split[e] == "train"])
    if not len(train_row_indices):
        raise ValueError("Training observations required")
    mean, scale, median = np.zeros(values.shape[1]), np.ones(values.shape[1]), np.zeros(values.shape[1])
    counts = np.zeros(values.shape[1], dtype=np.int64)
    for v in range(values.shape[1]):
        selected = values[train_row_indices, v]
        selected = selected[np.isfinite(selected)]
        counts[v] = len(selected)
        if len(selected):
            mean[v], median[v] = np.mean(selected, dtype=np.float64), np.median(selected)
            std = float(np.std(selected, dtype=np.float64))
            scale[v] = std if std > 1e-8 else 1.
    mean, scale, median = (x.astype(np.float32) for x in (mean, scale, median))
    observed = np.isfinite(values)
    normalized = np.where(observed, (values - mean) / scale, 0).astype(np.float32)
    rows = np.asarray(packed["target_rows"], dtype=np.int64)
    labels = values[np.maximum(rows, 0)].copy()
    labels[rows < 0] = np.nan
    times = np.asarray(packed["times"], dtype=np.float32)
    if ((starts < offsets[episode]) | (starts > origins) | (origins >= offsets[episode + 1])).any():
        raise ValueError("History crosses episode")
    if (times[origins] - times[starts] > 23.0001).any():
        raise ValueError("History longer than 24 hourly slots")
    for h, horizon in enumerate(packed["horizons"]):
        valid = rows[:, h] >= 0
        target = rows[valid, h]
        ep = episode[valid]
        if (target < offsets[ep]).any() or (target >= offsets[ep + 1]).any():
            raise ValueError("Target crosses episode")
        if not np.allclose(times[target] - times[origins[valid]], horizon):
            raise ValueError("Target must be at exact horizon")
    return NeuralData(torch.from_numpy(normalized), torch.from_numpy(observed), torch.from_numpy(times),
                      torch.from_numpy(origins), torch.from_numpy(starts), torch.from_numpy(labels),
                      torch.as_tensor(packed["horizons"], dtype=torch.float32),
                      tuple(str(x) for x in packed["variables"]), episode_split[episode], episode,
                      mean, scale, median, counts, train_row_indices)


def flatten_predictions(prediction, groups, variables):
    grouped_order = [v.removeprefix("SEPSIS2019::") for group in groups.values() for v in group]
    if len(set(grouped_order)) != len(variables) or set(grouped_order) != set(variables):
        raise ValueError("Groups must partition the variables exactly once")
    concatenated = torch.cat([prediction[group] for group in groups], dim=2)
    return concatenated[:, :, [grouped_order.index(v) for v in variables]]


def global_cell_loss(prediction, target, observed, global_counts, total_train_origins: int):
    """Unbiased minibatch estimate of the globally balanced variable/horizon loss."""
    counts = torch.as_tensor(global_counts, dtype=prediction.dtype, device=prediction.device)
    active = counts > 0
    if not active.any() or not observed.any():
        return None
    q = prediction.new_tensor([.1, .5, .9])
    error = target[..., None] - prediction
    losses = torch.maximum(q * error, (q - 1) * error).mean(-1)
    sums = (losses * observed).sum(0)
    return ((sums / counts.clamp_min(1))[active].sum() / active.sum()
            * (total_train_origins / prediction.shape[0]))
