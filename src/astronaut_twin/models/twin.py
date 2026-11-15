"""Small configurable causal encoders and multi-horizon Transformer heads."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TwinConfig:
    input_groups: Mapping[str, int]
    output_groups: Mapping[str, Sequence[str]]
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    decoder_layers: int = 1
    dropout: float = 0.1
    time_scale: float = 1.0

    def __post_init__(self):
        if not self.input_groups or not self.output_groups:
            raise ValueError("Input and output groups cannot be empty")
        if any(not isinstance(n, int) or n <= 0 for n in self.input_groups.values()):
            raise ValueError("Each input group needs a positive feature count")
        for name in (*self.input_groups, *self.output_groups):
            if not name or "." in name:
                raise ValueError("Group names must be nonempty and cannot contain '.'")
        targets = [target for group in self.output_groups.values() for target in group]
        if any(not group for group in self.output_groups.values()) or len(set(targets)) != len(targets):
            raise ValueError("Output targets must be nonempty and globally unique")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers < 1 or self.decoder_layers < 1:
            raise ValueError("At least one temporal and decoder layer is required")
        if not 0 <= self.dropout < 1 or not isfinite(self.time_scale) or not self.time_scale > 0:
            raise ValueError("Invalid dropout or time_scale")


class TimeFeatures(nn.Module):
    """Continuous elapsed-time features, in units fixed by config.time_scale."""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("frequencies", torch.logspace(-3, 1, 8))
        self.projection = nn.Linear(17, width)

    def forward(self, time: Tensor) -> Tensor:
        phase = time.unsqueeze(-1) * self.frequencies
        logged = torch.sign(time) * torch.log1p(time.abs())
        return self.projection(torch.cat((logged.unsqueeze(-1), phase.sin(), phase.cos()), -1))


class AstronautTwin(nn.Module):
    """Group encoders -> common causal temporal latent -> Transformer heads.

    Dictionary outputs are [batch, horizon, target_in_group, quantile], where
    quantiles are ordered (0.1, 0.5, 0.9). All times must share one unit.
    Available-time filtering and train-only preprocessing belong upstream.
    """

    def __init__(self, config: TwinConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.group_encoders = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(3 * count, d), nn.GELU(), nn.Linear(d, d))
            for name, count in config.input_groups.items()
        })
        self.fusion = nn.Linear(len(config.input_groups) * (d + 1), d)
        self.time_features = TimeFeatures(d)
        layer = nn.TransformerEncoderLayer(
            d, config.nhead, 4 * d, config.dropout, batch_first=True
        )
        self.temporal = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.decoders = nn.ModuleDict({
            name: nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d, config.nhead, 4 * d, config.dropout, batch_first=True
                ), config.decoder_layers
            ) for name in config.output_groups
        })
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(d, len(targets) * 3)
            for name, targets in config.output_groups.items()
        })
        # Transformer clones otherwise start with equal parameter values.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def encode_history(
        self,
        values: Mapping[str, Tensor],
        observed: Mapping[str, Tensor],
        ages: Mapping[str, Tensor],
        timestamps: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if timestamps.ndim != 2 or not timestamps.is_floating_point():
            raise ValueError("timestamps must be a floating [B,T] tensor")
        batch, steps = timestamps.shape
        if not batch or not steps:
            raise ValueError("History cannot be empty")
        if padding_mask is None:
            padding_mask = torch.zeros_like(timestamps, dtype=torch.bool)
        if padding_mask.shape != timestamps.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean [B,T]")
        if padding_mask[:, 0].any() or (padding_mask[:, :-1] & ~padding_mask[:, 1:]).any():
            raise ValueError("Padding must be on the right and every subject needs history")
        valid = ~padding_mask
        if not torch.isfinite(timestamps[valid]).all():
            raise ValueError("Valid timestamps must be finite")
        pairs = valid[:, 1:] & valid[:, :-1]
        differences = timestamps[:, 1:] - timestamps[:, :-1]
        if (differences[pairs] < 0).any():
            raise ValueError("Valid history must be chronologically ordered")
        expected = set(self.config.input_groups)
        if any(set(mapping) != expected for mapping in (values, observed, ages)):
            raise ValueError("Input dictionaries must exactly match configured groups")
        tokens = []
        for name, count in self.config.input_groups.items():
            value, mask, age = values[name], observed[name], ages[name]
            shape = (batch, steps, count)
            if value.shape != shape or mask.shape != shape or age.shape != shape:
                raise ValueError(f"Incorrect feature shape for {name}; expected {shape}")
            if mask.dtype != torch.bool or not value.is_floating_point() or not age.is_floating_point():
                raise ValueError("Values and ages must float; observed masks must be boolean")
            mask = mask & valid.unsqueeze(-1)
            if not torch.isfinite(value[mask]).all():
                raise ValueError("Observed values must be finite")
            valid_age = age[valid.unsqueeze(-1).expand_as(age)]
            if not torch.isfinite(valid_age).all() or (valid_age < 0).any():
                raise ValueError("Non-padding ages must be finite and nonnegative")
            safe_value = torch.where(mask, value, torch.zeros_like(value))
            safe_age = torch.where(valid.unsqueeze(-1), age, torch.zeros_like(age))
            present = mask.any(-1, keepdim=True)
            encoded = self.group_encoders[name](torch.cat((
                safe_value, mask.to(value.dtype), torch.log1p(safe_age / self.config.time_scale)
            ), -1))
            # No encoder bias is presented as a measurement of an absent group.
            tokens.extend((encoded * present, present.to(value.dtype)))
        fused = self.fusion(torch.cat(tokens, -1))
        elapsed = torch.where(valid, timestamps - timestamps[:, :1], torch.zeros_like(timestamps))
        fused = fused + self.time_features(elapsed / self.config.time_scale)
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=fused.device), 1)
        memory = self.temporal(fused, mask=causal, src_key_padding_mask=padding_mask)
        return memory.masked_fill(padding_mask.unsqueeze(-1), 0), padding_mask

    def forward(
        self,
        values: Mapping[str, Tensor],
        observed: Mapping[str, Tensor],
        ages: Mapping[str, Tensor],
        timestamps: Tensor,
        horizons: Tensor,
        padding_mask: Tensor | None = None,
        cutoff_index: int | Tensor | None = None,
    ) -> dict[str, Tensor]:
        memory, padding = self.encode_history(values, observed, ages, timestamps, padding_mask)
        batch, steps, _ = memory.shape
        if cutoff_index is None:
            cutoff = (~padding).sum(1) - 1
        elif isinstance(cutoff_index, int):
            cutoff = torch.full((batch,), cutoff_index, dtype=torch.long, device=memory.device)
        else:
            cutoff = cutoff_index.to(device=memory.device)
        if cutoff.shape != (batch,) or cutoff.dtype != torch.long:
            raise ValueError("cutoff_index must be int or long [B]")
        if (cutoff < 0).any() or (cutoff >= steps).any():
            raise ValueError("cutoff_index outside history")
        row = torch.arange(batch, device=memory.device)
        if padding[row, cutoff].any():
            raise ValueError("cutoff_index cannot point to padding")
        horizons = horizons.to(device=memory.device, dtype=memory.dtype)
        if horizons.ndim == 1:
            horizons = horizons.unsqueeze(0).expand(batch, -1)
        if horizons.ndim != 2 or horizons.shape[0] != batch or not horizons.shape[1]:
            raise ValueError("horizons must have shape [H] or [B,H]")
        if not torch.isfinite(horizons).all() or (horizons <= 0).any():
            raise ValueError("Horizons must be finite and positive")
        origin = timestamps[row, cutoff] - timestamps[:, 0]
        query = self.time_features((origin[:, None] + horizons) / self.config.time_scale)
        query = query + self.time_features(horizons / self.config.time_scale)
        excluded = padding | (torch.arange(steps, device=memory.device)[None, :] > cutoff[:, None])
        # Horizon queries are known covariates, but allowing them to attend one
        # another would make q(t+h) depend on which *other* horizons were asked
        # for. Diagonal-only self-attention keeps each query independent.
        query_mask = ~torch.eye(horizons.shape[1], dtype=torch.bool, device=memory.device)
        result = {}
        for name, targets in self.config.output_groups.items():
            decoded = self.decoders[name](
                query, memory, tgt_mask=query_mask, memory_key_padding_mask=excluded
            )
            raw = self.output_layers[name](decoded).reshape(batch, horizons.shape[1], len(targets), 3)
            median = raw[..., 1]
            result[name] = torch.stack((median - F.softplus(raw[..., 0]), median,
                                        median + F.softplus(raw[..., 2])), -1)
        return result


def masked_pinball_loss(
    prediction: Tensor,
    target: Tensor,
    observed: Tensor,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> Tensor:
    """Mean over observed target/quantile pairs; rejects an empty objective."""
    if prediction.shape[:-1] != target.shape or observed.shape != target.shape:
        raise ValueError("Prediction, target and mask shapes do not agree")
    if observed.dtype != torch.bool or prediction.shape[-1] != len(quantiles):
        raise ValueError("Invalid observed mask or quantile count")
    q = prediction.new_tensor(quantiles)
    if not torch.isfinite(q).all() or ((q <= 0) | (q >= 1)).any():
        raise ValueError("Quantiles must be finite and strictly between 0 and 1")
    if not observed.any():
        raise ValueError("Cannot score a batch without observed targets")
    selected_target = target[observed]
    selected_prediction = prediction[observed]
    if not torch.isfinite(selected_target).all() or not torch.isfinite(selected_prediction).all():
        raise ValueError("Observed targets and predictions must be finite")
    error = selected_target.unsqueeze(-1) - selected_prediction
    return torch.maximum(q * error, (q - 1) * error).mean()


@dataclass(frozen=True)
class GroupedPinballResult:
    """Explicit no-label result: a missing loss means skip the optimizer step."""

    loss: Tensor | None
    observed_counts: Mapping[str, int]
    active_heads: tuple[str, ...]
    reduction: str

    @property
    def should_step(self) -> bool:
        return self.loss is not None


def grouped_masked_pinball_loss(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    observed: Mapping[str, Tensor],
    reduction: str = "equal_head",
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> GroupedPinballResult:
    """Aggregate only labeled heads, with no autograd link to empty heads.

    equal_head: each head with >=1 label receives equal weight.
    per_target: every observed scalar target at a sample/horizon receives equal
    weight, so heads with more observed entries contribute proportionally.
    An all-missing batch returns loss=None: do not backward or optimizer.step.
    """
    if reduction not in {"equal_head", "per_target"}:
        raise ValueError("reduction must be 'equal_head' or 'per_target'")
    if not predictions or set(predictions) != set(targets) or set(predictions) != set(observed):
        raise ValueError("Nonempty head mappings must have exactly the same keys")
    if not quantiles or any(not isfinite(q) or not 0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be finite and strictly between 0 and 1")
    counts, losses = {}, {}
    for head, prediction in predictions.items():
        target, mask = targets[head], observed[head]
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("Grouped predictions must be [B,H,K,Q] and targets [B,H,K]")
        if prediction.shape[:-1] != target.shape or target.shape != mask.shape:
            raise ValueError(f"Prediction, target and mask shapes disagree for {head}")
        if mask.dtype != torch.bool or prediction.shape[-1] != len(quantiles):
            raise ValueError(f"Invalid observed mask or quantile count for {head}")
        counts[head] = int(mask.sum().item())
        if counts[head]:
            losses[head] = masked_pinball_loss(prediction, target, mask, quantiles)
    if not losses:
        return GroupedPinballResult(None, counts, (), reduction)
    if reduction == "equal_head":
        loss = torch.stack(list(losses.values())).mean()
    else:
        count = sum(counts.values())
        loss = torch.stack([value * (counts[head] / count) for head, value in losses.items()]).sum()
    return GroupedPinballResult(loss, counts, tuple(losses), reduction)
