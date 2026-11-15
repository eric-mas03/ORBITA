"""Residual identity, full temporal axis, safe full-past attention and reload."""
from dataclasses import asdict
from io import BytesIO

import numpy as np
import pytest
import torch

from astronaut_twin.hourly_architecture_trials import (ResidualHourlyTrial, TrialConfig,
    load_trial_checkpoint, ordered_residual_quantiles, prepare_frozen_trial_data)
from astronaut_twin.hourly_baselines import prepare_baseline_data


GROUPS = {"a": ("x",), "b": ("y",)}
KINDS = ("causal_residual", "window_bidirectional_residual", "group_temporal_residual")


def inputs():
    torch.manual_seed(7)
    values = {g: torch.randn(2, 6, 1) for g in GROUPS}
    observed = {g: torch.ones_like(v, dtype=torch.bool) for g, v in values.items()}
    ages = {g: torch.zeros_like(v) for g, v in values.items()}
    padding = torch.tensor([[False] * 3 + [True] * 3, [False] * 6])
    times = torch.tensor([[0., 1., 2., 0., 0., 0.], [0., 1., 3., 4., 7., 8.]])
    own = torch.randn(2, 2, 9)
    base = torch.stack((-torch.ones(2, 4, 2), torch.zeros(2, 4, 2), torch.ones(2, 4, 2)), -1)
    return (values, observed, ages, times, torch.tensor([1., 4., 8., 24.]), padding), own, base


@pytest.mark.parametrize("kind", KINDS)
def test_zero_residual_is_exact_baseline_and_preserves_temporal_axis(kind):
    torch.set_num_threads(2)
    model = ResidualHourlyTrial(GROUPS, ("x", "y"), TrialConfig(kind=kind, d_model=8, nhead=2, dropout=0)).eval()
    args, own, base = inputs()
    seen = []
    hook = model.core.fusion.register_forward_hook(lambda m, i, o: seen.append(tuple(o.shape)))
    with torch.no_grad():
        prediction = model(args, own, base)
    hook.remove()
    assert torch.equal(prediction, base)
    assert seen == [(2, 6, 8)]


@pytest.mark.parametrize("kind", KINDS)
def test_nonzero_residual_padding_invariant_horizon_independent_and_cutoff_safe(kind):
    torch.set_num_threads(2)
    model = ResidualHourlyTrial(GROUPS, ("x", "y"), TrialConfig(kind=kind, d_model=8, nhead=2, dropout=0)).eval()
    with torch.no_grad():
        for head in model.core.output_layers.values():
            head.weight.normal_(std=.03)
    args, own, base = inputs()
    with torch.no_grad():
        before = model(args, own, base)
        single = model((*args[:4], torch.tensor([8.]), args[-1]), own, base[:, 2:3])
        for value in args[0].values():
            value[0, 3:] = float("nan")
        args[3][0, 3:] = float("nan")
        after = model(args, own, base)
    torch.testing.assert_close(before, after, rtol=0, atol=1e-6)
    torch.testing.assert_close(before[:, 2], single[:, 0], rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="Pretruncate"):
        model(args, own, base, cutoff_index=1)
    assert (torch.diff(after, dim=-1) >= 0).all()


def test_ordered_residual_zero_gaps_and_extreme_corrections():
    base = torch.tensor([[[[-1., 0., 1.], [2., 2., 3.]]]])
    raw = torch.tensor([[[[100., -100., -100.], [-100., 100., 100.]]]])
    output = ordered_residual_quantiles(base, raw, TrialConfig())
    assert torch.isfinite(output).all() and (torch.diff(output, dim=-1) >= 0).all()
    assert output[0, 0, 1, 0] == output[0, 0, 1, 1]
    assert torch.equal(ordered_residual_quantiles(base, torch.zeros_like(raw), TrialConfig()), base)


def test_frozen_preparation_matches_original_features_without_refitting():
    values = np.array([[1., 2.], [3., np.nan], [5., 8.], [10., 20.], [30., 40.], [50., 80.]], np.float32)
    packed = {"values": values, "times": np.array([1., 2., 3.] * 2, np.float32),
        "episode_offsets": np.array([0, 3, 6]), "episode_split": np.array(["train", "dev"]),
        "origins": np.array([1, 4]), "history_start": np.array([0, 3]),
        "target_rows": np.array([[2], [5]]), "horizons": np.array([1]), "variables": np.array(["x", "y"])}
    reference = prepare_baseline_data(packed)
    scalers = {"mean": reference.mean, "scale": reference.scale, "median": reference.median,
               "variables": packed["variables"]}
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    np.testing.assert_array_equal(baseline.features, reference.features)
    np.testing.assert_array_equal(neural.scale, reference.scale)
    # Frozen application does not require any train episode in a new dataset.
    packed["episode_split"][:] = "confirmation"
    heldout, unchanged = prepare_frozen_trial_data(packed, scalers)
    np.testing.assert_array_equal(heldout.mean, reference.mean)
    np.testing.assert_array_equal(unchanged.features, reference.features)


def test_checkpoint_reload_reconstructs_configuration_and_predictions():
    torch.set_num_threads(2)
    config = TrialConfig(kind="window_bidirectional_residual", d_model=8, nhead=2, dropout=0)
    model = ResidualHourlyTrial(GROUPS, ("x", "y"), config).eval()
    path = BytesIO()
    torch.save({"state_dict": model.state_dict(), "config": asdict(config), "groups": GROUPS,
                "variables": ("x", "y"), "seed": 42}, path)
    path.seek(0)
    loaded, state = load_trial_checkpoint(path)
    args, own, base = inputs()
    with torch.no_grad():
        torch.testing.assert_close(model(args, own, base), loaded(args, own, base), rtol=0, atol=0)
