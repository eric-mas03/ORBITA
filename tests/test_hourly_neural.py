"""Software invariants for the controlled temporal-memory comparison."""
import numpy as np
import pytest
import torch

from astronaut_twin.hourly_neural import (flatten_predictions, global_cell_loss,
                                        make_hourly_twin, prepare_hourly_neural)


GROUPS = {"a": ("x",), "b": ("y",)}


@pytest.mark.parametrize("kind", ["gru_shared_readout", "transformer_grouped"])
def test_both_memories_causal_and_independent_of_other_horizons(kind):
    torch.set_num_threads(2)
    model = make_hourly_twin(GROUPS, kind, d_model=8, dropout=0).eval()
    torch.manual_seed(3)
    values = {group: torch.randn(2, 4, 1) for group in GROUPS}
    observed = {group: torch.ones_like(value, dtype=torch.bool) for group, value in values.items()}
    ages = {group: torch.zeros_like(value) for group, value in values.items()}
    time = torch.tensor([[1., 2., 3., 4.], [0., 1., 4., 8.]])
    with torch.no_grad():
        before = model(values, observed, ages, time, torch.tensor([4.]), cutoff_index=1)
        multiple = model(values, observed, ages, time, torch.tensor([1., 4., 24.]), cutoff_index=1)
        for value in values.values():
            value[:, 2:] += 100
        time[:, 2:] += 100
        after = model(values, observed, ages, time, torch.tensor([4.]), cutoff_index=1)
    for group in GROUPS:
        torch.testing.assert_close(before[group], after[group], rtol=0, atol=1e-6)
        torch.testing.assert_close(before[group][:, 0], multiple[group][:, 1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", ["gru_shared_readout", "transformer_grouped"])
def test_both_memories_ignore_right_padding_and_match_individual(kind):
    torch.set_num_threads(2)
    model = make_hourly_twin(GROUPS, kind, d_model=8, dropout=0).eval()
    values = {group: torch.arange(8.).reshape(2, 4, 1) for group in GROUPS}
    observed = {group: torch.ones_like(value, dtype=torch.bool) for group, value in values.items()}
    ages = {group: torch.zeros_like(value) for group, value in values.items()}
    time = torch.tensor([[1., 2., 0., 0.], [0., 1., 4., 8.]])
    padding = torch.tensor([[False, False, True, True], [False] * 4])
    with torch.no_grad():
        output = model(values, observed, ages, time, torch.tensor([1., 4.]), padding)
        individual = model({g: x[:1, :2] for g, x in values.items()},
                           {g: x[:1, :2] for g, x in observed.items()},
                           {g: x[:1, :2] for g, x in ages.items()}, time[:1, :2], torch.tensor([1., 4.]))
    for group in GROUPS:
        torch.testing.assert_close(output[group][:1], individual[group], rtol=0, atol=1e-6)
        assert (torch.diff(output[group], dim=-1) >= 0).all()


def test_initial_shared_parameters_exactly_match_across_memories():
    gru = make_hourly_twin(GROUPS, "gru_shared_readout")
    transformer = make_hourly_twin(GROUPS, "transformer_grouped")
    for name, parameter in gru.named_parameters():
        if not name.startswith("temporal."):
            torch.testing.assert_close(parameter, dict(transformer.named_parameters())[name], rtol=0, atol=0)


def test_global_cell_loss_matches_full_macro_under_minibatch_partition():
    prediction = torch.zeros(4, 2, 2, 3, requires_grad=True)
    target = torch.arange(16.).reshape(4, 2, 2)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 1, 1] = False
    mask[1:, 0, 1] = False
    counts = mask.sum(0)
    full = global_cell_loss(prediction, target, mask, counts, 4)
    split_a = global_cell_loss(prediction[:1], target[:1], mask[:1], counts, 4)
    split_b = global_cell_loss(prediction[1:], target[1:], mask[1:], counts, 4)
    torch.testing.assert_close(full, split_a / 4 + split_b * 3 / 4)
    expected = torch.stack([target[:, h, v][mask[:, h, v]].mean() * .5
                            for h, v in ((0, 0), (0, 1), (1, 0))]).mean()
    torch.testing.assert_close(full, expected)
    full.backward()
    assert (prediction.grad[~mask] == 0).all()
    assert global_cell_loss(prediction, target, torch.zeros_like(mask), counts, 4) is None


def test_batch_ages_masks_padding_target_order_and_gradient_alignment():
    raw = np.array([[1., np.nan], [2., 4.], [3., np.nan], [4., 8.], [5., 9.], [6., 10.]], dtype=np.float32)
    packed = {"values": raw, "times": np.array([1, 2, 3, 1, 2, 3], dtype=np.float32),
              "episode_offsets": np.array([0, 3, 6]), "episode_split": np.array(["train", "dev"]),
              "origins": np.array([1, 4]), "history_start": np.array([0, 3]),
              "target_rows": np.array([[2], [5]]), "horizons": np.array([1]), "variables": np.array(["x", "y"])}
    data = prepare_hourly_neural(packed)
    args, target, mask = data.batch([0, 1], GROUPS)
    assert args[-1].shape == (2, 24)
    assert args[-1][:, 2:].all()
    assert not args[-1][:, :2].any()
    assert not args[1]["b"][0, 0, 0]
    assert args[2]["b"][0, 1, 0] == 0
    assert not mask[0, 0, 1]
    assert target.shape == (2, 1, 2)
    first = torch.ones(2, 1, 1, 3, requires_grad=True)
    second = torch.full((2, 1, 1, 3), 2., requires_grad=True)
    flat = flatten_predictions({"b": second, "a": first}, {"b": ("y",), "a": ("x",)}, ("x", "y"))
    assert (flat[:, :, 0] == 1).all() and (flat[:, :, 1] == 2).all()
    flat.sum().backward()
    assert first.grad.sum() > 0 and second.grad.sum() > 0
