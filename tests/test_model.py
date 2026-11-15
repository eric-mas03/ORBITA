"""Synthetic tensors exercise mechanics only, never physiological validity."""

import pytest
import torch

from astronaut_twin.models import (AstronautTwin, TwinConfig,
                                  grouped_masked_pinball_loss, masked_pinball_loss)


@pytest.fixture
def example():
    torch.manual_seed(17)
    config = TwinConfig({"chemistry": 2, "immune": 1}, {"metabolic": ("a", "b"), "immune": ("c",)},
                        d_model=8, nhead=2, dropout=0)
    model = AstronautTwin(config).eval()
    times = torch.tensor([[0., 1., 4., 8.], [0., 2., 3., 9.]])
    values = {"chemistry": torch.randn(2, 4, 2), "immune": torch.randn(2, 4, 1)}
    observed = {key: torch.ones_like(value, dtype=torch.bool) for key, value in values.items()}
    ages = {key: torch.zeros_like(value) for key, value in values.items()}
    return model, values, observed, ages, times


def test_forward_shapes_order_and_gradient(example):
    model, values, observed, ages, times = example
    output = model(values, observed, ages, times, torch.tensor([1., 3.]))
    assert output["metabolic"].shape == (2, 2, 2, 3)
    assert output["immune"].shape == (2, 2, 1, 3)
    for prediction in output.values():
        assert torch.isfinite(prediction).all()
        assert (prediction[..., 0] <= prediction[..., 1]).all()
        assert (prediction[..., 1] <= prediction[..., 2]).all()
    prediction = output["metabolic"]
    target = torch.zeros_like(prediction[..., 0])
    loss = masked_pinball_loss(prediction, target, torch.ones_like(target, dtype=torch.bool))
    loss.backward()
    assert model.group_encoders["chemistry"][0].weight.grad.abs().sum() > 0
    assert model.decoders["metabolic"].layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0


def test_future_values_and_timestamps_cannot_change_past_forecast(example):
    model, values, observed, ages, times = example
    horizons = torch.tensor([1., 3.])
    before = model(values, observed, ages, times, horizons, cutoff_index=1)
    memory_before, _ = model.encode_history(values, observed, ages, times)
    changed = {key: value.clone() for key, value in values.items()}
    for value in changed.values():
        value[:, 2:] += 100
    times_changed = times.clone()
    times_changed[:, 2:] += 100
    after = model(changed, observed, ages, times_changed, horizons, cutoff_index=1)
    memory_after, _ = model.encode_history(changed, observed, ages, times_changed)
    torch.testing.assert_close(memory_before[:, :2], memory_after[:, :2], rtol=0, atol=1e-6)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=1e-6)
    # A truncated history must agree with explicitly excluding its future.
    truncated = model({key: value[:, :2] for key, value in values.items()},
                      {key: value[:, :2] for key, value in observed.items()},
                      {key: value[:, :2] for key, value in ages.items()}, times[:, :2], horizons)
    for key in before:
        torch.testing.assert_close(before[key], truncated[key], rtol=0, atol=1e-6)


def test_padding_and_missing_groups_ignore_arbitrary_values(example):
    model, values, observed, ages, times = example
    padding = torch.tensor([[False, False, True, True], [False, False, False, False]])
    observed["immune"][:] = False
    baseline = model(values, observed, ages, times, torch.tensor([2.]), padding)
    values["immune"][:] = float("nan")
    for value in values.values():
        value[0, 2:] = float("nan")
    for age in ages.values():
        age[0, 2:] = float("nan")
    times[0, 2:] = float("nan")
    actual = model(values, observed, ages, times, torch.tensor([2.]), padding)
    for key in actual:
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(baseline[key], actual[key])


def test_irregular_time_and_age_are_used(example):
    model, values, observed, ages, times = example
    baseline = model(values, observed, ages, times, torch.tensor([2.]))["metabolic"]
    new_times = times * 3
    time_output = model(values, observed, ages, new_times, torch.tensor([2.]))["metabolic"]
    assert not torch.allclose(baseline, time_output)
    ages["chemistry"][:] = 4
    age_output = model(values, observed, ages, times, torch.tensor([2.]))["metabolic"]
    assert not torch.allclose(baseline, age_output)


def test_masked_pinball_ignores_nan_and_has_expected_gradient():
    prediction = torch.tensor([[[0., 0., 0.], [float("nan")] * 3]], requires_grad=True)
    target = torch.tensor([[1., float("nan")]])
    mask = torch.tensor([[True, False]])
    loss = masked_pinball_loss(prediction, target, mask)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    torch.testing.assert_close(prediction.grad[0, 1], torch.zeros(3))
    assert torch.isfinite(prediction.grad).all()
    with pytest.raises(ValueError, match="without observed"):
        masked_pinball_loss(prediction, target, torch.zeros_like(mask))


def test_reject_invalid_history_and_horizon(example):
    model, values, observed, ages, times = example
    with pytest.raises(ValueError, match="right"):
        model(values, observed, ages, times, torch.tensor([1.]),
              torch.tensor([[False, True, False, False], [False] * 4]))
    with pytest.raises(ValueError, match="positive"):
        model(values, observed, ages, times, torch.tensor([-1.]))
    with pytest.raises(ValueError, match="chronologically"):
        model(values, observed, ages, times.flip(1), torch.tensor([1.]))


def test_translation_of_absolute_time_does_not_change_prediction(example):
    model, values, observed, ages, times = example
    before = model(values, observed, ages, times, torch.tensor([1.]))
    after = model(values, observed, ages, times + 1000, torch.tensor([1.]))
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_horizon_prediction_independent_of_other_requested_horizons(example):
    model, values, observed, ages, times = example
    single = model(values, observed, ages, times, torch.tensor([3.]))
    multiple = model(values, observed, ages, times, torch.tensor([1., 3., 7.]))
    reordered = model(values, observed, ages, times, torch.tensor([7., 1., 3.]))
    for key in single:
        torch.testing.assert_close(single[key][:, 0], multiple[key][:, 1], rtol=0, atol=1e-6)
        torch.testing.assert_close(single[key][:, 0], reordered[key][:, 2], rtol=0, atol=1e-6)


def test_mixed_batch_cutoffs_and_padding_equal_individual_histories(example):
    model, values, observed, ages, times = example
    padding = torch.tensor([[False, False, False, True], [False, False, False, False]])
    cutoffs = torch.tensor([1, 3])
    horizons = torch.tensor([[1., 5.], [2., 4.]])
    batched = model(values, observed, ages, times, horizons, padding, cutoff_index=cutoffs)
    for row, cutoff in enumerate(cutoffs.tolist()):
        end = cutoff + 1
        individual = model(
            {key: value[row:row + 1, :end] for key, value in values.items()},
            {key: value[row:row + 1, :end] for key, value in observed.items()},
            {key: value[row:row + 1, :end] for key, value in ages.items()},
            times[row:row + 1, :end], horizons[row:row + 1],
        )
        for key in batched:
            torch.testing.assert_close(batched[key][row:row + 1], individual[key], rtol=0, atol=1e-6)


def test_grouped_loss_reductions_and_explicit_empty_batch():
    prediction = {"a": torch.zeros(1, 1, 1, 3, requires_grad=True),
                  "b": torch.zeros(1, 1, 3, 3, requires_grad=True)}
    target = {"a": torch.ones(1, 1, 1), "b": torch.full((1, 1, 3), 3.)}
    mask = {key: torch.ones_like(value, dtype=torch.bool) for key, value in target.items()}
    equal = grouped_masked_pinball_loss(prediction, target, mask, "equal_head")
    pooled = grouped_masked_pinball_loss(prediction, target, mask, "per_target")
    torch.testing.assert_close(equal.loss, torch.tensor(1.))
    torch.testing.assert_close(pooled.loss, torch.tensor(1.25))
    assert equal.observed_counts == {"a": 1, "b": 3}
    assert equal.active_heads == ("a", "b") and equal.should_step
    empty = grouped_masked_pinball_loss(prediction, target,
                                        {key: torch.zeros_like(value) for key, value in mask.items()})
    assert empty.loss is None and not empty.should_step and not empty.active_heads
    assert all(value.grad is None for value in prediction.values())


def test_grouped_loss_masks_individual_samples_horizons_and_targets():
    prediction = torch.zeros(2, 3, 2, 3, requires_grad=True)
    target = torch.full((2, 3, 2), float("nan"))
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 1, 0] = True
    mask[1, 2, 1] = True
    target[mask] = 1.
    result = grouped_masked_pinball_loss({"a": prediction}, {"a": target}, {"a": mask})
    result.loss.backward()
    assert (prediction.grad[mask] != 0).all()
    assert (prediction.grad[~mask] == 0).all()
    assert result.observed_counts == {"a": 2}


def test_missing_head_has_no_gradient_but_shared_encoder_updates(example):
    model, values, observed, ages, times = example
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    horizons = torch.tensor([1., 3.])
    # Warm up both heads so Adam has momentum state before one head disappears.
    output = model(values, observed, ages, times, horizons)
    target = {key: torch.zeros_like(value[..., 0]) for key, value in output.items()}
    mask = {key: torch.ones_like(value, dtype=torch.bool) for key, value in target.items()}
    grouped_masked_pinball_loss(output, target, mask).loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    absent_parameters = list(model.decoders["immune"].parameters()) + list(model.output_layers["immune"].parameters())
    snapshot = [parameter.detach().clone() for parameter in absent_parameters]
    shared_before = model.fusion.weight.detach().clone()
    output = model(values, observed, ages, times, horizons)
    mask["immune"][:] = False
    target["immune"][:] = float("nan")
    result = grouped_masked_pinball_loss(output, target, mask)
    result.loss.backward()
    assert result.active_heads == ("metabolic",)
    assert all(parameter.grad is None for parameter in absent_parameters)
    assert model.fusion.weight.grad.abs().sum() > 0
    # Its measurements remain useful inputs even when its target labels vanish.
    assert model.group_encoders["immune"][0].weight.grad.abs().sum() > 0
    optimizer.step()
    for before, after in zip(snapshot, absent_parameters):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.equal(shared_before, model.fusion.weight)


def test_adamw_zero_gradient_can_update_unlabeled_parameter_but_none_skips():
    # This isolates the optimizer behavior behind the set_to_none requirement.
    parameter = torch.nn.Parameter(torch.tensor([1., 1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.1)
    parameter.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    before_zero = parameter.detach().clone()
    optimizer.step()  # Zero gradient still has Adam momentum and weight decay.
    assert not torch.equal(before_zero, parameter)
    optimizer.zero_grad(set_to_none=True)
    before_none = parameter.detach().clone()
    optimizer.step()
    torch.testing.assert_close(before_none, parameter, rtol=0, atol=0)


def test_unobserved_target_rows_have_zero_gradient_but_can_change_with_decay(example):
    model, values, observed, ages, times = example
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    output = model(values, observed, ages, times, torch.tensor([1.]))
    target = {key: torch.zeros_like(value[..., 0]) for key, value in output.items()}
    mask = {key: torch.zeros_like(value, dtype=torch.bool) for key, value in target.items()}
    mask["metabolic"][..., 0] = True
    result = grouped_masked_pinball_loss(output, target, mask)
    result.loss.backward()
    weight = model.output_layers["metabolic"].weight
    assert (weight.grad[3:6] == 0).all()
    before = weight[3:6].detach().clone()
    optimizer.step()
    assert not torch.equal(before, weight[3:6])


def test_grouped_loss_rejects_bad_head_keys_and_shape():
    pred = {"a": torch.zeros(1, 2, 1, 3)}
    target = {"a": torch.zeros(1, 2, 1)}
    mask = {"a": torch.zeros(1, 2, 1, dtype=torch.bool)}
    with pytest.raises(ValueError, match="keys"):
        grouped_masked_pinball_loss(pred, {}, mask)
    with pytest.raises(ValueError, match="reduction"):
        grouped_masked_pinball_loss(pred, target, mask, "anything")
    with pytest.raises(ValueError, match="shapes"):
        grouped_masked_pinball_loss(pred, target, {"a": torch.zeros(1, dtype=torch.bool)})
