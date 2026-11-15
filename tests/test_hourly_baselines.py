"""Mathematical toy sequences test software only; not synthetic physiology."""
import copy

import numpy as np
import pytest
import torch

from astronaut_twin.hourly_baselines import (fit_linear_quantiles, fit_persistence_bands,
    predict_linear, predict_persistence, prepare_baseline_data)


@pytest.fixture
def packed():
    values = np.tile(np.stack((np.arange(1, 9), np.arange(1, 9) ** 2), -1), (4, 1)).astype(np.float32)
    values[1, 1] = np.nan
    origins = np.array([e * 8 + k for e in range(4) for k in (1, 3, 5)])
    return {"values": values, "times": np.tile(np.arange(1, 9), 4).astype(np.float32),
            "episode_offsets": np.arange(0, 33, 8),
            "episode_split": np.array(["train", "dev", "calibration", "test"]),
            "origins": origins, "target_rows": origins[:, None] + np.array([1, 2])[None],
            "history_start": np.repeat(np.arange(0, 32, 8), 3),
            "horizons": np.array([1, 2])}


def test_training_only_scaler_and_causal_features(packed):
    before = prepare_baseline_data(packed)
    changed = copy.deepcopy(packed)
    changed["values"][8:] += 10000
    after = prepare_baseline_data(changed)
    np.testing.assert_array_equal(before.mean, after.mean)
    np.testing.assert_array_equal(before.median, after.median)
    np.testing.assert_array_equal(before.scale, after.scale)
    np.testing.assert_array_equal(before.features[:3], after.features[:3])
    # Changing future train rows cannot enter historical summaries at the first origin.
    changed = copy.deepcopy(packed)
    changed["values"][2:8] += 100
    # Keep scaler fixed conceptually by checking raw last/age/count, not z values.
    updated = prepare_baseline_data(changed)
    np.testing.assert_array_equal(before.last[0], updated.last[0])
    np.testing.assert_array_equal(before.counts[0], updated.counts[0])
    np.testing.assert_array_equal(before.features[0, :, 2], updated.features[0, :, 2])
    assert len(before.train_row_indices) == 8


def test_missing_own_history_uses_only_training_median(packed):
    packed["values"][8:10, 1] = np.nan
    data = prepare_baseline_data(packed)
    index = 3  # First dev origin, no own-variable history in channel 1.
    assert data.counts[index, 1] == 0
    assert data.last[index, 1] == data.median[1]
    assert data.features[index, 1, 1] == 0
    assert data.features[index, 1, 2] == 1
    assert np.isfinite(data.features).all()
    assert data.history_hours[index] == 2


def test_residual_bands_never_use_holdouts_and_keep_literal_persistence(packed):
    data = prepare_baseline_data(packed)
    original = fit_persistence_bands(data, minimum_count=2)
    changed = copy.deepcopy(data)
    changed.labels[changed.split != "train"] += 100000
    altered = fit_persistence_bands(changed, minimum_count=2)
    np.testing.assert_array_equal(original["bands"], altered["bands"])
    prediction = predict_persistence(data, original)
    np.testing.assert_array_equal(prediction[..., 1], np.broadcast_to(data.last[:, None], prediction.shape[:-1]))
    assert (prediction[..., 0] <= prediction[..., 1]).all()
    assert (prediction[..., 1] <= prediction[..., 2]).all()
    assert original["horizon_counts"].shape == (2, 2)


def test_training_and_selection_independent_of_test_and_calibration_labels(packed):
    torch.set_num_threads(2)
    data = prepare_baseline_data(packed)
    first = fit_linear_quantiles(data, learning_rates=(.01, .03), epochs=6)
    changed = copy.deepcopy(data)
    changed.labels[np.isin(changed.split, ["calibration", "test"])] += 100000
    second = fit_linear_quantiles(changed, learning_rates=(.01, .03), epochs=6)
    np.testing.assert_array_equal(first["weight"], second["weight"])
    np.testing.assert_array_equal(first["bias"], second["bias"])
    assert first["selection"] == second["selection"]
    bands = fit_persistence_bands(data)
    prediction = predict_linear(data, first, bands)
    assert prediction.shape == (12, 2, 2, 3)
    assert np.isfinite(prediction).all()
    assert (np.diff(prediction, axis=-1) >= 0).all()


def test_masked_linear_outputs_fallback_if_no_train_labels(packed):
    torch.set_num_threads(2)
    data = prepare_baseline_data(packed)
    data.labels[data.split == "train", :, 1] = np.nan
    artifact = fit_linear_quantiles(data, learning_rates=(.01,), epochs=3)
    assert (artifact["train_target_counts"][:, 1] == 0).all()
    bands = fit_persistence_bands(data)
    linear = predict_linear(data, artifact, bands)
    persistence = predict_persistence(data, bands)
    np.testing.assert_array_equal(linear[:, :, 1], persistence[:, :, 1])


def test_invalid_episode_or_target_alignment_rejected(packed):
    packed["target_rows"][0, 0] = 8
    with pytest.raises(ValueError, match="boundary"):
        prepare_baseline_data(packed)
