"""Audit preparation/batching on real OSD-575; tensor mutations test masks only."""

from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_exploratory_audit", ROOT / "scripts/train_exploratory.py")
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


@pytest.fixture(scope="module")
def prepared():
    wide = pd.read_csv(ROOT / "data/processed/inspiration4_wide.csv").set_index(
        ["subject_id", "visit_index", "timepoint"]
    ).sort_index()
    groups = json.loads((ROOT / "config/biomedical_groups.json").read_text(encoding="utf-8"))
    return wide, groups, training.prepare(wide, groups)


def test_real_partition_sizes_and_target_cutoffs(prepared):
    _, _, setup = prepared
    assert Counter(sample["split"] for sample in setup["samples"]) == {"train": 16, "val": 4, "test": 4}
    assert setup["training_scaler_rows"] == 20
    allowed = {"train": {"L-44", "L-3", "R+45"}, "val": {"R+82"}, "test": {"R+194"}}
    assert len({sample["subject_id"] for sample in setup["samples"]}) == 4
    for sample in setup["samples"]:
        assert sample["target_visit"] in allowed[sample["split"]]
        assert sample["target_visit"] != sample["origin"]
        assert len(sample["times"]) == len(sample["history"])
        assert sample["times"][-1] == int(sample["origin"][1:])
        # The fitted snapshot exists only after R+45; selecting it on R+82
        # means the test forecast can first be issued after that measurement.
        if sample["split"] == "val":
            assert sample["origin"] == "R+45"
        if sample["split"] == "test":
            assert sample["origin"] == "R+82"


def test_holdout_values_cannot_change_training_scaler_or_training_windows(prepared):
    wide, groups, original = prepared
    changed = wide.copy()
    holdout = changed.index.get_level_values("timepoint").isin(["R+82", "R+194"])
    changed.loc[holdout, original["features"]] = changed.loc[holdout, original["features"]] * 1000 + 123456
    altered = training.prepare(changed, groups)
    pd.testing.assert_series_equal(original["mean"], altered["mean"])
    pd.testing.assert_series_equal(original["scale"], altered["scale"])
    for before, after in zip(original["samples"], altered["samples"]):
        if before["split"] == "train":
            np.testing.assert_array_equal(before["history"], after["history"])
            np.testing.assert_array_equal(before["target"], after["target"])


def test_real_histories_never_cross_reference_or_include_target(prepared):
    wide, _, setup = prepared
    for sample in setup["samples"]:
        reference = sample["origin"][0]
        assert sample["target_visit"].startswith(reference)
        frame = wide.xs(sample["subject_id"], level="subject_id").sort_index()
        visits = frame.index.get_level_values("timepoint")
        origin_day = int(sample["origin"][1:])
        selected = [visit for visit in visits if visit.startswith(reference) and int(visit[1:]) <= origin_day]
        assert sample["target_visit"] not in selected
        np.testing.assert_array_equal(sample["times"], np.array([int(v[1:]) for v in selected]))
        expected = (frame.loc[frame.index.get_level_values("timepoint").isin(selected), setup["features"]]
                    - setup["mean"]) / setup["scale"]
        np.testing.assert_allclose(sample["history"], expected.to_numpy(dtype=np.float32), equal_nan=True)
        expected_target = (frame.xs(sample["target_visit"], level="timepoint")[setup["features"]].iloc[0]
                           - setup["mean"]) / setup["scale"]
        np.testing.assert_allclose(sample["target"], expected_target.to_numpy(dtype=np.float32), equal_nan=True)
        if sample["split"] in {"train", "val"}:
            assert "R+82" not in selected and "R+194" not in selected
        else:
            assert "R+194" not in selected


def test_horizons_are_real_positive_day_differences_within_reference(prepared):
    _, _, setup = prepared
    for sample in setup["samples"]:
        assert sample["horizon"] > 0
        expected = int(sample["target_visit"][1:]) - int(sample["origin"][1:])
        assert sample["horizon"] == expected
        assert np.all(np.diff(sample["times"]) > 0)
    test_horizons = [sample["horizon"] for sample in setup["samples"] if sample["split"] == "test"]
    train_horizons = [sample["horizon"] for sample in setup["samples"] if sample["split"] == "train"]
    assert min(test_horizons) > max(train_horizons)


def test_real_batch_masks_target_order_padding_and_day_units(prepared):
    _, _, setup = prepared
    samples = setup["samples"]
    arguments, targets, target_masks = training.batch(samples, setup)
    values, observed, ages, times, horizons, padding = arguments
    assert padding.dtype == torch.bool
    assert horizons.shape == (24, 1)
    torch.testing.assert_close(horizons[:, 0], torch.tensor([sample["horizon"] for sample in samples]))
    for row, sample in enumerate(samples):
        n = len(sample["times"])
        assert not padding[row, :n].any()
        assert padding[row, n:].all()
        torch.testing.assert_close(times[row, :n], torch.from_numpy(sample["times"]))
        for group, variables in setup["input_groups"].items():
            indices = [setup["features"].index(variable) for variable in variables]
            history = sample["history"][:, indices]
            assert np.array_equal(observed[group][row, :n].numpy(), np.isfinite(history))
            assert not observed[group][row, n:].any()
            assert (ages[group][row] >= 0).all()
            mask = observed[group][row, :n].numpy()
            np.testing.assert_array_equal(values[group][row, :n].numpy()[mask], history[mask])
        for group, variables in setup["output_groups"].items():
            indices = [setup["features"].index(variable) for variable in variables]
            actual = sample["target"][indices]
            mask = target_masks[group][row, 0].numpy()
            assert np.array_equal(mask, np.isfinite(actual))
            np.testing.assert_array_equal(targets[group][row, 0].numpy()[mask], actual[mask])


def test_missing_measurements_stay_masked_and_age_uses_only_prior_visits(prepared):
    _, _, setup = prepared
    # Mutations exercise absence mechanics; these are not synthetic health records.
    sample = copy.deepcopy(next(s for s in setup["samples"] if len(s["times"]) == 3))
    sample["history"][:, 0] = np.nan
    sample["history"][0, 0] = 0.0
    sample["history"][:, 1] = np.nan
    sample["target"][0] = np.nan
    (values, observed, ages, _, _, _), _, target_masks = training.batch([sample], setup)
    group = next(iter(setup["input_groups"]))
    torch.testing.assert_close(observed[group][0, :, 0], torch.tensor([True, False, False]))
    assert not observed[group][0, :, 1].any()
    torch.testing.assert_close(ages[group][0, :, 0], torch.from_numpy(sample["times"] - sample["times"][0]))
    torch.testing.assert_close(ages[group][0, :, 1], torch.from_numpy(sample["times"] - sample["times"][0]))
    assert (values[group][0, 1:, 0] == 0).all()
    first_target = setup["features"][0]
    for head, variables in setup["output_groups"].items():
        if first_target in variables:
            assert not target_masks[head][0, 0, variables.index(first_target)]


def test_trainer_does_not_fabricate_loss_for_unlabeled_batch():
    predictions = {"head": torch.ones(1, 2, 3, 3, requires_grad=True)}
    targets = {"head": torch.full((1, 2, 3), float("nan"))}
    masks = {"head": torch.zeros(1, 2, 3, dtype=torch.bool)}
    assert training.loss_function(predictions, targets, masks) is None
    masks["head"][0, 0, 0] = True
    targets["head"][0, 0, 0] = 2.0
    loss = training.loss_function(predictions, targets, masks)
    loss.backward()
    assert torch.isfinite(loss)
    assert predictions["head"].grad[0, 1].eq(0).all()
