"""Independent data-leakage and prediction-key audits; never fit a model."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from astronaut_twin.hourly_data import DYNAMIC

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def packed():
    with np.load(ROOT/'data/processed/hourly/hourly_dataset.npz', allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def training_rows(packed):
    offsets = packed['episode_offsets']
    return np.concatenate([np.arange(offsets[index], offsets[index+1])
                           for index, split in enumerate(packed['episode_split']) if split == 'train'])


def test_saved_baseline_scaler_is_unique_training_rows_only(packed):
    rows = training_rows(packed)
    assert len(rows) == len(np.unique(rows))
    values = packed['values'][rows].astype(np.float64)
    with np.load(ROOT/'outputs/hourly_baselines/scalers.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['train_row_indices'], rows)
        np.testing.assert_array_equal(saved['observed_counts'], np.isfinite(values).sum(axis=0))
        np.testing.assert_allclose(saved['mean'], np.nanmean(values, axis=0), rtol=2e-6)
        expected_scale = np.nanstd(values, axis=0)
        expected_scale[expected_scale <= 1e-8] = 1
        np.testing.assert_allclose(saved['scale'], expected_scale, rtol=2e-6)


@pytest.fixture(scope='module')
def neural(packed):
    from astronaut_twin.hourly_neural import prepare_hourly_neural
    return prepare_hourly_neural(packed)


def assert_model_args_equal(first, second):
    for a, b in zip(first, second):
        if isinstance(a, dict):
            assert set(a) == set(b)
            for key in a:
                torch.testing.assert_close(a[key], b[key], equal_nan=True)
        else:
            torch.testing.assert_close(a, b, equal_nan=True)


def test_neural_scalers_match_baseline_without_window_duplication(packed, neural):
    np.testing.assert_array_equal(neural.train_row_indices, training_rows(packed))
    for folder in ('hourly_baselines', 'hourly_neural'):
        with np.load(ROOT/'outputs'/folder/'scalers.npz', allow_pickle=False) as saved:
            for field in ('mean', 'scale', 'median'):
                np.testing.assert_allclose(getattr(neural, field), saved[field], rtol=2e-6)
            np.testing.assert_array_equal(neural.scaler_counts, saved['observed_counts'])
            np.testing.assert_array_equal(neural.train_row_indices, saved['train_row_indices'])


def test_fixed_split_hash_and_episode_assignments(packed):
    split_path = ROOT/'data/processed/hourly/splits.json'
    manifest = json.loads((ROOT/'data/processed/hourly/manifest.json').read_text())
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == manifest['split_sha256']
    saved = json.loads(split_path.read_text())
    actual = dict(zip(packed['episode_ids'].astype(str), packed['episode_split'].astype(str)))
    assert actual == saved['episode_split']
    dataset_hash = hashlib.sha256((ROOT/'data/processed/hourly/hourly_dataset.npz').read_bytes()).hexdigest()
    assert dataset_hash == manifest['dataset_sha256']
    baseline = json.loads((ROOT/'outputs/hourly_baselines/fit_manifest.json').read_text())
    neural = json.loads((ROOT/'outputs/hourly_neural/fit_manifest.json').read_text())
    assert dataset_hash == baseline['data_sha256'] == neural['dataset_sha256']
    assert neural['both_models_frozen_before_test_predictions'] is True


def test_heldout_values_cannot_change_training_scaler_features_or_splits(packed, neural):
    from astronaut_twin.hourly_neural import prepare_hourly_neural
    changed = {key: value.copy() for key, value in packed.items()}
    train_rows = training_rows(packed)
    heldout = np.ones(len(packed['values']), dtype=bool)
    heldout[train_rows] = False
    finite = np.isfinite(changed['values']) & heldout[:, None]
    changed['values'][finite] += 12345
    transformed = prepare_hourly_neural(changed)
    for field in ('mean', 'scale', 'median'):
        np.testing.assert_array_equal(getattr(neural, field), getattr(transformed, field))
    torch.testing.assert_close(neural.raw_values[train_rows], transformed.raw_values[train_rows])
    np.testing.assert_array_equal(neural.split, transformed.split)
    np.testing.assert_array_equal(packed['episode_split'], changed['episode_split'])
    selected = np.flatnonzero(neural.split == 'train')[:3]
    groups = {'all': DYNAMIC}
    before, _, _ = neural.batch(selected, groups)
    after, _, _ = transformed.batch(selected, groups)
    assert_model_args_equal(before, after)


def test_future_in_same_episode_cannot_change_origin_history_or_ages(packed, neural):
    from astronaut_twin.hourly_neural import prepare_hourly_neural
    candidates = np.flatnonzero((neural.split == 'dev') & (packed['target_rows'][:, -1] >= 0)
                               & (packed['origins']-packed['history_start'] >= 23))
    index = int(candidates[0])
    origin = int(packed['origins'][index])
    episode = int(packed['sample_episode'][index])
    stop = int(packed['episode_offsets'][episode+1])
    changed = {key: value.copy() for key, value in packed.items()}
    future = changed['values'][origin+1:stop]
    future[np.isfinite(future)] += 7654
    transformed = prepare_hourly_neural(changed)
    groups = {'all': DYNAMIC}
    before, labels_before, mask_before = neural.batch(np.array([index]), groups)
    after, labels_after, mask_after = transformed.batch(np.array([index]), groups)
    assert_model_args_equal(before, after)
    torch.testing.assert_close(mask_before, mask_after)
    assert not torch.allclose(labels_before, labels_after, equal_nan=True)


def test_sparse_labels_exact_rows_no_interpolation_and_padding(packed, neural):
    indices = np.array([0, len(packed['origins'])//2, len(packed['origins'])-1])
    groups = {'all': DYNAMIC}
    model_args, labels, mask = neural.batch(indices, groups)
    value_groups, observed_groups, ages_groups, times, horizons, padding = model_args
    expected = np.full((len(indices), 4, 34), np.nan, dtype=np.float32)
    for batch_index, sample in enumerate(indices):
        origin = int(packed['origins'][sample])
        start = int(packed['history_start'][sample])
        length = origin-start+1
        episode = int(packed['sample_episode'][sample])
        lower, upper = packed['episode_offsets'][episode:episode+2]
        assert lower <= start <= origin < upper
        history = packed['values'][start:origin+1]
        observed = np.isfinite(history)
        np.testing.assert_array_equal(observed_groups['all'][batch_index, :length].numpy(), observed)
        expected_values = (history-neural.mean)/neural.scale
        np.testing.assert_allclose(value_groups['all'][batch_index, :length].numpy()[observed], expected_values[observed], rtol=2e-6)
        assert not observed_groups['all'][batch_index, length:].any()
        assert not padding[batch_index, :length].any()
        assert padding[batch_index, length:].all()
        assert not ages_groups['all'][batch_index, length:].any()
        assert torch.isfinite(value_groups['all']).all()
        assert torch.isfinite(ages_groups['all']).all()
        # At every real step, age can depend on measurements at this step or
        # earlier only. No later step is allowed to reset an earlier age.
        for variable in range(34):
            previous = None
            for local in range(length):
                if observed[local, variable]:
                    previous = local
                actual_age = float(ages_groups['all'][batch_index, local, variable])
                if previous is not None:
                    expected_age = packed['times'][start+local]-packed['times'][start+previous]
                    assert actual_age == expected_age
        for h_index, row in enumerate(packed['target_rows'][sample]):
            if row >= 0:
                assert lower <= row < upper
                assert packed['times'][row]-packed['times'][origin] == packed['horizons'][h_index]
                expected[batch_index, h_index] = packed['values'][row]
    np.testing.assert_array_equal(mask.numpy(), np.isfinite(expected))
    restored = labels.numpy()*neural.scale[None, None, :]+neural.mean[None, None, :]
    np.testing.assert_allclose(restored[mask.numpy()], expected[mask.numpy()], rtol=2e-5, atol=2e-4)
    assert (~mask).any() and mask.any()  # Sparse supervision survives batching.


def test_real_biomedical_groups_preserve_variable_alignment(neural):
    from astronaut_twin.hourly_neural import flatten_predictions
    config = json.loads((ROOT/'config/hourly_task_candidates.json').read_text())
    groups = config['input_groups']
    tagged = {}
    for group, variables in groups.items():
        indices = [DYNAMIC.index(name.removeprefix('SEPSIS2019::')) for name in variables]
        tagged[group] = torch.tensor(indices, dtype=torch.float32)[None, None, :, None].expand(1, 4, -1, 3)
    restored = flatten_predictions(tagged, groups, DYNAMIC)
    torch.testing.assert_close(restored[0, 0, :, 0], torch.arange(34, dtype=torch.float32))
    args, _, _ = neural.batch(np.array([0]), groups)
    values = args[0]
    all_args, _, _ = neural.batch(np.array([0]), {'all': DYNAMIC})
    for group, variables in groups.items():
        columns = [DYNAMIC.index(name.removeprefix('SEPSIS2019::')) for name in variables]
        torch.testing.assert_close(values[group], all_args[0]['all'][:, :, columns])


def expected_export(packed):
    parts = []
    sample_splits = packed['episode_split'][packed['sample_episode']]
    for split in ('dev', 'calibration', 'test'):
        indices = np.flatnonzero(sample_splits == split)
        rows = packed['target_rows'][indices]
        labels = packed['values'][np.maximum(rows, 0)].copy()
        labels[rows < 0] = np.nan
        local, horizon, variable = np.nonzero(np.isfinite(labels))
        samples = indices[local]
        episodes = packed['sample_episode'][samples]
        parts.append(pd.DataFrame({'split': split, 'episode_id': packed['episode_ids'][episodes],
                                   'origin_hour': packed['times'][packed['origins'][samples]],
                                   'horizon_hours': packed['horizons'][horizon],
                                   'variable': packed['variables'][variable],
                                   'observed_expected': labels[local, horizon, variable]}))
    return pd.concat(parts, ignore_index=True)


def test_baseline_exports_cover_exact_same_observed_targets(packed):
    expected = expected_export(packed)
    keys = ['split', 'episode_id', 'origin_hour', 'horizon_hours', 'variable']
    predictions = pd.read_csv(ROOT/'outputs/hourly_baselines/predictions.csv',
                              usecols=keys+['method', 'observed', 'target_scale'])
    with np.load(ROOT/'outputs/hourly_baselines/scalers.npz', allow_pickle=False) as scaler:
        scales = dict(zip(scaler['variables'].astype(str), scaler['scale']))
    for method, subset in predictions.groupby('method'):
        assert not subset.duplicated(keys).any(), method
        joined = expected.merge(subset, on=keys, how='outer', validate='one_to_one', indicator=True)
        assert joined['_merge'].eq('both').all(), method
        np.testing.assert_allclose(joined.observed, joined.observed_expected, rtol=2e-6)
        np.testing.assert_allclose(joined.target_scale, joined.variable.map(scales), rtol=2e-6)


def test_neural_exports_match_baseline_keys_labels_scales_and_history(packed):
    keys = ['split', 'episode_id', 'origin_hour', 'horizon_hours', 'variable']
    columns = keys+['method', 'observed', 'target_scale', 'target_hour', 'own_history_observed',
                    'history_length', 'used_train_median_fallback', 'q10', 'q50', 'q90']
    baseline = pd.read_csv(ROOT/'outputs/hourly_baselines/predictions.csv', usecols=columns)
    reference = baseline[baseline.method.eq('persistence')].drop(columns=['method', 'q10', 'q50', 'q90', 'used_train_median_fallback'])
    actual = pd.read_csv(ROOT/'outputs/hourly_neural/predictions.csv', usecols=columns)
    assert set(actual.method) == {'gru_shared_readout', 'transformer_grouped'}
    assert not actual.used_train_median_fallback.any()
    for method, subset in actual.groupby('method'):
        assert not subset.duplicated(keys).any(), method
        joined = reference.merge(subset, on=keys, how='outer', validate='one_to_one', indicator=True,
                                 suffixes=('_reference', '_neural'))
        assert joined['_merge'].eq('both').all(), method
        for column in ('observed', 'target_scale'):
            np.testing.assert_allclose(joined[column+'_neural'], joined[column+'_reference'], rtol=2e-6, atol=1e-5)
        for column in ('own_history_observed', 'history_length', 'target_hour'):
            np.testing.assert_array_equal(joined[column+'_neural'], joined[column+'_reference'])
        np.testing.assert_array_equal(joined.target_hour_neural, joined.origin_hour+joined.horizon_hours)
        quantiles = joined[['q10', 'q50', 'q90']].to_numpy()
        assert np.isfinite(quantiles).all()
        assert (np.diff(quantiles, axis=1) >= 0).all()
        # Explicitly preserve zero-support cells in the34x4x3 coverage grid.
        grid = pd.MultiIndex.from_product([['dev', 'calibration', 'test'], DYNAMIC, [1, 4, 8, 24]],
                                          names=['split', 'variable', 'horizon_hours'])
        expected_counts = reference.groupby(['split', 'variable', 'horizon_hours']).size().reindex(grid, fill_value=0)
        actual_counts = subset.groupby(['split', 'variable', 'horizon_hours']).size().reindex(grid, fill_value=0)
        pd.testing.assert_series_equal(actual_counts, expected_counts)
        assert len(expected_counts) == 34*4*3
