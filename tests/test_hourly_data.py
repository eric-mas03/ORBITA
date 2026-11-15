"""Causal data boundaries and sparse label invariants; no model training."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from astronaut_twin.hourly_data import DYNAMIC, HORIZONS, PackedHourlyDataset, build_packed, fixed_splits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def working():
    parent = ROOT/'outputs/hourly_data'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='data-test-', dir=parent) as name:
        yield Path(name)


def episode(directory, name, hours, hr=None):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.nan, index=np.arange(len(hours)), columns=DYNAMIC)
    frame['ICULOS'] = hours
    frame['HR'] = hr if hr is not None else np.arange(len(hours))+60
    frame.to_csv(directory/name, sep='|', index=False)
    return frame


def build(working):
    destination = working/'packed.npz'
    summary = build_packed(working/'raw', destination, working/'splits.json')
    return PackedHourlyDataset(destination), summary


def test_sparse_labels_exact_hours_and_episode_isolation(working):
    episode(working/'raw/A', 'first.psv', [1, 2, 5])
    episode(working/'raw/A', 'second.psv', [1, 2], [90, 91])
    dataset, _ = build(working)
    item = dataset.get_window(0)
    assert item['origin_time'] == 1
    assert item['target_rows'].tolist() == [1, 2, -1, -1]
    assert item['label_mask'][:, DYNAMIC.index('HR')].tolist() == [True, True, False, False]
    assert item['label_mask'].sum() == 2
    assert len(item['values']) == 1  # Partial history is retained.
    for index in range(len(dataset)):
        item = dataset.get_window(index)
        episode_index = dataset.arrays['sample_episode'][index]
        lower, upper = dataset.arrays['episode_offsets'][episode_index:episode_index+2]
        positive = item['target_rows'][item['target_rows'] >= 0]
        assert ((positive >= lower) & (positive < upper)).all()
        for h, row in zip(HORIZONS, item['target_rows']):
            if row >= 0:
                assert dataset.arrays['times'][row] == item['origin_time']+h


def test_causal_history_future_mutation_and_right_padding(working):
    frame = episode(working/'raw/A', 'first.psv', list(range(1, 32)))
    dataset, _ = build(working)
    index = int(np.flatnonzero(dataset.arrays['origins'] == 24)[0])
    before = dataset.get_window(index)
    assert len(before['values']) == 24
    assert before['times'].tolist() == list(range(2, 26))
    frame.loc[25:, 'HR'] = 100
    frame.to_csv(working/'raw/A/first.psv', sep='|', index=False)
    changed, _ = build(working)
    np.testing.assert_equal(before['values'], changed.get_window(index)['values'])
    np.testing.assert_equal(dataset.arrays['episode_split'], changed.arrays['episode_split'])
    padded = dataset.get_window(0, pad_to=24)
    assert padded['padding_mask'].tolist() == [False]+[True]*23
    assert not padded['observed'][1:].any()
    assert np.isnan(padded['values'][1:]).all()
    with pytest.raises(ValueError, match='truncate'):
        dataset.get_window(index, pad_to=2)


def test_quality_invalid_masks_and_negative_base_excess(working):
    frame = episode(working/'raw/A', 'first.psv', [1, 2, 3], [60, 61, 62])
    frame['Glucose'] = [100, -2, np.nan]
    frame['BaseExcess'] = [-5, -3, np.nan]
    frame['O2Sat'] = [97, 110, np.nan]
    frame.to_csv(working/'raw/A/first.psv', sep='|', index=False)
    dataset, summary = build(working)
    glucose, base, oxygen = [DYNAMIC.index(name) for name in ('Glucose', 'BaseExcess', 'O2Sat')]
    item = dataset.get_window(0)
    assert not item['label_mask'][0, glucose]
    assert not item['label_mask'][0, oxygen]
    assert item['labels'][0, base] == -3
    assert dataset.arrays['quality_invalid'][1, glucose]
    assert dataset.arrays['source_observed'][1, glucose]
    assert not dataset.arrays['quality_invalid'][2, glucose]
    assert len(summary['quality_flags']) >= 2


def test_no_input_or_future_labels_does_not_create_sample(working):
    frame = episode(working/'raw/A', 'empty.psv', [1, 2, 3], [np.nan, np.nan, 60])
    frame.to_csv(working/'raw/A/empty.psv', sep='|', index=False)
    dataset, _ = build(working)
    assert len(dataset) == 0
    assert dataset.arrays['target_rows'].shape == (0, 4)


def test_fixed_split_site_stratification_and_value_independence(working):
    ids = [f'{site}/{index:03}' for site in ('A', 'B') for index in range(100)]
    sites = [name.split('/')[0] for name in ids]
    path = working/'splits.json'
    expected = fixed_splits(ids, sites, path)
    reversed_order = fixed_splits(ids[::-1], sites[::-1])
    assert expected == reversed_order
    for site in ('A', 'B'):
        actual = [expected[name] for name in ids if name.startswith(site+'/')]
        assert {label: actual.count(label) for label in set(actual)} == {'train': 60, 'dev': 15, 'calibration': 10, 'test': 15}
    assert fixed_splits(ids, sites, path) == expected
    with pytest.raises(ValueError, match='inventory'):
        fixed_splits(ids[:-1], sites[:-1], path)


def test_reject_noncausal_source_clock(working):
    episode(working/'raw/A', 'bad.psv', [1, 2, 2])
    with pytest.raises(ValueError, match='strictly increasing'):
        build(working)
