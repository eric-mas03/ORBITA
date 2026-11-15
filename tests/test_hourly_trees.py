"""Tree mechanics and actual development exports, without fitting full trials."""
import numpy as np
import pytest
import hashlib
import json
from pathlib import Path
import pandas as pd

from astronaut_twin.hourly_tree_models import fit_cell, predict_cell, feature_columns, ConstantQuantiles


def test_qrf_uses_leaf_response_distribution_not_tree_mean_percentiles():
    x = np.zeros((201, 2), dtype=np.float32)
    y = np.linspace(-10, 10, 201)
    forest = fit_cell(x, y, 'quantile_forest', dict(n_estimators=32, min_samples_leaf=5, max_depth=4), seed=11)
    q = predict_cell(forest, x[:1], 'quantile_forest')[0]
    means = np.array([tree.predict(x[:1])[0] for tree in forest.estimators_])
    naive = np.quantile(means, [.1, .5, .9])
    assert q[2]-q[0] > 12  # Outcome distribution is wide even for identical X.
    assert naive[2]-naive[0] < 4
    assert forest.max_samples_leaf is None
    assert np.isfinite(q).all() and (np.diff(q) >= 0).all()


def test_boosting_quantile_loss_has_no_internal_random_early_stop():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(160, 3)).astype(np.float32)
    y = x[:, 0]+rng.normal(size=160)
    models = fit_cell(x, y, 'hist_boost_quantile', dict(max_iter=12, max_leaf_nodes=7, min_samples_leaf=10))
    assert [model.quantile for model in models] == [.1, .5, .9]
    assert all(model.early_stopping is False and model.n_iter_ == 12 for model in models)
    prediction = predict_cell(models, x[:5], 'hist_boost_quantile')
    assert prediction.shape == (5, 3)
    assert np.isfinite(prediction).all() and (np.diff(prediction, axis=1) >= 0).all()


def test_group_feature_indices_include_exact_same_group_only():
    variables = ['HR', 'SBP', 'Glucose']
    groups = {'vitals': ['HR', 'SBP'], 'metabolic': ['Glucose']}
    assert feature_columns(1, 'hist_boost_quantile', variables, groups) == list(range(9, 18))
    assert feature_columns(0, 'hist_boost_group_features', variables, groups) == list(range(18))
    assert feature_columns(2, 'hist_boost_group_features', variables, groups) == list(range(18, 27))
    with pytest.raises(ValueError, match='partition'):
        feature_columns(0, 'hist_boost_group_features', variables, {'bad': ['Glucose']})


def test_single_observed_training_target_is_retained():
    model = fit_cell(np.zeros((1, 9)), np.array([7.]), 'quantile_forest', {})
    assert isinstance(model, ConstantQuantiles)
    np.testing.assert_array_equal(predict_cell(model, np.zeros((2, 9)), 'quantile_forest'), np.full((2, 3), 7.))


def test_real_tree_exports_use_development_only_and_identical_observed_keys():
    root = Path(__file__).resolve().parents[1]
    output = root/'outputs/hourly_trees'
    manifest = json.loads((output/'fit_manifest.json').read_text())
    assert manifest['status'] == 'development_only_tree_trials_complete'
    assert manifest['export_splits'] == ['dev']
    assert manifest['calibration_or_test_used'] is False
    assert manifest['dataset_sha256'] == hashlib.sha256((root/'data/processed/hourly/hourly_dataset.npz').read_bytes()).hexdigest()
    keys = ['split', 'episode_id', 'origin_hour', 'horizon_hours', 'variable']
    baseline = pd.read_csv(root/'outputs/hourly_baselines/predictions.csv',
                           usecols=keys+['method', 'observed', 'target_scale', 'own_history_observed'])
    baseline = baseline[baseline.split.eq('dev') & baseline.method.eq('persistence')].drop(columns='method')
    actual = pd.read_csv(output/'predictions.csv')
    assert set(actual.method) == {'hist_boost_quantile', 'quantile_forest', 'hist_boost_group_features'}
    assert actual.split.eq('dev').all()
    for method, frame in actual.groupby('method'):
        assert len(frame) == len(baseline) == 117844
        assert not frame.duplicated(keys).any()
        joined = baseline.merge(frame, on=keys, how='outer', validate='one_to_one', indicator=True, suffixes=('_baseline', '_tree'))
        assert joined['_merge'].eq('both').all()
        np.testing.assert_allclose(joined.observed_tree, joined.observed_baseline, rtol=2e-6)
        np.testing.assert_allclose(joined.target_scale_tree, joined.target_scale_baseline, rtol=2e-6)
        np.testing.assert_array_equal(joined.own_history_observed_tree, joined.own_history_observed_baseline)
        assert np.isfinite(frame[['q10', 'q50', 'q90']]).all().all()
        assert (np.diff(frame[['q10', 'q50', 'q90']].to_numpy(), axis=1) >= 0).all()
        empty = frame.train_target_count.eq(0)
        assert frame.loc[empty, 'prediction_status'].eq('no_train_labels_train_only_fallback').all()
    with np.load(output/'scalers.npz') as tree, np.load(root/'outputs/hourly_baselines/scalers.npz') as base:
        for key in ('mean', 'scale', 'median', 'train_row_indices', 'observed_counts'):
            np.testing.assert_array_equal(tree[key], base[key])
