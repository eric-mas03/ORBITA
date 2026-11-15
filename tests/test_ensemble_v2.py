"""Synthetic tests only: no META, CAL or FINAL observations are read."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from astronaut_twin import ensemble_v2 as e


def test_convex_weights_shared_preserve_order_and_reject_invalid_simplex():
    x = np.array([[[1., 2., 3.]], [[3., 4., 5.]]])
    np.testing.assert_allclose(e.convex_combine(x, [.25, .75]), [[2.5, 3.5, 4.5]])
    for weights in ([-.1, 1.1], [.2, .2], [float('nan'), 1.]):
        with pytest.raises(ValueError):
            e.convex_combine(x, weights)


def test_meta_weights_favor_better_component_and_ignore_missing_labels():
    labels = np.zeros((8, 2, 1))
    labels[0, 1, 0] = np.nan
    accurate = np.broadcast_to([-1., 0., 1.], labels.shape+(3,)).copy()
    biased = accurate+10
    predictions = np.stack([accurate, biased])
    first = e.fit_global_weights(predictions, labels, [1.], epochs=60)
    predictions[:, 0, 1, 0] = np.nan  # no label: must not enter optimizer
    second = e.fit_global_weights(predictions, labels, [1.], epochs=60)
    assert first['weights'][0] > .9
    np.testing.assert_array_equal(first['weights'], second['weights'])
    assert first['strategy'] == 'disjoint_META_holdout_not_OOF'
    np.testing.assert_array_equal(first['meta_counts'], [[8], [7]])


def test_empirical_expansion_pool_support_counts_and_median_unchanged():
    labels = np.array([[[5.], [np.nan]], [[5.], [5.]], [[5.], [5.]]])
    prediction = np.broadcast_to([0., 1., 2.], labels.shape+(3,)).copy()
    fitted = e.empirical_expansion(prediction, labels, minimum_labels=3, episode=['a', 'a', 'b'])
    assert fitted['source'][0, 0] == 'direct_variable_horizon'
    assert fitted['source'][1, 0] == 'pooled_same_variable_horizons'
    assert fitted['fitted_counts'][1, 0] == 5
    assert fitted['fitted_episode_counts'][1, 0] == 2
    adjusted = e.apply_expansion(prediction, fitted['expansion'])
    np.testing.assert_array_equal(adjusted[..., 1], prediction[..., 1])
    assert (adjusted[..., 0] <= prediction[..., 0]).all()
    assert (adjusted[..., 2] >= prediction[..., 2]).all()
    assert fitted['formal_coverage_claim'] is False


def test_insufficient_calibration_support_does_not_invent_expansion():
    labels = np.array([[[10.]]])
    fitted = e.empirical_expansion(np.array([[[[0., 1., 2.]]]]), labels, minimum_labels=30)
    assert fitted['expansion'][0, 0] == 0
    assert fitted['source'][0, 0] == 'insufficient_no_expansion'


def test_frozen_gate_has_no_packed_dependency_or_recursion():
    with patch.object(e, 'verify_components'), patch.object(Path, 'exists', return_value=True), \
         patch.object(e, 'sha256', return_value='bound'), \
         patch.object(e, '_source_freeze', return_value='source'), \
         patch.object(e, 'read_json', return_value={'components_frozen_sha256': 'bound', 'ensemble_source_frozen_sha256': 'source'}):
        assert e._frozen_gate(Path('synthetic'), 'ensemble_frozen.json')['components_frozen_sha256'] == 'bound'


def test_train_and_mixed_roles_rejected_before_inference_or_file_reads():
    for role in (['train'], ['meta', 'calibration'], ['test']):
        with patch.object(e, 'verify_components'), patch.object(e, '_config') as config:
            with pytest.raises(ValueError, match='isolated single nontraining'):
                e.infer_models('synthetic', {'episode_split': np.array(role)}, ['linear_quantile'])
            config.assert_not_called()


def test_final_cannot_stage_before_freezes():
    with patch.object(e, 'verify_components'), patch.object(e, '_frozen_gate', side_effect=FileNotFoundError), \
         patch.object(e, '_source_root') as source:
        with pytest.raises(FileNotFoundError):
            e.pack_role('synthetic', 'final')
        source.assert_not_called()


def test_external_final_npz_rejected_before_opening_final():
    with patch.object(e, 'verify_components'), patch.object(e, '_source_freeze'), \
         patch.object(e, '_frozen_gate', return_value={'ensemble_frozen_sha256': 'frozen', 'artifacts': {}}), \
         patch.object(e, 'sha256', return_value='frozen'), patch.object(Path, 'exists', return_value=False), \
         patch.object(e, 'pack_role') as pack:
        with pytest.raises(ValueError, match='canonical'):
            e.evaluate_final('synthetic', 'unrelated.npz')
        pack.assert_not_called()


def test_changed_source_binding_rejected():
    with patch.object(e, 'verify_components'), patch.object(Path, 'exists', return_value=True), \
         patch.object(e, 'sha256', return_value='bound'), patch.object(e, '_source_freeze', return_value='changed'), \
         patch.object(e, 'read_json', return_value={'components_frozen_sha256': 'bound', 'ensemble_source_frozen_sha256': 'old'}):
        with pytest.raises(ValueError, match='code/configuration'):
            e._frozen_gate(Path('synthetic'), 'calibration_frozen.json')


def test_export_masks_sparse_labels_reports_support_and_physical_flags():
    packed = {'sample_episode': np.array([0]), 'origins': np.array([0]), 'times': np.array([1]),
              'episode_ids': np.array(['synthetic']), 'sites': np.array(['site']),
              'variables': np.array(['FiO2']), 'horizons': np.array([1, 4])}
    baseline = SimpleNamespace(labels=np.array([[[50.], [np.nan]]]), scale=np.array([1.]),
        history_length=np.array([1]), history_hours=np.array([0]), counts=np.array([[1]]))
    metadata = {'variables': [{'source_key': 'FiO2', 'group': 'respiratory', 'source_unit_as_documented': 'ambiguous'}]}
    prediction = np.array([[[[-1., 50., 110.]], [[0., 1., 2.]]]])
    table = e.prediction_table('model_raw', prediction, packed, baseline, metadata,
                              train_counts=np.array([[0], [10]]))
    assert len(table) == 1  # the missing horizon is not fabricated
    assert table.iloc[0].prediction_status == 'no_train_labels_fallback_or_untrained'
    assert not bool(table.iloc[0].train_episode_support_known)
    assert bool(table.iloc[0].unit_review_required)
    assert bool(table.iloc[0].prediction_below_physical_support)
    assert bool(table.iloc[0].prediction_above_physical_support)
    assert table.iloc[0].interval_type == 'estimated_quantiles'
    np.testing.assert_array_equal(table[['q10', 'q50', 'q90']].to_numpy(), [[-1., 50., 110.]])
