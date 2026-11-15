"""Frozen V2 late ensemble: disjoint META fit, CAL expansion, one final read.

The global weights are shared by all quantiles, variables and horizons.
This is quantile averaging, not the quantiles of a distribution mixture.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch

QUANTILES = np.asarray([.1, .5, .9])
KEYS = ['episode_id', 'origin_hour', 'horizon_hours', 'variable']
TREE_METHODS = {'hist_boost_quantile', 'quantile_forest', 'hist_boost_group_features'}
NEURAL_METHODS = {'gru_shared_readout', 'transformer_grouped'}
RESIDUAL_METHODS = {'causal_residual', 'window_bidirectional_residual', 'group_temporal_residual'}
METHOD_DIRECTORIES = {'persistence': 'hourly_baselines', 'linear_quantile': 'hourly_baselines',
    **{method: 'hourly_trees' for method in TREE_METHODS},
    **{method: 'hourly_neural' for method in NEURAL_METHODS},
    **{method: 'hourly_architecture_trials' for method in RESIDUAL_METHODS},
    'own_summary_mlp_residual': 'hourly_residual_control'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assert_ordered(predictions):
    prediction = np.asarray(predictions)
    if prediction.shape[-1] != 3 or not np.isfinite(prediction).all() or (np.diff(prediction, axis=-1) < 0).any():
        raise ValueError('All component quantiles must be finite and ordered before combination')


def convex_combine(predictions, weights):
    predictions = np.asarray(predictions)
    weights = np.asarray(weights, dtype=np.float64)
    if predictions.ndim < 3 or weights.shape != (predictions.shape[0],):
        raise ValueError('Expected components first and one global weight per component')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1., atol=1e-7):
        raise ValueError('Weights must be nonnegative and sum to one')
    assert_ordered(predictions)
    return np.tensordot(weights, predictions, axes=(0, 0)).astype(np.float32)


def fit_global_weights(predictions, labels, scale, *, regularization=.01, learning_rate=.05, epochs=300):
    """Fit the predeclared shared simplex weights on disjoint META only.

    No selection of regularization/epochs on META, CAL or final is performed.
    Missing labels are masked independently for each variable and horizon.
    """
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    scale = np.asarray(scale, dtype=np.float64)
    if predictions.shape[1:] != labels.shape+(3,) or labels.ndim != 3:
        raise ValueError('Expected predictions[M,N,H,V,3] and labels[N,H,V]')
    if scale.shape != (labels.shape[-1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Frozen positive training scales required')
    if epochs < 1 or regularization < 0 or learning_rate <= 0:
        raise ValueError('Invalid predeclared optimizer configuration')
    observed = np.isfinite(labels)
    if not observed.any():
        raise ValueError('No observed META target labels')
    selected = predictions[:, observed]
    assert_ordered(selected)
    _, horizon_index, variable_index = np.nonzero(observed)
    cells = horizon_index*labels.shape[-1]+variable_index
    cell_counts = np.bincount(cells, minlength=labels.shape[1]*labels.shape[2])
    active = cell_counts > 0
    observation_weight = 1./(cell_counts[cells]*active.sum())
    x = torch.from_numpy((selected/scale[variable_index][None, :, None]).astype(np.float64))
    y = torch.from_numpy((labels[observed]/scale[variable_index]).astype(np.float64))
    importance = torch.from_numpy(observation_weight.astype(np.float64))
    q = torch.tensor(QUANTILES, dtype=torch.float64)
    logits = torch.zeros(predictions.shape[0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    trace = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = logits.softmax(dim=0)
        combined = torch.einsum('m,mlq->lq', weights, x)
        error = y[:, None]-combined
        pinball = (torch.maximum(q*error, (q-1)*error).mean(-1)*importance).sum()
        penalty = regularization*(weights-1./len(weights)).square().sum()
        objective = pinball+penalty
        if not torch.isfinite(objective):
            raise RuntimeError('Nonfinite META weight objective')
        objective.backward()
        optimizer.step()
        trace.append({'epoch': epoch+1, 'meta_macro_standardized_pinball': float(pinball.detach()),
                      'l2_uniform_penalty': float(penalty.detach()), 'objective': float(objective.detach())})
    weights = logits.detach().softmax(0).numpy()
    return {'weights': weights, 'trace': trace, 'meta_counts': cell_counts.reshape(labels.shape[1:]),
            'regularization': regularization, 'learning_rate': learning_rate, 'epochs': epochs,
            'objective': 'mean observed pinball within variable-horizon cell then equal active-cell mean + lambda*sum((w-1/M)^2)',
            'strategy': 'disjoint_META_holdout_not_OOF', 'quantile_weights_shared': True}


def empirical_expansion(predictions, labels, *, alpha=.2, minimum_labels=30, episode=None):
    """Nonnegative empirical interval expansion; no exchangeability claim."""
    prediction, labels = np.asarray(predictions), np.asarray(labels)
    if prediction.shape != labels.shape+(3,) or labels.ndim != 3:
        raise ValueError('Calibration shapes differ')
    assert_ordered(prediction)
    if not 0 < alpha < 1 or minimum_labels < 1:
        raise ValueError('Invalid calibration specification')
    valid = np.isfinite(labels)
    scores = np.maximum(np.maximum(prediction[..., 0]-labels, labels-prediction[..., 2]), 0.)
    scores[~valid] = np.nan
    shape = labels.shape[1:]
    expansion = np.zeros(shape, dtype=np.float32)
    counts = valid.sum(axis=0)
    fitted_counts = np.zeros(shape, dtype=np.int64)
    source = np.full(shape, 'insufficient_no_expansion', dtype='U48')
    episode_counts = np.zeros(shape, dtype=np.int64)
    fitted_episode_counts = np.zeros(shape, dtype=np.int64)
    for h in range(shape[0]):
        for v in range(shape[1]):
            selected = scores[:, h, v]
            selected = selected[np.isfinite(selected)]
            if episode is not None:
                episode_counts[h, v] = len(np.unique(np.asarray(episode)[valid[:, h, v]]))
            if len(selected) >= minimum_labels:
                source[h, v] = 'direct_variable_horizon'
                fitted_episode_counts[h, v] = episode_counts[h, v]
            else:
                pooled = scores[:, :, v].reshape(-1)
                selected = pooled[np.isfinite(pooled)]
                if len(selected) >= minimum_labels:
                    source[h, v] = 'pooled_same_variable_horizons'
                    if episode is not None:
                        selected_episodes = np.broadcast_to(np.asarray(episode)[:, None], valid[:, :, v].shape)
                        fitted_episode_counts[h, v] = len(np.unique(selected_episodes[valid[:, :, v]]))
            fitted_counts[h, v] = len(selected)
            if source[h, v] != 'insufficient_no_expansion':
                # Empirical finite-sample rank, capped at observed maximum.
                # Dependence prevents a formal conformal coverage guarantee.
                rank = min(len(selected), math.ceil((len(selected)+1)*(1-alpha)))
                expansion[h, v] = np.partition(selected, rank-1)[rank-1]
    return {'expansion': expansion, 'counts': counts, 'fitted_counts': fitted_counts,
            'source': source, 'episode_counts': episode_counts,
            'fitted_episode_counts': fitted_episode_counts,
            'alpha': alpha, 'minimum_labels': minimum_labels,
            'formal_coverage_claim': False}


def apply_expansion(predictions, expansion):
    prediction = np.asarray(predictions).copy()
    assert_ordered(prediction)
    expansion = np.asarray(expansion)
    if expansion.shape != prediction.shape[-3:-1] or not np.isfinite(expansion).all() or (expansion < 0).any():
        raise ValueError('Expected a nonnegative [horizon,variable] expansion')
    prediction[..., 0] -= expansion
    prediction[..., 2] += expansion
    return prediction


def verify_components(workspace):
    workspace = Path(workspace).resolve()
    path = workspace/'components_frozen.json'
    freeze = read_json(path)
    if (freeze.get('selection_split') != 'dev' or freeze.get('meta_used') is not False
            or freeze.get('calibration_used') is not False or freeze.get('final_used') is not False):
        raise ValueError('DEV-only component freeze is required before META/CAL access')
    methods = freeze.get('methods', [])
    if not methods or 'linear_quantile' not in methods or len(methods) != len(set(methods)) or not set(methods).issubset(METHOD_DIRECTORIES):
        raise ValueError('Frozen component list must include unique methods and linear reference')
    prepared_path = workspace/'prepared.json'
    if freeze.get('prepared_signature') != sha256(prepared_path):
        raise ValueError('Prepared campaign changed after component selection')
    prepared = read_json(prepared_path)
    if prepared.get('config') != read_json(workspace/'campaign_config.json'):
        raise ValueError('Runtime configuration differs from the prepared predeclaration')
    if freeze.get('dataset_sha256') != prepared.get('dataset_sha256'):
        raise ValueError('Frozen component dataset provenance differs')
    for relative, expected in prepared.get('training_source_sha256', {}).items():
        snapshot = (workspace/relative).resolve()
        actual = (Path(__file__).resolve().parents[2]/relative).resolve()
        if not snapshot.is_relative_to(workspace) or sha256(snapshot) != expected or sha256(actual) != expected:
            raise ValueError('Frozen inference/model source differs from training source')
    hashes = freeze.get('artifact_sha256', {})
    if not hashes:
        raise ValueError('Frozen component artifacts need content hashes')
    for relative, expected in hashes.items():
        artifact = (workspace/relative).resolve()
        if not artifact.is_relative_to(workspace) or sha256(artifact) != expected:
            raise ValueError('Frozen component artifact changed')
    return freeze


def _config(workspace):
    return read_json(Path(workspace)/'campaign_config.json')


def _source_root(workspace):
    prepared = read_json(Path(workspace)/'prepared.json')
    value = prepared.get('source_root') or prepared.get('project_root') or _config(workspace).get('source_root')
    if value is None:
        raise ValueError('Prepared campaign must record its source_root for exact role staging')
    return Path(value).resolve()


def _frozen_gate(workspace, name):
    workspace = Path(workspace)
    verify_components(workspace)
    if not (workspace/name).exists():
        raise FileNotFoundError(f'{name} required before this role can be opened')
    value = read_json(workspace/name)
    if value.get('components_frozen_sha256') != sha256(workspace/'components_frozen.json'):
        raise ValueError('Component freeze differs from downstream frozen state')
    if value.get('ensemble_source_frozen_sha256') != _source_freeze(workspace):
        raise ValueError('Downstream state differs from frozen ensemble code/configuration')
    return value


def pack_role(workspace, role):
    """Stage exact assigned files only; never glob the mixed expansion folder."""
    workspace = Path(workspace).resolve()
    if role not in {'meta', 'calibration', 'final'}:
        raise ValueError('Only explicitly gated nontraining roles can be staged')
    verify_components(workspace)
    if role in {'calibration', 'final'}:
        _frozen_gate(workspace, 'ensemble_frozen.json')
    if role == 'final':
        calibration = _frozen_gate(workspace, 'calibration_frozen.json')
        if calibration.get('ensemble_frozen_sha256') != sha256(workspace/'ensemble_frozen.json'):
            raise ValueError('Calibration is not bound to the frozen weights')
    root = _source_root(workspace)
    acquisition_path = root/'data/manifests/hourly_expansion_v2.json'
    acquisition = read_json(acquisition_path)
    prepared = read_json(workspace/'prepared.json')
    config = _config(workspace)
    audit_path = Path(config.get('audit_path', root/'data/processed/hourly_v2/audit.json'))
    roles_path = Path(config.get('roles_path', root/'data/processed/hourly_v2/roles.json'))
    audit_path = audit_path if audit_path.is_absolute() else root/audit_path
    roles_path = roles_path if roles_path.is_absolute() else root/roles_path
    if not prepared.get('expansion_manifest_sha256') or sha256(acquisition_path) != prepared['expansion_manifest_sha256']:
        raise ValueError('Expansion acquisition changed after preparation')
    roles_hash = prepared.get('roles_json_sha256', prepared.get('roles_sha256'))
    if not roles_hash or sha256(roles_path) != roles_hash:
        raise ValueError('Prepared role assignments are missing or changed')
    if not prepared.get('audit_sha256') or sha256(audit_path) != prepared['audit_sha256']:
        raise ValueError('Prepared data audit is missing or changed')
    audit = read_json(audit_path)
    if audit.get('source_manifest_sha256') != sha256(acquisition_path):
        raise ValueError('Acquisition differs from the audited manifest')
    with np.load(workspace/'nonfinal_dataset.npz', allow_pickle=False) as metadata:
        role_map = dict(zip(metadata['episode_ids'].astype(str), metadata['episode_split'].astype(str)))
    if role_map != read_json(roles_path):
        raise ValueError('Packed nonfinal roles differ from the frozen assignments')
    expansion = {entry['episode_id']: entry for entry in acquisition['files']}
    if role == 'final':
        records = [entry for entry in acquisition['files'] if entry['role'] == 'final']
        if len(records) != 300 or len({entry['episode_id'] for entry in records}) != 300:
            raise ValueError('Need exactly300 distinct roleFINAL acquisition records')
        if set(entry['episode_id'] for entry in records) & set(role_map):
            raise ValueError('Final overlaps nonfinal roles')
        final_ids = {entry['episode_id'] for entry in records}
        if final_ids != set(audit.get('final_episode_ids', [])):
            raise ValueError('Final IDs differ from the exact audited reservation')
        old_ids = set(read_json(root/'data/processed/hourly/splits.json')['episode_split'])
        previous_confirmation = read_json(root/'data/manifests/confirmation_holdout.json')
        consumed_ids = {str(Path(entry['path']).parent.name)+'/'+Path(entry['path']).stem
                        for entry in previous_confirmation['files'] if entry.get('kind') == 'episode'}
        if final_ids & (old_ids | consumed_ids):
            raise ValueError('Final reuses V1 or consumed confirmation episode identities')
    elif role == 'meta':
        names = {name for name, assigned in role_map.items() if assigned == role}
        records = [expansion[name] for name in names]
        if len(records) != 200 or any(entry['role'] != 'meta' for entry in records):
            raise ValueError('META must be the200 disjoint preassigned new episodes')
    else:
        names = {name for name, assigned in role_map.items() if assigned == role}
        original = read_json(root/'data/processed/hourly/source_hashes.json')
        original = {entry['episode_id']: entry for entry in original}
        records = []
        for name in names:
            source = original[name]
            source_path = Path(source.get('source_path', source.get('path', '')))
            if not source_path.is_absolute():
                source_path = root/source_path
            records.append({'episode_id': name, 'role': role, 'path': str(source_path),
                            'sha256': source['sha256'], 'bytes': source['bytes'], 'site': name.split('/')[0]})
        if len(records) != 100:
            raise ValueError('CAL must contain exactly the100 frozen calibration episodes')
    destination = workspace/'role_data'/role
    staging = destination/'sources'
    role_signature = hashlib.sha256(json.dumps(sorted((entry['episode_id'], entry['sha256']) for entry in records)).encode()).hexdigest()
    completed = destination/'complete.json'
    if completed.exists():
        saved = read_json(completed)
        if saved['role_source_sha256'] != role_signature or saved['packed_sha256'] != sha256(destination/'packed.npz'):
            raise ValueError('Previously staged role changed')
        return load_npz(destination/'packed.npz')
    for entry in records:
        source = Path(entry['path'])
        source = source if source.is_absolute() else root/source
        source = source.resolve()
        expected_directory = root/'data/raw/external'/('PHYSIONET_SEPSIS_2019_1.0.0' if role == 'calibration' else 'PHYSIONET_EXPANSION_V2')
        if not source.is_relative_to(expected_directory.resolve()):
            raise ValueError('Role source is outside its declared raw directory')
        if sha256(source) != entry['sha256'] or source.stat().st_size != entry['bytes']:
            raise ValueError('Role raw bytes differ from the acquisition hash')
        target = staging/entry['site']/source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != entry['sha256']:
            raise ValueError('Staged role file changed')
        if not target.exists():
            shutil.copyfile(source, target)
    from .hourly_data import build_packed
    summary = build_packed(staging, destination/'packed.npz', destination/'temporary_builder_splits.json')
    packed = load_npz(destination/'packed.npz')
    expected_ids = {entry['episode_id'] for entry in records}
    if set(packed['episode_ids'].astype(str)) != expected_ids:
        raise ValueError('Role staging includes unexpected files')
    packed['episode_split'] = np.full(len(packed['episode_ids']), role)
    np.savez_compressed(destination/'packed.npz', **packed)
    write_json_new(completed, {'role': role, 'episodes': len(records), 'origins': len(packed['origins']),
        'role_source_sha256': role_signature, 'packed_sha256': sha256(destination/'packed.npz'),
        'quality_invalid_cells': len(summary['quality_flags']), 'scalers_fitted': False})
    return packed


def infer_models(workspace, packed, methods, batch_size=256):
    """Apply frozen statistics/checkpoints only, including all11 V2 families."""
    workspace = Path(workspace)
    verify_components(workspace)
    roles = set(np.asarray(packed['episode_split']).astype(str))
    if len(roles) != 1 or not roles.issubset({'meta', 'calibration', 'final', 'dev'}):
        raise ValueError('Frozen inference requires an isolated single nontraining role')
    if roles & {'calibration', 'final'}:
        _frozen_gate(workspace, 'ensemble_frozen.json')
    if roles == {'final'}:
        _frozen_gate(workspace, 'calibration_frozen.json')
    dependency = _config(workspace).get('original_tree_dependency_path')
    if dependency and str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))
    from .hourly_architecture_trials import prepare_frozen_trial_data, load_trial_checkpoint
    from .hourly_baselines import predict_linear, predict_persistence
    from .hourly_neural import make_hourly_twin, flatten_predictions
    torch.set_num_threads(2)
    base = workspace/'outputs/hourly_baselines'
    scalers = load_npz(base/'scalers.npz')
    if not np.all(np.asarray(scalers['scale']) > 0):
        raise ValueError('Frozen scales must be positive')
    neural, baseline = prepare_frozen_trial_data(packed, scalers)
    linear, bands = load_npz(base/'linear_quantile.npz'), load_npz(base/'persistence.npz')
    reference = predict_linear(baseline, linear, bands)
    outputs = {}
    for method in methods:
        if method not in METHOD_DIRECTORIES:
            raise ValueError('Unknown frozen method')
        directory = workspace/'outputs'/METHOD_DIRECTORIES[method]
        if method == 'linear_quantile':
            prediction = reference
        elif method == 'persistence':
            prediction = predict_persistence(baseline, bands)
        elif method in TREE_METHODS:
            from .hourly_tree_models import load_tree_family, predict_tree_family
            artifact = load_tree_family(directory, method)
            for name in ('mean', 'scale', 'median'):
                np.testing.assert_array_equal(artifact['scalers'][name], scalers[name])
            prediction = predict_tree_family(artifact, baseline.features)
        else:
            if method in NEURAL_METHODS:
                # Only locally produced, hash-verified checkpoints are loaded.
                state = torch.load(directory/f'{method}.pt', map_location='cpu', weights_only=False)
                groups = state['groups']
                model = make_hourly_twin(groups, method, d_model=state['config']['d_model'], dropout=state['config']['dropout'])
                model.load_state_dict(state['state_dict'])
                model.eval()
            elif method in RESIDUAL_METHODS:
                model, state = load_trial_checkpoint(directory/f'{method}.pt')
                groups = state['groups']
            else:
                from .hourly_residual_control import load_summary_checkpoint
                model, state = load_summary_checkpoint(directory/f'{method}.pt')
                groups = read_json(workspace/'config/hourly_task_candidates.json')['output_groups']
            for name in ('mean', 'scale'):
                saved = state.get('scaler_'+name, state.get(name))
                if saved is None:
                    raise ValueError('Checkpoint lacks frozen normalization provenance')
                np.testing.assert_array_equal(np.asarray(saved), scalers[name])
            parts = []
            with torch.no_grad():
                for start in range(0, len(baseline.features), batch_size):
                    indices = np.arange(start, min(start+batch_size, len(baseline.features)))
                    model_args, _, _ = neural.batch(indices, groups)
                    if method in NEURAL_METHODS:
                        standardized = flatten_predictions(model(*model_args), groups, neural.variables)
                    else:
                        baseline_z = torch.from_numpy((reference[indices]-baseline.mean[None, None, :, None])/baseline.scale[None, None, :, None])
                        standardized = model(model_args, torch.from_numpy(baseline.features[indices]), baseline_z)
                    parts.append(standardized.numpy()*baseline.scale[None, None, :, None]+baseline.mean[None, None, :, None])
            prediction = np.concatenate(parts)
        assert_ordered(prediction)
        outputs[method] = prediction.astype(np.float32)
    return outputs, baseline


def _source_freeze(workspace):
    workspace = Path(workspace)
    source = workspace/'src/astronaut_twin/ensemble_v2.py'
    if not source.exists() or sha256(source) != sha256(Path(__file__)):
        raise ValueError('Use the exact frozen ensemble source copied into the V2 workspace')
    paths = [source, workspace/'campaign_config.json']
    for relative in ('src/astronaut_twin/hourly_data.py', 'src/astronaut_twin/hourly_quality.py',
                     'config/hourly_quality_policy.json', 'config/hourly_task_candidates.json'):
        staged = workspace/relative
        actual = Path(__file__).resolve().parents[2]/relative
        if not staged.exists() or sha256(staged) != sha256(actual):
            raise ValueError('Role builder, quality policy or task metadata differs from the frozen snapshot')
        paths.append(staged)
    evaluator = workspace/'scripts/evaluate_ensemble_v2.py'
    if evaluator.exists():
        paths.append(evaluator)
    current = {str(path.relative_to(workspace)): sha256(path) for path in paths}
    path = workspace/'ensemble_source_frozen.json'
    if path.exists():
        previous = read_json(path)
        if previous['sha256'] != current or previous.get('components_frozen_sha256') != sha256(workspace/'components_frozen.json'):
            raise ValueError('Ensemble code or predeclared configuration changed after source freeze')
    else:
        write_json_new(path, {'sha256': current, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                             'components_frozen_sha256': sha256(workspace/'components_frozen.json')})
    return sha256(path)


def run_meta(workspace):
    workspace = Path(workspace).resolve()
    freeze = verify_components(workspace)
    source_hash = _source_freeze(workspace)
    complete = workspace/'meta_complete.json'
    destination = workspace/'meta_predictions.npz'
    if complete.exists():
        saved = read_json(complete)
        if (saved['components_frozen_sha256'] != sha256(workspace/'components_frozen.json')
                or saved.get('ensemble_source_frozen_sha256') != source_hash
                or saved['predictions_sha256'] != sha256(destination)):
            raise ValueError('META cache changed after frozen inference')
        return saved
    packed = pack_role(workspace, 'meta')
    fit_path = workspace/'data/processed/hourly/hourly_dataset.npz'
    with np.load(fit_path, allow_pickle=False) as fitting:
        fitted_episode = np.searchsorted(fitting['episode_offsets'], fitting['origins'], side='right')-1
        fitted_ids = set(fitting['episode_ids'][fitted_episode].astype(str))
        fitted_roles = set(fitting['episode_split'][fitted_episode].astype(str))
    if not fitted_roles.issubset({'train', 'dev'}) or set(packed['episode_ids'].astype(str)) & fitted_ids:
        raise ValueError('META overlaps component fitting or selection records')
    outputs, baseline = infer_models(workspace, packed, freeze['methods'])
    np.savez_compressed(destination, methods=np.asarray(freeze['methods']),
        predictions=np.stack([outputs[method] for method in freeze['methods']]),
        labels=baseline.labels, scale=baseline.scale,
        episode_ids=packed['episode_ids'], origin_episode_ids=packed['episode_ids'][packed['sample_episode']])
    result = {'status': 'frozen_component_META_inference_complete', 'role': 'meta',
        'episodes': len(packed['episode_ids']), 'origins': len(packed['origins']),
        'component_methods': freeze['methods'], 'meta_disjoint_from_component_fit': True,
        'OOF_claim': False, 'components_frozen_sha256': sha256(workspace/'components_frozen.json'),
        'ensemble_source_frozen_sha256': source_hash, 'predictions_sha256': sha256(destination),
        'packed_sha256': sha256(workspace/'role_data/meta/packed.npz')}
    write_json_new(complete, result)
    return result


def fit_ensemble(workspace):
    workspace = Path(workspace).resolve()
    verify_components(workspace)
    _source_freeze(workspace)
    destination = workspace/'ensemble_frozen.json'
    if destination.exists():
        return _frozen_gate(workspace, 'ensemble_frozen.json')
    complete = read_json(workspace/'meta_complete.json')
    if complete.get('ensemble_source_frozen_sha256') != sha256(workspace/'ensemble_source_frozen.json'):
        raise ValueError('META cache came from different ensemble source/configuration')
    if complete['components_frozen_sha256'] != sha256(workspace/'components_frozen.json'):
        raise ValueError('META predictions are not from the current frozen components')
    if complete['predictions_sha256'] != sha256(workspace/'meta_predictions.npz'):
        raise ValueError('META predictions changed')
    data = load_npz(workspace/'meta_predictions.npz')
    specification = _config(workspace)['ensemble']
    if specification['training_strategy'] != 'disjoint_meta_holdout_not_OOF_this_round':
        raise ValueError('This round implements disjoint META, not an OOF experiment')
    torch.set_num_threads(2)
    fitted = fit_global_weights(data['predictions'], data['labels'], data['scale'],
        regularization=float(specification['regularization_strength']),
        learning_rate=float(specification['learning_rate']), epochs=int(specification['epochs']))
    pd.DataFrame(fitted['trace']).to_csv(workspace/'ensemble_weight_learning_curve.csv', index=False)
    result = {key: value for key, value in fitted.items() if key not in {'trace', 'weights', 'meta_counts'}}
    result.update(status='ensemble_weights_frozen', methods=data['methods'].astype(str).tolist(),
        weights=fitted['weights'].tolist(), meta_counts=fitted['meta_counts'].tolist(),
        meta_episode_count=len(data['episode_ids']), meta_disjoint_from_component_fit=True,
        components_frozen_sha256=sha256(workspace/'components_frozen.json'),
        meta_predictions_sha256=sha256(workspace/'meta_predictions.npz'),
        ensemble_source_frozen_sha256=sha256(workspace/'ensemble_source_frozen.json'),
        calibration_used=False, final_used=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat())
    write_json_new(destination, result)
    return result


def _reported_individuals(freeze):
    best = freeze.get('best_individual_dev')
    if not isinstance(best, str) or best not in METHOD_DIRECTORIES:
        raise ValueError('Best individual must be selected on DEV before calibration/final')
    return list(dict.fromkeys(['linear_quantile', best]))


def calibrate_ensemble(workspace):
    workspace = Path(workspace).resolve()
    freeze = verify_components(workspace)
    _source_freeze(workspace)
    ensemble = _frozen_gate(workspace, 'ensemble_frozen.json')
    destination = workspace/'calibration_frozen.json'
    if destination.exists():
        saved = _frozen_gate(workspace, 'calibration_frozen.json')
        if saved['ensemble_frozen_sha256'] != sha256(workspace/'ensemble_frozen.json'):
            raise ValueError('Calibration weights changed')
        return saved
    packed = pack_role(workspace, 'calibration')
    individuals = _reported_individuals(freeze)
    methods = list(dict.fromkeys(ensemble['methods']+individuals))
    predictions, baseline = infer_models(workspace, packed, methods)
    predictions['ensemble'] = convex_combine(np.stack([predictions[method] for method in ensemble['methods']]), ensemble['weights'])
    calibration_methods = ['ensemble']+individuals
    specification = _config(workspace)['calibration']
    fitted = {}
    directory = workspace/'calibration'
    directory.mkdir(exist_ok=True)
    rows = []
    origin_ids = packed['episode_ids'][packed['sample_episode']]
    for method in calibration_methods:
        values = empirical_expansion(predictions[method], baseline.labels,
            alpha=float(specification['alpha']), minimum_labels=int(specification['minimum_labels']), episode=origin_ids)
        path = directory/f'{method}.npz'
        np.savez_compressed(path, **{key: value for key, value in values.items() if isinstance(value, np.ndarray)})
        fitted[method] = {'path': str(path.relative_to(workspace)), 'sha256': sha256(path)}
        for h, horizon in enumerate(packed['horizons']):
            for v, variable in enumerate(packed['variables']):
                rows.append({'method': method, 'variable': variable, 'horizon_hours': int(horizon),
                    'expansion': float(values['expansion'][h, v]), 'direct_labels': int(values['counts'][h, v]),
                    'fit_labels': int(values['fitted_counts'][h, v]), 'direct_episodes': int(values['episode_counts'][h, v]),
                    'fit_episodes': int(values['fitted_episode_counts'][h, v]), 'source': values['source'][h, v]})
    pd.DataFrame(rows).to_csv(directory/'support_and_expansion.csv', index=False)
    result = {'status': 'empirical_interval_calibration_frozen', 'methods': calibration_methods,
        'alpha': float(specification['alpha']), 'minimum_labels': int(specification['minimum_labels']),
        'artifacts': fitted, 'episodes': len(packed['episode_ids']),
        'components_frozen_sha256': sha256(workspace/'components_frozen.json'),
        'ensemble_frozen_sha256': sha256(workspace/'ensemble_frozen.json'),
        'ensemble_source_frozen_sha256': sha256(workspace/'ensemble_source_frozen.json'),
        'calibration_packed_sha256': sha256(workspace/'role_data/calibration/packed.npz'),
        'formal_coverage_claim': False, 'final_used': False,
        'dependence_note': 'Hourly origins and labels are dependent; no distribution-free80percent coverage promise.',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json_new(destination, result)
    return result


def training_support(workspace):
    """Frozen fit counts; missing episode support is explicit, never adequate."""
    workspace = Path(workspace)
    base = workspace/'outputs/hourly_baselines'
    linear = load_npz(base/'linear_quantile.npz')
    if 'train_target_counts' not in linear:
        raise ValueError('Frozen baseline must expose actual training target counts')
    counts = np.asarray(linear['train_target_counts'], dtype=np.int64)
    if counts.shape != (4, 34) or (counts < 0).any():
        raise ValueError('Training target count shape is invalid')
    episodes = np.full(counts.shape, -1, dtype=np.int64)
    source = _source_root(workspace)/'outputs/data_audit_v2/training_labels_by_variable_horizon.csv'
    if source.exists():
        audit = pd.read_csv(source)
        variables = load_npz(base/'scalers.npz')['variables'].astype(str).tolist()
        for _, row in audit.iterrows():
            v = variables.index(str(row['variable']))
            h = [1, 4, 8, 24].index(int(row['horizon_hours']))
            if int(row['new_train_labels']) == counts[h, v]:
                episodes[h, v] = int(row['new_train_episodes'])
    return counts, episodes


def prediction_table(method, prediction, packed, baseline, metadata, split='final', calibration=None, train_counts=None, train_episodes=None):
    assert_ordered(prediction)
    local, h, v = np.nonzero(np.isfinite(baseline.labels))
    episode = packed['sample_episode'][local]
    origins = packed['origins'][local]
    variables = packed['variables'].astype(str)
    details = {entry['source_key']: entry for entry in metadata['variables']}
    q = prediction[local, h, v]
    if train_counts is None:
        raise ValueError('Actual frozen training label support must accompany every method')
    support = np.asarray(train_counts)[h, v]
    episode_support = np.asarray(train_episodes)[h, v] if train_episodes is not None else np.full(len(local), -1)
    table = pd.DataFrame({'split': split, 'method': method,
        'episode_id': packed['episode_ids'][episode], 'site': packed['sites'][episode],
        'origin_hour': packed['times'][origins], 'target_hour': packed['times'][origins]+packed['horizons'][h],
        'horizon_hours': packed['horizons'][h], 'variable': variables[v],
        'head': [details[name]['group'] for name in variables[v]],
        'unit': [details[name]['source_unit_as_documented'] for name in variables[v]],
        'observed': baseline.labels[local, h, v], 'q10': q[:, 0], 'q50': q[:, 1], 'q90': q[:, 2],
        'target_scale': baseline.scale[v], 'history_length': baseline.history_length[local],
        'history_hours': baseline.history_hours[local], 'own_history_observed': baseline.counts[local, v],
        'train_target_count': support, 'train_target_episode_count': episode_support,
        'train_episode_support_known': episode_support >= 0,
        'prediction_status': np.where(support == 0, 'no_train_labels_fallback_or_untrained',
            np.where(support < 20, 'low_train_label_support', 'fitted_train_support_reported')),
        'interval_type': 'empirical_adjusted_interval' if calibration is not None else 'estimated_quantiles',
        'unit_review_required': [name in {'FiO2', 'Magnesium', 'Lactate'} for name in variables[v]]})
    if calibration is not None:
        table['calibration_source'] = calibration['source'][h, v]
        table['calibration_direct_labels'] = calibration['counts'][h, v]
        table['calibration_fit_labels'] = calibration['fitted_counts'][h, v]
        table['calibration_fit_episodes'] = calibration['fitted_episode_counts'][h, v]
    policy = read_json(Path(__file__).resolve().parents[2]/'config/hourly_quality_policy.json')['variables']
    below, above = np.zeros(q.shape, dtype=bool), np.zeros(q.shape, dtype=bool)
    for column, name in enumerate(variables):
        rows = v == column
        bounds = policy[name]
        if bounds.get('minimum_inclusive') is not None:
            below[rows] = q[rows] < bounds['minimum_inclusive']
        if bounds.get('maximum_inclusive') is not None:
            above[rows] = q[rows] > bounds['maximum_inclusive']
    table['prediction_below_physical_support'] = below.any(axis=1)
    table['prediction_above_physical_support'] = above.any(axis=1)
    table['prediction_outside_physical_support'] = (below | above).any(axis=1)
    return table


def summarize_predictions(table, destination, best_individual=None):
    """Same actual keys, macro variable-horizon scores, paired differences."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    groups = []
    reference_keys = None
    for method, frame in table.groupby('method', sort=False):
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate evaluation keys')
        keyed = frame.set_index(KEYS).sort_index()
        if reference_keys is None:
            reference_keys = keyed[['observed', 'target_scale']]
        else:
            pd.testing.assert_index_equal(keyed.index, reference_keys.index)
            np.testing.assert_allclose(keyed.observed, reference_keys.observed, rtol=1e-6)
            np.testing.assert_allclose(keyed.target_scale, reference_keys.target_scale, rtol=1e-6)
        prediction = frame[['q10', 'q50', 'q90']].to_numpy()
        assert_ordered(prediction)
        error = frame.observed.to_numpy()[:, None]-prediction
        frame = frame.copy()
        frame['pinball_raw'] = np.maximum(QUANTILES*error, (QUANTILES-1)*error).mean(-1)
        frame['pinball_standardized'] = frame.pinball_raw/frame.target_scale
        frame['absolute_error'] = np.abs(frame.observed-frame.q50)
        frame['absolute_error_standardized'] = frame.absolute_error/frame.target_scale
        frame['covered'] = (frame.observed >= frame.q10) & (frame.observed <= frame.q90)
        frame['interval_width'] = frame.q90-frame.q10
        frame['interval_width_standardized'] = frame.interval_width/frame.target_scale
        groups.append(frame)
    scored = pd.concat(groups, ignore_index=True)
    cells = scored.groupby(['method', 'head', 'variable', 'horizon_hours']).agg(
        observed_labels=('observed', 'size'), episodes=('episode_id', 'nunique'),
        train_target_count=('train_target_count', 'first'), train_target_episode_count=('train_target_episode_count', 'first'),
        train_episode_support_known=('train_episode_support_known', 'first'),
        unit_review_required=('unit_review_required', 'first'),
        prediction_status=('prediction_status', 'first'), interval_type=('interval_type', 'first'),
        outside_physical_support=('prediction_outside_physical_support', 'sum'),
        below_physical_support=('prediction_below_physical_support', 'sum'),
        above_physical_support=('prediction_above_physical_support', 'sum'),
        pinball_raw=('pinball_raw', 'mean'), pinball_standardized=('pinball_standardized', 'mean'),
        mae_raw=('absolute_error', 'mean'), mae_standardized=('absolute_error_standardized', 'mean'),
        coverage80_empirical=('covered', 'mean'), width_raw=('interval_width', 'mean'),
        width_standardized=('interval_width_standardized', 'mean')).reset_index()
    cells.to_csv(destination/'metrics_by_variable_horizon.csv', index=False)
    normalized = ['pinball_standardized', 'mae_standardized', 'coverage80_empirical', 'width_standardized']
    heads = cells.groupby(['method', 'head', 'horizon_hours'])[normalized].mean().reset_index()
    heads.to_csv(destination/'metrics_by_head_horizon.csv', index=False)
    overall = cells.groupby('method')[normalized].mean().reset_index()
    overall.to_csv(destination/'method_scores.csv', index=False)
    difference_rows = []
    for variant in ('raw', 'calibrated'):
        references = list(dict.fromkeys(['linear_quantile_'+variant]+([best_individual+'_'+variant] if best_individual else [])))
        for reference_method in references:
            reference = cells[cells.method.eq(reference_method)]
            for method in cells.method.unique():
                if method == reference_method or not method.endswith('_'+variant):
                    continue
                selected = cells[cells.method.eq(method)]
                joined = selected.merge(reference, on=['head', 'variable', 'horizon_hours'], validate='one_to_one', suffixes=('', '_reference'))
                for _, row in joined.iterrows():
                    difference_rows.append({'method': method, 'reference': reference_method,
                    'head': row['head'], 'variable': row['variable'], 'horizon_hours': int(row['horizon_hours']),
                    'observed_labels': int(row['observed_labels']),
                    'pinball_standardized_difference': row['pinball_standardized']-row['pinball_standardized_reference'],
                    'mae_standardized_difference': row['mae_standardized']-row['mae_standardized_reference'],
                    'coverage_difference': row['coverage80_empirical']-row['coverage80_empirical_reference'],
                    'width_standardized_difference': row['width_standardized']-row['width_standardized_reference']})
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(destination/'paired_final_differences.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = overall.sort_values('pinball_standardized')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(plot.method, plot.pinball_standardized)
    ax.invert_yaxis()
    ax.set_xlabel('Macro pinball estandarizado · menor es mejor')
    ax.set_title('Evaluación final congelada · mismos objetivos observados')
    fig.tight_layout()
    fig.savefig(destination/'final_macro_pinball.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for method, subset in heads.groupby('method'):
        values = subset.groupby('horizon_hours').coverage80_empirical.mean()
        ax.plot(values.index, values.values, marker='o', label=method)
    ax.axhline(.8, linestyle='--', color='gray', label='Nominal0,8; sin garantía formal')
    ax.set_xlabel('Horizonte en horas')
    ax.set_ylabel('Cobertura empírica macro')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destination/'final_coverage_by_horizon.png', dpi=150)
    plt.close(fig)
    return {'observed_pairs_per_method': table.groupby('method').size().to_dict(),
            'scores': overall.to_dict(orient='records'), 'same_observed_keys': True}


def evaluate_final(workspace, final_npz=None):
    workspace = Path(workspace).resolve()
    freeze = verify_components(workspace)
    _source_freeze(workspace)
    ensemble = _frozen_gate(workspace, 'ensemble_frozen.json')
    calibration = _frozen_gate(workspace, 'calibration_frozen.json')
    frozen_hashes = {name: sha256(workspace/name) for name in
                     ('components_frozen.json', 'ensemble_frozen.json', 'calibration_frozen.json')}
    if calibration['ensemble_frozen_sha256'] != sha256(workspace/'ensemble_frozen.json'):
        raise ValueError('Calibration does not bind the frozen ensemble')
    for artifact in calibration['artifacts'].values():
        path = (workspace/artifact['path']).resolve()
        if not path.is_relative_to(workspace) or sha256(path) != artifact['sha256']:
            raise ValueError('Frozen calibration artifact changed')
    if (workspace/'final_complete.json').exists() or (workspace/'final_started.json').exists():
        raise FileExistsError('Final evaluation was already started; no automatic second evaluation')
    canonical = workspace/'role_data/final/packed.npz'
    if final_npz is not None and Path(final_npz).resolve() != canonical.resolve():
        raise ValueError('Only canonical role_data/final/packed.npz from exact audited staging is accepted')
    write_json_new(workspace/'final_started.json', {'status': 'one_shot_final_started',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'components_frozen_sha256': sha256(workspace/'components_frozen.json'),
        'ensemble_frozen_sha256': sha256(workspace/'ensemble_frozen.json'),
        'calibration_frozen_sha256': sha256(workspace/'calibration_frozen.json')})
    packed = pack_role(workspace, 'final')
    stage = read_json(workspace/'role_data/final/complete.json')
    if sha256(canonical) != stage['packed_sha256'] or not np.all(packed['episode_split'] == 'final'):
        raise ValueError('Final packed content differs from its verified role-stage record')
    individuals = _reported_individuals(freeze)
    needed = list(dict.fromkeys(ensemble['methods']+individuals))
    outputs, baseline = infer_models(workspace, packed, needed)
    outputs['ensemble'] = convex_combine(np.stack([outputs[method] for method in ensemble['methods']]), ensemble['weights'])
    metadata = read_json(workspace/'config/hourly_task_candidates.json')
    train_counts, train_episodes = training_support(workspace)
    tables = []
    for method in ['ensemble']+individuals:
        tables.append(prediction_table(method+'_raw', outputs[method], packed, baseline, metadata,
                                      train_counts=train_counts, train_episodes=train_episodes))
        fitted = load_npz(workspace/calibration['artifacts'][method]['path'])
        calibrated = apply_expansion(outputs[method], fitted['expansion'])
        tables.append(prediction_table(method+'_calibrated', calibrated, packed, baseline, metadata, calibration=fitted,
                                      train_counts=train_counts, train_episodes=train_episodes))
    predictions = pd.concat(tables, ignore_index=True)
    destination = workspace/'final_evaluation'
    destination.mkdir(exist_ok=True)
    predictions.to_csv(destination/'predictions.csv', index=False)
    summary = summarize_predictions(predictions, destination, best_individual=freeze['best_individual_dev'])
    verify_components(workspace)
    _source_freeze(workspace)
    if any(sha256(workspace/name) != expected for name, expected in frozen_hashes.items()):
        raise ValueError('Frozen state changed during final inference')
    result = {'status': 'one_shot_frozen_final_complete', 'episodes': len(packed['episode_ids']),
        'origins': len(packed['origins']), 'best_individual_dev': freeze['best_individual_dev'],
        'calibration_formal_coverage_claim': False, 'raw_and_calibrated_same_targets': True,
        'calibrated_bound_semantics': 'Adjusted empirical80percent interval endpoints; not certified marginal10th/90th quantiles. Median unchanged.',
        'components_frozen_sha256': sha256(workspace/'components_frozen.json'),
        'ensemble_frozen_sha256': sha256(workspace/'ensemble_frozen.json'),
        'calibration_frozen_sha256': sha256(workspace/'calibration_frozen.json'),
        'packed_sha256': sha256(canonical), 'predictions_sha256': sha256(destination/'predictions.csv'),
        'finished_at_utc': datetime.now(timezone.utc).isoformat(), **summary}
    write_json_new(workspace/'final_complete.json', result)
    return result
