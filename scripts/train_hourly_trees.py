"""Train tree quantile candidates on train, select/export development only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from astronaut_twin.hourly_baselines import prepare_baseline_data, FEATURE_NAMES
from astronaut_twin.hourly_tree_models import (METHODS, HGB_CONFIGS, QRF_CONFIGS, QUANTILES,
                                               fit_cell, predict_cell, feature_columns, ConstantQuantiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/hourly/hourly_dataset.npz')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/hourly_trees')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(args.data, allow_pickle=False) as archive:
        packed = {name: archive[name] for name in archive.files}
    with threadpool_limits(limits=2):
        data = prepare_baseline_data(packed)
    train = np.flatnonzero(data.split == 'train')
    dev = np.flatnonzero(data.split == 'dev')
    flat = data.features.reshape(len(data.features), -1)
    config_path = ROOT/'config/hourly_task_candidates.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    variables = packed['variables'].astype(str)
    groups = config['groups']
    metadata = {item['source_key']: item for item in config['variables']}
    scaler_values = dict(mean=data.mean, scale=data.scale, median=data.median,
        train_row_indices=data.train_row_indices, observed_counts=data.scaler_counts,
        variables=variables, feature_names=np.asarray(FEATURE_NAMES))
    if (args.out/'scalers.npz').exists():
        with np.load(args.out/'scalers.npz', allow_pickle=False) as previous:
            for key, value in scaler_values.items():
                np.testing.assert_array_equal(previous[key], value, err_msg='Cannot resume with changed scalers or training rows')
    else:
        np.savez_compressed(args.out/'scalers.npz', **scaler_values)
    signature = dict(dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        task_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), seed=args.seed,
        hgb_configs=HGB_CONFIGS, qrf_configs=QRF_CONFIGS, feature_names=list(FEATURE_NAMES))
    signature_path = args.out/'run_signature.json'
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError('Cannot resume artifacts with different frozen data/configuration')
    else:
        signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    signature_digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    manifest = {'status': 'development_only_tree_trials', 'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'task_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'implementation_sha256': hashlib.sha256((ROOT/'src/astronaut_twin/hourly_tree_models.py').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': args.seed, 'threads': 2, 'sklearn_version': sklearn.__version__, 'quantiles': QUANTILES.tolist(),
        'calibration_or_test_used': False, 'export_splits': ['dev'], 'selection': {}, 'candidates': [],
        'selection_metric': 'dev standardized pinball macro over nonempty variable-horizon cells',
        'features': {'own': list(FEATURE_NAMES), 'group': 'concatenate the same9 causal features of variables in the target biomedical group'},
        'label_policy': 'all valid exact-hour labels independently per output; no all-target completeness filter',
        'small_support_policy': 'single/constant target -> empirical train quantiles; zero target -> all unique train values of that variable',
        'boosting': 'HistGradientBoostingRegressor quantile losses; early_stopping=False; max_bins63',
        'qrf': 'leaf-response conditional CDF; max_samples_leaf=None,weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True; sqrt features,bootstrap fraction0.7',
        'qrf_reference': 'https://zillow.github.io/quantile-forest/user_guide/fit_predict.html',
        'groups': groups, 'variables': variables.tolist(), 'horizons_hours': packed['horizons'].tolist(),
        'train_target_counts': np.isfinite(data.labels[train]).sum(0).tolist(),
        'sample_counts': {'train': len(train), 'dev': len(dev)},
        'run_signature_sha256': signature_digest,
        'resume_policy': 'Preserve valid joblib cells; verify frozen scalers/signature, cell metadata and estimator parameters. Initial interrupted pre-signature artifacts adopted against their unchanged original frozen dataset and exact parameters.',
        'candidate_cache': 'Predictions may be NaN at unobserved development targets; exports contain every actual observed target.'}
    import quantile_forest
    manifest['quantile_forest_version'] = quantile_forest.__version__
    all_exports = []
    cell_audit = []
    for method in args.methods:
        candidates = QRF_CONFIGS if method == 'quantile_forest' else HGB_CONFIGS
        best = None
        for candidate_index, parameters in enumerate(candidates):
            candidate_started = time.monotonic()
            model_directory = args.out/method/f'candidate_{candidate_index}'
            model_directory.mkdir(parents=True, exist_ok=True)
            predictions = np.full((len(dev), 4, 34, 3), np.nan, dtype=np.float32)
            losses = []
            reused_cells = 0
            for h in range(4):
                for v in range(34):
                    columns = feature_columns(v, method, variables, groups)
                    valid = np.isfinite(data.labels[train, h, v])
                    y = (data.labels[train[valid], h, v]-data.mean[v])/data.scale[v]
                    x = flat[train[valid]][:, columns]
                    fallback = False
                    if not len(y):
                        fallback = True
                        source = packed['values'][data.train_row_indices, v]
                        source = source[np.isfinite(source)]
                        y = (source-data.mean[v])/data.scale[v] if len(source) else np.array([0.])
                        x = np.zeros((len(y), len(columns)), dtype=np.float32)
                    model_path = model_directory/f'h{h}_v{v}.joblib'
                    saved = None
                    if model_path.exists():
                        try:
                            saved = joblib.load(model_path)
                        except (EOFError, ValueError, OSError):
                            print(f'Refitting incomplete checkpoint {model_path}', flush=True)
                        if saved is not None:
                            if (saved['method'] != method or saved['variable'] != variables[v]
                                or saved['horizon_hours'] != int(packed['horizons'][h])
                                or saved['feature_columns'] != columns):
                                raise ValueError(f'Checkpoint metadata mismatch: {model_path}')
                            if saved.get('run_signature_sha256', signature_digest) != signature_digest:
                                raise ValueError('Checkpoint frozen-data signature mismatch')
                            model = saved['model']
                            if not isinstance(model, ConstantQuantiles):
                                estimators = [model] if method == 'quantile_forest' else model
                                for estimator in estimators:
                                    if any(estimator.get_params()[key] != value for key, value in parameters.items()):
                                        raise ValueError('Checkpoint hyperparameters differ')
                            reused_cells += 1
                    if saved is None:
                        model = fit_cell(x, y, method, parameters, args.seed+h*34+v)
                    target = (data.labels[dev, h, v]-data.mean[v])/data.scale[v]
                    observed = np.isfinite(target)
                    # Score/export only observed targets; do not spend QRF
                    # inference on missing labels or manufacture those labels.
                    predicted = predict_cell(model, flat[dev[observed]][:, columns], method) if observed.any() else np.empty((0, 3))
                    predictions[observed, h, v] = predicted*data.scale[v]+data.mean[v]
                    score = None
                    if observed.any():
                        error = target[observed, None]-predicted
                        score = float(np.maximum(QUANTILES*error, (QUANTILES-1)*error).mean())
                        losses.append(score)
                    if saved is None:
                        temporary = model_path.with_suffix('.joblib.partial')
                        joblib.dump({'model': model, 'feature_columns': columns, 'method': method,
                                     'variable': variables[v], 'horizon_hours': int(packed['horizons'][h]),
                                     'run_signature_sha256': signature_digest}, temporary, compress=3)
                        temporary.replace(model_path)
                    cell_audit.append({'method': method, 'candidate': candidate_index, 'variable': variables[v],
                        'horizon_hours': int(packed['horizons'][h]), 'train_labels': int(valid.sum()),
                        'dev_labels': int(observed.sum()), 'dev_standardized_pinball': score,
                        'train_only_fallback': fallback, 'feature_count': len(columns)})
                    if (h*34+v+1) % 17 == 0:
                        pd.DataFrame(cell_audit).to_csv(args.out/'candidate_cell_metrics.csv', index=False)
                        (args.out/'progress.json').write_text(json.dumps({'method': method, 'candidate': candidate_index,
                            'cells': h*34+v+1, 'reused_cells': reused_cells,
                            'elapsed_seconds': round(time.monotonic()-started, 1)}, indent=2), encoding='utf-8')
                        print(f'{method} candidate{candidate_index} cells={h*34+v+1}/136 seconds={time.monotonic()-candidate_started:.1f}', flush=True)
            objective = float(np.mean(losses))
            info = {'method': method, 'candidate': candidate_index, 'parameters': parameters,
                    'dev_macro_pinball': objective, 'seconds': round(time.monotonic()-candidate_started, 2),
                    'model_directory': str(model_directory.relative_to(args.out)), 'reused_cells': reused_cells}
            manifest['candidates'].append(info)
            if (model_directory/'dev_predictions.npz').exists():
                with np.load(model_directory/'dev_predictions.npz') as prior:
                    np.testing.assert_array_equal(prior['sample_indices'], dev)
                    mask = np.isfinite(data.labels[dev])
                    np.testing.assert_allclose(prior['predictions'][mask], predictions[mask], rtol=1e-6, atol=1e-5)
            else:
                np.savez_compressed(model_directory/'dev_predictions.npz', predictions=predictions, sample_indices=dev)
            if best is None or objective < best['dev_macro_pinball']:
                best = {**info, 'predictions': predictions}
            pd.DataFrame(cell_audit).to_csv(args.out/'candidate_cell_metrics.csv', index=False)
            print(f'FINISHED {method} candidate{candidate_index} dev_macro={objective:.6f}', flush=True)
        predictions = best.pop('predictions')
        manifest['selection'][method] = best
        (args.out/'fit_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        labels = data.labels[dev]
        local, h, v = np.nonzero(np.isfinite(labels))
        samples = dev[local]
        ep = data.episode[samples]
        origins = packed['origins'][samples]
        q = predictions[local, h, v]
        table = pd.DataFrame({'split': 'dev', 'method': method, 'episode_id': packed['episode_ids'][ep],
            'site': packed['sites'][ep], 'origin_hour': packed['times'][origins],
            'target_hour': packed['times'][origins]+packed['horizons'][h], 'horizon_hours': packed['horizons'][h],
            'head': [metadata[name]['group'] for name in variables[v]], 'variable': variables[v],
            'unit': [metadata[name]['source_unit_as_documented'] for name in variables[v]],
            'unit_review_required': np.isin(variables[v], ['FiO2', 'Magnesium', 'Lactate']),
            'observed': labels[local, h, v], 'q10': q[:, 0], 'q50': q[:, 1], 'q90': q[:, 2],
            'target_scale': data.scale[v], 'history_hours': data.history_hours[samples],
            'history_length': data.history_length[samples], 'own_history_observed': data.counts[samples, v],
            'used_train_median_fallback': data.counts[samples, v] == 0,
            'train_target_count': np.isfinite(data.labels[train]).sum(0)[h, v],
            'prediction_status': np.where(np.isfinite(data.labels[train]).sum(0)[h, v] == 0,
                'no_train_labels_train_only_fallback', np.where(np.isfinite(data.labels[train]).sum(0)[h, v] < 20, 'low_train_support', 'fitted'))})
        all_exports.append(table)
        pd.concat(all_exports, ignore_index=True).to_csv(args.out/'predictions.csv', index=False)
    manifest['elapsed_seconds'] = round(time.monotonic()-started, 2)
    manifest['prediction_rows'] = sum(len(table) for table in all_exports)
    manifest['status'] = 'development_only_tree_trials_complete'
    (args.out/'fit_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'elapsed_seconds': manifest['elapsed_seconds'], 'selection': manifest['selection']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
