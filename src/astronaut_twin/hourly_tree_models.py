"""Quantile boosting and leaf-distribution QRF on frozen causal features."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCY_PATH = ROOT/'.deps/tree_models'
if DEPENDENCY_PATH.exists():
    sys.path.insert(0, str(DEPENDENCY_PATH))
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
QUANTILES = np.asarray([.1, .5, .9])
METHODS = ('hist_boost_quantile', 'quantile_forest', 'hist_boost_group_features')
HGB_CONFIGS = [dict(max_iter=100, max_leaf_nodes=15, min_samples_leaf=20),
               dict(max_iter=200, max_leaf_nodes=31, min_samples_leaf=20)]
QRF_CONFIGS = [dict(n_estimators=64, min_samples_leaf=5, max_depth=10),
               dict(n_estimators=128, min_samples_leaf=20, max_depth=10)]


@dataclass
class ConstantQuantiles:
    quantiles: np.ndarray

    def predict(self, features):
        return np.broadcast_to(self.quantiles, (len(features), 3)).copy()


def feature_columns(variable: int, method: str, variables, groups) -> list[int]:
    if method not in METHODS:
        raise ValueError('Unknown tree method')
    selected = [variable]
    if method == 'hist_boost_group_features':
        target = str(variables[variable])
        memberships = [[str(name).removeprefix('SEPSIS2019::') for name in names] for names in groups.values()]
        matching = [members for members in memberships if target in members]
        if len(matching) != 1:
            raise ValueError('Biomedical groups must partition variables')
        selected = [list(variables).index(name) for name in matching[0]]
    return [v*9+feature for v in selected for feature in range(9)]


def fit_cell(features, target, method: str, config: dict, seed=42):
    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target, dtype=np.float64)
    if len(target) < 2 or np.ptp(target) <= 1e-12:
        if not len(target):
            raise ValueError('Caller must provide explicit train-only fallback for an empty target')
        return ConstantQuantiles(np.quantile(target, QUANTILES))
    with threadpool_limits(limits=2):
        if method == 'quantile_forest':
            from quantile_forest import RandomForestQuantileRegressor
            return RandomForestQuantileRegressor(**config, max_samples_leaf=None,
                max_features='sqrt', max_samples=.7, n_jobs=2, random_state=seed,
                bootstrap=True).fit(features, target)
        if method not in METHODS:
            raise ValueError('Unknown tree method')
        # No automatic random validation split or implicit stopping rule.
        return [HistGradientBoostingRegressor(loss='quantile', quantile=float(q),
                 **config, max_bins=63, learning_rate=.1, early_stopping=False,
                 random_state=seed).fit(features, target) for q in QUANTILES]


def predict_cell(model, features, method: str):
    with threadpool_limits(limits=2):
        if isinstance(model, ConstantQuantiles):
            prediction = model.predict(features)
        elif method == 'quantile_forest':
            # Quantiles of weighted response observations inside leaves,
            # NOT percentiles of the trees' conditional-mean predictions.
            prediction = model.predict(features, quantiles=QUANTILES.tolist(),
                weighted_quantile=True, weighted_leaves=True, aggregate_leaves_first=True)
        else:
            prediction = np.column_stack([estimator.predict(features) for estimator in model])
    return np.sort(np.asarray(prediction, dtype=np.float32), axis=-1)


def load_tree_family(directory: str | Path, method: str) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory/'fit_manifest.json').read_text(encoding='utf-8'))
    selection = manifest['selection'][method]
    with np.load(directory/'scalers.npz', allow_pickle=False) as archive:
        scalers = {name: archive[name] for name in archive.files}
    return {'directory': directory, 'method': method, 'selection': selection,
            'manifest': manifest, 'scalers': scalers}


def predict_tree_family(artifact: dict, features: np.ndarray) -> np.ndarray:
    """Fresh inference only; caller supplies causal baseline features[N,34,9]."""
    flat = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
    manifest, method = artifact['manifest'], artifact['method']
    result = np.empty((len(features), 4, 34, 3), dtype=np.float32)
    for h in range(4):
        for v in range(34):
            key = f'h{h}_v{v}'
            path = artifact['directory']/artifact['selection']['model_directory']/(key+'.joblib')
            saved = joblib.load(path)
            predicted = predict_cell(saved['model'], flat[:, saved['feature_columns']], method)
            result[:, h, v] = predicted*artifact['scalers']['scale'][v]+artifact['scalers']['mean'][v]
    return result
