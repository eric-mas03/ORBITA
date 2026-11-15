"""Fixed, honest presentation examples with dense forecasts and rolling historical replay.

Continuous future points are trained 15-minute queries, not interpolated 1/4/8/24h
predictions. Historical points are independent one-hour-ahead replays, each built
only from observations available at its own issue time. No point is forced onto
an actual observation to hide the switch of issue time at the presentation origin.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from .source_expansion import ROOT, SourceTwin, predict

DEFAULT_ARTIFACTS = ROOT / 'outputs/presentation_campaign'
TITLES = {
    'wearable': 'Daily health signals',
    'radiation': 'ISS radiation environment',
    'echo_c1': 'NASA cardiac measurements · C1',
    'echo_c3': 'NASA cardiac measurements · C3',
    'echo_cft': 'NASA cardiac measurements · CFT70',
    'echo_3d': 'NASA cardiac structure · 3D',
    'vascular_c1': 'NASA vascular protocol response',
}
VARIABLE_LABELS = {
    'glucose': 'Glucose', 'heart_rate': 'Heart rate',
    'absorbed_dose_rate': 'Absorbed dose rate',
    'LVDD': 'Ventricular diameter · diastole',
    'LVDS': 'Ventricular diameter · systole',
    'ME': 'Mitral flow · E', 'MA': 'Mitral flow · A',
    'MDT': 'Mitral deceleration time', 'PAT': 'Pulmonary acceleration time',
    'PET': 'Pulmonary ejection time', 'IVRT': 'Ventricular relaxation time',
    'LV mass': 'Ventricular mass', 'LVDV': 'Ventricular volume · diastole',
    'LVSV': 'Ventricular volume · systole',
    'arm': 'Brachial diameter · protocol', 'leg': 'Tibial diameter · protocol',
}


def _directory(path=None):
    return Path(path).resolve() if path is not None else DEFAULT_ARTIFACTS.resolve()


@lru_cache(maxsize=4)
def artifacts(directory=None):
    directory = _directory(directory)
    setup = json.loads((directory / 'setup.json').read_text(encoding='utf8'))
    series = json.loads((directory / 'series.json').read_text(encoding='utf8'))
    saved = torch.load(directory / 'source_twin.pt', map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    model = SourceTwin(setup['specs'], saved['width'])
    model.load_state_dict(saved['state'])
    model.eval()
    lookup = {row['id']: row for row in series}
    if len(lookup) != len(series):
        raise ValueError('Presentation series identifiers must be unique')
    for domain, spec in setup['specs'].items():
        if spec['clock'] == 'hours':
            if len(spec['horizons']) != 96 or not np.allclose(np.asarray(spec['horizons'], dtype=float), np.arange(1, 97) / 4):
                raise ValueError(f'{domain}: presentation requires 96 trained quarter-hour horizons')
    return setup, lookup, model


def _fixed_case(domain, lookup):
    candidates = [s for s in lookup.values() if s['domain'] == domain and s['role'] in ('test', 'temporal')]
    if not candidates:
        raise ValueError(f'No held-out presentation case for {domain}')
    # Fixed identity choice, not a search for the best-looking prediction/error.
    return sorted(candidates, key=lambda s: s['id'])[0]


def _fixed_origin(item, spec):
    n = len(item['times'])
    if spec['clock'] == 'hours':
        if item['role'] == 'temporal':
            # RadLab calibration ends at day 12. All one-hour historical issue
            # times for the previous 24h must lie after that boundary.
            origin = 13 * 96 + 4
        else:
            origin = min(max(n // 2, 196), n - 97)
        if not 100 <= origin < n:
            raise ValueError('Insufficient continuous context for the fixed presentation origin')
        return origin
    return min(n - 1, max(1, min(n // 2, n - 3)))


def _inference(model, samples, lookup, setup, method):
    chunks = []
    for start in range(0, len(samples), 32):
        chunks.append(predict(
            model, samples[start:start + 32], lookup, setup['specs'],
            setup['scalers'], setup['bands'], method,
            setup['ensemble_neural_weight'], setup['calibration_expansion_z'],
        ))
    return np.concatenate(chunks, axis=0)


def build_panel(domain, setup, lookup, model, method='ensemble'):
    """Build one fixed source panel; deliberately accepts no arbitrary patient picker."""
    if method not in ('ensemble', 'transformer', 'persistence'):
        raise ValueError('Unknown forecast method')
    spec = setup['specs'][domain]
    item = _fixed_case(domain, lookup)
    origin = _fixed_origin(item, spec)
    hourly = spec['clock'] == 'hours'
    lead = 1
    lead_steps = 4 if hourly else 1
    window_steps = 96 if hourly else 8
    history_targets = list(range(max(lead_steps, origin - window_steps), origin + 1))
    historical_issues = [target - lead_steps for target in history_targets]
    if hourly:
        keep = [(issue, target) for issue, target in zip(historical_issues, history_targets) if issue >= 95]
        if item['role'] == 'temporal':
            keep = [(issue, target) for issue, target in keep if issue >= 12 * 96]
        historical_issues = [pair[0] for pair in keep]
        history_targets = [pair[1] for pair in keep]
    samples = [{'series': item['id'], 'domain': domain, 'origin': origin}]
    samples.extend({'series': item['id'], 'domain': domain, 'origin': issue} for issue in historical_issues)
    estimates = _inference(model, samples, lookup, setup, method)
    lead_index = next(i for i, h in enumerate(spec['horizons']) if abs(float(h) - lead) < 1e-8)
    values = np.asarray(item['values'], dtype=float)
    first_row = max(0, origin - window_steps)
    last_row = min(len(values) - 1, origin + (96 if hourly else 2))
    variables = []
    for column, name in enumerate(spec['targets']):
        observed = [
            {'x': float(item['times'][i] - item['times'][origin]),
             'index': i, 'timestamp': item['labels'][i], 'value': float(values[i, column]),
             'future_comparison': i > origin}
            for i in range(first_row, last_row + 1) if np.isfinite(values[i, column])
        ]
        historical = []
        for sample_i, (issue, target) in enumerate(zip(historical_issues, history_targets)):
            q10, q50, q90 = map(float, estimates[sample_i + 1, lead_index, column])
            historical.append({
                'x': float(item['times'][target] - item['times'][origin]),
                'issue_index': issue, 'target_index': target,
                'issue_timestamp': item['labels'][issue], 'target_timestamp': item['labels'][target],
                'lead': lead, 'quantile_source': 'rolling_historical_replay',
                'q10': q10, 'q50': q50, 'q90': q90,
                'observed': float(values[target, column]) if np.isfinite(values[target, column]) else None,
            })
        future = []
        for h_index, horizon in enumerate(spec['horizons']):
            target = origin + int(round(float(horizon) * (4 if hourly else 1)))
            q10, q50, q90 = map(float, estimates[0, h_index, column])
            n = int(setup['calibration_counts'][domain][h_index][column])
            future.append({
                'x': float(horizon), 'horizon': float(horizon),
                'issue_index': origin, 'target_index': target,
                'issue_timestamp': item['labels'][origin],
                'target_timestamp': item['labels'][target] if target < len(values) else None,
                'quantile_source': 'dense_trained_query' if hourly else 'trained_visit_query',
                'q10': q10, 'q50': q50, 'q90': q90,
                'observed': float(values[target, column]) if target < len(values) and np.isfinite(values[target, column]) else None,
                'calibration_n': n,
                'calibration_status': 'adjusted' if method == 'ensemble' and n >= 20 else 'insufficient' if method == 'ensemble' else 'raw',
            })
        scored = [row for row in historical if row['observed'] is not None]
        error = [abs(row['observed'] - row['q50']) for row in scored]
        coverage = [row['q10'] <= row['observed'] <= row['q90'] for row in scored]
        available = np.flatnonzero(np.isfinite(values[:origin + 1, column]))
        current = float(values[available[-1], column]) if len(available) else None
        unit = spec['units'][column]
        if 'sin armonizar' in unit:
            unit = 'study units'
        variables.append({
            'id': name, 'label': VARIABLE_LABELS.get(name, name), 'unit': unit,
            'current': current, 'observed': observed,
            'historical': historical, 'future': future,
            'origin_forecast_anchor': historical[-1] if historical and historical[-1]['target_index'] == origin else None,
            'historical_metrics': {'n': len(scored), 'mae': float(np.mean(error)) if error else None,
                                   'coverage': float(np.mean(coverage)) if coverage else None,
                                   'scope': 'previous 24 hours · rolling 1-hour-ahead replay' if hourly else 'prior visits · rolling next-visit replay'},
        })
    return {
        'id': domain, 'title': TITLES.get(domain, domain),
        'case': item['id'], 'person': item['person'], 'role': item['role'],
        'origin_index': origin, 'origin_timestamp': item['labels'][origin],
        'clock': 'hours' if hourly else 'visits', 'historical_lead': lead,
        'future_step': .25 if hourly else 1, 'future_length': len(spec['horizons']),
        'domain': spec['domain'], 'source_url': spec['reference'],
        'evidence_label': 'ISS instrument · not personal dosimetry' if domain == 'radiation' else 'Ground wearable study · not astronaut telemetry' if domain == 'wearable' else 'NASA ground analogue · not in-flight telemetry',
        'history_label': 'Rolling 1-hour-ahead backtest' if hourly else 'Rolling next-visit backtest',
        'future_label': 'Forecast issued at the displayed origin',
        'seam_explanation': 'The point at zero was predicted one lead-time earlier. Subsequent points are forecasts issued at the displayed origin; no point is anchored to an actual measurement.',
        'is_logged_historical_forecast': False, 'is_clinically_validated': False,
        'variables': variables,
    }


@lru_cache(maxsize=8)
def presentation_payload(directory=None, method='ensemble'):
    """Seven fixed evidence panels for one coherent research concept demonstration."""
    setup, lookup, model = artifacts(directory)
    order = ['wearable', 'radiation', 'echo_cft', 'echo_3d', 'echo_c3', 'echo_c1', 'vascular_c1']
    panels = [build_panel(domain, setup, lookup, model, method) for domain in order if domain in setup['specs']]
    return {
        'title': 'ORBITA · A clearer view of crew health', 'panels': panels,
        'method': method, 'clinical_validation': False,
        'context': 'Independent evidence streams support the same research concept. They are not synchronized measurements of a single astronaut.',
        'research_goal': 'Collect more astronaut data, more frequently, to improve prediction quality and validate performance during spaceflight.',
        'continuous_contract': '96 trained 15-minute forecast steps; historical replay uses a fixed 1-hour lead and a fresh past-only window at every issue time.',
    }
