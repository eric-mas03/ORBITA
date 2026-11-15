"""Packed, causal hourly windows with frozen episode-level data splits.

No imputation, label interpolation, normalization or training is performed here.
An ICU file identifies an episode, not a guaranteed unique person.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hourly_quality import apply_quality

DYNAMIC = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
           'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets']
HORIZONS = (1, 4, 8, 24)
SPLITS = ('train', 'dev', 'calibration', 'test')
SPLIT_FRACTIONS = (.60, .15, .10, .15)
SPLIT_SEED = 'hourly-v1-2026-11-14'


def fixed_splits(episode_ids, sites, path: Path | None = None) -> dict[str, str]:
    """Hash-rank episodes within each site, independent of measured values."""
    episode_ids, sites = list(episode_ids), list(sites)
    if len(set(episode_ids)) != len(episode_ids) or len(episode_ids) != len(sites):
        raise ValueError('Episode IDs must be unique and align with sites')
    if path is not None and path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assignment = saved['episode_split']
        if set(assignment) != set(episode_ids):
            raise ValueError('Episode inventory differs from frozen splits; create an explicit new split version')
        if any(saved['episode_site'][name] != site for name, site in zip(episode_ids, sites)):
            raise ValueError('Episode site differs from frozen split metadata')
        if not set(assignment.values()).issubset(SPLITS):
            raise ValueError('Invalid saved split label')
        return assignment
    assignment = {}
    for site in sorted(set(sites)):
        members = [name for name, name_site in zip(episode_ids, sites) if name_site == site]
        members.sort(key=lambda name: hashlib.sha256(f'{SPLIT_SEED}|{site}|{name}'.encode()).hexdigest())
        exact = np.asarray(SPLIT_FRACTIONS) * len(members)
        counts = np.floor(exact).astype(int)
        remainder = len(members)-int(counts.sum())
        for index in np.argsort(-(exact-counts), kind='stable')[:remainder]:
            counts[index] += 1
        offset = 0
        for split, count in zip(SPLITS, counts):
            assignment.update({name: split for name in members[offset:offset+count]})
            offset += count
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'version': 1, 'seed': SPLIT_SEED,
                                   'method': 'site-stratified SHA256 rank; largest-remainder 60/15/10/15',
                                   'unit': 'ICU episode/file; unique person identity unavailable',
                                   'episode_site': dict(zip(episode_ids, sites)),
                                   'episode_split': assignment}, indent=2), encoding='utf-8')
    return assignment


def build_packed(raw: Path, destination: Path, split_path: Path) -> dict:
    files = sorted(raw.rglob('*.psv'))
    if not files:
        raise ValueError('No hourly PSV files found')
    ids = [f'{path.parent.name}/{path.stem}' for path in files]
    sites = [path.parent.name for path in files]
    assignment = fixed_splits(ids, sites, split_path)
    values, times, source_masks, quality_masks = [], [], [], []
    offsets = [0]
    origins, history_starts, targets, sample_episodes = [], [], [], []
    sources, flags, episodes = [], [], []
    for episode, (path, episode_id, site) in enumerate(zip(files, ids, sites)):
        frame = pd.read_csv(path, sep='|')
        if not set(DYNAMIC + ['ICULOS']).issubset(frame):
            raise ValueError(f'Missing required source columns in {episode_id}')
        clock = pd.to_numeric(frame.ICULOS, errors='coerce').to_numpy(dtype=float)
        if not len(clock) or not np.isfinite(clock).all() or not np.equal(clock, np.floor(clock)).all() or (np.diff(clock) <= 0).any():
            raise ValueError(f'ICULOS must be strictly increasing unique integer hours: {episode_id}')
        original = frame[DYNAMIC].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        cleaned, episode_flags = apply_quality(original, DYNAMIC)
        cleaned = np.asarray(cleaned, dtype=np.float32)
        if cleaned.shape != original.shape:
            raise ValueError('Quality policy changed data dimensions')
        observed = np.isfinite(cleaned)
        quality_invalid = np.zeros_like(observed)
        for flag in episode_flags:
            column = DYNAMIC.index(flag['variable'])
            quality_invalid[int(flag['row']), column] = True
            flags.append({**flag, 'episode_id': episode_id, 'source_row': int(flag['row'])+2,
                          'ICULOS': int(clock[int(flag['row'])])})
        start = offsets[-1]
        lookup = {int(t): i for i, t in enumerate(clock)}
        candidates = 0
        for local_origin, current in enumerate(clock):
            # Search real elapsed hours, never cross an episode boundary.
            local_start = int(np.searchsorted(clock, current-23, side='left'))
            if not observed[local_start:local_origin+1].any():
                continue
            future = [lookup.get(int(current)+horizon, -1) for horizon in HORIZONS]
            if not any(row >= 0 and observed[row].any() for row in future):
                continue
            origins.append(start+local_origin)
            history_starts.append(start+local_start)
            targets.append([start+row if row >= 0 else -1 for row in future])
            sample_episodes.append(episode)
            candidates += 1
        values.append(cleaned)
        times.append(clock.astype(np.int64))
        source_masks.append(np.isfinite(original))
        quality_masks.append(quality_invalid)
        offsets.append(start+len(frame))
        body = path.read_bytes()
        sources.append({'episode_id': episode_id, 'source_path': str(path), 'bytes': len(body),
                        'sha256': hashlib.sha256(body).hexdigest()})
        episodes.append({'episode_id': episode_id, 'site': site, 'split': assignment[episode_id],
                         'rows': len(frame), 'eligible_origins': candidates,
                         'gap_events': int((np.diff(clock)>1).sum()), 'quality_invalid_cells': int(quality_invalid.sum())})
    packed = {'values': np.concatenate(values), 'times': np.concatenate(times),
              'source_observed': np.concatenate(source_masks), 'quality_invalid': np.concatenate(quality_masks),
              'episode_offsets': np.asarray(offsets, dtype=np.int64), 'episode_ids': np.asarray(ids),
              'sites': np.asarray(sites), 'episode_split': np.asarray([assignment[name] for name in ids]),
              'origins': np.asarray(origins, dtype=np.int64), 'history_start': np.asarray(history_starts, dtype=np.int64),
              'target_rows': np.asarray(targets, dtype=np.int64).reshape(-1, len(HORIZONS)),
              'sample_episode': np.asarray(sample_episodes, dtype=np.int64),
              'variables': np.asarray(DYNAMIC), 'horizons': np.asarray(HORIZONS, dtype=np.int64)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **packed)
    return {'sources': sources, 'episodes': episodes, 'quality_flags': flags,
            'rows': len(packed['values']), 'samples': len(origins), 'variables': DYNAMIC,
            'horizons_hours': list(HORIZONS), 'context_hours': 24,
            'split_unit': 'ICU episode; person identity unavailable',
            'split_counts': {split: int((packed['episode_split']==split).sum()) for split in SPLITS},
            'sample_counts': {split: int((packed['episode_split'][packed['sample_episode']]==split).sum()) for split in SPLITS}}


class PackedHourlyDataset:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {name: archive[name] for name in archive.files}

    def __len__(self):
        return len(self.arrays['origins'])

    def get_window(self, index: int, pad_to: int | None = None) -> dict:
        """Return one causal history, optional right padding, and sparse labels."""
        data = self.arrays
        origin = int(data['origins'][index])
        start = int(data['history_start'][index])
        episode = int(data['sample_episode'][index])
        history = data['values'][start:origin+1].copy()
        clock = data['times'][start:origin+1].astype(float).copy()
        observed = np.isfinite(history)
        quality = data['quality_invalid'][start:origin+1].copy()
        padding = np.zeros(len(history), dtype=bool)
        if pad_to is not None:
            if pad_to < len(history):
                raise ValueError('pad_to cannot truncate causal history')
            extra = pad_to-len(history)
            history = np.pad(history, ((0, extra), (0, 0)), constant_values=np.nan)
            clock = np.pad(clock, (0, extra), constant_values=np.nan)
            observed = np.pad(observed, ((0, extra), (0, 0)), constant_values=False)
            quality = np.pad(quality, ((0, extra), (0, 0)), constant_values=False)
            padding = np.arange(pad_to) >= len(padding)
        rows = data['target_rows'][index]
        labels = np.full((len(HORIZONS), len(DYNAMIC)), np.nan, dtype=np.float32)
        present = rows >= 0
        labels[present] = data['values'][rows[present]]
        return {'values': history, 'observed': observed, 'times': clock,
                'relative_times': clock-float(data['times'][origin]), 'padding_mask': padding,
                'quality_invalid': quality, 'labels': labels, 'label_mask': np.isfinite(labels),
                'target_rows': rows.copy(), 'origin_row': origin, 'origin_time': int(data['times'][origin]),
                'episode_id': str(data['episode_ids'][episode]), 'site': str(data['sites'][episode]),
                'split': str(data['episode_split'][episode]), 'horizons': data['horizons'].copy()}


def get_window(dataset: PackedHourlyDataset, index: int, pad_to: int | None = None) -> dict:
    return dataset.get_window(index, pad_to=pad_to)
