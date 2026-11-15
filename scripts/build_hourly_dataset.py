"""Build fixed episode splits and causal hourly windows; does not train a model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from astronaut_twin.hourly_data import build_packed, PackedHourlyDataset, DYNAMIC, HORIZONS, SPLITS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0')
    parser.add_argument('--destination', type=Path, default=ROOT/'data/processed/hourly')
    parser.add_argument('--reports', type=Path, default=ROOT/'outputs/hourly_data')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    target = args.destination/'hourly_dataset.npz'
    split_path = args.destination/'splits.json'
    summary = build_packed(args.raw, target, split_path)
    data = PackedHourlyDataset(target).arrays
    rows = []
    sample_splits = data['episode_split'][data['sample_episode']]
    full_context = data['origins']-data['history_start']+1 == 24
    for split in SPLITS:
        selection = sample_splits == split
        for h_index, horizon in enumerate(HORIZONS):
            future = data['target_rows'][:, h_index]
            exists = future >= 0
            labels = np.zeros((len(future), len(DYNAMIC)), dtype=bool)
            labels[exists] = np.isfinite(data['values'][future[exists]])
            for column, variable in enumerate(DYNAMIC):
                active = selection & labels[:, column]
                rows.append({'split': split, 'variable': variable, 'horizon_hours': horizon,
                             'eligible_pairs': int(active.sum()),
                             'episodes_with_pair': int(len(np.unique(data['sample_episode'][active]))),
                             'pairs_full24_context': int((active & full_context).sum())})
    pd.DataFrame(rows).to_csv(args.reports/'eligibility_by_split_target.csv', index=False)
    flags = summary.pop('quality_flags')
    for flag in flags:
        if isinstance(flag.get('original'), (float, np.floating)) and not np.isfinite(flag['original']):
            flag['original'] = str(flag['original'])
    (args.reports/'quality_flags.json').write_text(json.dumps(flags, indent=2, allow_nan=False), encoding='utf-8')
    summary['quality_flag_reasons'] = dict(Counter(flag['reason'] for flag in flags))
    pd.DataFrame(summary['episodes']).to_csv(args.reports/'episode_inventory.csv', index=False)
    (args.destination/'source_hashes.json').write_text(json.dumps(summary.pop('sources'), indent=2), encoding='utf-8')
    policy_paths = [ROOT/'config/hourly_quality_policy.json', ROOT/'src/astronaut_twin/hourly_quality.py']
    summary.update({'built_at_utc': datetime.now(timezone.utc).isoformat(),
                    'status': 'dataset_built_no_model_training', 'dataset_path': str(target),
                    'dataset_bytes': target.stat().st_size, 'dataset_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                    'quality_policy_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in policy_paths},
                    'eligibility': 'At least one valid dynamic input in causal24hour window AND at least one valid future target across any variable/horizon.',
                    'missing_labels': 'Remain NaN with false masks; no fill or interpolation.',
                    'normalization': 'None; fit scalers with training episodes only in subsequent work.',
                    'arrays': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in data.items()}})
    summary.pop('episodes')
    (args.destination/'manifest.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (args.reports/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ('status', 'rows', 'samples', 'split_counts', 'sample_counts', 'quality_flag_reasons')}, indent=2))


if __name__ == '__main__':
    main()
