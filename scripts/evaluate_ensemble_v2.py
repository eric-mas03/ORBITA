"""Run gated V2 META/ensemble/CAL/final stages; never train base models here."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
# Resolve the frozen campaign's isolated tree dependency before importing models.
DEFAULT_WORKSPACE = ROOT if (ROOT/'prepared.json').exists() else ROOT/'outputs/model_campaign_v2/workspace'
early = argparse.ArgumentParser(add_help=False)
early.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
early_args, _ = early.parse_known_args()
runtime_config = early_args.workspace/'campaign_config.json'
if runtime_config.exists():
    tree_dependency = json.loads(runtime_config.read_text(encoding='utf-8')).get('original_tree_dependency_path')
    if tree_dependency:
        sys.path.insert(1, str(tree_dependency))
from astronaut_twin.ensemble_v2 import run_meta, fit_ensemble, calibrate_ensemble, evaluate_final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument('--stage', choices=['meta', 'ensemble', 'calibrate', 'final'], required=True)
    # An arbitrary external NPZ cannot bypass the exact FINAL role staging.
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    functions = {'meta': run_meta, 'ensemble': fit_ensemble, 'calibrate': calibrate_ensemble}
    result = evaluate_final(args.workspace, args.dataset) if args.stage == 'final' else functions[args.stage](args.workspace)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
