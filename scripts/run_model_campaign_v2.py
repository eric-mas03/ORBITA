"""Run an isolated, audited V2 campaign without writing V1 model artifacts."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.campaign_v2 import (prepare_campaign, run_individuals, run_worker,
    freeze_components, run_meta, fit_ensemble, calibrate_ensemble, evaluate_final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/model_campaign_v2.json")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--stage", choices=("prepare", "individuals", "select", "meta", "ensemble", "calibrate", "final", "status"), default="status")
    parser.add_argument("--worker", choices=("baselines", "neural", "trees", "architectures", "summary",
                                           "architecture_causal", "architecture_bidir", "architecture_group"))
    args = parser.parse_args()
    if args.worker:
        if not args.workspace:
            parser.error("A worker needs --workspace")
        run_worker(args.workspace.resolve(), args.worker)
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    workspace = (args.workspace or ROOT / config["workspace"]).resolve()
    if args.stage == "prepare":
        if args.dataset:
            config["dataset_path"] = str(args.dataset.resolve())
        if args.audit:
            config["audit_path"] = str(args.audit.resolve())
        result = prepare_campaign(ROOT, workspace, config)
    elif args.stage == "individuals":
        result = run_individuals(workspace)
    elif args.stage == "select":
        result = freeze_components(workspace)
    elif args.stage == "meta":
        result = run_meta(workspace)
    elif args.stage == "ensemble":
        result = fit_ensemble(workspace)
    elif args.stage == "calibrate":
        result = calibrate_ensemble(workspace)
    elif args.stage == "final":
        if not args.dataset:
            parser.error("Final requires an explicitly packed fresh final --dataset")
        result = evaluate_final(workspace, args.dataset.resolve())
    else:
        result = {"workspace": str(workspace), "prepared": (workspace / "prepared.json").exists(),
                  "individuals": (workspace / "individuals_complete.json").exists(),
                  "components": (workspace / "components_frozen.json").exists(),
                  "ensemble": (workspace / "ensemble_frozen.json").exists(),
                  "calibration": (workspace / "calibration_frozen.json").exists(),
                  "final": (workspace / "final_complete.json").exists()}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
