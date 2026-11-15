"""Descriptive paired episode uncertainty after frozen final evaluation."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_bootstrap", ROOT / "scripts/bootstrap_campaign_confirmation.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def main():
    workspace = ROOT / "outputs/model_campaign_v2/workspace"
    final = json.loads((workspace / "final_complete.json").read_text())
    if final["status"] != "one_shot_frozen_final_complete":
        raise ValueError("The frozen final evaluation must finish first")
    source = workspace / "final_evaluation/predictions.csv"
    import hashlib
    if hashlib.sha256(source.read_bytes()).hexdigest() != final["predictions_sha256"]:
        raise ValueError("Frozen final prediction artifact changed")
    fields = {"split", "method", "episode_id", "site", "origin_hour", "target_hour", "horizon_hours",
              "head", "variable", "observed", "q10", "q50", "q90", "target_scale"}
    frame = pd.read_csv(source, usecols=lambda key:key in fields)
    results = []
    for variant in ("raw", "calibrated"):
        candidate = f"ensemble_{variant}"
        references = list(dict.fromkeys([f"linear_quantile_{variant}", f"{final['best_individual_dev']}_{variant}"]))
        for reference in references:
            paired = frame[frame.method.isin([candidate, reference])].copy()
            paired["method"] = paired.method.replace({reference:"linear_quantile"})
            row = bootstrap.paired_intervals(paired).iloc[0].to_dict()
            row["reference"] = reference
            row["candidate"] = candidate
            row["variant"] = variant
            results.append(row)
    result = pd.DataFrame(results)
    destination = workspace / "final_evaluation/paired_episode_bootstrap.csv"
    result.to_csv(destination,index=False)
    notes = dict(source_predictions_sha256=final["predictions_sha256"], descriptive=True,
        seed=20261114,replicates=1000,level=.95,minimum_episodes_per_cell=10,
        selection_or_tuning_after_final=False,
        caveats=["Sparse cells excluded from this uncertainty calculation remain in the full final report.",
                 "Episode files do not establish unique people across admissions.",
                 "Episode resampling does not measure training-seed variation.",
                 "Intervals are descriptive comparisons, not a guarantee of clinical validity or coverage."])
    destination.with_suffix(".json").write_text(json.dumps(notes,indent=2),encoding="utf-8")
    print(result[["candidate","reference","supported_cells","relative_improvement","relative_improvement_ci95_lower","relative_improvement_ci95_upper"]].to_string(index=False))


if __name__ == "__main__":
    main()
