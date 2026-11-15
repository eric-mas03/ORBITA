"""Compare repeated DEV targets with the SAME V1 reporting scales.

V2 can learn its own train-only normalization. Cross-campaign improvements must
not be attributed to changing the denominator of standardized errors.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_report", ROOT / "scripts/report_model_campaign.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
DIRECTORIES = ("hourly_baselines", "hourly_neural", "hourly_trees", "hourly_architecture_trials", "hourly_residual_control")


def common_scale_scores(paths, scales):
    frame = report.read_split(paths, "dev")
    frame["target_scale"] = frame.variable.map(scales)
    report.utils.validate_shared_predictions(frame)
    scored = report.annotate(frame)
    identity = ["episode_id", "site", "origin_hour", "horizon_hours", "variable", "observed", "target_hour"]
    first = frame[frame.method.eq(sorted(frame.method.unique())[0])][identity].sort_values(identity[:5]).reset_index(drop=True)
    cells = report.utils.summarize(scored, ["method", "variable", "horizon_hours"])
    summary = cells.groupby("method").agg(macro_pinball=("normalized_pinball", "mean"),
        macro_mae=("normalized_mae", "mean"), macro_coverage=("coverage", "mean"),
        macro_width=("normalized_width", "mean"))
    return summary, cells, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs/model_campaign_v2/workspace")
    args = parser.parse_args()
    if not (args.workspace / "individuals_complete.json").exists():
        raise ValueError("Wait until every V2 family has completed")
    with np.load(ROOT / "outputs/hourly_baselines/scalers.npz", allow_pickle=False) as source:
        scales = dict(zip(map(str, source["variables"]), map(float, source["scale"])))
    old, old_cells, old_ids = common_scale_scores([ROOT / "outputs" / p / "predictions.csv" for p in DIRECTORIES], scales)
    new, new_cells, new_ids = common_scale_scores([args.workspace / "outputs" / p / "predictions.csv" for p in DIRECTORIES], scales)
    if not old_ids.iloc[:, :5].equals(new_ids.iloc[:, :5]):
        raise ValueError("Cross-campaign development target identities differ")
    for field in ("observed", "target_hour"):
        if not np.allclose(old_ids[field], new_ids[field], rtol=1e-6, atol=1e-5):
            raise ValueError(f"Cross-campaign {field} differs")
    result = old.add_prefix("v1_").join(new.add_prefix("v2_"), how="outer", validate="one_to_one")
    result["relative_pinball_improvement"] = 1 - result.v2_macro_pinball / result.v1_macro_pinball
    out = args.workspace.parent / "common_scale_comparison"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "v1_v2_dev_common_scale.csv")
    pd.concat([old_cells.assign(campaign="V1"), new_cells.assign(campaign="V2")]).to_csv(out / "by_variable_horizon.csv", index=False)
    metadata = dict(split="dev", shared_actual_targets=len(old_ids), reporting_scales="Frozen V1 training scales for BOTH campaigns",
        final_labels_used=False, training_normalizers_changed=True, optimization_budgets="Approximately matched maximum updates, different epochs",
        interpretation="Descriptive repeated-target comparison; multiple changes and reused development preclude causal attribution to dataset size alone.")
    (out / "comparison_contract.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Repetición V1 frente a V2 con escala común", "",
        "Se usan los mismos 150 episodios de desarrollo, los mismos objetivos observados y la escala de reporte V1 para ambas campañas. Cada campaña conserva sus propios normalizadores aprendidos solo en su entrenamiento. Así un cambio de desviación estándar no se presenta como una mejora del predictor.", "",
        "| Método | Pinball V1 | Pinball V2 | Mejora relativa | Cobertura V2 |", "|---|---:|---:|---:|---:|"]
    for name,row in result.sort_values("v2_macro_pinball").iterrows():
        lines.append(f"| {name} | {row.v1_macro_pinball:.6f} | {row.v2_macro_pinball:.6f} | {100*row.relative_pinball_improvement:+.2f} % | {100*row.v2_macro_coverage:.1f} % |")
    lines += ["", "Es una comparación descriptiva: aumentó la cohorte, cambiaron los normalizadores y se aproximó el presupuesto por número de actualizaciones en lugar de igualar épocas. Desarrollo se reutiliza para seleccionar. La prueba nueva se informa por separado después de congelar el ensemble y su calibración."]
    (ROOT / "docs/repeticion-v1-v2-escala-comun.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(result.to_string())


if __name__ == "__main__":
    main()
