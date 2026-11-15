"""Build a local retrospective DEV demonstration, without calling an LLM.

Uses only already exported predictions and aggregate development evaluation.
Future observed values never enter the evidence packet. This is not a clinical
report, new inference or a complete prospective prediction API.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from astronaut_twin.clinical_interpretation import Citation, build_evidence_packet, render_spanish_summary


def main():
    workspace = ROOT / "outputs/model_campaign_v2/workspace"
    directory = workspace / "outputs/hourly_baselines"
    if not (directory / "complete.json").is_file():
        raise ValueError("Wait until the frozen baseline export completes")
    source = directory / "predictions.csv"
    spec = importlib.util.spec_from_file_location("campaign_report", ROOT / "scripts/report_model_campaign.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    frame = report.read_split([source], "dev")
    frame = frame[frame.method.eq("linear_quantile")].copy()
    metrics = report.utils.summarize(report.annotate(frame.copy()), ["split", "method", "variable", "horizon_hours"])
    metrics["unit"] = metrics.variable.map(frame.groupby("variable").unit.first())
    examples = frame[frame.origin_hour.eq(24) & frame.horizon_hours.eq(4) & frame.variable.eq("HR")]
    if examples.empty:
        raise ValueError("No demonstration at the predefined origin/horizon")
    episode = sorted(examples.episode_id.unique())[0]
    selected = frame[frame.episode_id.eq(episode) & frame.origin_hour.eq(24) & frame.horizon_hours.eq(4) & frame.variable.isin(["HR", "Resp", "Temp"])]
    # Do not pass the retrospective future-label column, even though the
    # evidence builder independently ignores it.
    rows = selected.drop(columns=["observed"]).to_dict("records")
    out = ROOT / "outputs/clinical_interpretation_demo_v2"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "aggregate_development_metrics.csv"
    metrics.to_csv(metrics_path,index=False)
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    sources = [Citation("predictions", "Frozen V2 linear model · scored DEV forecasts", str(source.resolve()), digest(source)),
        Citation("development_evaluation", "Aggregate DEV evaluation; development is reused for selection", str(metrics_path.resolve()),digest(metrics_path))]
    context = dict(subject_key=episode, origin_hour=24, time_reference="retrospective ICULOS; research demonstration",
        method="linear_quantile", domain="terrestrial_ICU", trained_domain="terrestrial_ICU",
        evaluation_split="dev", evaluation_domain="terrestrial_ICU",
        prediction_source_id="predictions", evaluation_source_id="development_evaluation",
        physical_bounds=json.loads((ROOT / "config/hourly_quality_policy.json").read_text())["variables"],
        sensitivity="public_research_episode_not_anonymization_claim")
    packet = build_evidence_packet(rows, context=context, evaluation_rows=metrics.to_dict("records"), sources=sources)
    (out / "evidence.json").write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding="utf-8")
    preface = "# Demostración retrospectiva local\n\nEjemplo elegido por ID a las 24 h, horizonte 4 h. Solo muestra variables presentes en el CSV de evaluación; no es una API completa de pronóstico. No contiene las mediciones futuras ni es un informe clínico o una interpretación generada por un LLM.\n\n"
    (out / "summary.md").write_text(preface+render_spanish_summary(packet)+"\n",encoding="utf-8")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),method="linear_quantile",split="dev",
        retrospective=True,episode_selection="lexicographically first exported HR case at origin24/horizon4, no error ranking",
        facts=len(packet["facts"]),llm_executed=False,remote_transmission=False,used_meta_calibration_or_final=False,
        future_observed_values_passed=False,source_sha256={s.source_id:s.sha256 for s in sources})
    (out / "demo_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
