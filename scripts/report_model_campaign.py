"""Select individual candidates on development only; confirmation stays separate."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hourly_report_utils", ROOT / "scripts/report_hourly_baselines.py")
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def read_split(paths, split):
    columns = ["split", "method", "episode_id", "site", "origin_hour", "target_hour", "horizon_hours",
               "head", "variable", "unit", "unit_review_required", "observed", "q10", "q50", "q90",
               "history_hours", "own_history_observed", "used_train_median_fallback", "target_scale",
               "train_target_count", "prediction_status"]
    frames = []
    for path in paths:
        for frame in pd.read_csv(path, usecols=lambda name: name in columns, chunksize=100_000):
            selected = frame[frame.split.eq(split)]
            if len(selected):
                frames.append(selected)
    if not frames:
        raise ValueError(f"No {split} predictions")
    return pd.concat(frames, ignore_index=True)


def annotate(frame):
    finite = np.isfinite(frame[["observed", "q10", "q50", "q90", "target_scale"]]).all().all()
    if not finite or not frame.target_scale.gt(0).all():
        raise ValueError("Invalid prediction/label/scaling")
    if not ((frame.q10 <= frame.q50) & (frame.q50 <= frame.q90)).all():
        raise ValueError("Crossed quantiles")
    if not np.allclose(frame.target_hour - frame.origin_hour, frame.horizon_hours):
        raise ValueError("Incorrect target horizon")
    frame["absolute_error"] = (frame.observed - frame.q50).abs()
    for name, q in [("q10", .1), ("q50", .5), ("q90", .9)]:
        error = frame.observed - frame[name]
        frame["pinball_" + name] = np.maximum(q * error, (q - 1) * error)
        frame["below_" + name] = frame.observed.le(frame[name])
    frame["pinball"] = frame[["pinball_q10", "pinball_q50", "pinball_q90"]].mean(axis=1)
    frame["covered"] = frame.observed.ge(frame.q10) & frame.observed.le(frame.q90)
    frame["width"] = frame.q90 - frame.q10
    for metric, source in [("normalized_mae", "absolute_error"), ("normalized_pinball", "pinball"), ("normalized_width", "width")]:
        frame[metric] = frame[source] / frame.target_scale
    frame["support_violation"] = False
    frame["median_support_violation"] = False
    policies = json.loads((ROOT / "config/hourly_quality_policy.json").read_text(encoding="utf-8"))["variables"]
    for variable, rule in policies.items():
        rows = frame.variable.eq(variable)
        lower, upper = rule.get("minimum_inclusive"), rule.get("maximum_inclusive")
        if lower is not None:
            frame.loc[rows, "support_violation"] |= frame.loc[rows, "q10"].lt(lower)
            frame.loc[rows, "median_support_violation"] |= frame.loc[rows, "q50"].lt(lower)
        if upper is not None:
            frame.loc[rows, "support_violation"] |= frame.loc[rows, "q90"].gt(upper)
            frame.loc[rows, "median_support_violation"] |= frame.loc[rows, "q50"].gt(upper)
    return frame


def summarize_campaign(paths, split, out):
    config = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    frame = read_split(paths, split)
    shared = utils.validate_shared_predictions(frame)
    frame = annotate(frame)
    cells = utils.complete_metrics(utils.summarize(frame, ["split", "method", "variable", "horizon_hours"]), config, sorted(frame.method.unique()), [split])
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / f"{split}_by_variable_horizon.csv", index=False)
    macro = cells[cells.n_labels.gt(0)].groupby("method").agg(
        macro_pinball=("normalized_pinball", "mean"), macro_mae=("normalized_mae", "mean"),
        macro_coverage=("coverage", "mean"), macro_width=("normalized_width", "mean"),
        evaluated_cells=("variable", "size"), min_episodes_per_cell=("n_episodes", "min"))
    pair_rates = frame.groupby("method").agg(support_violation_fraction=("support_violation", "mean"),
                                               median_support_violation_fraction=("median_support_violation", "mean"))
    macro = macro.join(pair_rates).sort_values("macro_pinball")
    macro.to_csv(out / f"{split}_leaderboard.csv")
    heads = cells[cells.n_labels.gt(0)].groupby(["method", "head", "horizon_hours"]).agg(
        macro_pinball=("normalized_pinball", "mean"), coverage=("coverage", "mean"), width=("normalized_width", "mean"),
        labels=("n_labels", "sum"), variables=("variable", "size"))
    heads.to_csv(out / f"{split}_by_head_horizon.csv")
    utils.summarize(frame.assign(own_history_kind=np.where(frame.own_history_observed.gt(0), "present", "absent")),
                    ["split", "method", "variable", "horizon_hours", "own_history_kind"]).to_csv(out / f"{split}_by_own_history.csv", index=False)
    utils.summarize(frame.assign(context_kind=np.where(frame.history_hours.ge(24), "full24", "partial")),
                    ["split", "method", "variable", "horizon_hours", "context_kind"]).to_csv(out / f"{split}_by_context.csv", index=False)
    utils.summarize(frame, ["split", "method", "variable", "horizon_hours", "site"]).to_csv(out / f"{split}_by_site.csv", index=False)
    if "train_target_count" in frame:
        support = frame.groupby(["method", "variable", "horizon_hours"]).agg(
            train_target_count=("train_target_count", "first"), evaluated_labels=("observed", "size"),
            evaluated_episodes=("episode_id", "nunique")).reset_index()
        support["train_support_status"] = np.select([support.train_target_count.eq(0), support.train_target_count.lt(20)],
            ["no_train_labels_fallback_or_untrained", "low_train_support"], default="fitted")
        support.to_csv(out / f"{split}_train_support.csv", index=False)
    macro["paired_labels"] = pd.Series(shared)
    return macro, cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, nargs="+", default=[ROOT / "outputs/hourly_baselines/predictions.csv",
        ROOT / "outputs/hourly_neural/predictions.csv", ROOT / "outputs/hourly_trees/predictions.csv",
        ROOT / "outputs/hourly_architecture_trials/predictions.csv", ROOT / "outputs/hourly_residual_control/predictions.csv"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model_campaign")
    args = parser.parse_args()
    if not all(path.exists() for path in args.paths):
        raise ValueError("Wait until all candidate exports have completed, or pass an explicit completed subset")
    leaderboard, cells = summarize_campaign(args.paths, "dev", args.output)
    protocol = json.loads((ROOT / "config/model_campaign.json").read_text(encoding="utf-8"))
    reference = leaderboard.loc["linear_quantile"]
    gate = protocol["working_competitive_target"]
    leaderboard["relative_pinball_improvement"] = 1 - leaderboard.macro_pinball / reference.macro_pinball
    leaderboard["working_competitive_target_met"] = (
        leaderboard.relative_pinball_improvement.ge(gate["relative_pinball_improvement_minimum"]) &
        leaderboard.macro_coverage.ge(gate["macro_coverage_minimum"]) &
        leaderboard.support_violation_fraction.le(reference.support_violation_fraction + 1e-12))
    leaderboard.to_csv(args.output / "dev_leaderboard.csv")
    new = [name for name in leaderboard.index if name not in {"linear_quantile", "persistence", "gru_shared_readout", "transformer_grouped"}]
    candidates = [name for name in new if bool(leaderboard.loc[name, "working_competitive_target_met"])]
    # Freeze a small shortlist using development only, before opening the new
    # confirmation data. Always retain the pre-existing linear reference.
    shortlist = candidates[:2] if candidates else new[:1]
    selection = {"criterion": protocol["selection_metric"], "selection_split": "dev", "reference": "linear_quantile",
                 "reference_dev_pinball": float(reference.macro_pinball), "competitive_target": gate,
                 "selected_new_methods": shortlist, "selected_for_confirmation": ["linear_quantile"] + shortlist,
                 "target_met_methods": candidates, "selected_best_method": str(leaderboard.index[0]),
                 "confirmation_used_for_selection": False, "calibration_used_for_selection": False,
                 "ensemble_trained": False,
                 "new_method_note": "If target not met, the best new method may be measured descriptively; it is not declared competitive."}
    evidence = args.paths + [p.parent / "fit_manifest.json" for p in args.paths if (p.parent / "fit_manifest.json").exists()]
    selection["evidence_sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence}
    (args.output / "selection_frozen.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .45 * len(leaderboard))))
    labels = leaderboard.index.tolist()
    y = np.arange(len(labels))
    axes[0].barh(y, leaderboard.macro_pinball, color=["#217b5a" if name in candidates else "#507da6" for name in labels])
    axes[0].axvline(reference.macro_pinball, color="black", linestyle="--", label="Referencia lineal")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Desarrollo · pinball macro normalizada ↓")
    axes[0].legend(fontsize=8)
    axes[1].plot(leaderboard.macro_coverage, y, "o")
    axes[1].axvline(.8, color="gray", linestyle="--", label="Nominal 80 %")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Cobertura marginal media")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "dev_model_comparison.png", dpi=150)
    plt.close(fig)
    champion = str(leaderboard.index[0])
    reference_cells = cells[cells.method.eq("linear_quantile")].set_index(["variable", "horizon_hours"])
    champion_cells = cells[cells.method.eq(champion)].set_index(["variable", "horizon_hours"])
    difference = (champion_cells.normalized_pinball - reference_cells.normalized_pinball).unstack("horizon_hours")
    fig, ax = plt.subplots(figsize=(7, 11))
    limit = max(float(np.nanmax(np.abs(difference.to_numpy()))), 1e-6)
    im = ax.imshow(difference, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(difference)), difference.index, fontsize=8)
    ax.set_xticks(range(4), ["1 h", "4 h", "8 h", "24 h"])
    ax.set_title(f"Desarrollo · {champion} menos lineal\nAzul = menor pinball; rojo = mayor")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / "dev_champion_vs_linear.png", dpi=150)
    plt.close(fig)
    lines = ["# Campaña de modelos individuales", "", "Selección realizada únicamente en desarrollo. El ensemble no se entrena en esta campaña.", "",
             "Se conservan episodios, etiquetas, horizontes 1/4/8/24 h, máscaras y escalas. Cada red mantiene todos los pasos temporales. Las fuentes actuales son horarias: no hay mediciones de 15 min fabricadas.", "",
             "## Comparación de desarrollo", "", "| Método | Pinball macro ↓ | MAE macro ↓ | Cobertura media | Bandas fuera de soporte | Mejora pinball frente a lineal |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in leaderboard.iterrows():
        lines.append(f"| {name} | {row.macro_pinball:.6f} | {row.macro_mae:.4f} | {100*row.macro_coverage:.1f} % | {100*row.support_violation_fraction:.2f} % | {100*row.relative_pinball_improvement:.1f} % |")
    lines += ["", "## Selección congelada", "", "Referencia: lineal cuantílica. Nuevos candidatos para confirmar: " + ", ".join(shortlist) + ".", "",
              ("Cumplen el objetivo operativo de esta ronda: " + ", ".join(candidates) + ".") if candidates else
              "Ningún candidato nuevo cumple todos los criterios operativos de esta ronda. La confirmación del mejor nuevo será descriptiva; no se declara una mejora sólida ni se activa el ensemble.", "",
              "El objetivo operativo de esta ronda es mejorar pinball al menos un 5 % en desarrollo, con cobertura macro ≥75 % y tasa de bandas físicamente inválidas no superior a la referencia. Es un criterio de ingeniería propuesto para priorizar candidatos; no una validación clínica ni una garantía del 80 %. Se conservan resultados por variable/cabeza, incluidos los escasos.", "",
              "## Qué comparan las arquitecturas", "",
              "Los Transformers residuales conservan todos los latentes horarios y aprenden una corrección de la predicción lineal congelada, con rutas auxiliares del estado y resúmenes propios. El punto de partida sin corrección participa en la selección. El control `own_summary_mlp_residual` usa solo los nueve resúmenes causales propios y el horizonte; no tiene atención. Mejorar frente al Transformer original no demuestra por sí solo que la atención añada valor frente a la referencia lineal o al control sin atención.", "",
              "La atención bidireccional se limita al historial conocido hasta el origen. El modelo especializado añade una memoria temporal por grupo que recibe los latentes comunes fusionados; no aísla los grupos de entrada. Los tamaños, semillas y presupuestos figuran en los manifiestos. Una sola semilla constituye un ensayo inicial, y los intervalos sobre episodios no sustituyen a repetir entrenamientos.", "",
              "La nueva confirmación se abre solo después de guardar `selection_frozen.json`. No se usa para afinar ni seleccionar otra arquitectura. Las métricas antiguas de prueba permanecen como resultados exploratorios separados. Los datos siguen siendo UCI terrestre, con episodios no necesariamente deduplicables como personas.", "",
              "## Artefactos", "", "`outputs/model_campaign/` contiene tablas por variable/horizonte y cabeza, ranking de desarrollo, selección congelada y gráficos. Los modelos y sus presupuestos están en `outputs/hourly_trees/` y `outputs/hourly_architecture_trials/`. Las bandas todavía no están recalibradas. Las unidades pendientes de revisión siguen identificadas.", "",
              "Reproducir: `python scripts/report_model_campaign.py`. La inferencia de confirmación, si se ejecuta después de congelar modelos, se documenta en su informe separado."]
    (ROOT / "docs/campana-modelos-individuales.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(leaderboard.to_string())
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
