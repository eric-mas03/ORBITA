"""Evaluate frozen hourly baselines by variable, head and actual horizon."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 4, 8, 24]
DISPLAY_NAMES = {"persistence": "Persistencia", "linear_quantile": "Lineal cuantílica",
                 "gru_shared_readout": "GRU · readout común", "transformer_grouped": "Transformer agrupado"}


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.groupby(keys, dropna=False).agg(
        n_labels=("observed", "size"), n_episodes=("episode_id", "nunique"),
        mae=("absolute_error", "mean"), pinball=("pinball", "mean"),
        pinball_q10=("pinball_q10", "mean"), pinball_q50=("pinball_q50", "mean"), pinball_q90=("pinball_q90", "mean"),
        coverage=("covered", "mean"), width=("width", "mean"),
        normalized_mae=("normalized_mae", "mean"), normalized_pinball=("normalized_pinball", "mean"),
        normalized_width=("normalized_width", "mean"),
        frequency_below_q10=("below_q10", "mean"), frequency_below_q50=("below_q50", "mean"), frequency_below_q90=("below_q90", "mean"),
        fallback_fraction=("used_train_median_fallback", "mean"),
        support_violation_fraction=("support_violation", "mean"),
    ).reset_index()


def complete_metrics(scored: pd.DataFrame, config: dict, methods, splits) -> pd.DataFrame:
    """Keep unscored cells visible: no observations means NA, never zero error."""
    variables = [v["source_key"] for v in config["variables"]]
    grid = pd.MultiIndex.from_product([splits, methods, variables, HORIZONS], names=["split", "method", "variable", "horizon_hours"]).to_frame(index=False)
    result = grid.merge(scored, how="left", on=["split", "method", "variable", "horizon_hours"], validate="one_to_one")
    for key in ("n_labels", "n_episodes"):
        result[key] = result[key].fillna(0).astype(int)
    result["support_status"] = np.select([result.n_labels.eq(0), result.n_episodes.lt(10)],
                                          ["not_evaluable", "limited_episodes"], default="scored_exploratory")
    by_variable = {v["source_key"]: v for v in config["variables"]}
    result["head"] = result.variable.map(lambda v: by_variable[v]["group"])
    result["unit"] = result.variable.map(lambda v: by_variable[v]["source_unit_as_documented"])
    result["unit_review_required"] = result.variable.map(lambda v: by_variable[v].get("unit_review_required", False))
    return result


def validate_shared_predictions(pred: pd.DataFrame) -> dict:
    """A comparison must score identical targets, not easier model subsets."""
    keys = ["split", "episode_id", "origin_hour", "horizon_hours", "variable"]
    methods = sorted(pred.method.unique())
    reference = pred[pred.method.eq(methods[0])].set_index(keys).sort_index()
    if not reference.index.is_unique:
        raise ValueError("Duplicate reference prediction identities")
    checked = {}
    for method in methods:
        frame = pred[pred.method.eq(method)].set_index(keys).sort_index()
        if not frame.index.is_unique or not frame.index.equals(reference.index):
            raise ValueError(f"{method} does not score the same target identities")
        for column in ("observed", "target_scale", "target_hour"):
            if not np.allclose(frame[column], reference[column], rtol=1e-6, atol=1e-5, equal_nan=False):
                raise ValueError(f"{method} uses different {column} values")
        if not frame["site"].equals(reference["site"]) or not frame["head"].equals(reference["head"]):
            raise ValueError(f"{method} has different site or head identities")
        checked[method] = len(frame)
    return checked


def draw_heatmap(table, method, variable_order, metric, title, destination, cmap="viridis", limits=None):
    pivot = table[table.method.eq(method)].pivot(index="variable", columns="horizon_hours", values=metric).reindex(index=variable_order, columns=HORIZONS)
    fig, ax = plt.subplots(figsize=(7, 12))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#dedede")
    kwargs = dict(vmin=limits[0], vmax=limits[1]) if limits else {}
    image = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), aspect="auto", cmap=color_map, **kwargs)
    ax.set_xticks(range(4), [f"{h} h" for h in HORIZONS])
    ax.set_yticks(range(len(variable_order)), variable_order, fontsize=8)
    ax.set_title(title + "\nPrueba por episodios · gris = sin etiquetas", fontsize=11)
    ax.set_xlabel("Horizonte desde el origen")
    fig.colorbar(image, ax=ax, shrink=.7)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-neural", action="store_true", help="Compare frozen neural runs with baseline artifacts without retraining")
    args = parser.parse_args()
    directory = ROOT / "outputs/hourly_baselines"
    out = ROOT / ("outputs/hourly_comparison" if args.include_neural else "outputs/hourly_evaluation")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config/hourly_task_candidates.json").read_text(encoding="utf-8"))
    sources = [directory / "predictions.csv"]
    if args.include_neural:
        sources.append(ROOT / "outputs/hourly_neural/predictions.csv")
    pred = pd.concat([pd.read_csv(path) for path in sources], ignore_index=True)
    required = {"method", "split", "episode_id", "site", "origin_hour", "target_hour", "horizon_hours", "head", "variable", "unit", "observed", "q10", "q50", "q90", "history_hours", "own_history_observed", "used_train_median_fallback", "target_scale"}
    if not required.issubset(pred.columns):
        raise ValueError(f"Prediction contract missing columns: {required-set(pred.columns)}")
    if not np.isfinite(pred[["observed", "q10", "q50", "q90", "target_scale"]]).all().all():
        raise ValueError("Scoring requires finite actual labels and predictions")
    if not pred.target_scale.gt(0).all() or not ((pred.q10 <= pred.q50) & (pred.q50 <= pred.q90)).all():
        raise ValueError("Invalid scaling or unordered quantiles")
    if not np.allclose(pred.target_hour - pred.origin_hour, pred.horizon_hours):
        raise ValueError("Predicted horizon does not match actual source times")
    shared_counts = validate_shared_predictions(pred)
    pred["absolute_error"] = (pred.observed - pred.q50).abs()
    for name, q in (("q10", .1), ("q50", .5), ("q90", .9)):
        error = pred.observed - pred[name]
        pred["pinball_" + name] = np.maximum(q * error, (q - 1) * error)
        # Empirical <= frequencies need care with ties/zero-valued variables.
        pred["below_" + name] = pred.observed.le(pred[name])
    pred["pinball"] = pred[["pinball_q10", "pinball_q50", "pinball_q90"]].mean(axis=1)
    pred["covered"] = pred.observed.ge(pred.q10) & pred.observed.le(pred.q90)
    pred["width"] = pred.q90 - pred.q10
    policy = json.loads((ROOT / "config/hourly_quality_policy.json").read_text(encoding="utf-8"))["variables"]
    pred["support_violation"] = False
    for variable, rule in policy.items():
        rows = pred.variable.eq(variable)
        lower, upper = rule.get("minimum_inclusive"), rule.get("maximum_inclusive")
        if lower is not None:
            pred.loc[rows, "support_violation"] |= pred.loc[rows, "q10"].lt(lower)
        if upper is not None:
            pred.loc[rows, "support_violation"] |= pred.loc[rows, "q90"].gt(upper)
    for name, source in (("normalized_mae", "absolute_error"), ("normalized_pinball", "pinball"), ("normalized_width", "width")):
        pred[name] = pred[source] / pred.target_scale
    keys = ["split", "method", "variable", "horizon_hours"]
    metrics = complete_metrics(summarize(pred, keys), config, sorted(pred.method.unique()), sorted(pred.split.unique()))
    metrics.to_csv(out / "metrics_by_variable_horizon.csv", index=False)
    # Head/global summaries weight scored variables equally, not plentiful vital
    # sign rows. Raw errors with different units are never averaged together.
    eligible = metrics[metrics.n_labels.gt(0)]
    for name, groups in (("head", ["split", "method", "head", "horizon_hours"]),
                         ("horizon", ["split", "method", "horizon_hours"])):
        result = eligible.groupby(groups).agg(
            variables_scored=("variable", "nunique"), mean_variable_normalized_mae=("normalized_mae", "mean"),
            mean_variable_normalized_pinball=("normalized_pinball", "mean"),
            mean_variable_coverage=("coverage", "mean"), mean_variable_normalized_width=("normalized_width", "mean"),
            sum_label_pairs=("n_labels", "sum"))
        result.to_csv(out / ("metrics_by_" + name + ".csv"))
    summarize(pred, keys + ["site"]).to_csv(out / "metrics_by_site.csv", index=False)
    pred["context_kind"] = np.where(pred.history_hours.ge(24), "full24", "partial")
    summarize(pred, keys + ["context_kind"]).to_csv(out / "metrics_by_context.csv", index=False)
    pred["own_history_kind"] = np.where(pred.own_history_observed.gt(0), "own_measurement_in_context", "no_own_measurement_in_context")
    summarize(pred, keys + ["own_history_kind"]).to_csv(out / "metrics_by_own_history.csv", index=False)
    summarize(pred, keys + ["episode_id"]).to_csv(out / "metrics_by_episode.csv", index=False)
    pred.groupby(["split", "method", "variable", "horizon_hours", "episode_id", "origin_hour"]).size().rename("rows").reset_index().query("rows != 1").to_csv(out / "duplicate_prediction_checks.csv", index=False)
    if pred.duplicated(["split", "method", "variable", "horizon_hours", "episode_id", "origin_hour"]).any():
        raise ValueError("Duplicated scored prediction keys")
    test = metrics[metrics.split.eq("test")]
    order = [v["source_key"] for v in config["variables"]]
    methods = sorted(test.method.unique())
    pinball_max = float(test.normalized_pinball.max())
    for method in methods:
        draw_heatmap(test, method, order, "coverage", f"{method} · cobertura q10–q90 (nominal 0,80)", out / f"{method}_coverage.png", "RdYlBu", (0, 1))
        draw_heatmap(test, method, order, "normalized_pinball", f"{method} · pinball / escala de entrenamiento", out / f"{method}_pinball.png", limits=(0, pinball_max))
    draw_heatmap(test, methods[0], order, "n_labels", "Etiquetas de prueba por variable y horizonte", out / "test_label_counts.png", "Blues")
    horizon = pd.read_csv(out / "metrics_by_horizon.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for method in methods:
        sub = horizon[horizon.split.eq("test") & horizon.method.eq(method)].sort_values("horizon_hours")
        for ax, column in zip(axes, ["mean_variable_normalized_mae", "mean_variable_normalized_pinball", "mean_variable_coverage"]):
            ax.plot(sub.horizon_hours, sub[column], "o-", label=DISPLAY_NAMES.get(method, method))
            ax.set_xticks(HORIZONS)
            ax.set_xlabel("Horizonte (h)")
            ax.grid(alpha=.2)
    for ax, title in zip(axes, ["MAE normalizado ↓", "Pinball normalizada ↓", "Cobertura q10–q90"]):
        ax.set_title(title)
    axes[2].axhline(.8, color="gray", linestyle="--", label="Nominal 80 %")
    axes[0].legend(fontsize=8)
    axes[2].set_ylim(0, 1)
    fig.suptitle("Prueba horaria por episodios · media con igual peso por variable evaluada")
    fig.tight_layout()
    fig.savefig(out / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    head_metrics = pd.read_csv(out / "metrics_by_head.csv")
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 7), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        subset = head_metrics[head_metrics.split.eq("test") & head_metrics.method.eq(method)]
        matrix = subset.pivot(index="head", columns="horizon_hours", values="mean_variable_normalized_pinball").reindex(columns=HORIZONS)
        maximum = head_metrics.loc[head_metrics.split.eq("test"), "mean_variable_normalized_pinball"].max()
        im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap="viridis", vmin=0, vmax=maximum)
        ax.set_xticks(range(4), [f"{h} h" for h in HORIZONS])
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
        ax.set_title(DISPLAY_NAMES.get(method, method))
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle("Pinball normalizada por cabeza y horizonte · igual peso por variable evaluada")
    fig.tight_layout()
    fig.savefig(out / "head_horizon_pinball.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, h in zip(axes.flat, HORIZONS):
        for method in methods:
            cells = test[test.method.eq(method) & test.horizon_hours.eq(h) & test.n_labels.gt(0)]
            frequency = [cells["frequency_below_" + q].mean() for q in ["q10", "q50", "q90"]]
            ax.plot([.1, .5, .9], frequency, "o-", label=DISPLAY_NAMES.get(method, method))
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set(title=f"Horizonte {h} h", xlabel="Cuantil nominal", ylabel="Frecuencia observada Y ≤ q")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Calibración descriptiva en prueba · media por variable\nEmpates y variables discretas afectan la igualdad nominal")
    fig.tight_layout()
    fig.savefig(out / "quantile_calibration.png", dpi=150)
    plt.close(fig)
    # A deterministic rolling-origin example: choose the first episode with at
    # least two actual labels for each of these targets, for a single method.
    method = "transformer_grouped" if args.include_neural else "linear_quantile" if "linear_quantile" in methods else methods[-1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, variable in zip(axes.flat, ["HR", "Glucose", "Creatinine", "Hgb"]):
        sub = pred[pred.split.eq("test") & pred.method.eq(method) & pred.variable.eq(variable) & pred.horizon_hours.eq(4)]
        ids = sorted(sub.groupby("episode_id").size().loc[lambda x: x >= 2].index)
        if not ids:
            ax.text(.5, .5, "Sin episodio con dos etiquetas", ha="center", transform=ax.transAxes)
            ax.set_title(variable)
            continue
        sub = sub[sub.episode_id.eq(ids[0])].sort_values("target_hour")
        ax.fill_between(sub.target_hour, sub.q10, sub.q90, alpha=.2, label="q10–q90")
        ax.plot(sub.target_hour, sub.q50, "o-", markersize=3, label="q50 emitido 4 h antes")
        ax.plot(sub.target_hour, sub.observed, "x", color="black", label="Observado")
        source_site, source_id = ids[0].split("/", 1)
        site_label = source_site.replace("training_set", "hospital ")
        ax.set_title(f"{variable} · {site_label}/{source_id} · prueba", fontsize=9)
        ax.set(xlabel="Hora objetivo desde ingreso", ylabel=sub.unit.iloc[0])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{DISPLAY_NAMES.get(method, method)} · cada punto se predijo 4 h antes\nConexiones visuales entre etiquetas; no son mediciones interpoladas")
    fig.tight_layout()
    fig.savefig(out / "rolling_examples_4h.png", dpi=150)
    plt.close(fig)
    if args.include_neural:
        learning = pd.read_csv(ROOT / "outputs/hourly_neural/learning_curves.csv")
        fit = json.loads((ROOT / "outputs/hourly_neural/fit_manifest.json").read_text(encoding="utf-8"))
        neural_methods = list(learning.method.unique())
        fig, axes = plt.subplots(1, len(neural_methods), figsize=(6 * len(neural_methods), 4), squeeze=False)
        for ax, method in zip(axes.flat, neural_methods):
            sub = learning[learning.method.eq(method)]
            ax.plot(sub.epoch, sub.online_train_macro_pinball, label="Entrenamiento online, dropout activo")
            ax.plot(sub.epoch, sub.dev_macro_pinball, label="Desarrollo")
            ax.axvline(fit["models"][method]["selected_epoch"], color="gray", linestyle="--", label="Checkpoint elegido en desarrollo")
            ax.set(title=DISPLAY_NAMES.get(method, method), xlabel="Época", ylabel="Pinball macro normalizada")
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("Curvas de ajuste y selección · la prueba no interviene")
        fig.tight_layout()
        fig.savefig(out / "neural_learning_curves.png", dpi=150)
        plt.close(fig)
    summary = {"status": "hourly_neural_comparison_episode_holdout_exploratory" if args.include_neural else "hourly_baselines_episode_holdout_exploratory", "methods": methods,
               "verified_identical_target_pairs_per_method": shared_counts,
               "horizons_hours": HORIZONS, "test_episodes_scored": int(pred[pred.split.eq("test")].episode_id.nunique()),
               "test_cells": len(test), "test_cells_without_labels": int(test.n_labels.eq(0).sum()),
               "test_cells_under_10_episodes": int((test.n_labels.gt(0) & test.n_episodes.lt(10)).sum()),
               "unit_review_variables": sorted(metrics.loc[metrics.unit_review_required, "variable"].unique().tolist()),
               "test_support_violation_fraction_by_method": pred[pred.split.eq("test")].groupby("method").support_violation.mean().to_dict(),
               "calibration_statement": "No calibration guarantees; the reserved calibration split is not used to fit these uncalibrated baselines.",
               "aggregation": "Head/horizon aggregates average normalized per-variable metrics; report raw units separately.",
               "cautions": ["ICU episodes are not astronaut validation or confirmed unique people across admissions.",
                            "Repeated hourly labels and overlapping origins are dependent.",
                            "Rare targets may have no independently useful test support.",
                            "Published hourly target semantics and retrospective availability assumption retained."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Comparación horaria de memorias GRU y Transformer" if args.include_neural else "# Primer bloque horario: ventanas, particiones y baselines", "",
             "Experimento sobre el subconjunto clínico terrestre descargado. No es una validación en astronautas. Los pronósticos son del valor publicado del intervalo terminal a 1, 4, 8 y 24 horas.", "",
             "## Ejecución y contrato", "",
             "Contexto causal de hasta 24 slots horarios, admitiendo historias cortas y valores ausentes. Los objetivos conservan máscaras independientes por variable y horizonte. No se interpolan, rellenan ni fabrican etiquetas de laboratorio. Los mínimos/máximos de contexto resumen mediciones horarias observadas en el pasado; no son mínimos/máximos intrahorarios reconstruidos.", "",
             "La división se fija por episodio y hospital antes del escalado. Se reservan entrenamiento, desarrollo, calibración y prueba; las personas subyacentes a ingresos distintos no se pueden deduplicar con estos archivos. El optimizador y la selección usan entrenamiento/desarrollo. La calibración permanece reservada y las bandas de este bloque son sin calibrar.", "",
             "Persistencia usa el último valor realmente observado dentro del contexto; cuando falta, usa mediana de entrenamiento y lo marca. Sus bandas proceden de residuos de entrenamiento. La referencia lineal cuantílica utiliza características causales de cada variable y produce tres cuantiles por horizonte, con normalización de entrenamiento. Ver configuración y decisiones exactas en los artefactos de ajuste.", "",
             "## Prueba por horizonte", "",
             "Promedios normalizados con igual peso por variable evaluada. La cobertura es media de coberturas marginales; no garantía conjunta. Los recuentos suman parejas etiqueta/predicción, no personas independientes.", "",
             "| Método | Horizonte | Variables evaluadas | MAE normalizado | Pinball normalizada | Cobertura |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in horizon[horizon.split.eq("test")].sort_values(["method", "horizon_hours"]).iterrows():
        lines.append(f"| {row.method} | {int(row.horizon_hours)} h | {int(row.variables_scored)} | {row.mean_variable_normalized_mae:.3f} | {row.mean_variable_normalized_pinball:.3f} | {100*row.mean_variable_coverage:.1f} % |")
    if args.include_neural:
        macro = horizon.groupby(["split", "method"])["mean_variable_normalized_pinball"].mean()
        section = ["## Criterio de desarrollo y comparación congelada", "",
                   "La métrica de selección es pinball normalizada macro por variable y horizonte. La siguiente tabla separa desarrollo de prueba descriptiva; no se eligen hiperparámetros con la columna de prueba.", "",
                   "| Modelo | Pinball macro en desarrollo ↓ | Pinball macro en prueba ↓ |", "|---|---:|---:|"]
        for method in methods:
            section.append(f"| {DISPLAY_NAMES.get(method, method)} | {macro.loc[('dev', method)]:.6f} | {macro.loc[('test', method)]:.6f} |")
        section += ["", "Bajo este presupuesto, la referencia lineal conserva el menor criterio de desarrollo. Entre las dos memorias, Transformer mejora ligeramente a GRU, sin demostrar una ventaja estable con una sola semilla. Las dos redes quedan por detrás de la referencia lineal en el agregado de prueba; los CSV permiten inspeccionar diferencias individuales sin redefinir a posteriori los objetivos.", "",
                    "La decisión provisional es mantener la referencia lineal y estudiar en desarrollo si preservar explícitamente el último valor y resúmenes propios mediante una conexión residual, o revisar el cuello de botella latente, mejora el candidato neuronal. Es una hipótesis futura: este experimento no identifica cuál componente explica la diferencia ni ejecuta esa ampliación.", ""]
        insert_at = lines.index("## Prueba por horizonte")
        lines[insert_at:insert_at] = section
    lines += ["", "## Resultados inspeccionables", "",
              f"- `{out.relative_to(ROOT).as_posix()}/metrics_by_variable_horizon.csv`: las 136 celdas por modelo/split, con etiquetas, episodios, unidades, calibración y soporte. Sin etiquetas: NA, no error cero.",
              "- Métricas por cabeza, hospital, contexto parcial/completo, presencia/ausencia de historia propia y episodio en el mismo directorio. Las escalas originales permanecen en los resultados por variable.",
              "- Mapas de pinball, cobertura y número de etiquetas; mapa por cabeza/horizonte, frecuencias de cuantiles, comparación por horizonte y ejemplos de orígenes móviles a 4 h.",
              "- `outputs/hourly_baselines/`: predicciones y artefactos de ajuste; `data/processed/hourly/` y `outputs/hourly_data/`: datos y particiones del constructor.", "",
              f"La prueba puntúa {summary['test_episodes_scored']} episodios. Hay {summary['test_cells_without_labels']} celdas modelo–variable–horizonte sin etiquetas y {summary['test_cells_under_10_episodes']} con menos de diez episodios: el umbral solo marca escasez para inspección, no certifica suficiencia de las demás.", "",
              "## Interpretación y siguiente experimento", "",
              "Las diferencias se revisan por variable/cabeza/horizonte y junto a cobertura/anchura. No se elige un modelo mirando solo una media ni se retiran a posteriori las variables difíciles de prueba. La selección utiliza exclusivamente desarrollo. La posterior calibración usará su conjunto reservado; ampliar arquitectura no garantiza calibración ni transferencia espacial.", "",
              "Las unidades con revisión pendiente permanecen identificadas; no se convierten por plausibilidad. FiO2 y otros valores físicamente imposibles se tratan según la política explícita, conservando originales y registro de calidad. Los resultados no se interpretan como recomendaciones clínicas.", "",
              "Los modelos conservan sus cuantiles ordenados sin proyección posterior a soporte físico. `support_violation_fraction` registra la fracción de bandas que sobrepasan un límite físico de la política, por variable/horizonte. Es una limitación explícita; no se recortan predicciones tras mirar prueba para mejorar sus métricas. Es diferente del control de calidad aplicado a las observaciones de entrada.", "",
              "## Reproducir", "", "```powershell", "python scripts/build_hourly_dataset.py", "python scripts/train_hourly_baselines.py", "python scripts/report_hourly_baselines.py", "python -m pytest -q", "```"]
    if args.include_neural:
        lines += ["", "## Qué se compara en las redes", "",
                  "GRU y Transformer comparten encoders por categoría, fusión, codificación temporal, consultas de horizonte y decoders de atención cruzada. Cambia la memoria temporal recurrente frente a autoatención causal. La variante GRU conserva el readout con atención: no es una GRU pura sin atención. Se registran parámetros y presupuesto en el manifiesto neuronal.", "",
                  "Es una comparación inicial con una semilla y un presupuesto acotado. No hay preentrenamiento, aumento ni ajuste con prueba. Las diferencias de capacidad y el ajuste bajo este presupuesto limitan la atribución de resultados a la atención. Una red peor en prueba no justifica volver a seleccionar hiperparámetros usando esa misma prueba.", "",
                  "Reproducir el informe combinado con `python scripts/report_hourly_baselines.py --include-neural`. Entrenamiento y configuración exacta en `docs/modelos-neuronales-horarios.md` y `outputs/hourly_neural/`. El informe histórico del primer bloque y sus figuras permanecen separados."]
    report_name = "docs/comparacion-neuronal-horaria.md" if args.include_neural else "docs/primer-bloque-horario.md"
    (ROOT / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(horizon[horizon.split.eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
