"""Summarize a completed frozen confirmation; no model choice or refitting."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/model_campaign"
    selection = json.loads((out / "selection_frozen.json").read_text(encoding="utf-8"))
    run = json.loads((out / "confirmation_run_manifest.json").read_text(encoding="utf-8"))
    if run["status"] != "frozen_confirmation_complete":
        raise ValueError("Only summarize a complete frozen evaluation")
    dev = pd.read_csv(out / "dev_leaderboard.csv", index_col="method")
    confirmation = pd.read_csv(out / "confirmation_leaderboard.csv", index_col="method")
    bootstrap = pd.read_csv(out / "confirmation_paired_bootstrap.csv").set_index("method")
    horizons = pd.read_csv(out / "confirmation_by_horizon.csv")
    cells = pd.read_csv(out / "confirmation_by_variable_horizon.csv")
    reference = confirmation.loc["linear_quantile"]
    lines = ["# Confirmación de modelos individuales", "",
        "La selección se congeló en desarrollo antes de abrir 300 episodios nuevos (150 de cada fuente). Se reutilizaron los pesos, estadísticas de entrenamiento, calidad y tareas: no hubo ajuste con la confirmación. Los datos son UCI terrestre; no validan todavía un gemelo de astronautas.", "",
        "## Resultados globales", "",
        "Pinball y MAE normalizados por la escala de entrenamiento, calculados dentro de cada variable/horizonte y promediados con igual peso. Menor es mejor. La cobertura es la fracción de etiquetas dentro de q10–q90; su referencia nominal es 80 %, sin recalibración en esta ronda.", "",
        "| Método congelado | Pinball desarrollo | Pinball confirmación | Cobertura confirmación | Bandas fuera de soporte |",
        "|---|---:|---:|---:|---:|"]
    for method in selection["selected_for_confirmation"]:
        row = confirmation.loc[method]
        lines.append(f"| {method} | {dev.loc[method, 'macro_pinball']:.6f} | {row.macro_pinball:.6f} | {100*row.macro_coverage:.1f} % | {100*row.support_violation_fraction:.2f} % |")
    result = dict(selected_in_development=selection["selected_new_methods"],
        confirmation_used_for_selection=False, ensemble_trained=False, candidates={})
    for method in selection["selected_new_methods"]:
        row, ci = confirmation.loc[method], bootstrap.loc[method]
        relative = 1 - row.macro_pinball / reference.macro_pinball
        by_method = cells[cells.method.eq(method)].set_index(["variable", "horizon_hours"])
        by_reference = cells[cells.method.eq("linear_quantile")].set_index(["variable", "horizon_hours"])
        delta = (by_method.normalized_pinball - by_reference.normalized_pinball).dropna()
        wins, losses = int(delta.lt(-1e-8).sum()), int(delta.gt(1e-8).sum())
        result["candidates"][method] = dict(relative_confirmation_improvement=float(relative),
            variable_horizon_cells_improved=wins, variable_horizon_cells_worsened=losses,
            supported_cells_bootstrap=int(ci.supported_cells), bootstrap_delta_ci95=[float(ci.delta_ci95_lower), float(ci.delta_ci95_upper)])
        lines += ["", f"## Candidato preseleccionado: {method}", "",
            f"Cambio relativo de pinball en la confirmación completa: {100*relative:+.2f} % (positivo = mejora). Mejora en {wins} combinaciones variable/horizonte y empeora en {losses}; se conservan los empates y las celdas no evaluables en las tablas.", "",
            f"El bootstrap emparejado por episodio y estratificado por fuente usa {int(ci.supported_cells)} combinaciones con al menos diez episodios, dejando fuera {int(ci.omitted_sparse_cells)} combinaciones escasas solo de este cálculo de incertidumbre. Para ese subconjunto, mejora relativa {100*ci.relative_improvement:+.2f} %, intervalo descriptivo del 95 % [{100*ci.relative_improvement_ci95_lower:+.2f}, {100*ci.relative_improvement_ci95_upper:+.2f}] %. No es el mismo agregado que la tabla completa. Se ejecutan {int(ci.valid_replicates)} remuestreos válidos de {int(ci.requested_replicates)}. Los archivos identifican episodios, sin garantía de personas únicas.", "",
            "| Horizonte | Pinball lineal | Pinball candidato | Cobertura candidato |",
            "|---|---:|---:|---:|"]
        for horizon in (1, 4, 8, 24):
            a = horizons[horizons.method.eq("linear_quantile") & horizons.horizon_hours.eq(horizon)].iloc[0]
            b = horizons[horizons.method.eq(method) & horizons.horizon_hours.eq(horizon)].iloc[0]
            lines.append(f"| {horizon} h | {a.pinball:.6f} | {b.pinball:.6f} | {100*b.coverage:.1f} % |")
        lines += ["", ("La diferencia favorece al candidato en el subconjunto con más soporte y su intervalo no contiene cero." if ci.delta_ci95_upper < 0 else
                       "La comparación no establece una ventaja clara del candidato en el subconjunto con más soporte: el intervalo de diferencia incluye cero o favorece a la referencia."), ""]
    lines += ["## Alcance y siguiente fase", "",
        "La campaña compara once variantes finales, incluidas cuatro referencias previas. En esta ronda se entrenaron seis configuraciones de árboles (se eligió una por familia), tres arquitecturas Transformer residuales y un control MLP sin atención. Los Transformers reciben todos los latentes de hasta 24 rejillas horarias; los resúmenes son rutas auxiliares. No se han fabricado etiquetas ni rejillas de 15 minutos. La comparación de atención y capacidad procede de una semilla por arquitectura; el bootstrap de episodios no mide variación entre entrenamientos.", "",
        "No se ha entrenado un ensemble. Antes de combinar conviene medir estabilidad y complementariedad de errores mediante predicciones fuera de pliegue por episodio. Los futuros pesos no se ajustarán con esta confirmación ni con el conjunto de calibración. El conjunto de confirmación de esta ronda queda consumido para estos modelos individuales; una evaluación final del futuro ensemble requiere otra reserva independiente.", "",
        "Las variables escasas siguen visibles; el informe de desarrollo identifica ocho combinaciones con menos de diez episodios y dos combinaciones con menos de veinte etiquetas de entrenamiento (mínimo cinco). Las bandas se muestran sin recorte físico ni recalibración, junto a sus infracciones y las unidades pendientes de revisión.", "",
        "## Gráficas y reproducción", "",
        "- Desarrollo: `outputs/model_campaign/dev_model_comparison.png`, `dev_head_comparison.png`, `dev_learning_curves.png`.",
        "- Confirmación: `outputs/model_campaign/confirmation_horizons.png`, `confirmation_head_comparison.png`, `confirmation_forecast_example.png`.",
        "- Tablas completas por variable, cabeza, horizonte, fuente, historia propia y contexto: `outputs/model_campaign/`.",
        "- Pesos, presupuestos, versiones y hashes: manifiestos de `outputs/hourly_trees/`, `outputs/hourly_architecture_trials/`, `outputs/hourly_residual_control/` y `outputs/model_campaign/confirmation_run_manifest.json`.", "",
        "El evaluador de confirmación bloquea sobrescrituras para preservar esta ejecución. Los informes y figuras se pueden regenerar desde sus CSV sin volver a abrir ni ajustar modelos."]
    (ROOT / "docs/confirmacion-modelos-individuales.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "campaign_outcome.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
