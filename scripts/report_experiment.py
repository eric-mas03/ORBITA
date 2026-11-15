"""Render the completed technical experiment without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/training"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(out / "metrics.csv")
    pred = pd.read_csv(out / "predictions.csv")
    observations = pd.read_csv(ROOT / "data/processed/inspiration4_observations.csv")
    test = pred[pred.split.eq("test")]
    assert (test.q10 <= test.q50).all() and (test.q50 <= test.q90).all()
    assert test[["q10", "q50", "q90"]].ge(0).all().all()
    chosen = ["CMP::GLUCOSE", "CMP::CREATININE", "CBC::HEMOGLOBIN", "EVE_CARDIO::CRP"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.get_cmap("tab10").colors
    for ax, variable in zip(axes.flat, chosen):
        raw = observations[observations.variable.eq(variable) & observations.time_reference.eq("return")]
        p = test[test.variable.eq(variable) & test.method.eq("grouped_transformer")]
        for i, (subject, curve) in enumerate(raw.groupby("subject_id")):
            curve = curve.sort_values("relative_day")
            historical = curve[curve.relative_day.le(82)]
            color = colors[i]
            ax.plot(historical.relative_day, historical.value, "o-", color=color, label=subject)
            actual = curve[curve.relative_day.eq(194)]
            ax.scatter(actual.relative_day, actual.value, marker="x", s=45, color=color)
            point = p[p.subject_id.eq(subject)].iloc[0]
            x = 194 + (i - 1.5) * 2.5
            ax.errorbar(x, point.q50, yerr=[[point.q50 - point.q10], [point.q90 - point.q50]],
                        fmt="s", capsize=4, color=color, markersize=4)
        ax.axvline(82, color="gray", linestyle="--", linewidth=1)
        ax.set_title(variable)
        ax.set_ylabel(raw.unit.iloc[0])
        ax.set_xlabel("Días relativos al retorno · predicción desde R+82")
        ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Experimento técnico · horizonte 112 días, solo cuatro personas\nLínea: historial · ×: observado futuro · cuadrado/barras: q50 y q10–q90", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "forecast_examples.png", dpi=150)
    plt.close(fig)
    history = pd.read_csv(out / "loss_history.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.epoch, history.train_loss, label="Entrenamiento (dropout activo)")
    ax.plot(history.epoch, history.val_loss, label="Validación")
    ax.axvline(run["selected_epoch"], color="gray", linestyle="--", label="Checkpoint seleccionado")
    ax.set(xlabel="Época", ylabel="Pinball normalizada, equilibrada por grupo", title="Entrenamiento exploratorio")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=150)
    plt.close(fig)
    rows = metrics[metrics.split.eq("test")].set_index("method")
    model, base = rows.loc["grouped_transformer"], rows.loc["persistence"]
    mae_gain = 100 * (1 - model.normalized_mae / base.normalized_mae)
    pinball_gain = 100 * (1 - model.normalized_pinball / base.normalized_pinball)
    lines = ["# Primer entrenamiento técnico con datos reales", "",
             "Resultado exploratorio sobre Inspiration4. No valida uso clínico, transferencia a nuevas personas ni una cobertura garantizada.", "",
             "## Datos y arquitectura", "",
             f"109 variables en 13 grupos definidos por el agente biomédico. Encoders pequeños por grupo → fusión latente común → transformer temporal causal → 13 decoders transformer con consultas de horizonte → q10/q50/q90 ordenados. Dimensión latente 16, dos cabezas de atención y una capa; {run['parameter_count']:,} parámetros.", "",
             "Solo se usan grupos habilitados de CMP, CBC y Eve. Alamar se analiza en el EDA, pero sus 203 variables permanecen fuera del primer entrenamiento. Los valores se estandarizan por variable con 20 visitas-persona de entrenamiento; no se interpolan ni se rellenan con observaciones futuras. La pérdida asigna el mismo peso a cada grupo.", "",
             "## Partición y disponibilidad", "",
             "- Entrenamiento: 16 pares de historial/objetivo, con objetivos L-44, L-3 y R+45; horizontes 41, 44, 48 y 89 días.",
             "- Validación: cuatro sujetos, R+45 → R+82, horizonte 37 días. Selección de checkpoint exclusivamente con esta visita.",
             "- Prueba exploratoria: cuatro sujetos, R+82 → R+194, horizonte 112 días. Solo se emite después de disponer de R+82 y seleccionar el modelo.",
             "- Los segmentos de lanzamiento y retorno se modelan separados, sin fabricar el tiempo entre ambos. Se supone disponibilidad inmediata del resultado en cada visita; los archivos no prueban esa latencia operacional.", "",
             "Los pares históricos de entrenamiento son supervisión retrospectiva, no predicciones emitidas por un modelo ya entrenado en esas fechas. Los orígenes de validación/prueba respetan el momento de ajuste y selección del checkpoint. La normalización permanece congelada en entrenamiento.", "",
             f"Se ejecutaron {run['epochs_run']} épocas con semilla {run['seed']}, parada tras 15 épocas sin mejora y checkpoint de época {run['selected_epoch']}. PyTorch {run['versions']['torch']}, CPU. No se ajustaron hiperparámetros a la prueba.", "",
             "## Resultado de la prueba", "",
             "Promedios sobre variables en unidades estandarizadas con entrenamiento. No se promedian errores originales que mezclen mg/dL, células/uL y pg/ml. Las métricas originales por variable y horizonte están en CSV.", "",
             "| Método | MAE normalizado ↓ | Pinball normalizada ↓ | Cobertura q10–q90 | Anchura normalizada |", "|---|---:|---:|---:|---:|"]
    for name, label in [("persistence", "Persistencia"), ("grouped_transformer", "Transformer agrupado")]:
        row = rows.loc[name]
        lines.append(f"| {label} | {row.normalized_mae:.4f} | {row.normalized_pinball:.4f} | {100*row.coverage:.1f} % | {row.normalized_width:.4f} |")
    lines += ["", f"En esta ejecución el transformer reduce MAE un {mae_gain:.1f} % y pinball un {pinball_gain:.1f} % frente a persistencia. Es una comparación preliminar en los mismos cuatro sujetos; no establece una mejora generalizable.", "",
              f"La banda nominal del 80 % cubre solo {100*model.coverage:.1f} % de los valores evaluados: **todavía no está bien calibrada**. Sus {int(model.scored_pairs)} parejas sujeto-variable son dependientes y no equivalen a ese número de personas independientes. No se calcula un intervalo de confianza de eficacia con esta cohorte.", "",
              "La persistencia utiliza el último valor observado y bandas de residuos de entrenamiento agrupados, sin ajuste por horizonte. Ambos métodos aplican el mismo límite físico inferior cero a los cuantiles de estas 109 variables no negativas, antes de puntuar. Esto conserva el orden, pero no demuestra calibración. Los datos observados no se recortan. El decoder genérico produce salidas estandarizadas; la aplicación del soporte se realiza al invertir la escala en este pipeline.", "",
              "## Límites y siguiente decisión", "",
              "1. El horizonte de prueba de 112 días supera el máximo de 89 visto en entrenamiento. La arquitectura admite varias consultas simultáneas e independientes, verificadas por tests, pero esta prueba real solo evalúa un horizonte.",
              "2. El EDA descriptivo utilizó la cohorte completa antes del experimento. Esta prueba es exploratoria, no una confirmación con conjunto intacto. No se seleccionan grupos ni analitos por su correlación con la prueba.",
              "3. Solo hay cuatro personas y siete visitas por persona, sin estas medidas durante vuelo. Casi 79 mil parámetros frente a tan pocas trayectorias hace imprescindible buscar cohortes adicionales antes de interpretar la ventaja del modelo.",
              "4. Falta validación por personas/misiones nuevas y calibración con soporte suficiente. No se afina cobertura mirando la prueba actual.",
              "5. Antes de crecer el modelo: estudiar estabilidad por persona y lote, comparar un baseline cuantílico regularizado y hacer ablaciones de redundancias de hemograma. Los resultados de esas decisiones necesitarán evaluación externa.", "",
              "## Reproducir", "", "```powershell", "python scripts/download_osdr.py --studies 575 569", "python scripts/analyze_osd575.py", "python -m pytest -q", "python scripts/train_exploratory.py --epochs 80", "python scripts/report_experiment.py", "```", "",
              "Checkpoint: `outputs/checkpoints/exploratory_twin.pt`. Configuración, hashes, versiones, supuestos y cortes: `outputs/training/run.json`. Predicciones completas: `outputs/training/predictions.csv`. El checkpoint solo debe cargarse desde fuentes confiables."]
    (ROOT / "docs/primer-entrenamiento.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reported {len(test)} method/variable rows; four test subjects; MAE change {mae_gain:.1f}%")


if __name__ == "__main__":
    main()
