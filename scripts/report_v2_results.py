"""Report frozen V2 results and plot horizons with equal weight per variable."""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_report", ROOT / "scripts/report_model_campaign.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)
DIRECTORIES = ("hourly_baselines", "hourly_neural", "hourly_trees", "hourly_architecture_trials", "hourly_residual_control")


def main():
    workspace = ROOT / "outputs/model_campaign_v2/workspace"
    if not (workspace / "individuals_complete.json").exists():
        raise ValueError("Wait until the eleven individual variants finish")
    report.ROOT = workspace  # Use the audited V2 task and quality snapshots.
    destination = workspace.parent / "development"
    paths = [workspace / "outputs" / name / "predictions.csv" for name in DIRECTORIES]
    leaderboard, cells = report.summarize_campaign(paths, "dev", destination)
    leaderboard.to_csv(destination / "dev_leaderboard.csv")
    support = pd.read_csv(destination / "dev_train_support.csv")
    expected = pd.read_csv(ROOT / "outputs/data_audit_v2/training_labels_by_variable_horizon.csv")
    support = support.merge(expected[["variable", "horizon_hours", "new_train_labels", "new_train_episodes"]],
        on=["variable", "horizon_hours"],validate="many_to_one")
    known = support.train_target_count.notna()
    if not np.equal(support.loc[known,"train_target_count"],support.loc[known,"new_train_labels"]).all():
        raise ValueError("Exported train support differs from audited training")
    support["train_target_count"] = support.new_train_labels
    support["train_target_episode_count"] = support.new_train_episodes
    support["train_support_status"] = np.select([support.new_train_labels.eq(0),support.new_train_labels.lt(20)],
        ["no_train_labels_fallback_or_untrained","low_train_support"],default="fitted")
    support.to_csv(destination / "dev_train_support.csv",index=False)
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    y = np.arange(len(leaderboard))
    axes[0].barh(y,leaderboard.macro_pinball,color="#386d8b")
    axes[0].axvline(leaderboard.loc["linear_quantile","macro_pinball"],color="black",linestyle="--")
    axes[0].set_yticks(y,leaderboard.index)
    axes[0].invert_yaxis()
    axes[0].set_title("V2 · desarrollo · pinball macro ↓")
    axes[1].scatter(leaderboard.macro_coverage,y)
    axes[1].axvline(.8,color="gray",linestyle="--",label="Nominal 80 %")
    axes[1].set_yticks(y,[])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0,1)
    axes[1].set_title("Cobertura media por variable/horizonte")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(destination / "dev_model_comparison.png",dpi=150)
    plt.close(fig)
    lines = ["# Resultados de la campaña V2", "",
        "Se reconciliaron los cinco recursos oficiales NASA/CSA y se incorporaron sus documentos. No contenían tablas horarias nuevas. La ampliación de observaciones procede, por separado, de PhysioNet (UCI terrestre). El corpus documental se usa para recuperación y contexto; no se cuenta como pacientes.", "",
        "## Cohorte y repetición", "",
        "Entrenamiento: 1.600 episodios y 56.669 orígenes; desarrollo: 150 episodios. Se reservaron otros 200 para aprender pesos del ensemble, 100 para calibración y 300 nuevos para la evaluación final. Se excluyeron la prueba V1 y su confirmación consumida. Mínimo de entrenamiento por variable/horizonte: 23 etiquetas y 18 episodios. Las ventanas solapadas no son observaciones independientes.", "",
        "Se repitieron las once variantes con normalizadores nuevos ajustados solo en entrenamiento. Los límites de épocas neuronales aproximan el número anterior de actualizaciones (8/6/12 según familia), con parada anticipada y restauración del mejor estado. Se conservan los 24 latentes horarios, los horizontes 1/4/8/24 h y las máscaras. Las dos configuraciones de cada familia de árboles también se repitieron.", "",
        "## Desarrollo", "",
        "| Modelo | Pinball macro ↓ | Cobertura media |",
        "|---|---:|---:|"]
    for name,row in leaderboard.iterrows():
        lines.append(f"| {name} | {row.macro_pinball:.6f} | {100*row.macro_coverage:.1f} % |")
    lines += ["", "Los números anteriores usan la escala V2 y no se comparan directamente con una escala V1 distinta. La comparación sobre los mismos objetivos y escala común se informa en `docs/repeticion-v1-v2-escala-comun.md`.", ""]
    if (workspace / "final_complete.json").exists():
        final = json.loads((workspace / "final_complete.json").read_text())
        ensemble = json.loads((workspace / "ensemble_frozen.json").read_text())
        final_dir = workspace / "final_evaluation"
        final_cells = pd.read_csv(final_dir / "metrics_by_variable_horizon.csv")
        overall = pd.DataFrame(final["scores"])
        lines += ["## Ensemble y evaluación final", "",
            "Componentes elegidos únicamente en desarrollo; pesos globales no negativos aprendidos en los 200 episodios META disjuntos de los ajustes de componentes. Esta ronda usa stacking con reserva separada, no predicciones OOF. Los mismos pesos se aplican a los tres cuantiles, todas las variables y horizontes. La combinación de cuantiles no se presenta como cuantiles de una mezcla de distribuciones.", "",
            "Pesos congelados: " + ", ".join(f"{name}={float(weight):.4f}" for name,weight in zip(ensemble["methods"],ensemble["weights"])) + ".", "",
            "La calibración posterior usa los 100 episodios separados. Los límites ajustados son extremos empíricos de una banda nominal del 80 %; no se certifican percentiles marginales ni cobertura formal bajo dependencia temporal. q50 se mantiene. La referencia lineal y el mejor individual de desarrollo reciben la misma regla de ajuste.", "",
            f"Prueba final única de {final['episodes']} episodios nuevos, con modelos, pesos y calibración congelados antes de abrir sus valores. Mejor individual preseleccionado: `{final['best_individual_dev']}`.", "",
            "| Modelo / banda | Pinball macro ↓ | MAE normalizado ↓ | Cobertura | Bandas fuera de soporte |", "|---|---:|---:|---:|---:|"]
        physical = final_cells.groupby("method").agg(invalid=("outside_physical_support","sum"),pairs=("observed_labels","sum"))
        for row in overall.sort_values("pinball_standardized").to_dict("records"):
            rate = physical.loc[row["method"],"invalid"] / physical.loc[row["method"],"pairs"]
            lines.append(f"| {row['method']} | {row['pinball_standardized']:.6f} | {row['mae_standardized']:.4f} | {100*row['coverage80_empirical']:.1f} % | {100*rate:.2f} % |")
        lines += ["", "El ensemble raw mejora solo un 0,32 % frente a la lineal raw y un 0,04 % frente al mejor Transformer raw. Tras ajustar bandas, el Transformer obtiene menor pinball y MAE, mientras que el ensemble se acerca más a la cobertura nominal del 80 %. No hay un método que domine todas las métricas, ni se elige de nuevo a partir de estos resultados finales.", "",
            "Por cada método, la calibración tiene 117 combinaciones directas, 15 que agrupan horizontes de la misma variable y cuatro sin soporte suficiente, que conservan su banda sin expansión. Las predicciones fuera del soporte físico se conservan para auditoría; no deben convertirse en afirmaciones clínicas mediante la capa lingüística."]
        horizon = final_cells.groupby(["method","horizon_hours"]).agg(
            pinball=("pinball_standardized","mean"),mae=("mae_standardized","mean"),
            coverage=("coverage80_empirical","mean"),width=("width_standardized","mean")).reset_index()
        horizon.to_csv(final_dir / "horizons_equal_variable_weight.csv",index=False)
        fig,axes = plt.subplots(1,3,figsize=(15,4.5))
        for method,sub in horizon.groupby("method"):
            for ax,key in zip(axes,["pinball","mae","coverage"]):
                ax.plot(sub.horizon_hours,sub[key],"o-",label=method,linewidth=1,markersize=3)
        for ax,title in zip(axes,["Pinball normalizada ↓","MAE normalizado ↓","Cobertura empírica"]):
            ax.set_title(title)
            ax.set_xticks([1,4,8,24])
            ax.set_xlabel("Horizonte (h)")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=6)
        axes[2].axhline(.8,color="gray",linestyle="--")
        axes[2].set_ylim(0,1)
        fig.suptitle("V2 · prueba final congelada · igual peso por variable")
        fig.tight_layout()
        fig.savefig(final_dir / "final_horizons_equal_variable_weight.png",dpi=150)
        plt.close(fig)
        ci_path = final_dir / "paired_episode_bootstrap.csv"
        if ci_path.exists():
            lines += ["", "## Incertidumbre por episodio", "",
                "Remuestreo emparejado y estratificado por fuente: 1.000 réplicas, intervalo descriptivo del 95 %, solo combinaciones con al menos diez episodios. Las combinaciones escasas siguen presentes en la evaluación completa. Es un subconjunto distinto del ranking global; el remuestreo no mide variabilidad entre semillas ni asegura personas únicas por archivo.", "",
                "| Candidato | Referencia | Celdas | Mejora relativa | Intervalo 95 % |", "|---|---|---:|---:|---:|"]
            for row in pd.read_csv(ci_path).to_dict("records"):
                lines.append(f"| {row['candidate']} | {row['reference']} | {row['supported_cells']} | {100*row['relative_improvement']:+.2f} % | [{100*row['relative_improvement_ci95_lower']:+.2f}, {100*row['relative_improvement_ci95_upper']:+.2f}] % |")
            lines += ["", "Los cuatro intervalos de diferencia incluyen cero. Este análisis no establece una ventaja clara del ensemble frente a las referencias; no se afinan pesos o modelos después de verlo."]
    lines += ["", "## Interpretación y artefactos", "",
        "Se implementaron un paquete trazable de evidencia, un resumen determinista en español y un recuperador léxico TF-IDF con nueve documentos NASA/CSA y 48 fragmentos. La demostración en `outputs/clinical_interpretation_demo_v2/summary.md` usa predicciones de desarrollo, sin entregar mediciones futuras a la capa narrativa. No se entrenó ni ejecutó FLAN-T5 ni un LLM mediante API.", "",
        "La siguiente comparación propuesta es plantilla frente a RAG con modelo pequeño local y RAG con herramientas de lectura mediante API. El ajuste supervisado requiere resúmenes revisados por especialistas; los CSV numéricos y artículos no los sustituyen. Codex puede servir para desarrollo y orquestación; la validez clínica del producto exige evaluar por separado la explicación y la transferencia a astronautas.", "",
        "Fuentes y decisiones: `docs/auditoria-recursos-reto-2026.md`, `docs/ampliacion-horaria-v2.md`, `docs/ensemble-horario-v2.md`, `docs/capa-interpretacion-clinica.md`, `docs/agentes-producto-y-codex.md`. CSV, figuras, pesos y freezes están en `outputs/model_campaign_v2/`. Se conserva V1 intacta. No se ajustará el sistema con la prueba final ya consumida."]
    lines += ["", "### Figuras", "",
        "- [Once modelos en desarrollo](../outputs/model_campaign_v2/development/dev_model_comparison.png).",
        "- [Comparación final por horizonte](../outputs/model_campaign_v2/workspace/final_evaluation/final_horizons_equal_variable_weight.png).",
        "- [Comparación final por cabeza y horizonte](../outputs/model_campaign_v2/workspace/final_evaluation/final_head_comparison.png).",
        "- [Ensemble: variables, cobertura y soporte](../outputs/model_campaign_v2/workspace/final_evaluation/ensemble_variable_horizon_evaluation.png)."]
    (ROOT / "docs/resultados-campana-v2.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(leaderboard.to_string())


if __name__ == "__main__":
    main()
