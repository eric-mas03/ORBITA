# Resultados de la campaña V2

Se reconciliaron los cinco recursos oficiales NASA/CSA y se incorporaron sus documentos. No contenían tablas horarias nuevas. La ampliación de observaciones procede, por separado, de PhysioNet (UCI terrestre). El corpus documental se usa para recuperación y contexto; no se cuenta como pacientes.

## Cohorte y repetición

Entrenamiento: 1.600 episodios y 56.669 orígenes; desarrollo: 150 episodios. Se reservaron otros 200 para aprender pesos del ensemble, 100 para calibración y 300 nuevos para la evaluación final. Se excluyeron la prueba V1 y su confirmación consumida. Mínimo de entrenamiento por variable/horizonte: 23 etiquetas y 18 episodios. Las ventanas solapadas no son observaciones independientes.

Se repitieron las once variantes con normalizadores nuevos ajustados solo en entrenamiento. Los límites de épocas neuronales aproximan el número anterior de actualizaciones (8/6/12 según familia), con parada anticipada y restauración del mejor estado. Se conservan los 24 latentes horarios, los horizontes 1/4/8/24 h y las máscaras. Las dos configuraciones de cada familia de árboles también se repitieron.

## Desarrollo

| Modelo | Pinball macro ↓ | Cobertura media |
|---|---:|---:|
| group_temporal_residual | 0.150450 | 77.9 % |
| causal_residual | 0.150557 | 76.4 % |
| window_bidirectional_residual | 0.151002 | 76.3 % |
| quantile_forest | 0.151049 | 82.5 % |
| own_summary_mlp_residual | 0.151896 | 78.7 % |
| linear_quantile | 0.153027 | 79.0 % |
| hist_boost_group_features | 0.153616 | 72.5 % |
| hist_boost_quantile | 0.153942 | 74.0 % |
| transformer_grouped | 0.173908 | 79.6 % |
| gru_shared_readout | 0.175625 | 81.2 % |
| persistence | 0.180153 | 80.1 % |

Los números anteriores usan la escala V2 y no se comparan directamente con una escala V1 distinta. La comparación sobre los mismos objetivos y escala común se informa en `docs/repeticion-v1-v2-escala-comun.md`.

## Ensemble y evaluación final

Componentes elegidos únicamente en desarrollo; pesos globales no negativos aprendidos en los 200 episodios META disjuntos de los ajustes de componentes. Esta ronda usa stacking con reserva separada, no predicciones OOF. Los mismos pesos se aplican a los tres cuantiles, todas las variables y horizontes. La combinación de cuantiles no se presenta como cuantiles de una mezcla de distribuciones.

Pesos congelados: linear_quantile=0.2024, group_temporal_residual=0.2930, quantile_forest=0.2389, own_summary_mlp_residual=0.2658.

La calibración posterior usa los 100 episodios separados. Los límites ajustados son extremos empíricos de una banda nominal del 80 %; no se certifican percentiles marginales ni cobertura formal bajo dependencia temporal. q50 se mantiene. La referencia lineal y el mejor individual de desarrollo reciben la misma regla de ajuste.

Prueba final única de 300 episodios nuevos, con modelos, pesos y calibración congelados antes de abrir sus valores. Mejor individual preseleccionado: `group_temporal_residual`.

| Modelo / banda | Pinball macro ↓ | MAE normalizado ↓ | Cobertura | Bandas fuera de soporte |
|---|---:|---:|---:|---:|
| group_temporal_residual_calibrated | 0.170956 | 0.5023 | 78.8 % | 3.73 % |
| ensemble_raw | 0.171117 | 0.5047 | 78.2 % | 3.37 % |
| group_temporal_residual_raw | 0.171187 | 0.5023 | 75.3 % | 3.59 % |
| linear_quantile_raw | 0.171663 | 0.5073 | 78.0 % | 3.03 % |
| ensemble_calibrated | 0.171742 | 0.5047 | 80.4 % | 3.48 % |
| linear_quantile_calibrated | 0.172427 | 0.5073 | 80.9 % | 3.50 % |

El ensemble raw mejora solo un 0,32 % frente a la lineal raw y un 0,04 % frente al mejor Transformer raw. Tras ajustar bandas, el Transformer obtiene menor pinball y MAE, mientras que el ensemble se acerca más a la cobertura nominal del 80 %. No hay un método que domine todas las métricas, ni se elige de nuevo a partir de estos resultados finales.

Por cada método, la calibración tiene 117 combinaciones directas, 15 que agrupan horizontes de la misma variable y cuatro sin soporte suficiente, que conservan su banda sin expansión. Las predicciones fuera del soporte físico se conservan para auditoría; no deben convertirse en afirmaciones clínicas mediante la capa lingüística.

## Incertidumbre por episodio

Remuestreo emparejado y estratificado por fuente: 1.000 réplicas, intervalo descriptivo del 95 %, solo combinaciones con al menos diez episodios. Las combinaciones escasas siguen presentes en la evaluación completa. Es un subconjunto distinto del ranking global; el remuestreo no mide variabilidad entre semillas ni asegura personas únicas por archivo.

| Candidato | Referencia | Celdas | Mejora relativa | Intervalo 95 % |
|---|---|---:|---:|---:|
| ensemble_raw | linear_quantile_raw | 134 | +0.76 % | [-0.73, +2.04] % |
| ensemble_raw | group_temporal_residual_raw | 134 | +0.32 % | [-0.54, +1.25] % |
| ensemble_calibrated | linear_quantile_calibrated | 134 | +0.85 % | [-0.64, +2.12] % |
| ensemble_calibrated | group_temporal_residual_calibrated | 134 | -0.16 % | [-0.98, +0.75] % |

Los cuatro intervalos de diferencia incluyen cero. Este análisis no establece una ventaja clara del ensemble frente a las referencias; no se afinan pesos o modelos después de verlo.

## Interpretación y artefactos

Se implementaron un paquete trazable de evidencia, un resumen determinista en español y un recuperador léxico TF-IDF con nueve documentos NASA/CSA y 48 fragmentos. La demostración en `outputs/clinical_interpretation_demo_v2/summary.md` usa predicciones de desarrollo, sin entregar mediciones futuras a la capa narrativa. No se entrenó ni ejecutó FLAN-T5 ni un LLM mediante API.

La siguiente comparación propuesta es plantilla frente a RAG con modelo pequeño local y RAG con herramientas de lectura mediante API. El ajuste supervisado requiere resúmenes revisados por especialistas; los CSV numéricos y artículos no los sustituyen. Codex puede servir para desarrollo y orquestación; la validez clínica del producto exige evaluar por separado la explicación y la transferencia a astronautas.

Fuentes y decisiones: `docs/auditoria-recursos-reto-2026.md`, `docs/ampliacion-horaria-v2.md`, `docs/ensemble-horario-v2.md`, `docs/capa-interpretacion-clinica.md`, `docs/agentes-producto-y-codex.md`. CSV, figuras, pesos y freezes están en `outputs/model_campaign_v2/`. Se conserva V1 intacta. No se ajustará el sistema con la prueba final ya consumida.

### Figuras

- [Once modelos en desarrollo](../outputs/model_campaign_v2/development/dev_model_comparison.png).
- [Comparación final por horizonte](../outputs/model_campaign_v2/workspace/final_evaluation/final_horizons_equal_variable_weight.png).
- [Comparación final por cabeza y horizonte](../outputs/model_campaign_v2/workspace/final_evaluation/final_head_comparison.png).
- [Ensemble: variables, cobertura y soporte](../outputs/model_campaign_v2/workspace/final_evaluation/ensemble_variable_horizon_evaluation.png).
