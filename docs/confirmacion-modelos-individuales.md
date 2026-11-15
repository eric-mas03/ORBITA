# Confirmación de modelos individuales

La selección se congeló en desarrollo antes de abrir 300 episodios nuevos (150 de cada fuente). Se reutilizaron los pesos, estadísticas de entrenamiento, calidad y tareas: no hubo ajuste con la confirmación. Los datos son UCI terrestre; no validan todavía un gemelo de astronautas.

## Resultados globales

Pinball y MAE normalizados por la escala de entrenamiento, calculados dentro de cada variable/horizonte y promediados con igual peso. Menor es mejor. La cobertura es la fracción de etiquetas dentro de q10–q90; su referencia nominal es 80 %, sin recalibración en esta ronda.

| Método congelado | Pinball desarrollo | Pinball confirmación | Cobertura confirmación | Bandas fuera de soporte |
|---|---:|---:|---:|---:|
| linear_quantile | 0.156521 | 0.163205 | 78.4 % | 3.25 % |
| group_temporal_residual | 0.155083 | 0.162198 | 75.1 % | 3.94 % |

## Candidato preseleccionado: group_temporal_residual

Cambio relativo de pinball en la confirmación completa: +0.62 % (positivo = mejora). Mejora en 79 combinaciones variable/horizonte y empeora en 57; se conservan los empates y las celdas no evaluables en las tablas.

El bootstrap emparejado por episodio y estratificado por fuente usa 130 combinaciones con al menos diez episodios, dejando fuera 6 combinaciones escasas solo de este cálculo de incertidumbre. Para ese subconjunto, mejora relativa -0.24 %, intervalo descriptivo del 95 % [-1.26, +0.77] %. No es el mismo agregado que la tabla completa. Se ejecutan 1000 remuestreos válidos de 1000. Los archivos identifican episodios, sin garantía de personas únicas.

| Horizonte | Pinball lineal | Pinball candidato | Cobertura candidato |
|---|---:|---:|---:|
| 1 h | 0.145899 | 0.146095 | 75.2 % |
| 4 h | 0.159961 | 0.160211 | 75.1 % |
| 8 h | 0.162250 | 0.161282 | 75.7 % |
| 24 h | 0.184709 | 0.181202 | 74.3 % |

La comparación no establece una ventaja clara del candidato en el subconjunto con más soporte: el intervalo de diferencia incluye cero o favorece a la referencia.

## Alcance y siguiente fase

La campaña compara once variantes finales, incluidas cuatro referencias previas. En esta ronda se entrenaron seis configuraciones de árboles (se eligió una por familia), tres arquitecturas Transformer residuales y un control MLP sin atención. Los Transformers reciben todos los latentes de hasta 24 rejillas horarias; los resúmenes son rutas auxiliares. No se han fabricado etiquetas ni rejillas de 15 minutos. La comparación de atención y capacidad procede de una semilla por arquitectura; el bootstrap de episodios no mide variación entre entrenamientos.

No se ha entrenado un ensemble. Antes de combinar conviene medir estabilidad y complementariedad de errores mediante predicciones fuera de pliegue por episodio. Los futuros pesos no se ajustarán con esta confirmación ni con el conjunto de calibración. El conjunto de confirmación de esta ronda queda consumido para estos modelos individuales; una evaluación final del futuro ensemble requiere otra reserva independiente.

Las variables escasas siguen visibles; el informe de desarrollo identifica ocho combinaciones con menos de diez episodios y dos combinaciones con menos de veinte etiquetas de entrenamiento (mínimo cinco). Las bandas se muestran sin recorte físico ni recalibración, junto a sus infracciones y las unidades pendientes de revisión.

## Gráficas y reproducción

- Desarrollo: `outputs/model_campaign/dev_model_comparison.png`, `dev_head_comparison.png`, `dev_learning_curves.png`.
- Confirmación: `outputs/model_campaign/confirmation_horizons.png`, `confirmation_head_comparison.png`, `confirmation_forecast_example.png`.
- Tablas completas por variable, cabeza, horizonte, fuente, historia propia y contexto: `outputs/model_campaign/`.
- Pesos, presupuestos, versiones y hashes: manifiestos de `outputs/hourly_trees/`, `outputs/hourly_architecture_trials/`, `outputs/hourly_residual_control/` y `outputs/model_campaign/confirmation_run_manifest.json`.

El evaluador de confirmación bloquea sobrescrituras para preservar esta ejecución. Los informes y figuras se pueden regenerar desde sus CSV sin volver a abrir ni ajustar modelos.
