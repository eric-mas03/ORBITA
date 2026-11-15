# Ensayos reales con árboles cuantílicos

Se ejecutaron las tres familias sobre el dataset horario congelado, con las particiones existentes y sin utilizar calibración o prueba. Esta fase seleccionó configuraciones exclusivamente en desarrollo. Los artefactos se guardan en `outputs/hourly_trees/`.

## Familias y presupuesto

| Familia | Entradas de cada objetivo | Configuración A | Configuración B |
|---|---|---|---|
| `hist_boost_quantile` | Nueve características causales de la propia variable | 100 iteraciones, 15 hojas | 200 iteraciones, 31 hojas |
| `quantile_forest` | Las mismas nueve características | 64 árboles, mínimo 5 muestras por hoja | 128 árboles, mínimo 20 muestras por hoja |
| `hist_boost_group_features` | Concatenación de las nueve características de cada variable de su grupo biomédico | 100 iteraciones, 15 hojas | 200 iteraciones, 31 hojas |

Boosting utiliza tres regresores por variable y horizonte, con pérdida cuantílica q10, q50 y q90; mínimo de 20 muestras por hoja, 63 bins y `early_stopping=False`. No crea una partición aleatoria interna ni mezcla episodios para decidir el número de iteraciones. Las configuraciones se comparan externamente en desarrollo. Véase la [documentación de HistGradientBoostingRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

El bosque utiliza `quantile-forest==1.4.2`, compatible con el scikit-learn 1.8 instalado. Se retienen todas las observaciones de cada hoja (`max_samples_leaf=None`), y la predicción usa `weighted_quantile=True`, `weighted_leaves=True` y `aggregate_leaves_first=True`. Son cuantiles de la distribución condicional ponderada de las respuestas en hojas; **no percentiles de las medias predichas por árboles**. Profundidad máxima 10, `max_features='sqrt'`, bootstrap del 70 % y dos trabajos. La [documentación del proyecto](https://zillow.github.io/quantile-forest/user_guide/fit_predict.html) explica la retención de respuestas en hojas y su agregación.

Las características propias son último valor, indicador de historia disponible, antigüedad, media, mínimo, máximo, número de mediciones, desviación y duración de contexto, codificadas como en los baselines. Los agregados proceden de las filas horarias observadas de la ventana causal; no son estadísticas recuperadas de señales de 15 minutos. La variante grupal utiliza entre nueve y 54 características según su grupo, sin estadísticas calculadas desde etiquetas futuras.

## Datos, pérdida y selección

Se reutiliza `prepare_baseline_data` para mantener las mismas ventanas de hasta 24 slots, máscaras y etiquetas exactas a 1/4/8/24 horas. Las estadísticas de escalado se ajustan con filas únicas de los episodios train. Cada modelo se ajusta con las etiquetas válidas de su variable y horizonte, sin exigir las demás salidas completas ni excluir variables difíciles.

Si solo existe una respuesta de entrenamiento o la respuesta es constante, se conserva mediante cuantiles empíricos. Si faltasen todas las etiquetas de una celda, el fallback se obtiene exclusivamente de los valores únicos de entrenamiento de esa variable y se registra. Los rangos clínicos no recortan respuestas ni predicciones. Tras predecir, los tres cuantiles se ordenan para evitar cruces.

La selección usa la media de pinball estandarizada: primero media sobre las observaciones y cuantiles de cada celda variable-horizonte y después media sobre las celdas con etiquetas de desarrollo. Se elige **una configuración global por familia**, no una configuración independiente para cada una de las 136 salidas. `candidate_cell_metrics.csv` conserva los soportes y pérdidas de cada celda y candidato.

Los grupos con una sola variable contienen las mismas entradas que la familia propia. Si sus campeones globales escogen hiperparámetros diferentes, una diferencia entre esos campeones no se atribuye exclusivamente a relaciones entre variables: los candidatos emparejados permiten inspeccionar ese efecto por separado.

## Artefactos y reutilización

- `fit_manifest.json`: versiones, hashes de datos/configuración/código, candidatos, selección y alcance exclusivo de desarrollo.
- `scalers.npz`: estadísticas de train y filas únicas utilizadas.
- `{familia}/candidate_{0,1}/h{h}_v{v}.joblib`: modelos y columnas de entrada por salida.
- `{familia}/candidate_{0,1}/dev_predictions.npz`: predicciones del candidato y sus índices de muestra.
- `predictions.csv`: predicciones de desarrollo de la configuración elegida por familia, con el contrato de los baselines.
- `candidate_cell_metrics.csv`: diagnóstico completo por variable, horizonte y candidato.

La carga para una inferencia futura congelada queda disponible sin reentrenar:

```python
from astronaut_twin.hourly_tree_models import load_tree_family, predict_tree_family

artifact = load_tree_family('outputs/hourly_trees', 'quantile_forest')
quantiles = predict_tree_family(artifact, baseline_data.features[selected_origins])
# quantiles: [orígenes, 4 horizontes, 34 variables, 3 cuantiles], unidades originales
```

Esta API no selecciona muestras ni cambia particiones. La decisión de evaluar un campeón en nuevos conjuntos queda fuera de estos ensayos de desarrollo.

## Reproducción

```powershell
python -m pip install --target .deps/tree_models --no-deps --only-binary=:all: quantile-forest==1.4.2
python -m pytest tests/test_hourly_trees.py -q -p no:cacheprovider
python scripts/train_hourly_trees.py
```

La dependencia se instala dentro del proyecto; no se modifica Python global. Se limita el cálculo a dos hilos mediante `threadpoolctl` y `n_jobs=2`. Una prueba específica con entradas constantes y respuestas dispersas comprueba que QRF conserva la amplitud de la distribución de resultados aunque las medias de sus árboles sean similares.

## Resultados

Se completaron las seis configuraciones y las 136 salidas de cada una. Menor pinball estandarizado macro es mejor:

| Familia | Configuración | Pinball macro de desarrollo | Selección de su familia |
|---|---|---:|---|
| HGB propio | 100 iteraciones / 15 hojas | 0,16305527 | Sí |
| HGB propio | 200 iteraciones / 31 hojas | 0,16738492 | No |
| QRF | 64 árboles / mínimo 5 por hoja | 0,16115427 | No |
| QRF | 128 árboles / mínimo 20 por hoja | **0,15883638** | Sí |
| HGB del grupo | 100 iteraciones / 15 hojas | 0,16312073 | Sí |
| HGB del grupo | 200 iteraciones / 31 hojas | 0,16726264 | No |

QRF obtuvo la menor pérdida entre estas tres familias. La variante grupal de boosting no mejoró su contraparte propia en el agregado de desarrollo; eso no establece ausencia de relaciones biomédicas. Los resultados son de selección en desarrollo, no un nuevo resultado de prueba ni una garantía de cobertura q10–q90.

`predictions.csv` contiene **353.532 filas: las mismas 117.844 claves observadas de desarrollo por cada uno de los tres métodos**. La auditoría mediante unión uno a uno con persistencia confirmó ausencia de claves omitidas o duplicadas, igualdad de valores observados, escalas y recuentos de historia, y cuantiles finitos ordenados. Los escaladores y sus índices de filas train coinciden exactamente con los baselines. Los estados de salidas sin etiquetas train se identifican explícitamente como `no_train_labels_train_only_fallback`.

Se guardaron **816 artefactos joblib**, uno por celda, candidato y familia; cada artefacto HGB contiene sus tres regresores cuantílicos. El script permite continuar mediante checkpoints sin sobrescribir modelos válidos: verifica datos y configuración congelados, estadísticas train, columnas, variable, horizonte e hiperparámetros antes de reutilizarlos. El progreso se escribe cada 17 celdas.

La ejecución final reanudada duró 495,61 segundos y reutilizó 225 celdas HGB propias ya ajustadas. Ese tiempo incluye verificación y reutilización: **no es un benchmark de entrenamiento íntegro comparable entre familias**. El manifiesto registra cuántas celdas se reutilizaron por candidato.

Las cinco pruebas de `tests/test_hourly_trees.py` finalizaron correctamente, incluida la verificación de las exportaciones reales de las tres familias.
