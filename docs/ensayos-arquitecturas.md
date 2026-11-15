# Ensayos de arquitectura sobre una referencia lineal congelada

Esta campaña entrena modelos residuales únicamente en entrenamiento y los selecciona únicamente en desarrollo. Conserva los 34 objetivos, 13 categorías, horizontes 1/4/8/24h y las particiones congeladas del dataset horario. No se generan predicciones de calibración, prueba o confirmación en este script. No se ejecutan aumento, preentrenamiento ni el ensemble final de candidatos competitivos.

La referencia lineal, sus bandas de fallback y sus escaladores se cargan de los artefactos ya entrenados de `outputs/hourly_baselines/`, con hashes registrados. No se vuelven a ajustar. Cada arquitectura aprende una corrección sobre esa referencia: es un modelo híbrido residual con offset congelado, no una nueva elección de pesos de un ensemble final entre candidatos.

## Tres hipótesis controladas

| Método | Memoria temporal | Hipótesis |
|---|---|---|
| `causal_residual` | Transformer causal común de dimensión 32 | Puede aprender correcciones útiles sin tener que reaprender desde cero la regresión lineal. |
| `window_bidirectional_residual` | Transformer común que atiende toda la ventana ya pasada | Contextualizar entre sí todos los instantes disponibles al origen puede ayudar a la lectura por horizonte. |
| `group_temporal_residual` | Un Transformer temporal causal por grupo, leyendo la misma secuencia fusionada | Especializar la dinámica por grupo puede aportar valor frente a una memoria temporal común. |

En todas se conservan los encoders de categoría, presencia y fusión; las consultas de horizonte leen la memoria completa. **No se colapsa el eje temporal**: valores `[B,24,F_g]` pasan a embeddings `[B,24,32]` y permanecen así en la memoria. Las salidas son `[B,4,34,3]`, con el orden original de variables. Son 24 slots horarios reales, no 96 observaciones ficticias de quince minutos.

La lectura de cada cabeza recibe, además del contexto temporal de su horizonte, nueve resúmenes causales de su propia variable —los mismos usados por la referencia lineal— y el estado del slot actual: valor estandarizado, máscara y edad. Los resúmenes son último valor, disponibilidad, edad, media/mínimo/máximo, recuento, desviación estándar y duración del contexto. Las estadísticas se refieren al contexto observado, no a una señal intrahoraria desconocida.

La primera y segunda arquitectura comparten las mismas inicializaciones y componentes; cambia la máscara interna de atención. La tercera sustituye la memoria común por memorias independientes de grupo, aumentando parámetros y coste. Esa diferencia se registra; no se afirma igualdad de capacidad entre las tres.

## Corrección ordenada e identidad inicial

La cabeza residual se inicializa con pesos e interceptos cero. Si los cuantiles congelados son `b10,b50,b90`, el modelo aprende:

```text
q50 = b50 + tanh(delta50)
q50 - q10 = (b50 - b10) * exp(tanh(s10))
q90 - q50 = (b90 - b50) * exp(tanh(s90))
```

Las operaciones están en unidades estandarizadas con la escala original de entrenamiento. La corrección de mediana queda entre -1 y +1 desviaciones estándar; cada gap se multiplica por un factor entre `exp(-1)` y `exp(1)`. Se emplea una formulación con `expm1` para preservar exactamente los valores congelados cuando la corrección es cero. Se comprueba orden numérico final. No se proyecta a soportes físicos ni se recortan rangos clínicos.

Si un gap congelado es exactamente cero, este mecanismo multiplicativo lo conserva en cero; se registra su frecuencia en entrenamiento/desarrollo. El modelo no puede abrir ese gap sin cambiar la parametrización en otro experimento. En el primer paso, las últimas capas sí reciben gradiente; la representación previa recibe gradiente supervisado inicialmente cero por los pesos de corrección nulos y puede aprender después. Los coeficientes del modelo lineal nunca reciben gradientes.

**La época 0 participa en la selección.** Su estado y criterio se guardan antes de entrenar. Si ninguna época mejora desarrollo al menos 1e-6, se conserva esa identidad y el resultado lleva `no_added_value`. No se obliga a seleccionar una corrección peor que la referencia.

## Seguridad temporal de la ventana bidireccional

Toda posición que entra en el encoder debe pertenecer a una ventana ya terminada y disponible al origen. El dataset y los resúmenes se truncan antes del forward. La API rechaza cortes intermedios: no sería seguro entregar una ventana más larga y limitar únicamente el decoder, porque estados bidireccionales, resúmenes o la referencia lineal podrían contener información posterior al corte.

El padding derecho se excluye en todas las memorias y lecturas; las consultas de horizonte usan autoatención diagonal, manteniendo independencia del conjunto de horizontes solicitado. Los tests de invariancia usan correcciones **no nulas** para que comprobar solo la identidad inicial no oculte una fuga interna.

La causalidad de disponibilidad sigue dependiendo del constructor de datos. La fuente clínica no publica latencias completas de laboratorio, por lo que se conserva la hipótesis retrospectiva ya declarada de disponibilidad de la fila al emitir. Bidireccional dentro del pasado conocido no convierte esa hipótesis en una latencia medida.

## Ajuste, selección y presupuesto

Se retienen todos los 21.661 orígenes de entrenamiento y 5.324 de desarrollo, con máscaras por variable/horizonte. Las muestras de calibración/prueba se eliminan del conjunto de ventanas antes de preparar objetivos o ejecutar la referencia congelada. La normalización y medianas se aplican desde los artefactos de entrenamiento, sin recalcular estadísticas sobre desarrollo.

Cada candidato usa AdamW con tasa fija 0,0003, weight decay 0,001, clipping de norma a 1, minibatches de 256, máximo de 15 épocas y paciencia de cuatro. No hay barrido de tasas. El objetivo es pinball macro por variable × horizonte con recuentos globales de entrenamiento, usando la misma estrategia de contribución `N/B` de la comparación neuronal anterior. La pérdida de desarrollo acumula recuentos completos por celda, no promedios arbitrarios de minibatches.

Las cabezas enteramente sin etiquetas en un lote no reciben actualización de parámetros exclusivos por AdamW; los encoders comunes sí pueden aprender desde otras cabezas. Los objetivos de poco soporte se conservan y se marcan, sin convertirlos en evidencia robusta por darles igual peso en el objetivo macro.

La ejecución comenzó con un presupuesto orientativo de 600 segundos y un gate de coste para el tercer candidato. Antes de entrenar ese tercer candidato, root autorizó ampliar el presupuesto a 1.200 segundos para completar la especialización temporal por grupo. La ampliación no cambia tasa, pérdida, épocas o particiones y no utiliza resultados de conjuntos reservados. Si la primera ejecución ya cerró los dos candidatos iniciales, se continúa exclusivamente con el tercero, conservando estados/curvas y los hashes históricos originales.

Un intento anterior quedó interrumpido tras guardar únicamente la época 0, sin curvas o checkpoint entrenado persistido. Se registró en `interrupted_attempt.json` y se reinició el mismo protocolo; no se seleccionó entre resultados de varias ejecuciones entrenadas de ese intento.

El criterio operacional de interés de la campaña se mantiene separado de la selección de checkpoint: mejoras pequeñas se informan tal cual, sin elevarlas a un umbral prefijado. Un objetivo ingenieril de mejora/cobertura no implica utilidad clínica ni autoriza cambiar selectivamente los datos para alcanzarlo.

## Artefactos y uso posterior con datos reservados

En `outputs/hourly_architecture_trials/`:

- `fit_manifest.json`: fuentes y hashes, configuración, identidad inicial, criterio, resultados por método y cualquier extensión de presupuesto.
- `*_epoch0.pt`: identidad inicial explícita por arquitectura.
- `causal_residual.pt`, `window_bidirectional_residual.pt`, `group_temporal_residual.pt`: estados seleccionados cuando cada ensayo haya terminado.
- `learning_curves.csv`: criterio por época, incluyendo la época 0.
- `predictions.csv`: **solo desarrollo**, con las mismas claves y unidades de los otros modelos.
- `scalers.npz`: copia de los escaladores congelados.
- `training_run.log`: progreso persistido de la ejecución de los dos primeros candidatos; una continuación conserva su log aparte.

La primera fase se lanzó con `python scripts/train_hourly_architecture_trials.py --epochs 15 --batch-size 256 --budget-seconds 600`. La continuación autorizada utiliza `python scripts/train_hourly_architecture_trials.py --epochs 15 --batch-size 256 --budget-seconds 1200 --group-only`. Ese modo exige los dos checkpoints iniciales, impide repetir una tercera variante ya completada, comprueba que los artefactos congelados no cambiaron y añade curvas/predicciones del tercero. El script de la fase inicial se conserva con su hash original; la continuación registra su propia versión.

API para cargar un candidato congelado:

```python
model, state = load_trial_checkpoint(checkpoint_path)
neural, summary = prepare_frozen_trial_data(packed, frozen_scalers)
model_args, _, label_mask = neural.batch(rows, state["groups"])
baseline_raw = predict_linear(summary, frozen_linear, frozen_bands, rows)
baseline_z = (baseline_raw - neural.mean[None,None,:,None]) / neural.scale[None,None,:,None]
prediction_z = model(model_args, torch.from_numpy(summary.features[rows]),
                     torch.from_numpy(baseline_z))
```

`prediction_z` se devuelve como `[B,H,V,3]` en el orden de `state['variables']`; se desestandariza con `scale` y `mean`. `prepare_frozen_trial_data` puede operar sobre un paquete que contenga solo episodios reservados: **no exige ni ajusta una nueva partición de entrenamiento**. Esto evita recalcular escalas al confirmar un candidato más adelante. La evaluación reservada no se ejecuta aquí; corresponde a una fase posterior con selección ya congelada y protocolos independientes.

## Verificación

Nueve pruebas específicas comprobaron identidad exacta de época 0 para las tres arquitecturas, eje temporal conservado, orden con correcciones extremas y gaps cero, invariancia de padding y del conjunto de horizontes con correcciones no nulas, rechazo de cortes inseguros, coincidencia exacta de los resúmenes con los baselines originales, aplicación de escaladores en paquetes sin entrenamiento y recarga de checkpoints mediante `weights_only=True`.

Estos tests verifican software y seguridad del flujo de información. No son medidas de capacidad de pronóstico ni justifican crear etiquetas espaciales horarias a partir de visitas esporádicas.

## Resultado de la campaña completada

La referencia congelada obtuvo pinball macro estandarizada de desarrollo **0,156520754**. Los tres ensayos conservaron esa misma referencia exacta en época 0:

| Arquitectura | Época elegida / ejecutadas | Parámetros | Pinball macro de desarrollo | Mejora relativa frente a lineal | Ajuste y selección |
|---|---:|---:|---:|---:|---:|
| Causal residual | 5 / 9 | 269.934 | 0,155867040 | 0,418 % | 173,56 s |
| Ventana pasada bidireccional residual | 3 / 7 | 269.934 | 0,156001732 | 0,332 % | 143,34 s |
| Memoria temporal por grupo residual | 8 / 12 | 422.382 | 0,155082911 | 0,919 % | 573,99 s |

Las tres ejecuciones pararon por la paciencia prevista. El tiempo acumulado, con preparación y exportación, fue 903,60 segundos, dentro de la ampliación autorizada a 1.200. No se reentrenaron las dos primeras al añadir la tercera: se verificó que sus hashes permanecían exactamente iguales.

El CSV final contiene **353.532 filas solo de desarrollo**, 117.844 etiquetas iguales por método. La mayor mejora en pinball es pequeña y no basta por sí sola para decidir utilidad. El informe integrado de root identifica para el modelo por grupo cobertura del **74,11 %**, frente a **77,84 %** de la lineal, por lo que mejorar esta pérdida no significó mejorar simultáneamente la cobertura. La comparación de soporte físico y estabilidad por episodio se conserva en ese informe; no se han recalibrado las bandas ni cambiado parámetros tras conocer el resultado.

Root congeló la elección para una confirmación descriptiva independiente con el candidato por grupo y la referencia lineal. No se considera cumplido el objetivo ingenieril orientativo y no se entrenó un ensemble. Los artefactos seleccionados permanecen congelados; cualquier evaluación de confirmación queda fuera de este script y no puede usarse para modificar estos modelos.
