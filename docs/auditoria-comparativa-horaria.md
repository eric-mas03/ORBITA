# Auditoría independiente de la comparación horaria

## Alcance y estado

Se revisaron los datos empaquetados, las particiones fijadas, los escaladores de los baselines y el preprocesador compartido por GRU y Transformer. La auditoría no entrena modelos ni modifica datos, escaladores o predicciones. Las pruebas están en `tests/test_hourly_neural_data.py`.

Resultado final: **9 pruebas superadas** con los archivos y las exportaciones completas de la ejecución autorizada. El preprocesamiento y las predicciones exportadas de ambos modelos neurales se contrastaron con el NPZ y con los baselines congelados.

## Fronteras de datos comprobadas

1. **Escalado exclusivo de entrenamiento.** Se reconstruyeron los índices de las filas de los 600 episodios train: 23.178 filas únicas, no ventanas repetidas. Media, desviación, mediana y número de valores observados del preprocesador neural coinciden con `outputs/hourly_baselines/scalers.npz`. Se verificaron además las estadísticas guardadas frente al cálculo independiente sobre las filas originales de train.
2. **Mutación de held-out.** Se alteraron en memoria todos los valores finitos de desarrollo, calibración y prueba. No cambiaron las estadísticas de train, sus características normalizadas, su batch ni las asignaciones de partición. No se escribió esa copia en disco.
3. **Causalidad dentro del episodio.** Se eligió un origen real de desarrollo con 24 slots históricos y etiquetas posteriores; se modificaron únicamente los valores posteriores de ese episodio. Permanecieron iguales los valores históricos, máscaras, timestamps, edades y argumentos del modelo. Las etiquetas cambiaron, como debían. Se usa desarrollo en esta prueba para mantener congelado el ajuste global del escalador de train.
4. **Contexto y padding.** Los batches conservan hasta 24 slots reales desde `history_start` hasta `origin`, incluidos los historiales parciales. El padding se sitúa a la derecha, tiene máscara verdadera y no aporta observaciones ni antigüedad artificial. Los valores ausentes se representan por cero después de normalizar, acompañados por máscara falsa; ese cero no se presenta como una observación.
5. **Edades causales.** En cada paso y variable, la antigüedad se contrastó con el último valor observado en ese paso o antes. Antes de la primera medición, la implementación usa tiempo desde el inicio de contexto junto a máscara de observación falsa. Una observación futura no reinicia edades anteriores.
6. **Etiquetas independientes por variable y horizonte.** Se reconstruyeron las etiquetas desde las filas globales de `target_rows`; las inexistentes y las celdas ausentes permanecen enmascaradas. Se verificó `ICULOS[target]−ICULOS[origin] = h` para 1, 4, 8 y 24 horas, dentro del mismo episodio. No hay interpolación ni exigencia de que las 34 variables o los cuatro horizontes estén completos.
7. **Particiones y procedencia congeladas.** El SHA-256 de `splits.json` coincide con el manifiesto y cada identificador de episodio mantiene la asignación guardada. El SHA-256 del NPZ coincide con los manifiestos del constructor, de los baselines y de los modelos neurales. Ambos modelos quedaron congelados antes de exportar prueba. Son particiones por episodio y sitio, no una garantía de independencia entre personas si hubiera ingresos repetidos sin identidad enlazable.
8. **Alineación de grupos.** Los 13 grupos biomédicos reales cubren las 34 variables exactamente una vez. Se comprobó que agrupar entradas y volver a aplanar salidas conserva el orden de variables del dataset, evitando aplicar una escala o etiqueta a otra variable.

## Baselines: equivalencia de objetivos exportados

Se reconstruyó de forma independiente el conjunto de claves esperado desde el NPZ: `(split, episode_id, origin_hour, horizon_hours, variable)`. Para persistencia y regresión cuantílica lineal se comprobó mediante unión uno a uno que:

- No hay claves duplicadas, predicciones sin etiqueta observada ni etiquetas evaluables omitidas.
- Los valores `observed` coinciden con las celdas reales del dataset filtrado por calidad.
- `target_scale` corresponde al escalador train de la variable correcta.
- Ambas exportaciones juntas contienen 610.456 filas: son dos métodos evaluados sobre los mismos objetivos, no 610.456 observaciones independientes.

La elegibilidad de un par puede depender de que exista una etiqueta futura para evaluarlo; esa disponibilidad no se incorpora como característica de entrada. Los resultados describirán los objetivos realmente observados, sujetos a su patrón de muestreo.

## GRU y Transformer: unión final de predicciones

Se auditaron `outputs/hourly_neural/predictions.csv` y sus escaladores después de finalizar la ejecución. Cada método neural aporta exactamente las mismas **305.228 claves observadas** que cada baseline:

| Partición | Pares por método | Métodos contrastados |
|---|---:|---|
| Desarrollo | 117.844 | Persistencia, lineal cuantílico, GRU con lectura compartida y Transformer agrupado |
| Calibración | 71.240 | Los mismos cuatro |
| Prueba | 116.144 | Los mismos cuatro |

La unión uno a uno no produjo claves huérfanas ni duplicadas. Los valores observados y las escalas de objetivo coinciden, con tolerancia numérica de serialización. Los tiempos objetivo son exactamente origen más horizonte, y los recuentos de historia observada y la longitud de contexto coinciden sin tolerancia entre exportaciones. Los modelos neurales no declaran fallback a mediana; sus cuantiles son finitos y están ordenados.

Además del total, se contrastó la cuadrícula completa de **408 celdas: 34 variables × cuatro horizontes × tres particiones**, preservando explícitamente las celdas de soporte cero. Así se evita que la igualdad global esconda la omisión de una salida u horizonte difícil. Las cifras de los cuatro métodos se puntúan sobre los mismos objetivos y las mismas escalas de entrenamiento; esta auditoría no selecciona ni recalibra modelos a partir de prueba.

Los dos CSV de predicciones contienen conjuntamente 1.220.912 filas de métodos, correspondientes a cuatro pronósticos por cada una de las 305.228 claves compartidas; no representan ese número de pacientes ni muestras independientes.

## Límites de la comparación

Estas pruebas comprueban las fronteras de datos y su alineación. No demuestran capacidad predictiva, calibración de q10–q90, independencia de ventanas solapadas, transportabilidad desde UCI a astronautas ni disponibilidad operacional instantánea de análisis de laboratorio. Los intervalos siguen siendo marginales y su calibración requiere evaluación separada.

Las estadísticas de entrenamiento se ajustan una vez con el conjunto train completo para una evaluación offline congelada. La causalidad de la ventana se verifica manteniendo ese ajuste fijo; no se confunde con exigir que el entrenamiento offline ignore toda información posterior de sus propios episodios.

La configuración de candidatos mantiene metadatos históricos de planificación, incluido `left_pad_without_fabricating_observations`; la implementación y las pruebas actuales usan padding derecho. Esa diferencia documental no altera los batches verificados y debe describirse según el comportamiento implementado.

## Reproducir sin entrenar

```powershell
python -m pytest tests/test_hourly_neural_data.py -q -p no:cacheprovider
```

Solo requiere los datos y artefactos existentes. Las mutaciones de control ocurren en copias en memoria.
