# Primer bloque horario: ventanas, particiones y baselines

Experimento sobre el subconjunto clínico terrestre descargado. No es una validación en astronautas. Los pronósticos son del valor publicado del intervalo terminal a 1, 4, 8 y 24 horas.

## Ejecución y contrato

Contexto causal de hasta 24 slots horarios, admitiendo historias cortas y valores ausentes. Los objetivos conservan máscaras independientes por variable y horizonte. No se interpolan, rellenan ni fabrican etiquetas de laboratorio. Los mínimos/máximos de contexto resumen mediciones horarias observadas en el pasado; no son mínimos/máximos intrahorarios reconstruidos.

La división se fija por episodio y hospital antes del escalado. Se reservan entrenamiento, desarrollo, calibración y prueba; las personas subyacentes a ingresos distintos no se pueden deduplicar con estos archivos. El optimizador y la selección usan entrenamiento/desarrollo. La calibración permanece reservada y las bandas de este bloque son sin calibrar.

Persistencia usa el último valor realmente observado dentro del contexto; cuando falta, usa mediana de entrenamiento y lo marca. Sus bandas proceden de residuos de entrenamiento. La referencia lineal cuantílica utiliza características causales de cada variable y produce tres cuantiles por horizonte, con normalización de entrenamiento. Ver configuración y decisiones exactas en los artefactos de ajuste.

## Prueba por horizonte

Promedios normalizados con igual peso por variable evaluada. La cobertura es media de coberturas marginales; no garantía conjunta. Los recuentos suman parejas etiqueta/predicción, no personas independientes.

| Método | Horizonte | Variables evaluadas | MAE normalizado | Pinball normalizada | Cobertura |
|---|---:|---:|---:|---:|---:|
| linear_quantile | 1 h | 34 | 0.420 | 0.144 | 78.7 % |
| linear_quantile | 4 h | 34 | 0.460 | 0.156 | 77.8 % |
| linear_quantile | 8 h | 34 | 0.487 | 0.165 | 78.4 % |
| linear_quantile | 24 h | 34 | 0.595 | 0.193 | 77.1 % |
| persistence | 1 h | 34 | 0.430 | 0.159 | 81.2 % |
| persistence | 4 h | 34 | 0.490 | 0.175 | 80.3 % |
| persistence | 8 h | 34 | 0.528 | 0.187 | 80.0 % |
| persistence | 24 h | 34 | 0.578 | 0.205 | 78.8 % |

## Resultados inspeccionables

- `outputs/hourly_evaluation/metrics_by_variable_horizon.csv`: las 136 celdas por modelo/split, con etiquetas, episodios, unidades, calibración y soporte. Sin etiquetas: NA, no error cero.
- Métricas por cabeza, hospital, contexto parcial/completo y episodio en el mismo directorio. Las escalas originales permanecen en los resultados por variable.
- Mapas de pinball, cobertura y número de etiquetas; mapa por cabeza/horizonte, frecuencias de cuantiles, comparación por horizonte y ejemplos de orígenes móviles a 4 h.
- `outputs/hourly_baselines/`: predicciones y artefactos de ajuste; `data/processed/hourly/` y `outputs/hourly_data/`: datos y particiones del constructor.

La prueba puntúa 150 episodios. Hay 0 celdas modelo–variable–horizonte sin etiquetas y 8 con menos de diez episodios: el umbral solo marca escasez para inspección, no certifica suficiencia de las demás.

## Interpretación y siguiente experimento

Las diferencias se revisan por variable/cabeza/horizonte y junto a cobertura/anchura. No se elige un modelo mirando solo una media ni se retiran a posteriori las variables difíciles de prueba. El siguiente paso es comparar una dinámica recurrente pequeña y la arquitectura temporal agrupada con estas mismas particiones y objetivos, seleccionando exclusivamente en desarrollo. La posterior calibración usará su conjunto reservado; ampliar arquitectura no garantiza calibración ni transferencia espacial.

Las unidades con revisión pendiente permanecen identificadas; no se convierten por plausibilidad. FiO2 y otros valores físicamente imposibles se tratan según la política explícita, conservando originales y registro de calidad. Los resultados no se interpretan como recomendaciones clínicas.

Los dos baselines conservan sus cuantiles ordenados sin proyección posterior a soporte físico. `support_violation_fraction` registra la fracción de bandas que sobrepasan un límite físico de la política, por variable/horizonte. Es una limitación explícita a resolver en la siguiente comparación; no se recortan predicciones tras mirar prueba para mejorar sus métricas. Es diferente del control de calidad aplicado a las observaciones de entrada.

## Reproducir

```powershell
python scripts/build_hourly_dataset.py
python scripts/train_hourly_baselines.py
python scripts/report_hourly_baselines.py
python -m pytest -q
```
