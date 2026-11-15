# Comparación horaria de memorias GRU y Transformer

Experimento sobre el subconjunto clínico terrestre descargado. No es una validación en astronautas. Los pronósticos son del valor publicado del intervalo terminal a 1, 4, 8 y 24 horas.

## Ejecución y contrato

Contexto causal de hasta 24 slots horarios, admitiendo historias cortas y valores ausentes. Los objetivos conservan máscaras independientes por variable y horizonte. No se interpolan, rellenan ni fabrican etiquetas de laboratorio. Los mínimos/máximos de contexto resumen mediciones horarias observadas en el pasado; no son mínimos/máximos intrahorarios reconstruidos.

La división se fija por episodio y hospital antes del escalado. Se reservan entrenamiento, desarrollo, calibración y prueba; las personas subyacentes a ingresos distintos no se pueden deduplicar con estos archivos. El optimizador y la selección usan entrenamiento/desarrollo. La calibración permanece reservada y las bandas de este bloque son sin calibrar.

Persistencia usa el último valor realmente observado dentro del contexto; cuando falta, usa mediana de entrenamiento y lo marca. Sus bandas proceden de residuos de entrenamiento. La referencia lineal cuantílica utiliza características causales de cada variable y produce tres cuantiles por horizonte, con normalización de entrenamiento. Ver configuración y decisiones exactas en los artefactos de ajuste.

## Criterio de desarrollo y comparación congelada

La métrica de selección es pinball normalizada macro por variable y horizonte. La siguiente tabla separa desarrollo de prueba descriptiva; no se eligen hiperparámetros con la columna de prueba.

| Modelo | Pinball macro en desarrollo ↓ | Pinball macro en prueba ↓ |
|---|---:|---:|
| GRU · readout común | 0.178455 | 0.191133 |
| Lineal cuantílica | 0.156521 | 0.164499 |
| Persistencia | 0.179530 | 0.181548 |
| Transformer agrupado | 0.175066 | 0.188094 |

Bajo este presupuesto, la referencia lineal conserva el menor criterio de desarrollo. Entre las dos memorias, Transformer mejora ligeramente a GRU, sin demostrar una ventaja estable con una sola semilla. Las dos redes quedan por detrás de la referencia lineal en el agregado de prueba; los CSV permiten inspeccionar diferencias individuales sin redefinir a posteriori los objetivos.

La decisión provisional es mantener la referencia lineal y estudiar en desarrollo si preservar explícitamente el último valor y resúmenes propios mediante una conexión residual, o revisar el cuello de botella latente, mejora el candidato neuronal. Es una hipótesis futura: este experimento no identifica cuál componente explica la diferencia ni ejecuta esa ampliación.

## Prueba por horizonte

Promedios normalizados con igual peso por variable evaluada. La cobertura es media de coberturas marginales; no garantía conjunta. Los recuentos suman parejas etiqueta/predicción, no personas independientes.

| Método | Horizonte | Variables evaluadas | MAE normalizado | Pinball normalizada | Cobertura |
|---|---:|---:|---:|---:|---:|
| gru_shared_readout | 1 h | 34 | 0.538 | 0.183 | 75.6 % |
| gru_shared_readout | 4 h | 34 | 0.546 | 0.185 | 74.2 % |
| gru_shared_readout | 8 h | 34 | 0.570 | 0.193 | 73.3 % |
| gru_shared_readout | 24 h | 34 | 0.597 | 0.204 | 72.0 % |
| linear_quantile | 1 h | 34 | 0.420 | 0.144 | 78.7 % |
| linear_quantile | 4 h | 34 | 0.460 | 0.156 | 77.8 % |
| linear_quantile | 8 h | 34 | 0.487 | 0.165 | 78.4 % |
| linear_quantile | 24 h | 34 | 0.595 | 0.193 | 77.1 % |
| persistence | 1 h | 34 | 0.430 | 0.159 | 81.2 % |
| persistence | 4 h | 34 | 0.490 | 0.175 | 80.3 % |
| persistence | 8 h | 34 | 0.528 | 0.187 | 80.0 % |
| persistence | 24 h | 34 | 0.578 | 0.205 | 78.8 % |
| transformer_grouped | 1 h | 34 | 0.523 | 0.178 | 76.2 % |
| transformer_grouped | 4 h | 34 | 0.531 | 0.180 | 74.9 % |
| transformer_grouped | 8 h | 34 | 0.552 | 0.187 | 73.7 % |
| transformer_grouped | 24 h | 34 | 0.602 | 0.207 | 71.2 % |

## Resultados inspeccionables

- `outputs/hourly_comparison/metrics_by_variable_horizon.csv`: las 136 celdas por modelo/split, con etiquetas, episodios, unidades, calibración y soporte. Sin etiquetas: NA, no error cero.
- Métricas por cabeza, hospital, contexto parcial/completo, presencia/ausencia de historia propia y episodio en el mismo directorio. Las escalas originales permanecen en los resultados por variable.
- Mapas de pinball, cobertura y número de etiquetas; mapa por cabeza/horizonte, frecuencias de cuantiles, comparación por horizonte y ejemplos de orígenes móviles a 4 h.
- `outputs/hourly_baselines/`: predicciones y artefactos de ajuste; `data/processed/hourly/` y `outputs/hourly_data/`: datos y particiones del constructor.

La prueba puntúa 150 episodios. Hay 0 celdas modelo–variable–horizonte sin etiquetas y 16 con menos de diez episodios: el umbral solo marca escasez para inspección, no certifica suficiencia de las demás.

## Interpretación y siguiente experimento

Las diferencias se revisan por variable/cabeza/horizonte y junto a cobertura/anchura. No se elige un modelo mirando solo una media ni se retiran a posteriori las variables difíciles de prueba. La selección utiliza exclusivamente desarrollo. La posterior calibración usará su conjunto reservado; ampliar arquitectura no garantiza calibración ni transferencia espacial.

Las unidades con revisión pendiente permanecen identificadas; no se convierten por plausibilidad. FiO2 y otros valores físicamente imposibles se tratan según la política explícita, conservando originales y registro de calidad. Los resultados no se interpretan como recomendaciones clínicas.

Los modelos conservan sus cuantiles ordenados sin proyección posterior a soporte físico. `support_violation_fraction` registra la fracción de bandas que sobrepasan un límite físico de la política, por variable/horizonte. Es una limitación explícita; no se recortan predicciones tras mirar prueba para mejorar sus métricas. Es diferente del control de calidad aplicado a las observaciones de entrada.

## Reproducir

```powershell
python scripts/build_hourly_dataset.py
python scripts/train_hourly_baselines.py
python scripts/report_hourly_baselines.py
python -m pytest -q
```

## Qué se compara en las redes

GRU y Transformer comparten encoders por categoría, fusión, codificación temporal, consultas de horizonte y decoders de atención cruzada. Cambia la memoria temporal recurrente frente a autoatención causal. La variante GRU conserva el readout con atención: no es una GRU pura sin atención. Se registran parámetros y presupuesto en el manifiesto neuronal.

Es una comparación inicial con una semilla y un presupuesto acotado. No hay preentrenamiento, aumento ni ajuste con prueba. Las diferencias de capacidad y el ajuste bajo este presupuesto limitan la atribución de resultados a la atención. Una red peor en prueba no justifica volver a seleccionar hiperparámetros usando esa misma prueba.

Reproducir el informe combinado con `python scripts/report_hourly_baselines.py --include-neural`. Entrenamiento y configuración exacta en `docs/modelos-neuronales-horarios.md` y `outputs/hourly_neural/`. El informe histórico del primer bloque y sus figuras permanecen separados.
