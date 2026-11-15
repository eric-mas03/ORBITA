# Comparativa horaria GRU y Transformer agrupado

Este experimento entrena desde cero dos variantes sobre el mismo dataset horario terrestre y las mismas particiones de los baselines. No ejecuta aumento de datos, preentrenamiento externo, adaptación espacial ni calibración. La interpretación del resultado pertenece al dominio UCI; no valida un pronóstico horario en astronautas.

## Diferencia controlada de arquitectura

Ambas variantes usan los mismos encoders MLP por categoría, concatenación/proyección de embeddings y presencia, codificación de tiempo, consultas de horizonte y decoders Transformer por grupo de salida. Estos decoders mantienen atención cruzada al historial y autoatención diagonal de consultas: solicitar otro horizonte no debe modificar un horizonte existente durante evaluación.

La única parte sustituida es la memoria temporal:

- `transformer_grouped`: memoria Transformer de autoatención causal con producto escalar escalado, una capa, dimensión 16 y dos cabezas internas.
- `gru_shared_readout`: memoria GRU causal de una capa y estado 16, seguida de dropout y LayerNorm, adaptada a la interfaz de la memoria anterior. Conserva los decoders con atención cruzada por grupo; **no es una GRU pura sin atención**.

Las partes comunes tienen inicialización exactamente igual con semilla 42, verificada por test. Los parámetros de las memorias y, por tanto, el total difieren; se registran ambos recuentos. Se conserva anchura y profundidad de referencia, sin afirmar igualdad de capacidad. Dropout es 0,1 y la escala temporal 24 horas. El controlador temporal GRU usa longitudes reales con secuencias empaquetadas para que el padding derecho no modifique estados válidos.

## Mismos datos y preprocesamiento

Se usan las 34 variables dinámicas y grupos del contrato horario, todos los 21.661 orígenes de entrenamiento, 5.324 de desarrollo, 3.290 de calibración y 5.307 de prueba. Las particiones siguen episodios/hospitales congelados de los baselines. El archivo identifica un episodio, sin garantizar identidad única entre distintos ingresos.

Las etiquetas son las celdas reales físicamente válidas en `t+1`, `t+4`, `t+8` y `t+24`, interpretadas como valor publicado del intervalo terminal. No se generan nuevas etiquetas ni se arrastran resultados hacia el futuro. Las historias conservan hasta 24 slots reales `[t-23,t]`; las parciales se completan únicamente con padding a la derecha.

Se ajustan media/escala/mediana usando filas únicas de episodios de entrenamiento. La media y escala coinciden con el contrato de los baselines. Las entradas observadas se estandarizan; las ausentes reciben cero numérico normalizado con máscara falsa. **No es imputación mediante mediana de entrenamiento**, por lo que `used_train_median_fallback=False` en las filas neuronales. Se conservan los recuentos de observaciones propias para evaluar casos donde una cabeza predice a partir de otras modalidades sin historia propia.

Las edades se calculan causalmente desde la última medición dentro del contexto. Antes de la primera observación, edad significa tiempo desde el inicio del contexto, acompañada de la máscara; no fecha conocida de una medición anterior inexistente. Los tiempos son horas reales del episodio y los intervalos de laboratorio desconocidos no se rellenan. La disponibilidad inmediata de una fila al emitir es una hipótesis retrospectiva explícita, con latencia de laboratorio no publicada.

## Pérdida y presupuesto

El objetivo conserva balance por variable × horizonte sin favorecer automáticamente vitales densos. Primero se calcula el promedio de pinball entre cuantiles en cada etiqueta observada. Cada suma de etiquetas recibe el inverso del **recuento global de entrenamiento** de su celda variable/horizonte y después se divide entre las celdas con soporte. Una celda con cinco etiquetas conserva esas cinco, pero su estimación y gradiente tienen gran incertidumbre.

Para un minibatch de tamaño `B` sobre `N` orígenes, su contribución se multiplica por `N/B`: es una estimación del objetivo global con minibatches muestreados por permutación. No se divide por el recuento de una celda dentro de cada pequeño lote. Un test comprueba que las contribuciones ponderadas de minibatches recuperan el mismo objetivo completo para parámetros fijos.

Si un lote no tiene ninguna etiqueta se omite su actualización. Si una cabeza no tiene ninguna etiqueta en un lote, sus parámetros exclusivos de decoder/salida reciben `grad=None` antes de AdamW para evitar actualizaciones por momentum/decay. La representación compartida sí puede actualizarse desde otras cabezas. Las filas individuales sin etiqueta dentro de una cabeza activa siguen sujetas al comportamiento del tensor compartido y del optimizador: no se promete congelación por biomarcador.

Ambas variantes tienen el mismo presupuesto: AdamW con tasa 0,001 y weight decay 0,001, clipping de norma de gradiente a 1, minibatches de 256, máximo de 20 épocas y parada tras cinco sin mejora de al menos 1e-6. No hay búsqueda de tasas, semilla o profundidad. Cada modelo selecciona su checkpoint únicamente mediante pinball estandarizada macro variable/horizonte de desarrollo; no usa calibración ni prueba. El presupuesto máximo es igual, aunque la parada y época elegida pueden diferir.

La curva de entrenamiento registra una suma de contribuciones online durante la época: los parámetros cambian entre lotes, por lo que no es una evaluación estática de un único checkpoint. Desarrollo sí se evalúa después de la época con modelo fijo, dropout desactivado y acumulación global de pérdidas/recuentos. La selección y los estados se guardan antes de generar cualquier predicción de prueba.

## Salidas y límites

Las salidas mantienen cuantiles ordenados mediante la parametrización del modelo, en sus unidades originales tras desestandarizar. No se proyectan a límites físicos, no se recortan rangos clínicos y no se aplica CQR. Una anchura positiva no garantiza cobertura nominal del 80%; la tasa de valores pronosticados fuera de soporte físico debe revisarse como en los baselines.

El CSV usa las mismas etiquetas, claves, escalas y columnas necesarias para unir los dos baselines: método, split, episodio, hospital, origen, objetivo, horizonte, grupo, variable, unidad, revisión de unidades, observación, q10/q50/q90, longitud/duración del contexto y recuento propio. Se añaden estados para objetivos con poco o ningún soporte. Un objetivo sin etiquetas de entrenamiento se considera no entrenado, no un pronóstico validado por producir un número.

Las filas no observadas no reciben error cero; las métricas se calculan en los mismos pares reales para todos los métodos. El informe posterior puede separar contexto parcial/completo y ausencia de historia propia. Los múltiples orígenes que comparten resultados futuros no son observaciones independientes. La comparación de una única semilla y población tampoco mide estabilidad entre repeticiones o nuevas poblaciones.

## Archivos y reproducción

Comando: `python scripts/train_hourly_neural.py --epochs 20 --batch-size 256`.

En `outputs/hourly_neural/` se conservan:

- `fit_manifest.json`: hashes de datos/código/política, arquitectura, presupuesto, criterio, selección, recuentos y tiempos.
- `scalers.npz`: estadísticas, recuentos de observaciones y filas únicas de entrenamiento.
- `gru_shared_readout.pt` y `transformer_grouped.pt`: estado seleccionado y configuración de cada variante.
- `learning_curves.csv`: evolución por época de entrenamiento online, desarrollo, lotes y duración.
- `predictions.csv`: desarrollo, calibración y prueba para ambas variantes, después de congelar sus selecciones.

Los checkpoints son artefactos locales de este experimento; contienen también arrays NumPy de escalado, además del `state_dict`. El escalado se conserva asimismo en NPZ para inspección independiente.

## Pruebas

`tests/test_hourly_neural.py` verifica causalidad al modificar valores/fechas futuras, invariancia al conjunto de horizontes, padding y equivalencia con predicciones individuales para ambas memorias, inicialización común exacta, alineación y gradientes del orden de variables, edades/máscaras y objetivo global balanceado. Los ejemplos matemáticos son tests de software, no datos fisiológicos inventados para entrenar.

La auditoría independiente del dataset y de la comparación se documenta aparte. Los resultados cuantitativos finales deben leerse del informe comparativo producido desde predicciones guardadas; no se elige el método según la apariencia de una curva de entrenamiento.

## Ejecución completada

Ambas variantes completaron el presupuesto de 20 épocas sobre todos los orígenes de entrenamiento, sin evaluar prueba para seleccionar:

| Variante | Parámetros totales | Parámetros de memoria | Época elegida | Pinball macro estandarizada de desarrollo | Tiempo de ajuste |
|---|---:|---:|---:|---:|---:|
| GRU con readout de atención compartido en diseño | 69.814 | 1.664 | 19 | 0,178455 | 220,20 s |
| Transformer agrupado | 71.430 | 3.280 | 18 | 0,175066 | 229,53 s |

El resto de parámetros corresponde a los mismos componentes de representación y lectura, entrenados independientemente en cada variante. Los tiempos son de ajuste y selección, sin incluir toda la preparación/exportación. Esta ejecución de una semilla no demuestra una ventaja general de una arquitectura.

Se exportaron 610.456 filas para las dos variantes: 117.844 etiquetas de desarrollo, 71.240 de calibración y 116.144 de prueba por método. Las 16 pruebas del modelo horario y de la auditoría de datos neuronal pasaron tras completar la ejecución. El informe integrado compara los cuatro métodos sobre estas mismas etiquetas, con recuentos y calidad de soporte, en vez de decidir por las pérdidas de desarrollo de esta tabla.
