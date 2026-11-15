# Arquitectura y ablaciones para pronóstico horario

Actualización: 14 de noviembre de 2026. **Etapa de planificación**: este documento no cambia el modelo, no entrena y no implementa el dashboard. El contrato estructurado está en `config/experiment_plan.json`. Los estados `agreed` registran la dirección pedida por el usuario; `proposed` identifica elecciones pendientes de comprobar con los datos y acordar antes de ejecutarlas.

## Dirección y decisión inicial

Se toma como dirección acordada pronosticar q10/q50/q90 para horizontes de **1, 4, 8 y 24 horas**, utilizar todas las etiquetas válidas aunque sean parciales y analizar resultados por cabeza y horizonte. El contexto de referencia es de **24 horas**; su suficiencia no está demostrada. Se estudiarán alternativas de 6, 12 y 48 horas solo si el inventario de ventanas/etiquetas indica que son viables. La arquitectura de referencia mantiene encoders por categorías, representación temporal común y cabezas para distintos objetivos.

La propuesta inicial para datos publicados con resolución horaria es usar bins de una hora: 24 posiciones para el contexto de 24 horas. El candidato de bins de 15 minutos daría 96 posiciones, pero solo se abrirá si hay observaciones reales con esa resolución y cadencia útil en los objetivos. Repetir cada fila horaria cuatro veces o interpolarla no genera resolución de 15 minutos ni nuevas etiquetas. Que un archivo tenga miles de filas no demuestra que un paciente tenga 24 horas útiles de seguimiento.

Los análisis sanguíneos espaciales de visitas separadas por días o meses no permiten evaluar un objetivo a 1/4/8/24 horas. No se crearán etiquetas de vuelo horarias a partir de esas visitas. Hasta encontrar datos de vuelo con cadencia y duración suficientes, la tarea horaria podrá desarrollarse y evaluarse en su dominio clínico terrestre, manteniendo ese resultado separado del espacial.

## Origen y definición exacta de cada ejemplo

La propuesta es emitir en un borde de rejilla `t`, una vez completado el último bin histórico y disponibles sus datos. Para una resolución `Δ`, el último bin de entrada es `(t-Δ, t]`; la ventana de 24 horas contiene los bins completos hasta `t`. El uso de intervalos abiertos por la izquierda y cerrados por la derecha evita duplicar una medición entre bins. El origen de predicción y la fecha de disponibilidad se conservan explícitamente.

Solo se usan observaciones con `event_time <= t` y `available_time <= t`. Si un resultado llega tarde, no puede entrar retrospectivamente en una predicción ya emitida. Para preparar etiquetas, su disponibilidad debe ser compatible con la fecha del ajuste o evaluación simulados. No se supone que una fecha de extracción de sangre equivale a la fecha de publicación del resultado. Si el conjunto no tiene latencia, cualquier disponibilidad inmediata es una hipótesis declarada, no un hecho medido.

Los objetivos se deben fijar antes del entrenamiento. La primera definición propuesta es la **media de valores realmente observados en el bin futuro que termina en `t+h`**, es decir `(t+h-Δ, t+h]`. Con `Δ=1h`, el horizonte de 1h corresponde a `(t,t+1h]`; no a la media de las siguientes 24 horas. Otras tareas posibles son el mínimo o máximo de ese bin, un valor puntual con una regla temporal predefinida, o un agregado de todo `(t,t+h]`. Son objetivos distintos y no deben mezclarse bajo la misma etiqueta de horizonte.

Un bin sin observaciones válidas no tiene etiqueta. La media es una media de muestras observadas, no una media fisiológica continua garantizada; la cobertura y el mecanismo de muestreo pueden sesgarla. La regla de cobertura mínima por variable/ensayo se decide tras inspeccionar recuentos y frecuencia de medida, antes de comparar modelos. Para laboratorios de una muestra por bin, esa regla no puede ser la misma que para sensores densos.

## Media, mínimo y máximo de entrada

La propuesta es calcular media/mínimo/máximo de las observaciones históricas válidas dentro de cada bin, siempre que exista muestreo intrabin real suficiente. Añadir recuento y máscara de disponibilidad como metadatos, y edad de la última observación. No calcular estadísticas con un bin aún abierto ni ventanas centradas que atraviesen `t`.

Si el origen ya publica una única fila agregada por hora, no podemos recuperar el mínimo o máximo intrahorario de la señal original. En ese caso se empieza con el valor horario y sus máscaras; triplicar el mismo valor como media/mínimo/máximo sería redundante. Las columnas de origen que ya sean medias o máximos mantienen su semántica. Las estadísticas de entrada no obligan a predecir las tres estadísticas como objetivos; esa elección sigue propuesta y requiere un objetivo de uso concreto.

q10/q50/q90 son cuantiles de la distribución predictiva de la variable objetivo elegida. No son el mínimo, la mediana empírica ni el máximo de los valores dentro del bin. `[q10,q90]` conserva su interpretación de banda marginal nominal del 80%, pendiente de calibración.

## Representaciones y modalidades de distinta frecuencia

La arquitectura existente concatena embeddings MLP de categorías y presencia, proyecta a un latente, aplica autoatención temporal causal y utiliza decoders por grupo con consultas de horizonte y atención cruzada al historial. Su autoatención entre consultas es diagonal para que pedir otro horizonte no cambie el pronóstico anterior. No hay todavía atención explícita entre grupos ni sesgo relativo temporal en los logits de atención.

Para la fase horaria se propone conservar esa referencia pequeña y separar dos vistas cuando existan datos multimodales realmente enlazados:

- **Vista rápida:** vitales/sensores en bins de 1h inicialmente; 24 posiciones para la configuración de referencia. Cada posición contiene categorías disponibles, estadísticas justificadas, máscara, edad y recuento.
- **Vista lenta:** eventos originales de laboratorio u otras modalidades esporádicas, con marca de medida y disponibilidad reales. Su ventana puede ser distinta, condicionada al uso y a la relevancia temporal de cada variable. No se convierten en 24/96 mediciones nuevas repitiendo una analítica antigua.

La opción candidata es encoders por categoría y vista, memorias temporales con máscaras propias y un espacio latente de tamaño común. Una consulta de objetivo/horizonte podría leer ambas memorias mediante atención cruzada, distinguiendo modalidad y edad. Es una ampliación propuesta, no una función del código actual. También se comparará una referencia más simple que incorpore la última analítica disponible con su antigüedad; repetirla como contexto no equivale a observarla de nuevo ni autoriza una etiqueta nueva.

La identidad persona/episodio debe estar verificada para enlazar vistas. No emparejar laboratorios de una persona con sensores de otra, aunque pertenezcan a la misma misión. Una cabeza solo podrá aprender asociaciones entre modalidades con soporte real de coobservación o con una hipótesis de transferencia explícita y evaluada.

## Muestras parciales y ponderación de la pérdida

Mantener todas las ventanas elegibles según criterios causales, sin exigir que estén presentes todas las variables futuras. Cada objetivo en cada persona/origen/horizonte tiene su máscara; esa máscara no se confunde con la de inputs, padding, censura analítica o corrupción artificial. Una cabeza completamente sin etiquetas no contribuye a la pérdida del lote. Si no hay ninguna etiqueta válida, se omite la actualización.

La implementación actual ofrece `equal_head` y `per_target`. El primero iguala cabezas activas; el segundo iguala posiciones escalares observadas y puede favorecer variables medidas con mayor frecuencia. Para la tarea horaria se propone contrastar, **sin implementarlo aún**, una ponderación por variable/horizonte: calcular primero la media en sus muestras observadas y después promediar las celdas variable/horizonte activas, opcionalmente equilibrando grupos. No se declara equivalente a `per_target`. Con soporte muy desigual, una celda con una etiqueta no debería dominar por accidente; los mínimos de soporte o pesos quedan definidos con entrenamiento/desarrollo y documentados antes de prueba.

La ausencia de etiquetas no congela la representación común ni garantiza que la predicción de esa cabeza sea estable: los encoders pueden aprender de otras cabezas. `grad=None` y `zero_grad(set_to_none=True)` preservan el salto de parámetros exclusivos de una cabeza totalmente ausente en AdamW; filas de un objetivo ausente dentro de una cabeza activa pueden cambiar por decay/momentum. La explicación y pruebas están en `docs/aumento-enmascaramiento-atencion.md`.

## Comparación pequeña de arquitecturas, no una rejilla masiva

La selección final debe guiarse por la cobertura, longitud efectiva y diversidad de los datos, con estas comparaciones escalonadas:

1. Persistencia y regresión cuantílica regularizada sobre resúmenes causales; MLP cuantílica pequeña como referencia no recurrente opcional. Establecen qué aporta modelar dinámica.
2. GRU pequeña con máscaras/edades, frente al Transformer causal actual, usando los mismos inputs, objetivos, particiones y criterio de selección. No cambiar simultáneamente longitud y número de parámetros sin controlar el presupuesto.
3. Solo si existe un candidato temporal defendible y suficiente coobservación: fusión con atención entre grupos frente a concatenación/proyección. Las cabezas internas de atención son distintas de los grupos de salida.
4. Solo con longitud efectiva suficiente o irregularidad relevante: sesgo relativo basado en tiempo real; posteriormente atención local definida por horas frente a atención completa. Con 24 posiciones el coste no justifica por sí solo atención dispersa. Con 96 posiciones de datos reales se reevalúa la hipótesis, sin asumir mejora.

Las alternativas de contexto 6/12/24/48h se auditan primero por disponibilidad de muestras y sujetos; se compara después una o dos alternativas justificadas, no el producto cartesiano de resoluciones, modelos, capas, pérdidas y aumentos. Las pruebas de causalidad, padding e independencia del conjunto de horizontes siguen siendo obligatorias en cualquier futura variante. No se ha ejecutado ninguna de estas comparaciones horarias.

## Preentrenamiento externo y adaptación espacial

Separar tres funciones:

1. NHANES puede servir para preentrenar encoders de analitos mediante reconstrucción de valores originalmente observados. Sus sujetos transversales no forman una trayectoria: no se les asignan horas, retardos u objetivos temporales ficticios.
2. Una fuente clínica con seguimiento horario real puede preentrenar dinámica y cabezas compatibles de vitales/laboratorios. Particionar por persona y episodio, no por filas; comparar inicio desde cero frente a transferencia. La evidencia pertenece a esa población, con sus mecanismos de medición y ausencia.
3. Adaptación espacial exige variables armonizadas y datos espaciales con las etiquetas temporales que se pretenden evaluar. Las visitas espaciales esporádicas permiten estudiar adaptación de representación u otras tareas de visita, pero no validar las cuatro predicciones horarias de vuelo.

La ganancia de preentrenamiento se mide en un dominio objetivo reservado, con las mismas particiones y baseline. Se contempla transferencia negativa; no se seleccionará una fuente por mejorar solamente el entrenamiento externo. Los conjuntos existentes y sus restricciones están inventariados en `docs/ampliacion-y-siguiente-experimento.md`.

## Evaluación y gráficas antes del dashboard

Guardar predicciones fuera de muestra con persona/episodio, estudio/dominio, grupo, variable, unidad, origen, horizonte, ventana objetivo, valor real, máscara/calidad, q10/q50/q90 y versiones. Mantener por separado filas sin etiqueta para inspección de disponibilidad; no entran como aciertos ni errores cero.

Para cada cabeza, variable, horizonte, conjunto/dominio y persona cuando haya soporte, reportar MAE de q50, pinball por cuantil, cobertura q10–q90, anchura y frecuencias empíricas bajo cada cuantil. Acompañar **número de etiquetas, orígenes únicos, personas y episodios**, y ausencia por objetivo. La cobertura nominal no es una cobertura conjunta de toda la trayectoria. Los promedios globales deben indicar su ponderación para no ocultar objetivos dispersos.

Gráficas propuestas a partir de esas predicciones persistidas:

- Error y pinball frente a horizonte, una curva por cabeza/modelo con recuentos reales.
- Heatmap variable × horizonte para error, cobertura y número de etiquetas, marcando celdas sin soporte.
- Cobertura y anchura por horizonte junto a la referencia nominal del 80%, sin llamar calibrado a un modelo por una media global.
- Frecuencia empírica frente a 0.1/0.5/0.9 por horizonte y objetivos con suficiente soporte; estos tres puntos no constituyen una distribución completa.
- Trazas temporales de una persona con observaciones y pronósticos emitidos desde orígenes identificables, sin unir bandas de distintos horizontes como si fueran una única trayectoria conjunta.

Las particiones, la fecha de existencia del modelo y la selección del checkpoint deben respetar cada origen de emisión. Calibración, selección de arquitectura y prueba usan datos separados según soporte; no transformar la prueba en un tablero de selección repetida. La incertidumbre de métricas debe respetar agrupación por persona y dependencia temporal cuando el tamaño lo permita.

El dashboard queda como fase final, cuando existan tablas de predicciones, métricas y gráficos verificadas. Será una vista de resultados y calidad por persona/cabeza/horizonte; no calculará resultados de prueba de forma distinta ni rellenará los huecos de etiquetas para aparentar continuidad.
