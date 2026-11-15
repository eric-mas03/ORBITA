# Plan de modelado temporal probabilístico

Fecha: 14 de noviembre de 2026. Estado: propuesta metodológica condicionada a la auditoría de datos. No hay entrenamiento, métricas ni variables disponibles confirmadas en este documento.

## 1. Problema que se quiere resolver

Para cada persona `i`, instante de emisión `t`, variable objetivo `j` y horizonte `h > 0`, estimar:

`q(i,j,t,h,p) = Quantile_p[Y(i,j,t+h) | información realmente disponible en t]`, para `p ∈ {0.10, 0.50, 0.90}`.

La ventana de entrada puede ser `[t-L, t]`, o terminar estrictamente antes de `t` si así se define el producto. La convención debe fijarse antes de crear muestras y mantenerse en entrenamiento e inferencia. La variable objetivo histórica también puede ser una entrada si se midió y estaba disponible.

Los cuantiles son predicciones de valores futuros: q50 es la mediana y `[q10,q90]` es un intervalo predictivo central nominal del 80% para una variable y un horizonte. No es un intervalo de confianza sobre la media o un parámetro, ni asegura que la trayectoria completa o todas las variables caigan dentro con probabilidad del 80%. Esa cobertura requiere calibración y evaluación. [Referencia sobre distribuciones e intervalos predictivos](https://otexts.com/fpp3/toolbox.html).

## 2. Condiciones para poder entrenar

Para cada objetivo deben confirmarse mediciones repetidas, identificador estable de sujeto y misión, marcas temporales interpretables, unidades y suficientes observaciones futuras a cada horizonte. Varias filas técnicas del mismo espécimen no equivalen a medidas longitudinales. Miles de ventanas de una sola persona no equivalen a miles de personas independientes.

Si solo existen datos antes/después de misión o unos pocos puntos de laboratorio, replantear el objetivo como predicción de cambio entre visitas, modelo longitudinal de baja frecuencia o estudio descriptivo. No fabricar una serie densa interpolando etiquetas; tampoco unir personas o estudios sin una correspondencia válida. Datos sintéticos pueden verificar software y mostrar una demo identificada como sintética; no prueban capacidad predictiva en astronautas.

Los horizontes y el histórico se decidirán por variable después de conocer cadencia, duración, autocorrelación, frecuencia de medición y uso esperado. Los ejemplos de minutos, horas o días no constituyen todavía requisitos. Las variables con cadencias incompatibles pueden necesitar tareas y rejillas diferentes.

## 3. Contrato temporal y modalidades

Cada observación debe tener al menos `subject_id`, `mission_id`, `variable`, `value`, `unit`, `event_time`, `available_time`, calidad y procedencia. `event_time` indica cuándo ocurrió o se tomó la muestra; `available_time`, cuándo el sistema podía conocer el resultado. Para una emisión `t`, exigir `available_time <= t`. Un análisis de una muestra antigua publicado después de `t` no era una entrada disponible.

Clasificar las covariables:

- Estáticas: características registradas antes de la emisión, sin introducir metadatos corregidos retrospectivamente.
- Históricas observadas: sensores, resultados y eventos ya disponibles; incluir máscara de ausencia y tiempo desde la última observación.
- Futuras conocidas: calendario o plan disponible en `t`, guardando su versión. El plan no equivale a lo realmente realizado.
- Futuras desconocidas: no introducir sus valores realizados. Si se pronostican, evaluar el sistema completo propagando su error; si se fijan como escenario, etiquetarlo como condicional.

Unir modalidades mediante correspondencia de persona y tiempo verificada. No inferir que un registro ambiental de la estación es una medición personal. Imágenes, ómicas y texto pueden incorporarse como medidas estáticas o por visita con su disponibilidad temporal, si existen; la multimodalidad no obliga a una fusión profunda inicial. La separación entre covariables estáticas, observadas y futuras conocidas también está contemplada por [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363).

## 4. Preparación sin información futura

Definir unidades, duplicados, huso horario y relación entre UTC, tiempo de misión y visitas. Elegir agregaciones justificadas para cada variable; conservar observaciones originales y calidad. No usar filtros centrados, interpolación con puntos futuros, relleno hacia atrás ni estadísticas calculadas con toda la misión para construir una entrada histórica. El arrastre de último valor debe tener edad y límite de validez definidos por variable.

Escaladores, imputadores aprendidos, selección de variables, embeddings ajustados y parámetros de detección de valores anómalos se estiman con entrenamiento. La normalización personal usa únicamente el pasado accesible de esa persona. Las etiquetas ausentes se enmascaran; las puntuaciones deben calcularse sobre objetivos realmente observados, documentando su patrón de ausencia.

Crear ventanas dentro de cada persona y misión; nunca atravesar cambios de identidad o discontinuidades no justificadas. Para cada ventana registrar el intervalo de información de entrada, emisión, instante objetivo y disponibilidad de la etiqueta. La etiqueta usada al entrenar un modelo debe haber estado disponible en la fecha simulada de entrenamiento.

## 5. Validación: dos preguntas diferentes

**Continuar siguiendo a una persona conocida.** Backtesting con origen móvil, entrenando con el pasado disponible y evaluando emisiones posteriores. Reservar bloques cronológicos para desarrollo, calibración y prueba final. [La evaluación temporal de origen móvil](https://otexts.com/fpp3/toolbox.html) permite comparar errores por horizonte respetando el orden de observación.

**Predecir para una persona o misión nueva.** Separación externa por sujeto y, cuando sea posible, misión. Hiperparámetros y calibración se deciden dentro de los grupos de desarrollo. Reportar por separado inicio sin histórico personal y personalización tras adquirir una ventana inicial; esos datos iniciales solo pueden emplearse en el instante en que ya eran conocidos.

Si una misión comparte exposiciones o condiciones entre sujetos, una división únicamente por persona puede evaluar generalización dentro de esa misión; no demuestra transferencia a otra. La selección final de particiones debe reflejar el uso pretendido.

No dividir ventanas al azar. Purgar muestras de entrenamiento cuyas etiquetas o información se sitúen más allá del corte temporal. El intervalo de separación necesario depende del horizonte máximo, latencia de etiquetas y construcción de muestras: no basta un porcentaje fijo. El pasado previo al corte puede formar parte del contexto de una predicción de prueba si estará disponible en el despliegue; el problema es contaminar ajustes con resultados futuros. Ventanas de prueba solapadas requieren incertidumbre estadística que respete esa dependencia.

Con muy pocos sujetos, usar validación dejando sujetos fuera cuando sea viable y mostrar resultados individuales. Los intervalos de incertidumbre de las métricas se estiman por sujetos/misiones o bloques temporales, según el diseño; no con bootstrap de filas que suponga independencia. Si no quedan grupos suficientes para evaluar, declarar el trabajo exploratorio.

## 6. Escalera de modelos propuesta

1. Persistencia, mediana histórica y persistencia estacional solo si existe un ciclo confirmado. Construir bandas a partir de errores pasados fuera de muestra por horizonte, sin usar errores de prueba para ajustarlas.
2. Regresión cuantílica regularizada directa por objetivo y horizonte, con retardos, tendencias y resúmenes causales. Considerar modelos de efectos parciales o de estado si la estructura y el tamaño lo justifican.
3. Árboles con pérdida cuantílica sobre las mismas ventanas y particiones. Es una opción de complejidad moderada para relaciones no lineales, pendiente de evaluación.
4. Solo después: modelos de secuencia multiobjetivo, por ejemplo TCN, GRU o TFT. TFT es un candidato arquitectónico para combinar tipos de covariables y producir cuantiles a varios horizontes; no hay evidencia aún de que sea adecuado para estos datos.

La estrategia inicial propuesta es directa: producir cada horizonte sin utilizar valores futuros reales. Una salida conjunta puede compartir representación entre objetivos, pero tres cuantiles marginales por objetivo no especifican una distribución conjunta ni permiten generar trayectorias fisiológicamente coherentes por sí solos.

Promover un modelo avanzado solo si mejora de forma estable el baseline en validación externa y por horizonte, mantiene calibración, tiene datos suficientes según curvas de aprendizaje y cumple latencia y trazabilidad. No fijar un umbral universal de filas: cuentan sujetos independientes, duración, objetivos medidos y complejidad. Preentrenamiento externo requiere revisar dominio, unidades y posible contaminación de prueba.

## 7. Entrenamiento y evaluación

Optimizar pérdida pinball por objetivo, horizonte y cuantil, ponderando de forma explícita para evitar que una variable con unidades grandes domine. Reportar las métricas también en sus unidades originales. La selección de pesos debe hacerse con desarrollo y relevancia del uso, antes de abrir prueba.

Evaluar:

- MAE de q50 y mejora respecto a persistencia, por objetivo y horizonte.
- Pinball para q10, q50 y q90; no calcular CRPS exacto a partir de solo tres cuantiles.
- Frecuencias `P(Y <= q10)`, `P(Y <= q50)` y `P(Y <= q90)`, con tratamiento explícito de empates y variables discretas.
- Cobertura y anchura de `[q10,q90]`, y puntuación de intervalo que penalice amplitud y fallos.
- Resultados por sujeto, misión, fase, calidad del dato y longitud de histórico, cuando el tamaño lo permita.
- Frecuencia de cruces de cuantiles, abstenciones y violaciones de restricciones físicas.

El número de muestras y sujetos acompaña cada métrica. La cobertura agregada puede ocultar fallos en una persona. Comparar precisión y anchura a igual cobertura; ensanchar indiscriminadamente una banda no implica un buen predictor.

## 8. Calibración y consistencia

Reservar un bloque de calibración posterior al entrenamiento y anterior a prueba. Ajustar por objetivo y horizonte cuando haya soporte; si se agrupan, explicitar qué cobertura se evalúa. Imponer salidas ordenadas o un posprocesado validado para `q10 <= q50 <= q90`. Transformaciones para soportes no negativos o acotados deben definirse por variable; evaluar la cobertura tras cualquier transformación.

La calibración conformal es un candidato, no una garantía automática: la dependencia temporal viola la intercambiabilidad clásica y el cambio entre sujetos o misiones complica las garantías. Métodos adaptativos requieren evaluación secuencial y declarar sus supuestos. [Barber y Pananjady](https://proceedings.mlr.press/v313/barber26a.html) estudian el efecto de dependencia temporal sobre la cobertura; [Zaffran et al.](https://proceedings.mlr.press/v162/zaffran22a.html) estudian métodos adaptativos para series. Recalibrar únicamente con resultados ya observados y disponibles, considerando latencias de etiquetas.

Para alertas conjuntas, límites acumulados o simulación de trayectorias, diseñar otra evaluación y posiblemente otra salida probabilística: la banda marginal del 80% no cubre automáticamente esos usos.

## 9. Contrato de predicción y alcance del gemelo

Salida propuesta por fila: `subject_id`, `mission_id`, `issued_at`, `target_time`, `horizon`, `target`, `unit`, `q10`, `q50`, `q90`, `model_version`, `data_cutoff`, `quality_flags`, `prediction_status`, identificador de escenario y versión de calibración. Conservar la referencia a las entradas usadas para reproducir la predicción.

Abstenerse cuando falte el contexto mínimo, las entradas estén fuera del dominio evaluado o su calidad no permita estimar una banda útil; justificar los criterios con desarrollo. El gemelo inicial sería una representación longitudinal actualizada con pronóstico probabilístico. Afirmar el efecto de una intervención o recomendar un tratamiento exige evidencia adicional: una asociación aprendida no valida simulaciones causales ni decisiones clínicas.

## 10. Próximo experimento, condicionado a datos

Seleccionar uno o dos objetivos medidos longitudinalmente; fijar contrato temporal, particiones y horizontes; generar informe de ventanas utilizables; ejecutar persistencia y regresión cuantílica; evaluar errores y cobertura en backtesting. Solo entonces decidir si el forecasting es defendible y qué ampliación de modalidades o arquitectura merece implementarse.
