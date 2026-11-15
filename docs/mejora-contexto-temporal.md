# Mejorar el aprovechamiento del contexto temporal

> Aclaración posterior: el Transformer actual ya recibe y conserva N latentes, uno por rejilla. El latente 16 es por instante, no por ventana completa. Las rutas de resúmenes/estado/residuales propuestas son auxiliares y mantienen esa secuencia. El plan vigente de comparaciones y ensemble final está en [plan-comparaciones-secuencia-y-ensemble.md](plan-comparaciones-secuencia-y-ensemble.md).

Propuesta del 14 de noviembre de 2026, posterior a la primera comparación. No se han entrenado las variantes de este documento. Las mejoras son hipótesis que se decidirán en desarrollo; la prueba publicada queda congelada como referencia exploratoria.

## Hallazgos de la implementación actual

1. La referencia lineal ya usa contexto temporal: último valor, edad/presencia, media, mínimo, máximo, desviación, recuento y duración del contexto de hasta 24 h. La comparación existente no enfrenta historial contra ausencia de historial, sino representaciones/modelos distintos del pasado.
2. Las redes reciben valores observados, máscaras y edades por evento. No tienen acceso directo al vector de últimos valores conocidos y resúmenes de cada objetivo como la referencia lineal. Deben aprender a recuperarlos de la secuencia.
3. Los 13 encoders y sus indicadores de presencia se concatenan y proyectan a dimensión 16 por instante. Además, cuando un grupo no tiene ninguna observación nueva en una fila, su embedding se anula; sus edades no llegan directamente a la fusión en esa fila. El historial todavía puede contener su última medición, pero recuperar su estado es una tarea adicional para la memoria. Esto es una propiedad comprobada del código, no una demostración de que explique toda la diferencia de rendimiento.
4. Los modelos actuales están entrenados desde cero, con una semilla y un presupuesto de 20 épocas. Las curvas de entrenamiento y desarrollo no justifican asumir que aumentar épocas, anchura o atención produzca una mejora automáticamente.

## Secuencia propuesta de experimentos

### A. Completar la referencia no lineal

Añadir boosting con pérdida cuantílica y un bosque cuantílico usando primero exactamente las mismas nueve características causales por variable que la lineal. Así se prueba no linealidad sin atribuirle el beneficio de añadir entradas nuevas. Conservar horizontes, máscaras, particiones, etiquetas y criterio de desarrollo.

### B. Enriquecer y comprobar la información temporal

Construir vistas causales de 1/4/8/24 h: último valor realmente disponible, tiempo desde ese valor, número de observaciones, cambios entre mediciones, pendiente cuando existan suficientes puntos y resúmenes de cada escala. Las variables de laboratorio no se densifican: cada resumen registra su soporte y no se calcula una pendiente con una sola medición.

Comparar último valor/edad, resúmenes multiescala e información de otros grupos en pasos separados. La incorporación de otras variables necesita correspondencia de persona/tiempo y disponibilidad causal. Usar los mismos orígenes para medir el efecto de ventana, y reportar aparte los historiales cortos que se aprovechan con máscaras.

### C. Red residual con acceso directo al estado propio

Conservar un predictor base congelado y añadir una rama que aprenda correcciones mediante historial y grupos relacionados. Inicializar la corrección a cero para reproducir el predictor base inicialmente. Ofrecer directamente a las cabezas el último valor conocido, edad y resúmenes, además del contexto latente.

La arquitectura mantiene tres cantidades diferentes: observación nueva, último valor causal conocido y ausencia de información. Un valor conservado como estado no cuenta como nueva observación ni como etiqueta. La ruta directa evita exigir que toda la información propia atraviese una compresión común pequeña.

El origen inicial equivalente al baseline no garantiza que el modelo ajustado mejore fuera de muestra. Se seleccionará la variante residual frente al baseline únicamente con el criterio de desarrollo predefinido. Las salidas conservarán orden y soporte por variable con un diseño explícito, no con correcciones escogidas tras ver prueba.

### D. Ablaciones del latente y del contexto

Con la ruta directa controlada, contrastar dimensión 16/32/64 en una lista pequeña y presupuesto documentado. Comparar 6/12/24 h sobre orígenes comunes y medir resultados con/sin historia propia. Si el historial completo no supera al modelo de resúmenes, estudiar si existe señal temporal adicional aprovechable antes de atribuirlo a un tipo concreto de atención.

Que más contexto contenga más mediciones no garantiza información adicional condicionada al último estado. Variables no registradas, cambios entre poblaciones y resultados de laboratorio escasos pueden limitar cualquier modelo; esta explicación requiere evidencia por tarea, no debe darse por probada de antemano.

### E. Objetivos, ponderación y calibración

Revisar distribuciones y gradientes usando entrenamiento/desarrollo. Probar transformaciones monótonas apropiadas por variable cuando haya asimetría importante; invertirlas antes de comparar errores y cuantiles en unidades originales, manteniendo un criterio común. No cambiar a posteriori el criterio para favorecer una red.

Conservar todas las etiquetas válidas, pero estudiar por separado tareas de soporte mínimo y el efecto de ponderaciones estimadas en entrenamiento. No interpretar una celda con cinco etiquetas como evidencia comparable a una con miles. Mantener visibles los errores, denominadores y estados de soporte.

Una vez elegido el predictor, usar el conjunto reservado de calibración. La calibración puede ajustar cobertura; no demuestra por sí sola mayor precisión de la mediana ni mejor uso del contexto. El dominio espacial requiere evaluación propia y etiquetas con la frecuencia correspondiente.

## Criterio de éxito

Comparar pinball por variable/horizonte y agregado macro, error de q50, cobertura, anchura, soporte físico y estabilidad por episodio. Una mejora agregada con degradación importante de grupos o cobertura exige revisión. Confirmar el candidato final con episodios no usados en esta secuencia de desarrollo; no seguir afinando sobre la prueba ya publicada.
