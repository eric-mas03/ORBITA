# Baselines horarios sobre datos reales

Se implementan dos referencias para el subconjunto descargado de PhysioNet Challenge 2019: persistencia y regresión cuantílica lineal. La tarea usa las 34 variables dinámicas revisadas, horizontes de 1/4/8/24 horas y etiquetas reales de la fila futura exacta. No se entrena un Transformer, no se aumentan datos y no se ejecuta transferencia espacial en este experimento.

La fuente es terrestre UCI. Los archivos identifican episodios, sin demostrar identidades únicas de personas entre ingresos. El dataset empaquetado conserva 600 episodios de entrenamiento, 150 de desarrollo, 100 de calibración y 150 de prueba, estratificados por hospital y fijados antes de entrenar. La disponibilidad inmediata de cada fila al emitir se asume retrospectivamente; las latencias reales de laboratorio no están publicadas.

## Contexto, features y etiquetas

El contexto conserva hasta 24 slots horarios reales `[t-23,t]`, incluyendo historias parciales desde el inicio del episodio. Cada regresión utiliza **solo información de su propia variable**: último valor observado, indicador de existencia de historial, edad de ese último valor, media, mínimo, máximo, recuento y desviación estándar de las observaciones del contexto, más duración real del contexto. Son nueve features, incluyendo duración. No son estadísticas de una señal intrahoraria: la fuente ya publica una fila por hora.

Si no existe observación de la variable dentro del contexto, el último valor y los resúmenes usan la mediana de entrenamiento como entrada numérica, acompañada de máscara de ausencia, recuento cero y edad de 24h como centinela. Esto se marca en las predicciones; no se convierte en una observación real. Las demás variables presentes permiten conservar el origen aunque falte una variable concreta.

La media, escala y mediana se ajustan sobre filas únicas de episodios de entrenamiento, sin ponderar una fila varias veces porque aparezca en ventanas solapadas. No se usan filas de desarrollo, calibración ni prueba para ajustarlas. Escalas constantes se sustituyen por 1; una variable sin observaciones de entrenamiento queda señalada como tal, no aprendida. Las etiquetas mantienen NaN si no hubo una celda válida en el momento exacto `t+h`, sin relleno ni interpolación. La calidad física/analítica se aplica en el constructor del dataset, sin recortes de rango clínico normal.

## Persistencia y bandas

La mediana predictiva q50 es el último valor propio observado dentro de contexto, con mediana de entrenamiento si falta. Se ajustan cuantiles 0.1/0.9 de errores `objetivo - último_valor` utilizando solo transiciones de entrenamiento.

Para cada variable y horizonte se usan sus propios errores si hay al menos 20. Con menor soporte se agrupan errores de todos los horizontes de la misma variable, exclusivamente de entrenamiento, conservando un indicador. Si incluso ese conjunto tiene menos de 20 se calcula la banda empírica disponible con marca de bajo soporte. Sin errores se devuelve banda degenerada y estado sin evidencia. Se guardan tanto el recuento original variable/horizonte como el utilizado al ajustar y la procedencia de la banda.

Los residuos inferiores/superiores se limitan respectivamente a ser <=0 y >=0 para conservar la persistencia literal en q50 y bandas ordenadas. No se aplican restricciones fisiológicas a los valores pronosticados. Las bandas están **sin calibrar** y los residuos solapados no son independientes. Ni el umbral 20 ni agrupar horizontes establece una garantía de cobertura.

## Regresión cuantílica lineal

Cada variable/horizonte/cuantil tiene coeficientes lineales e intercepto propios sobre esas nueve features; no se comparten variables como predictores ni se usa una red temporal. Con 34 variables, cuatro horizontes y tres cuantiles son 4.080 parámetros, incluidos interceptos.

Se optimiza pinball sobre los objetivos estandarizados con la escala de entrenamiento. Primero se promedian muestras y cuantiles dentro de cada celda variable × horizonte y luego se promedian celdas no vacías. Así, vitales más densos no dominan automáticamente los laboratorios dispersos. Se añade una penalización `0.001 * mean(pesos²)`. Una celda sin etiquetas no aporta gradiente supervisado; su predicción final usa persistencia con estado explícito de ausencia de ajuste.

Se comparan únicamente dos tasas de aprendizaje de Adam, 0.01 y 0.03, hasta 80 épocas cada una y parada tras 12 épocas sin mejora. El checkpoint y la tasa se eligen mediante pinball estandarizada macro por variable × horizonte en desarrollo. Los cuantiles lineales se ordenan determinísticamente al predecir y en esa selección: la restricción no demuestra calibración. El optimizador es Adam, no un algoritmo de optimización convexa exacta; se informa de la duración y curvas, sin afirmar convergencia al óptimo global por completar 80 épocas.

El ajuste y su configuración seleccionada se guardan antes de generar predicciones de prueba. La prueba no se usa para selección. El split de calibración conserva predicciones independientes para una futura calibración, pero **no se aplica CQR ni otro ajuste de bandas** en esta primera ejecución. El código acepta un límite de épocas entre 1 y 80 y fija dos threads de CPU.

## Artefactos y reproducción

Comando desde la raíz del proyecto: `python scripts/train_hourly_baselines.py --epochs 80`. Produce en `outputs/hourly_baselines/`:

- `fit_manifest.json`: hash de datos/configuración, particiones, hipótesis, candidatos, selección, tiempos y limitaciones.
- `scalers.npz`: media/escala/mediana, recuentos y filas únicas de entrenamiento, orden de variables/features.
- `persistence.npz`: bandas, recuentos originales/utilizados y procedencia por variable/horizonte.
- `linear_quantile.npz`: coeficientes, interceptos, recuentos de etiquetas de entrenamiento y orden de features.
- `learning_curves.csv`: pérdida de entrenamiento y desarrollo de cada candidato y época.
- `label_availability.csv`: las 136 celdas variable × horizonte de cada split, incluso sin etiquetas, con episodios, orígenes y casos sin historia propia.
- `predictions.csv`: predicciones largas de desarrollo/calibración/prueba para ambos métodos en las mismas etiquetas observadas. Incluye episodio, hospital, origen, destino, horizonte, cabeza, variable/unidad, revisión de unidades, observación, cuantiles, contexto, historial propio, fallback, escala y soporte.

Las filas sin etiquetas no reciben error cero ni se añaden como aciertos; su soporte consta por separado en `label_availability.csv`. Las unidades FiO2/Magnesium/Lactate mantienen su declaración de origen y una marca de revisión; no se corrigen o mezclan automáticamente. Las métricas y gráficos finales se generan posteriormente leyendo los artefactos guardados, por cabeza/variable/horizonte, evitando confundir conteos de pares correlacionados con sujetos independientes.

## Verificación

Los tests de `tests/test_hourly_baselines.py` usan secuencias matemáticas pequeñas para comprobar el software, no simular fisiología: aislamiento del escalado respecto a holdouts; causalidad de último valor/edad/recuento; fallback solo con estadísticos de entrenamiento; bandas que no usan holdouts; parámetros y selección invariantes a cambios en etiquetas de calibración/prueba; cuantiles ordenados; fallback con cero etiquetas; y rechazo de objetivos que crucen episodios. No son evidencia clínica ni de rendimiento espacial.

## Ejecución realizada

La ejecución de 80 épocas máximas por candidato tardó 59,29 segundos en CPU con dos threads. Se eligió tasa 0,03, checkpoint de la época 54; ese candidato terminó tras 66 épocas por parada temprana. El criterio macro de desarrollo fue 0,156521 de pinball estandarizada. Este número sirve para seleccionar la referencia, no para afirmar rendimiento espacial ni calibración.

Se guardaron 610.456 filas de predicción para ambos métodos: 117.844 etiquetas de desarrollo, 71.240 de calibración y 116.144 de prueba por método. Son pares de evaluación con orígenes correlacionados, no ese número de personas. Las métricas y gráficos comparativos se generan por separado desde este CSV.

Las predicciones mantienen valores fuera de soportes físicos si el modelo los produce; ordenar cuantiles no impone positividad ni acota porcentajes. Esa tasa de violaciones debe auditarse por variable. No se recortan salidas ni se selecciona una transformación mirando prueba. La posible proyección de soporte sería una variante explícita con selección de desarrollo coherente y quedaría para otro experimento.
