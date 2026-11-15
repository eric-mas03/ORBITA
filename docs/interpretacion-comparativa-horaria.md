# Interpretación de la comparación horaria

14 de noviembre de 2026. Comparación exploratoria en UCI terrestre; no validación clínica ni de astronautas. Este documento separa el diseño revisado de los resultados congelados. La inspección del conjunto de prueba sirve para describir rendimiento, no para elegir arquitecturas, umbrales físicos o nuevos hiperparámetros.

## Qué se compara realmente

Los métodos neuronales comparten encoders por 13 grupos biomédicos, fusión, codificación temporal, representación de ausencias/edades y cabezas de salida con atención. En `gru_shared_readout` se sustituye únicamente la memoria temporal por una GRU causal de una capa, dropout y LayerNorm. Se conservan los decoders de atención: **no es una GRU pura frente a una arquitectura Transformer íntegramente distinta**.

| Propiedad | GRU con readout compartido | Transformer agrupado |
|---|---|---|
| Dimensión latente | 16 | 16 |
| Memoria | GRU causal, una capa | Encoder Transformer causal, una capa y dos cabezas internas |
| Readout por grupo | Decoder de atención, una capa y dos cabezas internas | Mismo diseño |
| Parámetros totales / de memoria | 69.814 / 1.664 | 71.430 / 3.280 |
| Origen del estado compartido inicial | Misma semilla 42 e inicialización común | Misma semilla 42 e inicialización común |

El resto del presupuesto previsto es común: AdamW, learning rate 0.001, weight decay 0.001, dropout 0.1, clip de norma de gradiente 1, minibatch 256, máximo 20 épocas y parada por cinco épocas sin mejora en desarrollo. Los dos usan entrenamiento desde cero, sin preentrenamiento, aumento artificial ni calibración posterior de intervalos. La pérdida equilibra las celdas variable×horizonte con etiquetas de entrenamiento; no da por observada una celda ausente ni aumenta el peso de una variable solo porque tenga más filas.

Persistencia y regresión cuantílica lineal son referencias con información de la propia variable: la lineal tiene 4.080 parámetros y resúmenes causales de hasta 24 slots. Las redes agrupadas también pueden utilizar otras variables. Por tanto, una diferencia frente a la lineal no aislaría el efecto de la atención: cambian acoplamiento multivariable, representación y capacidad. El contraste entre las dos redes acota mejor la diferencia de memoria, con un único tamaño y una única semilla; no demuestra superioridad universal de una familia.

## Controles de comparabilidad revisados

Se reutilizan el dataset horario filtrado, la partición por archivo/episodio y los orígenes: 600 episodios train, 150 desarrollo, 100 calibración y 150 prueba. La división está estratificada por hospital: ambos hospitales aparecen en entrenamiento y prueba, por lo que no equivale a validar en un hospital nuevo. No se ha establecido que episodios diferentes sean personas únicas. Los modelos se evalúan con ventanas parciales de hasta 24 slots, las mismas 34 variables y los mismos objetivos reales de intervalo terminal a 1/4/8/24 h.

Los historiales terminan en el origen; no atraviesan episodios. Escalas y normalizadores usan filas únicas de entrenamiento. Selección de checkpoint y parada usan desarrollo. Los conjuntos de calibración y prueba no se utilizan para ajustar los modelos de esta comparación. La presencia de una etiqueta se determina por celda; no se exige completar todas las cabezas. El tiempo de disponibilidad del resultado sigue siendo un supuesto retrospectivo de la fila horaria publicada.

La edad anterior a la primera medición del contexto representa tiempo desde el inicio de la ventana, acompañada de una máscara de ausencia: no es la fecha conocida de una medición anterior. Las entradas con relleno numérico no pasan a ser observadas. Las cuatro consultas de horizonte permanecen independientes, aunque comparten la memoria histórica.

Los criterios físicos se fijaron antes de esta comparación y se aplican a los datos, no como recorte posterior de pronósticos. q10≤q50≤q90 describe orden cuantílico; no garantiza que toda la banda respete soporte físico. Se auditan los cuantiles en unidades nativas, conservando los valores predichos originales. La frecuencia de q10 bajo el mínimo o q90 sobre el máximo es una métrica adicional, no un motivo para retocar después las predicciones de prueba.

La auditoría independiente confirmó igualdad de las **305.228 claves** por método entre cada red y las referencias: 117.844 pares en desarrollo, 71.240 en calibración y 116.144 en prueba. Coinciden episodios, sitios, origen/objetivo, horizonte, variable, observación, escala, cabeza, contexto y unidad. No hay comparaciones sobre subconjuntos favorables distintos. Los hashes de dataset y configuración coinciden; el manifiesto registra que ambos checkpoints se congelaron antes de exportar prueba. Todos los cuantiles exportados están ordenados.

## Lectura biomédica y soporte

Las diferencias se deben revisar por variable y horizonte además de por cabeza. Una cabeza renal no equivale a diagnóstico renal; correlaciones de señales y laboratorios no identifican mecanismos causales. Las predicciones de FiO2, Magnesium y Lactate conservan el estado de revisión de unidades; sus números en escala fuente no autorizan interpretación clínica ni conversiones automáticas.

En prueba a 24 h, EtCO2 tiene 29 etiquetas de cuatro episodios; bilirrubina directa, seis de cuatro; troponina I, 15 de ocho; fibrinógeno, 13 de nueve. Estos conteos incluyen contexto parcial y son distintos de los recuentos con 24 slots completos. Todas las variables permanecen visibles. Sus bandas y errores pueden calcularse, pero unas pocas secuencias no sostienen conclusiones firmes de calibración o de superioridad entre modelos.

Las métricas principales son pinball normalizado por escala de entrenamiento, MAE de q50 en unidades originales, cobertura y amplitud de q10–q90, con etiquetas/episodios. El promedio entre variables evita que un sensor frecuente domine al laboratorio; no elimina la incertidumbre de las celdas raras. Una cobertura cercana o superior al 80 % puede acompañarse de bandas excesivas o fuera de soporte. En las referencias existentes, por ejemplo, persistencia presenta violaciones de soporte en todas las bandas de SaO2 a 24 h y troponina I a 24 h; no debe confundirse su cobertura elevada con adecuación biomédica del intervalo.

## Resultados de la ejecución

**Desarrollo, usado para seleccionar checkpoints:** GRU seleccionó época 19 de 20, con 220,20 s de ajuste; Transformer, época 18 de 20, con 229,53 s. El pinball macro normalizado de las 136 celdas fue 0,178455 para GRU y 0,175066 para Transformer. La lineal congelada obtiene 0,156521 y persistencia 0,179530 con el mismo criterio. Por tanto, desarrollo favorece ligeramente la memoria Transformer entre las dos variantes neuronales, pero no demuestra mejora frente a la referencia lineal. No se elige otra configuración tras inspeccionar prueba.

Pinball normalizado en desarrollo, promediado entre las 34 variables en cada horizonte; menor es mejor:

| Método | 1 h | 4 h | 8 h | 24 h |
|---|---:|---:|---:|---:|
| Persistencia | 0,15523 | 0,16857 | 0,18412 | 0,21021 |
| Lineal cuantílica | 0,13803 | 0,14434 | 0,15826 | 0,18545 |
| GRU con readout compartido | 0,17604 | 0,17053 | 0,17434 | 0,19291 |
| Transformer agrupado | 0,17411 | 0,16892 | 0,17255 | 0,18468 |

**Prueba, descripción del resultado congelado:** el pinball macro global es 0,164499 en la lineal, 0,181548 en persistencia, 0,191133 en GRU y 0,188094 en Transformer. Las redes no aportan mejora global en esta ejecución. El Transformer tiene menor pinball que GRU en 81 de las 136 celdas, pero GRU obtiene menor promedio a 24 h: el resultado no es uniforme entre horizontes ni biomarcadores. Son comparaciones descriptivas, sin demostrar significación ni estabilidad entre semillas.

| Método | 1 h | 4 h | 8 h | 24 h |
|---|---:|---:|---:|---:|
| Persistencia | 0,15923 | 0,17498 | 0,18707 | 0,20491 |
| Lineal cuantílica | 0,14375 | 0,15649 | 0,16490 | 0,19287 |
| GRU con readout compartido | 0,18257 | 0,18463 | 0,19305 | 0,20429 |
| Transformer agrupado | 0,17776 | 0,17990 | 0,18747 | 0,20725 |

### Desglose por cabeza, sin esconder grupos difíciles

La siguiente vista de prueba a 4 h presenta las 13 cabezas; las métricas completas conservan cada variable y los cuatro horizontes. El promedio de una cabeza da el mismo peso a sus variables puntuadas. Los pares son etiquetas/orígenes repetidos, no personas nuevas; los episodios son la unión de episodios con alguna etiqueta del grupo.

| Cabeza | Lineal | GRU | Transformer | Pares / episodios |
|---|---:|---:|---:|---:|
| Ácido-base | 0,17924 | 0,20292 | 0,19056 | 820 / 90 |
| Laboratorio cardiaco | 0,02956 | 0,09322 | 0,06499 | 52 / 24 |
| Signos cardiovasculares | 0,16064 | 0,17416 | 0,16822 | 16.773 / 150 |
| Eritrocitos | 0,18467 | 0,18683 | 0,19149 | 825 / 136 |
| Leucocitos | 0,17624 | 0,19741 | 0,17950 | 315 / 135 |
| Plaquetas | 0,13130 | 0,18767 | 0,16283 | 293 / 135 |
| Electrolitos/minerales | 0,17065 | 0,20123 | 0,19429 | 1.504 / 138 |
| Hemostasia | 0,27705 | 0,29447 | 0,29762 | 170 / 64 |
| Hepatobiliar | 0,09850 | 0,12906 | 0,14853 | 262 / 47 |
| Lactato | 0,18248 | 0,21069 | 0,18586 | 103 / 34 |
| Renal/metabolismo | 0,10415 | 0,13803 | 0,14062 | 1.528 / 141 |
| Respiración/oxigenación | 0,16226 | 0,20090 | 0,18874 | 9.600 / 150 |
| Temperatura | 0,16394 | 0,17853 | 0,18081 | 1.736 / 146 |

En esta vista el Transformer mejora a GRU en varios grupos, pero no en eritrocitos, hemostasia, hepatobiliar, renal/metabolismo o temperatura. En desarrollo tampoco hay una ventaja uniforme: por ejemplo, el grupo respiratorio a 4 h obtiene 0,20174 con GRU y 0,20728 con Transformer, mientras en prueba cambia el orden. No se reorganizan grupos ni se seleccionan variables después de observar estos cambios.

### Variables y horizonte largo

Ejemplos de prueba a 24 h, siempre sobre los mismos pares para los tres métodos:

| Variable | Pares / episodios | Lineal | GRU | Transformer |
|---|---:|---:|---:|---:|
| HR | 2.055 / 107 | 0,19439 | 0,20676 | 0,20199 |
| Glucose | 359 / 91 | 0,16387 | 0,16656 | 0,16487 |
| Creatinine | 138 / 85 | 0,09382 | 0,10767 | 0,11892 |
| WBC | 144 / 86 | 0,19233 | 0,21179 | 0,21360 |
| Bilirubin_direct | 6 / 4 | 0,51511 | 0,16443 | 0,19193 |
| Fibrinogen | 13 / 9 | 0,59003 | 0,68819 | 0,78936 |

La aparente mejora de las redes sobre la lineal para bilirrubina directa a 24 h depende de solo seis etiquetas de cuatro episodios; persistencia obtiene 0,10748 en esos mismos pares. No es una base firme para recomendar una cabeza neuronal. En fibrinógeno, ambas redes cubren solo una de las 13 etiquetas —7,69 %—, lejos del nominal del 80 %. Se mantienen las variables, se muestran las deficiencias y se evita convertir una diferencia agregada en una interpretación de salud.

### Cobertura y soporte físico de las bandas originales

Cobertura macro entre variables y horizontes en prueba: persistencia 80,06 %, lineal 78,02 %, GRU 73,76 % y Transformer 73,99 %. Ninguno de estos promedios implica calibración por persona o variable, y la calibración reservada no se utilizó para corregir intervalos. A 24 h la cobertura macro baja a 72,01 % en GRU y 71,19 % en Transformer.

Auditoría de soporte sobre las 116.144 filas de etiqueta/pronóstico de cada método, ponderadas por fila —denominador distinto del promedio macro anterior—:

| Método | Banda con algún cuantil fuera de soporte | q50 fuera de soporte |
|---|---:|---:|
| Persistencia | 7.978; 6,869 % | 0 |
| Lineal cuantílica | 3.136; 2,700 % | 17 |
| GRU con readout compartido | 5.254; 4,524 % | 21 |
| Transformer agrupado | 4.041; 3,479 % | 9 |

Las 21 medianas neuronales GRU fuera de soporte corresponden a 20 predicciones de AST y una de bilirrubina directa; las nueve del Transformer corresponden a AST. En AST a 24 h, GRU cubre 27/33 observaciones —81,82 %—, pero 31/33 bandas incluyen cuantiles negativos —93,94 %—. Esto ilustra por qué cobertura cercana al 80 % no basta: una banda puede contener el dato y seguir representando mal una magnitud no negativa.

Se han conservado los cuantiles ordenados originales, sin clipping, proyección física, CQR ni ajuste posterior basado en prueba. Cualquier parametrización futura que imponga soporte sería otro experimento definido y seleccionado en desarrollo; no una corrección silenciosa de estos resultados.

## Alcance de la conclusión

Con este presupuesto, representación y semilla, el desarrollo no demuestra beneficio de sustituir la referencia lineal por las redes; la prueba documenta el mismo límite global y fallos distintos por cabeza. No demuestra que GRU o Transformer sean inadecuados en general ni que más atención, más entrenamiento o preentrenamiento vayan a resolverlo. Permanecen como incertidumbres la variación entre semillas, la escasez de algunos analitos, el sesgo de medición, las unidades pendientes y la transferencia a otros dominios. Las conclusiones se restringen al experimento UCI congelado.

Fuentes locales de resultados: `outputs/hourly_baselines/fit_manifest.json`, `outputs/hourly_neural/fit_manifest.json`, ambos `predictions.csv` y los desgloses de `outputs/hourly_evaluation/`. La auditoría de este documento recalculó pérdidas, coberturas y soporte desde las predicciones, sin reentrenar ni modificar modelos, políticas o configuraciones.
