# Protocolo biomédico para pronósticos a 1, 4, 8 y 24 horas

14 de noviembre de 2026. **Plan y auditoría de viabilidad, sin entrenamiento.** Se conservan todas las variables dinámicas y todos sus valores válidos, con máscaras independientes de entrada y etiqueta. La configuración operativa está en `config/hourly_task_candidates.json`; los recuentos proceden de `outputs/feasibility/target_availability.csv` y su desglose por hospital.

## Tarea concreta y fuentes que pueden sostenerla

Para una secuencia y un origen horario `t`, se solicitan q10/q50/q90 del valor de cada variable en el **intervalo terminal que termina en `t+h`**, con `h ∈ {1,4,8,24}`: esta semántica está confirmada por el usuario. En la rejilla horaria corresponde a `(t+h−1 h,t+h]`; con observaciones reales a 15 min, correspondería a `(t+h−15 min,t+h]`. No es el promedio acumulado de `(t,t+h]`.

En UCI se usa el valor publicado asociado a la fila `t+h`, sin afirmar que sea una media real, mínimo o máximo de esa hora si la fuente no documenta tal resumen. Cada etiqueta debe existir realmente en el intervalo terminal y conservar su ensayo/definición. Los laboratorios solo aportan sus resultados observados; no se busca una visita próxima fuera del intervalo ni se interpola entre resultados para inventar la etiqueta.

| Fuente ya disponible | Utilidad para este objetivo | Límite temporal |
|---|---|---|
| PhysioNet Challenge 2019, 1.000 secuencias UCI | Primera fuente con rejilla horaria y 34 variables dinámicas para auditar los cuatro horizontes | Muchas celdas de laboratorio ausentes; una celda horaria no demuestra una nueva toma de muestra. Dominio terrestre UCI |
| UTMB C3, 31 sujetos y 118 visitas-persona | Transferencia de encoders mediante 23 analitos clínicos comparables | Visitas con fecha, sin historial continuo de 24 h ni etiquetas identificables a 1/4/8/24 h |
| Inspiration4 y PBCseq | Estado basal, relaciones entre analitos y tareas longitudinales de visitas | Muestreo de días, semanas o meses; no se densifican laboratorios a horas |
| BIDMC, 53 registros de 46 IDs originales | Componentes cardiorrespiratorios numéricos y robustez a ausencia de señal | Cada registro dura ocho minutos: ninguno contiene un desenlace a una hora. No se unen registros separados para fabricar 24 h |
| NHANES y Mars500 | NHANES: representación transversal; Mars500: encoder microbiano longitudinal propio | No añaden etiquetas horarias clínicas. Meses de microbioma no se transforman en días exactos ni en horas |

El nombre de un grupo UTMB, “Chemistry profile for flight”, no cambia que su procedencia es bedrest terrestre. Entrenar y evaluar el contrato horario en UCI no valida un gemelo astronauta de 24 h.

## Todas las variables, diferenciando fenómeno y ensayo

Se mantienen como candidatas las 34 columnas dinámicas siguientes, sin seleccionar únicamente las más fáciles:

| Grupo candidato | Variables de la fuente |
|---|---|
| Signos cardiovasculares, nuevo | HR, SBP, MAP, DBP |
| Respiración y oxigenación, nuevo | O2Sat, Resp, EtCO2, FiO2, PaCO2, SaO2 |
| Temperatura, nuevo | Temp |
| Ácido-base | BaseExcess, HCO3, pH |
| Hepatobiliar | AST, Alkalinephos, Bilirubin_direct, Bilirubin_total |
| Renal/metabolismo energético | BUN, Creatinine, Glucose |
| Electrolitos/minerales | Calcium, Chloride, Magnesium, Phosphate, Potassium |
| Metabolito lactato, nuevo | Lactate |
| Laboratorio cardiaco, nuevo | TroponinI |
| Eritrocitos | Hct, Hgb |
| Hemostasia | PTT, Fibrinogen |
| Leucocitos | WBC |
| Plaquetas | Platelets |

Son grupos funcionales para organizar entradas y cabezas, no diagnósticos. Se conserva `SEPSIS2019::<columna>` como identidad del ensayo/dominio; una correspondencia molecular no elimina esa procedencia.

**Solapamientos:** los 23 destinos UTMB confirmados comprenden 14 variables CMP —sodio, potasio, cloruro, CO2 total, calcio total, BUN, creatinina, glucosa, ALT, AST, ALP, bilirrubina total, albúmina y proteína total— y nueve CBC —WBC, RBC, Hb, Hct, MCV, MCH, MCHC, RDW y plaquetas—. No son 23 variables horarias adicionales. El conjunto UCI comparte directamente solo 13 de esos destinos: nueve químicos y cuatro CBC, según el mapeo explícito del JSON. Le faltan, entre otros, sodio, ALT, albúmina/proteína total, RBC e índices eritrocitarios. No se rellenan esas variables ausentes como si se hubieran medido.

Las unidades originales constan por variable en el JSON y en el [diccionario primario de PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/). HCO3 no se renombra como CO2 total; calcio ionizado no es calcio total; O2Sat, SaO2, FiO2 y PaCO2 representan magnitudes diferentes. Fibrinógeno clínico y multiplex Eve requieren armonización de ensayo además de una conversión dimensional.

Hay cuestiones concretas de calidad que resolver antes de entrenar: FiO2 se documenta como porcentaje, pero 3.154 de 3.155 valores son ≤1 y aparece un 4.000 en `training_setB/p107096.psv`, ICULOS 2. Ese último valor no es una fracción ni un porcentaje físicamente válido y debe conservarse en raw con marca de calidad, no recortarse silenciosamente. La unidad del resto requiere verificación; tampoco se corrigen las unidades documentadas de Lactate o Magnesium solo porque sus valores parezcan pertenecer a otra escala. Los valores válidos extremos no se eliminan por ser difíciles de predecir.

## Entradas regulares y etiquetas dispersas son cosas distintas

Una rejilla facilita al encoder combinar información, pero no crea observaciones. Para la fuente UCI, el paso inicial es una hora. Cada celda conserva valor original, máscara observada/válida, calidad y edad desde la última observación disponible. Si se usa arrastre causal o imputación aprendida como representación de entrada, se identifica como estimado: no pasa a ser una medición real y no se convierte en etiqueta futura.

La edad de un resultado de laboratorio y su hora de toma no equivalen a su disponibilidad. El archivo no permite reconstruir todos los tiempos de entrega analítica: el experimento retrospectivo asumirá explícitamente que la fila t está disponible al emitir en t. Un análisis de sensibilidad a latencias es necesario antes de hablar de funcionamiento prospectivo.

La máscara de salida depende de **sujeto/secuencia, origen, variable y horizonte**. Un ejemplo puede aportar glucosa a 4 h y no creatinina a 4 h, o tener etiqueta a 1 h y ninguna a 24 h. Se aprovechan las etiquetas presentes sin exigir que toda la lista esté completa. Ausencia, censura, calidad inválida y padding tienen máscaras/estados distintos; un valor cero físicamente válido sigue siendo un valor observado. Un lote sin etiquetas supervisadas no produce una actualización artificial de pérdida cero.

La decisión de medir puede depender del estado del paciente. Una pérdida enmascarada aprovecha los resultados reales, pero no demuestra que los resultados no medidos sigan la misma distribución. La cobertura cuantílica obtenida se referirá a las etiquetas observadas del conjunto y protocolo, no a un estado fisiológico latente conocido en todas las horas.

## Contexto de 24 horas y agregados de 15 minutos

Para datos verdaderamente densos, 24 h a 15 min serían 96 intervalos. Se calcularían mean/min/max **solo con muestras válidas del intervalo causal cerrado al emitir**, acompañados de recuento observado/válido, cobertura si se conoce la frecuencia y edad de la última muestra. Un intervalo vacío tiene count=0 y máscara ausente; no mean=0 observado. Con una única muestra, mean=min=max no son tres mediciones independientes. La agregación no debe ocultar mala señal ni cambios de dispositivo.

**La fuente UCI descargada ya contiene una fila por hora:** no permite recuperar mean/min/max intrahorarios de 15 min. Repetir una fila cuatro veces generaría cuatro copias del mismo dato. El análisis inicial usa su resolución original. BIDMC sí tiene numerics a 1 Hz, pero sus ocho minutos no sostienen ni siquiera una serie de intervalos de 15 min completa durante 24 h. Para el diseño 96 intervalos hacen falta nuevas sesiones densas y largas con sujetos identificados.

En laboratorio se mantienen eventos, máscara y antigüedad; no se fabrican agregados intrahorarios entre visitas. Si se resumen varias mediciones reales de un mismo analito, debe fijarse el resumen y la procedencia. No mezclar resultados de laboratorio, i-STAT y otros ensayos promediándolos por proximidad temporal.

Se compararán contextos cortos —por ejemplo 1, 4, 8 o 12 h— con un máximo de 24 h, manteniendo los mismos orígenes cuando se estime el efecto del contexto. Se permitirá historial parcial/padding y se informará el rendimiento de esos comienzos por separado. Exigir 24 filas completas descarta buena parte de las secuencias cortas; eso no equivale a que sus observaciones sean inválidas.

La utilidad del histórico depende del analito, de su adquisición y del dominio. Un sensor puede captar variación rápida y artefactos; un laboratorio puede aportar pocas observaciones nuevas aunque la rejilla avance cada hora. Por otro lado, enfermedad aguda, intervenciones y diferencias analíticas pueden producir cambios de laboratorio en horas: no se asigna a priori un horizonte de semanas a todo un grupo. Las visitas largas de UTMB/PBC informan otra escala temporal y no revelan cómo evolucionó el marcador entre ellas. Se revisarán cambios observados, frescura, persistencia y errores por horizonte, en lugar de fijar memoria fisiológica a partir del nombre del órgano.

## Soporte cuantitativo real y elegibilidad

La auditoría encontró 1.000 secuencias, 38.153 filas, sin huecos ni duplicados de ICULOS. Los orígenes con futuro exacto son 37.153, 34.153, 30.153 y 15.690 para 1/4/8/24 h. Con contexto parcial que contiene alguna medición dinámica quedan 36.302, 33.302, 29.308 y 15.076; exigiendo 24 slots de historia quedan 15.690, 13.552, 10.893 y 3.306. Esto cuenta disponibilidad, no acierto ni muestras independientes.

Ejemplos de etiquetas reales con **24 slots históricos**; cada celda indica `pares / secuencias`:

| Variable | 1 h | 4 h | 8 h | 24 h |
|---|---:|---:|---:|---:|
| HR | 14.473 / 726 | 12.487 / 676 | 10.014 / 638 | 3.075 / 237 |
| Glucose | 2.367 / 619 | 2.029 / 574 | 1.611 / 496 | 459 / 124 |
| Creatinine | 874 / 557 | 779 / 515 | 617 / 419 | 187 / 87 |
| WBC | 929 / 554 | 829 / 512 | 656 / 417 | 179 / 79 |
| AST | 181 / 119 | 154 / 104 | 118 / 82 | 42 / 26 |
| Bilirubin_direct | 15 / 8 | 13 / 7 | 12 / 7 | 5 / 2 |
| Fibrinogen | 65 / 36 | 53 / 30 | 46 / 27 | 12 / 4 |

Los 136 pares variable×horizonte están conservados en el JSON, sin filtrar por una cifra arbitraria. Los conteos son de celdas numéricas finitas publicadas, pendientes de calidad/unidades y de partición; no equivalen al número final de etiquetas válidas para entrenamiento o prueba.

Cada variable/horizonte tendrá un estado explícito tras dividir datos: etiquetas disponibles en entrenamiento, desarrollo/calibración y prueba; número de secuencias; calidad/unidades revisadas; o evidencia insuficiente. Una cabeza sin etiquetas de entrenamiento permanece sin entrenar. Si no hay etiquetas de prueba, la métrica es NA, no error cero. Si solo quedan dos o cuatro secuencias, se mantienen los datos y la tarea candidata, pero no se puede afirmar una calibración fiable con tres cuantiles. Estas limitaciones justifican ampliar registros o usar contextos parciales, sin ocultar variables difíciles.

## Evaluación prevista

Se conservarán IDs y hospitales al particionar; se comparará seguimiento de personas/secuencias conocidas y generalización externa cuando el diseño lo permita. Todo escalado y ajuste se realiza en entrenamiento. Las ventanas que comparten futuro a través del corte no pueden contaminar selección o calibración. Un mismo resultado usado en varios orígenes sigue siendo una medición, aunque genere varios términos de pérdida.

Por variable, horizonte, hospital y contexto: número de etiquetas/secuencias, MAE de q50 en unidades originales, pinball de cada cuantil, cobertura de q10–q90 y amplitud, con persistencia como referencia. La incertidumbre de las métricas debe respetar la repetición por secuencia/persona; no tratar cada hora como paciente independiente. Reportar promedios normalizados además de resultados individuales para evitar que los sensores frecuentes o escalas grandes oculten analitos escasos.

q50 es mediana predictiva; q10–q90 es una banda marginal central nominal del 80 %, pendiente de calibración. No es un intervalo de confianza de un parámetro, una garantía conjunta sobre todas las variables ni un umbral clínico. El protocolo no recomienda decisiones médicas y no promete pronósticos horarios de astronautas con datos que no los contienen.
