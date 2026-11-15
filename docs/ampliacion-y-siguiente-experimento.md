# Ampliación de datos y siguiente experimento

Actualización del 14 de noviembre de 2026. Se ha priorizado adquirir y auditar datos, sin entrenar modelos nuevos ni ejecutar un barrido de atención. Los resultados anteriores de Inspiration4 no cambian por haber descargado nuevas fuentes.

## Datos obtenidos, no solo enlaces

| Fuente | Cantidad verificada localmente | Utilidad y limitación |
|---|---|---|
| NASA LSDA UTMB Campaign 3 | 31 identificadores, 118 visitas, 9.534 filas; 29 sujetos con visitas repetidas | Primera prioridad: 23 analitos comparables con CBC/CMP actuales. Es reposo en cama, no vuelo. |
| NASA LSDA UTMB Campaign 1, laboratorio | 3 sujetos, 12 visitas | Otra campaña clínica, conservada en su formato original; falta normalización completa. No se presupone identidad global entre campañas. |
| UTMB C1, actigrafía | Los mismos 3 sujetos, 207 días-persona; 190 valores por variable de sueño | Permite relacionar modalidades por identidad/fecha. No añade otras tres personas ni hace diarios los análisis de sangre. |
| NASA OSD-191, Mars500 | 6 participantes, 159 muestras, 24–28 visitas por persona | Microbioma longitudinal: 2.916 ASV y 12 categorías de phylum, requiere nueva modalidad composicional. |
| NASA OSD-174, ISS | 10 astronautas, 60 tablas procesadas, 6 ventanas por persona | Transcriptómica de cabello, no nuevas etiquetas de CBC/CMP; tiempos publicados como ventanas. |
| PhysioNet Challenge 2019, subconjunto | 1.000 secuencias de dos hospitales, 38.153 filas horarias | Datos vitales y laboratorio disperso para preentrenamiento temporal; población UCI. Son secuencias, no una nueva cohorte de astronautas. |
| PBCseq | 312 personas, 1.945 visitas; 285 personas con visitas repetidas | Trayectorias hepáticas y otros analitos; enfermedad crónica y unidades históricas que requieren revisión. |
| NHANES 2017–2018, adultos | 5.274 personas, 29 variables candidatas de laboratorio | Preentrenamiento transversal por reconstrucción enmascarada, sin aprendizaje de dinámica temporal. |
| BIDMC | 53 registros, 46 identificadores originales de paciente, 25.493 filas | HR, pulso, respiración y SpO2; no CBC/CMP. Particionar por paciente original, no por registro. |

También se descargaron procesados de OSD-53 y OSD-198; OSD-140 está incompleto y comparte la cohorte Mars500. Ax-1 solo tiene metadatos accesibles en la revisión: no se presenta como una descarga de resultados. No se suman las personas de distintas fuentes suponiendo identidades globales no verificadas.

Fuentes primarias: [UTMB C3 en NASA LSDA](https://nlsp.nasa.gov/explore/data/details/a36f6361-3492-5590-a675-a0257edf69ac?dataType=DATASETS), [UTMB C1](https://nlsp.nasa.gov/explore/data/details/042df6cd-2a1a-59f7-903f-c32ab3b7c9de?dataType=DATASETS), [actigrafía C1](https://nlsp.nasa.gov/explore/data/details/e26501eb-7d7a-5012-8c8e-c5bdb3eecaf6?dataType=DATASETS), [Mars500 OSD-191](https://osdr.nasa.gov/bio/repo/data/studies/OSD-191), [ISS OSD-174](https://osdr.nasa.gov/bio/repo/data/studies/OSD-174), [PhysioNet 2019](https://physionet.org/content/challenge-2019/1.0.0/), [PBCseq mantenido por sus autores](https://therneau.r-universe.dev/survival/data/pbcseq/csv), [NHANES bioquímica](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.htm), [BIDMC](https://physionet.org/content/bidmc/1.0.0/).

## Qué está preparado para analizar

UTMB C3 está normalizado en `data/processed/utmb_c3/`, con observaciones completas y tabla común candidata de 23 analitos. Se resuelve el encabezado `Parameter Name`/`ParameterName` antes de combinar archivos y se separan matriz/grupo, analito y unidad. Glucosa de orina, resultados i-STAT y albúmina SPE no se fusionan automáticamente con el panel sérico convencional.

Se usan las fechas de muestra para diferencias temporales; PRE_TEST tiene BR_Day positivo como magnitud y no puede convertirse ingenuamente en un eje global. Se conservan fase, día original, informe de laboratorio, notas y unidad. Hay 7.824 valores puntuales elegibles: 7.114 numéricos sin sufijo y 710 con banderas de rango H/L/CH/CL preservadas. No se eliminan valores por ser altos/bajos. Otros 131 números con notas alfabéticas locales se extraen pero permanecen excluidos de la pérdida puntual hasta revisar sus notas; desigualdades y rangos tampoco se convierten en etiquetas exactas.

El mapeo biomédico tiene 23 reglas confirmadas semánticamente y 18 provisionales/excluidas. Confirmado no significa ensayo idéntico ni ausencia de cambio de dominio: los metadatos se conservan y la incorporación al entrenamiento sigue deshabilitada. La tabla común permite entre 26 y 29 sujetos con al menos dos valores por analito, salvo WBC/potasio/proteínas con 28: véase resumen generado, sin fijar un número universal de ventanas útiles.

Mars500 está preparado en `data/processed/mars500/`: muestras enlazadas exactamente con ISA, recuentos ASV, taxonomía, recuentos por phylum y abundancias relativas. Se conservan 150 fechas relativas exactas; nueve visitas expresadas en meses no reciben días inventados. Los ceros de secuenciación no son una máscara de ausencia de ensayo. No se aplican logaritmos, pseudocuentas o filtrado supervisado antes de definir particiones.

El sueño C1 está preparado con duraciones convertidas desde HH:mm[:ss] a horas. Se conserva la tabla completa de 207 filas y no se añade su copia parcial Query_Result. Hay 17 días sin valores por variable: siguen ausentes. Los otros dominios se conservan como fuentes separadas, con resúmenes y manifiestos, sin concatenarlos al entrenamiento de astronautas.

## Enmascaramiento: dos problemas diferentes

**Datos/etiquetas ausentes.** La pérdida ahora recibe máscaras por muestra, horizonte, variable y cabeza. Puntúa solo objetivos observados y reparte el peso entre cabezas activas (`equal_head`) o entre posiciones observadas (`per_target`). Una cabeza sin etiquetas no recibe gradiente directo ni entra en el denominador; si todo el lote está vacío se omite la actualización. `zero_grad(set_to_none=True)` evita que AdamW actualice una cabeza totalmente desconectada por estados previos.

La representación común sí aprende desde las otras cabezas: por ello las predicciones de una cabeza inactiva pueden cambiar aunque sus parámetros exclusivos queden congelados. Dentro de una cabeza parcialmente etiquetada, filas sin gradiente local pueden cambiar por decay/momentum del tensor compartido. No se promete aislamiento total por variable; para exigirlo haría falta separar parámetros o controlar el estado del optimizador. Estos casos tienen pruebas específicas.

**Cobertura predictiva inferior al 80 %.** El 68 % anterior se refiere a la frecuencia con que los valores futuros cayeron entre q10 y q90. Es un problema de calibración/evaluación distinto de la completitud del dataset. Enmascarar etiquetas ausentes evita errores de entrenamiento, pero no recalibra esas bandas. Tampoco crea asociaciones entre modalidades que nunca se observaron juntas ni elimina ausencia informativa.

## Aumento de datos recomendado

Primero, aprovechar más personas y trayectorias reales. Después comparar, solo dentro de entrenamiento, ocultación artificial de entradas/modalidades y recortes de contexto pasado. Para reconstrucción enmascarada, la etiqueta debe ser un dato originalmente observado: ausencia real y corrupción artificial llevan máscaras distintas. Separar sujetos antes de escalar, ocultar o generar ejemplos.

El ruido solo se justifica con incertidumbre analítica documentada. No interpolar salidas, permutar visitas ni generar pseudo-personas a partir de cuatro sujetos para aumentar la cohorte declarada. TimeGAN, difusión o mixup son posibles líneas posteriores, sin evidencia actual de beneficio en nuestros cuantiles. La [revisión técnica de aumento y referencias primarias](aumento-enmascaramiento-atencion.md) concreta opciones y límites.

## Atención actual y comparación posterior

El modelo usa atención estándar de producto escalar escalado multica­beza: autoatención temporal causal sobre todo el historial permitido y atención cruzada desde cada horizonte al historial. En la ejecución inicial hubo dos cabezas internas de atención y 13 cabezas de salida por grupo: son conceptos diferentes. La autoatención entre consultas de horizonte tiene máscara diagonal, por lo que añadir otro horizonte no modifica uno ya solicitado.

Los encoders de categoría son MLP y la fusión entre categorías es concatenación más proyección lineal: todavía no hay atención explícita entre grupos. Tras preparar particiones y baselines, las primeras comparaciones propuestas son fusión con atención entre grupos, sesgo relativo de tiempo real, atención temporal local frente a completa y una GRU pequeña como referencia. No se ha ejecutado ese barrido ni se atribuye mejora a ninguna variante.

## Orden de trabajo propuesto

1. Usar UTMB C3 como primer conjunto ampliado para los 23 objetivos compatibles; definir evaluación por sujeto y subcampaña, latencia de resultados y escala ajustada solo con entrenamiento. Auditar cobertura por objetivo/horizonte y notas antes de incorporarlo al entrenador.
2. Normalizar los laboratorios C1 y enlazarlos por sujeto/fecha con sueño, sin rellenar análisis sanguíneos diarios que no se midieron.
3. Comparar entrenamiento específico frente a preentrenamiento de encoders con NHANES y, por separado, dinámica con datos clínicos longitudinales. Evaluar transferencia en astronautas manteniendo fuentes y sujetos separados. No presentar mejora en UCI como mejora en astronautas.
4. Añadir microbioma con un contrato composicional propio si se decide ampliar las salidas del gemelo; no sustituye etiquetas sanguíneas ausentes.
5. Ejecutar después las ablaciones de atención y aumento con las mismas particiones. Calibrar con datos distintos de prueba y reportar resultados por dominio, objetivo y horizonte.

Descargas, hashes, restricciones, licencias registradas y comandos: [inventario espacial](ampliacion-datos-espaciales.md), [fuentes y transferencia biomédica](ampliacion-datos-transferencia.md), [NHANES](nhanes-preentrenamiento.md). Las máscaras reforzadas están integradas en el entrenador, pero los nuevos conjuntos todavía no han producido métricas de entrenamiento.
