# Ampliación de personas y secuencias para transferencia

Revisión: 14 de noviembre de 2026. Se mantienen dominios separados: vuelo espacial, análogos terrestres, población general, enfermedad crónica y UCI. Las personas terrestres no aumentan el número de astronautas evaluados. No se ha entrenado aquí ningún modelo con las fuentes nuevas.

## Material efectivamente descargado y prioridad

| Prioridad y dominio | Tamaño verificado en archivos locales | Aportación concreta |
|---|---|---|
| 1. UTMB Campaign 3, análogo bedrest NASA | 31 sujetos, 118 visitas-persona por fecha, 9.534 filas de laboratorio; fases PRE_TEST, IN_TEST y POST_TEST | Ampliación más directamente compatible con química clínica y hemograma de Inspiration4: 23 analitos con correspondencia semántica/unidad confirmada |
| 2. Mars500 OSD-191, análogo de aislamiento | Auditoría del equipo: 6 personas, 159 muestras, 24–28 visitas por persona; misión de 520 días y seguimiento | Secuencias espaciales análogas prolongadas para un nuevo encoder microbiano. No añade CBC/CMP; OSD-140 corresponde a esas mismas seis personas y no se suma como otra cohorte |
| 3. PhysioNet Challenge 2019, UCI | 1.000 archivos de secuencia, 500 por cada sitio; 38.153 filas horarias. Longitud 8–336 filas, mediana 39 | Volumen útil para preentrenamiento temporal de partes de química/hemograma, con faltantes reales y separación por hospital |
| 4. PBCseq, ensayo Mayo de enfermedad hepática | 312 personas, 1.945 visitas, mediana 5 visitas/persona; 285 personas tienen al menos dos. Tiempo desde inclusión 0–5.152 días | Bioquímica longitudinal irregular de baja frecuencia: albúmina, bilirrubina, AST, fosfatasa alcalina y plaquetas, pendientes de armonizar unidades históricas |
| 5. NHANES, población general | Preparación independiente del equipo: 5.274 adultos y 29 solapamientos candidatos | Preentrenamiento transversal de relaciones entre analitos; no añade trayectorias individuales ni valida forecasting |
| 6. BIDMC, UCI | 53 registros de ocho minutos, 25.493 filas numéricas a 1 Hz, pero **46 identificadores originales de paciente únicos** | Encoder cardiorrespiratorio futuro: HR, PULSE, RESP y SpO2. No aporta CBC/CMP ni, en esta descarga, señales ECG/PPG crudas |

Los archivos de UTMB, Mars500 y NHANES pertenecen a los pipelines y manifiestos específicos del equipo. La descarga externa de este documento comprende PhysioNet, BIDMC y PBCseq: **1.111 archivos, 6.913.377 bytes**, con URL, SHA-256, procedencia y licencias en `data/manifests/external_downloads.json`. No hay fallos pendientes. Los índices descargados enumeran 20.336 secuencias A y 20.000 B del conjunto PhysioNet; solo se han descargado las 1.000 seleccionadas, no las 40.336 disponibles.

La selección UCI es aleatoria uniforme sin reemplazo, 500 registros por sitio y semilla 2026. No se ha equilibrado según etiquetas futuras. En BIDMC hay registros repetidos del mismo paciente —por ejemplo 06–09 comparten ID original—: el manifiesto contiene el mapa `recording_to_original_subject_id` y la partición debe usar ese ID, no el número de registro.

## UTMB: correspondencias revisadas y límites

`config/utmb_candidate_mapping.json` es una lista con campos `group`, `parameter`, `source_unit`, `target`, `unit`, `factor`, `status`, `auto_harmonize` y `reason`. Se revisaron únicamente combinaciones observadas y se comprobaron sus destinos contra el diccionario canónico existente:

- **23 confirmadas:** 14 químicas —incluida albúmina del grupo Other— y nueve de Hematology coulter. Son correspondencias de analito/unidad, no prueba de equivalencia entre instrumentos o poblaciones.
- **13 provisionales:** i-STAT, proteínas por electroforesis, variantes de CRP y una etiqueta Potassium D con nota local. Se conserva el posible destino, pero `auto_harmonize=false`.
- **Cinco porcentajes diferenciales excluidos por defecto:** son relativos y composicionales; no equivalen a neutrófilos/linfocitos absolutos. No se han inventado recuentos absolutos ni observaciones independientes multiplicando columnas.

`thou/mm3` equivale numéricamente a `Thousand/uL` y `mil/mm3` a `Million/uL`: el factor es 1 porque mm³ y µL son el mismo volumen. CRP mg/L→pg/ml tiene factor dimensional 1.000.000, pero su correspondencia con CRP multiplex Eve sigue siendo provisional por método y rango analítico. No se confunden creatinina/glucosa de orina con suero, calcio ionizado con calcio total, ni ALP ósea con ALP total.

Antes de concatenar archivos se armoniza el nombre de columna `Parameter Name`/`ParameterName`; dentro de etiquetas se reemplaza el texto literal `\\n`, se colapsa whitespace y se recortan extremos. No se eliminan indiscriminadamente sufijos de notas: los ReadMe contienen cambios de instrumento/rangos, incidencias de muestras y metodologías. `Test`, fuente, campaña y método siguen identificando cada observación. No se promedian resultados químicos, SPE e i-STAT de la misma fecha como si fueran réplicas equivalentes.

Las fechas originales permiten orden temporal; `BR_Day` positivo en PRE_TEST no se transforma automáticamente en día negativo. Se conserva fase y día tal como se publicaron. Las campañas son cohortes/condiciones distintas; cama inclinada no reproduce todos los riesgos de un vuelo.

## Qué encoder puede aprovechar cada dominio

| Encoder existente o futuro | Fuente y contenido aprovechable | Límite de la transferencia |
|---|---|---|
| Electrolitos/ácido-base | UTMB: Na/K/Cl/CO2/calcio total. UCI: K, Cl, calcio; bicarbonato/gasometría como variables nuevas explícitas | HCO3, PaCO2 y CO2 total no se renombran como una sola medición |
| Renal/metabolismo energético | UTMB y UCI: BUN, creatinina, glucosa | Función renal, terapias y selección de pacientes cambian entre dominios |
| Hepatobiliar | UTMB: AST/ALT/ALP/bilirrubina total. UCI: AST/ALP/bilirrubinas; PBCseq: AST/ALP/bili | Bilirrubina directa y total son objetivos diferentes; PBC tiene enfermedad hepática establecida |
| Proteínas séricas | UTMB albúmina/proteína total; PBCseq albúmina; NHANES como relaciones transversales | UCI Challenge 2019 carece de albúmina/proteína total: no preentrena ese bloque completo |
| Eritrocitos | UTMB: RBC/Hb/Hct/MCV/MCH/MCHC/RDW. UCI: Hb y Hct | UCI solo cubre parte del bloque; no rellenar RBC e índices como mediciones |
| Leucocitos | UTMB y UCI: WBC total | No se dispone aquí de todos los diferenciales absolutos de Inspiration4; porcentajes conservados aparte |
| Plaquetas | UTMB y UCI: recuento; PBCseq: platelet pendiente de unidad | MPV ausente en estas nuevas tablas: máscara explícita |
| Fase aguda/hemostasia | UTMB CRP candidato; UCI fibrinógeno y pruebas de coagulación | Ensayos clínicos no son automáticamente equivalentes a paneles Eve |
| Quimiocinas/interleucinas/IFN-TNF/factores de crecimiento | Estas descargas terrestres no aportan los paneles multiplex equivalentes | No alegar preentrenamiento específico de esos cuatro encoders a partir de signos vitales o CBC |
| Cardiorrespiratorio, nuevo | BIDMC numerics; UCI HR/Resp/O2Sat/temperatura/presión | BIDMC dura ocho minutos; no respalda predicciones de días o semanas |
| Microbioma, nuevo | Mars500 OSD-191 ASV/abundancias por muestra y tiempo relativo | Abundancias composicionales; cero de muestreo no prueba ausencia biológica. Conservar profundidad, taxonomía y tiempo publicado, sin inventar días exactos a partir de meses |
| Alamar NPQ/ómicas | No hay fuente externa homóloga confirmada en este lote | No sustituir NPQ por química sérica ni inventar emparejamiento molecular |

## Densidad real en la muestra UCI

Contar filas horarias sobreestima las etiquetas de laboratorio. Se han contado también secuencias con al menos **dos valores observados** de cada variable: glucosa 842; potasio 763; BUN 710; creatinina 692; Hb 711; Hct 735; WBC 683; plaquetas 672. AST solo aparece repetida en 141 secuencias, ALP en 142, bilirrubina total en 126 y fibrinógeno en 53. Esta cobertura hace viable estudiar subproblemas concretos; no demuestra que cada persona tenga un panel completo y denso.

Las filas son intervalos horarios con faltantes, no toma de sangre cada hora. Se mantienen tiempo desde última medida, máscara y hospital. `SepsisLabel` se excluye de entradas: la fuente lo define usando información del desenlace y desplaza la marca seis horas. `ICULOS` representa horas transcurridas hasta la fila, no duración final futura de la estancia. La tarea propuesta es pronóstico de analitos observados, no clasificación de sepsis. [Descripción primaria y diccionario PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/).

PBCseq se descargó directamente del [repositorio del mantenedor de survival](https://therneau.r-universe.dev/datasets), no de una copia de competición. Hay 285 personas con bilirrubina/albúmina/AST repetidas y 283 con plaquetas repetidas; no hay duplicados `(id,day)`. `futime` y `status` describen el desenlace final y no se usan como entradas históricas. Su documentación advierte visitas adicionales por empeoramiento y ausencias potencialmente informativas. Las unidades históricas de albúmina/AST y la escala de plaquetas necesitan contraste con el protocolo antes de mezclarse con OSDR; no se corrigen solo porque parezcan fisiológicamente plausibles. [Documentación primaria pbcseq](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbcseq.html).

## Estrategia operativa de preentrenamiento

1. Preparar cada dominio con IDs propios, ensayo, unidades originales y tiempos; dividir personas/campañas/hospitales antes de ajustar normalizadores o construir ventanas. No sumar filas de distintos dominios dentro de una trayectoria.
2. Preentrenar reconstrucción enmascarada exclusivamente sobre valores observados y pronóstico de siguiente medida con tiempo irregular. La ausencia de una etiqueta se enmascara; no se convierte en cero. Mantener adaptadores por dominio/ensayo y comparar con entrenamiento desde cero.
3. Transferir los encoders compatibles y sus máscaras; reinicializar o adaptar cabezas temporales al reloj y variables del dominio destino. El preentrenamiento UCI por horas no justifica reutilizar directamente horizontes o calibración de meses. NHANES solo aporta la parte transversal.
4. Priorizar adaptación y evaluación dentro de UTMB por sujeto/campaña; después analizar transferencia a Inspiration4 como experimento pequeño. Mantener Mars500 como tarea multimodal separada mientras no exista un vínculo de persona/visita con analitos clínicos.
5. Evaluar q10/q50/q90 y calibración en cada dominio por separado; reservar participantes completos y comparar persistencia. Solo el dominio espacial evalúa rendimiento en astronautas, y sus cuatro personas siguen limitando cualquier conclusión.

## Fuentes siguientes y acceso verificado

- **HERA humano:** prioridad científica para sueño, cognición y aislamiento, pero no se verificó una tabla individual abierta para descargar directamente. Una publicación primaria remite la solicitud de datos a LSDA por investigadores cualificados. No se confunde con el detector radiológico HERA ni con archivos astronómicos homónimos. [Estudio HERA y disponibilidad](https://pmc.ncbi.nlm.nih.gov/articles/PMC7198903/).
- **PPG-DaLiA:** 15 participantes, ECG/PPG/movimiento en actividades diarias; útil para un encoder de señal y robustez al movimiento. El repositorio ofrece 2.7 GB y CC BY 4.0. No descargado en esta fase, que prioriza analitos longitudinales. [UCI, fuente de los autores](https://archive.ics.uci.edu/dataset/495/ppg+dalia).
- **WESAD:** 15 participantes con señales de pecho/muñeca, estados experimentales y autoinformes. Útil para encoder fisiológico/conductual, no para crear etiquetas de salud astronauta. La ficha remite las condiciones de uso al sitio original; no descargado ni aceptados términos en nombre del usuario. [UCI WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).
- **MIMIC/eICU completos:** candidatos con laboratorios longitudinales, pero requieren credenciales personales, formación y acuerdo de uso. No se han descargado, solicitado credenciales ni firmado acuerdos. [PhysioNet FAQ](https://physionet.org/about/faqs/).

## Procedencia y reproducción

Ejecutar `python scripts/download_external.py --dataset all --sepsis-per-site 500 --workers 6`; reutiliza archivos existentes y genera manifiesto y conteos. `--dataset pbcseq` actualiza el inventario sin volver a consultar los índices UCI. Las nuevas descargas solo usan URLs públicas sin autenticación. Para aumentar la muestra UCI, se incrementa `--sepsis-per-site`; se debe reservar primero una partición de desarrollo/prueba estable y medir espacio y cobertura antes de ampliar indiscriminadamente.

Se preservan los textos de licencia. Hay una discrepancia de procedencia en PhysioNet Challenge 2019: la página declara CC BY 4.0, mientras el `LICENSE.txt` descargado comienza con ODC Open Database License (ODbL). Ambas indicaciones constan en el manifiesto; no se afirma una licencia única y debe aclararse antes de redistribuir ese material. BIDMC declara ODC-By 1.0 en la página y conserva su archivo de modificaciones/licencia. El fichero DESCRIPTION del paquete survival declara LGPL (>=2); se guarda junto al snapshot PBCseq sin reinterpretar sus condiciones. [Página primaria BIDMC](https://physionet.org/content/bidmc/1.0.0/).
