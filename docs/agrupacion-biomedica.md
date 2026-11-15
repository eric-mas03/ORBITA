# Agrupación biomédica para encoders y predicción temporal

Diseño dirigido desde la capa biomédica, 14 de noviembre de 2026. La asignación de variables se materializa en `config/biomedical_groups.json`; este documento explica sus decisiones. Los grupos son una partición operativa para EDA y arquitectura, no diagnósticos ni afirmaciones de independencia entre sistemas.

## Base de datos y evidencia

Se han inspeccionado los archivos reales descargados de OSD-575, suero de Inspiration4, y el hemograma de OSD-569. El esquema se deriva de sus columnas originales. La publicación de métodos distingue hemograma en sangre completa, panel metabólico en suero y paneles multiplex inmunitario y cardiovascular. Son ensayos diferentes; no se asumirá que comparten unidad, disponibilidad o calibración. [Publicación primaria de perfiles inmunes y métodos bioquímicos](https://www.nature.com/articles/s41467-024-49211-2).

Fuentes locales: `data/raw/OSD-575/*SUBMITTED.csv`, `LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv` y `data/raw/OSD-569/LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv`. El JSON contiene ruta exacta, columna y unidad por variable. Los paneles muestran cuatro sujetos y siete visitas L-92, L-44, L-3, R+1, R+45, R+82 y R+194; no contienen medidas durante vuelo. Alamar tiene 27 muestras y falta C003 en L-44. Un archivo con 28 filas por analito no equivale a 28 sujetos ni a observaciones densas.

Quedan **322 variables canónicas por ensayo**: CMP19, Eve cardiovascular9, Eve inmune71, Alamar203 y CBC20. La configuración habilita **109 variables en 13 grupos**, excluye cinco variables calculadas CMP y cinco porcentajes leucocitarios, y conserva Alamar203 como esquema confirmado con anotación biomédica pendiente, deshabilitado para el primer modelo. Habilitada significa asignada al contrato de software, no aprobada como objetivo estadísticamente viable: cada salida debe pasar el filtro de cobertura temporal y calidad.

La publicación de recolección describe muestras de suero para el panel metabólico y sus procedimientos de preparación. Tener muestras recogidas durante vuelo para alguna modalidad no significa que exista suero o panel metabólico durante vuelo. [Publicación primaria de biospecímenes](https://www.nature.com/articles/s41467-024-48806-z).

## Criterio de agrupación

Cada analito medido tiene un único grupo primario, que fija su encoder y su cabeza de salida. Los roles secundarios se guardan como etiquetas: IL-6, por ejemplo, no se convierte en dos entradas independientes por participar en varias vías. Una misma molécula medida por dos plataformas constituye dos mediciones distintas hasta verificar equivalencia analítica.

Dos decisiones explícitas de nomenclatura se apoyan en registros de referencia: IL-8 corresponde a [CXCL8, NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/3576), y la etiqueta IL-28A corresponde a [IFNL2, NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/282616). La clave original del ensayo se mantiene para evitar que un renombrado biológico rompa el enlace con el archivo.

La separación implementada usa los analitos reales y privilegia una interpretación clara:

| Grupo del JSON | N | Analitos reales o ejemplos | Interpretación para EDA |
|---|---:|---|---|
| `electrolytes_acid_base` | 5 | SODIUM, POTASSIUM, CHLORIDE, CARBON DIOXIDE, CALCIUM | Química sérica; CO2 total no equivale a presión parcial ni CO2 de cabina |
| `renal_energy_metabolism` | 3 | UREA NITROGEN (BUN), CREATININE, GLUCOSE | Marcadores con influencias múltiples, sin diagnóstico renal automático |
| `hepatobiliary_enzymes` | 4 | ALKALINE PHOSPHATASE, ALT, AST, BILIRUBIN; TOTAL | AST no es específica de hígado |
| `serum_proteins` | 2 | ALBUMIN, PROTEIN; TOTAL | Proteínas y estado de fluidos, sin inferir nutrición por sí solas |
| `chemokines_trafficking` | 24 | IL-8, IP-10, MCP-1, RANTES, MIG/CXCL9 y otras | Tráfico celular; los nombres originales permanecen en la clave |
| `interleukin_signaling` | 26 | IL-1α/β, IL-1RA, IL-2, IL-6, IL-10, IL-12p40/p70, IL-17A/F y otras | Señalización pleiotrópica, sin etiqueta pro/antiinflamatoria exclusiva |
| `interferon_tnf_signaling` | 7 | IFN-α2, IFNγ, IL-28A, TNFα, TNFβ, TRAIL, sCD40L | IL-28A se asigna por familia de interferón; sCD40L tiene roles secundarios plaquetarios |
| `growth_hematopoietic` | 14 | EGF, FGF-2, FLT-3L, G-CSF, GM-CSF, LIF, M-CSF, PDGF-AA, PDGF-AB/BB, SCF, TGFα, TPO, TSLP, VEGF-A | Crecimiento y hematopoyesis; PDGF/VEGF también tienen roles vasculares |
| `acute_phase_proteins` | 6 | a-2 Macroglobulin, AGP, CRP, Fetuin A36, Haptoglobin, SAP | Respuesta sistémica heterogénea, no un índice unidireccional de inflamación |
| `hemostasis_adhesion` | 3 | Fibrinogen, L-Selectin, PF4 | Hemostasia/adhesión; no estimación automática de trombosis |
| `cbc_erythrocytes` | 7 | RED BLOOD CELL COUNT, HEMOGLOBIN, HEMATOCRIT, MCV, MCH, MCHC, RDW | Serie roja e índices relacionados; no siete evidencias independientes |
| `cbc_leukocytes` | 6 | WHITE BLOOD CELL COUNT y ABSOLUTE NEUTROPHILS/LYMPHOCYTES/MONOCYTES/EOSINOPHILS/BASOPHILS | Total y diferencial absoluto con relaciones de suma |
| `cbc_platelets` | 2 | PLATELET COUNT, MPV | Recuento y volumen plaquetario |
| `alamar_exploratory` (deshabilitado) | 203 | Analitos completos enumerados en JSON | NPQ log2 de cuentas normalizadas IPC; no pg/ml ni nueva transformación log por defecto |

El JSON proporciona `input_groups` y `output_groups` como mapas `id_grupo -> [variable_key...]`, en orden estable para construir tensores. `variable_key` es `CMP::<ANALYTE>`, `EVE_CARDIO::<Analyte>`, `EVE_IMMUNE::<Analyte>`, `ALAMAR::<nombre original sin _concentration_npq>` o `CBC::<ANALYTE canónico>`. Cada variable tiene ensayo, unidad, fuente y habilitación. Los encabezados (FEMALE) del CBC se canonizan únicamente en tres analitos equivalentes, preservando la etiqueta original y los rangos específicos en la tabla auditada; no definen biomarcadores diferentes. Así, 23 etiquetas originales CBC son 20 analitos canónicos.

La configuración final puede fusionar bloques de uno o dos analitos para reducir parámetros. Esa decisión debe depender de la cobertura y del experimento, dejando persistente la anotación biomédica más fina para EDA.

## Redundancias y composición

El panel metabólico puede incluir variables calculadas: eGFR, cociente BUN/creatinina, globulina y cociente albúmina/globulina. Si aparecen, se conservarán en la tabla auditada con su fórmula y procedencia, pero quedarán fuera por defecto del conjunto independiente de entradas/salidas. Que un analito sea derivado no es fuga temporal por sí mismo; sí puede dar peso repetido a la misma información y crear tareas artificialmente fáciles cuando sus componentes contemporáneos se incluyen como entradas.

En el CBC confirmado, hemoglobina, hematocrito, RBC e índices eritrocitarios tienen dependencias algebraicas/fisiológicas; recuentos leucocitarios absolutos, porcentajes y total también. Los cinco porcentajes diferenciales se excluyen del modelo por defecto y se conservan para EDA composicional, sin interpretar correlaciones negativas inducidas por cierre como regulación biológica. HEMATOCRIT y RDW usan %, pero no son partes del diferencial leucocitario: no se excluyen por una regla genérica sobre unidades. Los siete índices de serie roja quedan anotados como redundantes; se propone una ablación reducida (RBC, hemoglobina, MCV, RDW) frente al bloque completo, con el mismo protocolo de evaluación. La reducción se decide en desarrollo, no a partir de resultados de prueba.

Las concentraciones de citocinas y las químicas séricas no forman, por defecto, una composición de suma constante. No aplicar CLR o normalización a suma uno por rutina. Un futuro bloque de abundancias microbianas o proporciones celulares sí requiere evaluación composicional, ceros, denominador y método de transformación. El identificador del ensayo y el tipo de matriz acompañan siempre a cualquier transformación.

Los campos `Percent` de Eve son otra cosa: normalización relativa a una referencia de la cohorte/preflight, no proporciones de células. Quedan excluidos para evitar duplicación y fuga de información de sujetos/visitas futuras. En CBC y CMP mandan las unidades del archivo SUBMITTED: por ejemplo, hemoglobina está en g/dL aunque un encabezado transformado pueda sugerir porcentaje. Los rangos de referencia del laboratorio se preservan para auditoría, sin transformarlos automáticamente en límites operacionales astronauta.

## Transformación para EDA y encoders

1. Guardar valor original, unidad, método, límite de detección y códigos de censura. Un valor menor que LOD no equivale a cero observado. Separar ausencias de censura e incidencias analíticas.
2. Revisar asimetría y rango por analito. Probar transformación logarítmica solo para concentraciones no negativas y cuando sea útil; el desplazamiento y la escala se estiman en entrenamiento. No truncar negativos de fondo silenciosamente.
3. Estandarizar por variable utilizando entrenamiento. Los cambios frente al basal personal usan únicamente visitas ya disponibles en el origen de predicción, con registro de cuántas medidas sustentan ese basal.
4. Conservar las observaciones irregulares. Crear un token por sujeto, visita y grupo a partir de valores, máscara y edad de cada variable; no fabricar una rejilla de mediciones interpoladas.
5. Evaluar cobertura por bloque, no solo por tabla. Un bloque sin datos en una visita recibe máscara de ausencia; no se presenta un vector imputado como si fuera una observación.

## EDA que guía la fusión

La primera matriz es analito × sujeto × visita con calidad y unidades. Después se trazan curvas individuales, distribución por ensayo, porcentaje de ausencias/censura y diferencias frente a la referencia personal. Se estudian correlaciones tanto dentro de bloque como entre bloques, identificando duplicados, relaciones calculadas y patrones compartidos por visita o lote.

Con pocos sujetos no usar una matriz de correlación de todas las filas como evidencia de asociación generalizable. Separar variación entre personas y cambios dentro de cada persona; informar número de sujetos y pares observados. Usar comparaciones descriptivas y estabilidad al excluir sujetos; ajustar selección de correlaciones y reducción dimensional solo en entrenamiento si alimentan el modelo. Las visitas próximas, réplicas técnicas y analitos del mismo ensayo no son observaciones independientes. Correlación entre bloques no determina causalidad ni necesidad automática de fusionarlos.

## Latente común y cabezas transformer

Propuesta aplicable al código: un encoder pequeño por bloque transforma `[valor, máscara, edad]` a una dimensión compartida. Una proyección con identidad de grupo y codificación del tiempo relativo permite fusionar bloques observados. El estado temporal usa máscaras causales y una rejilla de eventos reales; al emitir en `t` se excluye cualquier resultado con disponibilidad posterior, aunque la toma de muestra sea anterior.

Las consultas de salida contienen grupo objetivo, horizonte y variable; una cabeza transformer por grupo comparte contexto temporal y devuelve q10/q50/q90 por analito. Los grupos de salida no necesitan reproducir todas las entradas: las variables calculadas o sin suficientes etiquetas futuras se deshabilitan. Las pérdidas se enmascaran por etiqueta observada y se equilibran por analito/grupo para que un panel multiplex grande no domine al metabólico. Las pérdidas se calculan en escala transformada o normalizada explícita y las métricas finales se devuelven también a unidades originales.

El rango latente es compartido, pero no se fuerza que dos bloques tengan el mismo contenido. El alineamiento por visita no obliga a maximizar correlación de modalidades ni a reconstruir una desde otra sin evidencia. Se preservan identidades de grupo, tiempos y ausencias. Una dimensión inicial pequeña y pocas capas hacen viable probar el flujo; el ajuste de dimensiones se contrasta con baselines. Con cuatro sujetos y pocas visitas, que el software soporte transformer no demuestra que sus parámetros puedan estimarse con fiabilidad.

Los cuantiles se mantienen ordenados. El intervalo q10–q90 es nominalmente predictivo al 80 % para cada variable/horizonte; cobertura y amplitud deben evaluarse por salida. No se llama intervalo de confianza clínico ni se interpreta como cobertura conjunta de todos los analitos.

## Reglas de ampliación

Wearables y radiación permanecen como candidatos deshabilitados; microbios y ómicas requieren otra ampliación explícita. Alamar está descargado pero todavía deshabilitado como bloque de 203 analitos hasta anotar funciones y revisar normalización del ensayo. No se mezcla NPQ con concentraciones Eve ni se promedian analitos homónimos entre plataformas. Una categoría de órgano no justifica fabricar una variable ausente. Todo cambio del mapeo registra versión y motivo; un analito nuevo sin asignación se detiene para revisión en lugar de enviarse silenciosamente al encoder más parecido.
