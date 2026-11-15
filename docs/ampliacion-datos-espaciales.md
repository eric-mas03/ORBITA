# Ampliación real de datos espaciales y análogos

Revisión y descargas: 14 de noviembre de 2026. La ampliación más útil para las variables sanguíneas del proyecto es **NASA LSDA UTMB Bedrest Campaign 3**, seguida por Campaign 1. No son astronautas en vuelo: son personas en estudios humanos de reposo en cama como análogo espacial.

El inventario ejecutable es `python scripts/discover_osdr.py --inventory`. Produce `data/manifests/space_expansion_summary.json`, con fuentes, muestras, personas, cohortes, tamaños y hashes. Verifica SHA-256 y tamaños de **97 archivos descargados, 39.108.744 bytes**, incluyendo metadatos y datos anteriores de Inspiration4. Esto no equivale a 97 estudios ni a 97 personas. Los archivos LSDA descargados verificaron además el MD5 publicado por NASA.

## Prioridad de integración

| Fuente y carpeta | Personas identificadas en esa fuente | Muestras/visitas descargadas | Uso y límites |
| --- | ---: | ---: | --- |
| [UTMB Campaign 3 Clinical Laboratory Assessment](https://nlsp.nasa.gov/explore/data/details/a36f6361-3492-5590-a675-a0257edf69ac?dataType=DATASETS), `data/raw/LSDA-UTMB-C3` | 31 identificadores | 118 visitas; 9.534 filas | CBC, química y otros ensayos; 31 CSV y 8 ReadMe; subcampañas C3A–C3H |
| [UTMB Campaign 1 Clinical Laboratory Assessments](https://nlsp.nasa.gov/explore/data/details/042df6cd-2a1a-59f7-903f-c32ab3b7c9de?dataType=DATASETS), `data/raw/LSDA-UTMB-C1` | 3 | 12 visitas | 8 CSV anchos y ReadMe; cuatro visitas por identificador |
| [UTMB Campaign 1 Actigraphy](https://nlsp.nasa.gov/explore/data/details/e26501eb-7d7a-5012-8c8e-c5bdb3eecaf6?dataType=DATASETS), `data/raw/LSDA-UTMB-C1-ACTIGRAPHY` | Los mismos 3 de C1 | 207 filas; 69 fechas | Sueño diario; segunda tabla parcialmente duplicada, no sumar |
| [OSD-191 Mars500](https://osdr.nasa.gov/bio/repo/data/studies/OSD-191), `data/raw/OSD-191` | 6 | 159 muestras | Microbioma intestinal; 2.916 ASV, no CBC/CMP; encierro de 520 días y seguimiento |
| [OSD-174 ISS](https://osdr.nasa.gov/bio/repo/data/studies/OSD-174), `data/raw/OSD-174` | 10 astronautas | 60 tablas; seis ventanas por persona | Expresión en folículos de cabello; fechas expresadas como ventanas, no instantes exactos |
| [OSD-53](https://osdr.nasa.gov/bio/repo/data/studies/OSD-53), `data/raw/OSD-53` | 6 astronautas individuales | 12 muestras pre/post y 1 pool de controles | Expresión sanguínea; el pool representa cinco controles, no cinco series individuales |
| [OSD-198 PlanHab](https://osdr.nasa.gov/bio/repo/data/studies/OSD-198), `data/raw/OSD-198` | 13 | 68 muestras | miRNA muscular; diseño cruzado HAMB/HBR/NBR con PRE/POST, no seis visitas consecutivas |
| [OSD-140 Mars500](https://osdr.nasa.gov/bio/repo/data/studies/OSD-140), `data/raw/OSD-140` | Metadatos de 6 | Solo 2 tablas de las 36 declaradas | Metilación; descarga parcial: el otro procesado supera 160 MB. No usar como cohorte longitudinal completa |
| [OSD-943](https://osdr.nasa.gov/bio/repo/data/studies/OSD-943) y [OSD-903](https://osdr.nasa.gov/bio/repo/data/studies/OSD-903), `data/raw/OSD-943` y `data/raw/OSD-903` | Los mismos 2 de Ax-1 | Solo JSON de metadatos | Tres tiempos declarados, pero no observaciones descargadas: ISA devuelve HTTP 403 |

Las cifras proceden de los archivos descargados, no de sumar artículos o paneles. No se suman individuos entre estudios/misiones/campañas sin correspondencia validada. OSD-140 y OSD-191 pertenecen a Mars500; OSD-903 y OSD-943 describen las mismas dos personas. La actigrafía C1 comparte sujetos con sus análisis clínicos. Otras modalidades de Inspiration4 siguen siendo sus cuatro personas.

## Esquema y tiempo de UTMB C3

Los CSV usan `ID`, `Campaign`, `Subject`, `Test_Phase`, `BR_Day`, `Date`, `Test`, `Group`, `ParameterName` o `Parameter Name`, `Units`, `Normal_Range`, `Measured_Value`. **Las dos grafías del parámetro deben homologarse antes de concatenar**. Hay 160 etiquetas de parámetro originales; no necesariamente 160 variables biomédicas distintas, pues existen sinónimos, matrices y ensayos diferentes.

Hay entre una y seis visitas por persona. Se conservan identificadores como texto, incluidos ceros iniciales. La clave de una medición debe incluir contexto de ensayo/grupo, unidad, fecha y sujeto; no solo el nombre. Albúmina, glucosa y proteína, entre otros, presentan varias unidades/contextos: no mezclar sangre y orina.

`BR_Day` no es un eje global: PRE_TEST expresa una magnitud previa, IN_TEST usa días de intervención y POST_TEST días desde finalizarla. Por ejemplo, el sujeto 9032 tiene PRE_TEST/12 el 2 de abril de 2009, IN_TEST/28 el 11 de mayo, POST_TEST/0 el 13 de junio y POST_TEST/5 el 18 de junio. Una reconstrucción ingenua del origen desde PRE e IN difiere un día por indexación. **Usar Date para intervalos transcurridos, conservando fase y día publicado**. No asumir fecha de disponibilidad clínica del resultado.

Los ReadMe registran cambios de instrumentos y rangos, resultados verificados, ensayos cancelados o muestras fuera de plazo. BSAP cambia de metodología/unidad en parte del periodo; los valores no deben reinterpretarse automáticamente. Hay un ReadMe C3H de cero bytes en NASA: se conserva tal cual y su MD5 coincide con el publicado. Los intervalos normales describen referencias del laboratorio, no observaciones ni objetivos.

## UTMB C1 y sueño

Los análisis de C1 son tablas anchas de 12 filas con nombres y unidades en encabezados. Incluyen tres sujetos y cuatro visitas PRE10, IN28, POST0 y POST7. Se conservan originales sin asumir armonización terminada. Algunas celdas contienen anotaciones como `5.30A (H)`; un parser debe separar número y anotación y conservar el texto original, sin confundirlo con ausencia ni eliminar censura.

En actigrafía, `BRSMACT_Campagin_1_Actigraphy.csv` contiene tres sujetos, 69 fechas y latencia, duración y eficiencia de sueño. Aunque el encabezado dice «hours», el ReadMe especifica HH:mm; algunas celdas llevan HH:mm:ss. Es necesario convertir duraciones explícitamente. `...Query_Result.csv` contiene 191 filas con información parcialmente solapada para dos sujetos: no apilar ambos archivos como observaciones independientes. La vinculación con C1 clínica puede realizarse por identificador y fecha, sin inventar mediciones de sangre diarias.

## Descarga reproducible y acceso

```powershell
python scripts/discover_osdr.py --lsda-dataset a36f6361-3492-5590-a675-a0257edf69ac --lsda-label LSDA-UTMB-C3
python scripts/discover_osdr.py --lsda-dataset 042df6cd-2a1a-59f7-903f-c32ab3b7c9de --lsda-label LSDA-UTMB-C1
python scripts/discover_osdr.py --lsda-dataset e26501eb-7d7a-5012-8c8e-c5bdb3eecaf6 --lsda-label LSDA-UTMB-C1-ACTIGRAPHY
python scripts/discover_osdr.py --inventory
```

El cliente LSDA consulta `/api/v1/data/datasets/{id}/full` y descarga archivos explícitamente desbloqueados mediante `/api/v1/files/{id}/download`. No usa credenciales ni elude restricciones; verifica MD5 y guarda SHA-256, fuente estable, tamaño y fecha en `data/manifests/lsda_downloads.json`. Los catálogos completos quedan junto a los archivos.

La ampliación OSDR se puede reproducir con:

```powershell
python scripts/download_osdr.py --studies 174 198 191 140 53 943 903 --max-file-bytes 20000000 --include '*processed.1.zip' --include '*processed.2.zip' --include '*series_matrix.txt' --include '*taxonomy-and-counts.tsv' --include '*processed.zip'
```

Se han descargado solo procesados pequeños y metadatos, no secuencias crudas. Los errores HTTP 403 quedan en el manifiesto. Los archivos de metadatos Ax-1 no se cuentan como observaciones. OSD-353 se inspeccionó, pero su procesado supera 29 GB y no se descargó. Los catálogos de cada estudio muestran todos sus archivos aunque el resultado API tenga paginación por estudios; los estudios actuales solicitados individualmente están completos a nivel de catálogo.

Las consultas iniciales y respuestas están en `data/manifests/discovery/`. Incluyen búsquedas por Axiom, bed rest, HERA, SIRIUS, Twins, Polaris, Shenzhou, astronaut, hematology, cardiovascular, isolation y Mars500. El índice federado devuelve resultados irrelevantes: solo se aceptaron fuentes NASA y estudios humanos identificados; ausencia de coincidencias no demuestra inexistencia de datos. No se ha realizado un censo exhaustivo mundial.

El siguiente paso es armonizar únicamente variables compatibles, evaluar por dominio y retener separadas las evaluaciones de vuelo, análogo y población clínica terrestre. Las series más largas de microbioma o sueño no convierten automáticamente el problema de CBC/CMP en predicción diaria validada.
