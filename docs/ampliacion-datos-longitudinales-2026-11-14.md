# Búsqueda y adquisición adicional de datos longitudinales

Fecha: 14 de noviembre de 2026. Objetivo: historial de 24 h, rejilla de 15 min y cuantiles a 1, 4, 8 y 24 h, con máscaras por variable. La búsqueda se contrasta con los manifiestos existentes para no contar de nuevo personas, ensayos o documentos.

## Resultado de la búsqueda

| Fuente primaria | Dominio y cobertura declarada | Acceso comprobado | Decisión |
|---|---|---|---|
| [BIG IDEAs 1.1.3](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/) | Tierra; 16 personas, 8–10 días; CGM de 5 min, pulso, temperatura cutánea, EDA, movimiento y comidas | Archivos públicos; se descargaron CGM y comidas de las 16 personas, pulso de una y metadatos | Primera opción nueva para estudiar previsión horaria de glucosa y contexto alimentario. Aún no se han entrenado modelos con estos datos |
| [CGMacros 1.0.0](https://physionet.org/content/cgmacros/1.0.0/) | Tierra; 45 participantes, unos 10 días; CGM, pulso, actividad y macronutrientes | ZIP público; adquisición selectiva de tablas CSV, sin fotos, mediante rangos HTTP | Revisar interpolación de la tabla de 1 min antes de cualquier entrenamiento. No equivale a sensores nativos cada minuto |
| [PhysioCGM](https://api.figshare.com/v2/articles/28136294) y [repositorio de los autores](https://github.com/PSI-TAMU/PhysioCGM) | Tierra; 10 personas con diabetes tipo 1, hasta 17 días; CGM, ECG, respiración, pulso, PPG, EDA y movimiento | API pública verificada; archivos brutos suman unos 9 GB, además de procesados mayores. Se guardó catálogo, no señales | Candidato multimodal posterior; su población y tratamiento difieren de astronautas. Mantener el dominio y evitar duplicar bruto/procesado |
| [Inspiration4, estudio fisiológico](https://pmc.ncbi.nlm.nih.gov/articles/PMC11357997/) | Vuelo real; 4 personas. Describe medias horarias de HR, HRV y energía y mediciones de SpO₂; uso en vuelo del reloj breve | La publicación demuestra que existen mediciones, pero no se confirmó una descarga pública de las series individuales del reloj | Prioridad científica espacial, no se contabiliza como observaciones adquiridas ni se extraen puntos de una figura como si fueran datos originales |
| [TRISH EXPAND FAQ](https://www.bcm.edu/academic-centers/space-medicine/expand/expand-database-biorepository-faq) y [convocatoria 2026](https://spacehealth.bcm.edu/protected/nr/x2h4V/FAQ_Data_Request_Solicitation_-_04082026.pdf) | Salud y entorno antes/durante/después de múltiples misiones comerciales | Acceso controlado; la convocatoria de 2026 limita participación institucional a EE. UU. y prevé acceso en enero de 2027; no es descarga inmediata. La FAQ general indica que no acepta solicitudes en ese momento | Conservar como vía institucional, sin enviar solicitudes ni prometer acceso |
| [Spaceflight Standard Measures](https://pmc.ncbi.nlm.nih.gov/articles/PMC12612093/) | Vuelo real; fisiología cardiovascular, sueño, ejercicio, bioquímica y otras modalidades | Requiere solicitud formal, aprobación ética e institucional y acuerdo de uso; no se descargaron datos | La vía más alineada con el objetivo espacial amplio, sujeta a acceso y a comprobar la cadencia de cada modalidad |
| [MMAARS / UAM-L](https://xogi.ler.uam.mx/items/d1768c23-40e0-47fd-92e0-aa2115b7fc23) y [artículo](https://doi.org/10.3390/aerospace11100833) | Análogo terrestre, 5 personas, 4 días, sesiones ECG de 5 min por mañana y tarde; intervalos RR y actividades | Página institucional comprobada; enlace ZIP publicado devolvió 404. No hay observaciones adquiridas | Puede servir para HRV y contexto de misión; las sesiones cortas no permiten ventanas continuas de 24 h ni pronósticos horarios de todos los signos |
| [OpenMHC, documentación del productor](https://github.com/AshleyLab/OpenMHC/blob/main/DATASET.md) | Tierra; documentación anuncia 11.894 participantes y subconjunto de 593; datos wearable y tareas de previsión | El productor indica publicación del conjunto tras aceptación del artículo y condiciones DUA; tener código o checkpoints no acredita acceso a datos | Candidato a preentrenamiento, no descarga disponible verificada. No se instaló el paquete ni se contaron personas como adquiridas |

## Auditoría del piloto ya adquirido

El manifiesto `data/manifests/longitudinal_pilot_20261114.json` registra **40 archivos, 15.435.444 bytes**, sin errores de descarga. Son un subconjunto explícito de BIG IDEAs y diccionarios/licencia de CGMacros, no ambos conjuntos completos.

Para BIG IDEAs se verificaron **35 hashes de archivos contra SHA-256 publicados por el repositorio**. El archivo SHA256SUMS se conserva además como evidencia. Resultados locales:

- 16 participantes, **36.898 mediciones CGM**, **12.330 intervalos de 15 min con al menos una medición**.
- Las series realmente presentes abarcan aproximadamente **6,87–9,91 días**. La cobertura por sujeto en la rejilla es **75,84–100 %**; no se rellenaron huecos.
- Hay **1.422 filas de comidas**, que son alimentos/entradas del diario, no 1.422 comidas independientes. El archivo del sujeto 003 carece de cabecera fiable y queda fuera de cualquier integración nutricional automática.
- Con al menos 77/96 intervalos históricos observados en 24 h y una etiqueta futura disponible, se cuentan 10.343, 10.116, 9.843 y 8.870 ventanas a 1, 4, 8 y 24 h respectivamente. Son ventanas solapadas, no muestras independientes ni evidencia de precisión del modelo.
- El piloto HR de 001 contiene 634.188 filas y 714 intervalos de 15 min observados. Sus marcas de tiempo están expresadas en minutos y se repiten; no se inventan segundos. No se ha descargado HR de los otros 15 participantes ni todas las señales E4.

`outputs/research_expansion_20261114/pilot_summary.json` y `bigideas_cgm_coverage.csv` contienen los resultados reproducibles. La media por intervalo usa cierre y etiqueta por la derecha; no usa puntos posteriores al final de ese intervalo.

## Problema detectado en CGMacros

Se adquirieron **48 tablas CSV: 45 series de participantes y tres tablas auxiliares**, con 46.981.020 bytes descomprimidos. La lectura selectiva transfirió 7.118.570 bytes del ZIP público de 657.187.340 bytes. Las 45 series suman **687.580 filas de la rejilla publicada de 1 min y 1.706 filas con tipo de comida**. Esas filas no se contabilizan como 687.580 lecturas nativas del sensor ni como muestras independientes.

El diccionario distingue lectura nativa Dexcom de 5 min y Libre de 15 min, mientras que las tablas publicadas están a 1 min. En las tablas descargadas aparecen secuencias como 112, 111,8, 111,6, 111,4, 111,2, 111: evidencia numérica compatible con interpolación lineal. Una interpolación que utiliza el punto siguiente puede introducir futuro en una entrada o en una etiqueta.

Se conserva la fuente original y se marca **pendiente de validar anclas nativas; excluida de nuevas campañas de previsión hasta resolverlo**. No se afirma haber recuperado las mediciones nativas eliminando decimales o tomando una fila cada cinco minutos: el desfase y los huecos deben verificarse con el productor o un fichero nativo. Las tablas siguen siendo útiles para inspección de esquema y diseño de la integración de comidas, con esa limitación.

Ejemplo local CGMacros-001: 62,36 % de valores Dexcom no enteros y 82,56 % de tripletas válidas con segunda diferencia cero. Son indicadores de tramos lineales; por sí solos no determinan qué filas concretas fueron medidas. El resultado por persona está en `cgmacros_table_audit.csv`.

## Planteamiento propuesto con estos datos

1. Abrir una campaña independiente para BIG IDEAs, con división por persona y purga temporal; no reutilizar ni modificar las reservas de las campañas existentes.
2. Definir canales separados para glucosa intersticial y glucosa sanguínea de laboratorio. La temperatura de piel también permanece separada de la corporal central.
3. Estimar baselines de persistencia y estacionalidad diaria antes del Transformer. Comparar 24/48/72 h de contexto solo donde realmente haya historia.
4. Modelar glucosa, pulso y las modalidades disponibles mediante pérdidas enmascaradas; una máscara permite aprender de observaciones existentes, no crea etiquetas faltantes ni valida una transferencia a vuelo.
5. No incluir las comidas futuras registradas a posteriori como entradas conocidas. Los escenarios futuros deben ser introducidos explícitamente y evaluarse aparte. Datos observacionales y desayunos protocolizados no bastan para recomendar intervenciones personalizadas en astronautas.
6. Usar el conjunto NASA actual para contexto de misión, visualización y análisis por visita. No presentar sus análisis espaciados como etiquetas horarias.

No se ha entrenado, recalibrado ni revaluado ningún modelo en esta búsqueda. Los artefactos V1/V2 congelados permanecen intactos. La incorporación al dashboard requerirá identificar explícitamente el nuevo dominio terrestre y la procedencia de las previsiones.

## Ampliación posterior con prioridad NASA

Se inspeccionaron mediante la API oficial diez catálogos de experimentos LSDA: Apollo AP005, monitorización ambulatoria Mir, BRSMCF, BRSMAct, DSO484, medidas BHP, Holter/EVA del Shuttle, Concordia, Bio-Monitor y Biochemical Profile. La evidencia íntegra está en `data/raw/research_expansion_20261114/NASA_LSDA/` y el inventario en `data/manifests/nasa_longitudinal_search_20261114.json`.

**Radiación real de la ISS, descargada.** [RadLab de NASA OSDR](https://visualization.osdr.nasa.gov/radlab/gui/data-api/) ofrece datos instrumentales fechados. El piloto DosTel1 del 1 al 15 de marzo de 2022 contiene **16.807 filas**, de 2022-03-01 00:00:00 a 2022-03-14 23:59:41, 734.752 bytes. Hay 43 marcas de tiempo duplicadas con valores diferentes: se conservan los originales y se excluyen sus 86 filas del análisis derivado, sin promediarlas como si fueran mediciones inequívocas. Tras esa exclusión, la mediana de separación es 88 segundos; el hueco máximo es 1.398 segundos. Se observan 1.344 intervalos de 15 min de los 1.345 definidos por las etiquetas de cierre de la rejilla, sin interpolación. La unidad documentada es µGy/h. `radlab_summary.json` registra también los objetivos disponibles a los cuatro horizontes.

Esto añade una modalidad ambiental espacial continua. **No mide glucosa, pulso ni dosis personal de un tripulante**. No se empareja por fecha con las series terrestres anonimizadas. La convención de zona horaria debe confirmarse antes de unir calendarios externos. Los otros archivos `pilot_attempt_*.csv` son consultas de comprobación solapadas: no se suman como nuevos registros independientes.

**Función cardiovascular NASA en análogos, descargada.** Del experimento [BRSMCF](https://nlsp.nasa.gov/search/data/details/9af6f9ba-3730-54ab-a82a-1a32798667ae?dataType=EXPERIMENTS) se adquirieron 14 tablas públicas de tres catálogos, y se comprobaron sus 14 MD5 publicados:

| Catálogo | Tablas | Contenido verificado |
|---|---:|---|
| [UTMB C1 ecocardiografía](https://nlsp.nasa.gov/search/data/details/95186651-28d7-5c72-8f32-d40838766078?dataType=DATASETS) | 6 | Los mismos tres identificadores C1 previamente disponibles; mediciones y ensayos repetidos, medias y cambios relativos. No son seis cohortes ni nuevas personas por modalidad |
| [UTMB C3 ecocardiografía](https://nlsp.nasa.gov/search/data/details/c3bdfa7e-6661-5891-9ba0-0fc7c7a080d6?dataType=DATASETS) | 5 | Tabla principal: 141 filas y 15 identificadores; otras tablas incluyen medias agregadas y otras subcampañas. No sumar filas observadas y medias como etiquetas independientes |
| [CFT70](https://nlsp.nasa.gov/search/data/details/59501092-7252-5848-b79c-650e954b51ed?dataType=DATASETS) | 3 | 37 identificadores en la tabla 2D de 184 filas y los mismos 37 en 3D de 510 filas; el subconjunto de siete personas no añade participantes |

Son estudios humanos de reposo en cama publicados por NASA, **no astronautas en vuelo**. Aportan variables cardiovasculares y evolución por visitas/días; no signos vitales continuos a 15 min. No se suman personas entre campañas ni con los análisis previos sin correspondencia validada. El campo `SBP` es inferior a `DBP` en las 113 filas emparejadas de la tabla C3ABCD de observaciones y en 29/34 medias emparejadas: ambos campos quedan pendientes de revisión de etiquetas/unidades, sin intercambiarlos automáticamente. El inventario íntegro está en `nasa_echo_inventory.json`.

**Fuentes espaciales relevantes que no se pudieron adquirir como observaciones públicas.** [Holter/EVA del Shuttle](https://nlsp.nasa.gov/search/data/details/06699312-6756-5f70-a71b-f61abb300f3f?dataType=EXPERIMENTS) describe sesiones de 24 h en siete tripulantes; su catálogo declara todos los datos protegidos y disponibles por solicitud. [Concordia](https://nlsp.nasa.gov/search/data/details/6e611fcf-06df-5ef0-b76d-7862f6c1b499?dataType=EXPERIMENTS) describe actigrafía continua y ECG mensual de 24 h, también protegido. [Bio-Monitor](https://nlsp.nasa.gov/search/data/details/f9f4485d-656c-5980-9a5f-4f57c7ba383a?dataType=EXPERIMENTS) indica que el estudio internacional no archivó datos con LSDA. [Biochemical Profile](https://nlsp.nasa.gov/search/data/details/2a30c3f0-93d8-5060-b546-ba229f740cc3?dataType=EXPERIMENTS) ofrece muestras seriadas en vuelo, pero sus conjuntos son protegidos. El archivo abierto de Mir es un calendario de recogida; los de Apollo revisados son resúmenes/gráficas. Ninguno se contabiliza como nuevas etiquetas horarias individuales.

La adquisición y auditoría NASA se reproducen con:

```powershell
python scripts/discover_nasa_longitudinal.py
python scripts/acquire_nasa_followup.py
python scripts/audit_nasa_followup.py
```

El siguiente uso defendible sería estudiar una cabeza ambiental de radiación separada, enriquecer la representación cardiovascular con los análogos conservando sus días reales y, para signos vitales humanos horarios en vuelo, preparar un acceso institucional a una fuente apropiada. No hay nuevo modelo entrenado ni nueva validación clínica espacial en esta ampliación.

## Reproducción de la búsqueda inicial

```powershell
python scripts/research_additional_sources.py
python scripts/acquire_longitudinal_pilot.py
python scripts/audit_longitudinal_pilot.py
python scripts/acquire_cgmacros_tables.py
python scripts/audit_cgmacros_tables.py
```

La descarga selectiva de CGMacros valida la respuesta HTTP 206 y el rango exacto, limita los bytes transferidos, y deja un manifiesto por miembro CSV con SHA-256 local y CRC32 comprobado por ZIP. No descarga ni verifica el hash global del archivo completo, ni adquiere las fotografías. Las licencias originales se conservan; CGMacros está sujeto a CC BY-NC-SA 4.0 y BIG IDEAs a ODC Attribution 1.0.
