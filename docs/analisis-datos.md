# Auditoría inicial de datos

> Documento histórico anterior a la descarga. El estado vigente y el análisis de datos reales están en [fuentes-descarga.md](fuentes-descarga.md) y [eda-osd575.md](eda-osd575.md).

Fecha: 14 de noviembre de 2026. Estado: revisión documental; todavía no hay archivos de datos descargados ni análisis estadístico ejecutado.

## Qué se ha verificado

- El directorio del proyecto estaba vacío al comenzar la revisión.
- Existe el reto oficial 2026 [Create Health Monitoring Software for Astronauts on Space Missions](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/). Su resumen indexado propone reunir indicadores y facilitar la evaluación de salud. No establece por sí solo una tarea supervisada con entradas y salidas prefijadas.
- La página oficial y su pestaña Resources fallaron al abrirse con la herramienta web. Por tanto, no se ha confirmado un inventario oficial exhaustivo de datasets del reto. No debe confundirse falta de acceso de la herramienta con inexistencia de recursos.
- El artículo de la [CSA sobre CANS](https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp) describe instrumentación de radiación, pero no entrega en esa página una serie de dosis individual emparejada con resultados clínicos.

## Fuentes candidatas; no son datasets oficiales del reto confirmados

| Fuente | Contenido y acceso verificados | Utilidad y limitación temporal |
| --- | --- | --- |
| [NASA OSDR](https://science.nasa.gov/reference/osdr-faq/) | Estudios biológicos con metadatos y archivos; numerosas especies; acceso público y controlado según estudio/tipo de dato | Filtrar humanos y verificar sujeto, fechas y medidas repetidas antes de elegir variables |
| [NASA Life Sciences Portal / LSDA](https://nlsp.nasa.gov/) | Investigación humana de vuelos y análogos; archivo de salud e investigación | La existencia de un registro de estudio no garantiza acceso inmediato a sus datos individuales |
| [Procedimiento de acceso controlado OSDR](https://www.nasa.gov/reference/osdr-tutorials-requesting-controlled-access-datasets/) | Restricciones identificadas por archivo; solicitud de acceso | No planificar el MVP suponiendo autorizaciones ya obtenidas |
| [Inspiration4: publicación de acceso a datos](https://www.nature.com/articles/s41526-024-00393-5) | Cuatro participantes; misión de tres días; conjuntos multimodales longitudinales | Cohorte pequeña, visitas irregulares; cada modalidad requiere auditoría propia |

Inspiration4 ofrece estudios OSD-569, 570, 571, 572, 573, 574, 575, 630, 656 y 687. El artículo de acceso documenta para OSD-572 visitas L-92, L-44, L-3, FD1, FD2, FD3, R+1, R+45 y R+82. Son muestras microbianas, no una serie continua de constantes vitales. Documenta metadatos descargables CSV/ISA-Tab y acceso programático, con restricciones para ciertos datos humanos. La disponibilidad se verificará archivo a archivo. Los paneles sanguíneos (por ejemplo OSD-575) son candidatos multimodales, sin asumir que compartan ese calendario ni que tengan suficientes pasos para forecasting.

## Auditoría que debe producir el primer pipeline

Para cada archivo registrar URL/identificador y versión, fecha de descarga, hash, licencia/condiciones, especie, cohorte, misión, número de personas, número de visitas por persona y variable, instante de adquisición, instante en que el dato estaba disponible, unidad, límites de detección, calidad y porcentaje de ausencias. Distinguir mediciones, agregados y simulaciones.

Construir una tabla de observaciones en formato largo: `subject_id`, `mission_id`, `observed_at`, `available_at`, `variable`, `value`, `unit`, `quality_flag`, `source_id`. Conservar tiempo relativo a lanzamiento/regreso si no existen fechas absolutas. No crear correspondencias entre personas o misiones que la fuente no documente.

El análisis exploratorio debe mostrar duración por sujeto, intervalos entre medidas, huecos, simultaneidad entre modalidades, cobertura de cada posible salida, autocorrelación y cambio de distribución entre fases de misión. La mera cantidad de genes o ventanas solapadas no aumenta el número de participantes independientes.

## Criterios para decidir forecasting

**Adelante, de forma exploratoria**, para una variable y horizonte concretos, si hay observaciones ordenadas por persona, ventanas históricas y resultados futuros reales suficientes, trazabilidad de disponibilidad de entradas, particiones temporales viables y evaluación independiente. El mínimo de observaciones se decidirá con duración, autocorrelación, horizonte y curva de aprendizaje; no con una cifra universal.

**Reformular**, si solo hay pre/post, promedios de grupo, datos transversales, identidades no enlazables, horizontes mayores que el seguimiento o salidas medidas demasiado pocas veces. En esos casos estudiar cambios entre visitas, desviación frente a línea base o un simulador explícitamente etiquetado.

**No entrenar todavía**, mientras no se hayan inspeccionado archivos reales. No unir radiación ambiental de otra misión con biomarcadores humanos como si existiera un emparejamiento observado. No interpolar mediciones futuras para fabricar la entrada histórica.

## Validación prevista

- Separación cronológica con origen móvil; entrenamiento de normalizadores e imputadores exclusivamente con el tramo de entrenamiento.
- Dos evaluaciones distintas: futuro de una persona conocida y generalización a persona/misión nueva. Evitar ventanas solapadas a ambos lados de una partición cuando compartan información que invalide la evaluación.
- Comparar persistencia, línea base personal y modelos simples antes de ampliar arquitectura.
- Por variable y horizonte: pérdida pinball para q10/q50/q90, error de q50, cobertura y anchura del intervalo q10–q90, número de participantes y predicciones evaluadas.
- q10–q90 constituye un intervalo predictivo central nominal del 80 %, sujeto a calibración; no un intervalo de confianza de los parámetros ni una garantía clínica.

## Decisión provisional

El proyecto puede diseñarse ya; la viabilidad de entrenar predicción multihorizonte de salud astronauta sigue **pendiente de datos**. El siguiente hito es descargar un estudio candidato con acceso permitido, producir su manifiesto y medir cobertura temporal. Después se fijarán salidas, historial y horizontes. No hay resultados de entrenamiento que comunicar todavía.
