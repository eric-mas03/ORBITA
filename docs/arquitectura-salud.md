# Arquitectura y alcance biomédico del gemelo digital

> Diseño inicial anterior a la descarga. La agrupación aplicada está en [agrupacion-biomedica.md](agrupacion-biomedica.md), y la implementación en [modelo-implementado.md](modelo-implementado.md). La arquitectura descrita por el usuario sustituye la necesidad de aportar el TFM.

Fecha de revisión: 14 de noviembre de 2026. Documento de diseño inicial; no se han inspeccionado aquí archivos de mediciones ni validado modelos. Las variables y tecnologías siguientes son propuestas sujetas a la auditoría de datos y a la estructura del TFM que aporte el usuario.

## Evidencia disponible y límites

El artículo de la CSA del 26 de febrero de 2026 describe financiación para continuar desarrollando CANS, un instrumento autónomo de medición de exposición a neutrones. Es contexto científico y tecnológico: el artículo no entrega un conjunto longitudinal descargable de signos vitales y resultados clínicos individuales. No permite concluir que exista telemetría CANS pública utilizable para entrenar el proyecto. [CSA: Monitoring radiation for astronaut health](https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp).

La fuente principal facilitada para el reto es [Create Health Monitoring Software for Astronauts on Space Missions](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/). La apertura directa durante esta revisión devolvió HTTP 502; quedan pendientes la verificación del enunciado completo, los recursos y las condiciones específicas. No se interpreta este error como ausencia de recursos.

NASA SRAG explica que la exposición depende de entorno, ubicación de misión y blindaje. Diferencia sensores de área de los dosímetros activos de tripulantes, que registran exposición individual con fecha y hora. Por tanto, una serie ambiental no constituye por sí sola la dosis de una persona. NASA también describe herramientas especializadas de proyección de dosis en órganos y respuesta biológica; el prototipo no debe sustituirlas por una conversión arbitraria. [NASA SRAG: How do we protect the astronauts from space radiation?](https://srag.jsc.nasa.gov/spaceradiation/how/how.cfm).

NASA Standard Measures contempla actividad, luz, sueño, muestras biológicas, cognición, función sensorimotora y fuerza, con observaciones antes, durante y después de la misión. Que estas modalidades se estudien no confirma acceso público a todas ellas ni la frecuencia necesaria para forecasting. [NASA: Standard Measures](https://www.nasa.gov/reference/standard-measures/).

OSDR admite modalidades fisiológicas, conductuales, ambientales y ómicas, entre otras. Su FAQ señala que ALSDA contiene mayoritariamente organismos modelo y también astronautas humanos no NASA. En consecuencia, cada estudio debe auditarse por organismo, sujeto, misión, accesibilidad y temporalidad antes de usarlo. Datos animales o celulares no se incorporarán como si fueran mediciones de astronautas. [NASA OSDR FAQ](https://science.nasa.gov/reference/osdr-faq/).

## Separación de magnitudes

| Capa | Ejemplos posibles | Condición para incorporarla |
|---|---|---|
| Entorno externo | Flujo de partículas por energía; actividad geomagnética | Identificar instrumento, ubicación, bandas energéticas y tiempo. Son covariables del entorno |
| Entorno de cabina | Tasa de dosis del detector; CO2; temperatura | Conservar medio/material de medición, módulo, blindaje y calibración |
| Exposición individual | Dosis del dosímetro personal, dosis acumulada | Vinculación documentada entre dispositivo, sujeto y periodo de uso |
| Respuesta fisiológica | Frecuencia cardiaca, sueño, presión, actividad | Dispositivo/protocolo y contexto, con observaciones repetidas del mismo sujeto |
| Biomarcador | Hemograma, marcadores metabólicos o inmunitarios | Unidad, ensayo, muestra, lote, tiempo de toma y disponibilidad del resultado |
| Resultado de salud | Evento clínico, desempeño o resultado funcional definido | Etiqueta observada, definición verificable y seguimiento suficiente |

Propuesta de diccionario: conservar siempre valor y unidad originales además de su representación armonizada. La dosis absorbida (Gy) y las magnitudes de protección expresadas en Sv representan conceptos diferentes; no convertir unas en otras sin el modelo dosimétrico documentado. Tampoco sumar canales que ya estén incluidos en un total ni tratar cuentas del detector como dosis. Las tasas necesitan unidad temporal explícita y la integración requiere gestionar huecos, reinicios e incertidumbre del sensor.

Frecuencia cardiaca en latidos/min, HRV en ms y según métrica concreta, presión sistólica/diastólica en mmHg, temperatura en °C y sueño en minutos o horas son candidatos, no columnas confirmadas. Una escala de ánimo necesita instrumento y rango; una medición de glucosa necesita matriz, dispositivo y unidad; la densidad mineral ósea requiere modalidad y localización anatómica.

## Objetivos que dependen de la naturaleza del dato

1. **Telemetría densa:** predecir una medición o agregado definido a varios horizontes compatibles con su frecuencia real. Distinguir valor instantáneo, media y acumulado de la ventana futura.
2. **Muestras clínicas dispersas:** estudiar cambio frente al basal, tendencia por fase o próxima medición únicamente cuando haya visitas repetidas suficientes. Interpolar visitas no crea observaciones reales ni convierte una cohorte escasa en telemetría.
3. **Solo entorno/radiación:** construir un módulo de exposición ambiental. La personalización necesita datos de sujeto o un modelo físico explícito con supuestos visibles.
4. **Sin historial longitudinal utilizable:** construir ingestión, explorador y reproducción de datos; los datos sintéticos sirven para verificar software y demostrar interacción, sin presentar su rendimiento como evidencia sobre salud real.

Predecir un biomarcador no implica diagnosticar enfermedad. Estimar una asociación entre radiación y una señal fisiológica no identifica un efecto causal: sueño, actividad, medicación, fase de misión y otros factores pueden influir. Un cambio respecto al basal debe describirse como desviación observada; su importancia clínica requiere evaluación especializada. No se proponen límites clínicos universales ni recomendaciones terapéuticas en esta fase.

## Representación del astronauta

Se propone un estado por sujeto pseudonimizado con historial, basal estimado solo con información disponible, fase de misión, contexto, calidad y antigüedad de las observaciones. Separar el componente individual del componente compartido entre sujetos. Comparar primero modelos poblacionales simples con personalización y modelos jerárquicos; entrenar una red por astronauta puede ser inviable con pocos registros o pocas misiones.

Contrato mínimo de observación:

```text
subject_id?, mission_id, study_id, modality, variable
event_time, available_time, time_reference
value, unit, device_or_assay, quality_flags
source_file, source_row, source_version, provenance
data_kind = observed | simulated | derived
```

`event_time` es el tiempo del fenómeno o de toma de muestra; `available_time` indica cuándo habría estado disponible en el escenario de uso. Cuando no se conozca el segundo, documentar un supuesto de latencia o excluir esa variable del caso prospectivo. Si solo existe tiempo relativo a lanzamiento, conservarlo; no inventar fechas absolutas para emparejar estudios.

No unir estudios por fecha aproximada suponiendo que corresponden al mismo sujeto. La fusión entre modalidades requiere identidad y alineación verificables. La multimodalidad se añade por utilidad y disponibilidad, no por obligar a que el proyecto contenga imagen, texto y ómicas.

## Arquitectura propuesta

Para el primer prototipo conviene un repositorio modular y ejecución local reproducible. No hacen falta microservicios ni infraestructura distribuida antes de confirmar volumen y requisitos.

| Módulo | Responsabilidad | Salida revisable |
|---|---|---|
| Catálogo e ingestión | Descargar fuentes autorizadas, metadatos, licencia y checksums; conservar originales | Manifiesto por estudio y archivos inmutables |
| Calidad y armonización | Tipos, unidades, duplicados, irregularidad, reinicios, sincronización y sujetos | Tablas normalizadas, diccionario e informe de exclusiones |
| Constructor temporal | Ventanas, máscaras, retardos y disponibilidad hasta cada origen de predicción | Dataset versionado con sujetos, cortes y horizontes |
| Modelado | Baselines y modelos probabilísticos; ajuste solo en entrenamiento | Modelo, configuración y procedencia |
| Evaluación | Backtesting, error, pérdidas de cuantiles, cobertura y amplitud por sujeto/horizonte | Informe reproducible y criterio de aceptación |
| Estado del gemelo | Actualizar observaciones y estado estimado de cada sujeto en replay | Snapshot con versión y momento de corte |
| API | Consultar historial, calidad, estado y pronósticos con esquema validado | Contrato OpenAPI y respuestas trazables |
| Interfaz | Mostrar observado, estimado y predicho; explorar horizonte y fuente | Panel de reproducción de misión |

Tecnologías candidatas: Python con tablas Parquet y DuckDB para auditoría; bibliotecas de datos y aprendizaje elegidas tras fijar frecuencia y tamaño; FastAPI para servir resultados; frontend TypeScript si el TFM no exige otra pila. Seleccionar y fijar versiones al implementar. No instalar ni crear una aplicación prematuramente.

Contrato de predicción propuesto: sujeto, variable/unidad, origen temporal, horizonte, instante objetivo, q10/q50/q90, versión de datos/modelo/calibrador y advertencias de calidad. Las estimaciones deben cumplir q10 ≤ q50 ≤ q90. El intervalo q10–q90 es un intervalo predictivo central nominal del 80 %; no es un intervalo de confianza sobre una media ni asegura cobertura del 80 % en cada sujeto.

La UI mostrará banda q10–q90, mediana, datos observados y último dato disponible. Diferenciar ausencia de medición, mala señal y valor estimado; permitir que el sistema se abstenga cuando la información sea insuficiente. No interpretar q90 como umbral médico ni inventar una probabilidad de superar un límite a partir de solo tres cuantiles. Las alertas experimentales deben mostrar regla, fuente, duración de la desviación y revisión pendiente.

## Qué permitiría llamarlo gemelo digital

Definición operativa propuesta para este proyecto: representación de una persona identificable mediante pseudónimo, actualizada a partir de sus observaciones, con un estado explícito, pronósticos individuales evaluados y trazabilidad de sus actualizaciones. Un modelo general con un dashboard sigue siendo un monitor predictivo; un replay retrospectivo con estado por sujeto puede presentarse como prototipo de gemelo digital retrospectivo.

Los escenarios sobre ejercicio, blindaje o agenda se incorporarán solo con relaciones mecanísticas documentadas o validación causal apropiada. Variar una entrada de un predictor observacional no demuestra qué ocurriría si se aplicara una intervención. Conectar sensores y actualización prospectiva sería una fase posterior; no implica actuar automáticamente sobre la persona.

## Entregables y criterios de avance

1. Matriz de fuentes y variables: acceso, licencia, organismo, sujetos, periodos, frecuencia, unidades, cobertura y disponibilidad prospectiva.
2. Selección de uno o pocos objetivos con observaciones reales suficientes y horizontes justificables.
3. Dataset temporal auditado y particiones que permitan evaluar tiempo y generalización entre sujetos o misiones según el caso de uso.
4. Baseline reproducible y comparación probabilística sin mezclar datos sintéticos con el conjunto de evaluación real.
5. Replay de un sujeto con trazabilidad, calidad de datos y curvas q10/q50/q90.
6. Documentación de alcance y revisión por especialista de medicina aeroespacial/radioprotección antes de interpretar resultados como decisiones de salud.

La puerta de avance es la disponibilidad y vinculación real de las observaciones, no la elección de una arquitectura neuronal. El TFM del usuario permitirá adaptar módulos, interfaces y convenciones sin fijar anticipadamente una estructura incompatible.
