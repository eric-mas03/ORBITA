# Plan de decisiones y experimentos del gemelo digital

Versión 0.2 — 14 de noviembre de 2026. Incorpora las preferencias horarias del usuario. Este documento organiza decisiones y experimentos; la arquitectura final, resolución disponible y particiones siguen pendientes de evidencia.

## Preferencias incorporadas del usuario

**Aclaración posterior de arquitectura:** se conserva la secuencia de N latentes por rejilla hasta la atención. Esto ya ocurre en el código actual con N=24 y d=16; 24 h/15 min daría N=96 cuando los datos y el constructor lo respalden. Las propuestas de resúmenes/residuales añaden rutas auxiliares. El [plan vigente de comparaciones y ensemble final](plan-comparaciones-secuencia-y-ensemble.md) organiza las próximas fases, sin nuevos entrenamientos en esta revisión.

**Avance posterior autorizado:** ya se ejecutaron constructor, particiones fijas y primeros baselines horarios. Los [resultados del primer bloque](primer-bloque-horario.md) sustituyen el estado de solo planificación para esas tareas. La arquitectura temporal avanzada, transferencia, calibración y dashboard siguen como fases posteriores.

- Horizontes iniciales: **1, 4, 8 y 24 horas**, con q10/q50/q90 por variable.
- Aprovechar observaciones válidas incompletas mediante máscaras de entradas y objetivos; no exigir todos los grupos simultáneamente completos.
- Ventana candidata de **24 horas**; comparar otras duraciones en desarrollo para capturar evolución.
- Resolución candidata de **15 minutos**, con media/mínimo/máximo como posibles entradas si existen observaciones originales suficientes. El conjunto UCI descargado es horario y no permite reconstruir extremos intrahorarios.
- Evaluación y gráficas por variable, cabeza, horizonte y dominio, mostrando cuántas personas y etiquetas sustentan cada resultado.
- Arquitectura y transferencia guiadas por los agentes, contrastadas con baselines y ablaciones.
- Dashboard final para explorar historiales, predicciones y evaluación.

El usuario ha confirmado la semántica de salida: **valor del intervalo que termina en t+h**, sin promediar todo el periodo futuro desde t. Con rejilla de 1 h, la salida a 4 h corresponde a (t+3 h,t+4 h]; si hay datos reales de 15 min, a (t+3 h 45 min,t+4 h]. La estadística de cada variable respetará su método de adquisición y solo habrá etiqueta cuando exista observación válida.

La [hoja de ruta horaria](hoja-ruta-horaria.md) concreta las fases actuales. El experimento anterior de 112 días sigue siendo un prototipo histórico; sus métricas no se trasladan a horizontes horarios.

## Objetivo acordado

Predecir variables observables a varios horizontes utilizando información disponible hasta un origen temporal. Encoders de categorías biomédicas relacionadas, representación latente temporal común y cabezas de salida por objetivos o grupos. Salidas iniciales q10/q50/q90. La arquitectura existente es una referencia técnica que se puede contrastar, no una decisión cerrada.

## Cómo trabajaremos

En cada bloque: explicar alternativas, inspeccionar evidencia disponible, acordar una decisión y registrar cómo se comprobará. Mantener una tabla de decisiones con motivo y un experimento que pueda refutar cada elección. Las preguntas se resuelven antes de lanzar el experimento dependiente; las tareas independientes de auditoría pueden avanzar aparte.

Los datos adicionales ya están descargados y auditados a distinto nivel. UTMB C3 es la primera fuente candidata para ampliar CBC/CMP: 31 identificadores y 118 visitas, con 23 analitos comparables y 29 participantes con visitas repetidas. Sigue habiendo pocas visitas por persona. Sueño diario, microbioma y series horarias clínicas requieren contratos temporales propios. Fuentes terrestres, análogos y vuelo permanecen identificados y separados.

## 1. Definir el problema de predicción

**Preguntas para la conversación**

- ¿Qué variables serán entradas, salidas o ambas? ¿Qué papel tendrá cada grupo biomédico?
- ¿Queremos pronosticar una próxima visita de laboratorio, el valor a un plazo fijo o un agregado de un periodo futuro?
- ¿El uso principal será seguir a una persona conocida, comenzar con una nueva o transferir entre análogo y vuelo?
- ¿Qué información existirá realmente al emitir cada predicción, incluidas latencias de laboratorio y planes conocidos?

**Resultado:** una ficha por tarea con población/dominio, objetivo, unidad, origen de predicción, contexto disponible, definición del valor futuro y escenario de generalización. Empezar por un problema acotado antes de incorporar todas las modalidades.

## 2. Aclarar cobertura y ausencia

Separar conceptos que necesitan soluciones distintas:

| Concepto | Qué mediremos | Decisión relacionada |
|---|---|---|
| Cobertura de datos de entrada | Observaciones disponibles por variable, persona, fase y tiempo; antigüedad | Máscaras, elegibilidad del contexto, representación de ausencia |
| Cobertura de etiquetas | Resultados futuros realmente medidos por objetivo/horizonte/persona | Pérdida y denominador por cabeza; tareas evaluables |
| Cobertura predictiva | Fracción de resultados contenidos en q10–q90, por objetivo/horizonte | Evaluación y eventual calibración |
| Alcance de evaluación | Personas, misiones, dominios y horizontes presentes en prueba | Qué generalización podremos afirmar |

q10–q90 es una banda marginal central nominal del 80 %. No implica que el 80 % de todas las trayectorias completas o de todos los biomarcadores quede simultáneamente dentro. El 68 % del primer experimento describe cobertura predictiva, no porcentaje de datos presentes.

**Preguntas:** qué ausencias son esperables, cuáles dependen del estado de salud, si interesa una banda marginal o un criterio conjunto y cuándo el sistema debe indicar información insuficiente. Diferenciar ausencia, censura y valor cero.

**Resultado:** informe de cobertura y reglas de máscara por muestra/horizonte/variable/cabeza; métricas con sus denominadores. La pérdida enmascarada existente es el punto de partida. No imponer completar todas las salidas para aprovechar una muestra parcial.

## 3. Diseñar histórico, ventana y horizontes

**Alternativas de contexto:** últimas K observaciones, últimos D días o combinación de duración máxima y número máximo de eventos. Una visita no representa siempre la misma duración; días sin muestra no son observaciones nuevas.

**Preguntas:** histórico mínimo, longitud máxima, tratamiento del contexto escaso, reinicios/discontinuidades y si distintas modalidades deben tener ventanas diferentes. Decidir si el valor histórico de una salida puede usarse como entrada y cómo representar su antigüedad.

**Análisis previo:** contar ventanas y etiquetas utilizables para cada combinación candidata, desglosadas por persona/dominio. No fijar K o D por costumbre ni elegir un horizonte sin suficientes resultados observados. Para objetivos de plazo fijo con visitas irregulares, definir de antemano la tolerancia temporal o usar consultas al tiempo real de medición; no interpolar etiquetas para fabricar una evaluación.

**Resultado:** una o pocas configuraciones viables de ventana/horizonte. UTMB de laboratorio, sueño diario y UCI horaria se estudian en tareas distintas antes de compartir representaciones.

## 4. Fijar evaluación y referencias antes de comparar redes

- Reservar sujetos/subcampañas según la ficha de uso; distinguir continuación personal y sujeto nuevo.
- Respetar el tiempo de disponibilidad de datos, etiquetas, ajuste del modelo y selección de checkpoint.
- Separar desarrollo, calibración y prueba según el soporte real. No utilizar prueba para escoger ventanas o técnicas.
- Conservar escalas, imputación aprendida, selección de variables y aumento dentro de entrenamiento.
- Comparar persistencia y un modelo cuantílico regularizado; añadir una referencia recurrente pequeña si la tarea lo permite.

**Métricas:** MAE de q50, pinball, cobertura y anchura, por variable/horizonte/dominio; número de sujetos y de resultados. Estimar estabilidad respetando sujetos y dependencia temporal cuando el tamaño lo permita. Fijar el criterio de selección en desarrollo, no después de ver prueba.

**Resultado:** protocolo reproducible, particiones persistidas y baselines con los que comparar la arquitectura.

## 5. Revisar arquitectura y atención

### Ampliación del benchmark propuesta por el usuario: árboles

Incorporar bosques de regresión cuantílica y boosting con pérdida cuantílica a la lista de candidatos. Esta comparación estaba pendiente y debe formar parte de la referencia tabular antes de interpretar resultados como evidencia a favor o en contra de toda una familia neuronal. No se han entrenado árboles todavía.

Primero usar los mismos resúmenes causales de la referencia lineal: último valor, presencia/edad, media/mínimo/máximo, número de observaciones, desviación y duración del contexto. Mantener las mismas etiquetas, máscaras, unidades, horizontes y particiones; ajustar solo en entrenamiento/desarrollo. Después, una variante con información de otros grupos debe compararse por separado para distinguir el efecto del modelo del efecto de añadir entradas.

Un Random Forest de media no suministra por sí mismo los cuantiles condicionales solicitados. Usar un método cuantílico explícito; los percentiles de las predicciones medias de los árboles no equivalen automáticamente a cuantiles de la variable futura. Referencias primarias: [Quantile Regression Forests](https://www.jmlr.org/papers/v7/meinshausen06a.html) y [boosting cuantílico de scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).

La ejecución neuronal existente compara una configuración pequeña desde cero, con latente 16 y máximo 20 épocas; la lineal recibe resúmenes históricos explícitos y las redes reciben una secuencia comprimida mediante encoders. Por ello el resultado combina arquitectura, representación, optimización y tamaño efectivo de muestra. No aísla una inferioridad general del Transformer para predecir el futuro. La hipótesis de un cuello de botella latente sigue pendiente de comprobación.

Descomponer decisiones para que una comparación tenga una interpretación:

| Parte | Implementación de referencia | Pregunta experimental |
|---|---|---|
| Encoder biomédico | MLP por categoría, valores/máscaras/edades | ¿Qué representación aprende cada categoría? ¿Ayuda preentrenarla? |
| Fusión multimodal | Concatenación y proyección | ¿Aporta valor atención entre grupos observados? |
| Dinámica temporal | Transformer causal con producto escalar escalado | ¿Mejora frente a una dinámica recurrente más simple? |
| Tiempo irregular | Codificación continua y edades | ¿Ayuda un sesgo relativo basado en tiempo real? |
| Alcance temporal | Atención completa al pasado permitido | ¿Tiene sentido atención local con las longitudes reales disponibles? |
| Lectura del futuro | Consultas de horizonte y atención cruzada al historial | ¿Conviene compartir decoder o especializar por grupo/objetivo? |
| Cuantiles | Salidas ordenadas q10/q50/q90 | ¿Qué precisión/calibración logra cada objetivo? |

Las cabezas internas de atención y las cabezas de salida biomédicas son conceptos distintos. En la primera ejecución había dos internas y 13 de salida. La autoatención de consultas de horizonte es diagonal para que pedir otro horizonte no cambie la predicción del anterior.

**También por discutir:** tamaño latente, número de capas/cabezas, parámetros compartidos, especialización por dominio y ponderación de pérdidas. No incrementar varios componentes a la vez: una ablación cambia una decisión manteniendo particiones y presupuesto comparable. No tiene sentido evaluar atención de largo alcance si las historias solo contienen unas pocas visitas.

**Resultado:** un conjunto pequeño de comparaciones justificadas por la tarea y los datos, no una búsqueda indiscriminada de arquitecturas.

## 6. Decidir preentrenamiento y aumento

Evaluar por separado: entrenamiento específico, encoders preentrenados por reconstrucción de valores realmente observados y preentrenamiento temporal en una fuente longitudinal compatible. NHANES puede informar relaciones entre analitos; no enseña evolución individual.

Después, comparar una intervención de aumento por vez: ocultación artificial de entradas/modalidades o recortes causales de contexto. Conservar máscaras diferentes para ausencia real y corrupción artificial. Ruido necesita una justificación analítica. Generación sintética, mezcla y deformación temporal quedan pendientes de una hipótesis defendible y evaluación sobre sujetos reales reservados.

**Resultado:** saber si una técnica mejora transferencia y robustez en el dominio objetivo. El rendimiento en UCI o análogo se reporta separado del rendimiento en astronautas.

## 7. Integrar un gemelo evaluable

Tras seleccionar un candidato: reproducción de un historial, actualización del estado individual, pronósticos con origen/horizonte/unidad/versión, máscaras/calidad visibles y comparación posterior con observaciones reales. Definir cómo se recalibra usando solo resultados ya disponibles. Escenarios de intervención necesitan un planteamiento adicional: cambiar una entrada del predictor observacional no valida causalidad.

**Resultado:** prototipo trazable cuyo comportamiento y límites se puedan inspeccionar.

## Registro de decisiones

| ID | Decisión | Estado | Evidencia o validación prevista |
|---|---|---|---|
| D01 | Forecasting probabilístico desde historial, q10/q50/q90 | Dirección acordada | Precisar tareas en bloque 1 |
| D02 | Encoders por categorías y latente temporal común | Hipótesis de arquitectura acordada | Baselines y ablaciones |
| D02b | Conservar un latente por rejilla y dar la secuencia completa `[B,N,d]` al Transformer | Acordado y verificado para N=24 horario | Parametrizar N; 96 posiciones con datos reales de 15 min |
| D03 | Aprovechar etiquetas parciales con máscaras | Implementado, revisable | Pruebas de pérdida/gradientes y cobertura por cabeza |
| D04 | Mantener procedencia y separar dominios | Aplicado | Auditoría de cada fuente |
| D05 | Horizontes 1/4/8/24 h y q10/q50/q90 por variable en el intervalo que termina en t+h | Acordado | Auditar soporte y estadística publicada por variable/horizonte |
| D05b | Contexto 24 h y agregación 15 min con media/min/max | Candidatos del usuario | Comparar duraciones y usar resolución respaldada por la fuente; no fabricar 15 min a partir de datos horarios |
| D06 | Gráficas y métricas por variable, cabeza y horizonte | Acordado; particiones abiertas | Fijar separaciones por sujeto/hospital/dominio antes de entrenar |
| D07 | Atención/fusión/dimensión/capas definitivas | Abierto | Bloque 5 |
| D08 | Preentrenamiento externo y adaptación al dominio espacial | Hipótesis a contrastar | Comparar desde cero y transferencia, separando fuentes y objetivos con soporte real |
| D09 | Dashboard de resultados y reproducción de historiales | Objetivo acordado | Construir sobre predicciones y métricas versionadas después de validar el pipeline |
| D10 | Ensemble de candidatos individuales como fase final | Objetivo acordado, no implementado | Predicciones fuera de pliegue por episodio para pesos; calibración y prueba separadas |

El valor objetivo terminal está definido. Siguiente decisión de diseño: revisar el inventario horario de ventanas/etiquetas, resolución inicial, calidad y particiones del primer experimento horario.
