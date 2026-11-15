# Capa de interpretación para revisión profesional

14 de noviembre de 2026. Se implementó un contrato de evidencia y una plantilla descriptiva en español. No se descargaron pesos, entrenó un modelo de lenguaje, inventaron pacientes/informes ni transmitieron datos a una API. Las pruebas ejecutadas validan software; no constituyen evaluación clínica.

## Separar predicción de interpretación

El predictor conserva sus valores numéricos q10/q50/q90, máscaras, horizontes y evaluación. Una capa posterior organiza esa evidencia para que un especialista pueda revisarla. El modelo de lenguaje no calcula los cuantiles, modifica sus unidades, decide si un paciente está sano ni asigna tratamientos. Si una banda es inadecuada, un relato convincente no la corrige.

El paquete distingue `estimated_quantiles` de `empirical_adjusted_interval`. En el segundo caso, el resumen llama a los extremos límites inferior y superior ajustados, mantiene q50 sin modificar y no afirma que la expansión haya calibrado por separado los percentiles 10 y 90. Una observación requiere hora del evento y hora real de disponibilidad; se rechaza una disponibilidad anterior al evento o posterior al origen. La hora publicada ICULOS por sí sola no acredita el tiempo de respuesta real de un laboratorio.

La primera referencia es una **plantilla determinista verificable**. Después puede compararse con RAG local más FLAN-T5 y con otro adaptador de modelo por API sobre el mismo paquete, fuentes y criterios. La guía específica de OpenAI/Codex se mantiene aparte: tener herramientas/agentes para coordinar el trabajo no equivale a validar un sistema clínico.

## FLAN-T5 como candidato local

La ficha de `google/flan-t5-small` ofrece un modelo de 77 millones de parámetros, licencia Apache 2.0, uso mediante Transformers en CPU y español entre los idiomas declarados. Es una referencia pequeña de generación de texto guiada por instrucciones; no un predictor fisiológico ni un modelo clínico validado. Su tarjeta advierte limitaciones de evaluación en aplicaciones reales y necesidad de valorar riesgos específicos antes del uso aplicado. Que incluya español no demuestra fidelidad numérica, comprensión de unidades o calidad médica en español. [Ficha oficial del checkpoint](https://huggingface.co/google/flan-t5-small).

El trabajo original estudia ajuste por instrucciones y rendimiento en tareas generales; esas mejoras no validan automáticamente interpretar un panel clínico ni un astronauta concreto. Se fijarían checkpoint/revisión, tokenizer, longitud de entrada, generación y entorno antes del ensayo local. La memoria y latencia se medirían en esta máquina; no se han medido ni se promete que un tamaño concreto produzca relatos útiles. FLAN-T5 base podría ser un segundo tamaño si la referencia small resulta insuficiente, conservando idéntica tarea y evaluación. [Artículo original en JMLR](https://www.jmlr.org/papers/v25/23-0870.html), [documentación de integración Transformers](https://huggingface.co/docs/transformers/model_doc/flan-t5).

No existe ahora un conjunto autorizado de pares “historial + interpretación profesional” para entrenar un redactor clínico propio. Los CSV numéricos no sustituyen esos informes. Por ello no se propone SFT con diagnósticos inventados ni convertir automáticamente las salidas de otro modelo en etiquetas de especialistas. Si más adelante existen informes revisados y derechos de uso apropiados, su evaluación debe separarse por persona/episodio y del conjunto utilizado para redactar o ajustar ejemplos.

El contexto de 34 variables y cuatro horizontes puede ser largo para un modelo pequeño. El adaptador deberá medir tokens con su tokenizer y dividir por grupo biomédico y horizonte cuando sea necesario, conservando identificadores de hechos, ausencia de datos y referencias. No se truncará silenciosamente un paquete clínico para que quepa. La síntesis global también deberá declarar qué grupos no pudo cubrir.

## Qué aporta RAG y qué no garantiza

RAG puede suministrar definiciones, contexto de investigación y fragmentos verificables con su procedencia. La evaluación MIRAGE/MedRAG estudia distintas combinaciones de corpus, recuperación y modelos en preguntas médicas; sus resultados respaldan comparar esos componentes en lugar de asumir que cualquier RAG funciona igual. Un benchmark de preguntas no demuestra interpretación segura de historias individuales ni transferencia a medicina espacial. [Artículo primario MIRAGE/MedRAG](https://arxiv.org/abs/2402.13178).

Almanac incorpora recuperación de recursos médicos y evaluación por profesionales, mostrando un precedente de estudio con revisión especializada. No valida FLAN-T5 para este proyecto, ni convierte una cita generada en evidencia correcta por sí misma. Aquí habrá que comprobar qué fragmento sustenta cada afirmación y si corresponde al dominio, ensayo y pregunta. [Estudio primario Almanac](https://pmc.ncbi.nlm.nih.gov/articles/PMC10857783/).

El corpus inicial debería distinguir:

- Diccionarios, protocolos y notas de OSDR, UTMB y PhysioNet: qué variable se midió, matriz, unidad, frecuencia e incidencias.
- Registro del modelo congelado: selección, población, horizontes, métricas, calibración, máscaras, escasez y versiones.
- Investigación NASA/CSA y publicaciones primarias pertinentes: contexto científico, sin transformarlo en relaciones causales individuales.
- Documentación clínica autorizada y revisada, cuando sea pertinente: fecha, versión, ámbito de aplicación y licencia identificados, sin asumir acceso a contenidos no disponibles.

Cada fragmento recuperado requiere identificador, URI, localizador, versión/fecha y hash cuando proceda. La recuperación se filtra por dominio y finalidad. Los documentos son datos no confiables como instrucciones: un párrafo que ordena ignorar reglas o enviar información no debe convertirse en una acción del sistema.

## Prototipo implementado

`src/astronaut_twin/clinical_interpretation.py` no importa un LLM ni realiza llamadas de red. Expone:

| Elemento | Responsabilidad |
|---|---|
| `build_evidence_packet` | Construye un paquete para un episodio, origen, método y dominio; valida hechos y conserva motivos de abstención |
| `render_spanish_summary` | Produce un resumen determinista en español con números, procedencia, soporte y límites |
| `validate_generated_claims` | Compara afirmaciones estructuradas con valores/unidades/citas de campos existentes; rechaza hechos nuevos o alterados |
| `EvidenceRetriever` / `RetrievedPassage` | Contrato de un recuperador futuro y sus fragmentos con procedencia |
| `NarrativeModelAdapter` / `NarrativeDraft` | Contrato futuro de generación; no es un adaptador FLAN/API ya ejecutable |
| `make_model_request` | Serializa un pedido futuro; no transmite datos. La ubicación remota no se activa implícitamente |

### Recuperación local ya funcional

Se añadió `src/astronaut_twin/health_knowledge.py` y el constructor `scripts/build_health_knowledge_base.py`. Usan solo la biblioteca estándar y las fuentes documentales ya descargadas: verifican SHA-256 de cada HTML frente a `challenge_resources_v2.json`, permiten únicamente hosts primarios NASA/CSA y roles de recurso/seguimiento, y excluyen la ficha del reto y su mirror del corpus de conocimiento.

El índice `data/processed/health_knowledge_v2/` contiene **nueve documentos y 48 fragmentos**, de como máximo 120 palabras. Ocho páginas se extraen de article/main; Human Research Roadmap usa su bloque introductorio identificado explícitamente, dentro de su formulario estructural ASP.NET. No se usa todo el body como fallback ni se incluyen menús/formularios de búsqueda. Cada fragmento conserva URL, título, hash de fuente/texto y localizador en los bloques extraídos. Los localizadores no se presentan como líneas originales del HTML.

`LocalHealthKnowledgeRetriever.retrieve` satisface el contrato `EvidenceRetriever`: devuelve fragmentos con `Citation`, puntuación y dominio `space_health_background`. La búsqueda es **TF-IDF léxica local**, con normalización de acentos y un pequeño diccionario público de alias español→inglés, por ejemplo radiación→radiation y neutrones→neutron/neutrons. No son embeddings semánticos. Puntuación de similitud no significa veracidad ni confianza clínica. No devuelve resultados cuando no hay términos coincidentes o se pide otro dominio. El algoritmo, stopwords, alias, código e índice quedan identificados por metadatos/hashes; cambiar implementación exige reconstruir el índice.

La consulta de demostración “radiación neutrones salud astronautas” recupera documentación CANS, Radi-N2 y MOSFET con citas y hashes, guardada en `demo_radiacion.json`; no responde sobre ningún episodio ni ejecuta un LLM. Los documentos explican contexto espacial, no etiquetas de pacientes ni evidencia de rendimiento numérico. Su incorporación no cambia el predictor, los datos v1 o las particiones experimentales.

El paquete contiene dominio y dominio de entrenamiento, origen y referencia temporal, método/revisión, seudónimo del episodio, hechos por variable/horizonte, unidad, cuantiles disponibles, cobertura empírica por variable/horizonte, etiquetas/episodios de evaluación, historia propia y referencias por campo. El hash de seudónimo evita incluir la clave original directamente, pero **no anonimiza** la evidencia clínica ni garantiza que otras referencias no permitan vincularla.

No se copia la etiqueta `observed` futura de un CSV de evaluación retrospectiva al contexto prospectivo. Las observaciones históricas se suministran por separado y deben tener hora de toma y disponibilidad ≤origen. Una máscara falsa conserva `value=None`; un cero observado válido sigue siendo cero. Se rechazan episodios/orígenes mezclados, objetivos incompatibles con el horizonte, métricas ambiguas o de unidades distintas y cuantiles cruzados. Los originales del predictor no se alteran.

Los cuantiles incompletos se mantienen como ausentes, sin inventar la banda. Unidades pendientes, transferencia de dominio no validada, bandas fuera del soporte físico y evaluación ausente/escasa producen abstención explícita de interpretación. El mínimo de diez episodios es un criterio operativo de visibilidad de soporte, no un umbral clínico ni prueba de calibración al superarlo. El resumen puede seguir mostrando hechos numéricos válidos con su advertencia; no oculta la variable difícil.

El nominal q10–q90 se distingue de la cobertura empírica y de cualquier probabilidad de salud. No se interpreta una mediana futura elevada como una enfermedad ni una buena cobertura agregada como seguridad individual. El resultado siempre es descriptivo y requiere revisión profesional.

Las citas de hechos numéricos provienen de los artefactos de predicción/evaluación, no de un artículo genérico que mencione el analito. El validador exige la fuente que sustenta el campo concreto: citar un informe de métricas no basta para justificar un cuantil individual. No verifica automáticamente que un archivo externo sea auténtico, que una cita documental implique una afirmación médica ni que todo el texto libre de un modelo sea seguro. Esas comprobaciones siguen pendientes de implementación/revisión semántica y profesional; pasar el control estructurado no autoriza publicar el relato sin revisión.

## Comparación posterior sin inventar casos

Se utilizarán orígenes reales de los artefactos correspondientes a la versión actual, con historial disponible y fuentes registradas. Los ejemplos de desarrollo para adaptar una plantilla o un prompt se separarán de episodios reservados para evaluar redacción. No se crearán biografías, síntomas, diagnósticos ni informes de especialistas inexistentes. Una evaluación lingüística tampoco autoriza volver a seleccionar el predictor numérico con sus resultados de prueba.

| Variante | Qué se compara | Condición de evaluación |
|---|---|---|
| Plantilla determinista | Exactitud y claridad del relato de hechos | Referencia inicial, reproducible y sin generación libre |
| RAG + FLAN-T5 local | Si reformular con fuentes mejora utilidad/legibilidad conservando fidelidad | Mismo paquete, corpus versionado y tarea; medir contexto, memoria, latencia, citas y abstenciones |
| RAG + adaptador API | Diferencias de utilidad/fidelidad/coste operativo | Mismo contexto y corpus; transmisión solo dentro del alcance autorizado y condiciones de datos aplicables |

Las métricas deben incluir fidelidad de cada número/unidad/horizonte; ausencia de contaminación con resultados futuros; citas existentes y correctamente asociadas; cobertura de campos esenciales; abstención cuando falten datos; conservación de escasez y dominio; claridad en español; y detección de afirmaciones clínicas o causalidad no sustentadas. También se probarán instrucciones maliciosas en fragmentos recuperados y peticiones de inventar resultados. No basta puntuar similitud de texto o pedir a otro LLM que declare correcto al primero.

La revisión por un profesional deberá comprobar exactitud, omisiones potencialmente importantes, utilidad para inspeccionar evidencia y posibilidades de interpretación errónea, con criterios registrados y una comparación ciega cuando sea viable. Una validación clínica real requeriría su propio protocolo y población; todavía no se ha realizado revisión profesional de estos relatos ni medido desenlaces de salud.

## Privacidad y operación

Se separan el corpus documental público y la evidencia del episodio. El recuperador no debe indexar por defecto datos privados junto con documentación general; el modelo no recibe identificadores directos, notas completas o variables irrelevantes por comodidad. Los logs deben conservar versión/hash y mínimo contenido necesario, con acceso y retención definidos para el entorno real. Ejecutar localmente reduce transmisión, pero no resuelve acceso a archivos, logs, almacenamiento o licencias por sí solo.

En el prototipo no hay envío remoto. Un adaptador futuro debe respetar autorización y restricciones concretas de cada fuente; un artículo o dataset visible no autoriza enviar otros historiales privados. Una opción de API no debe reutilizar credenciales o acuerdos de MIMIC/eICU como permiso general. El contrato de transmisión y sus condiciones deben comprobarse antes de integrar datos sensibles reales.

## Estado y pruebas

`python -m pytest tests/test_clinical_interpretation.py -q`: **9 pruebas pasan**. Usan fixtures abstractos de software —variables `x/y`, reloj y unidad de prueba—, no pacientes ni casos médicos sintéticos. Verifican cuantiles/máscaras, ceros, ausencia y escasez, dominio/unidades/soporte, separación del futuro, coherencia temporal, citas por campo y bloqueo de transmisión remota implícita.

El recuperador añade **cinco pruebas**, para un total de **14 pruebas de esta capa sin fallos**: extracción sin navegación, fallback explícito del portal antiguo, tamaño/localizadores de chunks, hosts/roles permitidos, verificación de hashes, alias, orden determinista y cero resultados sin coincidencias. Los fixtures son texto documental de prueba, sin historiales ni desenlaces.

No se descargó ni ejecutó FLAN-T5 y no se midió aún la calidad de un modelo generativo o revisión por un especialista. Lo entregado es una referencia determinista utilizable, recuperación léxica documental local y el contrato para evaluar generación con fuentes conservando trazabilidad.
