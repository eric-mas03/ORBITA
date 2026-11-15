# Agentes y modelos de lenguaje para interpretar resultados

Revisión: 14 de noviembre de 2026. Propuesta de arquitectura, no evaluación de un modelo clínico ya entrenado.

## Recomendación

Primero separar la predicción numérica de la explicación. El predictor y el futuro ensemble producen cuantiles, unidades, horizonte, historial disponible y evidencia de evaluación. Una capa de datos construye un paquete verificable. Un recuperador consulta documentación NASA/CSA con URL, hash y localizador. El modelo de lenguaje redacta sobre esos datos y debe abstenerse de afirmar hechos que no puede respaldar. El especialista conserva la interpretación clínica.

La primera referencia es el resumen determinista implementado en `src/astronaut_twin/clinical_interpretation.py`. Permite medir qué aporta un modelo de lenguaje respecto a una salida factual reproducible. Los artículos NASA/CSA son conocimiento documental; no son informes clínicos emparejados con pacientes para entrenar un narrador supervisado.

Para el siguiente experimento, comparar esa plantilla con recuperación documental y un modelo pequeño local; después contrastar un modelo mediante API con las mismas herramientas de lectura y el mismo conjunto de evaluación. Los criterios son fidelidad a cifras/unidades/tiempo, citas verificables, omisiones relevantes, abstención apropiada, claridad en español y revisión por especialistas. ROUGE o fluidez no sustituyen a esos criterios.

## Dónde encaja Codex

Codex permite automatizar flujos mediante su SDK y ejecutar agentes programáticos. Es una opción técnica para agentes a medida; la integración no los convierte en especialistas clínicos validados. Para este proyecto lo usaría para implementar, auditar datos y mantener experimentos reproducibles. [Documentación del Codex SDK](https://learn.chatgpt.com/docs/codex-sdk).

La aplicación de apoyo al especialista necesita sobre todo herramientas acotadas: consultar historial conocido, predicciones ya calculadas, calidad/cobertura y documentos permitidos. Como decisión de diseño, propongo empezar con un flujo sencillo de lectura y validación, y añadir agentes especializados solo si una evaluación demuestra beneficio. El Agents SDK ofrece herramientas y coordinación de agentes; no es necesario dar a la capa clínica acceso a una consola ni permiso para modificar historias. [Guía de creación de agentes](https://developers.openai.com/tracks/building-agents).

Structured Outputs permite exigir un esquema de respuesta y facilita que la aplicación valide campos. La aplicación deberá comprobar además que los números, unidades y referencias se corresponden con el paquete de evidencia: respetar JSON Schema no garantiza veracidad ni calidad clínica. [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

No se han enviado observaciones clínicas a un proveedor externo, instalado credenciales ni ejecutado llamadas de pago en esta fase. Para una integración posterior habrá que fijar qué datos se envían, los controles de acceso y retención y el entorno de despliegue, además de evaluar el modelo concreto. El prototipo local funciona sin una clave API. [Inicio de integración con la API](https://developers.openai.com/api/docs/quickstart).

## FLAN-T5 y entrenamiento posterior

FLAN-T5 se conserva como candidato pequeño local, no como elección ya demostrada para español o medicina espacial. La comparación y fuentes del modelo se detallan en `docs/capa-interpretacion-clinica.md`. Antes de afinarlo hacen falta ejemplos de paquetes de evidencia y resúmenes revisados por profesionales, con particiones por persona y sin incluir resultados futuros en el contexto de una predicción. Los textos sintéticos podrían servir para comprobar formato o robustez, pero no se contarían como evidencia de desempeño clínico.

El orden de trabajo es: repetir predictores con más datos, medir el ensemble en una reserva independiente, cerrar sus límites y calibración, y después entrenar/evaluar la capa lingüística. El corpus de conocimientos y el contrato de evidencia pueden prepararse en paralelo.
