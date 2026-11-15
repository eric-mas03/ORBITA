# Equipo y fases del proyecto

## Avance ejecutado

La ampliación posterior ha añadido datos clínicos de análogos UTMB, sueño diario, microbioma Mars500, procesados de otras cohortes de vuelo y fuentes terrestres para preentrenamiento. Se reforzaron pérdidas enmascaradas e integración del entrenador. La suite vigente es de 38 pruebas aprobadas; los 22 casos del primer hito quedan como registro histórico. No se han ejecutado nuevos entrenamientos ni comparaciones de atención durante esta ampliación. Véase [ampliacion-y-siguiente-experimento.md](ampliacion-y-siguiente-experimento.md).

Tras la instrucción de continuar con encoders por categoría y espacio latente común, los tres agentes han completado descarga/auditoría de OSD-575 y OSD-569, agrupación biomédica, interpretación del EDA y modelo PyTorch. La coordinación integró el análisis reproducible, entrenamiento técnico y comparación con persistencia. La suite vigente contiene 22 pruebas aprobadas. La arquitectura del usuario es suficiente: ya no se requiere el TFM para continuar.

Resultados y limitaciones vigentes en [primer-entrenamiento.md](primer-entrenamiento.md). El siguiente trabajo científico es ampliar cohortes y contrastar baselines/ablaciones, con evaluación externa y calibración; el prototipo actual no acredita predicciones clínicas.

## Agentes activados para el análisis inicial

| Agente | Especialidad | Entregable | Responsabilidad |
|---|---|---|---|
| `data_audit` | Ingeniería y auditoría de datos | `analisis-datos.md` | Verificar fuentes, acceso, estructura temporal y viabilidad del conjunto de datos. |
| `forecasting` | ML probabilístico y series temporales | `plan-modelado.md` | Definir ventanas, objetivos, cuantiles, baselines, validación y calibración. |
| `health_architecture` | Análisis biomédico y arquitectura de software | `arquitectura-salud.md` | Delimitar variables de salud y exposición, interpretación y módulos del sistema. |
| Coordinación principal | Integración del proyecto | README y este documento | Contrastar conclusiones, registrar decisiones e incorporar posteriormente el TFM. |

Los agentes son asistentes de análisis y programación. Su trabajo no equivale a revisión por profesionales sanitarios ni a validación clínica. Los agentes se han activado para esta fase; no son procesos de vigilancia permanente.

## Especialidades de implementación previstas

Se asignarán tareas concretas de programación cuando la auditoría determine el alcance viable:

- Ingeniería de datos: adquisición reproducible, manifiesto de fuentes, diccionario, controles de calidad y transformación temporal.
- Ingeniería ML: baselines, entrenamiento, evaluación por horizonte, calibración y artefactos reproducibles.
- Backend/MLOps: contratos de predicción, versionado de datos/modelos y registro de procedencia.
- Frontend y visualización: historial individual, bandas predictivas, calidad de datos y comparación con observaciones.
- Revisión biomédica: interpretación de unidades y variables, límites de uso e hipótesis que requieren validación externa.

No se fija aún framework ni modelo avanzado: dependerán de datos y del TFM que aporte el usuario.

## Fases y criterios de salida

1. **Fuentes y viabilidad.** Inventario verificable con acceso, licencia o condiciones, sujetos, granularidad, cobertura y variables. Separar dato confirmado, candidato y no disponible.
2. **Análisis exploratorio sobre datos reales.** Tipos, unidades, duplicados, secuencias, faltantes, frecuencia efectiva, disponibilidad temporal, variabilidad por sujeto/misión y ventanas utilizables. Elegir objetivos y horizontes con evidencia.
3. **Baseline reproducible.** Separaciones temporales y por sujeto según escenario; último valor y referencias simples; primeras métricas por objetivo y horizonte sin fuga de información.
4. **Forecasting probabilístico.** Estimar q10/q50/q90, evaluar precisión y cobertura, detectar cruces y comparar con baselines. Pasar a modelos complejos solo con mejora demostrable.
5. **Prototipo individual actualizado.** Integrar observaciones, estado por astronauta y predicciones con fecha de origen, horizonte, unidad y versión de modelo.
6. **Adaptación al TFM.** Comparar arquitectura, modalidades y método de fusión del trabajo de referencia con lo que los datos realmente permiten.

## Preguntas abiertas que resolverá la evidencia

- ¿Hay series longitudinales humanas individuales con suficientes medidas repetidas?
- ¿Se busca pronosticar al mismo astronauta tras observar su historial, generalizar a otros astronautas, o ambos?
- ¿Qué variables se observan y cuándo están disponibles para el sistema?
- ¿Se pueden sincronizar las modalidades por identidad y tiempo de forma legítima?
- ¿Qué horizontes tienen sentido según frecuencia y continuidad real de las medidas?
- ¿Qué elementos del TFM deben conservarse cuando se facilite su documentación?

## Próxima acción de implementación

Tras identificar y obtener una fuente apta, crear un inventario y un informe exploratorio reproducible. Si no hay series suficientes, proponer una alternativa explícita: seguimiento descriptivo, modelado de cambios entre visitas o demostración sintética separada de la evaluación real.
