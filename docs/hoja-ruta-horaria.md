# Hoja de ruta acordada para predicción horaria

14 de noviembre de 2026. Estado: constructor, particiones, baselines y primera comparación de memorias GRU/Transformer ejecutados y evaluados. Siguen pendientes variantes adicionales, preentrenamiento, calibración y dashboard. Complementa el plan de decisiones. Las preferencias del usuario son objetivos de diseño; las propuestas siguientes necesitan evaluación.

**Comparación neuronal:** ambas variantes comparten encoders/fusión/readouts de atención y difieren en la memoria temporal. La selección se realizó en desarrollo; la referencia lineal sigue siendo mejor en ese criterio bajo el presupuesto inicial. Resultados y diagnóstico en [comparacion-neuronal-horaria.md](comparacion-neuronal-horaria.md). Las decisiones posteriores se estudian en desarrollo, sin afinar el modelo con la prueba publicada.

**Hito ejecutado:** 38.153 filas, 35.582 ventanas con contexto/etiquetas parciales, 600/150/100/150 episodios para entrenamiento/desarrollo/calibración/prueba y baselines de persistencia/regresión cuantílica. Evaluación y gráficas en [primer-bloque-horario.md](primer-bloque-horario.md). Se conservan las 34 variables y los cuatro horizontes; las celdas escasas quedan señaladas. La selección no usa prueba y la calibración se mantiene reservada. No hay Transformer horario ni adaptación espacial entrenados todavía.

## Qué queda definido

Para cada variable elegible se desean q10/q50/q90 a 1, 4, 8 y 24 horas. Una categoría biomédica puede contener varias variables de salida; cada variable conserva su unidad y métricas. No retirar una variable únicamente porque sea difícil de predecir: informar si hay evidencia suficiente, cómo se comporta y cuándo el modelo debe declarar falta de soporte.

Se quiere aprovechar información parcial. Una etiqueta ausente no aporta pérdida ni se imputa como un valor observado. Una entrada ausente lleva máscara y antigüedad. Una muestra no se descarta porque le falte otro grupo; si no hay ninguna etiqueta supervisada utilizable en el lote se omite la actualización. Los periodos de prueba permanecen reservados y no se incorporan al ajuste para cumplir literalmente con usar todos los datos.

La ventana inicial candidata es 24 h. Con observaciones originales cada 15 min serían 96 intervalos; con datos horarios son 24 intervalos. Se compararán duraciones alternativas en desarrollo, sin convertir una rejilla más fina en más observaciones reales.

## 1. Cerrar contrato de objetivos y comprobar viabilidad

El usuario ha elegido predecir el **intervalo que termina en t+h**. Para una rejilla de 1 h, el objetivo a 4 h está en (t+3 h,t+4 h]; para 15 min reales, en (t+3 h 45 min,t+4 h]. Los cuantiles representan incertidumbre sobre el valor futuro definido, no percentiles empíricos de las mediciones dentro del intervalo. Se conservará la semántica del valor publicado; una fila horaria no demuestra por sí sola que sea una media de mediciones crudas. En laboratorio, el objetivo solo existe cuando se ha medido: no se fabrica con interpolación o arrastre del último valor.

La primera auditoría cuenta parejas reales para cada variable y horizonte en las 1.000 secuencias horarias UCI descargadas, tanto con contexto completo de 24 posiciones como con historial parcial de hasta 24. Contar también cuántas secuencias/personas sustentan esas parejas, calidad y solapamiento. No exigir que las cuatro salidas temporales estén disponibles juntas.

Fuentes actuales y función:

- **UCI horaria:** base candidata para desarrollar y evaluar la tarea horaria. Sus laboratorios siguen siendo dispersos aunque exista una fila cada hora.
- **UTMB/Inspiration4/PBC:** medidas entre visitas, útiles para transferencia de encoders o tareas de otra frecuencia; no contienen etiquetas sanguíneas a 1/4/8/24 h por simple interpolación.
- **Sueño diario UTMB:** contexto de baja frecuencia y otra tarea; no se crean medidas intradiarias a partir de un total diario.
- **BIDMC:** registros de ocho minutos, insuficientes por sí solos para evaluar un horizonte de una hora con contexto de 24 h.
- **NHANES:** representación de perfiles de laboratorio, sin dinámica individual.

**Salida de fase:** matriz variable × horizonte × ventana con etiquetas y secuencias disponibles. Separar la disponibilidad numérica de la elegibilidad tras control de calidad/unidades.

### Auditoría inicial ya completada

Se revisaron las 1.000 secuencias clínicas descargadas: 38.153 filas, sin huecos ni duplicados de ICULOS. Para cada una de las 34 variables dinámicas se contaron etiquetas a los cuatro horizontes. Estos recuentos incluyen todo el subconjunto antes de separar entrenamiento/validación/prueba y antes de controles de calidad específicos.

| Horizonte | Orígenes con historial parcial de hasta 24 h | Orígenes con 24 slots completos |
|---|---:|---:|
| 1 h | 36.302 | 15.690 |
| 4 h | 33.302 | 13.552 |
| 8 h | 29.308 | 10.893 |
| 24 h | 15.076 | 3.306 |

Un origen de esta tabla no garantiza una etiqueta para todas las variables. Por ejemplo, glucosa a 24 h aporta 2.277 pares/606 secuencias con contexto parcial y 459 pares/124 secuencias con 24 slots completos. Fibrinógeno a ese horizonte aporta 64 pares/36 secuencias o 12 pares/cuatro secuencias respectivamente. Una celda escasa seguirá visible con su soporte; no se declarará evaluada solo porque existan otras variables en el mismo origen.

La tabla de 136 combinaciones y sus definiciones están en [viabilidad-horizontes-horarios.md](viabilidad-horizontes-horarios.md) y `outputs/feasibility/target_availability.csv`. Que existan 24 slots completos no significa que las 34 variables estén completas: se mantienen las máscaras.

## 2. Preparar series y máscaras

Definir intervalos que terminan en el origen t; todos los agregados de entrada usan exclusivamente información disponible hasta ese momento. Si una fuente suficientemente densa lo permite, resumir por variable e intervalo: media, mínimo, máximo, número de observaciones, cobertura del intervalo y edad de la última observación. Para sensores puede estudiarse variabilidad; no añadir estadísticas redundantes sin compararlas.

En la fuente horaria actual no se recuperan mínimos y máximos de los cuatro cuartos de hora. La primera resolución viable propuesta es 1 h; 15 min queda como variante condicionada a otra fuente original suficientemente densa y larga.

Mantener resolución y antigüedad por modalidad. Un análisis de sangre de hace 12 h puede aportar contexto con edad 12 h, si la regla de uso está definida; no se convierte en 48 mediciones de laboratorio. Diferenciar valor arrastrado, observación nueva y valor realmente ausente. Un cero fisiológico o de recuento tampoco equivale a ausencia.

Los historiales cortos pueden representarse con padding y máscaras; su uso y evaluación deben distinguirse de los de contexto completo. No desplazar orígenes ni usar observaciones futuras para alcanzar artificialmente 24 h.

**Salida de fase:** constructor temporal reproducible, controles de causalidad y cobertura por variable/cabeza/horizonte. Configuración de normalizadores y soporte físico por variable: no aplicar el mismo recorte a cero a variables que admitan negativos, como exceso de bases.

## 3. Fijar particiones y baselines

Reservar secuencias por identificador disponible y estudiar separación por hospital; declarar cuando la fuente no permite deduplicar personas entre ingresos. Crear particiones persistidas antes de ajustar escalas, seleccionar ventanas o preentrenar. Evaluar por separado sujetos nuevos y continuación temporal de sujetos conocidos si ambos usos interesan.

Primero persistencia y regresión cuantílica regularizada con características causales; después una referencia recurrente pequeña y la arquitectura agrupada actual. Mantener datos, particiones y métricas comparables. No equiparar buen resultado en UCI con capacidad demostrada en astronautas.

**Salida de fase:** experimento horario de referencia, sin transferencia, con q10/q50/q90 y conteos de etiquetas en cada celda evaluada.

## 4. Comparar arquitectura con agentes

- El agente biomédico revisa categorías, analitos, unidades y qué transferencia es razonable.
- El agente de datos revisa frecuencia, correspondencias temporales, calidad, ventanas y particiones.
- El agente de modelado diseña baselines y ablaciones, revisando causalidad y máscaras.
- La coordinación integra resultados y registra decisiones con el usuario.

Se conserva como hipótesis principal: encoders por categoría → representación común con tiempo/máscaras → memoria temporal → consultas de horizonte → cabezas de salida. A partir de ella comparar una decisión cada vez: fusión por atención entre grupos frente a concatenación, codificación temporal frente a sesgo relativo y atención completa frente a local cuando las secuencias lo justifiquen. Dimensión latente, capas y cabezas internas se ajustan con un presupuesto pequeño y curvas de aprendizaje.

**Salida de fase:** decidir qué componentes aportan mejora frente a referencias simples, con incertidumbre y estabilidad entre sujetos/dominios. Las cuatro consultas horarias deben seguir siendo independientes del conjunto de horizontes solicitado.

## 5. Preentrenar y estudiar transferencia

Comparar entrenamiento desde cero con dos rutas separadas: representación de perfiles mediante reconstrucción enmascarada (por ejemplo NHANES) y aprendizaje temporal en series clínicas compatibles. La máscara de corrupción artificial conserva el valor original como etiqueta solo cuando estaba observado; ausencia real no genera un objetivo de reconstrucción.

Transferir únicamente variables/encoders compatibles, conservando ensayo y dominio. Comparar encoders congelados y ajuste fino controlado si existe soporte. Las cabezas inmunitarias sin etiquetas externas no se consideran preentrenadas por disponer de hemogramas.

En el dominio espacial actual falta evidencia para validar estas cuatro horas de pronóstico en los paneles sanguíneos. La adaptación se evaluará en tareas respaldadas por sus visitas y, para forecasting horario espacial, requerirá una fuente con frecuencia suficiente. No se heredarán automáticamente ni métricas ni calibración del dominio terrestre.

**Salida de fase:** tabla de transferencia por bloque y dominio, con ganancias o pérdidas frente a entrenar desde cero. Ningún beneficio se presupone.

## 6. Evaluar gráficamente

Guardar predicciones por persona/secuencia, origen, instante objetivo, variable, unidad, horizonte, dominio, versión y máscara. Para cada variable y horizonte mostrar:

1. MAE de q50 y pinball frente a baselines, en unidades originales y con una escala normalizada documentada para comparar modelos.
2. Cobertura q10–q90 y anchura, además de calibración de cada cuantil; no sustituir esta evaluación por una media global.
3. Número de etiquetas y personas/secuencias, porcentaje de ausencia y resultados con contexto corto frente a completo.
4. Curvas de historial, observaciones futuras y bandas predictivas en orígenes seleccionados con una regla reproducible.
5. Mapas variable/cabeza × horizonte, mostrando como no evaluable una celda sin etiquetas.

Las métricas por cabeza complementan las individuales; una cabeza grande o una variable con unidades grandes no debe ocultar el resto. La dificultad se analiza en desarrollo; no retirar a posteriori de prueba los objetivos con peores resultados.

Calibración y selección usan conjuntos distintos de prueba. El intervalo nominal del 80 % sigue siendo marginal, no cobertura conjunta garantizada de todas las variables y horizontes.

## 7. Dashboard

Construirlo sobre los artefactos anteriores: selección de dominio, modelo, persona, variable, ventana y horizonte; historial y bandas; observaciones frente a predicciones; cobertura y errores; disponibilidad y antigüedad de datos; comparación de modelos y métricas por cabeza/horizonte. No mostrar como medido lo imputado o no observado ni ocultar una celda no evaluable.

Primero un visor local reproducible de resultados; publicación o integración con sensores sería una fase posterior. El diseño visual se concretará cuando existan artefactos horarios estables para inspeccionar.

## Orden inmediato

Semántica del objetivo confirmada → viabilidad auditada → calidad/particiones aplicadas → constructor y baselines ejecutados → primera evaluación terminada → revisar resultados en desarrollo y comparar arquitectura/contexto/preentrenamiento → calibración reservada → dashboard. La comparación de nuevos candidatos conserva particiones y evita seleccionar sobre prueba.
