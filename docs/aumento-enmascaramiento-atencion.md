# Datos adicionales, etiquetas ausentes y atención

Fecha: 14 de noviembre de 2026. Prioridad actual: aumentar datos reales compatibles y auditar su cobertura. No se ha ejecutado ningún entrenamiento nuevo, aumento de datos ni barrido de tipos de atención para esta revisión. Las recomendaciones siguientes son propuestas de diseño; las fuentes citadas no validan el modelo en astronautas.

## 1. Qué significa aumentar datos en este proyecto

El conjunto inicial OSD-575 aporta cuatro personas y siete visitas previstas. Recortar ventanas, generar ruido o repetir muestras no aumenta el número de personas independientes, ni crea observaciones durante el vuelo. La primera inversión debe ser localizar más sujetos/misiones con los mismos analitos, unidades, técnicas comparables, fechas y derechos de uso. Debe distinguirse ampliar la cohorte, añadir modalidades a las mismas personas y añadir puntos temporales: cada vía resuelve una carencia diferente.

Para combinar estudios se necesitan correspondencias explícitas de analito/ensayo, unidad, límites de detección, procesamiento, momento respecto a misión y disponibilidad temporal. Las variables no medidas en un estudio se conservan ausentes. Una matriz común con pérdida enmascarada permite aprovechar coberturas distintas; no permite identificar relaciones entre modalidades que nunca coexisten en personas o tiempos comparables. Registrar estudio y lote permite auditar confusión, pero introducirlos como covariables requiere evaluar también misiones/estudios nuevos.

La evaluación debe dejar fuera sujetos o misiones reales completos cuando el objetivo sea generalizar. Todo aumento se realiza después de esa división, solo dentro de entrenamiento. No normalizar con los datos reservados, no crear ventanas que salten de una persona a otra y no usar un generador entrenado sobre prueba.

## 2. Aumento artificial: opciones y restricciones

La literatura experimental compara transformaciones temporales, mezcla y generación, pero resultados de clasificación no se trasladan automáticamente a pronósticos cuantílicos de biomarcadores. [Iwana y Uchida](https://arxiv.org/abs/2007.15951) evaluaron doce técnicas en 128 conjuntos de clasificación y seis arquitecturas. Esa evidencia orienta candidatos; no demuestra beneficio para esta cohorte ni para calibración de q10/q90.

| Método candidato | Qué aportaría | Restricción propuesta aquí |
|---|---|---|
| Más ventanas de historias realmente observadas | Más pares de entrenamiento con distintos orígenes/horizontes | Mantener disponibilidad del modelo y de las etiquetas. Ventanas repetidas siguen correlacionadas. |
| Ocultar algunas entradas o modalidades de entrenamiento | Robustez a una cobertura de sensores/ensayos reducida | No crear etiquetas; conservar máscaras, recalcular edades causalmente y no borrar sistemáticamente toda la evidencia útil. Comparar con ausencia real. |
| Ruido compatible con incertidumbre analítica | Robustez frente a error de medida | Usar incertidumbre documentada por ensayo, unidades y límites; no elegir porcentajes arbitrarios ni perturbar etiquetas por defecto. |
| Recortar contexto pasado | Robustez a historias de longitud desigual | Solo recortar pasado; mantener el mismo instante objetivo y horizonte real y una longitud mínima definida. |
| Preentrenamiento con series reales adicionales | Aprender representaciones antes del ajuste específico | Particiones por sujeto/estudio y respeto a la causalidad. Más modalidades externas no son automáticamente el mismo dominio. |
| Mezcla de ejemplos o generación sintética | Regularización o exploración de software | Posponer hasta disponer de más soporte y validar transferencia real. No reportar sujetos sintéticos como cohorte real. |

Ocultar entradas y aprender representaciones con vistas aumentadas tiene precedentes: [TS2Vec](https://arxiv.org/abs/2106.10466) usa aprendizaje contrastivo jerárquico con vistas de contexto y evalúa representaciones en varias tareas. Es una motivación para estudiar preentrenamiento, no una autorización para usar contexto futuro en un predictor causal.

[Mixup](https://openreview.net/pdf?id=r1Ddp1-Rb) mezcla convexamente ejemplos y etiquetas como regularización. Aquí no asumiría que mezclar sujetos, dosis, fases de misión y biomarcadores conserva un estado fisiológico plausible, ni que mezclar etiquetas preserve los cuantiles de la población original. [TimeGAN](https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) propone un generador que busca preservar dinámica temporal mediante objetivos supervisados y adversariales; no elimina la necesidad de aprender esa dinámica a partir de una cohorte suficiente. Con cuatro personas, un generador podría reproducir sus particularidades sin aportar variación nueva defendible: esto es una preocupación de diseño, no un resultado medido aquí.

No propongo ahora permutar visitas, invertir el tiempo, deformar el calendario real de los análisis ni interpolar objetivos entre visitas. Esas operaciones modificarían la relación entre el origen, la fase de misión y el valor futuro que queremos pronosticar. Las transformaciones espectrales requieren además un tratamiento justificado de muestreo irregular; siete visitas de laboratorio no constituyen por sí solas una señal densa.

El primer experimento de aumento, cuando existan particiones suficientes, debería comparar ausencia de aumento frente a una sola transformación motivada por la adquisición real. Mantener los mismos sujetos reservados y medir pinball, cobertura y anchura por objetivo/horizonte. Ninguna ganancia media basta si aparece infracobertura importante. Este experimento queda pendiente; no se ha ejecutado.

## 3. Etiquetas ausentes: qué se entrena exactamente

Se añadió `grouped_masked_pinball_loss` en `src/astronaut_twin/models/twin.py`. Recibe diccionarios con las mismas claves: predicciones `[B,H,K,Q]`, objetivos `[B,H,K]` y máscaras booleanas `[B,H,K]`. La máscara puede variar por muestra, horizonte, objetivo y cabeza. Solo se puntúan posiciones observadas; NaN en posiciones no observadas no entra en el cálculo. Una etiqueta ausente no se sustituye por cero como objetivo real.

La función devuelve `GroupedPinballResult` con `loss`, `observed_counts`, `active_heads`, `reduction` y `should_step`:

- `equal_head`: calcula la media dentro de cada cabeza con etiquetas y luego da el mismo peso a esas cabezas activas. Una cabeza sin etiquetas no entra en el denominador. La cantidad relativa de observaciones de cada cabeza no determina su peso total.
- `per_target`: cada valor escalar observado de objetivo en una muestra/horizonte tiene el mismo peso, incluyendo sus cuantiles. Equivale a ponderar la pérdida media de cada cabeza por su número de posiciones observadas. Las variables o cabezas con más etiquetas pesan más. No significa igualar el peso de cada nombre de biomarcador independientemente de su cobertura.

La decisión es un criterio de entrenamiento, no un hecho biológico. Guardar el modo en la configuración del experimento y reportar también las coberturas de etiquetas por cabeza. Si una cabeza tiene solo una etiqueta, `equal_head` puede darle el mismo peso que otra con muchas; si una tiene muchas más variables, `per_target` puede hacerla dominante.

Si todo el lote carece de etiquetas, el resultado es `loss=None` y `should_step=False`. No se devuelve un cero artificial conectado al modelo. El entrenador debe omitir `backward()` y `optimizer.step()` para ese lote; también debe evitar contar ese lote como una actualización supervisada. Patrón de integración:

```python
optimizer.zero_grad(set_to_none=True)
result = grouped_masked_pinball_loss(predictions, targets, target_masks,
                                     reduction="equal_head")
if result.should_step:
    result.loss.backward()
    optimizer.step()
```

Una cabeza completamente sin etiquetas no recibe gradiente directo desde esta pérdida. Sus parámetros exclusivos mantienen `grad=None` después de limpiar gradientes con `set_to_none=True`. Sin embargo, los encoders y la memoria común sí reciben gradientes desde otras cabezas activas; incluso el encoder de una categoría sin etiquetas de salida puede aprender si sus entradas ayudan a predecir otros objetivos. Por tanto, una cabeza sin etiquetas puede emitir predicciones distintas tras actualizar la representación compartida, aunque sus parámetros exclusivos permanezcan iguales.

Dos matices del optimizador están verificados en tests:

1. En AdamW, `grad=None` permite saltar un parámetro. Un gradiente de cero puede seguir produciendo cambios por momentum anterior y weight decay. Usar `zero_grad(set_to_none=True)` evita arrastrar ceros en parámetros de cabezas enteras desconectadas. La diferencia entre cero y None está documentada por [PyTorch](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Optimizer.zero_grad.html).
2. Si falta solo una variable dentro de una cabeza activa, sus filas exclusivas de la capa de salida tienen gradiente local cero, pero pertenecen a un tensor que sí recibe gradiente para otras filas. AdamW puede cambiar las filas ausentes por decay o momentum. La pérdida no promete congelación por variable. Si se necesita congelación estricta habrá que separar parámetros por objetivo o diseñar explícitamente una política de actualización y de estado del optimizador; poner a cero el gradiente no basta.

Enmascarar permite entrenar con etiquetas parciales; no corrige el sesgo de selección cuando la decisión de medir depende del estado de salud ni resuelve automáticamente valores censurados por límites analíticos. Conservar las causas de ausencia. [GRU-D](https://www.nature.com/articles/s41598-018-24271-9) estudia precisamente máscaras e intervalos temporales para modelar ausencia informativa en series clínicas, en un contexto de clasificación distinto al de este proyecto.

## 4. Atención que usa el modelo actual

Se usa atención estándar multiproyección de producto escalar escalado, como la familia Transformer de [Vaswani et al.](https://arxiv.org/abs/1706.03762): `softmax(QKᵀ/√d + máscara)V`, en varias cabezas internas de atención. Esas cabezas de atención internas son distintas de las cabezas de salida por categoría biomédica.

| Bloque | Operación actual |
|---|---|
| Encoder de cada categoría | MLP; no atención entre sus biomarcadores. |
| Fusión de categorías | Concatenación de embeddings y presencia, seguida de proyección lineal; no atención explícita entre grupos. |
| Memoria temporal | Autoatención multica­beza causal completa sobre el pasado permitido, con padding excluido. |
| Consulta por horizonte | Tiempo real del objetivo y distancia desde el origen; sin valor objetivo futuro. |
| Decoder de cada grupo | Autoatención con máscara diagonal: cada consulta solo se atiende a sí misma. No mezcla los horizontes solicitados. |
| Decoder hacia historial | Atención cruzada desde cada consulta de horizonte a la memoria histórica, excluyendo posiciones posteriores al corte. |

La codificación continua del tiempo y las edades permite representar irregularidad. No es una atención relativa temporal explícita: no se añade actualmente un sesgo aprendido de diferencia de tiempo a cada par query/key. El caso diagonal tampoco es atención local temporal del encoder; son dos máscaras con propósitos diferentes. Los pesos de atención, si se inspeccionan en el futuro, no deben presentarse como efectos causales de los biomarcadores.

## 5. Comparaciones futuras, después de ampliar datos

Mantendría el modelo actual como referencia técnica y variaría una decisión cada vez, con particiones y presupuesto comparable:

- GRU pequeña con máscaras y edad, o variante GRU-D justificada, frente al encoder temporal Transformer. Determina si la atención aporta valor sobre una dinámica recurrente más simple.
- Atención entre tokens de categoría frente a concatenación/proyección. Evalúa interacción entre grupos sin perder sus máscaras ni su procedencia. Atención por canal es otra hipótesis distinta y necesitaría controlar el tamaño del panel.
- Sesgo relativo basado en diferencias reales de tiempo frente a codificación temporal actual. Debe conservar máscara causal y generalización a horizontes no vistos.
- Atención temporal local definida por días, frente a atención completa al pasado. Definirla por número de visitas puede representar duraciones muy diferentes en series irregulares. Con historias de una a tres visitas en el experimento inicial, esta comparación tiene muy poco soporte.

Se exigirán pruebas de invariancia al futuro, padding y conjunto de horizontes para cada variante, más evaluación probabilística por sujeto/estudio/horizonte. No se ha implementado ni probado ahora ninguna de estas variantes: la prioridad sigue siendo aumentar sujetos y cobertura temporal reales.
