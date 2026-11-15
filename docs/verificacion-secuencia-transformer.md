# Verificación de la secuencia temporal del Transformer

Revisión del 14 de noviembre de 2026. Se inspeccionaron el código actual y las formas reales de un forward del checkpoint seleccionado sobre dos ventanas completas del dataset. No se modificaron modelos, configuración o entrenamiento; no se lanzó un entrenamiento nuevo.

## Respuesta directa

**Sí: el Transformer recibe una secuencia completa de vectores latentes, uno por instante de la rejilla. El histórico no se comprime a un único vector antes de la atención temporal.** La implementación entrenada usa una rejilla horaria: una ventana completa tiene **N=24** posiciones. La arquitectura planteada para 24 horas con bins reales de 15 minutos tendría **N=96** posiciones. Esa segunda rejilla todavía no se ha entrenado ni preparado como dataset real.

`d_model=16` indica dieciséis coordenadas para representar **cada instante**; no dieciséis valores para representar toda la ventana. Son dos ejes distintos: longitud temporal `N` y anchura del latente `d`.

## Flujo comprobado

Con `B` ventanas, `F_g` variables en una categoría, `N` posiciones históricas, `d=16`, cuatro horizontes y `K_g` objetivos en la cabeza:

```text
Valores de categoría g         [B,N,F_g]
Máscara y edad de cada variable [B,N,F_g] cada una
              |
Concat de valor/máscara/edad    [B,N,3*F_g]
MLP de categoría g             [B,N,16]
              |
Fusión de las 13 categorías    [B,N,16]
              |
Encoder temporal Transformer   [B,N,16]
              |
Memoria completa de N estados  [B,N,16]
              |
4 consultas de horizonte       [B,4,16]
Atención cruzada a N estados   [B,4,16] por grupo
Cabeza de cuantiles            [B,4,K_g,3]
```

Las capas lineales/MLP operan sobre el último eje y conservan `B` y `N`. La fusión concatena categorías en el eje de características; no concatena ni promedia los instantes. El encoder temporal devuelve un estado contextualizado por cada posición. La reducción desde `N` estados a cuatro lecturas se produce **después**, cuando las consultas de los horizontes 1/4/8/24h leen la memoria por atención cruzada.

En la ejecución inspeccionada, la primera categoría tiene cuatro variables. Los hooks mostraron exactamente:

| Punto | Forma observada |
|---|---|
| Valores de primera categoría | `[2,24,4]` |
| Entrada MLP, con máscara/edad | `[2,24,12]` |
| Embedding de categoría | `[2,24,16]` |
| Concatenación para fusión | `[2,24,221]` |
| Salida de fusión | `[2,24,16]` |
| Entrada y salida del encoder temporal | `[2,24,16]` ambas |
| Memoria recibida por el decoder | `[2,24,16]` |
| Consultas y salida del decoder | `[2,4,16]` ambas |
| Cuantiles de primera categoría | `[2,4,4,3]` |

El ancho 221 procede de las 13 categorías, cada una con un embedding de 16 coordenadas y un indicador de presencia: `13*(16+1)`. Se proyecta a 16 en cada instante. Esto reduce el eje de características, preservando la secuencia de 24 posiciones. Una categoría enteramente ausente en un instante aporta un embedding anulado y su presencia falsa; no una medición inventada.

## Dónde están estas decisiones en el código

- `src/astronaut_twin/hourly_neural.py`, `NeuralData.batch`: construye las ventanas y actualmente contiene `torch.arange(24)`, con padding a la derecha.
- `src/astronaut_twin/models/twin.py`, `AstronautTwin.encode_history`: recibe `[B,T,F_g]`, aplica encoders, fusiona con `torch.cat(...,-1)` y devuelve memoria `[B,T,d]`.
- El mismo archivo, `AstronautTwin.forward`: crea consultas de horizonte y pasa **la memoria temporal completa** al decoder; la capa final produce `[B,H,K_g,3]`.

El modelo genérico usa `T=timestamps.shape[1]` y no fija la secuencia a 24. El preparador horario sí fija 24 posiciones, y sus controles de ventana están escritos para 24 slots horarios. Para 96 posiciones habrá que parametrizar preparador, duración de bin, política de contexto/padding y metadatos; no basta cambiar una constante del Transformer. No se ha hecho ese cambio en esta revisión.

Una ventana parcial conserva menos de 24 posiciones válidas y el resto son padding enmascarado. Eso permite aprovechar historias cortas, pero no equivale a disponer de 24 horas observadas. Con una rejilla completa de 15 minutos habría 96 **bins terminados**, no 97 puntos por incluir dos veces los bordes del intervalo.

## Qué implica una futura rejilla de 15 minutos

Se necesita evidencia de mediciones reales subhorarias y una definición de cada bin. No se pueden repetir las filas horarias existentes cuatro veces para declarar que el modelo vio 96 observaciones reales. Las estadísticas media/mínimo/máximo se calcularían dentro de cada bin solo si el origen permite recuperarlas; un valor publicado por hora no permite reconstruir sus extremos intrahorarios.

Las analíticas lentas conservan sus eventos, disponibilidad y antigüedad. Una analítica antigua puede representarse como contexto con edad conocida, pero no crea nuevas medidas o etiquetas cada 15 minutos. Una representación multirritmo puede mantener memoria rápida y lenta separadas, leídas por las mismas consultas, si las modalidades están enlazadas por persona/episodio.

La banda q10–q90 describe incertidumbre predictiva del objetivo terminal definido; no sustituye al mínimo/máximo medido dentro de un bin. La evaluación a 1/4/8/24h sigue exigiendo resultados reales válidos a esos horizontes o intervalos definidos de antemano.

## Atención causal interna frente a toda la ventana pasada

Actualmente cada estado temporal solo atiende su posición y las anteriores. Para pronosticar desde un origen `t`, es también una opción legítima permitir autoatención bidireccional **dentro de toda la ventana que ya era conocida en `t`**: una observación de `t-2h` puede ayudar a contextualizar la de `t-10h` sin introducir información posterior al origen. Esa alternativa no se ha implementado ni evaluado aquí.

La diferencia es entre causalidad de la información al emitir y causalidad de cada estado intermedio. Un encoder bidireccional de la ventana pasada satisface la primera si todas sus entradas cumplen `available_time <= t`; no representa necesariamente el estado que se habría calculado online en cada instante anterior.

Hay una precaución concreta en la API actual. `forward(...,cutoff_index=k)` excluye posiciones posteriores a `k` en la lectura del decoder, pero el encoder se ejecuta primero sobre el historial entregado. Con la máscara causal actual, los estados anteriores a `k` no pueden incorporar posiciones posteriores y ese corte es seguro. Si se elimina la máscara causal y se entrega una secuencia más larga que el corte, los estados anteriores podrían mezclar información posterior a `k`; enmascararla solo en el decoder **sería insuficiente**.

Antes de probar un encoder de ventana bidireccional habría que truncar o enmascarar las posiciones posteriores al corte **antes y durante su codificación**, por persona, además de aplicar padding y disponibilidad. La predicción desde un corte debe coincidir con la obtenida al proporcionar solo ese historial truncado. Debe mantenerse invariancia del pronóstico al alterar cualquier observación posterior al origen. No exigiríamos, en cambio, que todos los latentes de posiciones antiguas permanezcan iguales al añadir observaciones que ahora sí pertenecen a la ventana conocida: esa invariancia es propia del encoder causal y no del bidireccional de ventana.

## Prioridad propuesta para las siguientes ablaciones

1. Mantener registrada la distinción `N` frente a `d` y verificar cobertura/resolución real para el candidato de 96 bins. La verificación actual ya confirma que no se colapsa el eje temporal.
2. Comparar atención causal con atención sobre toda la ventana pasada, inicialmente en los 24 bins horarios reales. Conservar componentes, particiones, presupuesto, origen y salidas; corregir primero la semántica de corte descrita arriba. Es una comparación de una sola hipótesis.
3. Comparar la fusión lineal actual con atención entre categorías observadas, controlando máscaras y tamaño del latente. Eso modifica el eje de modalidades, distinto del eje temporal.
4. Estudiar memorias de distinta frecuencia si hay vitales y analíticas realmente enlazadas; después, sesgo relativo de tiempo o atención local frente a completa cuando la longitud efectiva y la resolución justifiquen la comparación. No abrir a la vez todas las combinaciones de resolución, arquitectura y pérdida.

Estas son propuestas de planificación, sin cambios de código ni nuevas métricas de entrenamiento. La comparación ya ejecutada GRU/Transformer cambia la memoria temporal; la GRU conserva lectura mediante atención a todos sus estados y tampoco comprime obligatoriamente la ventana al último estado antes del readout.

## Ensemble únicamente en la fase final

Un ensemble no reemplaza estas comprobaciones. Después de evaluar individualmente candidatos defendibles, sus pesos o un metamodelo necesitarían predicciones fuera de fold por episodio en el conjunto destinado a entrenamiento del ensemble. No se entrenan pesos sobre prueba ni se reutiliza el conjunto reservado para calibración como entrenamiento del metamodelo.

Se debe fijar la definición del ensemble probabilístico: promediar cuantiles ordenados es una combinación de estimaciones de cuantiles; no equivale en general a obtener los cuantiles de una mezcla de distribuciones. Si se usan pesos no negativos que suman uno, la combinación conserva el orden de los cuantiles componentes, pero no adquiere calibración automáticamente. La calibración posterior necesitaría su propio conjunto reservado y la prueba seguiría intacta para la evaluación final.

La elección entre combinación sencilla y metamodelo se deja para esa fase, condicionada al soporte independiente por episodio, variable y horizonte. No se ha entrenado ningún ensemble en esta revisión.
