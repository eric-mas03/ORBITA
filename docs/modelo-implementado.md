# Arquitectura PyTorch implementada

Estado: componente ejecutable y verificado con tensores de prueba. No está entrenado y no hay evidencia de precisión clínica. Los datos OSD-575 identificados por la auditoría tienen cuatro sujetos y siete visitas previstas irregulares, sin visitas durante el vuelo; la implementación no elimina esa limitación ni establece que un Transformer sea estimable con ellos.

## Estructura

`src/astronaut_twin/models/twin.py` implementa:

1. Un encoder MLP por categoría biomédica configurable. Recibe valores, máscara de observación y edad desde la última medición disponible. Cada categoría produce un vector del mismo tamaño.
2. Una proyección de fusión de los vectores y de indicadores explícitos de presencia de grupo. Un grupo completamente ausente aporta un vector cero. No se presenta el sesgo del encoder como una medición.
3. Un encoder Transformer temporal con atención causal, máscara de padding y representación continua del tiempo real transcurrido. No supone muestreo equidistante.
4. Un decoder Transformer independiente por grupo de objetivos, con consultas construidas a partir del origen y de los horizontes solicitados, y atención a la memoria histórica permitida. La autoatención entre consultas usa una máscara diagonal: cada horizonte solo atiende su propia consulta, además de la memoria histórica. Así, en evaluación, predecir un horizonte dado no depende de qué otros horizontes se soliciten ni de su orden. No recibe valores objetivo futuros; no es un decoder autorregresivo con teacher forcing.
5. Una capa por grupo que produce tres parámetros por objetivo. Se transforma en `q10 = q50-softplus(a)` y `q90 = q50+softplus(b)` para impedir cruces. Esta restricción no demuestra calibración.

Hay cabezas por grupo y una salida diferenciada por variable dentro de cada cabeza. No hay un Transformer separado por cada variable individual. Los grupos y el orden de sus variables quedan en la configuración; no se fijan nombres de biomarcadores en el código.

## Contrato de uso

`TwinConfig(input_groups, output_groups, d_model=32, nhead=4, num_layers=1, decoder_layers=1, dropout=0.1, time_scale=1.0)` configura tamaños. `input_groups` es un diccionario de nombre de grupo a número de variables; `output_groups`, de nombre de cabeza a secuencia ordenada de nombres de objetivos únicos.

`AstronautTwin.forward(values, observed, ages, timestamps, horizons, padding_mask=None, cutoff_index=None)` devuelve un diccionario por grupo con tensores `[B,H,K,3]`. `K` sigue el orden de la configuración y el último eje sigue `(0.1,0.5,0.9)`.

| Argumento | Forma y significado |
|---|---|
| `values` | Diccionario de tensores flotantes `[B,T,F_grupo]`, preprocesados con parámetros aprendidos solo en entrenamiento. NaN permitido únicamente en valores no observados. |
| `observed` | Diccionario booleano de las mismas formas, `True` si hay valor disponible. No confundir con padding. |
| `ages` | Diccionario flotante de las mismas formas: edad no negativa desde la última medida disponible, en la unidad común de tiempo. Debe ser finita fuera de padding; la política de edad cuando nunca hubo medición la define el preparador. |
| `timestamps` | Flotante `[B,T]`, orden cronológico no decreciente por sujeto, unidad común. Usar tiempos relativos, evitando Unix float32 por precisión. |
| `horizons` | Flotante `[H]` común o `[B,H]` individual, finito y estrictamente positivo. Misma unidad que timestamps y ages. |
| `padding_mask` | Booleano `[B,T]`, `True` para padding únicamente a la derecha. Cada sujeto debe conservar al menos un instante. |
| `cutoff_index` | Índice entero común o tensor long `[B]`; por defecto el último instante válido. Excluye memoria posterior de las cabezas. |

`time_scale` es una constante positiva finita que divide tiempos y edades antes de codificarlos: por ejemplo, si la unidad son días, puede mantenerse en 1. Debe guardarse en el checkpoint junto con el resto de la configuración. Tensores y modelo deben compartir dispositivo y dtype compatible; los tiempos de entrada deben estar en el mismo dispositivo que los valores.

Una categoría sin datos debe existir en los diccionarios con máscara completamente falsa. El preparador debe abstenerse ante historias insuficientes según una política explícita: el modelo puede calcular una salida con grupos ausentes, pero ello no demuestra que sea fiable.

## Responsabilidad de prevención de fuga

La memoria temporal en posición `k` no atiende posiciones posteriores y la cabeza no atiende posiciones posteriores a `cutoff_index`. La representación de tiempo usa el primer instante de la serie, no el final del historial completo, evitando que una fecha futura altere el origen temporal de posiciones anteriores.

Esto no corrige fugas previas al modelo. El pipeline debe filtrar por `available_time`, realizar joins causales, definir la ventana con corte real, excluir normalizaciones retrospectivas (por ejemplo porcentajes derivados con información preflight futura), usar escaladores entrenados dentro de cada fold y mantener sujetos separados cuando corresponda. El argumento `cutoff_index` no convierte valores construidos con información futura en valores válidos.

Los objetivos ausentes se excluyen mediante `masked_pinball_loss(prediction, target, observed)`, donde `target` y `observed` son `[B,H,K]`. La función promedia sobre objetivos y cuantiles observados, permite NaN en etiquetas no observadas y rechaza lotes sin etiquetas puntuables. Los pesos entre grupos y horizontes se definen en el entrenador; esta función no los fija.

Los errores de calibración y cobertura no se resuelven imponiendo orden a los cuantiles. La salida es marginal por objetivo/horizonte, no una distribución conjunta ni un simulador causal de intervenciones. El modelo tampoco impone por ahora restricciones fisiológicas entre variables.

## Verificación

Con Python 3.13.5, PyTorch 2.12.1 CPU y pytest disponibles, se ejecutó `python -m pytest tests/test_model.py -q` con `src` en `PYTHONPATH`: nueve pruebas pasaron.

Se comprobaron formas, cuantiles ordenados y gradientes hasta encoder y decoder; invariancia del pronóstico pasado y del latente causal al modificar valores y fechas futuras; equivalencia con truncar el historial; exclusión de padding y grupos ausentes con NaN; sensibilidad a tiempo irregular y edad; pinball enmascarada con gradiente válido; rechazo de entradas temporales inválidas; invariancia ante traslación del origen absoluto de tiempo; independencia respecto al conjunto y orden de horizontes solicitados; y equivalencia entre un batch con padding/cortes distintos y predicciones individuales de historiales truncados.

Los tensores sintéticos se limitan a estos tests de software: no son entrenamiento ni validación biomédica. Baselines, particiones por sujeto y evaluación de datos reales deben preceder cualquier conclusión sobre esta arquitectura.

La implementación utiliza las API oficiales [TransformerEncoder](https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerEncoder.html) y [TransformerDecoder](https://docs.pytorch.org/docs/2.12/generated/torch.nn.TransformerDecoder.html). La justificación del protocolo de validación está en `docs/plan-modelado.md`.
