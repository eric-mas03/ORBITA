# Calidad aplicada a los valores horarios

14 de noviembre de 2026. Se implementó una política conservadora por celda, independiente de las métricas o dificultad de cada variable. Su finalidad es identificar ausencias y valores fuera de un soporte inequívoco, no decidir normalidad clínica. No se han ejecutado entrenamientos desde este módulo.

## Contrato implementado

`src/astronaut_twin/hourly_quality.py` expone:

```python
cleaned, flags = apply_quality(values, variables)
valid = np.isfinite(cleaned)
```

`values` es un array numérico `[filas,variables]`; `variables` contiene los nombres originales o claves `SEPSIS2019::...`. Se devuelve una copia float32 del mismo tamaño, con NaN en celdas inválidas y una lista de registros `{row,variable,reason,original}`. `row` empieza en cero dentro del array recibido; el constructor añade archivo/episodio y línea fuente. No se altera el array de entrada, ni se elimina una fila, cabeza o variable.

Una ausencia NaN original permanece ausente sin generar millones de incidencias de calidad. Infinito positivo/negativo sí produce incidencia; la representación original se conserva como texto JSON seguro. Un desbordamiento al representar float32 también se marca explícitamente. `load_quality_policy()` devuelve reglas y metadatos versionados de `config/hourly_quality_policy.json`.

## Reglas y decisiones de soporte

| Magnitud | Regla implementada | Qué se conserva |
|---|---|---|
| Concentraciones, recuentos, frecuencias, presiones y otras magnitudes no negativas | Se enmascaran valores <0 | Ceros y valores altos finitos, sin recorte por rangos clínicos o percentiles |
| BaseExcess | Puede ser negativo; no se impone umbral inferior ni superior clínico | Exceso/déficit de base firmado |
| Temp en Celsius | Solo se enmascara por debajo del cero absoluto, −273.15 °C | No se aplica una regla genérica `Temp<0`; Celsius no es una concentración |
| O2Sat y SaO2 | Intervalo inclusivo 0–100 % | Valores dentro del soporte, aunque sean clínicamente extremos |
| Hct | Intervalo inclusivo 0–100 % como fracción de volumen | Sin usar el rango habitual de hematocrito como criterio de exclusión |
| FiO2 | Soporte conservador 0–100, compatible con la unión de fracción y porcentaje | Sin decidir silenciosamente qué representación usa cada fila; no se divide todo por 100 |
| pH sanguíneo | Intervalo inclusivo 0–14 como soporte amplio de esta fuente | Sin convertir un intervalo de referencia habitual en criterio de exclusión |

Las reglas no incluyen límites de frecuencia cardiaca, presión o analitos basados en normalidad clínica. No se cortan colas por z-score ni se elimina una variable escasa. Tampoco se imputan valores, se rellenan etiquetas por arrastre ni se convierten magnitudes entre ensayos.

Los metadatos de las 34 variables conservan grupo, unidad documentada y estado de revisión. FiO2, Magnesium y Lactate mantienen `review_required_native_values_preserved`. Sus números válidos pueden usarse en baselines descriptivos en escala nativa de la fuente, pero la unidad ambigua impide una interpretación clínica o conversión automática a otro dominio. El soporte 0–100 de FiO2 permite detectar el valor 4000 bajo cualquiera de las dos representaciones; no demuestra que los otros valores estén armonizados.

El diccionario de unidades y procedencia sigue la [fuente primaria PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/). La política de soporte es una decisión explícita del proyecto; la fuente no certifica clínicamente estas reglas ni el modelo.

## Auditoría real del subconjunto descargado

Se ejecutó la función sobre las **1.000 secuencias, 38.153 filas y 34 variables dinámicas**, sin modificar archivos originales:

- Antes: **258.323** celdas numéricas finitas.
- Después: **258.322** celdas válidas.
- Una exclusión: **FiO2=4000**, `training_setB/p107096.psv`, ICULOS 2, fila cero-basada 1, motivo `percentage_above_100`.
- Las otras **33 variables tienen cero exclusiones** bajo estas reglas. Sus conteos individuales, incluidos los ceros, constan en `audit_snapshot.invalid_by_variable` del JSON de política.

Esto no demuestra que las restantes celdas estén libres de artefactos, que una celda sea una nueva toma de muestra o que el estado del paciente esté completamente observado. Es el resultado de aplicar este conjunto limitado de controles físicos. El snapshot corresponde al inventario descargado de 1.000 secuencias, no a toda la base disponible.

El constructor usa `valid` tanto para entradas como para etiquetas reales del intervalo terminal. Si falta FiO2 en una celda, no se enmascaran los valores válidos de HR, creatinina u otras variables del mismo origen. NaN de ausencia, valor inválido y padding se conservan como estados distintos en la representación/auditoría del constructor.

## Verificación y revisión de integración

`python -m pytest tests/test_hourly_quality.py -q`: **8 pruebas pasan**. Comprueban aislamiento de la máscara por celda, conservación de otras variables y ceros, negativos legítimos de BaseExcess/Celsius, límites inclusivos y hematocrito porcentual, unidades sin conversión, NaN frente a infinito, desbordamiento, copia sin mutación y rechazo de nombres no revisados.

La revisión de los módulos horarios disponibles comprobó que los historiales se recortan hasta el origen, los objetivos se buscan exactamente en `ICULOS(t)+h`, sin tolerancia y dentro de la misma secuencia, y las estadísticas de normalización de los baselines se ajustan solo con entrenamiento. Sus resúmenes mean/min/max proceden de las observaciones históricas disponibles en la ventana; no son agregados intrahorarios de 15 min recuperados de una fuente de 1 h.

Los orígenes supervisados se seleccionan según existencia de alguna etiqueta futura; las métricas resultantes están condicionadas a que haya mediciones. Esto es coherente con el experimento retrospectivo, pero no representa vigilancia fisiológica completa durante todas las horas. Las cabezas que recurran a una referencia por ausencia de etiquetas de entrenamiento deben identificarse como fallback, con sus conteos y origen de bandas, sin presentarse como cabezas entrenadas o calibradas. La disponibilidad de laboratorio en la hora publicada continúa siendo un supuesto retrospectivo explícito.
