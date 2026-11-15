# Primer entrenamiento técnico con datos reales

Resultado exploratorio sobre Inspiration4. No valida uso clínico, transferencia a nuevas personas ni una cobertura garantizada.

## Datos y arquitectura

109 variables en 13 grupos definidos por el agente biomédico. Encoders pequeños por grupo → fusión latente común → transformer temporal causal → 13 decoders transformer con consultas de horizonte → q10/q50/q90 ordenados. Dimensión latente 16, dos cabezas de atención y una capa; 78,855 parámetros.

Solo se usan grupos habilitados de CMP, CBC y Eve. Alamar se analiza en el EDA, pero sus 203 variables permanecen fuera del primer entrenamiento. Los valores se estandarizan por variable con 20 visitas-persona de entrenamiento; no se interpolan ni se rellenan con observaciones futuras. La pérdida asigna el mismo peso a cada grupo.

## Partición y disponibilidad

- Entrenamiento: 16 pares de historial/objetivo, con objetivos L-44, L-3 y R+45; horizontes 41, 44, 48 y 89 días.
- Validación: cuatro sujetos, R+45 → R+82, horizonte 37 días. Selección de checkpoint exclusivamente con esta visita.
- Prueba exploratoria: cuatro sujetos, R+82 → R+194, horizonte 112 días. Solo se emite después de disponer de R+82 y seleccionar el modelo.
- Los segmentos de lanzamiento y retorno se modelan separados, sin fabricar el tiempo entre ambos. Se supone disponibilidad inmediata del resultado en cada visita; los archivos no prueban esa latencia operacional.

Los pares históricos de entrenamiento son supervisión retrospectiva, no predicciones emitidas por un modelo ya entrenado en esas fechas. Los orígenes de validación/prueba respetan el momento de ajuste y selección del checkpoint. La normalización permanece congelada en entrenamiento.

Se ejecutaron 60 épocas con semilla 42, parada tras 15 épocas sin mejora y checkpoint de época 45. PyTorch 2.12.1, CPU. No se ajustaron hiperparámetros a la prueba.

## Resultado de la prueba

Promedios sobre variables en unidades estandarizadas con entrenamiento. No se promedian errores originales que mezclen mg/dL, células/uL y pg/ml. Las métricas originales por variable y horizonte están en CSV.

| Método | MAE normalizado ↓ | Pinball normalizada ↓ | Cobertura q10–q90 | Anchura normalizada |
|---|---:|---:|---:|---:|
| Persistencia | 0.9430 | 0.3193 | 65.7 % | 2.1718 |
| Transformer agrupado | 0.7715 | 0.2524 | 68.0 % | 1.8003 |

En esta ejecución el transformer reduce MAE un 18.2 % y pinball un 20.9 % frente a persistencia. Es una comparación preliminar en los mismos cuatro sujetos; no establece una mejora generalizable.

La banda nominal del 80 % cubre solo 68.0 % de los valores evaluados: **todavía no está bien calibrada**. Sus 435 parejas sujeto-variable son dependientes y no equivalen a ese número de personas independientes. No se calcula un intervalo de confianza de eficacia con esta cohorte.

La persistencia utiliza el último valor observado y bandas de residuos de entrenamiento agrupados, sin ajuste por horizonte. Ambos métodos aplican el mismo límite físico inferior cero a los cuantiles de estas 109 variables no negativas, antes de puntuar. Esto conserva el orden, pero no demuestra calibración. Los datos observados no se recortan. El decoder genérico produce salidas estandarizadas; la aplicación del soporte se realiza al invertir la escala en este pipeline.

## Límites y siguiente decisión

1. El horizonte de prueba de 112 días supera el máximo de 89 visto en entrenamiento. La arquitectura admite varias consultas simultáneas e independientes, verificadas por tests, pero esta prueba real solo evalúa un horizonte.
2. El EDA descriptivo utilizó la cohorte completa antes del experimento. Esta prueba es exploratoria, no una confirmación con conjunto intacto. No se seleccionan grupos ni analitos por su correlación con la prueba.
3. Solo hay cuatro personas y siete visitas por persona, sin estas medidas durante vuelo. Casi 79 mil parámetros frente a tan pocas trayectorias hace imprescindible buscar cohortes adicionales antes de interpretar la ventaja del modelo.
4. Falta validación por personas/misiones nuevas y calibración con soporte suficiente. No se afina cobertura mirando la prueba actual.
5. Antes de crecer el modelo: estudiar estabilidad por persona y lote, comparar un baseline cuantílico regularizado y hacer ablaciones de redundancias de hemograma. Los resultados de esas decisiones necesitarán evaluación externa.

## Reproducir

```powershell
python scripts/download_osdr.py --studies 575 569
python scripts/analyze_osd575.py
python -m pytest -q
python scripts/train_exploratory.py --epochs 80
python scripts/report_experiment.py
```

Checkpoint: `outputs/checkpoints/exploratory_twin.pt`. Configuración, hashes, versiones, supuestos y cortes: `outputs/training/run.json`. Predicciones completas: `outputs/training/predictions.csv`. El checkpoint solo debe cargarse desde fuentes confiables.
