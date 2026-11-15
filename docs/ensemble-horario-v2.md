# Ensemble horario V2

La campaña conserva los experimentos V1 y separa cinco funciones: ajustar componentes, seleccionar en DEV, ajustar pesos en META, ampliar intervalos en CAL y evaluar una sola vez FINAL. Los episodios son unidades de separación; los identificadores no garantizan personas distintas. El dominio horario procede de UCI, no de astronautas.

## Protocolo fijado antes de las reservas

Los componentes se eligen únicamente con el pinball macro estandarizado DEV: hasta tres familias diversas y la referencia lineal. El mejor individual DEV se conserva además como comparador. Las predicciones se expresan en unidades originales con estadísticas ajustadas solo en entrenamiento.

Se ajusta un único vector de pesos no negativos con suma uno, compartido por variables, horizontes y los tres cuantiles. Es un promedio de cuantiles; no se afirma que sean los cuantiles de una mezcla de distribuciones. META contiene 200 episodios distintos de los usados para ajustar o seleccionar componentes. Esta ronda usa un conjunto META disjunto, sin afirmar validación OOF. La función objetivo asigna igual peso a cada celda variable/horizonte con etiqueta y promedia pinball dentro de cada celda, normalizado por la escala de entrenamiento. Incluye L2 hacia pesos uniformes de 0,01, Adam de 0,05 y 300 pasos predeclarados. Se conservan todas las etiquetas reales disponibles de forma independiente, sin exigir salidas completas.

Después de congelar los pesos, CAL contiene 100 episodios previamente asignados. Para cada método reportado se calcula la misma ampliación no negativa de extremos por variable/horizonte con alfa 0,2. Con menos de 30 etiquetas se agrupan horizontes de la misma variable; si tampoco hay soporte, se registra ausencia de ampliación. Se exportan recuentos directos y agrupados de etiquetas y episodios. La mediana permanece idéntica. Las horas y etiquetas se solapan: estos extremos son intervalos empíricamente ajustados, sin garantía formal de cobertura del 80 % ni certificación de cuantiles marginales 10/90.

## Reserva final y trazabilidad

FINAL solo se empaqueta después de los tres estados congelados. Se aceptan exactamente los 300 identificadores con rol final del manifiesto de adquisición y de la auditoría. Se exige exclusión de todos los episodios V1 y de la confirmación ya consumida. La carpeta de adquisición también contiene entrenamiento y META: el empaquetador copia exclusivamente la lista final declarada a un directorio aislado y verifica cada tamaño y SHA256. No admite un NPZ externo para saltarse este proceso.

Cada etapa comprueba los hashes de preparación, roles, adquisición, auditoría, modelos, código de inferencia, política de calidad y configuración. El inicio final queda registrado antes de leer la reserva; la CLI rechaza una segunda evaluación automática. Se comparan ensemble, lineal y mejor individual DEV, con sus versiones crudas y ajustadas, sobre exactamente las mismas claves observadas. Si el mejor individual es lineal no se duplica.

Los nuevos resultados incluyen soporte de entrenamiento por celda, soporte de episodios conocido o desconocido, unidades pendientes de revisión y violaciones del soporte físico por método. Estas últimas se auditan sin recortar predicciones ni retirar etiquetas. `interval_type` distingue `estimated_quantiles` de `empirical_adjusted_interval`. Las columnas q10/q50/q90 se conservan por compatibilidad; su semántica ajustada se declara explícitamente.

## Ejecución

El snapshot de campaña incorpora el módulo y la CLI antes de META, con un hash propio adicional al código de entrenamiento. Se ejecuta la CLI del snapshot, manteniendo primero en PYTHONPATH su directorio src y la dependencia de árboles aislada usada durante el ajuste.

```powershell
python outputs/model_campaign_v2/workspace/scripts/evaluate_ensemble_v2.py --stage meta
python outputs/model_campaign_v2/workspace/scripts/evaluate_ensemble_v2.py --stage ensemble
python outputs/model_campaign_v2/workspace/scripts/evaluate_ensemble_v2.py --stage calibrate
python outputs/model_campaign_v2/workspace/scripts/evaluate_ensemble_v2.py --stage final
```

La ruta workspace por defecto de la CLI del snapshot debe pasarse explícitamente si no coincide con el directorio desde el que se ejecuta. Los resultados se escriben únicamente en el workspace V2: estados congelados, curva de pesos, soporte CAL, predicciones finales, métricas por variable/cabeza/horizonte, diferencias pareadas y gráficos. Los tests `tests/test_ensemble_v2.py` emplean datos sintéticos y no abren reservas.
