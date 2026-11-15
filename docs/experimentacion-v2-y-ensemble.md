# Experimentación V2 y ensemble posterior

Esta vuelta repite las once familias de modelos sobre una ampliación real y auditada de datos horarios terrestres. Los recursos oficiales adicionales del reto recuperados son documentación, no nuevas mediciones; la expansión de entrenamiento procede de PhysioNet Challenge 2019 y se conserva como dominio UCI. No se presenta como una cohorte nueva de astronautas ni como validación del gemelo en vuelo.

## Datos y particiones de esta campaña

Se conservan los 600 episodios iniciales de entrenamiento, 150 de desarrollo y 100 de calibración. Se añaden **1.000 episodios nuevos de entrenamiento**, **200 para META** y **300 finales completamente nuevos**. Los 150 de prueba exploratoria V1 y los 300 de confirmación ya consumida no participan en la nueva selección o evaluación final.

El paquete no final tiene 2.050 episodios: entrenamiento 1.600, desarrollo 150, META 200 y calibración 100. Sus orígenes elegibles son respectivamente 56.669, 5.324, 7.081 y 3.290. Los 300 finales permanecen como archivos originales con roles y hashes; no se empaquetan ni se evalúan antes de congelar la selección, pesos y calibración del ensemble.

Se mantienen 34 variables, 13 categorías, horizontes 1/4/8/24h y q10/q50/q90. Se utilizan hasta **24 slots horarios reales**, con contexto parcial permitido y máscaras por variable/horizonte. Los modelos conservan la secuencia de latentes `[B,24,d]`. No se crean 96 observaciones de quince minutos repitiendo filas horarias; una fuente realmente subhoraria exigiría otro preparador y otra auditoría temporal.

## Aislamiento y trazabilidad

La ejecución está aislada en `outputs/model_campaign_v2/workspace/`. Se copian snapshots de código, grupos y política de calidad; V1 y sus artefactos canónicos permanecen intactos. `prepared.json` vincula hashes del NPZ auditado, auditoría, adquisición, roles y fuentes de entrenamiento. Los workers comprueban sus fuentes congeladas al comenzar o reanudar.

El snapshot de grupos conserva semántica y unidades, pero elimina los recuentos antiguos `observed_support`, estados de elegibilidad y evidencia de la cohorte V1 para no atribuirlos a V2. El archivo de origen se conserva como procedencia. El soporte V2 se audita por separado usando sus etiquetas reales.

Las vistas de ajuste filtran **orígenes y objetivos de entrenamiento/desarrollo antes de construir ventanas**. Las filas de otras particiones permanecen archivadas en el paquete fuente; nunca entran en el escalado o los batches de ajuste. Los preparadores calculan escalas sobre filas únicas de episodios de entrenamiento. Aplicar una transformación fija a una fila no habilita utilizarla como predictor de otra persona o episodio.

## Once familias y presupuestos comparables

Se repiten persistencia; regresión cuantílica lineal; GRU con readout de atención; Transformer agrupado; boosting cuantílico propio y por grupo; bosque cuantílico; residuales con memoria causal común, memoria bidireccional del pasado y memoria temporal por grupo; y el control MLP de resúmenes propios.

La ampliación aumenta los orígenes de entrenamiento de 21.661 a 56.669. Antes de ajustar se fija el presupuesto `ceil(épocas_V1 * 21.661 / 56.669)`, procurando comparar una cantidad aproximada de actualizaciones del optimizador, no el mismo número de épocas:

| Familia neuronal | Máximo V1 | Máximo V2 |
|---|---:|---:|
| GRU y Transformer desde cero | 20 | 8 |
| Tres arquitecturas residuales | 15 | 6 |
| MLP residual | 30 | 12 |

Se conservan tasas, tamaños, optimizadores, clipping, minibatches y paciencia de referencia. Los modelos residuales incluyen la época 0, idéntica a la lineal V2 congelada, como candidato. Si no mejoran desarrollo, conservan esa identidad. La lineal mantiene 80 épocas máximas y sus dos tasas; los árboles mantienen las dos configuraciones por familia.

Este protocolo **no identifica un efecto causal de añadir datos**: cambian cohorte, número de épocas y ejecución. Permite comparar los candidatos V2 en las mismas etiquetas de desarrollo con un presupuesto declarado y repetible.

## Ejecución y reanudación

Primero se ajustan los baselines nuevos, necesarios como offset de los residuales. Después se programan cuatro procesos de dos threads. La memoria por grupo arranca primero; las tres arquitecturas residuales tienen directorios independientes durante el ajuste y se consolidan al finalizar, evitando conflictos entre workers. Se aprovechan los ocho núcleos físicos disponibles sin dejar OpenMP a un número automático de threads.

Cada época neuronal conserva el estado actual, mejor estado, optimizador, generadores aleatorios, época y paciencia. Una reanudación verifica la firma de datos/configuración y continúa desde la última época completa. Los árboles conservan checkpoints por celda. Las exportaciones incompletas se reconstruyen desde estados seleccionados, evitando duplicar filas. Los logs persistentes permiten seguir la campaña aunque una sesión de herramientas termine.

El primer intento de lanzamiento desacoplado se cerró antes de crear workers; no produjo un modelo. La ejecución efectiva se inició bajo una sesión persistente, con `persistence.npz` y escaladores escritos a las 13:36 del 14 de noviembre de 2026. Cada familia marca `complete.json` y el conjunto final `individuals_complete.json`.

## Separación de fases

1. **Individuales:** únicamente entrenamiento y desarrollo. Se exporta solo DEV y se comparan todas las familias en las mismas claves, observaciones y escalas, verificando finitud y orden de cuantiles.
2. **Congelación de componentes:** ranking por pinball estandarizada macro variable/horizonte, hasta tres familias diversas y referencia lineal. Se guarda también el mejor individual de desarrollo. El archivo de selección vincula hashes de modelos y evidencia; no se presenta una posición en el ranking como prueba de superioridad clínica.
3. **META:** inferencia de componentes ya congelados en los 200 episodios disjuntos. Ninguno se utilizó para ajustar componentes o elegir sus checkpoints.
4. **Ensemble:** aprendizaje de pesos globales no negativos que suman uno, compartidos entre variables, horizontes y cuantiles, con regularización hacia pesos uniformes e hiperparámetros predeclarados. Se congelan antes de abrir calibración.
5. **Calibración:** ajuste separado con los 100 episodios CAL. Se calibran ensemble, referencia lineal y mejor individual de desarrollo, deduplicando métodos si coinciden. No se seleccionan hiperparámetros mirando CAL.
6. **Final:** solo después de los freezes previos se empaquetan exactamente los 300 IDs de rol final, se comprueba su procedencia y se realiza una evaluación congelada. Se reportan versiones raw/calibradas de ensemble, lineal y mejor individual, sin reentrenar a partir de ese resultado.

Esta vuelta utiliza **stacking con META reservado**, no predicciones OOF. Es válido porque META es disjunto del ajuste de componentes y se necesita para entrenar el ensemble. El plan OOF queda registrado para otra campaña: folds por episodio, modelos y selección de cada fold sin sus etiquetas reservadas y escalado ajustado solo dentro del entrenamiento de cada fold. No se simula OOF reutilizando predicciones de ajuste ni se añade su coste esta vuelta.

## Cuantiles y calibración

Una combinación convexa de cuantiles ordenados conserva su orden, pero no equivale en general a calcular los cuantiles de una mezcla de distribuciones. Tampoco adquiere cobertura del 80% por promediar modelos. La banda es marginal por objetivo/horizonte, no conjunta para una trayectoria completa.

La calibración propuesta utiliza expansión no negativa de bandas a partir de errores de calibración, con soporte por celda y fallback documentado si escasean etiquetas. La dependencia entre orígenes y la identidad no confirmada entre distintos ingresos impiden prometer una garantía clásica de cobertura por puntos intercambiables. Se reportan coberturas, anchuras, recuentos y estabilidad por episodio; no se convierten los huecos de etiquetas en aciertos.

## Comandos y salidas

El runner es `scripts/run_model_campaign_v2.py`. Las fases son `prepare`, `individuals`, `select`, `meta`, `ensemble`, `calibrate` y `final`; `status` consulta marcadores sin entrenar. El archivo de configuración es `config/model_campaign_v2.json` y el preparado efectivo queda congelado dentro del workspace.

Los modelos mantienen aliases de V1 dentro del **workspace nuevo**: `outputs/hourly_baselines`, `hourly_neural`, `hourly_trees`, `hourly_architecture_trials` y `hourly_residual_control`. Así se reutilizan loaders de artefactos sin tocar los resultados anteriores. `components_frozen.json` y su copia `outputs/model_campaign_v2/individual_selection.json` determinan los únicos modelos que pueden pasar a META.

Los tests del engine cubren aislamiento de snapshots, auditoría/hash de adquisición, filtrado de roles, cálculo del presupuesto, omisión de actualizaciones AdamW de cabezas sin etiquetas con aprendizaje compartido activo, serialización reanudable y selección diversa exclusivamente con evidencia DEV. Los tests de ensemble y evaluación son independientes. Los resultados de rendimiento V2 se añadirán únicamente tras completar sus etapas reales.

## Individuos V2 terminados y selección congelada

Los once métodos terminaron sus ajustes. Cada uno se evaluó en las mismas **117.844 etiquetas de desarrollo**, con observaciones y escalas comunes comprobadas antes de seleccionar. Se generaron 1.296.284 filas DEV entre todos los métodos; son pares correlacionados, no personas independientes.

| Método | Pinball estandarizada macro DEV V2 |
|---|---:|
| Residual con memoria temporal por grupo | 0,150450 |
| Residual causal común | 0,150557 |
| Residual bidireccional de ventana pasada | 0,151002 |
| Bosque cuantílico | 0,151049 |
| MLP residual de resúmenes propios | 0,151896 |
| Regresión cuantílica lineal | 0,153027 |
| Boosting con características del grupo | 0,153616 |
| Boosting con características propias | 0,153942 |
| Transformer agrupado desde cero | 0,173908 |
| GRU con lectura de atención | 0,175625 |
| Persistencia | 0,180153 |

La mejora del mejor individual respecto a la lineal **dentro de V2** es aproximadamente 1,68 %. No se utiliza esta tabla para atribuir una mejora a añadir datos frente a V1: cambió la escala de entrenamiento y se necesita el informe de comparación con una escala de reporte compartida. La clasificación tampoco sustituye cobertura, anchura, soporte físico o incertidumbre por episodio.

Por la regla predeclarada de diversidad se congelaron lineal, residual temporal por grupo, bosque cuantílico y MLP residual. La selección y todos sus hashes están en `components_frozen.json`, SHA-256 `0130dcc7b41a4dd71d747df987f68e94aba9649797ce50a1cd43b65151c53f33`. El preparado vinculado tiene SHA-256 `891db1013f7d5809bfe251d2d32c86a6a88431fe4a81bb36a4b1b69bf9fa35cb`.

Se verificó que los diez artefactos de evidencia vinculados por la selección V1 conservaron sus hashes originales tras esta campaña. Ningún modelo o resultado V1 se sobrescribió.

La ejecución posterior queda en un único evaluator, para evitar carreras de acceso a reservas. Usa el entrypoint **copiado dentro del snapshot** `outputs/model_campaign_v2/workspace/scripts/evaluate_ensemble_v2.py`, con el workspace explícito y la ruta de dependencias de árboles original. Ese detalle garantiza que modelos, configuración derivada y guardas se importen desde el snapshot V2. Los comandos raíz del runner son adecuados para preparación/ajuste/selección; para las fases reservadas se emplea este entrypoint congelado. En el momento de entregar componentes, el engine de entrenamiento no había ejecutado META, CAL ni FINAL.
