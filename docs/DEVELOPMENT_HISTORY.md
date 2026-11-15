# Gemelo digital de astronautas

Prototipo de investigación para NASA Space Apps 2026: encoders de categorías biomédicas → espacio latente temporal común → cabezas Transformer por grupo → predicciones q10/q50/q90 condicionadas al historial y horizonte.

## Estado — 14 de noviembre de 2026

La presentación vigente está en **http://127.0.0.1:8765/** (`python scripts/dashboard.py`). La vista inicial muestra **2 horas de histórico y 4 horas de futuro**, con controles independientes de histórico (1/2/24 h) y horizonte (1/4/8/24 h). **Explore measurements** permite buscar y filtrar 32 salidas por estudio. ORBITA se abre a la derecha a altura completa y la gráfica se adapta en escritorio. Incluye una experiencia unificada en inglés, histórico de predicciones reproducidas, 96 consultas futuras entrenadas cada 15 minutos hasta 24 h, bandas q10/q50/q90, comparación con observaciones, modo claro/oscuro y todas las salidas actuales en un panel secundario. Los objetivos cardíacos escasos mantienen la resolución por visitas.

El [paquete de presentación](presentation/README.md) contiene el [vídeo de 30 segundos](presentation/orbita-30s.mp4) con narración inglesa, [guion](presentation/storyboard-30s.md) y [arquitectura editable draw.io](presentation/architecture.drawio). La nueva campaña independiente está en `outputs/presentation_campaign/`; conserva los artefactos V3. **165 pruebas aprobadas**, además de comprobaciones visuales y funcionales del navegador. El objetivo sigue siendo recoger más participantes y mediciones de astronautas más frecuentes y emparejadas para mejorar y validar las predicciones; no se atribuye validación clínica.

### Hitos anteriores y contexto de desarrollo

**Arquitectura inicial y plan de aquella fase:** el Transformer recibe todos los latentes de la ventana: `[B,N,d]`. Actualmente `[B,24,16]` por datos horarios; 24 h/15 min serían 96 posiciones. No se comprime toda la ventana en un único vector antes de la atención. La [traza verificada](outputs/architecture/temporal_shape_trace.json), la [verificación independiente](docs/verificacion-secuencia-transformer.md) y el [plan de comparaciones con ensemble final](docs/plan-comparaciones-secuencia-y-ensemble.md) documentan el contrato. No se han lanzado nuevos entrenamientos al actualizar este plan. El ensemble está previsto para la fase final, tras comparar candidatos individuales.

**Primera comparación neuronal completada:** GRU y Transformer con encoders/fusión/readouts comunes, mismas 34 variables y horizontes, mismos splits y escalado. Se verificó igualdad de las 305.228 parejas por método y se generaron métricas y gráficos conjuntos. El criterio de desarrollo favorece la referencia lineal (0,156521) frente a Transformer (0,175066) y GRU (0,178455); la prueba es descriptiva y confirma que ambas redes no mejoran el agregado lineal bajo este presupuesto. Véase [comparación y selección](docs/comparacion-neuronal-horaria.md), [interpretación por grupos](docs/interpretacion-comparativa-horaria.md) y [auditoría independiente](docs/auditoria-comparativa-horaria.md). **76 pruebas aprobadas**. No hay calibración, preentrenamiento ni adaptación espacial ejecutados en esta fase.

**Primer bloque horario:** constructor de ventanas causales de hasta 24 h, particiones fijas 600/150/100/150 episodios y baselines de persistencia/regresión cuantílica lineal para 34 variables a 1/4/8/24 h. Se generaron 610.456 filas de predicción y evaluación por variable, cabeza, horizonte, hospital, contexto y episodio. La prueba contiene 150 episodios; la calibración queda reservada. Véase [resultados y gráficos](docs/primer-bloque-horario.md). Este hito tuvo 58 pruebas aprobadas. La resolución usada es 1 h; no se han inventado medidas a 15 min.

La [hoja de ruta horaria](docs/hoja-ruta-horaria.md) organiza las siguientes comparaciones de arquitectura, transferencia y dashboard. El [protocolo de baselines](docs/baselines-horarios.md) describe ajuste, selección solo en desarrollo, datos parciales y límites. El mejor modelo puede variar según objetivo y métrica: la referencia lineal mejora pinball macro en todos los horizontes, mientras persistencia tiene menor MAE macro a 24 h. Hay etiquetas escasas y unidades pendientes de revisión; se conservan visibles y no se extrapola validación a astronautas.

**Ampliación posterior:** ya se han descargado y auditado más fuentes. UTMB C3 añade 31 identificadores y 118 visitas de un análogo de reposo en cama, con 23 analitos comparables; C1 añade laboratorio y sueño en los mismos tres sujetos; Mars500 aporta 159 muestras de seis participantes. Hay además datos de otros astronautas en modalidades ómicas, y fuentes clínicas terrestres para estudiar preentrenamiento. Los dominios permanecen separados: no se han reentrenado los modelos con la ampliación. La suite vigente contiene **38 pruebas aprobadas**.

El [informe de ampliación y siguiente experimento](docs/ampliacion-y-siguiente-experimento.md) reúne los datos efectivamente obtenidos, prioridades, aumento artificial, atención actual y máscaras por cabeza. El inventario siguiente describe el primer experimento de Inspiration4, conservado como referencia.

- Descargados 19 archivos públicos de NASA OSDR, 1,78 MB, con manifiesto, URLs y SHA-256.
- Datos humanos reales de Inspiration4: cuatro personas, siete visitas irregulares, 28 visitas-persona. No hay medidas durante vuelo en estos paneles.
- Auditadas 322 variables canónicas de ensayo: CMP19, CBC20, Eve cardiovascular9, Eve inmune71, Alamar203; 8.783 valores numéricos observados.
- El agente biomédico definió 109 variables activas en 13 grupos. Alamar permanece fuera del primer modelo hasta completar anotación.
- EDA ejecutado: cobertura, unidades, faltantes, trayectorias, 51.360 pares con Spearman y ajuste descriptivo de rangos por persona.
- Modelo PyTorch implementado y primer entrenamiento técnico completado, con baseline de persistencia y resultados reproducibles. La prueba real es de 112 días en cuatro personas; todavía no demuestra generalización ni calibración adecuada.

q50 es la mediana. q10–q90 es una banda predictiva central nominal del 80 %, cuya cobertura debe verificarse por variable y horizonte; no es un intervalo de confianza ni cobertura conjunta de todas las variables.

## Ejecución

Bloque horario actual, usando las fuentes ya descargadas:

```powershell
python scripts/build_hourly_dataset.py
python scripts/train_hourly_baselines.py
python scripts/report_hourly_baselines.py
python -m pytest -q
```

Comparación neuronal posterior, con las mismas particiones:

```powershell
python scripts/train_hourly_neural.py --epochs 20 --batch-size 256
python scripts/report_hourly_baselines.py --include-neural
```

Estados seleccionados, escalas y predicciones en `outputs/hourly_neural/`; comparación de cuatro modelos, curvas y métricas en `outputs/hourly_comparison/`. La GRU conserva readouts con atención: se compara la memoria temporal, no una GRU pura sin atención. Los intervalos siguen sin recalibrarse y las violaciones del soporte físico se reportan sin recortes posteriores.

Datos, splits y hashes: `data/processed/hourly/`; modelos, escalas y predicciones: `outputs/hourly_baselines/`; métricas y PNGs: `outputs/hourly_evaluation/`. Las bandas de esta primera referencia no están recalibradas. Las violaciones del soporte físico de predicciones se registran; no se recortan a posteriori para mejorar la evaluación.

El siguiente bloque conserva la reproducción del experimento histórico de Inspiration4 entre visitas:

Entorno usado: Python 3.13.5, PyTorch 2.12.1 CPU, pandas 2.3.3, NumPy 2.2.6. Dependencias en `pyproject.toml`; versiones del experimento en `outputs/training/run.json`. Ejecutar desde el directorio del proyecto con un entorno que tenga esas dependencias:

```powershell
python scripts/download_osdr.py --studies 575 569
python scripts/analyze_osd575.py
python -m pytest -q
python scripts/train_exploratory.py --epochs 80
python scripts/report_experiment.py
```

La descarga acepta únicamente archivos públicos pequeños y registra los descartados.

## Datos y decisiones

Se conservan los originales en `data/raw/`. Las tablas normalizadas son `data/processed/inspiration4_observations.csv` y `inspiration4_wide.csv`. La unidad de hemoglobina procede del original g/dL, ya que su copia transformada la etiqueta incorrectamente como porcentaje. No se usan las normalizaciones porcentuales Eve calculadas con visitas/personas de referencia que podrían introducir información futura.

Las referencias temporales L (lanzamiento) y R (retorno) se conservan. El modelo usa segmentos separados sin inventar el tiempo entre ambas. Se supone disponibilidad inmediata en cada visita para el experimento retrospectivo; no conocemos la latencia real de publicación de cada resultado.

Los encoders consumen valores estandarizados solo con entrenamiento, máscaras y antigüedad de la observación. La atención temporal es causal. Los decoders consultan horizontes positivos sin utilizar valores futuros realizados y producen cuantiles ordenados. Solicitar otro horizonte no modifica una predicción existente. El pipeline experimental impone soporte no negativo al invertir la escala de las 109 variables activas.

## Documentación vigente

- [Plan de comparaciones: secuencia completa y ensemble final](docs/plan-comparaciones-secuencia-y-ensemble.md).
- [Configuración del plan](config/comparison_roadmap.json).
- [Verificación del eje temporal del Transformer](docs/verificacion-secuencia-transformer.md).

- [Comparación GRU/Transformer y baselines](docs/comparacion-neuronal-horaria.md).
- [Método y ejecución neuronal](docs/modelos-neuronales-horarios.md).
- [Interpretación de la comparación](docs/interpretacion-comparativa-horaria.md).
- [Auditoría de datos y claves compartidas](docs/auditoria-comparativa-horaria.md).

- [Primer bloque horario: resultados y gráficos](docs/primer-bloque-horario.md).
- [Baselines horarios: ajuste y selección](docs/baselines-horarios.md).
- [Calidad horaria aplicada](docs/calidad-horaria-aplicada.md).

- [Plan de decisiones actualizado](docs/plan-decisiones-y-experimentos.md).
- [Hoja de ruta horaria](docs/hoja-ruta-horaria.md).
- [Viabilidad de 1/4/8/24 h sobre archivos reales](docs/viabilidad-horizontes-horarios.md).
- [Protocolo biomédico horario](docs/protocolo-horario-biomedico.md).
- [Arquitectura y ablaciones horarias](docs/arquitectura-y-ablaciones-horarias.md) y [plan de experimentos estructurado](config/experiment_plan.json).

- [Ampliación de datos y siguiente experimento](docs/ampliacion-y-siguiente-experimento.md).
- [Descargas espaciales y análogos](docs/ampliacion-datos-espaciales.md).
- [Transferencia y compatibilidad biomédica](docs/ampliacion-datos-transferencia.md).
- [Pérdidas enmascaradas, aumento y atención](docs/aumento-enmascaramiento-atencion.md).
- [NHANES para preentrenamiento de encoders](docs/nhanes-preentrenamiento.md).

- [Fuentes y descarga reproducible](docs/fuentes-descarga.md).
- [Análisis exploratorio de las tablas reales](docs/eda-osd575.md).
- [Interpretación biomédica de correlaciones y redundancias](docs/interpretacion-eda.md).
- [Agrupación biomédica de encoders](docs/agrupacion-biomedica.md) y [configuración ejecutable](config/biomedical_groups.json).
- [Modelo implementado y contrato de tensores](docs/modelo-implementado.md).
- [Primer entrenamiento, resultados y limitaciones](docs/primer-entrenamiento.md).
- [Equipo, responsabilidades y fases](docs/equipo-y-fases.md).

Los documentos `analisis-datos.md`, `plan-modelado.md` y `arquitectura-salud.md` conservan el análisis inicial anterior a las descargas; los informes anteriores describen el estado vigente. La arquitectura propuesta por el usuario ya es suficiente como referencia: no se necesita documentación del TFM para continuar.

## Preparación de fuentes nuevas

Los scripts de descubrimiento y descarga, con comandos completos, están documentados en el inventario espacial y el plan de transferencia. Tras descargar las fuentes correspondientes:

```powershell
python scripts/prepare_lsda_utmb.py
python scripts/prepare_utmb_sleep.py
python scripts/prepare_mars500.py
python scripts/download_nhanes.py
python -m pytest -q
```

Los datos normalizados conservan dominio, identidad local, fecha real o referencia temporal original, unidades, calidad y procedencia. Los mapeos candidatos no activan automáticamente el entrenamiento ni hacen equivalentes los ensayos. Los scripts de preparación no aceptan como etiquetas puntuales intervalos, resultados censurados o notas analíticas sin revisar.

El entrenador utiliza `grouped_masked_pinball_loss`; `--loss-reduction equal_head` es el modo por defecto y `per_target` la alternativa por posición observada. Una cabeza sin etiquetas se omite del objetivo; un lote totalmente sin etiquetas no actualiza el optimizador. La representación compartida sigue aprendiendo desde otras cabezas. Este cambio no constituye una recalibración del intervalo q10–q90.

## Artefactos de evaluación

La [campaña V2](docs/resultados-campana-v2.md) ya está completada: se verificaron y descargaron los cinco recursos oficiales del reto, se amplió el entrenamiento de 600 a 1.600 episodios y se repitieron las once variantes. Los cinco enlaces son documentación, no nuevas observaciones horarias; la ampliación clínica adicional es PhysioNet, UCI terrestre. La [auditoría de recursos](docs/auditoria-recursos-reto-2026.md) y la [ampliación de datos](docs/ampliacion-horaria-v2.md) mantienen esa distinción.

El ensemble se aprendió en 200 episodios META separados; después se ajustaron las bandas en 100 de calibración y se evaluó una sola vez sobre 300 episodios nuevos. La cobertura macro del ensemble ajustado es 80,39 %, sin garantía formal bajo dependencia temporal. Las mejoras de precisión son pequeñas y los intervalos de comparación por episodio incluyen cero; el ensemble no domina al mejor Transformer en todas las métricas. Los pesos, hashes y tablas por variable/horizonte están en `outputs/model_campaign_v2/workspace/`.

La [comparación V1/V2 con escala común](docs/repeticion-v1-v2-escala-comun.md) evita atribuir mejoras al cambio de normalizadores. Se conserva V1 intacta. La configuración captura las condiciones iniciales; el estado real se consulta con `python scripts/run_model_campaign_v2.py --stage status`. No se debe volver a abrir la prueba final para ajustar modelos.

Para la siguiente fase se implementaron un [contrato de evidencia y resumen en español](docs/capa-interpretacion-clinica.md) y recuperación local de nueve documentos NASA/CSA con citas. Hay una demostración retrospectiva en `outputs/clinical_interpretation_demo_v2/summary.md`. No se entrenó ni ejecutó FLAN-T5 ni un modelo mediante API. La [propuesta de agentes y modelos](docs/agentes-producto-y-codex.md) compara las opciones y qué falta para una evaluación con especialistas.

Como antecedente, la campaña V1 comparó once variantes, entrenó diez configuraciones nuevas y comprobó el candidato preseleccionado sobre 300 episodios adicionales. Véanse el [ranking de desarrollo V1](docs/campana-modelos-individuales.md) y la [confirmación V1 con gráficos e incertidumbre](docs/confirmacion-modelos-individuales.md).

El Transformer residual con memoria temporal por grupo conserva hasta 24 latentes horarios y mejora pinball un 0,62 % en la confirmación completa frente a la lineal. La cobertura baja del 78,4 % al 75,1 %; en las combinaciones con más soporte, el bootstrap no demuestra una ventaja clara. Se conservan ambos modelos y las variables difíciles, sin entrenar todavía el ensemble. Estos resultados proceden de UCI terrestre.

Los CSV, figuras por cabeza/horizonte, selección congelada y hashes están en `outputs/model_campaign/`. Los checkpoints nuevos están en `outputs/hourly_architecture_trials/`, `outputs/hourly_residual_control/` y `outputs/hourly_trees/`. El evaluador de confirmación ya se ejecutó y protege sus resultados contra sobrescritura. Para regenerar el informe desde sus CSV:

```powershell
python scripts/bootstrap_campaign_confirmation.py
python scripts/report_campaign_confirmation.py
```

`outputs/eda/` contiene diccionario, correlaciones, cobertura, intervalos y PNGs. `outputs/training/` contiene predicciones, métricas por variable/horizonte, particiones, escalas, curva de pérdida y ejemplos visuales. El checkpoint está en `outputs/checkpoints/exploratory_twin.pt`.

La partición del experimento es de 16 pares de entrenamiento, cuatro de validación y cuatro de prueba. Los resultados de validación seleccionan el checkpoint; la prueba se emite desde R+82 hacia R+194. Solo hay cuatro personas, los biomarcadores son dependientes y el EDA previo ha inspeccionado la cohorte completa: los resultados son exploratorios. Los modelos no ofrecen decisiones clínicas ni simulaciones causales de intervenciones.

## Fuentes

- [Reto oficial 2026](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/): resumen localizado; su página completa y recursos no fueron recuperables con la herramienta web durante esta revisión.
- [NASA OSD-575](https://osdr.nasa.gov/bio/repo/data/studies/OSD-575) y [NASA OSD-569](https://osdr.nasa.gov/bio/repo/data/studies/OSD-569): fuentes efectivamente descargadas; se incorporan como datos NASA del proyecto, sin atribuirlas como un dataset prescrito del reto.
- [CSA: Monitoring radiation for astronaut health](https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp): contexto de instrumentación; no se empareja radiación ambiental sin identificación con la salud de estos sujetos.
