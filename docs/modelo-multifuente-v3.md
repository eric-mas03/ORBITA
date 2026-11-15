# Modelo multifuente V3 y dashboard

Se ha entrenado una campaña independiente en `outputs/source_campaign_v3/`, sin sustituir los pesos ni las reservas de las campañas anteriores. **Seguimiento por horas** consume su checkpoint mediante inferencia real. La página inicial muestra la tripulación Inspiration4; los datos continuos se abren con el botón de seguimiento. Radiación, revisiones cardíacas y exploración avanzada están en **Más opciones**. Los selectores de fuente/modelo y la evaluación quedan dentro de apartados cerrados por defecto. Las rutas `?mode=nasa`, `?mode=icu` y `?mode=continuous` permiten abrir una vista concreta.

## Qué entra en el modelo

| Rama | Entradas y salidas | Reloj |
|---|---|---|
| BIG IDEAs | Entradas de glucosa intersticial, pulso y macronutrientes/calorías de comidas ya registradas; salidas de glucosa y pulso | Contexto de 24 h = 96 pasos de 15 min; horizontes 1/4/8/24 h |
| NASA RadLab ISS | Tasa de dosis absorbida de DosTel1 | Contexto de 24 h = 96 pasos; horizontes 1/4/8/24 h; evaluación temporal dentro de un instrumento |
| NASA UTMB C1 | Ocho medidas ecocardiográficas | Hasta ocho visitas conocidas; una o dos visitas futuras |
| NASA UTMB C3 | Las ocho medidas disponibles de observaciones ABCD y de E/F | Visitas reales ordenadas por fase y día publicado |
| NASA CFT70 2D | Ocho medidas ecocardiográficas | Una o dos visitas futuras |
| NASA CFT70 3D | Masa y volúmenes ventriculares | Media de lecturas por visita, no un sujeto por analista |
| NASA C1 vascular SLNTG | Diámetros bajo protocolo, al minuto 3, promediando frames | Respuesta al protocolo en visitas posteriores; no diámetro basal ni recomendación de intervención |

Las variables ecocardiográficas seleccionadas son diámetro ventricular diastólico/sistólico, velocidades mitrales E/A, tiempo de desaceleración mitral, tiempos pulmonares de aceleración/eyección y relajación isovolumétrica. Sus unidades se mantienen como pendientes de armonización clínica y la vista no impone rangos normales inventados.

Se completó la descarga pública de pulso de las 16 personas BIG IDEAs. El manifiesto de la adquisición ahora contiene 55 archivos y 251.263.775 bytes, incluyendo las adquisiciones anteriores y diccionarios. Algunos sujetos tienen tiempos en segundos y otros etiquetas de minuto repetidas: en estos últimos se usa el final del minuto para impedir que el resumen de un intervalo incorpore segundos todavía futuros.

Las siete ramas no implican siete cohortes independientes: CFT70 2D y 3D comparten 37 identificadores; C1 eco y vascular comparten tres. El modelo no empareja filas de personas distintas ni une radiación de la ISS con las fechas anonimizadas terrestres.

## Arquitectura y máscaras

Encoders de grupos de variables con valores, máscara de presencia y tiempo desde última observación; fusión por fuente y embedding de dominio; una memoria Transformer compartida de ancho 32 con cuatro cabezas de atención; atención cruzada desde las consultas de horizonte; salidas de cuantiles ordenados. Cada rama tiene su propia salida residual sobre una persistencia probabilística calculada solo con entrenamiento. Los tiempos internos son coordenadas normalizadas y explícitas por dominio: horas divididas por 24 en las series continuas y posiciones de visita en eco. No representan un mismo calendario entre dominios.

La salida no recibe valores futuros reales. Las comidas futuras tampoco entran. La pérdida solo usa etiquetas observadas y las otras cabezas permanecen sin gradiente directo cuando se entrena una rama. Todos los pasos de la ventana llegan a la atención; no se sustituye la secuencia por una media.

Las diferencias de edad/población, ensayo y condiciones de medición persisten aunque exista un latente compartido. Esta campaña demuestra una implementación y una evaluación por fuente; no demuestra transferencia clínica a astronautas ni que cada fuente mejore a las demás. Esa atribución necesita una ablación de entrenamiento separado frente a compartido.

## Particiones y selección

- Personas separadas en entrenamiento, validación, calibración y prueba para las fuentes humanas. La asignación se comparte entre las dos ramas CFT70 y las dos C1.
- RadLab: ocho primeros días para ajuste, dos siguientes para validación, dos para calibración y dos para prueba. Cada etiqueta permanece dentro de su bloque. El historial conocido puede venir del bloque anterior; no sus valores futuros.
- Escalas ajustadas sobre filas originales de entrenamiento, sin contar varias veces los contextos solapados.
- Selección de checkpoint solo con validación: época 17. El ensemble explora pesos de Transformer 0/0,25/0,5/0,75/1 solo en validación.
- Expansión del intervalo en calibración independiente, por variable y horizonte; no se ajusta si hay menos de 20 etiquetas. En las ramas pequeñas se muestra **calibración insuficiente**.
- El test no selecciona pesos ni checkpoint. Los intervalos nominales del 80 % no garantizan cobertura individual y la correlación entre ventanas limita una interpretación de incertidumbre basada en conteos simples.

## Resultado de prueba

Promedios de pérdida pinball normalizada entre variables y horizontes, con las escalas de entrenamiento. La tabla no compara MAE entre unidades distintas:

| Rama | Persistencia | Transformer | Ensemble |
|---|---:|---:|---:|
| Glucosa/pulso | 0,2729 | 0,2265 | 0,2264 |
| Radiación ISS | 0,1216 | 0,1192 | 0,1203 |
| NASA CFT70 2D | 0,2741 | 0,2619 | 0,2619 |
| NASA CFT70 3D | 0,2224 | 0,2071 | 0,2071 |
| NASA C3 eco | 0,2788 | 0,2668 | 0,2686 |
| NASA C1 eco | 0,5592 | 0,5312 | 0,5312 |
| NASA C1 vascular | 0,3772 | 0,4049 | 0,3772 |

El peso elegido en validación para C1 vascular es cero: se conserva persistencia. El ensemble de C3 usa peso 0,75; el resto usa peso 1 con ajuste de bandas solo donde la calibración es suficiente. Que una combinación elegida en validación no supere al mejor componente en prueba no justifica elegir otra mirando ese test.

La cobertura media del ensemble es aproximadamente 78 % en glucosa/pulso y 91 % en radiación; las ramas eco pequeñas quedan entre 65 % y 75 %, con calibración insuficiente en muchas celdas. `metrics.csv` contiene MAE, pinball, cobertura, anchura y número de etiquetas por rama, variable, horizonte y modelo. `predictions.csv` permite reproducir cada cifra. Las figuras exportadas están en `figures/`.

## Qué sigue registrado pero no produce etiquetas de entrenamiento

CGMacros queda en cuarentena por interpolación no resuelta. El archivo de comida 003 de BIG IDEAs no tiene cabecera fiable; solo se excluye esa modalidad, no su glucosa/pulso. Los 43 timestamps conflictivos de RadLab se excluyen conservando las 86 filas originales. Los campos NASA de presión/pulso sospechosos y la derivada WS no entran. Medias agregadas, porcentajes de cambio, la copia de siete participantes y el duplicado C3D no se añaden como personas o etiquetas independientes. Los catálogos de PhysioCGM y fuentes de acceso controlado no se presentan como observaciones adquiridas.

## Dashboard y límites de interpretación

Selector de fuente, persona/instrumento, origen, modelo y horizonte; tarjetas actual → previsto; gráfica con observaciones previas, comparaciones reales posteriores, línea vertical del origen, previsión discontinua y banda continua; tabla de los cuantiles y valores posteriores; evaluación por variable/horizonte en prueba. Permite elegir el último dato para mostrar un futuro todavía sin observación real.

Las previsiones anteriores se reconstruyen con el checkpoint actual y se identifican así; no se afirma que hubiesen sido emitidas en las fechas históricas. Las tarjetas identifican tierra, análogo NASA o entorno de la ISS. La radiación no se presenta como dosis personal; los participantes de reposo en cama no se presentan como astronautas. El asistente de la nueva vista resume los valores reales del panel con respuestas predefinidas; no es un nuevo LLM entrenado ni valida tratamientos.

## Reproducción

```powershell
python scripts/acquire_longitudinal_pilot.py
python scripts/train_source_campaign_v3.py --epochs 24
python scripts/report_source_campaign_v3.py
python -m pytest tests/test_source_expansion.py tests/test_dashboard_data.py -q
python scripts/dashboard.py
```

`--refinalize` reutiliza los pesos ya congelados y recompone selección en validación, calibración e informes sin entrenar. La regla de mezcla utilizada en selección y servicio es la misma: primero mezcla los cuantiles y después aplica soporte no negativo. Se conserva la semilla 2026, los roles y hashes de archivos fuente en `fit_manifest.json`.
