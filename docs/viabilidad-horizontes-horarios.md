# Viabilidad real de horizontes de 1, 4, 8 y 24 horas

Auditoría descriptiva del subconjunto descargado de PhysioNet Challenge 2019 v1.0.0. **No se ha entrenado ni cambiado ningún modelo, partición o métrica anterior.** Los cuantiles solicitados son q10, q50 y q90; esta auditoría mide disponibilidad de etiquetas, no su precisión o calibración.

## Definición de los recuentos

Se analizaron 1.000 archivos de UCI: 500 del hospital A y 500 del B, con 38.153 filas horarias. Cada archivo es una secuencia; no se afirma que sean personas independientes entre posibles ingresos. No se encontraron ICULOS duplicados, desordenados, no enteros ni huecos horarios. Ninguna secuencia fue excluida. Las longitudes van de 8 a 336 filas; mediana 39, cuartiles 24 y 47.

El origen se sitúa al final de la fila horaria actual ICULOS=t. Para h se exige una fila de la misma secuencia exactamente en t+h; no se usa la siguiente medición más cercana. La etiqueta de cada variable debe ser un valor finito publicado en esa fila. No hay forward fill, interpolación ni requisito de que las demás salidas estén disponibles. Un valor publicado no demuestra por sí solo una extracción de sangre nueva e independiente en esa hora.

La ventana candidata comprende hasta 24 slots horarios [t−23,t], incluyendo la fila actual. Contexto parcial: entre uno y 24 slots existentes y al menos una medición finita en alguna de las 34 variables dinámicas dentro de la ventana. Contexto completo: existen los 24 slots y al menos una medición dinámica. **Completo describe cobertura de slots, no ausencia de valores perdidos**: todas las celdas restantes pueden mantenerse con máscaras. No se exige que el propio objetivo se haya medido previamente. Edad, género, unidad, tiempo de ingreso, ICULOS y SepsisLabel no cuentan como mediciones dinámicas para habilitar el contexto.

## Orígenes candidatos antes de exigir un objetivo concreto

| Horizonte | Fila futura exacta | Contexto parcial | 24 slots completos | Secuencias con fila futura |
|---|---:|---:|---:|---:|
| 1 h | 37153 | 36302 | 15690 | 1000 |
| 4 h | 34153 | 33302 | 13552 | 1000 |
| 8 h | 30153 | 29308 | 10893 | 994 |
| 24 h | 15690 | 15076 | 3306 | 732 |

## Disponibilidad por salida

Cada celda indica **pares con contexto parcial / pares con 24 slots completos**. Las columnas finales muestran cuántas secuencias aportan al menos un par a 24 h. El CSV contiene además las etiquetas futuras sin requisito de contexto y el número de secuencias para cada variable, horizonte y tipo de contexto.

| Variable publicada | 1 h: P/F | 4 h: P/F | 8 h: P/F | 24 h: P/F | Secuencias a 24 h: P/F |
|---|---:|---:|---:|---:|---:|
| HR | 33416/14473 | 30603/12487 | 26895/10014 | 13917/3075 | 713/237 |
| O2Sat | 32263/13944 | 29523/12042 | 25924/9683 | 13411/3015 | 710/235 |
| Temp | 12383/4969 | 11249/4268 | 9750/3420 | 4769/941 | 666/202 |
| SBP | 31953/13684 | 29249/11803 | 25654/9467 | 13148/2975 | 688/227 |
| MAP | 32551/14053 | 29808/12121 | 26190/9730 | 13511/3005 | 711/235 |
| DBP | 25556/11126 | 23474/9637 | 20670/7769 | 10698/2582 | 547/177 |
| Resp | 31471/13662 | 28837/11795 | 25338/9458 | 13138/2898 | 707/230 |
| EtCO2 | 966/319 | 849/281 | 698/232 | 308/98 | 24/6 |
| BaseExcess | 1977/643 | 1684/547 | 1337/445 | 624/170 | 143/32 |
| HCO3 | 1507/620 | 1345/558 | 1168/452 | 609/145 | 296/44 |
| FiO2 | 2950/1132 | 2552/1004 | 2134/846 | 1099/386 | 178/58 |
| pH | 2463/788 | 2092/662 | 1676/536 | 763/198 | 208/46 |
| PaCO2 | 1983/637 | 1666/540 | 1326/437 | 615/169 | 185/41 |
| SaO2 | 1258/384 | 1055/320 | 844/243 | 372/61 | 130/27 |
| AST | 579/181 | 486/154 | 403/118 | 172/42 | 114/26 |
| BUN | 2463/991 | 2209/884 | 1904/704 | 969/220 | 549/89 |
| Alkalinephos | 570/180 | 480/153 | 398/117 | 171/41 | 114/26 |
| Calcium | 2060/823 | 1814/727 | 1529/573 | 799/180 | 496/78 |
| Chloride | 1656/666 | 1470/596 | 1264/481 | 652/149 | 301/46 |
| Creatinine | 2177/874 | 1959/779 | 1682/617 | 853/187 | 547/87 |
| Bilirubin_direct | 66/15 | 57/13 | 46/12 | 14/5 | 7/2 |
| Glucose | 6215/2367 | 5565/2029 | 4728/1611 | 2277/459 | 606/124 |
| Lactate | 877/236 | 707/189 | 546/153 | 228/46 | 98/17 |
| Magnesium | 2264/968 | 2035/865 | 1745/693 | 946/210 | 518/83 |
| Phosphate | 1441/610 | 1280/551 | 1087/437 | 597/139 | 393/58 |
| Potassium | 3346/1311 | 2966/1149 | 2509/899 | 1273/281 | 569/98 |
| Bilirubin_total | 537/166 | 450/140 | 374/108 | 157/37 | 115/25 |
| TroponinI | 353/77 | 286/60 | 217/44 | 73/13 | 46/10 |
| Hct | 3179/1210 | 2853/1042 | 2393/814 | 1179/215 | 562/94 |
| Hgb | 2673/1050 | 2396/919 | 2047/717 | 1026/186 | 551/84 |
| PTT | 1022/365 | 881/324 | 740/261 | 356/71 | 214/34 |
| WBC | 2313/929 | 2079/829 | 1774/656 | 909/179 | 545/79 |
| Fibrinogen | 220/65 | 191/53 | 155/46 | 64/12 | 36/4 |
| Platelets | 2150/858 | 1935/761 | 1639/599 | 837/159 | 543/79 |

## Qué permiten concluir estas cifras

Las constantes vitales tienen más etiquetas horarias que muchos ensayos de laboratorio. Para glucosa a 24 h hay 2.277 pares con contexto parcial repartidos en 606 secuencias, frente a 459 pares con 24 slots en 124 secuencias. Para fibrinógeno a 24 h quedan 64 pares en 36 secuencias, o 12 pares en cuatro secuencias con 24 slots. No puede decidirse una validación suficiente contando pares solapados como si fueran participantes independientes.

Permitir historial parcial conserva bastantes ejemplos; se deben registrar por separado longitud efectiva, máscaras, antigüedad del último valor y variable/horizonte evaluados. Estos recuentos son anteriores a cualquier partición, selección de objetivos o requisito adicional de contexto; el entrenamiento y la evaluación dispondrían de menos ejemplos. El subconjunto no es una estimación representativa de toda PhysioNet ni de astronautas.

## Resolución de 15 minutos y otras fuentes

**No se pueden reconstruir cuatro observaciones de 15 minutos ni sus mean/min/max a partir de una única fila horaria.** Repetir o interpolar cada hora no crea información observada. Para esta fuente, la planificación debe conservar la resolución publicada de una hora; los horizontes 1/4/8/24 h sí están expresados en ese reloj.

BIDMC aporta registros cortos de aproximadamente ocho minutos en el conjunto disponible y no proporciona etiquetas reales a una hora o más dentro de cada registro. UTMB e Inspiration4 tienen visitas clínicas separadas por días o semanas: no justifican etiquetar resultados sanguíneos a 1/4/8/24 horas. La actigrafía diaria de UTMB tampoco permite convertir sueño diario en medidas de 15 minutos.

## Artefactos y reproducción

- `outputs/feasibility/target_availability.csv`: 136 combinaciones de variable y horizonte; pares y secuencias para etiquetas futuras, contexto parcial y completo.
- `outputs/feasibility/target_availability_by_hospital.csv`: los mismos recuentos por hospital y agregados.
- `outputs/feasibility/origin_availability.csv`: orígenes candidatos antes de seleccionar salidas.
- `outputs/feasibility/sequence_audit.csv`: orden, duplicados, huecos, longitudes y disponibilidad por archivo.
- `outputs/feasibility/length_histogram.csv` y `length_bins.csv`: distribución exacta y agrupada de longitudes.
- `outputs/feasibility/hourly_summary.json`: configuración fija, definiciones y resumen verificable.

Reproducir con `python scripts/audit_hourly_horizons.py`. Se lee exclusivamente el subconjunto local; no se descarga nada ni se modifican los archivos originales.

Fuente: [PhysioNet Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/).
