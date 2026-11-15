# NHANES: ampliación para preentrenar encoders

Descargados y leídos los archivos públicos CBC_J, BIOPRO_J y DEMO_J del ciclo 2017–2018, junto con sus diccionarios CDC. Fuentes: [hemograma](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.htm), [bioquímica](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.htm), [demografía y diseño](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm).

## Resultado verificado sobre archivos

- 5.274 adultos con al menos una variable disponible entre las 29 seleccionadas.
- 14 medidas candidatas CMP y 15 CBC, alineadas por identidad de participante `SEQN` dentro del mismo ciclo. No son 29 series temporales: cada participante aporta una observación transversal.
- 151.172 valores elegibles según la política de observación; se excluyen de objetivos exactos los resultados con flag documentado bajo límite de detección. Otros límites/códigos no resueltos conservan su advertencia y requieren revisión antes de entrenar.
- Los recuentos absolutos del diferencial CBC de NHANES están calculados y redondeados; se convierten de miles de células/uL a células/uL multiplicando por 1.000. No son nueva evidencia independiente del total y porcentajes.
- Se conservan los pesos, estratos y PSU de la encuesta para posteriores análisis; no se presentan estimaciones poblacionales sin su diseño muestral.

## Uso defendible

Preentrenamiento de representación mediante reconstrucción de analitos observados ocultados artificialmente: ocultar parte de un perfil de laboratorio y reconstruir únicamente esos valores conocidos. Separar personas para entrenamiento/validación/prueba antes de ajustar escalas o muestrear máscaras. La máscara de corrupción es diferente de la máscara de ausencia real: un dato realmente ausente no se convierte en etiqueta por ocultar otro dato.

Después puede evaluarse transferencia de esos encoders a las observaciones de astronautas. Esto es una hipótesis a contrastar, no un beneficio medido. No añade astronautas ni enseña dinámica temporal, radiación, estado en microgravedad o evolución individual. No crear pseudo-visitas ordenando personas por edad ni concatenarlas a historiales de Inspiration4.

## Compatibilidad por revisar

La correspondencia en `config/nhanes_candidate_mapping.json` es explícitamente **candidata y deshabilitada**. Instrumentos, población, manejo de muestras, estado de ayuno y condiciones preanalíticas pueden diferir. La equivalencia dimensional no establece equivalencia de ensayo. En particular, bicarbonato se relaciona con CO2 total del CMP pero no debe fusionarse sin revisar métodos. Tampoco una muestra terrestre debe compartir identidad de sujeto o misión con una de vuelo.

Los dos flags de detección utilizados se validan como categorías 0/1. La conversión SAS/IBM del lector pandas puede expresar el cero como aproximadamente 5,4e-79; el script valida y redondea solo esas categorías antes de crear la máscara, para no descartar todos los resultados como si no estuvieran observados. No se modifica la medida original por este ajuste de lectura de flags.

## Reproducir y localizar

`python scripts/download_nhanes.py` descarga o verifica la caché. Los XPT/HTML originales están en `data/raw/external/NHANES_2017_2018/`; los hashes y URLs en `data/manifests/nhanes_downloads.json`. La tabla y su máscara separada están en `data/processed/external/nhanes_adult_labs.csv` y `nhanes_adult_labs_observed_mask.csv`. Resumen por variable en `outputs/data_expansion/nhanes_summary.json`.

No se ha entrenado ni modificado el modelo de astronautas con estos datos. Se prioriza completar el catálogo y auditar transferencia antes de iniciar nuevos experimentos.
