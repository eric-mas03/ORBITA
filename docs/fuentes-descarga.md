# Fuentes y descargas verificadas

Descarga realizada el 14 de noviembre de 2026. Se conservaron archivos originales sin modificar y catálogo/metadatos de NASA OSDR. Los hashes SHA-256, tamaños, URLs estables, fecha UTC y evidencia de acceso están en `data/manifests/osdr_downloads.json`. Las URLs finales se guardan sin credenciales temporales de firma.

## Reproducir

```powershell
python scripts/download_osdr.py --studies 575 569
```

El script usa biblioteca estándar, verifica `restricted == false`, limita cada archivo a 2 MB y acepta tablas CSV/TSV/XLSX/TXT y metadatos ISA.zip. No solicita ni elude acceso controlado. Los archivos `.txt` de checksums incluidos no son observaciones. Los catálogos se consultan en `https://osdr.nasa.gov/osdr/data/osd/files/{id}` y los metadatos en `https://osdr.nasa.gov/osdr/data/osd/meta/{id}`.

## Datos humanos reales disponibles

### OSD-575: suero, Inspiration4

[Estudio NASA](https://osdr.nasa.gov/bio/repo/data/studies/OSD-575). Todos los nueve archivos del catálogo se descargaron. Hay dos JSON adicionales (`meta.json`, `files.json`).

| Panel | Original | Transformado | Observaciones |
| --- | --- | --- | --- |
| CMP | `LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv` | `LSDS-8_Comprehensive_Metabolic_Panel_CMP_TRANSFORMED.csv` | Original largo; transformado 28 filas y 58 columnas, con 19 analitos y rangos de referencia |
| Cardiovascular Eve | `LSDS-8_Multiplex_serum.cardiovascular.EvePanel_SUBMITTED.csv` | `LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv` | Transformado 28 filas, nueve concentraciones y nueve porcentajes derivados |
| Inmune Eve | `LSDS-8_Multiplex_serum.immune.EvePanel_SUBMITTED.csv` | `LSDS-8_Multiplex_serum_immune_EvePanel_TRANSFORMED.csv` | Transformado 28 filas, 71 concentraciones y 71 porcentajes derivados |
| Inmune Alamar | `LSDS-8_Multiplex_serum.immune.AlamarPanel_SUBMITTED.xlsx` | `LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv` | Original: hoja `SpaceX_serum_non_bridged`, 5481 filas; 203 analitos × 27 muestras |

Rutas bajo `data/raw/OSD-575/`. Las filas representan mediciones humanas reales depositadas, no datos simulados. Los originales y transformados describen las mismas muestras y no deben sumarse como observaciones independientes.

Alamar original tiene columnas `Analyte`, `Concentration`, `Timepoint`, `ID`, `Unit`, `Timepoint2`, `Type`, `Percent_normalized_value`. El transformado usa `Sample ID`, 203 columnas `*_concentration_npq` y 203 `*_percent_normalized_value`. NPQ es el log2 de recuentos normalizados por control de placa, según el protocolo ISA: **no es pg/mL ni concentración absoluta**.

Los cuatro sujetos son C001–C004. Las siete visitas son L-92, L-44, L-3, R+1, R+45, R+82 y R+194. No hay visita en vuelo para estos paneles. En Alamar falta C003/L-44. No debe confundirse este calendario con el de otro estudio Inspiration4.

El ISA confirma que los porcentajes Eve se normalizaron por la media preflight de toda la tripulación. Por ello se excluyen de entradas temporales: pueden incorporar otras personas o visitas posteriores al origen de predicción. Se usarán concentraciones originales con unidades y se ajustará cualquier normalización exclusivamente con entrenamiento.

### OSD-569: hemograma completo, Inspiration4

[Estudio NASA](https://osdr.nasa.gov/bio/repo/data/studies/OSD-569). Archivos clínicos:

- `data/raw/OSD-569/LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv`: 553 filas, nueve columnas `ANALYTE`, `VALUE`, `RANGE_MIN`, `RANGE_MAX`, `UNITS`, `TEST_TYPE`, `SUBJECT_ID`, `SEX`, `TEST_DATE`.
- `data/raw/OSD-569/LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv`: 28 filas, 61 columnas; 20 medidas más rangos.
- Cuatro sujetos y las mismas siete visitas. El original tiene 23 etiquetas de analito porque tres tienen variante `(FEMALE)`; al homologarlas corresponden a 20 medidas. Algunos analitos tienen menos de 28 filas.
- Hay una discrepancia importante: el transformado llama `hemoglobin_value_percent` a la hemoglobina; el original declara `g/dL`. Para unidades y valores se prioriza el original y se conserva la discrepancia en la auditoría.

Además se conservaron metadatos y dos archivos pequeños de checksums. Los archivos ómicos grandes y reportes comprimidos no clínicos se omitieron. Se consultó también `data/raw/OSD-570/files.json`: no se seleccionaron datos ómicos de ese estudio para esta fase.

## Alcance, acceso y condiciones

El indicador público `restricted: false` documenta acceso abierto en el catálogo, no una licencia adicional inventada por el proyecto. Se deben mantener las atribuciones NASA y publicaciones asociadas, y revisar [términos OSDR](https://science.nasa.gov/reference/osdr-help-terms-and-conditions/) para usos posteriores. No se descargaron archivos restringidos ni secuencias crudas.

Los archivos contienen tiempo de muestra relativo a lanzamiento/regreso. No se ha encontrado fecha operacional de disponibilidad de cada resultado; no debe afirmarse que un ensayo de laboratorio estuviera disponible a bordo al tomarse la muestra. Esta limitación condiciona cualquier interpretación prospectiva.

La página Resources del reto oficial falló al abrirse con la herramienta web en varias variantes. La [réplica del organizador local HackAthlone](https://hackathlone.com/challenges/health-monitoring-software) remite a documentación HRP/CSA. OSD-575/569 se incorporan como **fuentes NASA candidatas del proyecto**, sin afirmar que sean una tabla de entrenamiento prescrita por el reto.
