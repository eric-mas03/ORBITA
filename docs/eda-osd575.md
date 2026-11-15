# Análisis exploratorio real de Inspiration4: OSD-575 y OSD-569

Generado por `python scripts/analyze_osd575.py`. Datos publicados, sin imputación ni interpolación.

Se han analizado **4 personas, 28 visitas-persona, 322 variables de ensayo y 8783 valores numéricos observados**.

| Ensayo | Variables | Visitas-persona | Valores observados |
|---|---:|---:|---:|
| ALAMAR | 203 | 27 | 5481 |
| CBC | 20 | 28 | 553 |
| CMP | 19 | 28 | 509 |
| EVE_CARDIO | 9 | 28 | 252 |
| EVE_IMMUNE | 71 | 28 | 1988 |

## Estructura temporal y calidad

Visitas: L-92, L-44, L-3, R+1, R+45, R+82, R+194. No hay observaciones durante el vuelo en estas tablas. Los intervalos son irregulares: L-92→L-44 son 48 días; L-44→L-3 son 41; R+1→R+45 son 44; R+45→R+82 son 37; R+82→R+194 son 112. El salto entre L y R requiere fechas exactas adicionales y no se fabrica.

La matriz conjunta tiene un 2.58 % de celdas ausentes. Alamar no contiene C003 en L-44. BUN/CREATININE RATIO solo contiene 6 valores numéricos de 28 visitas; no se completa usando futuros valores. Las cifras y rangos publicados se conservan sin convertir desviaciones en diagnósticos.

Se excluyen todas las columnas Percent, percent y percent_normalized_value, todos los rangos de referencia como entradas y las copias transformadas de ensayos ya incluidos. Alamar se conserva en NPQ log2 y nunca se mezcla con pg/ml como si fueran la misma medición. Valores cero se conservan; ausencia de flags de límite de detección no demuestra ausencia de censura.

## Correlaciones exploratorias

`correlations.csv` contiene Spearman agrupado (rangos recalculados por par), correlación de rangos marginales centrados por persona (centrado sobre pares completos), número de parejas, sujetos y rango de correlaciones individuales. Los rangos marginales se calculan con todas las observaciones disponibles de cada variable, por lo que la segunda métrica no es exactamente una correlación Spearman de pares completos. Este centrado elimina diferencias de nivel entre sujetos, pero no elimina tendencia de visita, confusión ni dependencia temporal. No se publican p-valores de independencia ni se presentan asociaciones como causales. Los valores máximos entre miles de pares sufren selección múltiple.

Los siguientes pares entre ensayos son ejemplos extremos para inspección; no una selección de features ni evidencia confirmatoria:

| Variable A | Variable B | Pares | rho agrupado | rho centrado por persona |
|---|---|---:|---:|---:|
| EVE_CARDIO::PF4 | EVE_IMMUNE::Eotaxin | 28 | 0.866 | 0.870 |
| EVE_CARDIO::a-2 Macroglobulin | EVE_IMMUNE::Eotaxin | 28 | 0.859 | 0.851 |
| EVE_CARDIO::SAP | EVE_IMMUNE::Eotaxin | 28 | 0.833 | 0.840 |
| ALAMAR::mpo | EVE_IMMUNE::IL-1RA | 27 | 0.368 | 0.822 |
| EVE_CARDIO::PF4 | EVE_IMMUNE::Fractalkine | 28 | 0.828 | 0.818 |
| EVE_CARDIO::a-2 Macroglobulin | EVE_IMMUNE::MCP-3 | 28 | 0.815 | 0.815 |
| EVE_CARDIO::SAP | EVE_IMMUNE::Fractalkine | 28 | 0.805 | 0.794 |
| ALAMAR::il17rb | EVE_IMMUNE::IL-1β | 27 | -0.795 | -0.791 |
| EVE_CARDIO::a-2 Macroglobulin | EVE_IMMUNE::Fractalkine | 28 | 0.809 | 0.790 |
| EVE_CARDIO::CRP | EVE_IMMUNE::Eotaxin | 28 | 0.714 | 0.775 |

## Implicación para el modelo

Se puede implementar y verificar la arquitectura por encoders y cabezas temporales con estas observaciones. Cuatro personas y siete visitas no sostienen una validación convincente de un transformer de salud ni predicciones continuas por horas. La primera ejecución de entrenamiento será un experimento técnico de siguiente visita con horizonte irregular, comparado con persistencia y etiquetado como exploratorio.

Los rangos históricos de eGFR por categoría racial son variables derivadas del archivo original; no se usarán como objetivos ni para personalización. Las razones derivadas tampoco serán el criterio de agrupamiento. El agrupamiento se realiza con criterio biomédico y ensayo, y la correlación sirve para revisarlo, sin reemplazar ese criterio.

El tiempo de disponibilidad del resultado no consta: el experimento será retrospectivo bajo un supuesto explícito. Los datos del conjunto se han explorado completos, por lo que cualquier partición posterior será una comprobación preliminar, no una prueba confirmatoria intacta.

El hemograma se toma del archivo original de OSD-569: HEMOGLOBIN usa g/dL, corrigiendo la etiqueta errónea percent de la copia transformada sin cambiar el valor. Se armonizan los sufijos (FEMALE) de eritrocitos, hemoglobina y hematocrito, preservando la etiqueta original. No se añaden sexo ni rangos como predictores.

Fuentes: https://osdr.nasa.gov/bio/repo/data/studies/OSD-575 y https://osdr.nasa.gov/bio/repo/data/studies/OSD-569 . URLs, tamaños y SHA-256 en `data/manifests/osdr_downloads.json`.
