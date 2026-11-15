# Interpretación biomédica descriptiva del EDA

Revisión del 14 de noviembre de 2026 sobre `data/processed/inspiration4_observations.csv`, los archivos originales y `outputs/eda/correlations.csv`. Cuatro personas y siete visitas; los resultados siguientes describen esta muestra, sin inferencia clínica, causal ni selección de grupos a partir de correlaciones. Todos los coeficientes de las tablas son Spearman recalculado sobre pares completos.

## Redundancia metabólica confirmada

- `GLOBULIN = PROTEIN; TOTAL − ALBUMIN` en las 28 visitas-persona, con discrepancia máxima de redondeo informático inferior a 1e-15 g/dL. Es información calculada, no un tercer marcador independiente.
- `ALBUMIN/GLOBULIN RATIO` reproduce el cociente con diferencia absoluta máxima 0.0483, compatible con redondeo a una decimal. El cociente BUN/creatinina tiene solo seis valores numéricos; su diferencia máxima frente a la división de componentes es 0.455, compatible con el redondeo entero publicado. No se interpreta el patrón de ausencia como dato aleatorio demostrado.
- Creatinina y eGFR NON-AFR. AMERICAN muestran rho agrupado −0.294, pero rho −0.991 en cada una de las cuatro personas. La mezcla de niveles entre personas oculta una relación intrapersonal casi determinista y derivada. No es evidencia de un mecanismo nuevo; respalda excluir ambas variantes históricas de eGFR y los cocientes del conjunto independiente de objetivos.

Los bloques renales, proteínas y enzimas permanecen separados por significado y ensayo. La correlación no modifica su asignación. Los rangos del laboratorio tampoco se convierten en límites de salud astronauta.

## Eotaxin: patrón amplio, heterogéneo entre personas

Frente a los otros 79 analitos Eve, Eotaxin presenta 18 correlaciones agrupadas >0.6 y siete >0.8; la mediana es 0.167 y 30 son negativas. Por tanto, existe un subconjunto de asociaciones elevadas, pero no un aumento común de todo el panel. Son concentraciones originales, sin utilizar el campo Percent.

| Pareja con Eotaxin | n | rho agrupado | C001 | C002 | C003 | C004 |
|---|---:|---:|---:|---:|---:|---:|
| EVE_CARDIO PF4 | 28 | 0.866 | 0.714 | 0.857 | 0.857 | 0.464 |
| EVE_CARDIO a-2 Macroglobulin | 28 | 0.859 | 0.536 | 0.964 | 0.857 | −0.143 |
| EVE_CARDIO SAP | 28 | 0.833 | 0.393 | 0.786 | 0.714 | 0.607 |
| EVE_IMMUNE Fractalkine | 28 | 0.885 | 0.893 | 0.714 | 0.893 | 0.000 |
| EVE_IMMUNE MCP-3 | 28 | 0.893 | 0.821 | 0.857 | 0.893 | −0.071 |

Cada correlación individual tiene solo siete pares. La dirección PF4/Eotaxin es positiva en las cuatro personas; las asociaciones con a-2 Macroglobulin, Fractalkine y MCP-3 no son igualmente estables en C004. No deben presentarse como relaciones personales universales.

**Comprobación de visita:** L-44 es el mínimo de Eotaxin en las cuatro personas; su mediana es 45.96 pg/ml, frente a 101.97 en L-92 y 90.02 en L-3. La mediana por visita de las puntuaciones z por analito también es especialmente baja en L-44 para los paneles Eve inmune y cardiovascular. Esta estandarización usa todo el conjunto solo para descripción, nunca para crear features del modelo.

El patrón compartido justifica revisar factores preanalíticos, visita y ensayo, pero no prueba un artefacto de lote. Al retirar L-44, Eotaxin/PF4 mantiene rho 0.806 con n=24. Al ajustar rangos globales mediante regresión con indicadores de sujeto y visita, su correlación residual es 0.881; a-2 Macroglobulin y SAP mantienen 0.814 y 0.794. Estas correlaciones residuales son métricas descriptivas, **no Spearman ordinario ni control de toda confusión**. La visita compartida no basta para explicar esas parejas. Faltan metadatos analíticos vinculados —por ejemplo placa, lote y orden de procesamiento— para contrastar una explicación técnica; tampoco se ha demostrado una explicación biológica.

## Homólogos entre Alamar y Eve

Se revisan tres parejas candidatas por identidad del analito, no por seleccionar correlaciones máximas. Se emparejan sujeto y visita; n=27, porque Alamar carece de C003 en L-44. Las correlaciones individuales usan n=7 excepto C003, n=6.

| ALAMAR / EVE_IMMUNE | rho agrupado | C001 | C002 | C003 | C004 |
|---|---:|---:|---:|---:|---:|
| il6 / IL-6 | 0.443 | 0.714 | 0.786 | 0.200 | 0.893 |
| ccl2 / MCP-1 | 0.304 | 0.607 | −0.071 | 0.543 | 0.393 |
| ifng / IFNγ | 0.226 | −0.643 | −0.643 | 0.429 | 0.393 |

IL-6 ofrece concordancia de orden positiva en las cuatro personas, de intensidad variable. CCL2 y especialmente IFNγ muestran heterogeneidad de dirección. Esto impide asumir intercambiabilidad de plataformas, incluso para el mismo nombre molecular. Alamar expresa NPQ en escala log2 de cuentas normalizadas IPC; Eve informa pg/ml. Spearman compara orden, **no concordancia cuantitativa, sesgo ni equivalencia de unidades**. No se convierten NPQ a pg/ml ni se promedian los resultados. Las parejas son candidatas de control de consistencia y posteriores análisis analíticos, no calibraciones ya validadas.

## Consecuencia operativa

Se mantiene el agrupamiento biomédico actual y la identidad de cada ensayo. La siguiente comparación debe informar cambios y errores por persona, ablaciones de variables derivadas/redundantes y sensibilidad a visitas concretas. Alamar continúa deshabilitado para el primer modelo mientras se completa su anotación. Los resultados de este EDA completo son exploratorios; ninguna división posterior del mismo conjunto recupera una prueba confirmatoria intacta.

Reproducción de los coeficientes de las tablas: pivotar `value` por índice `(subject_id,timepoint)` y columnas `variable`, seleccionar cada pareja y descartar ausencias, calcular `DataFrame.corr(method='spearman')` sobre todos los pares y repetir tras agrupar por sujeto. Para el ajuste de visita, rankear globalmente cada pareja completa, ajustar ambos rangos por mínimos cuadrados a intercepto e indicadores de sujeto y visita, y correlacionar los residuos con Pearson. No se han calculado p-valores suponiendo independencia de las 28 visitas-persona.
