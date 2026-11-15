# Protocolo de campaña y confirmación reservada

14 de noviembre de 2026. Se amplían ensayos individuales; el ensemble queda para una fase posterior. Los nuevos candidatos se ajustan en train y se seleccionan en desarrollo. La antigua prueba ya inspeccionada conserva valor descriptivo, pero no se reutiliza como si fuese una confirmación intacta ni dirige nuevos hiperparámetros.

## Criterio operativo previo a la confirmación

La referencia es la regresión cuantílica lineal congelada, con pinball macro estandarizado de desarrollo **0,156520754**. Se propone para priorizar candidatos individuales una mejora relativa de al menos **5 %**, equivalente a **≤0,148694716** con esta referencia. El contrato de campaña añade cobertura marginal macro **≥75 %** y tasa de bandas fuera del soporte físico no superior a la referencia en desarrollo.

Estos límites son criterios de ingeniería del proyecto, no umbrales clínicos ni una declaración de beneficio sanitario aprobada por el usuario. El nominal q10–q90 continúa siendo 80 %; pasar el mínimo operativo del 75 % no significa estar calibrado. Si ningún modelo nuevo pasa la puerta, se puede confirmar descriptivamente el mejor candidato nuevo previamente seleccionado, identificándolo como no competitivo; no se rebaja después el umbral para presentar un éxito.

Se informa conjuntamente de pinball, MAE, cobertura, anchura, frecuencia bajo cada cuantil y violaciones de soporte. Una mejora de cobertura obtenida solo ensanchando bandas no basta para afirmar mejor pronóstico. Los cuantiles permanecen ordenados, pero eso no garantiza soporte físico; no se aplica clipping posterior a resultados de prueba. Un eventual método que imponga soporte o recalibre bandas tendría que especificarse y congelarse con datos de desarrollo/calibración antes de la confirmación.

Las 34 variables y cuatro horizontes permanecen visibles. Cada celda conserva etiquetas, episodios, máscara, calidad y estado de unidades, incluso si es difícil o escasa. Una celda sin etiquetas no recibe error cero y una salida sin entrenamiento se identifica como fallback/no entrenada. Las unidades pendientes de FiO2, Magnesium y Lactate no se reinterpretan clínicamente.

## Ensayos y controles de lectura

- Los árboles usan las mismas nueve características causales por variable; la variante de grupo concatena esas características de miembros del bloque biomédico. HGB ajusta pérdidas cuantílicas sin validación aleatoria interna. El bosque cuantílico usa distribuciones de respuestas dentro de hojas, no cuantiles de predicciones medias de árboles.
- Las redes residuales parten de los cuantiles de la lineal congelada. La corrección cero reproduce exactamente la referencia y constituye un candidato de época cero; solo se conserva una actualización si mejora desarrollo. Mantienen la secuencia histórica, el estado propio y las máscaras.
- La atención bidireccional sobre una ventana ya truncada en el origen es legal: todos sus tokens pertenecen al pasado disponible. No permite solicitar después un corte intermedio conservando resúmenes calculados con datos posteriores a ese corte.
- La memoria específica de grupo recibe el historial latente fusionado; no constituye aislamiento causal de sistemas biológicos. Los contrastes modifican capacidad y estructura además del nombre del bloque.
- El control residual MLP sobre resúmenes propios comprueba si una ganancia requiere memoria/atención o puede aparecer con una corrección más simple. No debe confundirse con un ensemble aprendido.

El presupuesto, semillas, características, pérdida y candidatos se registran antes de abrir confirmación. El promedio macro da peso igual a las celdas variable×horizonte con soporte; no vuelve independientes las ventanas superpuestas ni convierte una celda con pocas etiquetas en evidencia fuerte. Repetir búsquedas en desarrollo puede optimizar particularidades de ese conjunto: por eso se reserva una confirmación nueva y no se presupone que una mejora pequeña sea reproducible.

## Reserva nueva efectivamente adquirida

Se congeló una selección uniforme sin reemplazo con semilla **20261114**, distinta de la usada para el subconjunto inicial, a partir de los índices públicos cacheados. Contiene **300 episodios nuevos, 150 por cada sitio**, excluyendo los nombres de archivo existentes tanto en raw como en el manifiesto previo. Se descargaron **301 archivos**, incluida licencia, por **1.878.958 bytes**, sin fallos y con **cero solapamientos de IDs publicados** con los 1.000 episodios iniciales.

La adquisición inspeccionó únicamente cabecera, número de campos/filas y hashes de bytes. No se resumieron valores, desenlaces, calidad numérica, frecuencias de etiquetas ni rendimiento; no se evaluó ningún modelo. Las longitudes no se utilizaron para volver a seleccionar episodios favorables. Los índices originales se referencian por hash sin duplicar sus 4,68 MB en la reserva.

Los archivos están en `data/raw/external/PHYSIONET_CONFIRMATION/`; el manifiesto inmutable de adquisición es `data/manifests/confirmation_holdout.json`. La selección tiene SHA-256 `ae6ff54db91039e43506c51f8a82e60569ba1c797cc39e6e0e5ec13a6c2c73e5`; se verificó la correspondencia uno a uno de archivos y hashes. El inventario permanece fuera del dataset horario original y de sus particiones train/dev/calibration/antigua prueba.

Se trata de nuevos archivos/episodios de los mismos dos sistemas UCI. No se puede establecer identidad única de personas entre ingresos a partir de los metadatos disponibles, y no se infieren identidades mediante valores clínicos. No es validación en un hospital nuevo, cohorte astronauta ni futuro calendario prospectivo. La discrepancia entre licencia declarada en la página y LICENSE distribuido permanece documentada; no se deduce una autorización distinta de redistribución.

## Apertura y evaluación única

El evaluador `scripts/evaluate_confirmation_campaign.py` exige la ruta canónica **`outputs/model_campaign/selection_frozen.json`**. Una selección de smoke test en otra carpeta no sirve. Debe registrar selección solo en desarrollo, hashes de evidencias, lineal de referencia y uno o dos nuevos candidatos. El evaluador verifica esos hashes y las fuentes/modelos antes de inferir; registra pesos, selección, contrato y código utilizados. Una salida previa de confirmación o una ejecución iniciada bloquean una repetición/sobrescritura automática.

Solo después de esa congelación y del aviso de coordinación se abre la reserva para evaluación. El constructor se ejecuta sobre su directorio aislado; todos los `episode_split` pasan a **confirmation antes de cualquier preparación**. Se reutilizan exclusivamente media, escala, mediana, bandas y pesos del entrenamiento original. No se llama a rutinas de ajuste, se combinan nuevos episodios con train ni se aprende un normalizador de la reserva. El packed horario original no se reescribe.

Se exportan exclusivamente los métodos de `selected_for_confirmation` con el mismo contrato: episodio, sitio, origen, intervalo terminal, horizonte, variable, unidad, etiqueta observada, q10/q50/q90, contexto, historia propia, escala congelada y soporte de entrenamiento. `summarize_campaign(..., split='confirmation')` calcula comparaciones sobre las mismas claves reales; la selección no se modifica según el ranking resultante. Se conservan métricas por variable/horizonte, cabeza, sitio, contexto y presencia de historia propia, incluidos los casos escasos.

Los hashes de modelos, selección, código y fuentes se vuelven a verificar al terminar. El manifiesto de evaluación registra la apertura posterior a la congelación; el manifiesto de adquisición mantiene su descripción histórica de inspección estructural. No se recalibra, optimiza un ensemble ni selecciona una arquitectura a partir de esta reserva. Si el resultado no confirma la mejora, se informa así; una nueva ronda necesitaría otra reserva realmente no inspeccionada.

Los intervalos de incertidumbre de diferencias se calculan con bootstrap emparejado por episodios y estratificado por sitio, conservando la dependencia de filas de un episodio. El análisis de incertidumbre con soporte suficiente puede omitir celdas con menos de diez episodios, pero debe mostrar cuáles y mantenerlas en el informe completo. Ese subconjunto no debe confundirse con la macro de todas las variables/horizontes.

## Condición para estudiar ensemble después

No se entrena un ensemble en esta campaña. Cuando existan componentes individuales competitivos, una comparación nueva podrá estudiar agregación limitada y pesos seleccionados solo con desarrollo. Pesos no negativos comunes entre q10/q50/q90 preservan orden si los componentes ya están ordenados; promediar cuantiles no es, por sí solo, obtener los cuantiles de una mezcla de distribuciones. La amplitud y soporte deberán auditarse también tras agregar. Ninguna ponderación se aprende con confirmación y su resultado no se usa para elegir retrospectivamente componentes.

## Verificación de software

La adquisición incluye seis pruebas de selección reproducible/exclusión, congelación previa, integridad, estructura opaca y rechazo de sobrescritura/cambios. El evaluador añade ocho pruebas de ruta canónica, evidencia, shortlist, reserva de salidas, particiones confirmation, packing aislado, estadísticas congeladas y exportación de etiquetas reales sin reparar cruces. Se ejecutaron **14 pruebas sin fallos**, con fixtures de software ajenos a los archivos reservados. La inferencia real de confirmación aún no se ha ejecutado desde este documento.
