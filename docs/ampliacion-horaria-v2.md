# Auditoría de ampliación horaria V2

Se verificó el contenido oficial de Resources: cinco enlaces documentales, sin tablas horarias nuevas. Se descargaron las cinco fuentes y enlaces de seguimiento para contexto y recuperación documental; no se contaron como observaciones.

La ampliación observacional independiente proviene de PhysioNet Challenge 2019, UCI terrestre. Se congelaron las identidades antes de descargarlas: 1.000 episodios nuevos de entrenamiento, 200 para aprender el ensemble y 300 para prueba final. No hay solapamiento con los 1.000 episodios originales ni los 300 de confirmación consumidos.

El NPZ no final reúne 1.600 episodios de entrenamiento (56,669 orígenes), 150 de desarrollo, 200 META y 100 de calibración. Se excluyeron la prueba original y la confirmación anterior. Los 300 finales solo tienen comprobación estructural y hashes; no se empaquetan ni evalúan antes del freeze final.

Entrenamiento: 60,765 filas fuente únicas, 1,216,008 pares etiqueta–origen–horizonte frente a 467,033 antes. El mínimo por variable/horizonte pasa a 23 etiquetas, y quedan 0 combinaciones con menos de veinte. Los pares solapados no son observaciones independientes.

Se conservan 34 variables, horizontes 1/4/8/24 h y contexto de hasta 24 horas con datos horarios. No se fabrica una resolución de 15 minutos ni se emparejan registros de personas diferentes. Se aplica la misma política de calidad congelada.

Las tablas en `outputs/data_audit_v2/` incluyen cobertura y etiquetas por variable/horizonte, correlaciones y número de pares simultáneos. Solo se han analizado valores del entrenamiento para este EDA; META, calibración y prueba final no dirigen la exploración. Las correlaciones dependen del muestreo clínico, y no demuestran relaciones causales.

El script de adquisición guarda roles y hashes en `data/manifests/hourly_expansion_v2.json`; el dataset, sus roles y auditoría están en `data/processed/hourly_v2/`. Se preservan los originales y todos los artefactos V1.
