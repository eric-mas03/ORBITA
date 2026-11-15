# Auditoría de recursos del reto NASA Space Apps 2026

## Resultado verificado

La pestaña [Resources del reto oficial](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/?tab=resources) contiene **cinco enlaces científicos: tres NASA y dos CSA**. Se reconciliaron los cinco con la réplica del organizador local y se adquirieron sus páginas completas. Son recursos documentales; no se ha encontrado en ellos una tabla descargable de observaciones longitudinales de salud ni nuevas etiquetas horarias para entrenamiento.

La revisión añade cobertura documental y trazabilidad. **No añade por sí sola nuevas mediciones humanas para predecir a 1, 4, 8 o 24 horas.** Los diez archivos HTML descargados no se contabilizan como diez datasets ni como observaciones de entrenamiento.

## Evidencia de la página oficial

La captura oficial está en `data/manifests/challenge_page_v2/official_resources_page.html`: respuesta HTTP 200, 76.664 bytes. El contenido relevante estaba incluido en cadenas JSON de los scripts React Server Components `self.__next_f.push`, por lo que una extracción limitada al DOM visible podía omitirlo. El programa decodifica esos argumentos con `json.loads`; no ejecuta JavaScript de la página.

Se extrajo el registro de texto RSC `19`, declarado como `T a48` en hexadecimal: **2.632 bytes UTF-8**. Contiene exactamente los cinco enlaces del catálogo. El bloque se conserva en `data/raw/challenge_resources_v2/official_resources_extracted.html`; el manifiesto registra hashes del original y de la extracción. El campo observado `lastPublishedAt` es **2026-11-13T03:46:22.373143Z**.

La captura oficial se obtuvo con `Invoke-WebRequest -SkipCertificateCheck` después de que el cliente local no pudiera validar el emisor del certificado. Se registra expresamente **`tls_verification=false` para esa captura**, no como fallo del servidor. El listado se corroboró con la [réplica HackAthlone](https://hackathlone.com/challenges/health-monitoring-software) y con las cinco páginas primarias. Las diez adquisiciones posteriores emplearon verificación TLS habilitada. El error inicial de conexión de CSA se resolvió mediante otro cliente HTTP normal, sin credenciales ni elusión de acceso restringido.

La completitud verificada se refiere al bloque Resources de esa captura. No es una afirmación de haber inventariado todos los datos existentes de NASA, todos los estudios citados indirectamente ni futuros cambios de la web.

## Clasificación de los cinco recursos oficiales

| Recurso oficial | Página adquirida | Naturaleza y decisión para modelado |
|---|---:|---|
| [5 Hazards of Human Spaceflight](https://www.nasa.gov/hrp/hazards/) | 315.818 bytes | Descripción de peligros y artículos relacionados; contexto científico, no serie persona-tiempo-valor |
| [Risks of Human Spaceflight](https://www.nasa.gov/hrp/risks/) | 325.067 bytes | Descripciones de riesgos; no etiquetas clínicas horarias descargables identificadas en la página |
| [Ongoing HRP Research](https://www.nasa.gov/hrp/ongoing-hrp-research/) | 342.198 bytes | Investigaciones y enlaces de evidencia; una ficha de investigación no equivale a sus datos individuales |
| [Monitoring Radiation for Astronaut Health](https://www.asc-csa.gc.ca/eng/news/articles/2026/2026-02-26-monitoring-radiation-astronaut-health.asp) | 22.994 bytes | Instrumentación y exposición radiológica; no tabla de dosis personal emparejada con resultados clínicos |
| [Why Do We Conduct Science Experiments in Space?](https://www.asc-csa.gc.ca/eng/iss/science/why-do-we-conduct-science-experiments-in-space.asp) | 25.253 bytes | Divulgación y fundamento experimental; no tabla observacional reutilizable confirmada |

Los tamaños corresponden a los bytes de las capturas de esta adquisición. Los hashes SHA-256 y URLs resueltas están en `data/manifests/challenge_resources_v2.json`. La clasificación no implica que las páginas carezcan de cifras científicas; significa que no ofrecen por sí mismas una tabla longitudinal de entrenamiento verificada.

## Seguimiento de enlaces relevantes

También se adquirieron las páginas directamente relacionadas de [Human Research Roadmap](https://humanresearchroadmap.nasa.gov/), [MOSFET](https://www.asc-csa.gc.ca/eng/astronauts/space-medicine/mosfet.asp), [EVARM](https://www.asc-csa.gc.ca/eng/iss/evarm.asp) y [Radi-N2](https://www.asc-csa.gc.ca/eng/sciences/radi-n2.asp). Son catálogos de riesgo/evidencia o descripciones de experimentos. En estas páginas tampoco se identificó un archivo directo CSV, TSV, XLS/XLSX o API de observaciones clínicas persona-tiempo-valor que añada objetivos horarios al proyecto.

El Roadmap enlaza documentos, riesgos, tareas y evidencia del programa. Se conservaron esos enlaces como candidatos secundarios con su disposición y razón, sin asumir que un informe o ficha pública implique acceso a datos médicos individuales. Radi-N2 enlaza una actividad educativa; no se ha demostrado un emparejamiento con resultados de salud de astronautas. El catálogo no presenta esas páginas ni la actividad como nuevas cohortes humanas.

No se realizó un rastreo ilimitado de todos los artículos y enlaces de NASA. Los candidatos secundarios no inspeccionados exhaustivamente se marcan como tales; no se declara que en todo el repositorio sea imposible encontrar datos.

## Inventario anterior y brechas que permanecen

Se leyeron únicamente cuatro manifiestos previos, sus metadatos y documentación existente. **No se abrieron observaciones clínicas, episodios de reserva, `confirmation_holdout.json` ni predicciones/artefactos de la campaña V1.**

| Manifiesto previo | Entradas registradas como descargadas | Bytes declarados | Observación |
|---|---:|---:|---|
| `osdr_downloads.json` | 43 | 37.257.762 | Incluye metadatos y procesados; además registra 1.566 archivos omitidos y dos errores de acceso |
| `lsda_downloads.json` | 54 | 1.850.982 | UTMB clínica y actigrafía, con metadatos y ReadMe; los paneles no añaden personas por separado |
| `external_downloads.json` | 1.111 | 6.913.377 | BIDMC, PBCSEQ y subconjunto PhysioNet; son fuentes adicionales terrestres, no datos prescritos por el reto |
| `nhanes_downloads.json` | 6 | 7.354.422 | Tablas poblacionales transversales; no convierten sujetos en series horarias |

Esos números describen entradas de manifiestos, no pacientes ni mediciones independientes. En esta fase no se verificó el contenido clínico de los archivos anteriores: se preservó la separación de la reserva y de la campaña V1.

Las brechas conocidas siguen registradas explícitamente: OSD-140 tiene solo una parte del procesado descargada; su otra parte supera 160 MB. OSD-353 tiene un procesado superior a 29 GB. Ax-1 OSD-903/943 conserva metadatos pero las descargas ISA anteriores devolvieron HTTP 403. No se descargaron secuencias ómicas masivas ni se eludieron restricciones. Ninguna de esas incidencias se transforma en datos sintéticos o etiquetas inventadas.

Inspiration4 y UTMB ya aportaban observaciones de salud al proyecto, pero con sus cadencias reales. Los enlaces documentales del reto no cambian esas cadencias ni justifican interpolarlas a horas. Una ampliación de PhysioNet, si se ejecuta para una nueva experimentación, debe identificarse como fuente observacional terrestre adicional y mantenerse en artefactos V2 separados; no atribuirse a un nuevo dataset NASA encontrado en esos cinco enlaces.

## Artefactos V2 y reproducción

- `config/challenge_resources_v2.json`: catálogo reconciliado, condiciones de acceso, límites y enlaces de seguimiento.
- `scripts/acquire_challenge_resources.py`: adquisición pública limitada, extracción RSC y auditoría de manifiestos.
- `data/manifests/challenge_resources_v2.json`: estado por recurso, hashes, tamaños, intentos fallidos históricos, inventario y disposición de candidatos secundarios.
- `data/raw/challenge_resources_v2/`: diez capturas HTML y el bloque Resources oficial extraído.

```powershell
python scripts/acquire_challenge_resources.py
python scripts/acquire_challenge_resources.py --offline
```

La segunda orden actualiza la auditoría y reconciliación con los archivos existentes, sin red ni lectura de observaciones. Se verificaron los hashes y tamaños de las **diez capturas descargadas, 1.624.231 bytes**. El script conserva un máximo por archivo y requiere una lista explícita de descargas de datos con acceso público confirmado; no realiza una descarga recursiva de datos médicos ni de archivos masivos.

Resultado para la siguiente campaña: **cinco de cinco recursos oficiales incorporados documentalmente; cero nuevos datasets observacionales y cero nuevas etiquetas horarias de salud confirmadas en esos recursos y páginas de seguimiento adquiridas**. La campaña V1 permanece intacta.
