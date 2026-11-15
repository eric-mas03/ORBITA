# Campaña de modelos individuales

Selección realizada únicamente en desarrollo. El ensemble no se entrena en esta campaña.

Se conservan episodios, etiquetas, horizontes 1/4/8/24 h, máscaras y escalas. Cada red mantiene todos los pasos temporales. Las fuentes actuales son horarias: no hay mediciones de 15 min fabricadas.

## Comparación de desarrollo

| Método | Pinball macro ↓ | MAE macro ↓ | Cobertura media | Bandas fuera de soporte | Mejora pinball frente a lineal |
|---|---:|---:|---:|---:|---:|
| group_temporal_residual | 0.155083 | 0.4567 | 74.1 % | 4.07 % | 0.9 % |
| causal_residual | 0.155867 | 0.4586 | 75.8 % | 3.70 % | 0.4 % |
| own_summary_mlp_residual | 0.155920 | 0.4590 | 77.8 % | 4.03 % | 0.4 % |
| window_bidirectional_residual | 0.156002 | 0.4591 | 77.1 % | 3.29 % | 0.3 % |
| linear_quantile | 0.156521 | 0.4607 | 77.8 % | 3.46 % | 0.0 % |
| quantile_forest | 0.158836 | 0.4774 | 84.4 % | 0.00 % | -1.5 % |
| hist_boost_quantile | 0.163055 | 0.4816 | 70.8 % | 0.17 % | -4.2 % |
| hist_boost_group_features | 0.163121 | 0.4801 | 69.1 % | 0.11 % | -4.2 % |
| transformer_grouped | 0.175066 | 0.5247 | 75.5 % | 4.00 % | -11.8 % |
| gru_shared_readout | 0.178455 | 0.5363 | 75.6 % | 4.83 % | -14.0 % |
| persistence | 0.179530 | 0.5004 | 79.8 % | 7.04 % | -14.7 % |

## Selección congelada

Referencia: lineal cuantílica. Nuevos candidatos para confirmar: group_temporal_residual.

Ningún candidato nuevo cumple todos los criterios operativos de esta ronda. La confirmación del mejor nuevo será descriptiva; no se declara una mejora sólida ni se activa el ensemble.

El objetivo operativo de esta ronda es mejorar pinball al menos un 5 % en desarrollo, con cobertura macro ≥75 % y tasa de bandas físicamente inválidas no superior a la referencia. Es un criterio de ingeniería propuesto para priorizar candidatos; no una validación clínica ni una garantía del 80 %. Se conservan resultados por variable/cabeza, incluidos los escasos.

## Qué comparan las arquitecturas

Los Transformers residuales conservan todos los latentes horarios y aprenden una corrección de la predicción lineal congelada, con rutas auxiliares del estado y resúmenes propios. El punto de partida sin corrección participa en la selección. El control `own_summary_mlp_residual` usa solo los nueve resúmenes causales propios y el horizonte; no tiene atención. Mejorar frente al Transformer original no demuestra por sí solo que la atención añada valor frente a la referencia lineal o al control sin atención.

La atención bidireccional se limita al historial conocido hasta el origen. El modelo especializado añade una memoria temporal por grupo que recibe los latentes comunes fusionados; no aísla los grupos de entrada. Los tamaños, semillas y presupuestos figuran en los manifiestos. Una sola semilla constituye un ensayo inicial, y los intervalos sobre episodios no sustituyen a repetir entrenamientos.

La nueva confirmación se abre solo después de guardar `selection_frozen.json`. No se usa para afinar ni seleccionar otra arquitectura. Las métricas antiguas de prueba permanecen como resultados exploratorios separados. Los datos siguen siendo UCI terrestre, con episodios no necesariamente deduplicables como personas.

## Artefactos

`outputs/model_campaign/` contiene tablas por variable/horizonte y cabeza, ranking de desarrollo, selección congelada y gráficos. Los modelos y sus presupuestos están en `outputs/hourly_trees/` y `outputs/hourly_architecture_trials/`. Las bandas todavía no están recalibradas. Las unidades pendientes de revisión siguen identificadas.

Reproducir: `python scripts/report_model_campaign.py`. La inferencia de confirmación, si se ejecuta después de congelar modelos, se documenta en su informe separado.
