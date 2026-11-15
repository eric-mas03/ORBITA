# Repetición V1 frente a V2 con escala común

Se usan los mismos 150 episodios de desarrollo, los mismos objetivos observados y la escala de reporte V1 para ambas campañas. Cada campaña conserva sus propios normalizadores aprendidos solo en su entrenamiento. Así un cambio de desviación estándar no se presenta como una mejora del predictor.

| Método | Pinball V1 | Pinball V2 | Mejora relativa | Cobertura V2 |
|---|---:|---:|---:|---:|
| group_temporal_residual | 0.155083 | 0.150175 | +3.16 % | 77.9 % |
| causal_residual | 0.155867 | 0.150436 | +3.48 % | 76.4 % |
| window_bidirectional_residual | 0.156002 | 0.150821 | +3.32 % | 76.3 % |
| quantile_forest | 0.158836 | 0.151465 | +4.64 % | 82.5 % |
| own_summary_mlp_residual | 0.155920 | 0.151830 | +2.62 % | 78.7 % |
| linear_quantile | 0.156521 | 0.152963 | +2.27 % | 79.0 % |
| hist_boost_group_features | 0.163121 | 0.153820 | +5.70 % | 72.5 % |
| hist_boost_quantile | 0.163055 | 0.154200 | +5.43 % | 74.0 % |
| transformer_grouped | 0.175066 | 0.172485 | +1.47 % | 79.6 % |
| gru_shared_readout | 0.178455 | 0.175867 | +1.45 % | 81.2 % |
| persistence | 0.179530 | 0.179325 | +0.11 % | 80.1 % |

Es una comparación descriptiva: aumentó la cohorte, cambiaron los normalizadores y se aproximó el presupuesto por número de actualizaciones en lugar de igualar épocas. Desarrollo se reutiliza para seleccionar. La prueba nueva se informa por separado después de congelar el ensemble y su calibración.
