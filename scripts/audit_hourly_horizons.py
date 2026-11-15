"""Audit published hourly labels at exact horizons; never train or fill labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
           'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
           'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
           'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
           'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
           'Fibrinogen', 'Platelets']
HORIZONS = (1, 4, 8, 24)
CONTEXT_SLOTS = 24


def audit(raw: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    counter = {}
    origin_counter = {}
    sequences = []
    files = sorted(raw.rglob('*.psv'))
    for path in files:
        frame = pd.read_csv(path, sep='|')
        times = pd.to_numeric(frame.ICULOS, errors='coerce').to_numpy(dtype=float)
        finite_time = np.isfinite(times).all()
        integral_time = finite_time and np.equal(times, np.floor(times)).all()
        ordered = finite_time and (np.diff(times) > 0).all()
        duplicate_count = int(pd.Series(times).duplicated().sum())
        missing = [name for name in DYNAMIC if name not in frame]
        valid = integral_time and ordered and not missing
        gap_count = int((np.diff(times) > 1).sum()) if finite_time else None
        gap_slots = int(np.maximum(np.diff(times)-1, 0).sum()) if finite_time else None
        values = frame[DYNAMIC].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        present = np.isfinite(values)
        hospital = path.parent.name
        sequence_id = f'{hospital}/{path.stem}'
        info = {'sequence_id': sequence_id, 'hospital': hospital, 'rows': len(frame),
                'time_min': float(times.min()) if finite_time else None,
                'time_max': float(times.max()) if finite_time else None,
                'strictly_increasing': bool(ordered), 'integer_times': bool(integral_time),
                'duplicate_ICULOS': duplicate_count, 'gap_events': gap_count,
                'missing_hour_slots': gap_slots, 'valid_for_counts': bool(valid),
                'rows_with_any_dynamic': int(present.any(axis=1).sum()),
                'nonfinite_nonmissing_values': int((~np.isfinite(values) & ~np.isnan(values)).sum())}
        sequences.append(info)
        if not valid:
            continue
        # The sequence itself supplies the hourly slots. For a gap, complete
        # context requires every integer slot in [t-23,t]; no gap filling.
        lookup = {int(t): index for index, t in enumerate(times)}
        contexts = []
        full_flags = []
        for t in times.astype(int):
            history = [lookup[hour] for hour in range(t-CONTEXT_SLOTS+1, t+1) if hour in lookup]
            contexts.append(bool(present[history].any()))
            full_flags.append(len(history) == CONTEXT_SLOTS)
        contexts = np.asarray(contexts)
        full_flags = np.asarray(full_flags) & contexts
        info['origins_partial_context'] = int(contexts.sum())
        info['origins_full24_context'] = int(full_flags.sum())
        for horizon in HORIZONS:
            origins = np.array([index for index, t in enumerate(times.astype(int)) if t+horizon in lookup], dtype=int)
            targets = np.array([lookup[int(times[index])+horizon] for index in origins], dtype=int)
            partial = contexts[origins]
            full = full_flags[origins]
            y_present = present[targets]
            for group in ('ALL', hospital):
                key = (group, horizon)
                counts = origin_counter.setdefault(key, {'exact_future_origins': 0, 'partial_context_origins': 0,
                                                         'full24_context_origins': 0, 'sequences_with_future': 0})
                counts['exact_future_origins'] += len(origins)
                counts['partial_context_origins'] += int(partial.sum())
                counts['full24_context_origins'] += int(full.sum())
                counts['sequences_with_future'] += int(len(origins) > 0)
                for col, variable in enumerate(DYNAMIC):
                    key = (group, variable, horizon)
                    totals = counter.setdefault(key, np.zeros(6, dtype=np.int64))
                    available = y_present[:, col]
                    n_any = int(available.sum())
                    n_partial = int((available & partial).sum())
                    n_full = int((available & full).sum())
                    totals += (n_any, int(n_any > 0), n_partial, int(n_partial > 0), n_full, int(n_full > 0))
    names = ['future_labels', 'sequences_with_future_labels', 'pairs_partial_context',
             'sequences_partial_context', 'pairs_full24_context', 'sequences_full24_context']
    table = pd.DataFrame([{'hospital': hospital, 'variable': variable, 'horizon_hours': horizon,
                           **dict(zip(names, counts.tolist()))}
                          for (hospital, variable, horizon), counts in counter.items()])
    table.to_csv(output/'target_availability_by_hospital.csv', index=False)
    table[table.hospital.eq('ALL')].drop(columns='hospital').to_csv(output/'target_availability.csv', index=False)
    origin_table = pd.DataFrame([{'hospital': hospital, 'horizon_hours': horizon, **counts}
                                for (hospital, horizon), counts in origin_counter.items()])
    origin_table.to_csv(output/'origin_availability.csv', index=False)
    sequence_table = pd.DataFrame(sequences)
    sequence_table.to_csv(output/'sequence_audit.csv', index=False)
    histogram = sequence_table.groupby(['hospital', 'rows']).size().rename('sequences').reset_index()
    histogram.to_csv(output/'length_histogram.csv', index=False)
    bins = [0, 8, 24, 25, 48, 72, 168, 336, float('inf')]
    binned = sequence_table.assign(length_bin=pd.cut(sequence_table.rows, bins=bins, right=False))
    binned.groupby(['hospital', 'length_bin'], observed=False).size().rename('sequences').to_csv(output/'length_bins.csv')
    summary = {'status': 'feasibility_audit_only_no_training', 'source': 'PhysioNet Challenge 2019 v1.0.0 downloaded subset',
               'source_url': 'https://physionet.org/content/challenge-2019/1.0.0/',
               'sequences': len(files), 'rows': int(sequence_table.rows.sum()), 'dynamic_variables': DYNAMIC,
               'horizons_hours': list(HORIZONS), 'context_hour_slots': CONTEXT_SLOTS,
               'invalid_sequences_excluded': int((~sequence_table.valid_for_counts).sum()),
               'duplicate_ICULOS_rows': int(sequence_table.duplicate_ICULOS.sum()),
               'sequences_with_gaps': int(sequence_table.gap_events.gt(0).sum()),
               'length_descriptive': sequence_table.rows.describe().to_dict(),
               'hospitals': sequence_table.groupby('hospital').agg(sequences=('rows','size'), rows=('rows','sum')).reset_index().to_dict('records'),
               'origins_all_hospitals': origin_table[origin_table.hospital.eq('ALL')].drop(columns='hospital').to_dict('records'),
               'definitions': {
                   'origin': 'End of current hourly row at integer ICULOS t; label exactly at ICULOS t+h in the same sequence.',
                   'future_labels': 'Nonmissing finite published target cells at exact horizon, without requiring any input history.',
                   'partial_context': 'Between one and24 existing hourly slots in [t-23,t], with at least one finite observed value across the34 dynamic variables in that window.',
                   'full24_context': 'All24 integer hourly slots [t-23,t] exist and at least one dynamic measurement exists anywhere in that window; individual cells may be missing.',
                   'sequence_counts': 'ICU records/files, not asserted unique people across admissions.',
                   'label_independence': 'A nonmissing published hourly value is not proof of a new independently drawn sample at that hour.',
                   'uncertainty': 'q10/q50/q90 requested; sample counts do not establish predictive calibration.',
                   'aggregation': '15minute observations or within15minute mean/min/max cannot be recovered from hourly rows.'}}
    (output/'hourly_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_report(summary, table[table.hospital.eq('ALL')])
    print(json.dumps({key: value for key, value in summary.items() if key not in ('dynamic_variables', 'definitions')}, indent=2))
    return summary


def write_report(summary: dict, targets: pd.DataFrame) -> None:
    lines = ['# Viabilidad real de horizontes de 1, 4, 8 y 24 horas', '',
             'Auditoría descriptiva del subconjunto descargado de PhysioNet Challenge 2019 v1.0.0. **No se ha entrenado ni cambiado ningún modelo, partición o métrica anterior.** Los cuantiles solicitados son q10, q50 y q90; esta auditoría mide disponibilidad de etiquetas, no su precisión o calibración.', '',
             '## Definición de los recuentos', '',
             'Se analizaron 1.000 archivos de UCI: 500 del hospital A y 500 del B, con 38.153 filas horarias. Cada archivo es una secuencia; no se afirma que sean personas independientes entre posibles ingresos. No se encontraron ICULOS duplicados, desordenados, no enteros ni huecos horarios. Ninguna secuencia fue excluida. Las longitudes van de 8 a 336 filas; mediana 39, cuartiles 24 y 47.', '',
             'El origen se sitúa al final de la fila horaria actual ICULOS=t. Para h se exige una fila de la misma secuencia exactamente en t+h; no se usa la siguiente medición más cercana. La etiqueta de cada variable debe ser un valor finito publicado en esa fila. No hay forward fill, interpolación ni requisito de que las demás salidas estén disponibles. Un valor publicado no demuestra por sí solo una extracción de sangre nueva e independiente en esa hora.', '',
             'La ventana candidata comprende hasta 24 slots horarios [t−23,t], incluyendo la fila actual. Contexto parcial: entre uno y 24 slots existentes y al menos una medición finita en alguna de las 34 variables dinámicas dentro de la ventana. Contexto completo: existen los 24 slots y al menos una medición dinámica. **Completo describe cobertura de slots, no ausencia de valores perdidos**: todas las celdas restantes pueden mantenerse con máscaras. No se exige que el propio objetivo se haya medido previamente. Edad, género, unidad, tiempo de ingreso, ICULOS y SepsisLabel no cuentan como mediciones dinámicas para habilitar el contexto.', '',
             '## Orígenes candidatos antes de exigir un objetivo concreto', '',
             '| Horizonte | Fila futura exacta | Contexto parcial | 24 slots completos | Secuencias con fila futura |',
             '|---|---:|---:|---:|---:|']
    for row in summary['origins_all_hospitals']:
        lines.append(f"| {row['horizon_hours']} h | {row['exact_future_origins']} | {row['partial_context_origins']} | {row['full24_context_origins']} | {row['sequences_with_future']} |")
    lines += ['', '## Disponibilidad por salida', '',
              'Cada celda indica **pares con contexto parcial / pares con 24 slots completos**. Las columnas finales muestran cuántas secuencias aportan al menos un par a 24 h. El CSV contiene además las etiquetas futuras sin requisito de contexto y el número de secuencias para cada variable, horizonte y tipo de contexto.', '',
              '| Variable publicada | 1 h: P/F | 4 h: P/F | 8 h: P/F | 24 h: P/F | Secuencias a 24 h: P/F |',
              '|---|---:|---:|---:|---:|---:|']
    for variable in DYNAMIC:
        rows = targets[targets.variable.eq(variable)].set_index('horizon_hours')
        cells = [f"{int(rows.loc[h, 'pairs_partial_context'])}/{int(rows.loc[h, 'pairs_full24_context'])}" for h in HORIZONS]
        last = rows.loc[24]
        lines.append('| ' + variable + ' | ' + ' | '.join(cells) + f" | {int(last.sequences_partial_context)}/{int(last.sequences_full24_context)} |")
    lines += ['', '## Qué permiten concluir estas cifras', '',
              'Las constantes vitales tienen más etiquetas horarias que muchos ensayos de laboratorio. Para glucosa a 24 h hay 2.277 pares con contexto parcial repartidos en 606 secuencias, frente a 459 pares con 24 slots en 124 secuencias. Para fibrinógeno a 24 h quedan 64 pares en 36 secuencias, o 12 pares en cuatro secuencias con 24 slots. No puede decidirse una validación suficiente contando pares solapados como si fueran participantes independientes.', '',
              'Permitir historial parcial conserva bastantes ejemplos; se deben registrar por separado longitud efectiva, máscaras, antigüedad del último valor y variable/horizonte evaluados. Estos recuentos son anteriores a cualquier partición, selección de objetivos o requisito adicional de contexto; el entrenamiento y la evaluación dispondrían de menos ejemplos. El subconjunto no es una estimación representativa de toda PhysioNet ni de astronautas.', '',
              '## Resolución de 15 minutos y otras fuentes', '',
              '**No se pueden reconstruir cuatro observaciones de 15 minutos ni sus mean/min/max a partir de una única fila horaria.** Repetir o interpolar cada hora no crea información observada. Para esta fuente, la planificación debe conservar la resolución publicada de una hora; los horizontes 1/4/8/24 h sí están expresados en ese reloj.', '',
              'BIDMC aporta registros cortos de aproximadamente ocho minutos en el conjunto disponible y no proporciona etiquetas reales a una hora o más dentro de cada registro. UTMB e Inspiration4 tienen visitas clínicas separadas por días o semanas: no justifican etiquetar resultados sanguíneos a 1/4/8/24 horas. La actigrafía diaria de UTMB tampoco permite convertir sueño diario en medidas de 15 minutos.', '',
              '## Artefactos y reproducción', '',
              '- `outputs/feasibility/target_availability.csv`: 136 combinaciones de variable y horizonte; pares y secuencias para etiquetas futuras, contexto parcial y completo.',
              '- `outputs/feasibility/target_availability_by_hospital.csv`: los mismos recuentos por hospital y agregados.',
              '- `outputs/feasibility/origin_availability.csv`: orígenes candidatos antes de seleccionar salidas.',
              '- `outputs/feasibility/sequence_audit.csv`: orden, duplicados, huecos, longitudes y disponibilidad por archivo.',
              '- `outputs/feasibility/length_histogram.csv` y `length_bins.csv`: distribución exacta y agrupada de longitudes.',
              '- `outputs/feasibility/hourly_summary.json`: configuración fija, definiciones y resumen verificable.', '',
              'Reproducir con `python scripts/audit_hourly_horizons.py`. Se lee exclusivamente el subconjunto local; no se descarga nada ni se modifican los archivos originales.', '',
              'Fuente: [PhysioNet Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/).']
    (ROOT/'docs/viabilidad-horizontes-horarios.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw/external/PHYSIONET_SEPSIS_2019_1.0.0')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/feasibility')
    args = parser.parse_args()
    audit(args.raw, args.output)
