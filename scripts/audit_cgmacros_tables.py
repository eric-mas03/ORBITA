"""Screen downloaded release tables for timing and interpolation concerns, without training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/research_expansion_20261114/CGMACROS_1.0.0/tables'
OUT=ROOT/'outputs/research_expansion_20261114'

if __name__=='__main__':
    rows=[]
    for path in sorted(DATA.glob('CGMacros-*.csv')):
        frame=pd.read_csv(path)
        if 'Timestamp' not in frame:continue
        time=pd.to_datetime(frame['Timestamp'],errors='coerce')
        glucose=pd.to_numeric(frame['Dexcom GL'],errors='coerce')
        values=glucose.dropna().to_numpy()
        valid_triplet=glucose.notna()&glucose.shift(1).notna()&glucose.shift(-1).notna()
        linear=((glucose.shift(-1)-2*glucose+glucose.shift(1)).abs()<1e-8)&valid_triplet
        rows.append({'subject':path.stem,'rows':len(frame),'span_days':(time.max()-time.min()).total_seconds()/86400,
                     'median_row_interval_seconds':float(time.diff().dt.total_seconds().median()),
                     'dexcom_noninteger_fraction':float((np.abs(values-np.round(values))>1e-8).mean()),
                     'dexcom_linear_interior_fraction':float(linear.sum()/max(1,valid_triplet.sum())),
                     'meal_rows':int(frame['Meal Type'].notna().sum()),
                     'forecasting_eligibility':'quarantined_pending_native_sensor_anchors'})
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'cgmacros_table_audit.csv',index=False)
    summary={'dataset':'CGMacros 1.0.0','domain':'terrestrial_wearable','participant_tables_downloaded':len(rows),
             'table_rows':sum(r['rows'] for r in rows),'meal_rows':sum(r['meal_rows'] for r in rows),
             'forecasting_eligibility':'quarantined_pending_native_sensor_anchors',
             'reason':'One-minute table cadence is denser than native 5-min Dexcom / 15-min Libre. Fractional glucose and long linear runs are observed. Do not treat interpolated inputs or targets as independent real measurements.',
             'example':rows[0] if rows else None,'fitted':False,'frozen_campaigns_modified':False}
    (OUT/'cgmacros_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
