"""Audit genuinely observed CGM labels and meal alignment; do not interpolate or fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/research_expansion_20261114/BIGIDEAS_1.1.3'
OUT=ROOT/'outputs/research_expansion_20261114'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0] for line in (DATA/'SHA256SUMS.txt').read_text().splitlines() if line.strip()}
    verified=[]
    for path in DATA.rglob('*'):
        if not path.is_file() or path.name=='SHA256SUMS.txt':continue
        name=path.relative_to(DATA).as_posix()
        assert name in checks,f'Missing upstream checksum: {name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==checks[name],f'Checksum mismatch: {name}'
        verified.append(name)
    rows=[]
    for sid in range(1,17):
        subject=f'{sid:03d}'
        frame=pd.read_csv(DATA/subject/f'Dexcom_{subject}.csv')
        timestamp=pd.to_datetime(frame['Timestamp (YYYY-MM-DDThh:mm:ss)'],errors='coerce')
        value=pd.to_numeric(frame['Glucose Value (mg/dL)'],errors='coerce')
        valid=timestamp.notna()&value.notna()
        series=pd.Series(value[valid].to_numpy(),index=pd.DatetimeIndex(timestamp[valid])).sort_index()
        duplicate_count=int(series.index.duplicated().sum())
        series=series.groupby(level=0).mean()
        grid=series.resample('15min',label='right',closed='right').mean()
        history_count=grid.notna().rolling(96,min_periods=96).sum()
        food=pd.read_csv(DATA/subject/f'Food_Log_{subject}.csv')
        food_schema_ok='time_begin' in food.columns
        if food_schema_ok:
            food_time=pd.to_datetime(food['time_begin'],format='mixed',errors='coerce')
        else:
            # Subject 003 has no usable header. Keep bytes intact; do not guess nutrient columns.
            food=pd.read_csv(DATA/subject/f'Food_Log_{subject}.csv',header=None)
            food_time=pd.Series(pd.NaT,index=food.index,dtype='datetime64[ns]')
        row={'subject':subject,'cgm_measurements':int(valid.sum()),'duplicate_timestamps':duplicate_count,
             'first':series.index.min().isoformat(),'last':series.index.max().isoformat(),
             'span_days':(series.index.max()-series.index.min()).total_seconds()/86400,
             'median_interval_minutes':float(series.index.to_series().diff().dt.total_seconds().median()/60),
             'observed_15min_bins':int(grid.notna().sum()),'total_15min_bins':len(grid),
             'coverage_15min':float(grid.notna().mean()),'food_rows':len(food),
             'food_schema_verified':food_schema_ok,
             'food_rows_inside_cgm_span':int(food_time.between(series.index.min(),series.index.max()).sum()) if food_schema_ok else None}
        for horizon in (1,4,8,24):
            future=grid.shift(-horizon*4).notna()
            row[f'observed_target_pairs_{horizon}h']=int((future&grid.notna()).sum())
            row[f'pairs_24h_history_80pct_{horizon}h']=int((future&(history_count>=77)).sum())
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'bigideas_cgm_coverage.csv',index=False)
    hr=pd.read_csv(DATA/'001/HR_001.csv');hr.columns=hr.columns.str.strip()
    times=pd.to_datetime(hr['datetime'],format='%m/%d/%y %H:%M',errors='coerce')
    hr_values=pd.to_numeric(hr['hr'],errors='coerce')
    hr_grid=pd.Series(hr_values.to_numpy(),index=pd.DatetimeIndex(times)).dropna().resample('15min',label='right',closed='right').mean()
    summary={'dataset':'BIG IDEAs 1.1.3','domain':'terrestrial_wearable','participants':len(result),'upstream_hashes_verified':len(verified),
             'cgm_measurements':int(result.cgm_measurements.sum()),'food_rows':int(result.food_rows.sum()),
             'observed_15min_bins':int(result.observed_15min_bins.sum()),
             'span_days_min':float(result.span_days.min()),'span_days_max':float(result.span_days.max()),
             'coverage_15min_min':float(result.coverage_15min.min()),'coverage_15min_max':float(result.coverage_15min.max()),
             'food_schema_issues':result.loc[~result.food_schema_verified,'subject'].tolist(),
             'pairs_24h_history_80pct':{str(h):int(result[f'pairs_24h_history_80pct_{h}h'].sum()) for h in (1,4,8,24)},
             'pulse_pilot':{'participants_downloaded':1,'rows':len(hr),'unparsed_timestamps':int(times.isna().sum()),'observed_15min_bins':int(hr_grid.notna().sum()),'timestamp_resolution':'minute labels repeated across approximately second-rate samples; do not invent subminute timestamps'},
             'limitations':['Interstitial glucose differs from laboratory blood glucose.','Skin temperature differs from core temperature.','Cohort is terrestrial and not representative of astronauts.','Overlapping windows are not independent samples; split by person and purge time boundaries.','Future actual meals must not enter forecasts; future meal scenarios must be supplied explicitly.','No fitting, test evaluation, interpolation, imputation or causal intervention estimation performed.']}
    (OUT/'pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
