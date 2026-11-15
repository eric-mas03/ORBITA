"""Audit new NASA observations, preserving domains, reported labels and time scales."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/research_expansion_20261114'
OUT=ROOT/'outputs/research_expansion_20261114'

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/manifests/nasa_followup_20261114.json').read_text())
    verified=0
    for group in manifest:
        for record in group['records']:
            if record['status']!='downloaded':continue
            payload=(ROOT/record['path']).read_bytes()
            assert hashlib.sha256(payload).hexdigest()==record['sha256']
            if record.get('catalog_md5'):
                assert hashlib.md5(payload).hexdigest()==record['catalog_md5'];verified+=1
    records=[]
    for label in ('BRSMCF_C1','BRSMCF_C3','BRSMCF_CFT70'):
        for path in sorted((RAW/'NASA_LSDA'/label).glob('*.csv')):
            frame=pd.read_csv(path,dtype=str)
            subject='Subject' if 'Subject' in frame else None
            metadata=[c for c in ('Subject','Campaign','Test_Phase','BR_Day','Day','Date') if c in frame]
            row={'dataset':label,'file':path.name,'rows':len(frame),'columns':len(frame.columns),
                 'subject_ids':sorted(frame[subject].dropna().unique().tolist()) if subject else [],
                 'subject_count':int(frame[subject].nunique()) if subject else None,
                 'distinct_visit_keys':len(frame[metadata].drop_duplicates()) if metadata else None,
                 'has_calendar_date':'Date' in frame,'has_hourly_timestamps':False,
                 'column_names':list(frame.columns),'domain':'NASA_ground_space_analog'}
            if {'SBP','DBP'}.issubset(frame):
                sbp=pd.to_numeric(frame['SBP'],errors='coerce');dbp=pd.to_numeric(frame['DBP'],errors='coerce')
                row['paired_bp_rows']=int((sbp.notna()&dbp.notna()).sum());row['sbp_less_than_dbp_rows']=int((sbp<dbp).sum())
                row['bp_use']='quarantined_labels_require_review_no_automatic_swap'
            records.append(row)
    (OUT/'nasa_echo_inventory.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    path=RAW/'NASA_RADLAB/ISS_DosTel1_20220301_20220315.csv'
    frame=pd.read_csv(path);time=pd.to_datetime(frame['timestamp'],errors='coerce');value=pd.to_numeric(frame['absorbed_dose_rate'],errors='coerce')
    assert time.notna().all() and value.notna().all() and np.isfinite(value).all()
    assert set(frame['instrument_id'])=={'DosTel1'}
    assert time.min()>=pd.Timestamp('2022-03-01') and time.max()<pd.Timestamp('2022-03-15')
    duplicate_count=int(time.duplicated().sum())
    raw_series=pd.Series(value.to_numpy(),index=pd.DatetimeIndex(time)).sort_index()
    conflicts=raw_series.groupby(level=0).nunique()
    conflicting_timestamps=conflicts[conflicts>1].index
    # Preserve all original rows; exclude ambiguous timestamps in this derived audit.
    series=raw_series[~raw_series.index.isin(conflicting_timestamps)]
    series=series[~series.index.duplicated(keep='first')]
    grid=series.resample('15min',label='right',closed='right').mean();count=grid.notna().rolling(96,min_periods=96).sum()
    summary={'source':'NASA OSDR RadLab','instrument':'DosTel1','domain':'ISS_instrument_radiation_not_personal_dose',
             'rows':len(frame),'first':series.index.min().isoformat(),'last':series.index.max().isoformat(),
             'duplicate_timestamps':duplicate_count,'median_interval_seconds':float(series.index.to_series().diff().dt.total_seconds().median()),
             'conflicting_timestamps_excluded':len(conflicting_timestamps),'conflicting_rows_excluded':int(raw_series.index.isin(conflicting_timestamps).sum()),
             'max_gap_seconds':float(series.index.to_series().diff().dt.total_seconds().max()),
             'negative_dose_rate_rows':int((value<0).sum()),'unit_as_documented':'microGy/hour',
             'observed_15min_bins':int(grid.notna().sum()),'total_15min_bins':len(grid),
             'pairs_24h_history_80pct':{str(h):int(((count>=77)&grid.shift(-h*4).notna()).sum()) for h in (1,4,8,24)},
             'limitations':['Instrument readings are not astronaut personal dosimetry or health outcomes.','Do not pair by absolute dates with date-shifted terrestrial clinical datasets.','A single 14-day instrument series cannot establish cross-mission clinical prediction.','Timezone convention must be verified before external calendar alignment.','No new models fitted.']}
    (OUT/'radlab_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('NASA source MD5s verified:',verified)
    print('Echo tables:',[(r['dataset'],r['file'],r['rows'],r['subject_count']) for r in records])
    print('Blood pressure label flags:',[(r['file'],r['sbp_less_than_dbp_rows'],r['paired_bp_rows']) for r in records if 'sbp_less_than_dbp_rows' in r])
    print(json.dumps(summary,indent=2))
