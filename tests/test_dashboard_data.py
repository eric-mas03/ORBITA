"""Data provenance and temporal isolation checks for the local dashboard."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('dashboard_app', ROOT/'scripts/dashboard.py')
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


def test_nasa_crew_and_real_visits_are_not_hourly_icu_cases():
    obs,pred,variables = app.nasa_data()
    assert {r['subject_id'] for r in obs} == {'C001','C002','C003','C004'}
    assert len(variables) > 100
    visits = ['L-92','L-44','L-3','R+1','R+45','R+82','R+194']
    assert all(r['timepoint'] in visits for r in obs)
    assert all(visits.index(r['origin']) < visits.index(r['target_visit']) for r in pred)


def test_reference_ranges_keep_original_person_visit_analyte_and_units():
    import csv
    obs,_,_ = app.nasa_data()
    original = {}
    for assay,path in [('CMP','OSD-575/LSDS-8_Comprehensive_Metabolic_Panel_CMP.upload_SUBMITTED.csv'),('CBC','OSD-569/LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv')]:
        with (ROOT/'data/raw'/path).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                try:
                    low,high = float(row['RANGE_MIN']),float(row['RANGE_MAX'])
                except ValueError:
                    continue
                if np.isfinite(low) and np.isfinite(high) and low<=high:
                    original[(assay,row['SUBJECT_ID'],row.get('timepoint',row.get('TEST_DATE')),row['ANALYTE'],row['UNITS'])]=(low,high)
    matched=unmatched=0
    for row in obs:
        key=(row['assay'],row['subject_id'],row['timepoint'],row['original_analyte'] or row['analyte'],row['unit'])
        if key in original:
            matched+=1
            low,high=original[key]
            assert (row['reference_min'],row['reference_max']) == (low,high)
            expected='below_reference' if float(row['value'])<low else 'above_reference' if float(row['value'])>high else 'within_reference'
            assert row['reference_status']==expected
        else:
            unmatched+=1
            assert row['reference_min'] is None and row['reference_max'] is None
            assert row['reference_status']=='no_reference'
    assert matched>900 and unmatched>0


def test_hourly_prediction_does_not_use_actual_future_values():
    if not (app.WORKSPACE/'role_data/final/packed.npz').exists():
        pytest.skip('Frozen research assets not installed')
    data = app.final_data()
    episode = str(data['episode_ids'][0])
    origin = app.origins_for(episode)[3]
    before = app.forecast_prediction(episode,origin,'group_temporal_residual_raw').copy()
    overview = app.overview_at(episode,origin,4,'group_temporal_residual_raw')
    assert all(row['hour'] is None or row['hour']<=origin for row in overview)
    assert all(row['target_hour']==origin+4 and row['horizon']==4 for row in overview)
    ep = 0
    first,end = data['episode_offsets'][ep:ep+2]
    rows = np.arange(first,end)[data['times'][first:end] > origin]
    saved = data['values'][rows].copy()
    try:
        data['values'][rows] = np.where(np.isfinite(saved),saved+100,saved)
        app.forecast_prediction.cache_clear()
        after = app.forecast_prediction(episode,origin,'group_temporal_residual_raw')
        np.testing.assert_array_equal(before,after)
        assert app.overview_at(episode,origin,4,'group_temporal_residual_raw') == overview
        for row in overview:
            col=int(np.flatnonzero(data['variables']==row['variable'])[0])
            known=np.flatnonzero((data['times'][first:end]<=origin)&np.isfinite(data['values'][first:end,col]))
            assert row['value']==float(data['values'][first+known[-1],col]) if len(known) else row['value'] is None
        with pytest.raises(ValueError,match='Horizonte'):
            app.overview_at(episode,origin,7,'group_temporal_residual_raw')
    finally:
        data['values'][rows] = saved
        app.forecast_prediction.cache_clear()
