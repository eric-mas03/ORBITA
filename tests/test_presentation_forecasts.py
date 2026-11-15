"""Dense presentation contracts must not manufacture forecasts or hide error at now."""
import copy
import numpy as np
import pytest

from astronaut_twin import presentation_forecasts as presentation
from astronaut_twin.source_expansion import SourceTwin, make_batch


def fixture(domain='wearable'):
    features=['glucose'] if domain=='wearable' else ['absorbed_dose_rate']
    spec={'clock':'hours','context':96,'time_scale':24.,'features':features,'targets':features,
          'groups':[[0]],'horizons':(np.arange(1,97)/4).tolist(),
          'labels':['measurement'],'units':['mg/dL'] if domain=='wearable' else ['µGy/h'],
          'domain':'terrestrial_wearable' if domain=='wearable' else 'ISS_environment','reference':'https://example.test/source'}
    n=1345 if domain=='radiation' else 800
    item={'id':'ISS:DosTel1' if domain=='radiation' else 'BIG:011','person':'test-person','role':'temporal' if domain=='radiation' else 'test',
          'domain':domain,'values':[[float(i%30)] for i in range(n)],'times':(np.arange(n)*.25).tolist(),'labels':[f't{i}' for i in range(n)]}
    setup={'specs':{domain:spec},'scalers':{domain:{'mean':[10.],'scale':[2.]}},
           'bands':{domain:np.tile([-1.,0.,1.],(96,1,1)).tolist()},
           'ensemble_neural_weight':{domain:1.},'calibration_expansion_z':{domain:np.zeros((96,1)).tolist()},
           'calibration_counts':{domain:np.full((96,1),25).tolist()}}
    return setup,{item['id']:item},item


def fake_inference(model,samples,lookup,setup,method):
    horizon_count=len(setup['specs'][samples[0]['domain']]['horizons'])
    result=np.zeros((len(samples),horizon_count,1,3))
    for i,sample in enumerate(samples):
        median=100+sample['origin']/100
        result[i,:,0]=[median-2,median,median+2]
    return result


def test_dense_future_and_complete_historical_replay_without_forced_anchor(monkeypatch):
    setup,lookup,item=fixture()
    monkeypatch.setattr(presentation,'_inference',fake_inference)
    panel=presentation.build_panel('wearable',setup,lookup,None)
    variable=panel['variables'][0]
    assert len(variable['future'])==96
    assert variable['future'][0]['x']==.25 and variable['future'][-1]['x']==24
    assert len(variable['historical'])==97
    assert variable['historical'][0]['x']==-24 and variable['historical'][-1]['x']==0
    for point in variable['historical']:
        assert point['target_index']-point['issue_index']==4
        assert point['quantile_source']=='rolling_historical_replay'
    anchor=variable['origin_forecast_anchor']
    assert anchor['issue_index']==panel['origin_index']-4
    assert anchor['q50']!=variable['current']
    assert variable['future'][0]['q50']!=variable['current']
    assert all(p['issue_index']==panel['origin_index'] for p in variable['future'])


def test_radiation_historical_issuances_follow_calibration_and_future_truth_stays_missing(monkeypatch):
    setup,lookup,item=fixture('radiation')
    monkeypatch.setattr(presentation,'_inference',fake_inference)
    panel=presentation.build_panel('radiation',setup,lookup,None)
    variable=panel['variables'][0]
    assert panel['origin_index']==1252
    assert min(p['issue_index'] for p in variable['historical'])==12*96
    assert all(p['issue_index']>=12*96 for p in variable['historical'])
    assert variable['future'][-1]['observed'] is None
    assert 'not personal dosimetry' in panel['evidence_label']
    assert panel['is_logged_historical_forecast'] is False


def test_every_rolling_window_ignores_its_own_future():
    setup,lookup,item=fixture()
    origin=300;domain='wearable';samples=[{'series':item['id'],'domain':domain,'origin':origin}]
    before,_,_=make_batch(samples,lookup,setup['specs'],setup['scalers'],setup['bands'])
    changed=copy.deepcopy(lookup);changed[item['id']]['values'][origin+1:]=[[1e8]]*(len(item['values'])-origin-1)
    after,_,_=make_batch(samples,changed,setup['specs'],setup['scalers'],setup['bands'])
    for first,second in zip(before[1:],after[1:]):
        np.testing.assert_array_equal(first.numpy(),second.numpy())


def test_fixed_case_uses_identity_order_instead_of_forecast_quality():
    _,lookup,item=fixture()
    earlier=copy.deepcopy(item);earlier['id']='BIG:009';earlier['person']='earlier-test'
    training=copy.deepcopy(item);training['id']='BIG:001';training['role']='train'
    lookup.update({earlier['id']:earlier,training['id']:training})
    assert presentation._fixed_case('wearable',lookup)['id']=='BIG:009'


def test_trained_dense_artifacts_if_available():
    if not (presentation.DEFAULT_ARTIFACTS/'source_twin.pt').exists():
        pytest.skip('Isolated dense campaign not yet trained')
    data=presentation.presentation_payload()
    for panel in data['panels']:
        if panel['clock']=='hours':
            for variable in panel['variables']:
                assert len(variable['future'])==96
                assert all(p['q10']<=p['q50']<=p['q90'] for p in variable['future'])
    assert any('Ground wearable' in p['evidence_label'] for p in data['panels'])
    assert any('ISS instrument' in p['evidence_label'] for p in data['panels'])
