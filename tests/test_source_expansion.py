"""Temporal isolation, identity splits and active-head updates in the source campaign."""
import copy
import json
import numpy as np
import pytest
import torch
from astronaut_twin.source_expansion import OUT,SourceTwin,history,person_roles,samples_for,make_batch


def toy():
    spec={'clock':'hours','context':96,'time_scale':24.,'features':['a'],'targets':['a'],'groups':[[0]],'horizons':[1,4,8,24]}
    values=np.arange(240,dtype=float)[:,None];values[30]=np.nan
    item={'id':'one','person':'one','domain':'a','role':'train','values':values.tolist(),'times':(np.arange(240)*.25).tolist()}
    return spec,item


def test_future_values_and_labels_cannot_change_inference_inputs():
    spec,item=toy();sample={'domain':'a','series':'one','origin':100}
    scalers={'a':{'mean':[0.],'scale':[100.]}};bands={'a':np.tile([-1,0,1],(4,1,1)).tolist()}
    before,_,_=make_batch([sample],{'one':item},{'a':spec},scalers,bands)
    changed=copy.deepcopy(item);changed['values'][101:]=[[1e8]]*(len(changed['values'])-101)
    after,_,_=make_batch([sample],{'one':changed},{'a':spec},scalers,bands)
    for a,b in zip(before[1:],after[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    raw,clock=history(item,spec,100)
    assert raw.shape==(96,1) and clock[-1]==25 and clock[0]==1.25


def test_radiation_training_targets_never_cross_fit_boundary():
    spec,item=toy();item.update(role='temporal',values=[[1.]]*1345,times=(np.arange(1345)*.25).tolist())
    samples=samples_for(item,spec)
    bounds={'train':(0,768),'dev':(768,960),'calibration':(960,1152),'test':(1152,1345)}
    for sample in samples:
        low,high=bounds[sample['split']]
        assert low<=sample['origin']<high
        for row,y in zip(sample['target_rows'],sample['target']):
            assert np.isfinite(y).any()==(row<high)
            if np.isfinite(y).any():assert sample['origin']<row<high


def test_inactive_source_head_keeps_parameters_with_adamw():
    torch.set_num_threads(2);spec,item=toy();specs={'a':spec,'b':copy.deepcopy(spec)};model=SourceTwin(specs,width=8)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    frozen={k:v.detach().clone() for k,v in model.heads['b'].state_dict().items()}
    args,_,_=make_batch([{'domain':'a','series':'one','origin':100}],{'one':item},specs,{'a':{'mean':[0.],'scale':[100.]}},{'a':np.tile([-1,0,1],(4,1,1)).tolist()})
    optimizer.zero_grad(set_to_none=True);model(*args).square().mean().backward();optimizer.step()
    for name,value in model.heads['b'].state_dict().items():torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)
    assert all(p.grad is None for p in model.heads['b'].parameters())


def test_same_cft70_person_has_same_role_in_2d_and_3d():
    assert person_roles(['x','y','z'])==person_roles(['z','y','x'])
    path=OUT/'fit_manifest.json'
    if not path.exists():pytest.skip('Source campaign not trained in this installation')
    manifest=json.loads(path.read_text(encoding='utf8'));roles={}
    for row in manifest['roles'].values():roles.setdefault(row['person'],set()).add(row['role'])
    assert all(len(values)==1 for values in roles.values())
    assert manifest['test_used_for_fitting'] is False


def test_served_forecast_quantiles_and_unobserved_future():
    if not (OUT/'source_twin.pt').exists():pytest.skip('Source campaign not trained')
    from astronaut_twin import source_dashboard
    cat=source_dashboard.catalog();case=next(s for s in cat['sources'] if s['id']=='radiation')['cases'][0]['id']
    origin=source_dashboard.origins(case)[-1]['index'];result=source_dashboard.forecast(case,origin,'ensemble',24)
    for variable in result['variables']:
        for forecast in variable['forecast']:
            assert forecast['q10']<=forecast['q50']<=forecast['q90']
            assert forecast['observed'] is None and forecast['index']>origin
    with pytest.raises(ValueError,match='Horizonte'):
        source_dashboard.forecast(case,origin,'ensemble',2)
