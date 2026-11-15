"""Read-only serving for the isolated source campaign; real cutoff-based inference."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import torch
from .source_expansion import OUT,SourceTwin,predict


@lru_cache(maxsize=1)
def artifacts():
    setup=json.loads((OUT/'setup.json').read_text(encoding='utf8'))
    series=json.loads((OUT/'series.json').read_text(encoding='utf8'))
    checkpoint=torch.load(OUT/'source_twin.pt',map_location='cpu',weights_only=True)
    model=SourceTwin(setup['specs'],checkpoint['width']);model.load_state_dict(checkpoint['state']);model.eval()
    return setup,{s['id']:s for s in series},model


def catalog():
    setup,lookup,_=artifacts();sources=[]
    for domain,spec in setup['specs'].items():
        cases=[{'id':s['id'],'person':s['person'],'role':s['role'],'steps':len(s['times'])} for s in lookup.values() if s['domain']==domain]
        cases.sort(key=lambda s:({'test':0,'temporal':0,'dev':1,'calibration':2,'train':3}.get(s['role'],4),s['person']))
        sources.append({'id':domain,**spec,'cases':cases,'ensemble_neural_weight':setup['ensemble_neural_weight'][domain]})
    sources.sort(key=lambda s:(s['id']!='radiation',s['id']=='wearable',s['label']))
    metrics=pd.read_csv(OUT/'metrics.csv').replace({np.nan:None}).to_dict('records')
    return {'sources':sources,'metrics':metrics,'summary':json.loads((OUT/'summary.json').read_text(encoding='utf8')),
            'integration':json.loads((OUT/'integration.json').read_text(encoding='utf8')),'model':'SourceTwin V3 · memoria compartida, encoders y salidas por fuente'}


def origins(case):
    setup,lookup,_=artifacts()
    if case not in lookup:raise ValueError('Caso desconocido')
    item=lookup[case];spec=setup['specs'][item['domain']];n=len(item['times'])
    options=list(range(95 if spec['clock']=='hours' else 0,n,4 if spec['clock']=='hours' else 1))
    if item['role']=='temporal':options=[i for i in options if i>=12*96]
    if n-1 not in options:options.append(n-1)
    return [{'index':i,'label':item['labels'][i],'last':i==n-1} for i in options]


@lru_cache(maxsize=256)
def forecast(case,origin,method='ensemble',horizon=1):
    setup,lookup,model=artifacts()
    if case not in lookup:raise ValueError('Caso desconocido')
    item=lookup[case];domain=item['domain'];spec=setup['specs'][domain]
    if origin not in {o['index'] for o in origins(case)}:raise ValueError('Origen no disponible')
    if method not in ('ensemble','transformer','persistence'):raise ValueError('Modelo desconocido')
    if horizon not in spec['horizons']:raise ValueError('Horizonte no entrenado para esta cadencia')
    selected=[{'series':case,'domain':domain,'origin':origin}]
    prior=[i for i in [o['index'] for o in origins(case)] if origin-(96 if spec['clock']=='hours' else 8)<=i<origin and i+int(horizon*4 if spec['clock']=='hours' else horizon)<=origin]
    selected.extend({'series':case,'domain':domain,'origin':i} for i in prior)
    p=predict(model,selected,lookup,setup['specs'],setup['scalers'],setup['bands'],method,setup['ensemble_neural_weight'],setup['calibration_expansion_z'])
    values=np.asarray(item['values'],dtype=float);rows=[];v=len(spec['targets'])
    start=max(0,origin-(95 if spec['clock']=='hours' else 7));end=min(len(values),origin+int(max(spec['horizons'])*(4 if spec['clock']=='hours' else 1))+1)
    for k in range(v):
        known=[{'index':i,'x':item['times'][i],'label':item['labels'][i],'value':float(values[i,k])} for i in range(start,end) if np.isfinite(values[i,k])]
        latest=np.flatnonzero(np.isfinite(values[:origin+1,k]));current=float(values[latest[-1],k]) if len(latest) else None
        forecasts=[]
        for hi,h in enumerate(spec['horizons']):
            index=origin+int(h*(4 if spec['clock']=='hours' else 1));q10,q50,q90=map(float,p[0,hi,k]);n=setup['calibration_counts'][domain][hi][k]
            forecasts.append({'horizon':h,'index':index,'x':item['times'][origin]+h,'target_label':item['labels'][index] if index<len(values) else 'Futuro sin medición',
                              'q10':q10,'q50':q50,'q90':q90,'observed':float(values[index,k]) if index<len(values) and np.isfinite(values[index,k]) else None,
                              'calibration_n':n,'calibration_status':'adjusted' if method=='ensemble' and n>=20 else 'insufficient' if method=='ensemble' else 'raw'})
        hi=spec['horizons'].index(horizon)
        previous=[{'origin':i,'index':i+int(horizon*(4 if spec['clock']=='hours' else 1)),'x':item['times'][i]+horizon,
                   **dict(zip(('q10','q50','q90'),map(float,p[j+1,hi,k])))} for j,i in enumerate(prior)]
        rows.append({'id':spec['targets'][k],'label':spec['labels'][k],'unit':spec['units'][k],'history':known,'current':current,
                     'current_index':int(latest[-1]) if len(latest) else None,'forecast':forecasts,'previous':previous})
    return {'case':case,'person':item['person'],'role':item['role'],'source':domain,'source_label':spec['label'],'clock':spec['clock'],'horizons':spec['horizons'],
            'origin':origin,'origin_x':item['times'][origin],'origin_label':item['labels'][origin],'method':method,'variables':rows,
            'past_forecasts':'Reconstrucciones retrospectivas con el modelo actual; no registros de predicciones emitidas entonces.',
            'clinical_validation':False}
