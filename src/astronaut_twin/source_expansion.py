"""Independent multi-source experiment: common temporal memory, source-specific encoders/heads.

Quarter-hour physiological/environmental windows and ordinal echo visits are never joined.
Time is dimensionless inside the memory; each source has an explicit clock contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/research_expansion_20261114'
OUT=ROOT/'outputs/source_campaign_v3'
ECHO_KEYS=['LVDD','LVDS','ME','MA','MDT','PAT','PET','IVRT']
ECHO_LABELS=['Diámetro ventricular diastólico','Diámetro ventricular sistólico','Velocidad mitral E','Velocidad mitral A','Desaceleración mitral','Aceleración pulmonar','Eyección pulmonar','Relajación isovolumétrica']
ECHO_LONG=['Left Ventricular Diameter Diastole','Left Ventricular Diameter Systole','Mitral E Wave Velocity','Mitral A Wave Velocity','Mitral Deceleration Time','Pulmonic Valve Acceleration Time','Pulmonic Valve Ejection Time','Isovolumic Relaxation Time']


def person_roles(names,seed=2026):
    names=sorted(set(names));order=np.random.default_rng(seed).permutation(len(names))
    if len(names)<4:
        roles=['train','dev','test'][:len(names)]
    else:
        counts=[max(1,int(len(names)*.1))]*3
        roles=['train']*(len(names)-sum(counts))+['dev']*counts[0]+['calibration']*counts[1]+['test']*counts[2]
    return {names[int(i)]:role for i,role in zip(order,roles)}


def frame_series(frame,domain,features,source,clock='visits',scope=''):
    """Aggregate repeated readings within one visit; do not treat replicates as people."""
    result=[];frame=frame.copy();frame['Subject']=frame.Subject.astype(str)
    phase={'PRE_TEST':0,'IN_TEST':1,'POST_TEST':2}
    frame['_phase']=frame.Test_Phase.map(phase)
    day='BR_Day' if 'BR_Day' in frame else 'Day'
    frame['_day']=pd.to_numeric(frame[day],errors='coerce')
    frame['_order']=np.where(frame['_phase']==0,-frame['_day'],frame['_day'])
    for f in features:frame[f]=pd.to_numeric(frame[f],errors='coerce')
    for subject,data in frame.groupby('Subject',sort=True):
        visits=data.groupby(['_phase','_order'],dropna=True)[features].mean().reset_index().sort_values(['_phase','_order'])
        if len(visits)<2:continue
        result.append({'id':domain+'|'+scope+subject,'person':scope+subject,'domain':domain,'source':source,'clock':clock,
                       'times':np.arange(len(visits),dtype=float).tolist(),
                       'labels':[f"{['PRE','IN','POST'][int(r['_phase'])]} {abs(r['_order']):g}" for _,r in visits.iterrows()],
                       'values':visits[features].to_numpy(float).tolist()})
    return result


def load_sources():
    specs={};series=[];disposition=[]
    specs['wearable']={'label':'Glucosa, pulso y comidas · BIG IDEAs','domain':'terrestrial_wearable','clock':'hours','horizons':[1,4,8,24],
        'features':['glucose','heart_rate','carbs','protein','fat','fiber','sugar','calories'],
        'targets':['glucose','heart_rate'],'labels':['Glucosa intersticial','Frecuencia cardíaca'],'units':['mg/dL','beats/min'],
        'groups':[[0,1],[2,3,4,5,6,7]],'context':96,'time_scale':24.,
        'reference':'https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/'}
    for sid in range(1,17):
        subject=f'{sid:03d}';directory=RAW/'BIGIDEAS_1.1.3'/subject
        d=pd.read_csv(directory/f'Dexcom_{subject}.csv')
        time=pd.to_datetime(d['Timestamp (YYYY-MM-DDThh:mm:ss)'],errors='coerce');value=pd.to_numeric(d['Glucose Value (mg/dL)'],errors='coerce')
        valid=time.notna()&value.notna()
        glucose=pd.Series(value[valid].to_numpy(),index=pd.DatetimeIndex(time[valid])).groupby(level=0).mean().sort_index()
        grid=glucose.resample('15min',label='right',closed='right').mean().to_frame('glucose')
        h=pd.read_csv(directory/f'HR_{subject}.csv');h.columns=h.columns.str.strip()
        ht=pd.to_datetime(h['datetime'],format='mixed',errors='coerce');hv=pd.to_numeric(h['hr'],errors='coerce')
        # Source exposes minute labels for second-rate HR. Put minute k in the bin
        # ending strictly after k, so a boundary cannot include unknown later seconds.
        minute_only=~h['datetime'].astype(str).str.contains(r'\d:\d{2}:\d{2}',regex=True)
        ht=ht+pd.to_timedelta(minute_only.astype(int),unit='min')
        good=ht.notna()&hv.notna()
        hr=pd.Series(hv[good].to_numpy(),index=pd.DatetimeIndex(ht[good])).resample('15min',label='right',closed='right').mean()
        grid['heart_rate']=hr.reindex(grid.index)
        food=pd.read_csv(directory/f'Food_Log_{subject}.csv')
        mapping={'carbs':'total_carb','protein':'protein','fat':'total_fat','fiber':'dietary_fiber','sugar':'sugar','calories':'calorie'}
        for name in mapping:grid[name]=np.nan
        if 'time_begin' in food:
            food['_time']=pd.to_datetime(food.time_begin,format='mixed',errors='coerce')
            for name,col in mapping.items():
                if col not in food:continue
                values=pd.to_numeric(food[col],errors='coerce');ok=food['_time'].notna()&values.notna()
                events=pd.Series(values[ok].to_numpy(),index=pd.DatetimeIndex(food.loc[ok,'_time']))
                grid[name]=events.resample('15min',label='right',closed='right').sum(min_count=1).reindex(grid.index)
        else:disposition.append({'source':f'Food_Log_{subject}','status':'excluded_meal_inputs','reason':'No verified header; CGM and HR remain eligible'})
        series.append({'id':'BIG:'+subject,'person':'BIG:'+subject,'domain':'wearable','source':'BIG IDEAs 1.1.3','clock':'hours',
                       'times':(np.arange(len(grid))*.25).tolist(),'labels':[t.isoformat() for t in grid.index],
                       'values':grid[specs['wearable']['features']].to_numpy(float).tolist()})
    specs['radiation']={'label':'Radiación de la ISS · DosTel1','domain':'ISS_environment','clock':'hours','horizons':[1,4,8,24],
        'features':['absorbed_dose_rate'],'targets':['absorbed_dose_rate'],'labels':['Tasa de dosis absorbida'],'units':['µGy/h'],
        'groups':[[0]],'context':96,'time_scale':24.,'reference':'https://visualization.osdr.nasa.gov/radlab/gui/data-api/'}
    d=pd.read_csv(RAW/'NASA_RADLAB/ISS_DosTel1_20220301_20220315.csv');d['_time']=pd.to_datetime(d.timestamp)
    counts=d.groupby('_time').absorbed_dose_rate.nunique();conflicts=counts[counts>1].index
    d=d[~d['_time'].isin(conflicts)].drop_duplicates('_time').sort_values('_time')
    grid=d.set_index('_time').absorbed_dose_rate.resample('15min',label='right',closed='right').mean()
    series.append({'id':'ISS:DosTel1','person':'ISS:DosTel1','domain':'radiation','source':'NASA RadLab · DosTel1','clock':'hours',
                   'times':(np.arange(len(grid))*.25).tolist(),'labels':[t.isoformat() for t in grid.index],'values':grid.to_numpy()[:,None].tolist()})
    disposition.append({'source':'RadLab','status':'excluded_ambiguous_timestamps','count':len(conflicts),'reason':'Conflicting readings at identical timestamps; original preserved'})
    nasa=RAW/'NASA_LSDA'
    def echo_spec(key,label,features,labels=None):
        specs[key]={'label':label,'domain':'NASA_ground_analog','clock':'visits','horizons':[1,2],
            'features':features,'targets':features,'labels':labels or features,'units':['unidad del ensayo · sin armonizar']*len(features),
            'groups':[list(range(len(features)))],'context':8,'time_scale':1.,
            'reference':'https://nlsp.nasa.gov/search/data/details/9af6f9ba-3730-54ab-a82a-1a32798667ae?dataType=EXPERIMENTS'}
    echo_spec('echo_c1','NASA UTMB C1 · ecocardiografía',ECHO_KEYS,ECHO_LABELS)
    d=pd.read_csv(nasa/'BRSMCF_C1/BRSMCF_Campaign_1_Echocardiography_CF.csv',dtype=str);d['Exam_Type']=d.Exam_Type.str.strip();d['Measurement']=pd.to_numeric(d.Measurement,errors='coerce')
    keys=['Subject','Test_Phase','BR_Day']
    wide=d.pivot_table(index=keys,columns='Exam_Type',values='Measurement',aggfunc='mean').reset_index()
    wide=wide.rename(columns=dict(zip(ECHO_LONG,ECHO_KEYS)))
    series+=frame_series(wide,'echo_c1',ECHO_KEYS,'NASA LSDA C1 CF',scope='NASA:C1:')
    echo_spec('echo_c3','NASA UTMB C3 · ecocardiografía',ECHO_KEYS,ECHO_LABELS)
    c3=[pd.read_csv(nasa/'BRSMCF_C3'/name,dtype=str) for name in ('BRSMCF_C3ABCD_Echocardiography_obsv.csv','BRSMCF_C3E_Echo_stills.csv','BRSMCF_C3F_Echo_stills.csv')]
    series+=frame_series(pd.concat(c3,ignore_index=True),'echo_c3',ECHO_KEYS,'NASA LSDA C3 observations/E/F',scope='NASA:C3:')
    echo_spec('echo_cft','NASA CFT70 · ecocardiografía 2D',ECHO_KEYS,ECHO_LABELS)
    d=pd.read_csv(nasa/'BRSMCF_CFT70/BRSMCF_CFT70_2D_Echo_All.csv',dtype=str).rename(columns=dict(zip(ECHO_LONG,ECHO_KEYS)))
    series+=frame_series(d,'echo_cft',ECHO_KEYS,'NASA LSDA CFT70 2D',scope='NASA:CFT70:')
    echo_spec('echo_3d','NASA CFT70 · ecocardiografía 3D',['LV mass','LVDV','LVSV'],['Masa ventricular','Volumen ventricular diastólico','Volumen ventricular sistólico'])
    d=pd.read_csv(nasa/'BRSMCF_CFT70/BRSMCF_CFT70_3D_Echo_All.csv',dtype=str)
    series+=frame_series(d,'echo_3d',specs['echo_3d']['features'],'NASA LSDA CFT70 3D · media de lecturas por visita',scope='NASA:CFT70:')
    # SLNTG is a protocol response, not a resting arterial diameter. Keep its own head.
    echo_spec('vascular_c1','NASA C1 · respuesta vascular SLNTG',['arm','leg'],['Diámetro braquial bajo protocolo','Diámetro tibial bajo protocolo'])
    vascular=[]
    for name,col,label in [('BRSMCF_Campaign_1_Echocardiography_SLN_Arm.csv','SLNTG Brachial Artery Diameter','arm'),('BRSMCF_Campaign_1_Echocardiography_SLN_Leg.csv','SLNTG Anterior Tibia Diameter','leg')]:
        d=pd.read_csv(nasa/'BRSMCF_C1'/name,dtype=str);d=d[pd.to_numeric(d.Minute,errors='coerce')==3].copy();d[label]=pd.to_numeric(d[col],errors='coerce')
        vascular.append(d.groupby(keys)[label].mean())
    wide=pd.concat(vascular,axis=1).reset_index();series+=frame_series(wide,'vascular_c1',['arm','leg'],'NASA C1 SLNTG · minuto 3, media de frames',scope='NASA:C1:')
    disposition.extend([
        {'source':'NASA derived means / changes / 7-subject subset / C3D duplicate','status':'registered_not_separate_labels','reason':'Avoid duplicated visits, aggregate labels or repeated participants'},
        {'source':'NASA C3 SBP/DBP/WS/RHR/LHR','status':'quarantined','reason':'Unresolved labels, inconsistent pressure order or suspect heart-rate fields'},
        {'source':'CGMacros 1.0.0','status':'registered_quarantined','reason':'Published minute series interpolated; native sensor anchors not verified'},
        {'source':'PhysioCGM / NASA controlled-access studies','status':'metadata_only','reason':'No eligible raw observations downloaded; cannot train on a catalogue'}])
    roles={}
    for family in ('BIG:','NASA:C1:','NASA:C3:','NASA:CFT70:'):
        roles.update(person_roles([s['person'] for s in series if s['person'].startswith(family)]))
    for item in series:
        item['role']=roles.get(item['person'],'temporal')
    return specs,series,disposition


def samples_for(item,spec):
    values=np.asarray(item['values'],dtype=np.float32);samples=[];n=len(values);hourly=spec['clock']=='hours'
    for origin in range(95 if hourly else 0,n,4 if hourly else 1):
        split=item['role'];lower,upper=0,n
        if split=='temporal':
            cuts=[0,8*96,10*96,12*96,n];index=min(3,np.searchsorted(cuts[1:],origin,side='right'))
            split=['train','dev','calibration','test'][index];lower,upper=cuts[index:index+2]
        rows=[origin+int(h*4 if hourly else h) for h in spec['horizons']]
        mask=np.array([r<upper for r in rows]);safe=np.minimum(rows,n-1)
        target=values[safe,:len(spec['targets'])].copy();target[~mask]=np.nan
        if np.isfinite(target).any():samples.append({'series':item['id'],'domain':item['domain'],'origin':origin,'split':split,'target':target,'target_rows':rows})
    return samples


def history(item,spec,origin):
    """No forward filling and no access to rows later than issuance."""
    start=max(0,origin-spec['context']+1)
    return np.asarray(item['values'][start:origin+1],dtype=np.float32),np.asarray(item['times'][start:origin+1],dtype=np.float32)


class SourceTwin(nn.Module):
    def __init__(self,specs,width=32):
        super().__init__();self.specs=specs;self.encoders=nn.ModuleDict();self.fusions=nn.ModuleDict();self.heads=nn.ModuleDict()
        self.embedding=nn.ParameterDict({k:nn.Parameter(torch.randn(width)*.02) for k in specs})
        for domain,spec in specs.items():
            self.encoders[domain]=nn.ModuleList([nn.Sequential(nn.Linear(3*len(group),width),nn.GELU(),nn.Linear(width,width)) for group in spec['groups']])
            self.fusions[domain]=nn.Linear(width*len(spec['groups']),width)
            self.heads[domain]=nn.Linear(width,len(spec['targets'])*3)
            nn.init.zeros_(self.heads[domain].weight);nn.init.zeros_(self.heads[domain].bias)
        self.time=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width))
        self.memory=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,.1,batch_first=True),1,enable_nested_tensor=False)
        self.attention=nn.MultiheadAttention(width,4,.1,batch_first=True)

    def forward(self,domain,values,observed,ages,times,padding,horizons,baseline):
        spec=self.specs[domain];tokens=[]
        for group,encoder in zip(spec['groups'],self.encoders[domain]):
            mask=observed[:,:,group];token=encoder(torch.cat([values[:,:,group]*mask,mask.float(),torch.log1p(ages[:,:,group])],-1))
            tokens.append(token*mask.any(-1,keepdim=True))
        fused=self.fusions[domain](torch.cat(tokens,-1))+self.embedding[domain]
        def timing(t):return torch.stack([t,torch.sin(t*2*np.pi),torch.cos(t*2*np.pi)],-1)
        memory=self.memory(fused+self.time(timing(times)),mask=torch.triu(torch.ones(times.shape[1],times.shape[1],device=times.device,dtype=torch.bool),1),src_key_padding_mask=padding)
        query=self.time(timing(horizons))+self.embedding[domain]
        decoded,_=self.attention(query,memory,memory,key_padding_mask=padding,need_weights=False)
        raw=self.heads[domain](decoded).reshape(len(values),horizons.shape[1],len(spec['targets']),3)
        median=baseline[...,1]+torch.tanh(raw[...,1])
        return torch.stack([median-(baseline[...,1]-baseline[...,0])*torch.exp(.7*torch.tanh(raw[...,0])),median,
                            median+(baseline[...,2]-baseline[...,1])*torch.exp(.7*torch.tanh(raw[...,2]))],-1)


def make_batch(items,lookup,specs,scalers,bands):
    domain=items[0]['domain'];spec=specs[domain];mean,scale=map(np.asarray,(scalers[domain]['mean'],scalers[domain]['scale']))
    length=spec['context'];b=len(items);f=len(mean);v=len(spec['targets']);h=len(spec['horizons'])
    values=np.zeros((b,length,f),np.float32);observed=np.zeros_like(values,bool);ages=np.zeros_like(values);times=np.zeros((b,length),np.float32);padding=np.ones((b,length),bool)
    targets=np.zeros((b,h,v),np.float32);masks=np.zeros_like(targets,bool);bases=np.zeros((b,h,v,3),np.float32)
    for i,sample in enumerate(items):
        raw,clock=history(lookup[sample['series']],spec,sample['origin']);n=len(raw);mask=np.isfinite(raw)
        normalized=(raw-mean)/scale;values[i,:n]=np.nan_to_num(normalized);observed[i,:n]=mask;padding[i,:n]=False
        times[i,:n]=(clock-clock[-1])/spec['time_scale'];last_clock=np.full(f,clock[0]);last_value=np.zeros(f)
        for step in range(n):
            last_clock=np.where(mask[step],clock[step],last_clock);last_value=np.where(mask[step],normalized[step],last_value)
            ages[i,step]=(clock[step]-last_clock)/spec['time_scale']
        bases[i]=last_value[:v][None,:,None]+np.asarray(bands[domain])
        if 'target' in sample:
            y=(sample['target']-mean[:v])/scale[:v];masks[i]=np.isfinite(y);targets[i]=np.nan_to_num(y)
    args=[domain,*map(torch.from_numpy,(values,observed,ages,times,padding)),torch.tensor(spec['horizons'],dtype=torch.float32)[None].expand(b,-1)/spec['time_scale'],torch.from_numpy(bases)]
    return args,torch.from_numpy(targets),torch.from_numpy(masks)


def predict(model,items,lookup,specs,scalers,bands,method='transformer',weights=None,expansion=None):
    if not items:return np.empty((0,))
    args,_,_=make_batch(items,lookup,specs,scalers,bands);domain=items[0]['domain'];v=len(specs[domain]['targets'])
    model.eval()
    with torch.no_grad():neural=model(*args).numpy()
    base=args[-1].numpy();w=(weights or {}).get(domain,1.)
    result=base if method=='persistence' else neural if method=='transformer' else w*neural+(1-w)*base
    if method=='ensemble' and expansion is not None:
        offset=np.asarray(expansion[domain]);result[...,0]-=offset;result[...,2]+=offset
    result=result*np.asarray(scalers[domain]['scale'])[:v][None,None,:,None]+np.asarray(scalers[domain]['mean'])[:v][None,None,:,None]
    return np.maximum(result,0.)
