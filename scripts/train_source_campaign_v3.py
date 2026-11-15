"""Train a separate, auditable shared-memory multi-source model; preserve campaigns V1/V2."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from astronaut_twin.source_expansion import OUT,RAW,SourceTwin,load_sources,samples_for,history,make_batch,predict


def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj


def dump(path,obj):path.write_text(json.dumps(clean(obj),ensure_ascii=False,allow_nan=False),encoding='utf8')


def chunks(items,size=64):
    for i in range(0,len(items),size):yield items[i:i+size]


def pinball(y,p):
    error=y[...,None]-p
    return float(np.nanmean(np.maximum(np.array([.1,.5,.9])*error,(np.array([.1,.5,.9])-1)*error)))


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=24);parser.add_argument('--width',type=int,default=32);parser.add_argument('--refinalize',action='store_true',help='Reuse frozen checkpoint; rebuild only dev selection/calibration/reports')
    parser.add_argument('--output-dir',default=str(OUT));parser.add_argument('--dense-quarter-hour',action='store_true');parser.add_argument('--warm-start',type=Path);args=parser.parse_args()
    OUT=Path(args.output_dir).resolve()
    if not OUT.is_relative_to(ROOT/'outputs'):raise ValueError('Campaign output must stay in workspace outputs')
    if args.dense_quarter_hour and OUT.name=='source_campaign_v3':raise ValueError('Dense trajectory requires a separate campaign directory')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(2026);rng=np.random.default_rng(2026);started=time.time()
    specs,series,disposition=load_sources();lookup={s['id']:s for s in series}
    if args.dense_quarter_hour:
        for spec in specs.values():
            if spec['clock']=='hours':spec['horizons']=(np.arange(1,97)/4).tolist()
    assert len(lookup)==len(series)
    samples=[r for s in series for r in samples_for(s,specs[s['domain']])]
    by={(d,split):[r for r in samples if r['domain']==d and r['split']==split] for d in specs for split in ('train','dev','calibration','test')}
    scalers={};bands={}
    for domain,spec in specs.items():
        train_rows=[]
        for item in series:
            if item['domain']!=domain:continue
            if item['role']=='train':train_rows.extend(item['values'])
            elif item['role']=='temporal':train_rows.extend(item['values'][:8*96])
        values=np.asarray(train_rows,dtype=float);mean=np.nanmean(values,axis=0);scale=np.nanstd(values,axis=0)
        mean=np.nan_to_num(mean);scale=np.where(np.isfinite(scale)&(scale>1e-6),scale,1)
        scalers[domain]={'mean':mean.tolist(),'scale':scale.tolist(),'observed_train_counts':np.isfinite(values).sum(0).tolist()}
        residuals=[]
        for sample in by[domain,'train']:
            raw,_=history(lookup[sample['series']],spec,sample['origin']);last=np.full(len(spec['targets']),np.nan)
            for row in raw:last=np.where(np.isfinite(row[:len(last)]),row[:len(last)],last)
            residuals.append((sample['target']-last)/scale[:len(last)])
        residuals=np.asarray(residuals);band=np.zeros((len(spec['horizons']),len(spec['targets']),3))
        for h in range(band.shape[0]):
            for v in range(band.shape[1]):
                values=residuals[:,h,v];values=values[np.isfinite(values)]
                band[h,v]=np.quantile(values,[.1,.5,.9]) if len(values) else [-1,0,1]
                band[h,v,0]=min(band[h,v,0],band[h,v,1]-1e-3);band[h,v,2]=max(band[h,v,2],band[h,v,1]+1e-3)
        bands[domain]=band.tolist()
    dump(OUT/'series.json',series);dump(OUT/'specs.json',specs)
    dump(OUT/'integration.json',{'disposition':disposition,'source_samples':{d:{s:len(by[d,s]) for s in ('train','dev','calibration','test')} for d in specs},'clinical_hourly_astronaut_validation':False})
    model=SourceTwin(specs,args.width);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    if args.warm_start:
        pretrained=torch.load(args.warm_start,map_location='cpu',weights_only=True)
        model.load_state_dict(pretrained['state'])
    best=float('inf');best_state=None;best_epoch=0;trace=[]
    if args.refinalize:
        frozen=torch.load(OUT/'source_twin.pt',map_location='cpu',weights_only=True)
        prior=json.loads((OUT/'setup.json').read_text(encoding='utf8'))
        model.load_state_dict(frozen['state']);best_state=copy.deepcopy(model.state_dict());best_epoch=prior['best_epoch'];trace=prior['train_trace']
    def predictions(items,method='transformer',weights=None,expansion=None):
        return np.concatenate([predict(model,b,lookup,specs,scalers,bands,method,weights,expansion) for b in chunks(items)])
    for epoch in range(0 if args.refinalize else args.epochs):
        model.train();losses=[]
        for domain in rng.permutation(list(specs)):
            train=by[domain,'train'];selection=rng.permutation(len(train))[:768]
            for batch in chunks([train[i] for i in selection]):
                arguments,target,mask=make_batch(batch,lookup,specs,scalers,bands)
                optimizer.zero_grad(set_to_none=True);prediction=model(*arguments);error=target.unsqueeze(-1)-prediction
                q=torch.tensor([.1,.5,.9]);loss=torch.maximum(q*error,(q-1)*error)[mask].mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        scores=[]
        for domain in specs:
            items=by[domain,'dev']
            if not items:continue
            y=np.asarray([s['target'] for s in items]);p=predictions(items);v=len(specs[domain]['targets']);mean=np.asarray(scalers[domain]['mean'])[:v];scale=np.asarray(scalers[domain]['scale'])[:v]
            scores.append(pinball((y-mean)/scale,(p-mean[None,None,:,None])/scale[None,None,:,None]))
        score=float(np.mean(scores));trace.append({'epoch':epoch+1,'train_pinball':float(np.mean(losses)),'dev_equal_source_pinball':score})
        print(json.dumps(trace[-1]),flush=True)
        if score<best-1e-5:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if epoch+1-best_epoch>=6:break
    model.load_state_dict(best_state);weights={};expansions={};calibration_counts={}
    for domain,spec in specs.items():
        dev=by[domain,'dev'];v=len(spec['targets']);scale=np.asarray(scalers[domain]['scale'])[:v];mean=np.asarray(scalers[domain]['mean'])[:v]
        if dev:
            y=np.asarray([r['target'] for r in dev])
            # Select the exact served blend (blend before clipping), not a blend
            # of separately clipped quantiles, which is a different estimator.
            candidates={float(w):pinball((y-mean)/scale,(predictions(dev,'ensemble',{domain:float(w)})-mean[None,None,:,None])/scale[None,None,:,None]) for w in (0,.25,.5,.75,1)}
            weights[domain]=min(candidates,key=candidates.get)
        else:weights[domain]=0.
        offset=np.zeros((len(spec['horizons']),v));counts=np.zeros_like(offset,dtype=int);cal=by[domain,'calibration']
        if cal:
            y=np.asarray([r['target'] for r in cal]);p=predictions(cal,'ensemble',weights)
            nonconformity=np.maximum(np.maximum(p[...,0]-y,y-p[...,2]),0)/scale
            for h in range(len(spec['horizons'])):
                for j in range(v):
                    values=nonconformity[:,h,j];values=values[np.isfinite(values)];counts[h,j]=len(values)
                    if len(values)>=20:offset[h,j]=np.quantile(values,min(1,np.ceil((len(values)+1)*.8)/len(values)),method='higher')
        expansions[domain]=offset.tolist();calibration_counts[domain]=counts.tolist()
    torch.save({'state':model.state_dict(),'width':args.width},OUT/'source_twin.pt')
    setup={'specs':specs,'scalers':scalers,'bands':bands,'ensemble_neural_weight':weights,'calibration_expansion_z':expansions,'calibration_counts':calibration_counts,'best_epoch':best_epoch,'train_trace':trace}
    dump(OUT/'setup.json',setup)
    records=[]
    for domain,spec in specs.items():
        for split in ('dev','calibration','test'):
            items=by[domain,split]
            if not items:continue
            for method in ('persistence','transformer','ensemble'):
                p=predictions(items,method,weights,expansions if method=='ensemble' else None)
                for i,sample in enumerate(items):
                    item=lookup[sample['series']]
                    for h,horizon in enumerate(spec['horizons']):
                        for v,variable in enumerate(spec['targets']):
                            y=float(sample['target'][h,v])
                            if not np.isfinite(y):continue
                            q10,q50,q90=map(float,p[i,h,v]);records.append({'source':domain,'case':sample['series'],'person':item['person'],'split':split,'method':method,'origin':sample['origin'],'horizon':horizon,'clock':spec['clock'],'variable':variable,'observed':y,'q10':q10,'q50':q50,'q90':q90,'absolute_error':abs(y-q50),'covered':q10<=y<=q90,'pinball':pinball(np.asarray(y),np.asarray([q10,q50,q90])),'normalized_pinball':pinball(np.asarray(y),np.asarray([q10,q50,q90]))/scalers[domain]['scale'][v],'width':q90-q10,'calibration_n':calibration_counts[domain][h][v]})
    frame=pd.DataFrame(records);frame.to_csv(OUT/'predictions.csv',index=False)
    metrics=frame.groupby(['source','split','method','variable','horizon','clock'],as_index=False).agg(n=('observed','size'),mae=('absolute_error','mean'),pinball=('pinball','mean'),normalized_pinball=('normalized_pinball','mean'),coverage=('covered','mean'),width=('width','mean'))
    metrics.to_csv(OUT/'metrics.csv',index=False)
    test=metrics[metrics.split=='test'];summary=test.groupby(['source','method'],as_index=False).agg(normalized_pinball=('normalized_pinball','mean'),coverage=('coverage','mean'),scored_targets=('n','sum'))
    dump(OUT/'summary.json',{'models':summary.to_dict('records'),'best_epoch':best_epoch,'ensemble_neural_weight':weights,'seconds':time.time()-started,'warning':'Exploratory domain-specific evaluation. No hourly astronaut clinical validation. Radiation test is temporal within one instrument.'})
    raw_files=list((RAW/'NASA_LSDA').rglob('*.csv'))+list((RAW/'NASA_RADLAB').glob('ISS_DosTel1_*.csv'))+list((RAW/'BIGIDEAS_1.1.3').rglob('*.csv'))
    dump(OUT/'fit_manifest.json',{'seed':2026,'source_files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw_files],
          'roles':{s['id']:{'person':s['person'],'role':s['role']} for s in series},'scalers_fit':'unique source rows from train persons; first 8 days only for radiation','selection':'dev only','ensemble':'dev only','calibration':'separate calibration roles; no expansion if n<20','test_used_for_fitting':False,'source_campaigns_v1_v2_modified':False,'dense_quarter_hour':args.dense_quarter_hour,'warm_start':str(args.warm_start) if args.warm_start else None,'warm_start_sha256':hashlib.sha256(args.warm_start.read_bytes()).hexdigest() if args.warm_start else None,'model_parameters':sum(p.numel() for p in model.parameters())})
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
