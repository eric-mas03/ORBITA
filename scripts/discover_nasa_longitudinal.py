"""Inspect public LSDA experiment catalogues; preserve per-file access and cadence evidence."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/research_expansion_20261114/NASA_LSDA'
EXPERIMENTS={
 'AP005_Apollo_cardiovascular':'e1fa5c55-ddfe-5548-91ba-864080319f44',
 'MIR_ambulatory_monitoring':'f8e3f755-5caa-522c-b628-7153cfe446f5',
 'BRSMCF_cardiovascular':'9af6f9ba-3730-54ab-a82a-1a32798667ae',
 'BRSMAct_actigraphy':'cd8a4a16-fcfd-5c32-a61f-22c590a10b35',
 'DSO484_circadian':'1bb44ffb-b00d-521a-bbeb-d8fcd036ad83',
 'BHP_standard_measures':'4591d315-4c81-520f-ba41-3608ef0c6dba',
 'SHUTTLE_Holter_EVA':'06699312-6756-5f70-a71b-f61abb300f3f',
 'CONCORDIA_BHP':'6e611fcf-06df-5ef0-b76d-7862f6c1b499',
 'ISS_BioMonitor':'f9f4485d-656c-5980-9a5f-4f57c7ba383a',
 'ISS_BiochemProfile':'2a30c3f0-93d8-5060-b546-ba229f740cc3',
}

def retrieve(pair):
    label,identifier=pair;url=f'https://nlsp.nasa.gov/api/v1/data/experiments/{identifier}/full'
    try:
        path=DEST/(label+'.json')
        if path.exists():body=path.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=35) as response:body=response.read(5_000_001)
        if len(body)>5_000_000:raise ValueError('Metadata exceeds bounded 5 MB')
        data=json.loads(body);path.write_bytes(body)
        related=data.get('related',{})
        groups={}
        for key,group in related.items():
            if key not in ('data_files','datasets','data','open_access_data'):continue
            if isinstance(group,list):groups[key]={'total':len(group),'items':group}
            else:groups[key]={'total':group.get('total'),'items':group.get('items',[])}
        return {'label':label,'id':identifier,'url':url,'path':str(path.relative_to(ROOT)),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'related_groups':list(related),'groups':groups,'status':'metadata_downloaded'}
    except Exception as exc:return {'label':label,'id':identifier,'url':url,'status':'not_downloaded','error':str(exc)}

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(retrieve,EXPERIMENTS.items()))
    (ROOT/'data/manifests/nasa_longitudinal_search_20261114.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    for r in results:
        print(r['label'],r['status'],r.get('error',''),r.get('related_groups',[]))
        for group,items in r.get('groups',{}).items():
            print(group,items['total'])
            for item in items['items'][:30]:print(item.get('key',item.get('id')),item.get('label',item.get('element_name')),item.get('meta',''))
