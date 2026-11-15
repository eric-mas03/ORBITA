"""Acquire explicitly public NASA cardiovascular tables and a bounded RadLab pilot."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/research_expansion_20261114'
DATASETS={'BRSMCF_C1':'95186651-28d7-5c72-8f32-d40838766078','BRSMCF_C3':'c3bdfa7e-6661-5891-9ba0-0fc7c7a080d6','BRSMCF_CFT70':'59501092-7252-5848-b79c-650e954b51ed'}

def fetch(url,limit=20_000_000):
    with urllib.request.urlopen(url,timeout=45) as response:
        data=response.read(limit+1)
        if len(data)>limit:raise ValueError('Bounded download exceeded')
    return data

def store(data,path,url,**extra):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'url':url,'path':str(path.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded',**extra}

def dataset(pair):
    label,identifier=pair;url=f'https://nlsp.nasa.gov/api/v1/data/datasets/{identifier}/full';records=[]
    try:
        body=fetch(url);data=json.loads(body);records.append(store(body,DEST/'NASA_LSDA'/label/'catalog.json',url,kind='catalogue'))
        if data['record'].get('__is_locked') is not False:raise ValueError('Dataset not explicitly public')
        group=data['related']['data_files']
        if group.get('total')!=len(group['items']):raise ValueError('File list truncated')
        for item in group['items']:
            filename=item.get('__original_file_name',item.get('file_name',''))
            if item.get('__is_locked') is not False or not filename.lower().endswith(('.csv','.txt','.xlsx','.xls','.tsv')):continue
            file_url=f"https://nlsp.nasa.gov/api/v1/files/{item['id']}/download"
            try:
                body=fetch(file_url)
                if body.lstrip().lower().startswith((b'<html',b'<!doctype html')):raise ValueError('HTML instead of table')
                md5=item.get('__file_md5sum')
                if md5 and hashlib.md5(body).hexdigest()!=md5:raise ValueError('NASA MD5 mismatch')
                records.append(store(body,DEST/'NASA_LSDA'/label/Path(filename).name,file_url,kind='public_table',catalog_md5=md5))
            except Exception as exc:records.append({'url':file_url,'status':'not_downloaded','error':str(exc)})
    except Exception as exc:records.append({'url':url,'status':'not_downloaded','error':str(exc)})
    return {'label':label,'records':records}

def radlab():
    # Use the documented parenthesized syntax. This API rejects '+' as encoded spaces.
    url='https://visualization.osdr.nasa.gov/radlab/api?query=(timestamp%3E=2022-03-01)AND(timestamp%3C2022-03-15)AND(instrument_id=DosTel1)&fields+=absorbed_dose_rate&format=csv'
    try:return {'label':'RADLAB_ISS_DosTel1_14days','records':[store(fetch(url),DEST/'NASA_RADLAB/ISS_DosTel1_20220301_20220315.csv',url,kind='radiation_measurements',domain='spacecraft_environment_not_personal_dose')]}
    except Exception as exc:return {'label':'RADLAB_ISS_DosTel1_14days','records':[{'url':url,'status':'not_downloaded','error':str(exc)}]}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(dataset,pair) for pair in DATASETS.items()]+[pool.submit(radlab)]
        results=[job.result() for job in jobs]
    (ROOT/'data/manifests/nasa_followup_20261114.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    for result in results:
        print(result['label'])
        for record in result['records']:print(Path(record.get('path','')).name,record['status'],record.get('bytes',record.get('error')))
