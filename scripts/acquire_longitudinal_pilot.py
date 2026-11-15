"""Acquire explicitly selected open longitudinal files for feasibility, not model fitting."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/research_expansion_20261114'
BIG='https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/'
CG='https://physionet.org/files/cgmacros/1.0.0/'
FILES=[(f'BIGIDEAS_1.1.3/{sid:03d}/{kind}_{sid:03d}.csv',BIG+f'{sid:03d}/{kind}_{sid:03d}.csv') for sid in range(1,17) for kind in ('Dexcom','Food_Log')]
FILES += [(f'BIGIDEAS_1.1.3/{name}',BIG+name) for name in ('Demographics.csv','LICENSE.txt','SHA256SUMS.txt','001/HR_001.csv')]
FILES += [(f'BIGIDEAS_1.1.3/{sid:03d}/HR_{sid:03d}.csv',BIG+f'{sid:03d}/HR_{sid:03d}.csv') for sid in range(2,17)]
FILES += [(f'CGMACROS_1.0.0/{name}',CG+name) for name in ('DataDictionary_Bio.csv','DataDictionary_CGMacros-00X.csv','LICENSE.txt','SHA256SUMS.txt')]

def acquire(item):
    name,url=item; path=DEST/name; path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if path.exists():payload=path.read_bytes()
        else:
            with urllib.request.urlopen(url,timeout=45) as response:
                payload=response.read(20_000_001)
            if len(payload)>20_000_000:raise ValueError('File exceeds explicit 20 MB cap')
            path.write_bytes(payload)
        return {'path':str(path.relative_to(ROOT)),'url':url,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'status':'downloaded'}
    except Exception as exc:return {'path':str(path.relative_to(ROOT)),'url':url,'status':'not_downloaded','error':str(exc)}

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(acquire,FILES))
    (ROOT/'data/manifests/longitudinal_pilot_20261114.json').write_text(json.dumps({'purpose':'feasibility_only','domain':'terrestrial_wearable','frozen_campaigns_modified':False,'files':results},indent=2),encoding='utf8')
    ok=[r for r in results if r['status']=='downloaded']
    print(json.dumps({'files_downloaded':len(ok),'bytes':sum(r['bytes'] for r in ok),'errors':[r for r in results if r['status']!='downloaded']},indent=2))
