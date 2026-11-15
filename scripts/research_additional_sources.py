"""Capture a bounded set of public source catalogues without changing frozen experiments."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/research_expansion_20261114'
SOURCES={
 'cgmacros_index.html':'https://physionet.org/files/cgmacros/1.0.0/',
 'bigideas_index.html':'https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/',
 'bigideas_subject001.html':'https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3/001/',
 'physiocgm_article.json':'https://api.figshare.com/v2/articles/28136294',
 'openmhc_dataset.md':'https://raw.githubusercontent.com/AshleyLab/OpenMHC/main/DATASET.md',
 'analog_hrv_article.html':'https://www.mdpi.com/2226-4310/11/10/833',
 'cgmacros_description.html':'https://physionet.org/content/cgmacros/1.0.0/',
 'bigideas_description.html':'https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/',
}

def retrieve(name,url):
    target=DEST/name
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'AstronautTwin-Research/1.0'})
        with urllib.request.urlopen(request,timeout=30) as response:
            payload=response.read(5_000_001)
            if len(payload)>5_000_000: raise ValueError('Catalogue exceeds 5 MB cap')
            resolved=response.url
        target.write_bytes(payload)
        return {'file':str(target.relative_to(ROOT)),'url':url,'resolved_url':resolved,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'status':'downloaded_catalogue'}
    except Exception as exc:
        return {'file':str(target.relative_to(ROOT)),'url':url,'status':'not_downloaded','error':str(exc)}

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(lambda pair:retrieve(*pair),SOURCES.items()))
    manifest=ROOT/'data/manifests/research_expansion_20261114.json'
    manifest.write_text(json.dumps({'catalogues':results,'observations_added':False,'frozen_campaigns_modified':False},indent=2),encoding='utf8')
    for r in results:print(Path(r['file']).name,r['status'],r.get('bytes',r.get('error')))
