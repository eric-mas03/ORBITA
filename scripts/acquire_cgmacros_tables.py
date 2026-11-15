"""Read only CSV members of the open CGMacros ZIP through verified HTTP byte ranges."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip'
DEST=ROOT/'data/raw/research_expansion_20261114/CGMACROS_1.0.0/tables'

class PublicRangeReader(io.RawIOBase):
    def __init__(self):
        with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=30) as r:self.size=int(r.headers['Content-Length'])
        self.position=0;self.transferred=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=(0 if whence==0 else self.position if whence==1 else self.size)+offset
        if self.position<0:raise ValueError('Negative seek')
        return self.position
    def read(self,size=-1):
        size=min(self.size-self.position,size if size>=0 else self.size-self.position)
        if size<=0:return b''
        if size>10_000_000 or self.transferred+size>60_000_000:raise ValueError('Bounded partial-download budget exceeded')
        start=self.position;end=start+size-1
        request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=45) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server does not support exact byte ranges; full archive not downloaded')
            payload=r.read(size+1)
        if len(payload)!=size:raise ValueError('Unexpected range length')
        self.position+=size;self.transferred+=size;return payload

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True);remote=PublicRangeReader();records=[];names=set()
    with zipfile.ZipFile(remote) as archive:
        selected=[f for f in archive.infolist() if f.filename.lower().endswith('.csv') and not f.filename.startswith('__MACOSX/')]
        if sum(f.file_size for f in selected)>100_000_000:raise ValueError('Selected tables exceed 100 MB')
        for item in selected:
            if item.file_size>10_000_000:raise ValueError('Individual table exceeds 10 MB')
            filename=Path(item.filename).name
            if filename in names:raise ValueError('Duplicate output filename')
            names.add(filename);payload=archive.read(item)  # zipfile verifies member CRC32
            path=DEST/filename;path.write_bytes(payload)
            records.append({'member':item.filename,'path':str(path.relative_to(ROOT)),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'zip_crc32_verified':True})
    manifest={'source':URL,'mode':'public_HTTP_ranges_CSV_members_only','full_zip_downloaded':False,'archive_bytes':remote.size,
              'network_bytes':remote.transferred,'files':records,'domain':'terrestrial_wearable','frozen_campaigns_modified':False}
    (ROOT/'data/manifests/cgmacros_tables_20261114.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'csv_files':len(records),'table_bytes':sum(r['bytes'] for r in records),'network_bytes':remote.transferred,'archive_bytes':remote.size}))
