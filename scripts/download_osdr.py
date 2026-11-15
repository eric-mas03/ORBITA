"""Download small, explicitly unrestricted OSDR tables and their provenance.

Run: python scripts/download_osdr.py --studies 575 570
No credentials, no controlled access, no large sequencing data.
"""
import argparse
import hashlib
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qs, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://osdr.nasa.gov'


def get(url, max_bytes=10_000_000):
    with urlopen(Request(url, headers={'User-Agent': 'AstronautTwinDataAudit/0.1'}), timeout=90) as response:
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError('Response exceeds configured download limit')
        return body, response.geturl(), response.headers.get('Content-Type', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs='+', type=int, default=[575])
    parser.add_argument('--max-file-bytes', type=int, default=2_000_000)
    parser.add_argument('--include', action='append', default=[], help='Optional filename glob allowlist; repeat for several patterns')
    args = parser.parse_args()
    manifest = []
    for study in args.studies:
        directory = ROOT / 'data' / 'raw' / f'OSD-{study}'
        directory.mkdir(parents=True, exist_ok=True)
        catalog = None
        for kind in ('meta', 'files'):
            url = f'{BASE}/osdr/data/osd/{kind}/{study}'
            body, final_url, content_type = get(url)
            target = directory / f'{kind}.json'
            target.write_bytes(body)
            manifest.append(record(target, url, final_url, body, content_type, 'public_metadata'))
            if kind == 'files':
                catalog = json.loads(body)
        for item in catalog['studies'][f'OSD-{study}']['study_files']:
            filename = item['file_name']
            explicitly_selected = any(fnmatch.fnmatchcase(filename, pattern) for pattern in args.include)
            allowed_suffix = filename.lower().endswith(('.csv', '.tsv', '.xlsx', '.txt', '-isa.zip')) or (explicitly_selected and filename.lower().endswith('.zip'))
            selected = not args.include or explicitly_selected or filename.lower().endswith('-isa.zip')
            if item.get('restricted') is not False or not selected or not allowed_suffix or item['file_size'] > args.max_file_bytes:
                manifest.append({'study': f'OSD-{study}', 'filename': filename, 'status': 'skipped',
                                 'restricted': item.get('restricted'), 'catalog_bytes': item['file_size']})
                continue
            url = urljoin(BASE, item['remote_url'])
            try:
                body, final_url, content_type = get(url, args.max_file_bytes)
                if body.lstrip().lower().startswith((b'<!doctype html', b'<html')):
                    raise ValueError('Received HTML instead of data file')
                target = directory / Path(filename).name
                target.write_bytes(body)
                entry = record(target, url, final_url, body, content_type, 'catalog_restricted_false')
                entry.update(category=item.get('category'), catalog_bytes=item['file_size'], restricted=False)
                manifest.append(entry)
                print(f'Downloaded {target.relative_to(ROOT)} ({len(body)} bytes)')
            except Exception as error:
                manifest.append({'study': f'OSD-{study}', 'filename': filename, 'url': url,
                                 'status': 'error', 'error': str(error)})
                print(f'Failed {filename}: {error}')
    output = ROOT / 'data' / 'manifests' / 'osdr_downloads.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(output.read_text(encoding='utf-8')) if output.exists() else []
    touched = {f'OSD-{study}' for study in args.studies}
    previous = [entry for entry in previous if entry.get('study') not in touched]
    output.write_text(json.dumps(previous + manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def record(path, url, final_url, body, content_type, access):
    parsed = urlsplit(final_url)
    version = parse_qs(parsed.query).get('versionId')
    safe_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path,
                          urlencode({'versionId': version[0]}) if version else '', ''))
    return {'study': path.parent.name, 'path': str(path.relative_to(ROOT)), 'url': url,
            'resolved_url': safe_url, 'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
            'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
            'content_type': content_type, 'access_evidence': access, 'status': 'downloaded'}


if __name__ == '__main__':
    main()
