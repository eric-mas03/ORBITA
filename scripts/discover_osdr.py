"""Save reproducible public OSDR discovery queries, without downloading raw omics."""
import argparse
import hashlib
import csv
import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/manifests/discovery'


def save(url, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=90) as response:
        body = response.read(20_000_000)
    (OUT / name).write_bytes(body)
    log = OUT / 'queries.jsonl'
    with log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'url': url, 'path': str((OUT/name).relative_to(ROOT)),
                                 'at_utc': datetime.now(timezone.utc).isoformat(), 'bytes': len(body)})+'\n')
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terms', nargs='*', default=['Axiom', 'bed rest', 'HERA', 'SIRIUS', 'Twins', 'Polaris', 'Shenzhou'])
    parser.add_argument('--lsda-dataset', help='Public NASA LSDA dataset UUID to download')
    parser.add_argument('--lsda-label', help='Local directory name under data/raw')
    parser.add_argument('--inventory', action='store_true', help='Summarize current space datasets and verify downloaded hashes')
    args = parser.parse_args()
    if args.inventory:
        inventory()
        return
    if args.lsda_dataset:
        download_lsda(args.lsda_dataset, args.lsda_label)
        return
    index = save('https://visualization.genelab.nasa.gov/data/', 'visualization_index.html').decode()
    for content in re.findall(r'<script[^>]*>(.*?)</script>', index, flags=re.S):
        if content.lstrip().startswith(('{', '[')):
            try:
                obj = json.loads(content)
                print('Embedded index:', str(obj)[:2500])
            except ValueError:
                pass
    for term in args.terms:
        url = 'https://osdr.nasa.gov/osdr/data/search?' + urlencode({'term': term, 'size': 100})
        name = re.sub(r'\W+', '_', term.lower()) + '.json'
        try:
            result = json.loads(save(url, name))
            hits = result.get('hits', {}).get('hits', [])
            relevant = [hit for hit in hits if hit.get('_source', {}).get('Data Source Type') in ('cgene', 'alsda')]
            print(term, 'total', result.get('hits', {}).get('total'), 'OSDR candidates',
                  [(x.get('_source', {}).get('Accession'), x.get('_source', {}).get('Study Title'), x.get('_source', {}).get('organism')) for x in relevant])
        except Exception as error:
            print(term, type(error).__name__, str(error))


def download_lsda(dataset_id, label=None):
    """Download only public, unlocked, small tabular files using LSDA's API."""
    base = 'https://nlsp.nasa.gov/api/v1'
    url = f'{base}/data/datasets/{dataset_id}/full'
    with urlopen(url, timeout=90) as response:
        body = response.read(20_000_000)
    catalog = json.loads(body)
    if catalog['record'].get('__is_locked') is not False:
        raise ValueError('Dataset is not explicitly unlocked')
    local_name = label or ('LSDA-' + dataset_id)
    if Path(local_name).name != local_name or local_name in ('.', '..'):
        raise ValueError('Directory label must be a single safe component')
    directory = ROOT / 'data/raw' / local_name
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'clinical_catalog.json'
    target.write_bytes(body)

    def record(path, source_url, data, **extra):
        return {'dataset': dataset_id, 'path': str(path.relative_to(ROOT)), 'url': source_url,
                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'downloaded', **extra}

    records = [record(target, url, body, access_evidence='public_dataset_metadata')]
    files = catalog.get('related', {}).get('data_files', {})
    items = files.get('items', [])
    if files.get('total', 0) != len(items):
        raise ValueError('Catalog is truncated; resolve pagination before downloading')

    def download(item):
        name = item.get('__original_file_name', item.get('file_name', ''))
        entry = {'dataset': dataset_id, 'file_id': item.get('id'), 'filename': name}
        if item.get('__is_locked') is not False or not name.lower().endswith(('.csv', '.tsv', '.txt', '.xlsx', '.xls')):
            return {**entry, 'status': 'skipped', 'reason': 'locked_or_not_tabular'}
        source_url = f"{base}/files/{item['id']}/download"
        try:
            with urlopen(source_url, timeout=90) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000 or data.lstrip().lower().startswith((b'<html', b'<!doctype html')):
                raise ValueError('Oversized file or HTML response')
            expected_md5 = item.get('__file_md5sum')
            if expected_md5 and hashlib.md5(data).hexdigest() != expected_md5:
                raise ValueError('Downloaded bytes do not match NASA catalog MD5')
            path = directory / Path(name).name
            path.write_bytes(data)
            print('Downloaded', path.relative_to(ROOT), len(data), flush=True)
            return record(path, source_url, data, access_evidence='catalog_file_explicitly_unlocked',
                          file_id=item['id'], catalog_md5=expected_md5)
        except Exception as error:
            print('Failed', name, str(error), flush=True)
            return {**entry, 'url': source_url, 'status': 'error', 'error': str(error)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        records.extend(executor.map(download, items))
    manifest_path = ROOT / 'data/manifests/lsda_downloads.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    previous = [entry for entry in previous if entry.get('dataset') != dataset_id]
    manifest_path.write_text(json.dumps(previous + records, indent=2), encoding='utf-8')


def inventory():
    """Report downloaded samples separately from people and declared metadata."""
    datasets = []
    clinical = ROOT / 'data/raw/LSDA-UTMB-C3'
    records = []
    for path in clinical.glob('*.csv'):
        with path.open(newline='', encoding='utf-8-sig') as stream:
            records.extend(csv.DictReader(stream))
    visits = {(r['Campaign'], r['Subject'], r['Test_Phase'], r['BR_Day'], r['Date']) for r in records}
    datasets.append({'dataset': 'LSDA-UTMB-C3', 'priority': 1, 'modality': 'clinical_CBC_CMP_and_other',
                     'subjects_local_ids': len({r['Subject'] for r in records}),
                     'campaigns': sorted({r['Campaign'] for r in records}), 'rows': len(records),
                     'subject_visits': len(visits), 'parameter_labels': len({r.get('ParameterName', r.get('Parameter Name')) for r in records}),
                     'csv_files': len(list(clinical.glob('*.csv'))), 'readme_files': len(list(clinical.glob('*ReadMe.txt'))),
                     'cohort_key': 'UTMB_C3', 'source_url': 'https://nlsp.nasa.gov/explore/data/details/a36f6361-3492-5590-a675-a0257edf69ac?dataType=DATASETS'})
    c1 = ROOT / 'data/raw/LSDA-UTMB-C1'
    with (c1/'MR010G_Campaign_1_Flight_Chem_Profile.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    datasets.append({'dataset': 'LSDA-UTMB-C1', 'priority': 2, 'modality': 'clinical_CBC_CMP_and_other',
                     'subjects_local_ids': len({r['Subject'] for r in rows}), 'subject_visits': len(rows),
                     'csv_files': len(list(c1.glob('*.csv'))), 'cohort_key': 'UTMB_C1',
                     'source_url': 'https://nlsp.nasa.gov/explore/data/details/042df6cd-2a1a-59f7-903f-c32ab3b7c9de?dataType=DATASETS'})
    actigraphy = ROOT/'data/raw/LSDA-UTMB-C1-ACTIGRAPHY/BRSMACT_Campagin_1_Actigraphy.csv'
    with actigraphy.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    datasets.append({'dataset': 'LSDA-UTMB-C1-ACTIGRAPHY', 'priority': 3, 'modality': 'daily_sleep',
                     'subjects_local_ids': len({r['Subject'] for r in rows}), 'subject_visits': len(rows),
                     'distinct_dates': len({r['Date'] for r in rows}), 'cohort_key': 'UTMB_C1',
                     'new_people_relative_to_C1': 0,
                     'source_url': 'https://nlsp.nasa.gov/explore/data/details/e26501eb-7d7a-5012-8c8e-c5bdb3eecaf6?dataType=DATASETS'})
    for sid, subject_key, cohort, modality in [
            (191, 'Characteristics[Subject]', 'MARS500', 'gut_microbiome_counts'),
            (174, 'Source Name', 'ISS_JAXA_HAIR', 'hair_transcriptomics'),
            (53, 'Characteristics[Study Subject]', 'STS_SHUTTLE_BLOOD', 'blood_expression'),
            (198, 'Characteristics[Study Subject]', 'PLANHAB_CROSSOVER', 'muscle_miRNA'),
            (140, 'Characteristics [individual identifier]', 'MARS500', 'blood_methylation')]:
        directory = ROOT/'data/raw'/f'OSD-{sid}'
        archive = zipfile.ZipFile(next(directory.glob('*ISA.zip')))
        study_file = next(name for name in archive.namelist() if name.startswith('s_'))
        rows = list(csv.DictReader(io.StringIO(archive.read(study_file).decode('utf-8-sig')), delimiter='\t'))
        item = {'dataset': f'OSD-{sid}', 'cohort_key': cohort, 'modality': modality,
                'metadata_samples': len(rows), 'metadata_subject_labels': sorted({r[subject_key] for r in rows}),
                'source_url': f'https://osdr.nasa.gov/bio/repo/data/studies/OSD-{sid}'}
        if sid == 191:
            with next(directory.glob('*taxonomy-and-counts.tsv')).open(newline='') as stream:
                counts = csv.reader(stream, delimiter='\t')
                columns = next(counts)
                item.update(downloaded_sample_columns=len(columns)-8, asv_rows=sum(1 for _ in counts), people=6)
        elif sid == 174:
            data_zip = zipfile.ZipFile(next(path for path in directory.glob('*.zip') if 'ISA' not in path.name))
            item.update(downloaded_sample_tables=len(data_zip.namelist()), people=10, visits_per_person=6,
                        temporal_precision='collection_windows_not_exact_dates')
        elif sid == 53:
            data_zip = zipfile.ZipFile(next(path for path in directory.glob('*.zip') if 'ISA' not in path.name))
            item.update(downloaded_sample_tables=len(data_zip.namelist()), individually_resolved_astronauts=6,
                        astronaut_visits=12, pooled_control_samples=1, people_in_control_pool=5)
        elif sid == 198:
            item.update(people=13, downloaded_samples=68, phases_per_campaign=2,
                        campaigns=['HAMB', 'HBR', 'NBR'], warning='Crossover campaigns are not six consecutive visits')
        elif sid == 140:
            data_zip = zipfile.ZipFile(next(path for path in directory.glob('*.zip') if 'ISA' not in path.name))
            item.update(metadata_people=6, metadata_visits_per_person=6, downloaded_sample_tables=len(data_zip.namelist()),
                        complete_longitudinal_data_downloaded=False, warning='Only processed part2 fits the download budget; do not train on an assumed complete cohort')
        datasets.append(item)
    for sid in (943, 903):
        datasets.append({'dataset': f'OSD-{sid}', 'cohort_key': 'AXIOM1_BLOOD_TWO_PEOPLE',
                         'people_declared': 2, 'visits_per_person_declared': 3,
                         'data_downloaded': False, 'status': 'metadata_JSON_only_ISA_download_HTTP403',
                         'source_url': f'https://osdr.nasa.gov/bio/repo/data/studies/OSD-{sid}'})
    manifests = []
    for name in ('osdr_downloads.json', 'lsda_downloads.json'):
        for entry in json.loads((ROOT/'data/manifests'/name).read_text(encoding='utf-8')):
            if entry.get('status') == 'downloaded':
                data = (ROOT/entry['path']).read_bytes()
                assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
                manifests.append(entry)
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'datasets': datasets,
              'verified_downloaded_files': len(manifests), 'verified_bytes': sum(r['bytes'] for r in manifests),
              'files': manifests, 'warning': 'Never sum people across studies/campaigns without a validated identity crosswalk; omics panels do not add people'}
    output = ROOT/'data/manifests/space_expansion_summary.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
