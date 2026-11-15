"""Versioned challenge-resource acquisition; never read held-out clinical records.

Only reads the specified existing manifests, snapshots public resource pages,
and downloads explicit allowlisted assets. No recursive dataset crawl.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def sanitized(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def inventory(config):
    result = []
    for relative in config['inventory_manifests']:
        path = ROOT/relative
        body = path.read_bytes()
        document = json.loads(body)
        entries = document if isinstance(document, list) else document.get('files', [])
        counts = Counter(entry.get('status', 'downloaded' if entry.get('sha256') else 'unspecified') for entry in entries)
        groups = Counter(entry.get('study', entry.get('dataset', 'other')) for entry in entries
                         if entry.get('status', 'downloaded') == 'downloaded')
        result.append({'manifest': relative, 'manifest_sha256': hashlib.sha256(body).hexdigest(),
                       'entries': len(entries), 'statuses': dict(counts), 'downloaded_entries_by_source': dict(groups),
                       'reported_downloaded_bytes': sum(entry.get('bytes', 0) for entry in entries
                             if entry.get('status', 'downloaded') == 'downloaded'),
                       'verification_scope': 'Manifest metadata only; no raw observations or held-out records opened.'})
    return result


def reconcile_official(config, raw):
    """Decode RSC JSON string payloads, never execute page JavaScript."""
    snapshot = ROOT/config.get('official_snapshot', 'missing')
    if not snapshot.is_file():
        return {'status': 'official_snapshot_not_available'}
    body = snapshot.read_bytes()
    page = body.decode('utf-8')
    parts = []
    for argument in re.findall(r'self\.__next_f\.push\((.*?)\)</script>', page, re.S):
        try:
            value = json.loads(argument)
            if len(value) > 1 and isinstance(value[1], str):
                parts.append(value[1])
        except (ValueError, TypeError):
            continue
    decoded = ''.join(parts)
    encoded = decoded.encode('utf-8')
    matches = []
    for marker in re.finditer(br'([0-9a-f]+):T([0-9a-f]+),', encoded):
        length = int(marker[2], 16)
        content = encoded[marker.end():marker.end()+length]
        if b'NASA Data &amp; Resources' in content:
            matches.append((marker[1].decode(), length, content))
    if len(matches) != 1:
        return {'status': 'unresolved_resources_block', 'matching_blocks': len(matches)}
    record, declared_length, content = matches[0]
    parser = Links()
    parser.feed(content.decode('utf-8'))
    links = list(dict.fromkeys(parser.links))
    expected = [resource['url'] for resource in config['resources']]
    verified = set(links) == set(expected)
    raw.mkdir(parents=True, exist_ok=True)
    extracted = raw/'official_resources_extracted.html'
    extracted.write_bytes(content)
    published = None
    index = decoded.find('lastPublishedAt')
    if index >= 0:
        date = re.search(r'20\d{2}-\d\d-\d\dT[\d:.]+Z', decoded[index:index+100])
        published = date[0] if date else None
    return {'status': 'verified_matches_catalogue' if verified else 'catalogue_mismatch',
            'resource_links': links, 'resource_count': len(links), 'rsc_record': record,
            'declared_utf8_bytes': declared_length, 'actual_utf8_bytes': len(content),
            'official_snapshot_path': str(snapshot.relative_to(ROOT)),
            'official_snapshot_bytes': len(body),
            'official_snapshot_sha256': hashlib.sha256(body).hexdigest(),
            'extracted_path': str(extracted.relative_to(ROOT)), 'extracted_sha256': hashlib.sha256(content).hexdigest(),
            'last_published_at': published,
            'http_status': config.get('official_snapshot_transport', {}).get('http_status'),
            'tls_verification': config.get('official_snapshot_transport', {}).get('tls_verification'),
            'transport': config.get('official_snapshot_transport'),
            'scope': 'Exact Resources block in retrieved official snapshot; not an exhaustive NASA dataset census.'}


def fetch(url, destination, max_bytes):
    result = {'url': url, 'attempted_at_utc': datetime.now(timezone.utc).isoformat(),
              'access': 'public_without_credentials', 'tls_verification': True}
    try:
        request = Request(url, headers={'User-Agent': 'AstronautTwinChallengeResourceAudit/2.0'})
        try:
            with urlopen(request, timeout=60) as response:
                length = response.headers.get('Content-Length')
                if length and int(length) > max_bytes:
                    return {**result, 'status': 'skipped_size_limit', 'reported_bytes': int(length), 'max_bytes': max_bytes}
                body = response.read(max_bytes+1)
                content_type = response.headers.get('Content-Type', '')
                result.update(http_status=response.status, resolved_url=sanitized(response.geturl()), content_type=content_type)
        except Exception as first_error:
            # Different public HTTP client for connection compatibility, with
            # TLS verification still enabled and no authentication bypass.
            import requests
            result['first_client_error'] = f'{type(first_error).__name__}: {first_error}'
            with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=40, stream=True) as response:
                response.raise_for_status()
                chunks, count = [], 0
                for chunk in response.iter_content(65536):
                    chunks.append(chunk)
                    count += len(chunk)
                    if count > max_bytes:
                        break
                body = b''.join(chunks)
                content_type = response.headers.get('Content-Type', '')
                result.update(http_status=response.status_code, resolved_url=sanitized(response.url),
                              content_type=content_type, client='requests_TLS_verified')
        if len(body) > max_bytes:
            return {**result, 'status': 'skipped_size_limit', 'observed_bytes_at_least': len(body), 'max_bytes': max_bytes}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        result.update(status='downloaded', path=str(destination.relative_to(ROOT)), bytes=len(body),
                      sha256=hashlib.sha256(body).hexdigest())
        if 'html' in content_type or body.lstrip().lower().startswith((b'<html', b'<!doctype html')):
            parser = Links()
            parser.feed(body.decode('utf-8', errors='replace'))
            links = sorted(set(urljoin(url, href) for href in parser.links if not href.startswith(('mailto:', 'javascript:', '#'))))
            result['outbound_links'] = links
            result['data_or_evidence_link_candidates'] = [href for href in links if re.search(
                r'\.(csv|tsv|xlsx?|json|xpt|zip|pdf)(?:[?#]|$)|osdr|nlsp|humanresearchroadmap|/data/|radi-n', href, re.I)]
            result['observations_added'] = 0
        return result
    except HTTPError as error:
        return {**result, 'status': 'http_error', 'http_status': error.code,
                'reason': 'Access not bypassed. HTTP403/401 does not by itself prove a dataset-level restriction.'}
    except Exception as error:
        return {**result, 'status': 'network_error', 'reason': f'{type(error).__name__}: {error}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/challenge_resources_v2.json')
    parser.add_argument('--offline', action='store_true', help='Only inventory prior manifests; do not fetch pages')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    raw = ROOT/config['raw_directory']
    if ROOT.resolve() not in raw.resolve().parents:
        raise ValueError('Output directory must be inside this workspace')
    reconciliation = reconcile_official(config, raw)
    config['official_list_verified'] = reconciliation['status'] == 'verified_matches_catalogue'
    if config['official_list_verified']:
        config['completeness_claim'] = 'All five links in the retrieved official Resources block are reconciled; not an exhaustive inventory of all NASA datasets.'
        args.config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    acquisitions = []
    if not args.offline:
        if config['official_list_verified']:
            acquisitions.append({'role': 'official_challenge_listing', 'status': 'official_snapshot_reconciled',
                'url': config['challenge_url'], **reconciliation})
        else:
            acquisitions.append({'role': 'official_challenge_listing', **fetch(config['challenge_url'], raw/'official_resources.html', config['max_document_bytes'])})
        acquisitions.append({'role': 'local_organizer_mirror', **fetch(config['listing_evidence_url'], raw/'organizer_mirror.html', config['max_document_bytes'])})
        for resource in config['resources']:
            acquisitions.append({'resource_id': resource['id'], 'role': 'resource_page',
                                 'classification': resource['kind'], 'training_eligibility': resource['training_eligibility'],
                                 'eligibility_reason': resource['reason'],
                                 **fetch(resource['url'], raw/(resource['id']+'.html'), config['max_document_bytes'])})
            for asset in resource.get('direct_downloads', []):
                if asset.get('restricted') is not False:
                    acquisitions.append({'resource_id': resource['id'], 'url': asset['url'],
                        'status': 'skipped_restricted_or_unverified_access', 'reason': 'Explicit public access evidence required.'})
                    continue
                name = Path(asset['filename']).name
                acquisitions.append({'resource_id': resource['id'], 'role': 'explicit_data_asset',
                    'classification': asset.get('kind', 'unclassified'),
                    **fetch(asset['url'], raw/'assets'/name, config['max_file_bytes'])})
        for resource in config.get('follow_up_pages', []):
            acquisitions.append({'resource_id': resource['id'], 'parent_resource': resource['parent_resource'],
                'role': 'directly_linked_follow_up_page', 'classification': resource['kind'],
                'eligibility_reason': resource['reason'],
                **fetch(resource['url'], raw/'follow_up'/(resource['id']+'.html'), config['max_document_bytes'])})
    previous_path = ROOT/'data/manifests/challenge_resources_v2.json'
    prior_failures = []
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        prior_failures = previous.get('historical_failed_attempts', []) + [item for item in previous.get('acquisitions', []) if item.get('status') in ('network_error', 'http_error')]
        if args.offline:
            acquisitions = previous.get('acquisitions', [])
    report = {'schema_version': '2.0', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'official_list_verified': config['official_list_verified'],
              'completeness_claim': config['completeness_claim'], 'challenge_url': config['challenge_url'],
              'resource_catalogue': config['resources'], 'official_reconciliation': reconciliation,
              'prior_inventory': inventory(config), 'historical_failed_attempts': prior_failures,
              'acquisitions': acquisitions, 'campaign_v1_modified': False,
              'protected_records_read': False,
              'scope': 'Documentation pages and explicit public allowlisted assets only. No automatic recursive claim of resource completeness.',
              'observational_dataset_gaps': [
                  {'source': 'Inspiration4', 'state': 'already_downloaded', 'reason': 'Four people; sparse clinical visits, not additional hourly astronaut labels.'},
                  {'source': 'NASA LSDA UTMB', 'state': 'already_downloaded', 'reason': 'Clinical visits and daily sleep; preserve their actual sampling cadence.'},
                  {'source': 'OSD-140', 'state': 'partial_download', 'reason': 'Only two processed sample tables fit prior budget; other archive exceeds160MB; not a complete additional cohort.'},
                  {'source': 'OSD-353', 'state': 'catalogue_only', 'reason': 'Processed proteomic archive exceeds29GB; raw omics not useful as direct hourly clinical labels.'},
                  {'source': 'Ax-1 OSD-903/943', 'state': 'metadata_only', 'reason': 'Prior ISA downloads returned403; no access bypass; metadata not observations.'},
                  {'source': 'challenge_resource_pages', 'state': 'no_new_verified_hourly_health_table', 'reason': 'Five official resources are documentary pages; no new labeled hourly health table found in the resource pages.'}]}
    report['secondary_link_dispositions'] = []
    for acquired in acquisitions:
        for link in acquired.get('data_or_evidence_link_candidates', []):
            if any(link == other.get('url') for other in acquisitions):
                disposition, reason = 'already_catalogued_or_acquired', 'The linked landing page has its own acquisition record.'
            elif 'participation-kit' in link:
                disposition, reason = 'not_observational_event_document', 'Organizer participation kit; not a health dataset or one of the five official scientific resources.'
            elif 'humanresearchroadmap.nasa.gov' in link:
                disposition, reason = 'evidence_or_research_catalogue_not_confirmed_health_table', 'Risk/evidence/programme links, not a verified person-time-value dataset. Recursive content beyond acquired landing page not exhaustively audited.'
            elif 'radi-n2-and-you' in link:
                disposition, reason = 'educational_environmental_follow_up_not_paired_clinical_labels', 'Activity link; no verified astronaut health outcome table paired with personal exposure.'
            else:
                disposition, reason = 'not_verified_as_observational_dataset', 'Link retained for review; no direct public health observation file confirmed.'
            report['secondary_link_dispositions'].append({'parent_resource': acquired.get('resource_id', acquired.get('role')),
                'url': link, 'disposition': disposition, 'reason': reason, 'download_size_bytes': None})
    report['summary'] = {'official_resource_count': reconciliation.get('resource_count'),
        'official_resources_reconciled': bool(config['official_list_verified']),
        'downloaded_resource_and_follow_up_pages': sum(item.get('status') == 'downloaded' for item in acquisitions),
        'downloaded_snapshot_bytes': sum(item.get('bytes', 0) for item in acquisitions if item.get('status') == 'downloaded'),
        'new_verified_observational_datasets': 0,
        'new_verified_hourly_health_labels': 0,
        'interpretation': 'Documentation coverage improved; no new training observations confirmed in the five resources or acquired linked landing pages.'}
    destination = ROOT/'data/manifests/challenge_resources_v2.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'official_list_verified': report['official_list_verified'],
                      'resources_catalogued': len(config['resources']),
                      'acquisitions': Counter(item['status'] for item in acquisitions),
                      'prior_manifests_inventoried': len(report['prior_inventory'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
