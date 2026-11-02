"""Read small public metadata from the user-identified GCS bucket only.

No credentials, dataset shard downloads, training, or numeric sample inspection.
Snapshots distinguish actual bucket releases from local TFDS importer versions.
"""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_bucket_metadata_20261020'
LIMIT = 1024 * 1024
HEADERS = {'User-Agent': 'OXE-public-metadata-audit/1.0'}


def fetch(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = response.read(LIMIT + 1)
    if len(payload) > LIMIT:
        raise ValueError('Metadata exceeded 1 MiB limit')
    return payload


def feature_leaves(node, prefix=''):
    if 'featuresDict' in node:
        leaves = {}
        for name, child in node['featuresDict']['features'].items():
            leaves.update(feature_leaves(child, prefix+'/'+name))
        return leaves
    if 'sequence' in node:
        return feature_leaves(node['sequence']['feature'], prefix)
    return {prefix:node}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    attempts = []
    releases = {}
    for dataset, versions, local_folder in [
        ('language_table', ['0.0.1', '0.1.0'], 'language_table_real'),
        ('bc_z', ['0.1.0', '1.0.0'], 'bc_z'),
    ]:
        local_dir = ROOT/'training_cache/oxe_core'/local_folder
        local_info = json.loads((local_dir/'dataset_info.json').read_text())
        local_features = json.loads((local_dir/'features.json').read_text())
        local_leaves = feature_leaves(local_features)
        releases[dataset] = {'local_version':local_info['version'],
                             'local_module':local_info.get('moduleName'),
                             'local_metadata_sha256':hashlib.sha256((local_dir/'dataset_info.json').read_bytes()).hexdigest(),
                             'local_features_sha256':hashlib.sha256((local_dir/'features.json').read_bytes()).hexdigest(),
                             'remote_versions':{}}
        for version in versions:
            record = {}
            for filename in ['dataset_info.json', 'features.json']:
                url = f'https://storage.googleapis.com/gresearch/robotics/{dataset}/{version}/{filename}'
                attempt = {'url':url}
                try:
                    payload = fetch(url)
                    value = json.loads(payload)
                    snapshot = f'{dataset}_{version}_{filename}'
                    (OUT/snapshot).write_bytes(payload)
                    attempt.update(success=True, bytes=len(payload), snapshot=snapshot,
                                   sha256=hashlib.sha256(payload).hexdigest())
                    if filename == 'dataset_info.json':
                        record.update(name=value.get('name'), version=value.get('version'),
                                      module=value.get('moduleName'), release_notes=value.get('releaseNotes'),
                                      splits=value.get('splits'), metadata_snapshot=snapshot)
                    else:
                        leaves = feature_leaves(value)
                        common = local_leaves.keys() & leaves.keys()
                        record.update(features_snapshot=snapshot,
                                      feature_json_equal_to_local=value == local_features,
                                      local_only_features=sorted(local_leaves.keys()-leaves.keys()),
                                      remote_only_features=sorted(leaves.keys()-local_leaves.keys()),
                                      changed_common_feature_definitions=sorted(k for k in common if local_leaves[k] != leaves[k]),
                                      time_named_fields=[k for k in leaves if any(t in k.lower() for t in ['timestamp','time_stamp','elapsed','time_us','time_ms'])],
                                      unit_documentation={k:v for k,v in leaves.items() if any(t in json.dumps(v).lower() for t in ['meter','radian','millimet','centimet'])})
                except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
                    attempt.update(success=False, error_type=type(error).__name__, error=str(error))
                attempts.append(attempt)
            releases[dataset]['remote_versions'][version] = record
    report = {
        'bucket_source_confirmed_by_user':'gs://gresearch/robotics',
        'user_language_table_link_version':'0.0.1',
        'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'attempts':attempts, 'releases':releases,
        'all_metadata_reads_succeeded':all(a['success'] for a in attempts),
        'source_shards_downloaded':False, 'source_payloads_parsed':False,
        'production_code_changed':False, 'training_or_inference_run':False,
        'all_export_units_and_timestamps_verified':False, 'training_ready':False,
    }
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'successes':sum(a['success'] for a in attempts), 'attempts':len(attempts),
                      'releases':{d:{v:{k:r[k] for k in ['version','module','feature_json_equal_to_local','remote_only_features','local_only_features','changed_common_feature_definitions','time_named_fields','unit_documentation'] if k in r}
                                     for v,r in s['remote_versions'].items()} for d,s in releases.items()}}, indent=2))


if __name__ == '__main__':
    main()
