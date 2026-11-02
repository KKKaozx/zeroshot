"""Enumerate existing cached train episodes without reading TFRecord payloads.

Candidate pool only: no window selection, split assignment, source contract
acceptance, model, training, or final task access. Existing old heldouts excluded
by shard/record identity; raw-origin alias checks remain required.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/oxe_baseline_candidates_20261020'
SOURCES = {'Bridge': 'bridge_v2', 'Language Table': 'language_table_real',
           'BC-Z': 'bc_z', 'Fractal': 'rt1_fractal'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_headers(path):
    size, rows, digest = path.stat().st_size, [], hashlib.sha256()
    with path.open('rb') as stream:
        while stream.tell() < size:
            offset = stream.tell()
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError('Truncated TFRecord header: ' + str(path))
            length = struct.unpack('<Q', header[:8])[0]
            end = offset + 12 + length + 4
            if end > size:
                raise ValueError('Record exceeds cached file boundary: ' + str(path))
            digest.update(struct.pack('<Q', offset) + header)
            rows.append({'record_index': len(rows), 'payload_offset': offset + 12,
                         'payload_bytes': length})
            stream.seek(end)
        if stream.tell() != size:
            raise ValueError('File boundary does not coincide with TFRecord boundary')
    return rows, digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    inventory_path = ROOT / 'reports/multisource_local_inventory_20261007.json'
    inventory = json.loads(inventory_path.read_text())
    plan_path = ROOT / 'configs/bridge_demonstration_expansion_eligible_20261017.json'
    plan = json.loads(plan_path.read_text())
    protected = {}
    for partition in ('development', 'consumed_test_identity_only',
                      'future_test_candidates_identity_and_metadata_only', 'source_ineligible'):
        for row in plan[partition]:
            protected[(row['shard'], row['record_index'])] = partition
    candidates, excluded, sources = [], [], {}
    for source, cached_subdir in SOURCES.items():
        metadata_path = Path(inventory['sources'][source]['tfds_metadata'][0]['path'])
        metadata = json.loads(metadata_path.read_text())
        train = next(s for s in metadata['splits'] if s['name'] == 'train')
        lengths = list(map(int, train['shardLengths']))
        paths = sorted((ROOT / 'training_cache/oxe_core' / cached_subdir).glob('*.tfrecord*'))
        if not paths:
            raise ValueError('Existing cache missing for ' + source)
        files, count = [], 0
        for path in paths:
            matched = re.fullmatch(re.escape(metadata['name']) + r'-train\.tfrecord-(\d+)-of-(\d+)', path.name)
            if not matched or int(matched[2]) != len(lengths):
                raise ValueError('Cache is not an identified official train shard: ' + str(path))
            shard_index = int(matched[1])
            rows, headers_sha = read_headers(path)
            if len(rows) != lengths[shard_index]:
                raise ValueError('Cached count differs from source metadata: ' + str(path))
            files.append({'path': str(path), 'shard': path.name, 'bytes': path.stat().st_size,
                          'episodes': len(rows), 'header_stream_sha256': headers_sha,
                          'source_shard_index': shard_index})
            for row in rows:
                identity = {'source': source, 'shard': path.name, **row}
                if source == 'Bridge' and (path.name, row['record_index']) in protected:
                    excluded.append({**identity, 'reason': protected[path.name, row['record_index']]})
                else:
                    candidates.append({**identity, 'partition': 'unassigned_candidate'})
                    count += 1
        sources[source] = {'dataset_name': metadata['name'], 'version': metadata['version'],
                           'source_root': inventory['sources'][source]['root'],
                           'metadata_path': str(metadata_path), 'metadata_sha256': sha(metadata_path),
                           'full_train_episodes_from_metadata': sum(lengths),
                           'cached_files': files, 'cached_bytes': sum(p['bytes'] for p in files),
                           'cached_episodes': sum(p['episodes'] for p in files), 'candidate_episodes': count}
        print('OXE_EXISTING_CACHE', source, 'candidate episodes', count, flush=True)
    manifest = {'format': 'oxe-unassigned-episode-candidates-v1', 'training_ready': False,
                'selection_rule': 'All pre-existing core cache official-train shards; exclude known protected Bridge row identities',
                'episode_rows': candidates, 'excluded_rows': excluded,
                'not_done': ['Raw-origin/content alias deduplication', 'Length/terminal/language eligibility',
                             'Numeric source contracts and source frame/time compatibility',
                             'Grouped train/validation split', 'Window starts', 'Full payload hashes/CRC checks']}
    manifest_path = OUT / 'episode_candidates.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    report = {'complete': True, 'scope': 'Existing cached TFRecord headers and source TFDS metadata only',
              'sources': sources, 'total_candidate_episodes': len(candidates),
              'excluded_by_reason': dict(Counter(x['reason'] for x in excluded)),
              'total_cached_bytes': sum(v['cached_bytes'] for v in sources.values()),
              'candidate_manifest_sha256': sha(manifest_path),
              'prior_inventory_sha256': sha(inventory_path), 'protected_plan_sha256': sha(plan_path),
              'payloads_read': False, 'crc_or_numeric_integrity_verified': False,
              'final_targets_read': False, 'new_data_copied_or_downloaded': False,
              'training_updates': 0, 'training_ready': False}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OXE_CANDIDATE_REPORT', OUT / 'report.json')
    print('CANDIDATES', len(candidates), 'EXCLUDED', len(excluded), 'TRAINING_READY', False)


if __name__ == '__main__':
    main()
