"""Read-only Bridge metadata census. Skips image/numeric feature payloads by seek.

Uses the official protobuf Feature parser for the three retained fields, and
checks the selective reader against full official Example parsing on six records.
No image decoding, model imports, inference, exports or dataset writes.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.core.example import example_pb2, feature_pb2

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('E:/dataset/bridge_v2_0.0.1/0.0.1')
OUT = ROOT/'reports/bridge_additional_inventory_20261017'
FIELDS = {'steps/language_instruction', 'episode_metadata/file_path', 'episode_metadata/episode_id'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def varint(stream, end):
    value = 0
    for shift in range(0, 70, 7):
        assert stream.tell() < end, 'Truncated varint'
        b = stream.read(1)[0]
        value |= (b & 127) << shift
        if b < 128:
            return value
    raise ValueError('Overlong varint')


def fields(stream, end):
    while stream.tell() < end:
        tag = varint(stream, end)
        number, wire = tag >> 3, tag & 7
        assert number > 0
        if wire == 2:
            length = varint(stream, end)
            start = stream.tell()
            stop = start+length
            assert stop <= end
            yield number, start, stop
            stream.seek(stop)
        elif wire == 0:
            varint(stream, end)
        elif wire in (1, 5):
            assert stream.tell()+(8 if wire == 1 else 4) <= end
            stream.seek(8 if wire == 1 else 4, 1)
        else:
            raise ValueError(('Unsupported protobuf wire type', wire))
    assert stream.tell() == end


def selective_metadata(stream, start, end):
    values = {}
    stream.seek(start)
    for outer, features_start, features_end in fields(stream, end):
        if outer != 1:
            continue
        stream.seek(features_start)
        for entry, entry_start, entry_end in fields(stream, features_end):
            if entry != 1:
                continue
            stream.seek(entry_start)
            key, span = None, None
            for field, first, last in fields(stream, entry_end):
                if field == 1:
                    stream.seek(first)
                    key = stream.read(last-first).decode('utf-8')
                elif field == 2:
                    span = first, last
            if key in FIELDS:
                assert key not in values and span is not None
                first, last = span
                stream.seek(first)
                values[key] = feature_pb2.Feature.FromString(stream.read(last-first))
            stream.seek(entry_end)
        stream.seek(features_end)
    assert set(values) == FIELDS
    return values


def normalized_metadata(values):
    instructions = list(values['steps/language_instruction'].bytes_list.value)
    origin = list(values['episode_metadata/file_path'].bytes_list.value)
    eid = list(values['episode_metadata/episode_id'].int64_list.value)
    assert len(origin) == len(eid) == 1
    task_strings = sorted({' '.join(x.decode('utf-8').lower().split()) for x in instructions})
    return dict(steps=len(instructions), instructions=task_strings,
                origin_file_path=origin[0].decode('utf-8'), episode_id=int(eid[0]))


def records(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        index = 0
        while stream.tell() < size:
            header = stream.read(12)
            assert len(header) == 12
            length = struct.unpack('<Q', header[:8])[0]
            start = stream.tell()
            end = start+length
            assert length > 0 and end+4 <= size
            yield index, stream, start, end
            stream.seek(end+4)
            index += 1
        assert stream.tell() == size


def validate_reader():
    comparisons = []
    for shard in (0, 256, 1023):
        path = DATA/f'bridge_data_v2-train.tfrecord-{shard:05d}-of-01024'
        for index, stream, first, last in records(path):
            small = normalized_metadata(selective_metadata(stream, first, last))
            stream.seek(first)
            full = example_pb2.Example.FromString(stream.read(last-first))
            official = normalized_metadata({k: full.features.feature[k] for k in FIELDS})
            assert small == official
            comparisons.append(dict(shard=path.name, record_index=index, metadata_equal=True))
            if index == 1:
                break
    return comparisons


def main():
    started = time.monotonic()
    OUT.mkdir(exist_ok=True)
    (OUT/'shards').mkdir(exist_ok=True)
    plan = read(ROOT/'configs/bridge_expansion_eligible_plan_v1.json')
    tasks = set(plan['tasks'])
    census = {}
    sources = []

    # Reuse prior metadata scans; do not read the same256 shards again.
    p = ROOT/'results/bridge_multitask_pilot_plan_v1/bridge_multitask_original_inventory.json'
    old = read(p)
    sources.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    for shard in range(32):
        name = f'bridge_data_v2-train.tfrecord-{shard:05d}-of-01024'
        selected = [e for e in old['episodes'] if e['shard'] == name and e['instruction'] in tasks]
        census[name] = dict(episodes=selected, metadata_source='prior_first32_inventory')
    p = ROOT/'results/bridge_expansion_metadata_32_63.json'
    old = read(p)
    sources.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    for shard in range(32,64):
        name = f'bridge_data_v2-train.tfrecord-{shard:05d}-of-01024'
        census[name] = dict(episodes=[e for e in old['episodes'] if e['shard'] == name], metadata_source='prior32_63_inventory')
    p = ROOT/'results/bridge_expansion_metadata_64_255.json'
    old = read(p)
    sources.append(dict(path=str(p.relative_to(ROOT)), sha256=sha(p)))
    assert len(old) == 192
    census.update({k: dict(v,metadata_source='prior64_255_inventory') for k,v in old.items()})
    assert len(census) == 256
    validation = validate_reader()
    for shard in range(256,1024):
        name = f'bridge_data_v2-train.tfrecord-{shard:05d}-of-01024'
        path = DATA/name
        cache = OUT/'shards'/(name+'.json')
        stat = path.stat()
        binding = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if cache.exists():
            result = read(cache)
            assert result['source_stat'] == binding
        else:
            result = dict(source_stat=binding, scanned_episodes=0, episodes=[], metadata_source='selective_official_feature_parse')
            for index, stream, first, last in records(path):
                m = normalized_metadata(selective_metadata(stream, first, last))
                result['scanned_episodes'] += 1
                if m['steps'] >= 17 and len(m['instructions']) == 1 and m['instructions'][0] in tasks:
                    result['episodes'].append(dict(shard=name,record_index=index,steps=m['steps'],
                        instruction=m['instructions'][0],origin_file_path=m['origin_file_path'],episode_id=m['episode_id']))
            write(cache,result)
        census[name] = result
        if (shard-255)%32 == 0:
            print('METADATA',shard+1,'/1024','new_candidates',sum(len(v['episodes']) for k,v in census.items() if int(k.split('-')[-3])>=256),
                  'seconds',round(time.monotonic()-started,1),flush=True)

    by_record = {(e['shard'],e['record_index']):e for v in census.values() for e in v['episodes']}
    assert len(by_record) == sum(len(v['episodes']) for v in census.values())
    # Include all former selected rows, including the seven known numeric rejects,
    # in exclusions. None may silently re-enter a new training candidate pool.
    original = read(ROOT/'configs/bridge_expansion_plan_v1.json')
    excluded_rows = [e for v in original['split_episodes'].values() for e in v]
    excluded_records = {(e['shard'],e['record_index']) for e in excluded_rows}
    excluded_origins = {(e['origin_file_path'],e['episode_id']) for e in excluded_rows}
    new, duplicates, blocked = [], [], []
    seen = set(excluded_origins)
    for key,e in sorted(by_record.items()):
        origin = e['origin_file_path'],e['episode_id']
        if key in excluded_records or origin in excluded_origins:
            blocked.append(e)
        elif origin in seen:
            duplicates.append(e)
        else:
            seen.add(origin)
            new.append(e)
    def task_counts(rows):
        return {task:sum(e['instruction']==task for e in rows) for task in sorted(tasks)}
    report = dict(date='2026-10-17',purpose='metadata_only_additional_same5task_demonstrations',
        training=False,new_inference=False,gpu_used=False,images_decoded=False,numeric_action_values_inspected=False,
        full_record_bytes_read_for_six_parser_validation_cases=True,
        metadata_reader_verified_against_official_example=validation,
        crc_payload_verification=False,numeric_and_production_reader_eligibility_verified=False,
        existing_train_count=405,existing_development_count=10,consumed_test_count=13,
        original_rejected_or_ineligible_count=7,
        existing_train_by_task=task_counts(plan['split_episodes']['train']),
        all_train_shards_present=len(list(DATA.glob('bridge_data_v2-train.tfrecord-*-of-01024'))),
        reused_metadata_shards=256,newly_scanned_metadata_shards=768,
        scanned_new_episodes=sum(v['scanned_episodes'] for k,v in census.items() if v['metadata_source']=='selective_official_feature_parse'),
        all_matching_metadata_candidates=len(by_record),all_matching_counts_by_task=task_counts(list(by_record.values())),
        excluded_selected_records=len(blocked),duplicate_new_origins=len(duplicates),
        additional_unique_candidates=len(new),additional_counts_by_task=task_counts(new),
        additional_windows_from_lengths=sum(len(range(0,e['steps']-16,4)) for e in new),
        eligibility_note='Candidate means same normalized instruction, length>=17 and unique source-path/episode-id outside original selected435 rows. Not proof of valid numeric targets, image validity, new physical scene, or absence of every historical unrecorded use.',
        candidate_order='lexicographic shard and record; no model score or numeric target selection',
        consumed_holdout_208056_excluded=True,cache_sources=sources,
        original_eligible_plan_sha256=sha(ROOT/'configs/bridge_expansion_eligible_plan_v1.json'),
        original_selected_plan_sha256=sha(ROOT/'configs/bridge_expansion_plan_v1.json'),
        candidates=new,duplicate_rows=duplicates,elapsed_seconds=time.monotonic()-started)
    write(OUT/'inventory.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidates','duplicate_rows','cache_sources')},indent=2),flush=True)


if __name__=='__main__':
    main()
