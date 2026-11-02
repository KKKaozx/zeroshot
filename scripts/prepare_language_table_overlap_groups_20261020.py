"""Group the existing real Language Table candidate pool by shared JPEG bytes.

No image decoding, model, partition assignment, training or benchmark access.
TFRecordDataset verifies record CRCs. Exact encoded-frame matches are a lower
bound on alias detection, not proof that different groups are independent.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/language_table_overlap_20261020'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def overlap_groups(rows):
    """Conservative connected components; labels do not affect grouping."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    shared = set()
    for i, row in enumerate(rows):
        for frame in set(row['encoded_frame_sha256']):
            if frame in owners:
                other = owners[frame]
                parents[find(i)] = find(other)
                shared.add(frame)
            else:
                owners[frame] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row['episode_key'])
    output = []
    for members in groups.values():
        members.sort()
        output.append({'group_id': 'lt-shared-jpeg-' + digest('\n'.join(members).encode()),
                       'episode_keys': members})
    output.sort(key=lambda g: g['episode_keys'][0])
    return output, len(owners), len(shared)


def check_grouping():
    fixture = [{'episode_key': 'a', 'encoded_frame_sha256': ['x', 'x', 'y']},
               {'episode_key': 'b', 'encoded_frame_sha256': ['y', 'z']},
               {'episode_key': 'c', 'encoded_frame_sha256': ['z', 'u']},
               {'episode_key': 'd', 'encoded_frame_sha256': ['separate']}]
    result, unique, shared = overlap_groups(fixture)
    assert [g['episode_keys'] for g in result] == [['a', 'b', 'c'], ['d']]
    assert unique == 5 and shared == 2
    assert overlap_groups(list(reversed(fixture)))[0] == result


def main():
    started = time.perf_counter()
    check_grouping()
    OUT.mkdir(exist_ok=False)
    manifest_path = ROOT / 'reports/oxe_baseline_candidates_20261020/episode_candidates.json'
    inventory_path = ROOT / 'reports/oxe_baseline_candidates_20261020/report.json'
    manifest = json.loads(manifest_path.read_text())
    inventory = json.loads(inventory_path.read_text())['sources']['Language Table']
    assert inventory['dataset_name'] == 'language_table'
    metadata_path = Path(inventory['metadata_path'])
    assert digest(metadata_path.read_bytes()) == inventory['metadata_sha256']
    metadata = json.loads(metadata_path.read_text())
    assert metadata['name'] == 'language_table'
    expected = {(r['shard'], r['record_index']): r for r in manifest['episode_rows']
                if r['source'] == 'Language Table'}
    assert all(r['partition'] == 'unassigned_candidate' for r in expected.values())
    rows, errors, file_checks = [], [], []
    action_phase = {offset: {'squared_error': 0., 'max_abs': 0., 'components': 0}
                    for offset in (-1, 0, 1)}
    first_hash_before = None
    for file in inventory['cached_files']:
        path = Path(file['path'])
        assert path.stat().st_size == file['bytes']
        offset, count = 12, 0
        payload_hash = hashlib.sha256()
        for index, raw in enumerate(tf.data.TFRecordDataset([str(path)])):
            payload = bytes(raw.numpy())
            candidate = expected[path.name, index]
            assert candidate['payload_offset'] == offset and candidate['payload_bytes'] == len(payload)
            payload_hash.update(len(payload).to_bytes(8, 'little') + payload)
            example = tf.train.Example()
            example.ParseFromString(payload)
            if first_hash_before is None:
                prior = json.loads((ROOT / 'reports/core_source_fixed_record_checks.json').read_text())
                first_hash_before = digest(example.SerializeToString(deterministic=True))
                assert first_hash_before == prior['sources']['language_table']['payload_sha256']
            f = example.features.feature
            images = f['steps/observation/rgb'].bytes_list.value
            n = len(images)
            row = {'episode_key': f'{path.name}::{index}', 'shard': path.name,
                   'record_index': index, 'payload_sha256': digest(payload), 'steps': n,
                   'partition': 'unassigned_candidate', 'errors': []}
            row['encoded_frame_sha256'] = [digest(image) for image in images]
            for name in ('effector_translation', 'effector_target_translation'):
                vals = np.array(f['steps/observation/' + name].float_list.value, dtype=np.float32)
                if vals.size != n * 2 or not np.isfinite(vals).all():
                    row['errors'].append(name + '_shape_or_nonfinite')
            flags = {}
            for name in ('is_first', 'is_last', 'is_terminal'):
                flags[name] = list(f['steps/' + name].int64_list.value)
                if len(flags[name]) != n or not set(flags[name]).issubset({0, 1}):
                    row['errors'].append(name + '_shape_or_nonbinary')
            if n:
                if flags['is_first'] != [1] + [0] * (n - 1): row['errors'].append('first_boundary')
                if flags['is_last'] != [0] * (n - 1) + [1]: row['errors'].append('last_boundary')
                if any(flags['is_terminal'][:-1]): row['errors'].append('internal_terminal')
            else:
                row['errors'].append('no_frames')
            tokens = np.array(f['steps/observation/instruction'].int64_list.value)
            texts = []
            if tokens.size != n * 512 or np.any((tokens < 0) | (tokens > 255)):
                row['errors'].append('instruction_shape_or_bytes')
            else:
                for token_row in tokens.reshape(n, 512):
                    data = bytes(token_row.astype(np.uint8)).split(b'\0', 1)[0]
                    try:
                        texts.append(data.decode('utf-8').strip())
                    except UnicodeDecodeError:
                        texts.append('')
                        row['errors'].append('instruction_invalid_utf8')
            row['instruction_sha256'] = sorted({digest(t.encode()) for t in texts if t})
            # Candidate starts only, not admitted training windows or a split.
            starts = [i for i in range(0, max(0, n - 16), 4)
                      if len(texts) == n and texts[i]] if n >= 17 else []
            row['full16_candidate_starts_stride4'] = starts
            row['has_valid_full16_candidate'] = bool(starts) and not row['errors']
            raw_action = np.array(f['steps/action'].float_list.value, dtype=np.float32)
            if raw_action.size != n * 2 or not np.isfinite(raw_action[:-2]).all():
                row['errors'].append('action_shape_or_nonfinite_nonterminal')
            elif n > 1 and not row['errors']:
                target = np.array(f['steps/observation/effector_target_translation'].float_list.value,
                                  dtype=np.float32).reshape(n, 2).astype(np.float64)
                actions = raw_action.reshape(n, 2).astype(np.float64)
                delta = np.diff(target, axis=0)
                # Diagnose index phase; never fit a transform or repair labels.
                for lag, stats in action_phase.items():
                    indices = np.arange(n - 1)
                    keep = (indices + lag >= 0) & (indices + lag < n - 1)
                    err = actions[indices[keep] + lag] - delta[indices[keep]]
                    if err.size:
                        stats['squared_error'] += float(np.square(err).sum())
                        stats['max_abs'] = max(stats['max_abs'], float(np.abs(err).max()))
                        stats['components'] += err.size
            row['has_valid_full16_candidate'] = bool(starts) and not row['errors']
            if row['errors']: errors.append({'episode_key': row['episode_key'], 'errors': row['errors']})
            rows.append(row)
            offset += len(payload) + 16
            count += 1
        assert count == file['episodes'] and offset - 12 == path.stat().st_size
        file_checks.append({'shard': path.name, 'records': count,
                            'length_delimited_payload_sha256': payload_hash.hexdigest(),
                            'tfrecord_reader_crc_checks_completed': True})
        print('LT_GROUP_INPUT', path.name, count, 'total', len(rows), flush=True)
    assert len(rows) == len(expected)
    groups, unique_frames, shared_frames = overlap_groups(rows)
    by_key = {r['episode_key']: r for r in rows}
    mapping = {}
    for group in groups:
        for key in group['episode_keys']: mapping[key] = group['group_id']
        group['episodes'] = len(group['episode_keys'])
        group['distinct_instruction_hashes'] = len({h for key in group['episode_keys']
                                                   for h in by_key[key]['instruction_sha256']})
    owners = {}
    for row in rows:
        row['group_id'] = mapping[row['episode_key']]
        for h in row['encoded_frame_sha256']:
            assert owners.setdefault(h, row['group_id']) == row['group_id']
    for stats in action_phase.values():
        stats['rmse_native_units'] = (stats.pop('squared_error') / stats['components']) ** .5
    dataset = {'format': 'language-table-exact-jpeg-overlap-groups-v1',
               'grouping_is_proof_of_independent_origins': False,
               'source': 'language_table_real_0.1.0_existing_train_cache',
               'training_ready': False, 'groups': groups, 'episode_rows': rows}
    group_path = OUT / 'episode_groups.json'
    group_path.write_text(json.dumps(dataset, indent=2) + '\n')
    multi = [g for g in groups if g['episodes'] > 1]
    report = {'complete': True, 'all_record_structures_valid': not errors,
              'candidate_manifest_sha256': digest(manifest_path.read_bytes()),
              'source_metadata_sha256': inventory['metadata_sha256'],
              'script_sha256': digest(Path(__file__).read_bytes()),
              'group_manifest_sha256': digest(group_path.read_bytes()),
              'records': len(rows), 'frame_occurrences': sum(r['steps'] for r in rows),
              'unique_encoded_frames': unique_frames, 'shared_encoded_frames': shared_frames,
              'groups': len(groups), 'multi_episode_groups': len(multi),
              'episodes_in_multi_episode_groups': sum(g['episodes'] for g in multi),
              'groups_with_multiple_language_labels': sum(g['distinct_instruction_hashes'] > 1 for g in multi),
              'largest_group_episodes': max(g['episodes'] for g in groups),
              'group_size_histogram': dict(sorted(Counter(g['episodes'] for g in groups).items())),
              'records_with_full16_candidate': sum(r['has_valid_full16_candidate'] for r in rows),
              'full16_candidate_windows_stride4': sum(len(r['full16_candidate_starts_stride4'])
                                                    for r in rows if not r['errors']),
              'input_gripper_features': [0, 0], 'supervised_dimensions': ['source_x', 'source_y'],
              'gripper_supervision': False, 'rotation_supervision': False, 'height_supervision': False,
              'setpoint_phase_diagnostic_by_action_row_offset': action_phase,
              'paper_collection_frequency_hz': 5, 'per_record_timestamps_available': False,
              'real_tool_frame_calibration_obtained': False, 'main_mixed_training_admitted': False,
              'grouping_fixture_transitive_and_order_invariance_passed': True,
              'shared_encoded_frame_never_crosses_groups': True,
              'partitions_assigned': False, 'images_decoded': 0,
              'models_loaded': 0, 'training_updates': 0, 'final_benchmark_access': False,
              'file_checks': file_checks, 'errors': errors,
              'limitations': ['Only eight existing train shards, not full Language Table',
                              'Different JPEG encodings of the same image are not detected',
                              'No original collection/session IDs exist in the released local schema',
                              'A static identical frame may conservatively connect separate episodes',
                              'Components must be recomputed when extending the pool; new edges can merge groups',
                              'Do not split these groups yet: no complete physical-origin independence guarantee'],
              'elapsed_seconds': time.perf_counter() - started}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('file_checks', 'limitations', 'errors')}, indent=2))


if __name__ == '__main__':
    main()
