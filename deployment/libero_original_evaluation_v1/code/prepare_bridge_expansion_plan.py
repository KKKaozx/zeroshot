"""Select additional Bridge training episodes from metadata; preserve holdouts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def identity(row):
    return row['origin_file_path'], row['episode_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--base-plan', type=Path, required=True)
    parser.add_argument('--prior-candidates', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=32)
    parser.add_argument('--stop', type=int, default=256)
    parser.add_argument('--per-task', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new plan path; existing selections are immutable')
    if not 0 <= args.start < args.stop <= 1024 or args.per_task < 1:
        raise ValueError('Invalid scan bounds or training cap')
    base = json.loads(args.base_plan.read_text(encoding='utf-8'))
    tasks = base['tasks']
    cache = json.loads(args.cache.read_text(encoding='utf-8')) if args.cache.exists() else {}
    import tensorflow as tf
    for shard_index in range(args.start, args.stop):
        name = f'bridge_data_v2-train.tfrecord-{shard_index:05d}-of-01024'
        if name in cache:
            continue
        rows, count = [], 0
        for record_index, raw in enumerate(tf.data.TFRecordDataset(str(args.data / name))):
            count += 1
            example = tf.train.Example()
            example.ParseFromString(bytes(raw.numpy()))
            fields = example.features.feature
            instructions = fields['steps/language_instruction'].bytes_list.value
            if len(instructions) < 17:
                continue
            normalized = {' '.join(t.decode('utf-8').lower().split()) for t in instructions}
            if len(normalized) != 1 or next(iter(normalized)) not in tasks:
                continue
            origin = fields['episode_metadata/file_path'].bytes_list.value
            eid = fields['episode_metadata/episode_id'].int64_list.value
            if len(origin) != 1 or len(eid) != 1:
                raise ValueError(f'Missing unique episode origin: {name}:{record_index}')
            rows.append(dict(shard=name, record_index=record_index, steps=len(instructions),
                instruction=next(iter(normalized)), origin_file_path=origin[0].decode('utf-8'),
                episode_id=int(eid[0])))
        cache[name] = dict(scanned_episodes=count, episodes=rows)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.cache.with_suffix('.partial')
        temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.cache)
        if (shard_index - args.start + 1) % 8 == 0:
            print('SCANNED', shard_index + 1, 'TASK CANDIDATES',
                dict(Counter(r['instruction'] for v in cache.values() for r in v['episodes'])), flush=True)
    train = list(base['split_episodes']['train'])
    blocked = {identity(r) for rows in base['split_episodes'].values() for r in rows}
    counts = Counter(r['instruction'] for r in train)
    candidates = (json.loads(args.prior_candidates.read_text(encoding='utf-8'))['episodes']
        if args.prior_candidates else [])
    candidates += [r for i in range(args.start, args.stop)
        for r in cache[f'bridge_data_v2-train.tfrecord-{i:05d}-of-01024']['episodes']]
    duplicate_count = 0
    for row in sorted(candidates, key=lambda r: (r['shard'], r['record_index'])):
        if identity(row) in blocked:
            duplicate_count += 1
            continue
        blocked.add(identity(row))
        if counts[row['instruction']] < args.per_task:
            train.append(row)
            counts[row['instruction']] += 1
    plan = dict(base)
    plan.update(status='metadata_selected_pending_numeric_audit',
        purpose='Bridge-only training expansion with unchanged pilot holdouts',
        base_plan_sha256=hashlib.sha256(args.base_plan.read_bytes()).hexdigest(),
        split_episodes={**base['split_episodes'], 'train': train},
        counts=dict(train=len(train), development=len(base['split_episodes']['development']),
            reserved_test=len(base['split_episodes']['reserved_test'])),
        counts_by_task={task: {part: sum(r['instruction'] == task for r in rows)
            for part, rows in {**base['split_episodes'], 'train': train}.items()} for task in tasks},
        expansion_selection=dict(scan_start=args.start, scan_stop_exclusive=args.stop,
            prior_candidates_sha256=(hashlib.sha256(args.prior_candidates.read_bytes()).hexdigest()
                if args.prior_candidates else None),
            maximum_training_episodes_per_task=args.per_task, candidate_count=len(candidates),
            duplicate_origins_excluded=duplicate_count,
            policy='Preserve original train; add unique origins in shard/record order up to per-task cap; no numerical target or model-error selection',
            metadata_only=True, numeric_eligibility_verified=False, new_training_started=False),
        not_done=['Numeric/RGB/production-loader audit of selected train/development',
            'Verified subset export', 'Training and comparison on fixed development set'])
    plan['limits'] = ['Same five instruction strings; no new dataset or task.',
        'Fixed episode holdouts do not establish scene or unseen-task independence.',
        'Bounded shard scan may leave some tasks below cap.',
        'Metadata selection does not guarantee valid numeric/image targets.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('PLAN', args.output, plan['counts'], plan['counts_by_task'], flush=True)


if __name__ == '__main__':
    main()
