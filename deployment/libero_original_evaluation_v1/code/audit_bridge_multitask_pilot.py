"""Audit only pilot train/development records; never load reserved-test targets."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
from PIL import Image
import torch
from dataset import UnifiedRobotDataset, POSITION_SCALE_METERS


def rotation_euler(v):
    x, y, z = np.asarray(v, dtype=np.float64)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_quaternion(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new report path')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    selected = []
    for part in ('train', 'development'):
        selected.extend({**row, 'partition': 'train' if part == 'train' else 'validation'}
            for row in plan['split_episodes'][part])
    reserved = {(e['shard'], e['record_index']) for e in plan['split_episodes']['reserved_test']}
    assert not reserved & {(e['shard'], e['record_index']) for e in selected}
    report = dict(passed=False, purpose='pilot_train_development_source_and_production_loader_audit',
        trained=False, model_loaded=False, reserved_test_targets_read=False,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(), episodes=[],
        dataset_source_sha256=hashlib.sha256(Path(__file__).with_name('dataset.py').read_bytes()).hexdigest(),
        limitations=['No physical controller calibration or benchmark execution.',
            'Same-task episode holdout does not establish scene/task independence.',
            'Output range mismatches are recorded, not silently clipped.'])
    started = time.monotonic()
    torch.set_num_threads(2)
    try:
        dataset = UnifiedRobotDataset(data_dir=str(args.data), chunk_size=16, stride=4,
            sources=['tfrecord'], min_trajectory_steps=17, exclude_path_parts=[], exclude_schemas=[],
            tfrecord_splits=['train'], bridge_gripper_policy='reverse_scan_valid_steps_v2',
            bridge_current_gripper='continuous', bridge_episode_selection=selected)
        groups = defaultdict(list)
        for i, sample in enumerate(dataset.samples):
            groups[(Path(sample['file_path']).name, sample['record_index'])].append(i)
        if set(groups) != {(e['shard'], e['record_index']) for e in selected}:
            raise ValueError('Missing or extra episode windows')
        for row in selected:
            key = row['shard'], row['record_index']
            path = args.data / row['shard']
            example = dataset._load_tfrecord_example(str(path), row['record_index'])
            fields = example.features.feature
            n = row['steps']
            state = np.array(fields['steps/observation/state'].float_list.value, np.float32).reshape(n, 7)
            command = np.array(fields['steps/action'].float_list.value, np.float32).reshape(n, 7)
            first = np.asarray(fields['steps/is_first'].int64_list.value)
            last = np.asarray(fields['steps/is_last'].int64_list.value)
            terminal = np.asarray(fields['steps/is_terminal'].int64_list.value)
            if not (np.isfinite(state).all() and np.isfinite(command).all()):
                raise ValueError(f'Nonfinite source: {key}')
            if (first.shape != (n,) or last.shape != (n,) or terminal.shape != (n,)
                    or np.flatnonzero(first).tolist() != [0] or np.flatnonzero(last).tolist() != [n-1]
                    or not np.isin(first, [0, 1]).all() or not np.isin(last, [0, 1]).all()
                    or not np.isin(terminal, [0, 1]).all()):
                raise ValueError(f'RLDS boundaries differ: {key}')
            images = fields['steps/observation/image_0'].bytes_list.value
            if len(images) != n:
                raise ValueError(f'Image count differs: {key}')
            for encoded in images:
                with Image.open(io.BytesIO(encoded)) as image:
                    image.load()
                    if image.size != (256, 256) or image.mode != 'RGB':
                        raise ValueError(f'RGB image differs: {key}')
            # Independent reverse scan over valid commands, excluding the last dummy action.
            binary = np.empty(n-1, np.float32)
            carry = command[n-2, 6]
            for t in range(n-2, -1, -1):
                if command[t, 6] > .95: carry = 1.
                elif command[t, 6] < .05: carry = 0.
                binary[t] = carry
            if not np.isin(binary, [0., 1.]).all():
                raise ValueError(f'Unresolved gripper tail: {key}')
            position_error = matrix_error = 0.
            outside = opened = 0
            starts = []
            for index in groups[key]:
                start = dataset.samples[index]['start_index']
                if start+16 >= n:
                    raise ValueError('Padded or cross-episode target')
                language, image, current, target, mask = dataset[index]
                action = target.numpy()
                # Inventory groups instructions using the loader's normalized metadata key.
                # The actual encoder input keeps the original spelling and capitalization.
                if (' '.join(language.lower().split()) != row['instruction'] or tuple(image.shape) != (3, 224, 224)
                        or action.shape != (16, 8) or not torch.isfinite(image).all()
                        or not np.isfinite(action).all() or not torch.all(mask == 1)):
                    raise ValueError(f'Production loader contract failed: {key}')
                np.testing.assert_allclose(current.numpy(), [2*state[start, 6]-1], atol=1e-6)
                reference = rotation_euler(state[start, 3:6])
                for j in range(16):
                    reached = start+j+1
                    reconstructed = state[start, :3] + reference @ (action[j, :3]*POSITION_SCALE_METERS)
                    position_error = max(position_error, float(np.max(np.abs(reconstructed-state[reached, :3]))))
                    actual = reference @ rotation_quaternion(action[j, 3:7])
                    matrix_error = max(matrix_error, float(np.max(np.abs(actual-rotation_euler(state[reached, 3:6])))))
                    if action[j, 7] != 2*binary[reached-1]-1:
                        raise ValueError(f'Gripper command time differs: {key}')
                outside += int(np.any(np.abs(action[:, :3]) > 3, axis=1).sum())
                opened += int((action[:, 7] > 0).sum())
                starts.append(start)
            if position_error > 1e-5 or matrix_error > 1e-5:
                raise ValueError(f'Independent pose reconstruction failed: {key}')
            report['episodes'].append(dict(shard=row['shard'], record_index=row['record_index'],
                partition=row['partition'], instruction=row['instruction'], steps=n, windows=len(starts),
                starts=starts, decoded_rgb_frames=n, position_max_abs_error_m=position_error,
                rotation_matrix_max_abs_error=matrix_error, out_of_30cm_target_views=outside,
                open_target_views=opened,
                payload_sha256=hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()))
            print('VERIFIED', row['partition'], row['instruction'], key, len(starts), flush=True)
        report['partitions'] = {p: dict(episodes=sum(e['partition']==p for e in report['episodes']),
            windows=sum(e['windows'] for e in report['episodes'] if e['partition']==p),
            decoded_rgb_frames=sum(e['steps'] for e in report['episodes'] if e['partition']==p),
            out_of_30cm_target_views=sum(e['out_of_30cm_target_views'] for e in report['episodes'] if e['partition']==p))
            for p in ('train', 'validation')}
        report['passed'] = True
        print(json.dumps(report['partitions']), flush=True)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        print('Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
