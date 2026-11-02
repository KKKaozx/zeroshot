"""Evaluate saved trajectories with path, horizon-endpoint and event metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

POSITION_CM = 10.0


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def pose_errors(prediction, target):
    position = np.linalg.norm(prediction[..., :3] - target[..., :3], axis=-1) * POSITION_CM
    pq = prediction[..., 3:7]
    tq = target[..., 3:7]
    pq = pq / np.maximum(np.linalg.norm(pq, axis=-1, keepdims=True), 1e-8)
    tq = tq / np.maximum(np.linalg.norm(tq, axis=-1, keepdims=True), 1e-8)
    cosine = np.clip(np.abs(np.sum(pq * tq, axis=-1)), 0, 1)
    rotation = 2 * np.arccos(cosine) * 180 / np.pi
    return position, rotation


def scalar_metrics(prediction, target):
    position, rotation = pose_errors(prediction, target)
    true_open = target[..., 7] > 0
    predicted_open = prediction[..., 7] > 0
    result = dict(
        windows=len(target), action_targets=int(np.prod(target.shape[:2])),
        path_position_cm=float(position.mean()), path_rotation_deg=float(rotation.mean()),
        endpoint_position_cm=float(position[:, -1].mean()),
        endpoint_rotation_deg=float(rotation[:, -1].mean()),
        endpoint_gripper_accuracy=float((predicted_open[:, -1] == true_open[:, -1]).mean()),
    )
    for name, before, after in [('grasp', True, False), ('release', False, True)]:
        event = (true_open[:, :-1] == before) & (true_open[:, 1:] == after)
        predicted_event = ((predicted_open[:, :-1] == before)
                           & (predicted_open[:, 1:] == after))
        count = int(event.sum())
        result[name + '_events'] = count
        result[name + '_exact_timing_accuracy'] = float(predicted_event[event].mean()) if count else None
        result[name + '_position_cm_at_true_event'] = float(position[:, 1:][event].mean()) if count else None
        result[name + '_rotation_deg_at_true_event'] = float(rotation[:, 1:][event].mean()) if count else None
    return result


def oracle_minimum(predictions, target):
    """Ground-truth oracle; each scalar may select a different draw."""
    positions, rotations = zip(*(pose_errors(draw, target) for draw in predictions))
    position = np.stack(positions)
    rotation = np.stack(rotations)
    return dict(
        draws=len(predictions),
        path_position_cm=float(position.mean(axis=-1).min(axis=0).mean()),
        path_rotation_deg=float(rotation.mean(axis=-1).min(axis=0).mean()),
        endpoint_position_cm=float(position[..., -1].min(axis=0).mean()),
        endpoint_rotation_deg=float(rotation[..., -1].min(axis=0).mean()),
        selection='Ground-truth oracle independently minimizes each scalar per window; unavailable at deployment without a selector.',
    )


def markley_mean(poses):
    result = np.empty(poses.shape[1:], np.float32)
    result[..., :3] = poses[..., :3].mean(axis=0)
    q = poses[..., 3:7].astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    for step in range(q.shape[1]):
        _, vectors = np.linalg.eigh(q[:, step].T @ q[:, step])
        result[step, 3:7] = vectors[:, -1]
    result[..., 7] = np.where(poses[..., 7].mean(axis=0) >= 0, 1, -1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--training-report', type=Path, required=True)
    parser.add_argument('--window-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.training_report.read_text())
    reference = json.loads(args.window_reference.read_text())
    rows = reference['window_selection']
    assert report['passed'] and report['phase'] == 'train' and not report['reserved_test_targets_read']
    assert report['protocol']['window_sha256'] == hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode()).hexdigest()
    arrays = {}
    targets = None
    sources = {}
    for head in ('regression', 'diffusion'):
        path = args.run / (head + '-predictions.npz')
        with np.load(path) as saved:
            prediction = saved['predictions'].astype(np.float32)
            current_targets = saved['targets'].astype(np.float32)
        assert prediction.ndim == 4 and prediction.shape[1:] == (len(rows), 16, 8)
        assert np.isfinite(prediction).all() and np.isfinite(current_targets).all()
        assert len(prediction) == len(report['groups'][head]['sampling_seeds'])
        if targets is None:
            targets = current_targets
        else:
            assert np.array_equal(targets, current_targets)
        arrays[head] = prediction
        sources[head] = dict(path=str(path), sha256=sha256(path), shape=list(prediction.shape))
    groups = {}
    for part in ('train', 'validation'):
        groups[part + '/overall'] = [i for i, row in enumerate(rows) if row['partition'] == part]
        for task in sorted({row['task'] for row in rows if row['partition'] == part}):
            groups[part + '/task/' + task] = [i for i, row in enumerate(rows)
                if row['partition'] == part and row['task'] == task]
        for shard, record in sorted({(row['shard'], row['record_index']) for row in rows if row['partition'] == part}):
            groups[part + '/episode/' + shard + '::' + str(record)] = [i for i, row in enumerate(rows)
                if row['partition'] == part and (row['shard'], row['record_index']) == (shard, record)]
    train = groups['train/overall']
    tasks = sorted({row['task'] for row in rows})
    task_means = {task: markley_mean(targets[[i for i in train if rows[i]['task'] == task]]) for task in tasks}
    task_mean_prediction = np.stack([task_means[row['task']] for row in rows])
    static = np.zeros_like(targets)
    static[..., 6] = 1
    static[..., 7] = 1
    output = dict(stage='saved_prediction_path_endpoint_event_evaluation', trained=False,
        weights_loaded=False, reserved_test_targets_read=False,
        training_report_sha256=sha256(args.training_report),
        window_reference_sha256=sha256(args.window_reference), sources=sources,
        definitions=dict(path='Mean over all 16 targets (ADE-like).',
            endpoint='Target index 15 relative to the input observation tool frame; not necessarily the task endpoint.',
            event='Pose error at target t+1 where ground truth gripper changes within the 16-target window.',
            oracle='Best of saved diffusion draws after seeing ground truth; coverage diagnostic only.'),
        groups={})
    for group, indices in groups.items():
        target = targets[indices]
        values = dict(static=scalar_metrics(static[indices], target),
            train_task_mean=scalar_metrics(task_mean_prediction[indices], target))
        for head, prediction in arrays.items():
            draws = prediction[:, indices]
            values[head] = dict(per_draw=[scalar_metrics(draw, target) for draw in draws])
            if len(draws) > 1:
                values[head]['oracle_min_at_k'] = oracle_minimum(draws, target)
        output['groups'][group] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for group in ('train/overall', 'validation/overall'):
        print(group, json.dumps(output['groups'][group], ensure_ascii=False), flush=True)
    print('SAVED TRAJECTORY ENDPOINT AND EVENT EVALUATION: PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
