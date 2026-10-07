"""Train-only, horizon-specific mean pose control; no learned model is loaded."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np


def mean_pose(poses):
    """Arithmetic xyz and sign-invariant Markley quaternion mean per horizon."""
    result = np.empty(poses.shape[1:], dtype=np.float32)
    result[..., :3] = poses[..., :3].mean(axis=0)
    q = poses[..., 3:7].astype(np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms <= 1e-6):
        raise ValueError('Degenerate target quaternion')
    q = q / norms
    for h in range(q.shape[1]):
        _, vectors = np.linalg.eigh(q[:, h].T @ q[:, h])
        result[h, 3:7] = vectors[:, -1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['USE_TF'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    sys.path.insert(0, str(args.pack.resolve()))
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import torch
    torch.set_num_threads(4)
    from train import bridge_plan_selection, bridge_plan_splits
    from dataset import UnifiedRobotDataset
    from evaluate_multitask_pilot import metrics
    from probe_bridge_conditioning import select_windows, sha256
    reference = json.loads(args.reference.read_text())
    selected = bridge_plan_selection(args.pack / 'manifest.json')
    dataset = UnifiedRobotDataset(data_dir=str(args.pack / 'data'), chunk_size=16, stride=4,
        sources=['tfrecord'], min_trajectory_steps=17, exclude_path_parts=[], exclude_schemas=[],
        tfrecord_splits=['train'], bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous', bridge_episode_selection=selected)
    splits = bridge_plan_splits(dataset)
    expected = json.loads((args.pack / 'manifest.json').read_text())['expected_windows']
    assert {k: len(v) for k, v in splits.items()} == expected
    rows = select_windows(dataset, splits, selected, 8, all_train=True)
    assert rows == reference['window_selection']
    assert {name: sha256(args.pack/name) for name in reference['module_sha256']} == reference['module_sha256']
    targets = []
    for row in rows:
        item = dataset[row['dataset_index']]
        assert torch.all(item[4] == 1)
        targets.append(item[3].numpy()[..., :7])
    targets = np.stack(targets)
    assert np.isfinite(targets).all()
    train = [i for i, row in enumerate(rows) if row['partition'] == 'train']
    tasks = sorted({row['task'] for row in rows})
    fitted = {task: mean_pose(targets[[i for i in train if rows[i]['task'] == task]]) for task in tasks}
    global_mean = mean_pose(targets[train])
    groups = {}
    for part in ('train', 'validation'):
        groups[part+'/overall'] = [i for i,r in enumerate(rows) if r['partition'] == part]
        for task in tasks:
            groups[part+'/task/'+task] = [i for i,r in enumerate(rows) if r['partition'] == part and r['task'] == task]
    results = {}
    for name, prediction in [('task_mean', np.stack([fitted[r['task']] for r in rows])),
                             ('global_mean', np.broadcast_to(global_mean, targets.shape))]:
        results[name] = {}
        for group, indices in groups.items():
            # Gripper is outside this pose-only diagnostic.
            p, t = prediction[indices], targets[indices]
            extra = np.zeros(t.shape[:-1]+(1,), np.float32)
            values = metrics(np.concatenate([p,extra],-1), np.concatenate([t,extra],-1))
            assert abs(values['static_position_cm']-reference['static_baselines'][group]['position_cm']) < 1e-4
            assert abs(values['static_rotation_deg']-reference['static_baselines'][group]['rotation_deg']) < 1e-4
            results[name][group] = {k: values[k] for k in ('windows','action_targets','position_cm','rotation_deg')}
    report = dict(stage='train_only_mean_pose_control', model_loaded=False, trained=False,
        reserved_test_targets_read=False, reference_sha256=sha256(args.reference),
        module_sha256=reference['module_sha256'], counts=expected,
        mean_policy='Train windows only, per horizon; arithmetic xyz; sign-invariant Markley quaternion mean',
        train_scores_are_in_sample=True, development_used_for_fitting=False,
        task_means={k:v.tolist() for k,v in fitted.items()}, baselines=results,
        one_step_model={g:{k:sum(x[k] for x in values)/len(values) for k in ('position_cm','rotation_deg')}
            for g,values in reference['one_step_pure_noise'].items()},
        limits=['No image, phase, current state or test target enters the mean control.',
            'Mean uses overlapping training windows with equal weights; train evaluation is descriptive in-sample.',
            'Markley quaternion mean is a rotation mean, not an optimizer of the angular absolute-error metric.',
            'Aggregate errors cannot establish equal predictions or prove the learned model ignores images.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{g:v for g,v in values.items() if g.endswith('/overall')} for k,values in results.items()},indent=2),flush=True)
    print('TRAIN-ONLY MEAN POSE CONTROL: PASSED', args.output, flush=True)


if __name__ == '__main__':
    main()
