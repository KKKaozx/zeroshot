"""Fixed CPU readouts evaluated on existing independent Bridge demonstrations.

Exploratory representation/readout diagnostic, not a replacement diffusion policy.
Uses saved arrays only; never loads neural weights or reserved test data.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bridge_generalization_readout_20261015'
BASE = ROOT / 'reports/bridge-state-fit-202278/real'
AUDIT = ROOT / 'reports/bridge_condition_input_audit_20261013'
VISUAL = ROOT / 'reports/bridge_growth128_generalization_201663_20261013'
RIDGE = 1.0
ARMS = {'state_only': ['xyz', 'rotation'],
        'clip_state': ['clip', 'xyz', 'rotation'],
        'learned_condition_state': ['context', 'xyz', 'rotation']}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotations(rpy):
    r, p, y = rpy.T
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
    return np.stack([cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr], axis=-1)


def readout(features, target_train_cm, tasks, fit, query, blocks):
    """Task intercepts and one fixed ridge solve per task. No query labels."""
    assert max(fit) < 128
    prediction = np.empty((len(query), 48))
    max_solver_delta = 0.
    for task in sorted(set(tasks[query])):
        train = fit[tasks[fit] == task]
        at = np.flatnonzero(tasks[query] == task)
        assert len(train) >= 2
        a, b = [], []
        for block in blocks:
            x = features[block][train]
            mean = x.mean(0)
            scale = max(float(np.sqrt(np.square(x-mean).sum(1).mean())), 1e-12)
            a.append((x-mean)/scale)
            b.append((features[block][query[at]]-mean)/scale)
        a = np.concatenate(a, axis=1)/np.sqrt(len(blocks))
        b = np.concatenate(b, axis=1)/np.sqrt(len(blocks))
        y = target_train_cm[train].reshape(len(train), 48)
        center = y.mean(0)
        coef = np.linalg.solve(a@a.T + RIDGE*np.eye(len(train)), y-center)
        predicted = center + b@a.T@coef
        # Independent SVD expression verifies the closed-form linear solve.
        u, singular, vt = np.linalg.svd(a, full_matrices=False)
        weights = vt.T @ ((singular/(singular**2+RIDGE))[:, None] * (u.T@(y-center)))
        delta = float(np.max(np.abs(predicted-(center+b@weights))))
        max_solver_delta = max(max_solver_delta, delta)
        assert delta < 1e-8
        prediction[at] = predicted
    return prediction.reshape(len(query), 16, 3), max_solver_delta


def task_mean(target_train_cm, tasks, fit, query):
    return np.asarray([target_train_cm[fit[tasks[fit] == tasks[i]]].mean(0) for i in query])


def main():
    started = time.monotonic()
    if OUT.exists():
        # The first attempt stopped at an input-index assertion before any solve.
        assert {p.name for p in OUT.iterdir()} == {'plan.json', 'entry_snapshot.py'}, 'Do not overwrite computed results.'
        (OUT/'input_guard_attempt_plan.json').write_bytes((OUT/'plan.json').read_bytes())
        (OUT/'input_guard_attempt_entry.py').write_bytes((OUT/'entry_snapshot.py').read_bytes())
    manifest_path = ROOT/'cluster_packages/bridge_growth128_v1/manifest.json'
    manifest = read(manifest_path)
    raw = read(AUDIT/'raw_audit.json')
    verified = read(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')
    encoding = read(VISUAL/'encoding.json')
    assert raw['passed'] and verified['verified'] and verified['both_training_fit']
    assert sha(manifest_path) == raw['manifest_sha256'] == encoding['manifest_sha256']
    assert sha(AUDIT/'observed_states.npz') == raw['observed_states_sha256']
    assert sha(AUDIT/'input_windows.csv') == raw['input_windows_sha256']
    assert sha(VISUAL/'frozen_visual_descriptors.npz') == encoding['features_sha256']
    audit_rows = list(csv.DictReader((AUDIT/'input_windows.csv').open(encoding='utf-8-sig')))
    rows = manifest['train'] + manifest['development']
    tasks = np.array([r['task'] for r in rows])
    episodes = np.array([r['shard']+'::'+str(r['record_index']) for r in rows])
    assert len(rows) == 168 and len(set(episodes[:128])) == 64 and len(set(episodes[128:])) == 10
    assert not set(episodes[:128]) & set(episodes[128:])
    for i, (row, audited) in enumerate(zip(rows, audit_rows)):
        assert audited['episode'] == episodes[i] and audited['task'] == row['task']
        assert int(audited['input_index']) == row['start_index'] and int(audited['window']) == i
    paths = [manifest_path, AUDIT/'raw_audit.json', AUDIT/'observed_states.npz',
             AUDIT/'input_windows.csv', VISUAL/'encoding.json', VISUAL/'frozen_visual_descriptors.npz']
    for split in ['train', 'development']:
        p = BASE/f'reduced_u8192_{split}.npz'
        assert sha(p) == verified['artifact_sha256'][f'real/{p.name}']
        paths.append(p)
    # Freeze choices before reading any development prediction/target arrays below.
    OUT.mkdir(exist_ok=True)
    plan = dict(stage='fixed_readout_independent_demonstration_generalization',
        entry_sha256=sha(Path(__file__)), source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        reused_model='202278 real-state final, initialization43, update8192',
        train_windows=128, train_episodes=64, development_windows=40, development_episodes=10,
        arms=ARMS, ridge_lambda=RIDGE, lambda_selected_on_development=False,
        algorithm='Per-task intercept; each feature block centered and normalized to training mean squared row norm1; equally weighted blocks; fixed ridge1.0; target XYZ in centimetres.',
        target='All16 future tool-relative measured XYZ targets, same existing labels; no quaternion/gripper readout.',
        train_leave_episode_out='All windows of each of64 train episodes excluded from readout fit and feature statistics. Supervised context encoder still saw these train labels; its auxiliary CV is not end-to-end validation.',
        baselines=['no_motion', 'training_task_mean', 'existing_final_ddpm100'],
        limits=['CLIP pooled descriptor is not full visual tokens.',
                'Saved learned context includes Adapter, repeated-current/history and real-state projections; cannot uniquely blame Adapter.',
                'Different readout architecture/objective; performance difference does not prove a unique defective module.',
                'Development repeatedly used in prior diagnostics; no untouched final test.',
                'No rotation/gripper/physical success or complete policy evaluation.'],
        stopping='One fixed CPU analysis. No lambda sweep, neural training, new encoding, downloads, checkpoints or simulations.')
    write(OUT/'plan.json', plan)
    (OUT/'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    with np.load(AUDIT/'observed_states.npz') as z:
        current = z['current_state'].astype(float)  # Deliberately do not load future_state.
    with np.load(VISUAL/'frozen_visual_descriptors.npz') as z:
        visual = z['features'].astype(float)
    data = {}
    for split in ['train', 'development']:
        with np.load(BASE/f'reduced_u8192_{split}.npz') as z:
            data[split] = {k:z[k].copy() for k in ['predictions', 'targets', 'contexts', 'row_indices']}
        expected = np.arange(128) if split == 'train' else np.arange(128, 168)
        assert np.array_equal(data[split]['row_indices'], expected)
    contexts = np.concatenate([data[s]['contexts'] for s in ['train', 'development']]).astype(float)
    features = dict(xyz=current[:, :3], rotation=rotations(current[:, 3:6]), clip=visual, context=contexts)
    y = data['train']['targets'][..., :3].astype(float)*10
    all_train, dev = np.arange(128), np.arange(128, 168)
    dev_pred, train_pred, cross_pred = {}, {}, {}
    solver_delta = 0.
    for arm, blocks in ARMS.items():
        both, delta = readout(features, y, tasks, all_train, np.arange(168), blocks)
        train_pred[arm], dev_pred[arm] = both[:128], both[128:]
        solver_delta = max(solver_delta, delta)
        cross_pred[arm] = np.empty_like(y)
    for episode in sorted(set(episodes[:128])):
        query = np.flatnonzero(episodes[:128] == episode)
        fit = np.flatnonzero(episodes[:128] != episode)
        assert not set(episodes[fit]) & set(episodes[query])
        for arm, blocks in ARMS.items():
            result, delta = readout(features, y, tasks, fit, query, blocks)
            cross_pred[arm][query] = result
            solver_delta = max(solver_delta, delta)
    # Target means depend only on fit labels. No development mean is ever used.
    dev_pred['training_task_mean'] = task_mean(y, tasks, all_train, dev)
    train_pred['training_task_mean'] = task_mean(y, tasks, all_train, all_train)
    cross_pred['training_task_mean'] = np.empty_like(y)
    for episode in sorted(set(episodes[:128])):
        query = np.flatnonzero(episodes[:128] == episode)
        fit = np.flatnonzero(episodes[:128] != episode)
        cross_pred['training_task_mean'][query] = task_mean(y, tasks, fit, query)
    # Development labels are accessed only for scoring after all readout solves.
    truth = np.concatenate([y, data['development']['targets'][..., :3].astype(float)*10])
    errors, prediction_payload = {}, {'train_targets_cm': y, 'development_targets_cm': truth[128:]}
    for split, predictions, actual in [('train_resubstitution', train_pred, y),
                                       ('train_episode_excluded_readout', cross_pred, y),
                                       ('development', dev_pred, truth[128:])]:
        errors[split] = {}
        for arm, value in predictions.items():
            assert value.shape == actual.shape and np.isfinite(value).all()
            errors[split][arm] = np.linalg.norm(value-actual, axis=-1)
            prediction_payload[f'{split}__{arm}'] = value
        errors[split]['no_motion'] = np.linalg.norm(actual, axis=-1)
    for split, key in [('train_resubstitution', 'train'), ('development', 'development')]:
        # Average distances over3 existing samples, not distance of their mean.
        policy = data[key]['predictions'][..., :3].astype(float)*10
        actual = y if key == 'train' else truth[128:]
        errors[split]['existing_final_ddpm100'] = np.linalg.norm(policy-actual[None], axis=-1).mean(0)
    assert abs(errors['development']['existing_final_ddpm100'].mean()-verified['development']['real']['path_position_cm']) < 1e-8
    summaries, window_rows, episode_rows, task_rows = {}, [], [], []
    for split, values in errors.items():
        offset = 128 if split == 'development' else 0
        ids = np.arange(offset, offset+len(next(iter(values.values()))))
        summaries[split] = {}
        for arm, e in values.items():
            demo_values = []
            for episode in sorted(set(episodes[ids])):
                at = np.flatnonzero(episodes[ids] == episode)
                val = float(e[at].mean()); demo_values.append(val)
                static = float(values['no_motion'][at].mean())
                episode_rows.append(dict(split=split, arm=arm, episode=episode, task=tasks[ids[at[0]]],
                    windows=len(at), path_position_cm=val, static_cm=static, improvement_cm=static-val))
            summaries[split][arm] = dict(path_position_cm=float(e.mean()), endpoint_position_cm=float(e[:, -1].mean()),
                episode_equal_position_cm=float(np.mean(demo_values)), episodes=len(demo_values),
                episodes_better_than_static=sum(r['path_position_cm'] < r['static_cm'] for r in episode_rows if r['split']==split and r['arm']==arm))
            for task in sorted(set(tasks[ids])):
                at = np.flatnonzero(tasks[ids] == task)
                task_rows.append(dict(split=split, arm=arm, task=task, windows=len(at), path_position_cm=float(e[at].mean()), static_cm=float(values['no_motion'][at].mean())))
            for at, i in enumerate(ids):
                window_rows.append(dict(split=split, arm=arm, window=int(i), episode=episodes[i], task=tasks[i],
                    path_position_cm=float(e[at].mean()), endpoint_position_cm=float(e[at, -1])))
    np.savez_compressed(OUT/'predictions.npz', **prediction_payload)
    table(OUT/'per_window.csv', window_rows); table(OUT/'per_episode.csv', episode_rows); table(OUT/'per_task.csv', task_rows)
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest
    report = dict(complete=True, source_identity_checks_passed=True, identical128_40_split=True,
        train_only_readout_statistics=True, no_same_episode_in_readout_cv_fit=True,
        fixed_lambda=RIDGE, max_dual_svd_prediction_difference_cm=solver_delta,
        summaries=summaries, auxiliary_linear_readouts_solved_on_training_labels=True,
        new_neural_training=False, new_policy_inference=False, new_clip_encoding=False,
        reserved_test_used=False, physical_rollout=False, development_used_for_scoring_only=True,
        context_cv_is_not_end_to_end_validation=True, elapsed_seconds=time.monotonic()-started)
    write(OUT/'report.json', report)
    print(json.dumps({'development':summaries['development'], 'elapsed_seconds':report['elapsed_seconds'], 'solver_delta_cm':solver_delta}, indent=2))


if __name__ == '__main__':
    main()
