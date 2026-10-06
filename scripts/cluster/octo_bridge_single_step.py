"""Frozen Octo diagnostic on sparse Bridge pairs; no training or robot control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

IDENTITY = '4b91633128d732dc054e13f2d672877d2a904a1ccf4f1dd545571af5b40a1ce8'
SEEDS = (0, 1, 2)
BATCH = 8


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(root):
    """Read original sparse pairs without interpreting their record order as time."""
    import tensorflow as tf
    manifest = json.loads((root / 'export_manifest.json').read_text())
    check = json.loads((root / 'read_check.json').read_text())
    if manifest['dataset_identity'] != IDENTITY or manifest['test_targets_used'] or not check['passed']:
        raise ValueError('Unexpected subset identity or read-check')
    if check['test_targets_used'] or set(manifest['sample_order']) != {'train', 'validation'}:
        raise ValueError('Unexpected partitions')
    data = {}
    for part, directory, expected, episode_count in [('train', 'train', 316, 17), ('validation', 'val', 55, 3)]:
        relative = f'data/sweep_into_pile/{directory}/out.tfrecord'
        path = root / relative
        metadata = manifest['files'][relative]
        if path.stat().st_size != metadata['bytes'] or sha256(path) != metadata['sha256']:
            raise ValueError('Subset file changed')
        episodes = [e for e in manifest['episodes'] if e['partition'] == part]
        if len(episodes) != episode_count:
            raise ValueError('Episode count differs')
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        records = tf.data.TFRecordDataset(str(path)).with_options(options).as_numpy_iterator()
        groups = []
        for number, raw in enumerate(records):
            if number >= len(episodes):
                raise ValueError('Extra episode record')
            episode = episodes[number]
            example = tf.train.Example.FromString(raw)
            fields = example.features.feature
            def tensor(key, dtype):
                return tf.io.parse_tensor(fields[key].bytes_list.value[0], dtype).numpy()
            state = tensor('observations/state', tf.float32)
            nxt = tensor('next_observations/state', tf.float32)
            command = tensor('actions', tf.float32)
            image = tensor('observations/images0', tf.uint8)
            language = tensor('language', tf.string)[0].decode('utf-8')
            count = len(episode['indices'])
            if state.shape != (count, 7) or nxt.shape != state.shape or command.shape != state.shape:
                raise ValueError('State/action shapes differ')
            if image.shape != (count, 256, 256, 3) or language != episode['instruction']:
                raise ValueError('Image/language contract differs')
            if not all(np.isfinite(x).all() for x in (state, nxt, command)):
                raise ValueError('Nonfinite data')
            if not np.isin(command[:, 6], [0., 1.]).all():
                raise ValueError('Exported gripper must already be binary')
            target = np.concatenate([nxt[:, :6] - state[:, :6], command[:, 6:7]], axis=1)
            groups.append(dict(image=image, state=state, next_state=nxt, target=target,
                language=np.repeat(language, count), indices=np.array(episode['indices']),
                starts=np.array(episode['starts']),
                episode=np.repeat(f"{episode['shard']}::{episode['record_index']}", count)))
        if len(groups) != episode_count:
            raise ValueError('Missing episode record')
        data[part] = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
        if len(data[part]['indices']) != expected or data[part]['indices'].tolist() != manifest['sample_order'][part]:
            raise ValueError('Window coverage/order differs')
        if len(set(data[part]['indices'])) != expected:
            raise ValueError('Duplicate indices')
    if set(data['train']['indices']) & set(data['validation']['indices']):
        raise ValueError('Partition overlap')
    return data, manifest


def metrics(prediction, subset):
    state, nxt, target = (subset[k] for k in ('state', 'next_state', 'target'))
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError('Prediction contract failed')
    position = np.linalg.norm(state[:, :3] + prediction[:, :3] - nxt[:, :3], axis=1) * 100
    def quaternion(eulers):
        roll, pitch, yaw = (eulers / 2).T
        cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
        sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
        return np.stack([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy], axis=1)
    predicted_q = quaternion(state[:, 3:6] + prediction[:, 3:6])
    reached_q = quaternion(nxt[:, 3:6])
    vector = predicted_q[:, 3:4]*reached_q[:, :3] - reached_q[:, 3:4]*predicted_q[:, :3] - np.cross(predicted_q[:, :3], reached_q[:, :3])
    scalar = np.sum(predicted_q * reached_q, axis=1)
    rotation = np.rad2deg(2*np.arctan2(np.linalg.norm(vector, axis=1), np.abs(scalar)))
    opened, predicted_open = target[:, 6] >= .5, prediction[:, 6] >= .5
    tp, tn = int(np.sum(opened & predicted_open)), int(np.sum(~opened & ~predicted_open))
    positives, negatives = int(opened.sum()), int((~opened).sum())
    open_recall = tp / positives if positives else None
    closed_recall = tn / negatives if negatives else None
    return dict(windows=len(target), position_cm=float(position.mean()), rotation_deg=float(rotation.mean()),
        gripper_accuracy=float(np.mean(opened == predicted_open)), open_recall=open_recall,
        closed_recall=closed_recall,
        balanced_accuracy=(open_recall + closed_recall) / 2 if positives and negatives else None,
        confusion=dict(true_open_pred_open=tp, true_open_pred_closed=positives-tp,
            true_closed_pred_open=negatives-tn, true_closed_pred_closed=tn),
        true_open_count=positives, predicted_open_count=int(predicted_open.sum()))


def summarize(predictions, subset):
    results = [metrics(p, subset) for p in predictions]
    mean = {key: float(np.mean([r[key] for r in results])) if results[0][key] is not None else None
        for key in ('position_cm', 'rotation_deg', 'gripper_accuracy', 'balanced_accuracy')}
    return dict(per_sampling_seed={str(s): r for s, r in zip(SEEDS, results)}, mean_metrics=mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, trained=False, robot_controlled=False, test_targets_used=False,
        unseen_data_generalization=False, sampling_seeds=list(SEEDS), batch_size=BATCH,
        evaluated_action_index=0, observation_history=1, sparse_pairs=True,
        limitations=['Bridge may overlap Octo pretraining; no unseen-data claim.',
            'Octo bridge_dataset uses its own updated Bridge release; this subset is older OXE Bridge V2, not an identical benchmark distribution.',
            'Sparse pairs do not support four-step targets or gripper switching metrics.',
            'Different complete policies, pretraining and inputs; not an Adapter/head ablation.',
            'Export gripper was binarized before sparse selection by the earlier author-loader audit; full raw scan is not rerun here.',
            'Offline reached-state errors do not prove controller execution or robotic success.'])
    try:
        plan = json.loads(args.plan.read_text())
        commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != plan['code_commit']:
            raise ValueError('Official code commit differs')
        subprocess.run(['git', '-C', str(args.source), 'diff', '--exit-code', 'HEAD', '--', 'octo'], check=True)
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        import jax
        if jax.default_backend() != 'gpu':
            raise RuntimeError('JAX fell back to CPU')
        from octo.model.octo_model import OctoModel
        from octo.data.utils.data_utils import relabel_actions
        data, manifest = read_pairs(args.data)
        for part, subset in data.items():
            # Call the unchanged official relabeler separately on each source pair.
            for state, nxt, target in zip(subset['state'], subset['next_state'], subset['target']):
                official = relabel_actions(dict(observation={'state': tf.constant(np.stack([state, nxt]))},
                    action=tf.constant(np.stack([target, target]))))['action'].numpy()[0]
                np.testing.assert_array_equal(official, target)
            print(f'CONTRACT {part}: {len(subset["target"])} real pairs verified', flush=True)
        assets = json.loads((args.assets / 'assets.json').read_text())
        if Path(assets['model_path']).name != plan['model_revision'] or assets['t5_revision'] != plan['t5_revision']:
            raise ValueError('Asset revisions differ')
        model = OctoModel.load_pretrained(assets['model_path'])
        stats = model.dataset_statistics['bridge_dataset']['action']
        np.testing.assert_array_equal(stats['mask'], [True]*6 + [False])
        report.update(source_commit=commit, model_revision=plan['model_revision'],
            dataset_identity=manifest['dataset_identity'], source_manifest_sha256=sha256(args.data/'export_manifest.json'),
            backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
            action_statistics={k: np.asarray(v).tolist() for k, v in stats.items()}, partitions={})
        for partition_number, (part, subset) in enumerate(data.items()):
            start_time = time.monotonic()
            n = len(subset['target'])
            predictions = np.empty((len(SEEDS), n, 7), np.float32)
            for start in range(0, n, BATCH):
                stop = min(start + BATCH, n)
                indices = np.minimum(np.arange(start, start+BATCH), n-1)
                observation = dict(image_primary=subset['image'][indices, None],
                    timestep_pad_mask=np.ones((BATCH, 1), bool))
                tasks = model.create_tasks(texts=subset['language'][indices].tolist())
                for j, seed in enumerate(SEEDS):
                    key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), partition_number), start)
                    actions = np.asarray(jax.device_get(model.sample_actions(observation, tasks,
                        unnormalization_statistics=stats, rng=key)))
                    if actions.shape != (BATCH, 4, 7) or not np.isfinite(actions).all():
                        raise ValueError('Native action chunk check failed')
                    predictions[j, start:stop] = actions[:stop-start, 0]
                print(f'INFERENCE {part}: {stop}/{n}', flush=True)
            np.savez_compressed(args.output/f'{part}-first-actions.npz', predictions=predictions,
                targets=subset['target'], state=subset['state'], next_state=subset['next_state'],
                indices=subset['indices'], starts=subset['starts'], episode=subset['episode'])
            baseline = metrics(np.zeros_like(subset['target']), subset)
            episodes = {}
            for episode in np.unique(subset['episode']):
                selected = subset['episode'] == episode
                group = {k: v[selected] for k, v in subset.items()}
                episodes[episode] = dict(octo=summarize(predictions[:, selected], group),
                    zero_motion_always_closed=metrics(np.zeros_like(group['target']), group))
            report['partitions'][part] = dict(octo=summarize(predictions, subset),
                zero_motion_always_closed=baseline, per_episode=episodes,
                evaluated_windows=n, elapsed_seconds=time.monotonic()-start_time)
            print(f'RESULT {part}: '+json.dumps(report['partitions'][part]['octo']['mean_metrics']), flush=True)
        report['passed'] = True
        print('OCTO BRIDGE SINGLE STEP EVALUATION: PASSED (not task success)', flush=True)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('Report:', args.output/'report.json', flush=True)


if __name__ == '__main__':
    main()
