"""Euler-wrap subset variant using the unmodified author LCBC agent; original labels retained."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def metrics(prediction, target, state, next_state, mean, std):
    prediction = np.asarray(prediction, dtype=np.float64)
    assert prediction.shape == target.shape and prediction.shape[1] == 7
    assert np.isfinite(prediction).all()
    positions = state[:, :3] + prediction[:, :3]
    def quaternion(eulers):
        roll, pitch, yaw = (eulers / 2).T
        cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
        sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
        return np.stack([sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                         cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy], axis=1)
    predicted_q = quaternion(state[:, 3:6] + prediction[:, 3:6])
    true_q = quaternion(next_state[:, 3:6])
    relative_vector = (predicted_q[:, 3:4]*true_q[:, :3] -
                       true_q[:, 3:4]*predicted_q[:, :3] -
                       np.cross(predicted_q[:, :3], true_q[:, :3]))
    relative_scalar = np.sum(predicted_q*true_q, axis=1)
    angle = np.rad2deg(2*np.arctan2(np.linalg.norm(relative_vector, axis=1), np.abs(relative_scalar)))
    opened = target[:, 6] >= 0.5
    predicted_open = prediction[:, 6] >= 0.5
    tp = int(np.sum(opened & predicted_open))
    tn = int(np.sum(~opened & ~predicted_open))
    positives = int(opened.sum())
    negatives = int((~opened).sum())
    open_recall = tp / positives if positives else None
    close_recall = tn / negatives if negatives else None
    return dict(windows=len(target), position_cm=float(np.linalg.norm(positions-next_state[:, :3], axis=1).mean()*100),
        rotation_deg=float(angle.mean()), gripper_accuracy=float((opened == predicted_open).mean()),
        gripper_balanced_accuracy=(open_recall+close_recall)/2 if positives and negatives else None,
        open_recall=open_recall, close_recall=close_recall,
        gripper_confusion=dict(true_open_pred_open=tp, true_open_pred_closed=positives-tp,
                              true_closed_pred_closed=tn, true_closed_pred_open=negatives-tn),
        normalized_mse_sum=float(np.mean(np.sum(((prediction-target)/std)**2, axis=1))))


class CyclingSampler:
    def __init__(self, count, seed):
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(count)
        self.cursor = 0
        self.frequency = np.zeros(count, dtype=np.int64)

    def take(self, size):
        chunks = []
        needed = size
        while needed:
            n = min(needed, self.count-self.cursor)
            chunks.append(self.order[self.cursor:self.cursor+n])
            self.cursor += n
            needed -= n
            if self.cursor == self.count:
                self.order = self.rng.permutation(self.count)
                self.cursor = 0
        indices = np.concatenate(chunks)
        np.add.at(self.frequency, indices, 1)
        return indices


def wrap_rotation_targets(data, original_stats):
    """Canonicalize out-of-range Euler component differences, preserving physical targets."""
    original_mean = np.asarray(original_stats['action']['mean'], np.float32)
    original_std = np.asarray(original_stats['action']['std'], np.float32)
    checks = {}
    for partition, subset in data.items():
        raw = subset['target'].copy()
        wrapped = raw.copy()
        rotation = raw[:, 3:6].astype(np.float64)
        outside = np.abs(rotation) > np.pi
        canonical = rotation.copy()
        canonical[outside] = (rotation[outside] + np.pi) % (2*np.pi) - np.pi
        wrapped[:, 3:6] = canonical.astype(raw.dtype)
        np.testing.assert_array_equal(wrapped[:, [0, 1, 2, 6]], raw[:, [0, 1, 2, 6]])
        assert np.max(np.abs(wrapped[:, 3:6])) <= np.pi + 1e-7
        rows = np.flatnonzero(np.any(outside, axis=1))
        errors = []
        for i in rows:
            state = subset['state'][i:i+1]
            original_next = state.astype(np.float64).copy()
            original_next[:, :6] += raw[i:i+1, :6]
            error = metrics(wrapped[i:i+1], raw[i:i+1], state, original_next,
                            original_mean, original_std)['rotation_deg']
            assert error < 1e-4, 'Canonicalization changed the target physical pose.'
            errors.append(error)
        subset['target_raw'] = raw
        subset['target'] = wrapped
        checks[partition] = dict(windows=len(raw), changed_windows=int(len(rows)),
            changed_indices=subset['indices'][rows].tolist(),
            max_physical_rotation_difference_deg=max(errors, default=0.0),
            position_and_gripper_identical=True)
    assert checks['train']['changed_indices'] == [363]
    assert checks['validation']['changed_indices'] == []
    # All seven statistics are derived from the same 316 training windows.
    mean = data['train']['target'].mean(axis=0)
    std = data['train']['target'].std(axis=0)
    assert np.all(std > 0) and np.isfinite(mean).all() and np.isfinite(std).all()
    np.testing.assert_array_equal(mean[[0, 1, 2, 6]], original_mean[[0, 1, 2, 6]])
    np.testing.assert_array_equal(std[[0, 1, 2, 6]], original_std[[0, 1, 2, 6]])
    stats = dict(action=dict(mean=mean.tolist(), std=std.tolist()),
                 proprio=original_stats['proprio'],
                 computed_from='316 selected training windows only; Euler-wrap variant')
    return mean, std, stats, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--language-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.steps == 1000 and args.batch_size == 16, 'This v1 experiment uses the declared fixed schedule.'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    import jax
    from flax import serialization
    from jaxrl_m.data.bridge_dataset import BridgeDataset
    from jaxrl_m.data.text_processing import MULTI_MODULE
    from jaxrl_m.agents.continuous.lc_bc import LCBCAgent
    from jaxrl_m.vision import encoders
    from experiments.configs.train_config import get_config

    gpu = jax.devices('gpu')[0]
    assert gpu.platform == 'gpu'
    print('JAX GPU:', gpu, flush=True)
    root = args.data_dir
    manifest = json.loads((root/'export_manifest.json').read_text())
    stats = json.loads((root/'train_action_stats.json').read_text())
    assert manifest['test_targets_used'] is False
    commit = 'bc60a35b701a12021c8c95e9d8601274d3acd928'
    assert manifest['official_commit'] == commit
    mean = np.asarray(stats['action']['mean'], np.float32)
    std = np.asarray(stats['action']['std'], np.float32)
    assert mean.shape == std.shape == (7,) and np.all(std > 0)
    with np.load(args.language_cache, allow_pickle=False) as cache:
        assert cache['dataset_identity'].item() == manifest['dataset_identity']
        assert cache['source_url'].item() == MULTI_MODULE
        languages = cache['languages'].tolist()
        embedding = cache['embeddings'].copy()
    assert len(languages) == 1 and embedding.shape == (1, 512) and np.isfinite(embedding).all()
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    episode_map = {}
    for episode in manifest['episodes']:
        key = f"{episode['shard']}::{episode['record_index']}"
        for index, start in zip(episode['indices'], episode['starts']):
            assert index not in episode_map
            episode_map[index] = (key, start, episode['partition'])
    data = {}
    for partition, directory, expected in [('train', 'train', 316), ('validation', 'val', 55)]:
        relative = f'data/sweep_into_pile/{directory}/out.tfrecord'
        path = root/relative
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        assert digest.hexdigest() == manifest['files'][relative]['sha256']
        reader = BridgeDataset([str(path)], seed=args.seed, batch_size=1, train=False,
            augment=False, load_language=True, skip_unlabeled=True, relabel_actions=True,
            action_proprio_metadata=None, goal_relabeling_strategy='uniform',
            goal_relabeling_kwargs={'reached_proportion': 0.0})
        images, states, next_states, targets = [], [], [], []
        for batch in reader.tf_dataset.with_options(options).as_numpy_iterator():
            assert batch['goals']['language'][0].decode('utf-8') == languages[0]
            images.append(batch['observations']['image'][0])
            states.append(batch['observations']['proprio'][0])
            next_states.append(batch['next_observations']['proprio'][0])
            targets.append(batch['actions'][0])
        indices = np.asarray(manifest['sample_order'][partition], dtype=np.int64)
        assert len(images) == expected == len(indices)
        assert all(episode_map[int(i)][2] == partition for i in indices)
        data[partition] = dict(image=np.stack(images), state=np.stack(states), next_state=np.stack(next_states),
                              target=np.stack(targets), indices=indices,
                              episode_keys=np.array([episode_map[int(i)][0] for i in indices]),
                              starts=np.array([episode_map[int(i)][1] for i in indices]))
        np.testing.assert_allclose(data[partition]['target'][:, :6], data[partition]['next_state'][:, :6]-data[partition]['state'][:, :6])
        assert set(np.unique(data[partition]['target'][:, 6])) <= {0., 1.}
        print(f'Loaded {partition}: {expected} fixed windows', flush=True)
    assert len(np.unique(data['validation']['episode_keys'])) == 3
    np.testing.assert_allclose(data['train']['target'].mean(0), mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(data['train']['target'].std(0), std, rtol=1e-5, atol=1e-6)

    original_stats = stats
    mean, std, stats, wrap_checks = wrap_rotation_targets(data, original_stats)
    write_json(args.output_dir/'rotation_label_check.json', wrap_checks)
    write_json(args.output_dir/'train_action_stats.json', stats)
    for p, check in wrap_checks.items():
        print(f"Euler wrap {p}: changed={check['changed_indices']} max_pose_difference={check['max_physical_rotation_difference_deg']:.8f}deg", flush=True)
    print('Wrapped training yaw std:', float(std[5]), flush=True)

    original_config = get_config('lc_bc').to_dict()
    model_config = get_config('lc_bc').to_dict()
    model_config['agent_kwargs']['warmup_steps'] = 50
    model_config['agent_kwargs']['decay_steps'] = args.steps
    settings = dict(purpose='controlled_single_step_author_lcbc_euler_wrap_subset', official_commit=commit,
        dataset_identity=manifest['dataset_identity'], steps=args.steps, batch_size=args.batch_size, seed=args.seed,
        learning_rate=3e-4, warmup_steps=50, eval_interval=100, augmentation=False,
        label_variant='Euler component differences wrapped to [-pi, pi]',
        original_action_stats=original_stats, action_stats=stats, rotation_label_checks=wrap_checks,
        base_runner_sha256='4ab5372931b6afe845d099607af3616cb73a8f887a75cb7f16f6238cc1565c2d',
        original_author_config=original_config, agent_kwargs=model_config['agent_kwargs'],
        encoder=model_config['encoder'], encoder_kwargs=model_config['encoder_kwargs'],
        sampling='shuffle without replacement, retain epoch tails across batches',
        checkpoint_selection='minimum validation normalized_mse_sum, excluding untrained step 0',
        no_motion_baseline='zero xyz/Euler delta and gripper always closed',
        test_targets_used=False, transition_metrics_supported=False,
        limitations=['One task instruction; cannot establish zero-shot language generalization.',
                     'Sparse first-step windows; no continuous gripper transition or closed-loop success metric.',
                     'Euler-wrap is an explicit label variant; normalized losses across raw/wrapped statistics are not directly comparable.',
                     'Different architecture from the project model; this is not a module ablation or full paper reproduction.'])
    write_json(args.output_dir/'experiment.json', settings)

    def batch_for(indices):
        train = data['train']
        return dict(observations={'image': train['image'][indices]},
                    goals={'language': np.repeat(embedding, len(indices), axis=0)},
                    actions=(train['target'][indices]-mean)/std)

    example = batch_for(np.arange(args.batch_size))
    agent = LCBCAgent.create(rng=jax.random.PRNGKey(args.seed), observations=example['observations'],
        goals=example['goals'], actions=example['actions'],
        encoder_def=encoders[model_config['encoder']](**model_config['encoder_kwargs']),
        **model_config['agent_kwargs'])
    print('Author LCBC agent created; first update will compile.', flush=True)

    def prediction_for(current_agent, subset):
        predictions = []
        for start in range(0, len(subset['image']), args.batch_size):
            images = subset['image'][start:start+args.batch_size]
            real_count = len(images)
            if real_count < args.batch_size:
                images = np.concatenate([images, np.repeat(images[-1:], args.batch_size-real_count, axis=0)])
            normalized = np.asarray(current_agent.sample_actions(
                {'image': images}, {'language': np.repeat(embedding, args.batch_size, axis=0)},
                seed=jax.random.PRNGKey(args.seed), argmax=True))[:real_count]
            predictions.append(normalized*std+mean)
        result = np.concatenate(predictions)
        assert len(result) == len(subset['indices'])
        return result

    def summarize(prediction, subset):
        total = metrics(prediction, subset['target'], subset['state'], subset['next_state'], mean, std)
        total['by_episode'] = {}
        for key in np.unique(subset['episode_keys']):
            mask = subset['episode_keys'] == key
            total['by_episode'][str(key)] = metrics(prediction[mask], subset['target'][mask],
                subset['state'][mask], subset['next_state'][mask], mean, std)
        return total

    def save_predictions(name, predictions):
        with (args.output_dir/f'{name}_predictions.npz').open('wb') as stream:
            payload = {}
            for partition, prediction in predictions.items():
                payload[f'{partition}_prediction_native7'] = prediction
                for field in ['state', 'next_state', 'target', 'target_raw', 'indices', 'episode_keys', 'starts']:
                    payload[f'{partition}_{field}'] = data[partition][field]
            np.savez_compressed(stream, **payload)

    baselines = {partition: summarize(np.zeros_like(subset['target']), subset) for partition, subset in data.items()}
    history = []
    best_mse = float('inf')
    best_step = None
    def evaluate(step):
        nonlocal best_mse, best_step
        predictions = {p: prediction_for(agent, subset) for p, subset in data.items()}
        report = dict(step=step, partitions={p: summarize(predictions[p], data[p]) for p in data})
        history.append(report)
        write_json(args.output_dir/'history.json', dict(baselines=baselines, evaluations=history))
        for p in data:
            m = report['partitions'][p]
            balanced = m['gripper_balanced_accuracy']
            print(f"step={step} {p} n={m['windows']} position={m['position_cm']:.3f}cm rotation={m['rotation_deg']:.3f}deg gripper={m['gripper_accuracy']:.3f} balanced={balanced}", flush=True)
        criterion = report['partitions']['validation']['normalized_mse_sum']
        if step > 0 and criterion < best_mse:
            best_mse, best_step = criterion, step
            weights = args.output_dir/'best_params.msgpack'
            temporary = weights.with_suffix('.part')
            temporary.write_bytes(serialization.to_bytes(agent.state.params))
            temporary.replace(weights)
            write_json(args.output_dir/'best_metrics.json', report)
            save_predictions('best', predictions)
        if step == args.steps:
            (args.output_dir/'final_params.msgpack').write_bytes(serialization.to_bytes(agent.state.params))
            save_predictions('final', predictions)
            write_json(args.output_dir/'final_metrics.json', report)

    evaluate(0)
    sampler = CyclingSampler(len(data['train']['image']), args.seed)
    started = time.monotonic()
    for step in range(1, args.steps+1):
        agent, info = agent.update(batch_for(sampler.take(args.batch_size)))
        if step == 1 or step % 50 == 0:
            loss = float(info['actor_loss'])
            assert np.isfinite(loss)
            print(f'update={step}/{args.steps} actor_loss={loss:.6f} lr={float(info["lr"]):.8f} elapsed={time.monotonic()-started:.1f}s', flush=True)
        if step % 100 == 0:
            evaluate(step)
    assert sampler.frequency.min() > 0
    write_json(args.output_dir/'training_coverage.json', dict(windows=316, draws=int(sampler.frequency.sum()),
        min_draws=int(sampler.frequency.min()), max_draws=int(sampler.frequency.max()),
        indices=data['train']['indices'].tolist(), draws_by_window=sampler.frequency.tolist()))
    write_json(args.output_dir/'summary.json', dict(completed_updates=args.steps, best_step=best_step,
        best_validation_normalized_mse_sum=best_mse, elapsed_seconds=time.monotonic()-started,
        covered_train_windows=316, covered_validation_windows=55, validation_episodes=3,
        test_targets_used=False, results_dir=str(args.output_dir)))
    print('CONTROLLED EULER-WRAP LCBC RUN COMPLETED; evaluate metrics against baselines, not training loss.', flush=True)
    print('Results:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
