"""Single training episode replay: no held-out evaluation or checkpoint selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from run_bridge_lcbc_wrap import CyclingSampler, metrics, write_json

EPISODE = 'bridge_data_v2-train.tfrecord-00003-of-01024::38'
COMMIT = 'bc60a35b701a12021c8c95e9d8601274d3acd928'


def prepare_targets(raw):
    target = raw.copy()
    rotation = raw[:, 3:6].astype(np.float64)
    outside = np.abs(rotation) > np.pi
    rotation[outside] = (rotation[outside] + np.pi) % (2*np.pi) - np.pi
    target[:, 3:6] = rotation.astype(raw.dtype)
    np.testing.assert_array_equal(target, raw)  # Selected episode has no branch crossing.
    assert target.shape == (21, 7) and np.isfinite(target).all()
    assert int((target[:, 6] == 1).sum()) == 3
    assert int((target[:, 6] == 0).sum()) == 18
    mean, std = target.mean(0), target.std(0)
    assert np.all(std > 1e-8)
    return target, mean, std


def decide(final, baseline):
    checks = dict(
        position_reduction_at_least_50_percent=final['position_cm'] <= baseline['position_cm']*.5,
        rotation_reduction_at_least_50_percent=final['rotation_deg'] <= baseline['rotation_deg']*.5,
        open_recall_at_least_95_percent=final['open_recall'] is not None and final['open_recall'] >= .95,
        close_recall_at_least_95_percent=final['close_recall'] is not None and final['close_recall'] >= .95)
    return dict(passed=all(checks.values()), checks=checks,
        thresholds=dict(position_cm_max=baseline['position_cm']*.5,
                        rotation_deg_max=baseline['rotation_deg']*.5,
                        open_recall_min=.95, close_recall_min=.95),
        scope='Same 21 training windows; fitting diagnostic, not generalization or robot success.',
        checkpoint_step=1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--language-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
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
    assert jax.devices('gpu')[0].platform == 'gpu'
    print('JAX GPU:', jax.devices('gpu')[0], flush=True)
    manifest = json.loads((args.data_dir/'export_manifest.json').read_text())
    assert manifest['official_commit'] == COMMIT and manifest['test_targets_used'] is False
    episodes = [e for e in manifest['episodes'] if
                f"{e['shard']}::{e['record_index']}" == EPISODE]
    assert len(episodes) == 1 and episodes[0]['partition'] == 'train'
    selected = episodes[0]
    assert len(selected['indices']) == len(selected['starts']) == 21
    selected_indices = set(selected['indices'])
    order = manifest['sample_order']['train']
    assert len(order) == len(set(order)) == 316 and selected_indices <= set(order)
    relative = 'data/sweep_into_pile/train/out.tfrecord'
    path = args.data_dir/relative
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    assert digest.hexdigest() == manifest['files'][relative]['sha256']
    with np.load(args.language_cache, allow_pickle=False) as cache:
        assert cache['dataset_identity'].item() == manifest['dataset_identity']
        assert cache['source_url'].item() == MULTI_MODULE
        languages = cache['languages'].tolist()
        embedding = cache['embeddings'].copy()
    assert len(languages) == 1 and embedding.shape == (1, 512) and np.isfinite(embedding).all()
    reader = BridgeDataset([str(path)], seed=42, batch_size=1, train=False,
        augment=False, load_language=True, skip_unlabeled=True, relabel_actions=True,
        action_proprio_metadata=None, goal_relabeling_strategy='uniform',
        goal_relabeling_kwargs={'reached_proportion': 0.0})
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    collected = dict(image=[], state=[], next_state=[], target_raw=[], indices=[])
    count = 0
    for offset, batch in enumerate(reader.tf_dataset.with_options(options).as_numpy_iterator()):
        assert offset < len(order), 'Author reader yielded extra windows.'
        index = order[offset]
        count += 1
        if index not in selected_indices:
            continue
        assert batch['goals']['language'][0].decode('utf-8') == languages[0]
        collected['image'].append(batch['observations']['image'][0])
        collected['state'].append(batch['observations']['proprio'][0])
        collected['next_state'].append(batch['next_observations']['proprio'][0])
        collected['target_raw'].append(batch['actions'][0])
        collected['indices'].append(index)
    assert count == 316 and collected['indices'] == selected['indices']
    data = {field: np.stack(values) for field, values in collected.items()}
    data['starts'] = np.asarray(selected['starts'])
    data['episode_keys'] = np.full(21, EPISODE)
    assert data['image'].shape == (21, 256, 256, 3) and data['image'].dtype == np.uint8
    np.testing.assert_allclose(data['target_raw'][:, :6],
                              data['next_state'][:, :6]-data['state'][:, :6])
    data['target'], mean, std = prepare_targets(data['target_raw'])
    baseline = metrics(np.zeros_like(data['target']), data['target'],
                       data['state'], data['next_state'], mean, std)
    config = get_config('lc_bc').to_dict()
    config['agent_kwargs']['warmup_steps'] = 50
    config['agent_kwargs']['decay_steps'] = 1000
    helper = Path(__file__).with_name('run_bridge_lcbc_wrap.py')
    settings = dict(purpose='single_training_episode_replay', episode=EPISODE,
        dataset_identity=manifest['dataset_identity'], official_commit=COMMIT,
        windows=21, open_targets=3, closed_targets=18, steps=1000, batch_size=16, seed=42,
        learning_rate=3e-4, warmup_steps=50, eval_interval=100, augmentation=False,
        author_config=config, action_stats_source='Only the selected 21 training windows',
        action_mean=mean.tolist(), action_std=std.tolist(),
        training_file_windows_read=316, training_windows_used_for_updates=21,
        validation_targets_read=False, test_targets_used=False,
        checkpoint_selection='Fixed final update 1000; no best checkpoint selection',
        label_variant='Wrapped Euler component differences; no changed labels in this episode',
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline=baseline, predeclared_criteria=decide(baseline, baseline)['thresholds'])
    write_json(args.output_dir/'experiment.json', settings)
    print('Selected:', EPISODE, '21 windows, open=3 closed=18', flush=True)
    print('Predeclared thresholds:', settings['predeclared_criteria'], flush=True)

    def batch_for(indices):
        return dict(observations={'image': data['image'][indices]},
                    goals={'language': np.repeat(embedding, len(indices), axis=0)},
                    actions=(data['target'][indices]-mean)/std)
    example = batch_for(np.arange(16))
    agent = LCBCAgent.create(rng=jax.random.PRNGKey(42), observations=example['observations'],
        goals=example['goals'], actions=example['actions'],
        encoder_def=encoders[config['encoder']](**config['encoder_kwargs']),
        **config['agent_kwargs'])
    history = []
    final_decision = None

    def evaluate(step):
        nonlocal final_decision
        predictions = []
        for start in range(0, 21, 16):
            images = data['image'][start:start+16]
            n = len(images)
            images = np.concatenate([images, np.repeat(images[-1:], 16-n, axis=0)])
            output = np.asarray(agent.sample_actions(
                {'image': images}, {'language': np.repeat(embedding, 16, axis=0)},
                seed=jax.random.PRNGKey(42), argmax=True))[:n]
            predictions.append(output*std+mean)
        prediction = np.concatenate(predictions)
        score = metrics(prediction, data['target'], data['state'], data['next_state'], mean, std)
        report = dict(step=step, partitions={'train_replay': score})
        history.append(report)
        write_json(args.output_dir/'history.json',
                   dict(baselines={'train_replay': baseline}, evaluations=history))
        print(f"step={step} train_replay n=21 position={score['position_cm']:.4f}cm "
              f"rotation={score['rotation_deg']:.4f}deg open_recall={score['open_recall']:.3f} "
              f"close_recall={score['close_recall']:.3f}", flush=True)
        if step == 1000:
            write_json(args.output_dir/'final_metrics.json', report)
            final_decision = decide(score, baseline)
            write_json(args.output_dir/'diagnostic_decision.json', final_decision)
            np.savez_compressed(args.output_dir/'final_predictions.npz',
                train_replay_prediction_native7=prediction,
                **{f'train_replay_{key}': value for key, value in data.items() if key != 'image'})
            (args.output_dir/'final_params.msgpack').write_bytes(serialization.to_bytes(agent.state.params))

    evaluate(0)
    sampler = CyclingSampler(21, 42)
    started = time.monotonic()
    for step in range(1, 1001):
        agent, info = agent.update(batch_for(sampler.take(16)))
        if step == 1 or step % 50 == 0:
            loss = float(info['actor_loss'])
            assert np.isfinite(loss)
            print(f'update={step}/1000 actor_loss={loss:.6f} lr={float(info["lr"]):.8f} '
                  f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        if step % 100 == 0:
            evaluate(step)
    assert sampler.frequency.sum() == 16000
    assert sampler.frequency.min() == 761 and sampler.frequency.max() == 762
    write_json(args.output_dir/'training_coverage.json', dict(
        windows=21, draws=16000, indices=data['indices'].tolist(),
        min_draws=761, max_draws=762, draws_by_window=sampler.frequency.tolist()))
    write_json(args.output_dir/'summary.json', dict(completed_updates=1000,
        covered_train_replay_windows=21, covered_validation_windows=0,
        test_targets_used=False, elapsed_seconds=time.monotonic()-started,
        diagnostic_passed=final_decision['passed'], results_dir=str(args.output_dir)))
    print('SINGLE EPISODE REPLAY COMPLETED; diagnostic_passed=', final_decision['passed'], flush=True)
    print('Results:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
