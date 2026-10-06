"""Pinned official Octo notebook single-image inference, without robot control."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

IMAGE_URL = ('https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/'
             'datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/'
             'traj_group0/traj0/images0/im_12.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path')
    plan = json.loads(args.plan.read_text())
    report = dict(stage='asset_setup' if args.prepare else 'official_single_image_inference',
                  trained=False, robot_controlled=False, generalization_evaluated=False,
                  model_used=False, passed=False, plan=plan)
    try:
        commit = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != plan['code_commit']:
            raise ValueError('Official code commit differs')
        report['source_commit'] = commit
        args.assets.mkdir(parents=True, exist_ok=True)
        if args.prepare:
            from huggingface_hub import snapshot_download
            import requests
            model_path = snapshot_download(plan['model_repository'], revision=plan['model_revision'],
                allow_patterns=['config.json', 'dataset_statistics.json', 'example_batch.msgpack', '300000/**'])
            # Seed main cache for the unchanged official calls to 't5-base'.
            # Require that its resolved revision equals the version fixed here.
            t5_path = snapshot_download('t5-base', revision='main', allow_patterns=[
                'config.json', 'tokenizer.json', 'tokenizer_config.json', 'spiece.model',
                'special_tokens_map.json', 'added_tokens.json'])
            if Path(t5_path).name != plan['t5_revision']:
                raise ValueError('T5 revision changed; do not silently use different resources')
            response = requests.get(IMAGE_URL, timeout=60)
            response.raise_for_status()
            image_path = args.assets / 'official_fork.jpg'
            image_path.write_bytes(response.content)
            assets = dict(model_path=model_path, t5_revision=Path(t5_path).name,
                          image_url=IMAGE_URL, image_sha256=hashlib.sha256(response.content).hexdigest())
            (args.assets / 'assets.json').write_text(json.dumps(assets, indent=2))
            report['assets'] = assets
        else:
            import numpy as np
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            import jax
            if jax.default_backend() != 'gpu':
                raise RuntimeError('JAX fell back to CPU')
            from PIL import Image
            from octo.model.octo_model import OctoModel
            assets = json.loads((args.assets / 'assets.json').read_text())
            image_path = args.assets / 'official_fork.jpg'
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != assets['image_sha256']:
                raise ValueError('Official example image changed')
            if Path(assets['model_path']).name != plan['model_revision']:
                raise ValueError('Model asset revision differs')
            if assets['t5_revision'] != plan['t5_revision']:
                raise ValueError('T5 asset revision differs')
            start = time.monotonic()
            model = OctoModel.load_pretrained(assets['model_path'])
            report['model_used'] = True
            image = np.asarray(Image.open(image_path).convert('RGB').resize((256, 256)))
            observation = dict(image_primary=image[None, None], timestep_pad_mask=np.array([[True]]))
            task = model.create_tasks(texts=['pick up the fork'])
            stats = model.dataset_statistics['bridge_dataset']['action']
            action = np.asarray(jax.device_get(model.sample_actions(observation, task,
                unnormalization_statistics=stats, rng=jax.random.PRNGKey(0))))
            if list(action.shape) != plan['expected_action_shape'] or not np.isfinite(action).all():
                raise ValueError('Official action shape/finiteness check failed')
            report.update(action_shape=list(action.shape), native_actions=action.tolist(),
                language='pick up the fork', history_length=1, sampling_seed=0,
                backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                elapsed_seconds=time.monotonic()-start, assets=assets,
                normalization_source='model.dataset_statistics[bridge_dataset][action]',
                action_statistics={k:np.asarray(v).tolist() for k,v in stats.items()},
                versions={n:importlib.metadata.version(n) for n in ['jax','jaxlib','flax','numpy',
                    'tensorflow','transformers','huggingface-hub','orbax-checkpoint']})
        report['passed'] = True
        print(report['stage'].upper(), 'PASSED', flush=True)
    except Exception as error:
        report['error'] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
