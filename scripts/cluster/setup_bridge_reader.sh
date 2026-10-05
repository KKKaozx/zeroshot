#!/bin/bash
#SBATCH --job-name=bridge-reader-check
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-reader-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export JAX_PLATFORMS=cpu
mkdir -p "$TMPDIR" /projects/Zeroshot/baselines
BRIDGE_REPO=/projects/Zeroshot/baselines/bridge_data_v2
BRIDGE_COMMIT=bc60a35b701a12021c8c95e9d8601274d3acd928
if [ ! -e "$BRIDGE_REPO" ]; then
  git clone --no-checkout https://github.com/rail-berkeley/bridge_data_v2.git "$BRIDGE_REPO"
  git -C "$BRIDGE_REPO" checkout --detach "$BRIDGE_COMMIT"
fi
[ "$(git -C "$BRIDGE_REPO" rev-parse HEAD)" = "$BRIDGE_COMMIT" ]
git -C "$BRIDGE_REPO" diff --quiet
git -C "$BRIDGE_REPO" diff --cached --quiet
export PYTHONPATH="$BRIDGE_REPO${PYTHONPATH:+:$PYTHONPATH}"

/projects/Zeroshot/envs/zeroshot/bin/python -u - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.agents.continuous.lc_bc import LCBCAgent

root = Path('/projects/Zeroshot/data/bridge_single_step_subset')
manifest = json.loads((root / 'export_manifest.json').read_text())
stats = json.loads((root / 'train_action_stats.json').read_text())
assert manifest['test_targets_used'] is False
assert manifest['official_commit'] == 'bc60a35b701a12021c8c95e9d8601274d3acd928'
mean = np.asarray(stats['action']['mean'], dtype=np.float32)
std = np.asarray(stats['action']['std'], dtype=np.float32)
assert mean.shape == std.shape == (7,) and np.all(std > 0)
options = tf.data.Options()
options.threading.private_threadpool_size = 2

spec = {
 'observations/images0': tf.uint8, 'next_observations/images0': tf.uint8,
 'observations/state': tf.float32, 'next_observations/state': tf.float32,
 'actions': tf.float32, 'terminals': tf.bool, 'language': tf.string,
}

def decoded_pairs(path):
    raw = tf.data.TFRecordDataset(str(path)).with_options(options)
    for payload in raw.as_numpy_iterator():
        example = tf.train.Example.FromString(payload)
        values = {k: tf.io.parse_tensor(
            example.features.feature[k].bytes_list.value[0], dtype).numpy()
            for k, dtype in spec.items()}
        for index in range(len(values['actions'])):
            yield values, index

def reader(path, metadata):
    ds = BridgeDataset([str(path)], seed=42, batch_size=1, train=False,
        augment=False, load_language=True, skip_unlabeled=True,
        relabel_actions=True, action_proprio_metadata=metadata,
        goal_relabeling_strategy='uniform',
        goal_relabeling_kwargs={'reached_proportion': 0.0})
    return ds.tf_dataset.unbatch().batch(8, drop_remainder=False).with_options(options)

results = {}
for partition, directory, expected_count in [('train', 'train', 316), ('validation', 'val', 55)]:
    relative = f'data/sweep_into_pile/{directory}/out.tfrecord'
    path = root / relative
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == manifest['files'][relative]['sha256']
    partition_result = {}
    for normalized in [False, True]:
        expected = iter(decoded_pairs(path))
        count = 0
        sizes = []
        for batch in reader(path, stats if normalized else None).as_numpy_iterator():
            sizes.append(len(batch['actions']))
            for j, action in enumerate(batch['actions']):
                values, i = next(expected)
                state = values['observations/state'][i]
                next_state = values['next_observations/state'][i]
                native = np.concatenate([next_state[:6] - state[:6], values['actions'][i, 6:7]])
                target = (native - mean) / std if normalized else native
                np.testing.assert_allclose(action, target, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(batch['observations']['image'][j], values['observations/images0'][i])
                np.testing.assert_array_equal(batch['next_observations']['image'][j], values['next_observations/images0'][i])
                proprio = batch['observations']['proprio'][j]
                expected_proprio = (state - np.asarray(stats['proprio']['mean'])) / np.asarray(stats['proprio']['std']) if normalized else state
                np.testing.assert_allclose(proprio, expected_proprio, rtol=1e-5, atol=1e-6)
                assert batch['goals']['language'][j] == values['language'][0]
                assert bool(batch['terminals'][j]) == bool(values['terminals'][i])
                assert native[6] in (0.0, 1.0)
                count += 1
        assert count == expected_count == len(manifest['sample_order'][partition])
        assert next(expected, None) is None
        partition_result['normalized' if normalized else 'raw'] = {'windows': count, 'batch_sizes': sizes}
    results[partition] = partition_result
    print(f'{partition}: {count}/{expected_count}; raw and normalized fields PASSED; tail={sizes[-1]}', flush=True)

report = {'passed': True, 'tensorflow': tf.__version__, 'official_commit': manifest['official_commit'],
          'partitions': results, 'test_targets_used': False, 'augmentation_enabled': False,
          'lc_bc_agent_imported': True, 'model_trained': False}
output = Path('/projects/Zeroshot/baseline_setup') / f"cluster-reader-{os.environ['SLURM_JOB_ID']}.json"
output.write_text(json.dumps(report, indent=2))
print('Report:', output)
print('AUTHOR CODE IMPORT AND SUBSET READER: PASSED', flush=True)
PY
