#!/bin/bash
#SBATCH --job-name=bridge-language
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-language-%j.out

set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export TFHUB_CACHE_DIR=/projects/Zeroshot/.tmp/tfhub
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
mkdir -p "$TMPDIR" "$TFHUB_CACHE_DIR"
BRIDGE_REPO=/projects/Zeroshot/baselines/bridge_data_v2
[ "$(git -C "$BRIDGE_REPO" rev-parse HEAD)" = bc60a35b701a12021c8c95e9d8601274d3acd928 ]
git -C "$BRIDGE_REPO" diff --quiet
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
from jaxrl_m.data.text_processing import MuseEmbedding, MULTI_MODULE

root = Path('/projects/Zeroshot/data/bridge_single_step_subset')
manifest = json.loads((root / 'export_manifest.json').read_text())
assert manifest['test_targets_used'] is False
options = tf.data.Options()
options.threading.private_threadpool_size = 2
instructions = set()
for relative in sorted(manifest['files']):
    path = root / relative
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == manifest['files'][relative]['sha256']
    for payload in tf.data.TFRecordDataset(str(path)).with_options(options).as_numpy_iterator():
        example = tf.train.Example.FromString(payload)
        feature = example.features.feature['language'].bytes_list.value[0]
        strings = tf.io.parse_tensor(feature, tf.string).numpy()
        assert len(strings) == 1
        instructions.add(strings[0].decode('utf-8'))
assert instructions == {episode['instruction'] for episode in manifest['episodes']}
languages = sorted(instructions)
assert len(languages) == 1, 'This fixed subset should contain one task instruction.'
output = Path('/projects/Zeroshot/baseline_setup/muse_fixed_subset.npz')
if output.exists():
    with np.load(output, allow_pickle=False) as saved:
        assert saved['dataset_identity'].item() == manifest['dataset_identity']
        assert saved['languages'].tolist() == languages
        assert saved['source_url'].item() == MULTI_MODULE
        embeddings = saved['embeddings'].copy()
    print('Reusing verified language cache:', output, flush=True)
else:
    print('Loading author MUSE model:', MULTI_MODULE, flush=True)
    print('Persistent model cache:', os.environ['TFHUB_CACHE_DIR'], flush=True)
    processor = MuseEmbedding()
    embeddings = np.asarray(processor.encode(languages), dtype=np.float32)
    assert embeddings.shape == (len(languages), 512)
    assert np.isfinite(embeddings).all()
    temporary = output.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, languages=np.array(languages), embeddings=embeddings,
            dataset_identity=np.array(manifest['dataset_identity']), source_url=np.array(MULTI_MODULE))
    temporary.replace(output)
assert embeddings.shape == (len(languages), 512) and np.isfinite(embeddings).all()
report = {'passed': True, 'source_url': MULTI_MODULE, 'dataset_identity': manifest['dataset_identity'],
          'languages': languages, 'embedding_shape': list(embeddings.shape),
          'tensorflow': tf.__version__, 'cache_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
          'test_targets_used': False, 'model_trained': False}
output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Language count:', len(languages), 'Embedding shape:', embeddings.shape, flush=True)
print('Saved:', output, flush=True)
print('AUTHOR MUSE LANGUAGE CACHE: PASSED', flush=True)
PY
