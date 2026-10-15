"""Freeze small dependencies for one native context-boundary inference job."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'cluster_packages/bridge_project_condition_probe_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def prepare():
    assert not (PACK / 'protocol.json').exists()
    run = ROOT / 'reports/bridge-state-fit-202278/real'
    report, plan = read(run / 'report.json'), read(run / 'plan.json')
    assert report['execution_checks_passed'] and report['state_arm'] == 'real'
    checkpoint = report['checkpoints'][0]
    assert checkpoint['path'] == 'reduced_u8192_trainable.pt' and checkpoint['roundtrip_exact']
    cache_dirs = ['bridge_batch_input_audit_v1/reference', 'bridge_growth64_v1/added_reference', 'bridge_growth128_v1/new_reference']
    saved = []
    for d in cache_dirs:
        with np.load(ROOT / 'cluster_packages' / d / 'arrays.npz') as z:
            saved.append(z['currents'].copy())
    old, added, new = saved
    currents = np.concatenate([old[:32], added, new, old[32:]])
    assert currents.shape == (168, 1) and currents.dtype == np.float32
    expected = [report['data_checks'][i if i < 32 else i + 40 if i < 128 else i - 96]['current_sha256'] for i in range(168)]
    assert [hashlib.sha256(v.tobytes()).hexdigest() for v in currents] == expected
    np.savez_compressed(PACK / 'currents.npz', currents=currents)
    required = {'runs/bridge-state-fit-202278/real/' + n: sha(run / n) for n in
                ['report.json', 'plan.json', 'reduced_u8192_train.npz', 'reduced_u8192_development.npz']}
    required['runs/bridge-state-fit-202278/real/' + checkpoint['path']] = checkpoint['sha256']
    required.update({'bridge_growth128_v1/' + n: h for n, h in report['source_sha256'].items()})
    for name, h in report['source_sha256'].items():
        assert sha(ROOT / 'cluster_packages/bridge_growth128_v1' / name) == h
    fixed = read(ROOT / 'reports/bridge_broccoli_observation_audit_203699/frozen_observation_intervention_plan.json')['donors']
    assert len(fixed) == 9
    protocol = dict(stage='prepared_frozen_project_context_probe', run_directory='runs/bridge-state-fit-202278/real',
                    checkpoint_name=checkpoint['path'], required_file_sha256=required,
                    runtime_config=plan['runtime_config'], manifest_sha256=report['manifest_sha256'],
                    torch_version=report['torch_version'], sampling_seeds=report['sampling_seeds'],
                    trace_steps=[99, 74, 49, 24, 0], batch_size=2, pairs=fixed,
                    current_checks=expected, replay_tolerance=1e-4, updates=0,
                    expected_sampler_calls=150, job_wall_minutes=15, output_limit_mib=5,
                    clip_forward_passes=0, adapter_forward_passes=0,
                    condition='Original saved1024-dimensional fused context, including real current state; no repeated fusion',
                    noise='Paired initial and per-step RNG; unchanged native stochastic DDPM100',
                    limits='Fixed nine post-hoc diagnostic pairs, not an untouched test or closed-loop success evaluation; expert path is not the only valid task path')
    (PACK / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8', newline='\n')
    (PACK / 'probe.sh').write_text('''#!/bin/bash
#SBATCH --job-name=bridge-project-condition
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=00:15:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-project-condition-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/project-condition-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_project_condition_probe_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u probe_project_conditions.py \\
  --project /projects/Zeroshot --output "/projects/Zeroshot/runs/bridge-project-condition-${SLURM_JOB_ID}"
''', encoding='utf-8', newline='\n')
    print('Prepared: existing202278 checkpoint, fixed9 pairs, no new training or weights.')


def package():
    proof = read(ROOT / 'reports/bridge_project_condition_probe_preparation.json')
    assert proof['passed'] and proof['bash_syntax_passed'] and proof['protocol_sha256'] == sha(PACK / 'protocol.json')
    for n, h in proof['source_sha256'].items():
        assert sha(PACK / n) == h
    (PACK / 'local_preparation.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (PACK / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='ascii', newline='\n')
    files.append(PACK / 'SHA256SUMS')
    archive = PACK.with_suffix('.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:
            assert z.read(p.relative_to(PACK.parent).as_posix()) == p.read_bytes()
    assert archive.stat().st_size < 1024 * 1024
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='ascii', newline='\n')
    print(json.dumps(dict(bytes=archive.stat().st_size, sha256=sha(archive), files=len(files))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'package')
    globals()[sys.argv[1]]()
