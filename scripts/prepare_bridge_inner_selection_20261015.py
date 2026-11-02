"""Freeze one internal checkpoint-selection trial; no training or downloads."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_inner_selection_v1'
sys.path.insert(0, str(PACK))
from selection_contract import sha, verify_split, observed_statistics, training_schedule


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    assert not (PACK/'protocol.json').exists(), 'Preserve fixed protocol'
    plan_path = ROOT/'reports/bridge_inner_selection_plan_20261015/plan.json'
    plan = read(plan_path)
    base = ROOT/'cluster_packages/bridge_growth128_v1'
    manifest = read(base/'manifest.json')
    assert sha(base/'manifest.json') == plan['source_manifest_sha256']
    indices = verify_split(manifest['train']+manifest['development'], plan)
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz', allow_pickle=False) as z:
        stats = observed_statistics(z['current_pose'], indices['optimization_indices'])
    inherited = {f'bridge_growth128_v1/{n}':sha(base/n) for n in manifest['source_sha256']}
    inherited['bridge_growth128_v1/manifest.json'] = sha(base/'manifest.json')
    state = ROOT/'cluster_packages/bridge_state_conditioning_v1'
    inherited.update({f'bridge_state_conditioning_v1/{n}':sha(state/n)
                      for n in ['fit_state_arm.py','state_input_contract.py','protocol.json','current_pose.npz']})
    original = read(ROOT/'reports/bridge-state-fit-202278/real/report.json')
    local_original = read(ROOT/'reports/bridge_state_prior_preflight_v2_20261015/report.json')
    schedule = training_schedule(indices['optimization_indices'])
    protocol = dict(date='2026-10-15', stage='one_inner_episode_action_selection_trial',
                    inherited_sha256=inherited, plan=plan, plan_sha256=sha(plan_path),
                    state_statistics=stats,
                    schedule_sha256=hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),
                    original_initial_shared_sha256=original['initial_shared_sha256'],
                    initialization_sha256_by_torch={original['torch_version']:original['initial_shared_sha256'],
                        local_original['torch_version']:local_original['initial_shared_sha256']},
                    preserve='Frozen growth128 direct x0/DDPM100 modules, initial43, real current-state port, frozenCLIP/gripper; no residual prior.',
                    controls='Within one96-window trajectory selected-vs-final, not a causal comparison against old128-window training.',
                    scaling='Only96 optimization-window current XYZ fit state scaling; action scale0.1m fixed before split, quaternion unchanged. Old128 stats are checked only as raw input provenance and discarded.',
                    cache='May reuse audited full frozen CLIP tokens; encoder has no train-fitted normalization and all weights stay frozen. Cache contains observations only, not actions.',
                    preflight='Eight unobserved updates vs the same eight with inner selection at4/8, exact parameter/loss/Adam replay; selected-file reload must exactly replay32 inner predictions. No outer scoring; temporary selected checkpoint removed.',
                    storage='At most best.pt+final.pt, combined<1200MiB; shared token cache<256MiB; no optimizer checkpoints; no weight/data downloads.',
                    stopping='Exactly one6144-update formal trajectory; no extension even if train fit is poor or selected checkpoint fails outer metrics.',
                    limitations=['Original40 outer windows have been reused extensively; this is exploratory development, not an untouched test.',
                                 'Only pose, no learned gripper or physical robot task success.',
                                 'Inner and outer labels are decoded by existing identity audit, but never used for optimizer gradients or normalization. Inner labels only select weights; outer labels only score after lock.'])
    (PACK/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8', newline='\n')
    for stage in ['preflight','fit']:
        extra = '' if stage == 'preflight' else ': "${SELECTION_PREFLIGHT_JOB:?Supply successful preflight ID}"\n'
        reference = '' if stage == 'preflight' else ' \\\n  --preflight-reference "/projects/Zeroshot/runs/bridge-selection-preflight-${SELECTION_PREFLIGHT_JOB}"'
        shell = f'''#!/bin/bash
#SBATCH --job-name=bridge-selection-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage == 'preflight' else '03:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-selection-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{extra}PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_inner_selection_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_selection.py \\
  --token-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt \\
  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-selection-{stage}-${{SLURM_JOB_ID}}"{reference}
'''
        assert not any(line.startswith('+') for line in shell.splitlines())
        (PACK/f'{stage}.sh').write_text(shell, encoding='utf-8', newline='\n')
    print('PLAN/SPLIT/STATS/SCHEDULE FROZEN', sha(PACK/'protocol.json'))


def package():
    local = read(ROOT/'reports/bridge_inner_selection_preflight_v2_20261015/report.json')
    audit_path = ROOT/'reports/bridge_inner_selection_preflight_v2_20261015/independent_audit.json'
    audit = read(audit_path)
    assert local['execution_checks_passed'] and local['observer_replay_exact'] and local['selected_checkpoint_replay_exact']
    assert audit['verified'] and audit['protocol_sha256'] == sha(PACK/'protocol.json')
    for name, expected in local['entry_sha256'].items(): assert sha(PACK/name) == expected
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(
        passed=True, report_sha256=audit['report_sha256'], independent_audit_sha256=sha(audit_path),
        protocol_sha256=sha(PACK/'protocol.json'), operational_updates=16,
        observer_replay_exact=True, selected_checkpoint_replay_max_difference=0.,
        elapsed_seconds=local['elapsed_seconds'], peak_reserved_gib=local['peak_reserved_gib'],
        outer_scoring_executed=False, cluster_job_submitted=False), indent=2)+'\n', encoding='utf-8', newline='\n')
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    for path in files:
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'))
        assert path.suffix in {'.py','.sh','.json','.md'}
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='ascii', newline='\n')
    archive = PACK.with_suffix('.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in files+[PACK/'SHA256SUMS']:
            z.write(path, f'{PACK.name}/{path.name}')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in files+[PACK/'SHA256SUMS']:
            assert z.read(f'{PACK.name}/{path.name}') == path.read_bytes()
    assert archive.stat().st_size < 1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='ascii', newline='\n')
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))))


if __name__ == '__main__':
    assert sys.argv[1:] in [['prepare'], ['package']]
    globals()[sys.argv[1]]()
