"""Package ONE frozen-CLIP post-Adapter preflight; never submit a job."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'cluster_packages/bridge_prefix_pair_v1'
OUT=ROOT/'cluster_packages/bridge_spatial_readout_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name,text):
    (OUT/name).write_bytes(text.replace('\r\n','\n').encode('utf-8'))


def main():
    archive=OUT.with_suffix('.zip')
    if OUT.exists() or archive.exists():raise FileExistsError('Preserve existing release')
    OUT.mkdir();(OUT/'frozen').mkdir()
    sources={}
    def copy(source,name):
        destination=OUT/name
        shutil.copyfile(source,destination)
        assert sha(source)==sha(destination)
        sources[name]=dict(source=str(source.relative_to(ROOT)),sha256=sha(source))
    for name in ('adapter.py','models.py','diffusion_decoder.py','dataset.py','state_input_contract.py'):
        copy(PRE/'frozen'/name,'frozen/'+name)
    for source,name in (
        ('bridge_spatial_readout_prototype_v1.py','bridge_spatial_readout_prototype_v1.py'),
        ('bridge_spatial_readout_interface_v1.py','bridge_spatial_readout_interface_v1.py'),
        ('run_bridge_spatial_preflight_v1.py','run_preflight.py'),
        ('bridge_prefix_input_contract_v1.py','bridge_prefix_input_contract_v1.py'),
        ('bridge_partial_future_contract_v1.py','bridge_partial_future_contract_v1.py')):
        copy(ROOT/'scripts'/source,name)
    for source,name in (('old_rows.json','old_rows.json'),('train_manifest.json','train_manifest.json'),
        ('state_protocol.json','state_protocol.json'),('base_protocol.json','base_protocol.json'),
        ('old_report.json','old_report.json'),('input_reference/report.json','input_audit.json')):
        copy(PRE/source,name)
    base=json.loads((ROOT/'configs/active_experiment_reference.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'cluster_packages/bridge_coverage_replication_v1/schedule.npz',allow_pickle=False) as z:
        batches=z['batches'][4096:8192].copy()
    assert batches.shape==(4096,8) and batches.max()<1836
    np.savez_compressed(OUT/'schedule.npz',batches=batches)
    protocol=dict(date='2026-10-18',stage='preflight_release_only',
        objective='Does learned patch weighting after frozen CLIP improve near-term new-demonstration pose prediction without losing full16 performance?',
        checkpoint=dict(relative_path='runs/bridge-coverage-rep-fit-207558/final_trainable.pt',
            sha256=base['checkpoint_sha256'],bytes=base['checkpoint_bytes']),
        training_reference=dict(relative_path='runs/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz',
            sha256=sha(ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz')),
        hard_constraints=dict(CLIP_visual_and_text_frozen=True,diffusion_required=True,
            gripper_frozen=True,final25_closed=True,original207558_unchanged=True),
        difference=dict(control='Original equal patch weights; new score parameters fixed at zero',
            candidate='Only1026 new post-Adapter score parameters learn; same base trainables',
            same_new_module_in_both_arms=True,content_or_coordinate_variants_allowed=False),
        planned_formal=dict(start='same207558, zero score port, fresh Adam per arm; no optimizer state guessed',
            arms=['control','candidate'],updates_each=4096,batch_size=8,learning_rate=1e-5,
            optimizer='Adam',weight_decay=0.,gradient_clip=1.,train_mode='same original model.train(), frozenCLIP.eval()',
            original_base_trainables=['adapter','diffusion_decoder','history_projection','state_projection'],
            input_windows=1836,training_episodes=405,horizon=16,
            schedule='Exact original207558 batches4097..8192 reused as a new paired continuation schedule',
            context_seed='430000 + (8192 + update)',diffusion_seed='420000 + (8192 + update)',
            no_partial_future_or_request_port=True,final_weights_only=True,
            no_selection_or_early_stopping=True,sampling_seeds=[1101,1102,1103],sampling_steps=100,
            validation='Same already-used209481 new10 development episodes/50 original full16 windows, including all start0; NOT a fresh final test',
            unchanged_baseline='Reuse209481 saved original207558 full16 predictions after input/replay checks; no repeat old1836 scoring',
            primary='Episode-equal FIRST target position and rotation, within original full16 outputs',
            full16_and_endpoint='Must not worsen position/rotation versus control and original207558',
            admission_gates=[
                'FIRST target episode-equal position strictly lower than control, original207558 AND no-motion',
                'FIRST target episode-equal rotation no worse than control, original207558 AND no-motion',
                'At least6 of the SAME10 episodes have FIRST target position lower than BOTH control and original207558',
                'Full16 path and endpoint position/rotation episode-equal no worse than BOTH control and original207558',
                'Window-weighted FIRST target and full16 path position/rotation no worse than BOTH control and original207558'],
            fail='Retain207558; close this readout branch, no extra seeds/budget/variants/final25 scoring',
            pass_decision='Development candidate only; any final25 assessment is separate and not automatic'),
        preflight=dict(updates_each=8,development_scoring=False,final25_payload_read=False,
            original_replay_windows=[0,32,64,96],original_replay_seed=1101,
            context_atol=1e-4,prediction_atol=1e-4,rtol=0.,
            second_replay='Zero initialized new readout through same original full16 sampler',
            restore='Each arm after4 updates: save/reload weights+Adam and compare next calculation',
            frozen_parameters='CLIP and all original nontrainables hash unchanged; no gradients',
            new_parameters='Candidate first gradients finite nonzero; score weights changed. Control zeros unchanged',
            peak_reserved_gib_limit=7.5,estimated_formal_seconds_limit=7200,
            no_new_encoding=True,no_new_persistent_weights=True,no_automatic_formal_submit=True),
        schedule_sha256=sha(OUT/'schedule.npz'),source_provenance=sources)
    write('protocol.json',json.dumps(protocol,indent=2)+'\n')
    write('preflight.sh','''#!/bin/bash
#SBATCH --job-name=bridge-spatial-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-spatial-preflight-%j.out
set -euo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
SPATIAL_USED_BYTES=$(du --apparent-size -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( SPATIAL_USED_BYTES + 8*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Planning headroom insufficient; no live quota claim"
  exit 1
fi
cd /projects/Zeroshot/bridge_spatial_readout_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_preflight.py \\
  --project /projects/Zeroshot \\
  --output "/projects/Zeroshot/runs/bridge-spatial-preflight-${SLURM_JOB_ID}"
''')
    write('submit_preflight_once.sh','''#!/bin/bash
set -euo pipefail
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE"
exec 9>submit_preflight.lock
flock -x 9
if [[ -s submitted_preflight.job ]]; then
  printf 'Existing spatial preflight: %s\\n' "$(cat submitted_preflight.job)"
  exit 0
fi
sha256sum -c SHA256SUMS
SPATIAL_REPLY=$(sbatch --parsable "$CODE/preflight.sh")
SPATIAL_NUMBER="${SPATIAL_REPLY%%;*}"
[[ "$SPATIAL_NUMBER" =~ ^[0-9]+$ ]]
printf '%s\\n' "$SPATIAL_NUMBER" > submitted_preflight.job
printf 'SPATIAL_PREFLIGHT: %s\\n' "$SPATIAL_NUMBER"
''')
    write('verify_preflight.py','''"""Validate one complete operational report; never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
CODE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
r=json.loads((args.directory/'report.json').read_text())
p=json.loads((CODE/'protocol.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert r['stage']=='spatial_readout_preflight' and r['execution_checks_passed']
assert r['protocol_sha256']==sha(CODE/'protocol.json')
assert r['package_sums_sha256']==sha(CODE/'SHA256SUMS')
assert r['actual_checkpoint_sha256']==p['checkpoint']['sha256']
for flag in ('development_scored','final25_payloads_read','formal_training_executed','gripper_trained',
             'CLIP_training_allowed','downloads','final_weights_saved','validation_payloads_read'):
    assert not r[flag],flag
assert r['clip_unchanged'] and r['other_frozen_unchanged'] and r['optimization_operations']==16
assert r['images_encoded']==0 and r['numeric_inputs']['episodes']==405 and r['numeric_inputs']['windows']==1836
assert r['numeric_inputs']['target_max_abs']<=1e-6
for name in ('original207558_replay','zero_readout_full_sampler_replay'):
    assert r[name]['passed'] and r[name]['context_max_abs']<=1e-4 and r[name]['prediction_max_abs']<=1e-4
assert [v['arm'] for v in r['runs']]==['control','candidate']
for v in r['runs']:
    assert v['logical_updates']==len(v['losses'])==8
    assert v['base_parameters_changed'] and v['weights_Adam_and_next_calculation_restore_exact']
assert not r['runs'][0]['score_parameters_changed'] and r['runs'][1]['score_parameters_changed']
assert all(v>0 for v in r['runs'][1]['new_score_gradient_norms'].values())
assert r['runs'][0]['initial_base_sha256']==r['runs'][1]['initial_base_sha256']
assert r['runs'][0]['losses'][0]==r['runs'][1]['losses'][0]
assert r['runs'][1]['trainable_parameters']-r['runs'][0]['trainable_parameters']==1026
assert r['peak_reserved_gib']<=7.5 and r['estimated_paired_formal_seconds']<=7200
print('SPATIAL PREFLIGHT VERIFIED; no generalization verdict or automatic formal submission')
print(json.dumps({k:r[k] for k in ('peak_reserved_gib','warm_update_seconds','estimated_paired_formal_seconds')},indent=2))
''')
    write('README.md','''# Frozen-CLIP post-Adapter spatial readout: ONE preflight

CLIP visual/text weights MUST remain frozen. Same original405 training episodes/1836 windows, full16 target and native DDPM100. The candidate learns1026 readout scores after the original eight Adapter blocks; control scores remain zero. Both keep the same original base trainables. No late-input/requested-horizon port, image augmentation, new labels or dataset scan.

This release has NO formal-training entry. protocol.json declares the bounded planned4096 updates/arm, unchanged data/schedule and development admission gates BEFORE any development scoring. The one preflight only replays actual207558, zero readout+native sampler, and runs8 updates per arm with weights/Adam restore checks. It does not score development or read final25. A pass is operational evidence, not improvement.

Existing dependencies: original207558 checkpoint and saved train predictions, bridge_training_coverage_v1/full_tokens.pt, bridge_expansion_training_v1/data, frozen local CLIP, existing multitask-preflight-v1 environment. Missing/stale assets STOP; no download or fallback/re-encoding. Only the405 pinned training payloads are parsed numerically; no training images decoded. Old40 development token rows are present in the inherited cache for identity checking but are not sampled or scored.

Run on cluster after unpacking to /projects/Zeroshot/bridge_spatial_readout_v1:

    bash /projects/Zeroshot/bridge_spatial_readout_v1/submit_preflight_once.sh

Inspect output and verify once completed:

    sacct -j "$SPATIAL_PREFLIGHT" --format=JobID,State,ExitCode,Elapsed,NodeList
    tail -n 30 "/projects/Zeroshot/logs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT.out"
    /projects/Zeroshot/envs/multitask-preflight-v1/bin/python /projects/Zeroshot/bridge_spatial_readout_v1/verify_preflight.py "/projects/Zeroshot/runs/bridge-spatial-preflight-$SPATIAL_PREFLIGHT"

The submit-once helper prints the job ID and records it; it does not export a variable into your parent shell. Set SPATIAL_PREFLIGHT to that printed numeric ID. Do not resubmit automatically after a failed report.

No weights in this ZIP; temporary restore checkpoints are removed by exact filename after successful verification. Keep the effective207558 reference. No grasp-success or unseen-task claims.
''')
    for path in OUT.rglob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    files=sorted(p for p in OUT.rglob('*') if p.is_file())
    write('SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in files))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent).as_posix())
    summary=dict(package=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha(archive),
        files=sum(p.is_file() for p in OUT.rglob('*')),job_submitted=False,
        GPU_used=False,CLIP_frozen=True,formal_entry_present=False)
    (ROOT/'reports/bridge_spatial_preflight_release_20261018.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
