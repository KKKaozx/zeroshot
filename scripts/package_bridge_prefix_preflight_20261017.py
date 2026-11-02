"""Build one self-contained preflight archive, without weights or final data."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_prefix_pair_v1'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(source,target):target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def main():
    assert not PACK.exists(),'Do not overwrite an uploaded/frozen package'
    audit=ROOT/'reports/bridge_prefix_input_audit_20261017'
    r=read(audit/'report.json');assert r['passed'] and not r['final25_payloads_read'] and not r['validation_scored']
    p=read(ROOT/'experiments/bridge_prefix_coverage_pair_v1/protocol.json')
    assert sha(ROOT/'experiments/bridge_prefix_coverage_pair_v1/protocol.json')==r['protocol_sha256']
    for name,h in r['source_sha256'].items():assert sha(ROOT/name)==h
    PACK.mkdir();(PACK/'frozen').mkdir();(PACK/'input_reference').mkdir()
    for name in ('models.py','adapter.py','diffusion_decoder.py','dataset.py'):
        copy(ROOT/'cluster_packages/bridge_growth128_v1'/name,PACK/'frozen'/name)
    copy(ROOT/'cluster_packages/bridge_state_conditioning_v1/state_input_contract.py',PACK/'frozen/state_input_contract.py')
    for name in ('bridge_partial_future_contract_v1.py','bridge_requested_prefix_diffusion_v1.py',
        'bridge_prefix_model_interface_v1.py','bridge_prefix_input_contract_v1.py','bridge_prefix_preflight_inputs_v1.py'):
        copy(ROOT/'scripts'/name,PACK/name)
    copy(ROOT/'scripts/run_bridge_prefix_preflight_v1.py',PACK/'run_preflight.py')
    copy(ROOT/'scripts/verify_bridge_prefix_preflight_v1.py',PACK/'verify_prefix_preflight.py')
    for name in ('protocol.json','matched_schedule.npz'):
        copy(ROOT/'experiments/bridge_prefix_coverage_pair_v1'/name,PACK/name)
    copy(ROOT/'reports/bridge_partial_future_contract_20261017/candidate_rows.json',PACK/'candidate_rows.json')
    copy(ROOT/'training_cache/exports/bridge_expansion_training_v1/manifest.json',PACK/'train_manifest.json')
    copy(ROOT/'cluster_packages/bridge_coverage_replication_v1/protocol.json',PACK/'base_protocol.json')
    copy(ROOT/'cluster_packages/bridge_coverage_replication_v1/rows.json',PACK/'old_rows.json')
    copy(ROOT/'cluster_packages/bridge_state_conditioning_v1/protocol.json',PACK/'state_protocol.json')
    copy(ROOT/'reports/bridge-coverage-rep-fit-207558/report.json',PACK/'old_report.json')
    for name in ('report.json','train_delta_reference.npz','validation_reference.npz','validation_rows.json'):
        copy(audit/name,PACK/'input_reference'/name)
    for path in (audit/'validation_data').iterdir():
        assert sha(path)==r['validation_export_sha256'][path.name]
        copy(path,PACK/'validation_data'/path.name)
    # Reuse known old numerical arrays; no raw-data re-audit or model run.
    with np.load(ROOT/'reports/bridge-demo806-fit-208207/reduced_u8192_train.npz',allow_pickle=False) as z:
        old_targets=z['targets'][:1836].copy()
    with np.load(ROOT/'reports/bridge_xyz_frame_audit_20261017/current_pose_and_prior.npz',allow_pickle=False) as z:
        poses=z['current_pose'][:1836].copy()
    path=PACK/'frozen/state_input_contract.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='state_features')
    ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    features=ns['state_features'](poses,read(PACK/'state_protocol.json')['state_features'])
    np.savez_compressed(PACK/'input_reference/old_training_reference.npz',targets=old_targets,state_features=features)
    bash='''#!/bin/bash
#SBATCH --job-name=bridge-prefix-preflight
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-prefix-preflight-%j.out
set -euo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
PROJECT_USED_BYTES=$(du --apparent-size -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 8*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then
  echo "Planning headroom insufficient; no live quota claim"
  exit 1
fi
cd /projects/Zeroshot/bridge_prefix_pair_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_preflight.py \\
  --project /projects/Zeroshot \\
  --output "/projects/Zeroshot/runs/bridge-prefix-preflight-${SLURM_JOB_ID}"
'''
    (PACK/'preflight.sh').write_text(bash,encoding='utf-8',newline='\n')
    copy(ROOT/'scripts/cluster/bridge_prefix_pair_v1/submit_preflight_once.sh',PACK/'submit_preflight_once.sh')
    (PACK/'README.md').write_text('''# One preflight only

No fit/automatic-submission entry is included. The raw validation_data directory
contains exactly the10 newly qualified development records with empty index
placeholders. The final25 are metadata only in protocol.json; no payloads.

Reuse remote207558 final_trainable.pt, reduced_u8192_train.npz, and existing
bridge_training_coverage_v1/full_tokens.pt. The frozenCLIP asset must already
exist; no installation/downloads. Only1558 new train images are encoded once.
Both CLIP token caches stay onCPU and only requested batches transfer toGPU.

The8 updates per arm are operational probes, NOT the4096-update formal run.
The temporary trainable/Adam serialization file is removed after exact restore.
Only the new visual token cache persists; no new final model is saved.

After uploading the ZIP, extract under/projects/Zeroshot and submit preflight.sh
once. On completion run verify_prefix_preflight.py with its run directory.
Do not submit another job if a prior preflight is still queued or running.
Execution passing is not a generalization verdict.
''',encoding='utf-8',newline='\n')
    sums=[]
    for path in sorted(PACK.rglob('*')):
        if path.is_file():sums.append(f'{sha(path)}  {path.relative_to(PACK).as_posix()}')
    (PACK/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(PACK.rglob('*')):
            if path.is_file():z.write(path,PACK.name+'/'+path.relative_to(PACK).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    archive.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        preflight_only=True,validation_records=10,final25_payloads=0),indent=2))


if __name__=='__main__':main()
