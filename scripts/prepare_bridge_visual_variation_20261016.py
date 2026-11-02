"""Package one controlled visual-input intervention without a new model implementation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_visual_variation_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def prepare():
    assert not (PACK/'protocol.json').exists()
    plan=read(ROOT/'reports/bridge_visual_variation_plan_20261016/plan.json')
    base=ROOT/'cluster_packages/bridge_growth128_v1'
    paths=[base/n for n in plan['source_sha256']]+[base/'manifest.json']
    state=ROOT/'cluster_packages/bridge_state_conditioning_v1'
    paths += [state/n for n in ['fit_state_arm.py','state_input_contract.py','protocol.json','current_pose.npz']]
    paths += [ROOT/'cluster_packages/bridge_inner_selection_v1/selection_contract.py']
    protocol=dict(stage='one_matched_visual_variation_control',date='2026-10-16',plan=plan,
        inherited_sha256={p.relative_to(PACK.parent).as_posix():sha(p) for p in paths},
        preflight='4 common+4constant+4restored-same-LR+4reduced updates. Four fixed training windows, no development scoring. Verify loss/parameter/prediction restore; raw inputs unchanged before visual replacement.',
        formal='One8192-update mean-visual arm; reuse real-state202278 control; final128/40 arrays; no fitting-driven extension.',
        mean_visual='mean of original128 training visual token sequences, same for every query, stored in final checkpoint. Real instruction and current-state inputs retained.',
        storage='Reuse existing frozen token cache; one final nonCLIP checkpoint including1MiB mean tokens, small reports, no installations/downloads.')
    (PACK/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8',newline='\n')
    for stage in ['preflight','fit']:
        prefix=f'''#!/bin/bash
#SBATCH --job-name=bridge-visual-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage=='preflight' else '03:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-visual-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
'''
        if stage=='fit': prefix+=': "${VISUAL_PREFLIGHT_JOB:?Supply successful preflight ID}"\n'
        prefix+='''PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_visual_variation_v1
sha256sum -c SHA256SUMS
nvidia-smi
'''
        command=['/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_visual_control.py',
            '  --token-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt',
            '  --manifest ../bridge_growth128_v1/manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data',
            '  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1',
            '  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference',
            '  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json',
            '  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json',
            '  --state-arm real --state-protocol ../bridge_state_conditioning_v1/protocol.json',
            f'  --stage {stage} --output "/projects/Zeroshot/runs/bridge-visual-{stage}-${{SLURM_JOB_ID}}"']
        if stage=='fit': command.append('  --preflight-reference "/projects/Zeroshot/runs/bridge-visual-preflight-${VISUAL_PREFLIGHT_JOB}"')
        shell=prefix+(' '+chr(92)+'\n').join(command)+'\n'
        assert not any(line.startswith('+') for line in shell.splitlines())
        (PACK/f'{stage}.sh').write_text(shell,encoding='utf-8',newline='\n')
    print('FIXED VISUAL CONTROL PROTOCOL',sha(PACK/'protocol.json'))


def package():
    pre=ROOT/'reports/bridge_visual_variation_local_preflight_20261016'
    report=read(pre/'report.json'); audit=read(pre/'independent_audit.json')
    assert report['execution_checks_passed'] and report['branch_restore_passed'] and audit['verified']
    assert audit['entry_sha256']==sha(PACK/'run_visual_control.py')
    assert audit['protocol_sha256']==sha(PACK/'protocol.json')
    (PACK/'local_preflight_summary.json').write_text(json.dumps(dict(passed=True,report_sha256=sha(pre/'report.json'),
        audit_sha256=sha(pre/'independent_audit.json'),elapsed_seconds=report['elapsed_seconds'],
        peak_reserved_gib=report['peak_reserved_gib'],neural_operational_updates=16,development_scored=False),indent=2)+'\n',encoding='utf-8',newline='\n')
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    for p in files:
        assert p.suffix in {'.py','.json','.sh','.md'}
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files),encoding='ascii',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[PACK/'SHA256SUMS']:z.write(p,p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(n)==(PACK.parent/n).read_bytes() for n in z.namelist())
    assert archive.stat().st_size<1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print('ZIP',archive.stat().st_size,'SHA256',sha(archive))


if __name__=='__main__':
    assert sys.argv[1:] in [['prepare'],['package']]
    globals()[sys.argv[1]]()
