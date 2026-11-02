"""Freeze and package one training-only image augmentation trial; no downloads."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_image_augmentation_v1'
LOCAL = ROOT/'reports/bridge_image_augmentation_local_preflight_20261016'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')


def prepare():
    assert not (PACK/'protocol.json').exists(), 'Never overwrite a frozen protocol'
    prior = read(ROOT/'cluster_packages/bridge_visual_variation_v1/protocol.json')
    for name, expected in prior['inherited_sha256'].items(): assert sha(PACK.parent/name) == expected, name
    assert sha(ROOT/'reports/bridge-state-fit-202278/real/report.json') == prior['plan']['control']['report_sha256']
    protocol = dict(date='2026-10-16', stage='single_training_image_augmentation_arm',
        objective='Improve independent-episode offline generalization. Training fitting is a secondary diagnostic only.',
        evidence_gap='Successful PushT checkpoint has crop_shape84x84 for96x96 images and eval_fixed_crop=true; adapted author Bridge reference203699 disabled crop_shape, and current project uses fixed view tokens. Training-time image augmentation has not been tested in the evidence index.',
        official=dict(commit='5ba07ac6661db573af695b419a7947ecb704690f',
            implementation='https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/vision/crop_randomizer.py',
            checkpoint_config_sha256=sha(ROOT/'reports/official_pusht_195527/checkpoint_config.yaml'),
            adapted_bridge_config_sha256=sha(ROOT/'cluster_packages/bridge_author_reference_v1/training_config.yaml'),
            distinction='This conservative CLIP-compatible pad/shift is inspired by the train-random/eval-fixed principle, not exact reproduction of the84/96 crop or proof that missing augmentation is the root cause.'),
        inherited_sha256=prior['inherited_sha256'], manifest_sha256=prior['plan']['manifest_sha256'],
        control=prior['plan']['control'],
        input=dict(train_windows=128, train_episodes=64, development_windows=40, development_episodes=10,
            intervention='On normalized224x224 training RGB only: edge-replicate pad8 then crop224x224 with independent per-row integer top-left offsets0..16 inclusive. Fixed8-pixel strength; no interpolation, flips, color changes or label/action/state transforms. Border content can be removed by at most8 pixels.',
            augmentation_seed='CPU-local torch.Generator seed510000+update, independent of unchanged dropout430000+update and diffusion420000+update seeds.',
            paired_context='Both repeated-current Adapter calls receive the same augmented token sequence for that optimization row.',
            evaluation='Use original unshifted cached visual tokens, unchanged language and state. Center offset8 exactly reproduces original pixels; check online center encoding against original cache on4 training rows before updates.',
            no_leakage='Only indices0..127 enter optimization augmentation. No train statistics from development observations or targets.'),
        unchanged='CLIP frozen in eval mode; same8layerAdapter,history and real-state ports,multiscale x0-DDPM100,actions,losses,gripper frozen,raw128/40 split and model initialization. Original control202278 reused without retraining.',
        recipe=prior['plan']['recipe'], screen=prior['plan']['screen'],
        primary='Original40-window path position AND rotation, all10 episodes and5 tasks; also report endpoint, episode-equal and per-task summaries. State-only ridge6.2798cm remains an auxiliary position reference, not a matched neural control.',
        preflight='16 optimization operations:4 common+4constant+4restored-same-LR+4reduced; four fixed TRAIN windows[0,32,64,96] only. Confirm checkpoint branch restore, RNG isolation, frozen CLIP and unchanged targets/eval cache. No development scoring, no millimetre fitting requirement.',
        formal='One8192-update candidate with original512exposures/window; final original128/40 predictions and one nonCLIP checkpoint only. No inner selection, no old-control retraining.',
        resources='Reuse original token cache, existing models/data/environment. Augmented training views encoded online in batches2; retain frozen visual encoder on GPU. No multiview cache or downloads. Formal allowed only after preflight peak<7.5GiB and conservative time estimate<9000s. One checkpoint<1GiB; retain15GiB planning headroom within200GiB.',
        stopping='One strength, initialization and budget. No additional crops, color changes, seeds or fitting-driven extensions if primary screen fails. Close prior mean-visual, prior-residual and selection candidates.',
        interpretation='Positive paired primary results would support this fixed augmentation recipe, not uniquely prove a module bug. Negative results close this trial; do not claim all augmentation ineffective.',
        limits=['Repeatedly used40-window development set, not an untouched test or new-task zero shot.',
                'Gripper frozen; no grasp success, object localization or online robot evaluation.',
                'One training initialization; three sampling seeds are not three independent training runs.'])
    write(PACK/'protocol.json', protocol)
    for stage in ['preflight', 'fit']:
        script = f'''#!/bin/bash
#SBATCH --job-name=bridge-aug-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage=='preflight' else '03:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-aug-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
'''
        if stage == 'fit': script += ': "${AUG_PREFLIGHT_JOB:?Supply successful preflight ID}"\n'
        script += '''PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_image_augmentation_v1
sha256sum -c SHA256SUMS
nvidia-smi
'''
        if stage == 'fit':
            script += '/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py "/projects/Zeroshot/runs/bridge-aug-preflight-${AUG_PREFLIGHT_JOB}"\n'
        command = ['/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_image_augmentation.py',
            '  --token-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt',
            '  --manifest ../bridge_growth128_v1/manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data',
            '  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1',
            '  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference',
            '  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json',
            '  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json',
            '  --state-arm real --state-protocol ../bridge_state_conditioning_v1/protocol.json',
            f'  --stage {stage} --output "/projects/Zeroshot/runs/bridge-aug-{stage}-${{SLURM_JOB_ID}}"']
        if stage == 'fit': command.append('  --preflight-reference "/projects/Zeroshot/runs/bridge-aug-preflight-${AUG_PREFLIGHT_JOB}"')
        script += (' '+chr(92)+'\n').join(command)+'\n'
        assert not any(line.startswith('+') for line in script.splitlines())
        (PACK/f'{stage}.sh').write_text(script, encoding='utf-8', newline='\n')
    print('AUGMENTATION PROTOCOL FROZEN', sha(PACK/'protocol.json'))


def package():
    report, audit = read(LOCAL/'report.json'), read(LOCAL/'independent_audit.json')
    assert report['execution_checks_passed'] and report['branch_restore_passed'] and audit['verified']
    assert audit['entry_sha256'] == sha(PACK/'run_image_augmentation.py')
    assert audit['protocol_sha256'] == sha(PACK/'protocol.json')
    assert report['augmentation_calls'] == 16 and report['original_evaluation_tokens_unchanged']
    assert report['peak_reserved_gib'] < 7.5 and report['estimated_formal_seconds'] < 9000
    write(PACK/'local_preflight_summary.json', dict(passed=True, report_sha256=sha(LOCAL/'report.json'),
        audit_sha256=sha(LOCAL/'independent_audit.json'), peak_reserved_gib=report['peak_reserved_gib'],
        elapsed_seconds=report['elapsed_seconds'], warm_update_seconds=report['warm_update_seconds'],
        conservative_formal_seconds=report['estimated_formal_seconds'], optimization_operations=16,
        development_scored=False, generalization_improvement_not_tested=True))
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    for p in files:
        assert p.suffix in {'.py', '.json', '.sh', '.md'}
        if p.suffix == '.py': ast.parse(p.read_text(encoding='utf-8'))
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='ascii', newline='\n')
    archive = PACK.with_suffix('.zip'); assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+[PACK/'SHA256SUMS']: z.write(p, p.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(z.read(n) == (PACK.parent/n).read_bytes() for n in z.namelist())
    assert archive.stat().st_size < 1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='ascii', newline='\n')
    print('ZIP', archive.stat().st_size, 'SHA256', sha(archive))


if __name__ == '__main__':
    assert sys.argv[1:] in [['prepare'], ['package']]
    globals()[sys.argv[1]]()
