"""Export the existing state readout and freeze one Bridge residual-DDPM trial."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_state_prior_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def prepare():
    import sys
    sys.path.insert(0, str(PACK))
    from prior_contract import predict_prior, rotation_features
    assert not (PACK/'protocol.json').exists(), 'Do not recreate frozen protocol.'
    baseline = ROOT/'reports/bridge_generalization_readout_20261015'
    assert read(baseline/'independent_audit.json')['verified']
    base = ROOT/'cluster_packages/bridge_growth128_v1'
    manifest = read(base/'manifest.json')
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz') as z:
        observed = z['current_pose'].astype(float)
    with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:
        y = z['targets'][..., :3].astype(float)
    rows = manifest['train']+manifest['development']
    tasks = np.array([r['task'] for r in rows])
    rot = rotation_features(observed)
    spec = dict(fit_rows=list(range(128)), ridge_lambda=1., output_units='0.1m tool-relative XYZ',
                inputs='Current observed XYZ/RPY and instruction task only; no image/future/episode id.', tasks={})
    for task in sorted(set(tasks[:128])):
        ix = np.flatnonzero(tasks[:128] == task)
        xyz = observed[ix, :3]; rotation = rot[ix]
        xm, rm = xyz.mean(0), rotation.mean(0)
        xs = max(float(np.sqrt(np.square(xyz-xm).sum(1).mean())), 1e-12)
        rs = max(float(np.sqrt(np.square(rotation-rm).sum(1).mean())), 1e-12)
        x = np.concatenate([(xyz-xm)/xs, (rotation-rm)/rs], 1)/np.sqrt(2)
        target = y[ix].reshape(len(ix), 48); intercept = target.mean(0)
        weights = x.T@np.linalg.solve(x@x.T+np.eye(len(ix)), target-intercept)
        spec['tasks'][task] = dict(xyz_mean=xm.tolist(), xyz_scale=xs,
            rotation_mean=rm.tolist(), rotation_scale=rs, weights=weights.tolist(), intercept=intercept.tolist())
    (PACK/'prior.json').write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8')
    values = predict_prior(observed, tasks, spec)
    with np.load(baseline/'predictions.npz') as z:
        saved = np.concatenate([z['train_resubstitution__state_only'],z['development__state_only']])/10
    delta = float(np.abs(values-saved).max())
    assert delta < 1e-6
    train_residual = y-values[:128]
    assert np.abs(train_residual).max() <= 3., 'Stop: residual clipping would exclude training labels.'
    inherited = {f'bridge_growth128_v1/{n}':sha(base/n) for n in manifest['source_sha256']}
    inherited['bridge_growth128_v1/manifest.json'] = sha(base/'manifest.json')
    state = ROOT/'cluster_packages/bridge_state_conditioning_v1'
    inherited.update({f'bridge_state_conditioning_v1/{n}':sha(state/n) for n in ['fit_state_arm.py','state_input_contract.py','protocol.json','current_pose.npz']})
    protocol = dict(stage='one_fixed_observed_state_prior_diffusion_residual_candidate', date='2026-10-15',
        manifest_sha256=sha(base/'manifest.json'), inherited_sha256=inherited, prior_sha256=sha(PACK/'prior.json'),
        baseline_analysis_sha256=sha(baseline/'independent_audit.json'), prior_export_max_difference_units=delta,
        all_prior_float32_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        train_residual_max_abs_units=float(np.abs(train_residual).max()),
        only_change='Train XYZ residual=(original tool-relative XYZ - fixed observed-state prior); add the same prior after native DDPM100 and before original-unit evaluation; quaternion targets unchanged.',
        preserved=['Same128 train/40 development windows and64/10 independent demonstrations',
            'SameCLIP/8layerAdapter/history and real-state projection', 'Same initial model43, Adam, gradient clipping and original8192 batch/noise seeds',
            'Same4096 updates at1e-4 then4096 at1e-5', 'Same nativeDDPM100 and3 sampling seeds; frozen gripper'],
        train_prior_statistics_only=True, use_cross_fitted_prior_for_neural_training=False,
        prior_training_interpretation='Prior fitted on same training split; in-sample prior for train, fixed extrapolated prior for development. No future observation supplied as input.',
        clipping='Same +/-3 project units applied to sampled residual XYZ, then +/-3 to reconstructed XYZ. Report final clipping counts; no post-hoc bound changes.',
        preflight='4 common updates +4 constant +4 restored-same-LR replay +4 reduced-LR updates; four fixed training windows for execution checks, no development scoring, no checkpoint.',
        formal='One new8192-update arm only. Reuse202278 real-state raw control; final128/40 offline arrays and one trainable checkpoint. No mid-run development selection.',
        primary='Original40-window path position and rotation after reconstruction; per-demo means secondary; original train pose gate a quality constraint, not main objective.',
        candidate_screen=dict(position_below_static_cm=8.21232607389037, position_below_original_cm=8.349126682028293,
            rotation_no_worse_than_original_deg=15.256048338593287+.02, episodes_position_better_than_original_at_least=6,
            preserve_original_training_pose_gate=True),
        stronger_reference='Also report distance to fixed state-only6.279815835cm baseline; improvement over old DDPM alone does not show added value over the prior.',
        stopping='One fixed budget, final weights only. No extra seeds/updates/prior strength/mixture or regularization sweeps. Failure or partial benefit closes this candidate.',
        reserved_test_used=False, physical_rollout=False, gripper_claim=False,
        storage='One shared full token cache<256MiB and one final trainable checkpoint<1GiB; no optimizer/intermediate checkpoints; no model/data downloads.')
    (PACK/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    for stage in ['preflight','fit']:
        extra = '' if stage == 'preflight' else ': "${PRIOR_PREFLIGHT_JOB:?Supply successful preflight ID}"\n'
        reference = '' if stage == 'preflight' else ' \\\n  --preflight-reference "/projects/Zeroshot/runs/bridge-prior-preflight-${PRIOR_PREFLIGHT_JOB}"'
        shell = f'''#!/bin/bash
#SBATCH --job-name=bridge-prior-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage=='preflight' else '03:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-prior-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
{extra}PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 2*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient"; exit 1; fi
cd /projects/Zeroshot/bridge_state_prior_v1
sha256sum -c SHA256SUMS
nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_prior.py \\
  --token-cache /projects/Zeroshot/bridge_state_prior_v1/full_tokens.pt \\
  --manifest ../bridge_growth128_v1/manifest.json --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --origin-report /projects/Zeroshot/runs/bridge-trace-fit-200039/report.json \\
  --retained-report /projects/Zeroshot/runs/bridge-growth64-fit-200600/report.json \\
  --state-arm real --state-protocol ../bridge_state_conditioning_v1/protocol.json \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-prior-{stage}-${{SLURM_JOB_ID}}"{reference}
'''
        assert not any(line.startswith('+') for line in shell.splitlines())
        (PACK/f'{stage}.sh').write_text(shell, encoding='utf-8', newline='\n')
    print(json.dumps({'prior_export_max_difference_units':delta,'train_residual_max_abs_units':protocol['train_residual_max_abs_units']}, indent=2))


def package():
    files = sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    assert {p.suffix for p in files} <= {'.py','.json','.sh','.md'}
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='ascii')
    archive = PACK.with_suffix('.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+[PACK/'SHA256SUMS']:
            z.write(p, f'{PACK.name}/{p.name}')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    assert archive.stat().st_size < 1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='ascii')
    print('ARCHIVE', archive, archive.stat().st_size, sha(archive))


if __name__ == '__main__':
    import sys
    assert sys.argv[1:] in [['prepare'], ['package']]
    globals()[sys.argv[1]]()
