"""Freeze a single resumed806-pool update-mask intervention; no model execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_fixed_context_v1'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert not (PACK/'protocol.json').exists(), 'Frozen experiment already prepared.'
    r = read(ROOT/'reports/bridge-budget-fit-208586/report.json')
    audit = read(ROOT/'reports/bridge_budget_analysis_208586/analysis.json')
    readout = read(ROOT/'reports/bridge_806_context_readout_20261017/report.json')
    assert audit['verified'] and readout['verified'] and all(v['positive_position_boundary'] for v in readout['results'].values())
    spec = importlib.util.spec_from_file_location('fixed_wrapper', PACK/'run_fixed_context.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = ROOT/'reports/bridge-demo806-fit-208207/effective_runner_snapshot.py'
    effective = module.adapted_source(original.read_text())
    old, new = ast.parse(original.read_text()), ast.parse(effective)
    def find(tree, name): return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('context', 'check'):
        assert ast.dump(find(old, name)) == ast.dump(find(new, name)), name
    # Remove exactly the two deliberate callbacks; all remaining update AST matches.
    advance = find(new, 'advance')
    removed = []
    for node in ast.walk(advance):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                clean = []
                for child in value:
                    if (isinstance(child, ast.Expr) and isinstance(child.value, ast.Call)
                        and isinstance(child.value.func, ast.Attribute) and isinstance(child.value.func.value, ast.Name)
                        and child.value.func.value.id == 'FIXED'):
                        removed.append(child.value.func.attr)
                    else: clean.append(child)
                setattr(node, field, clean)
    assert removed == ['mask_updates', 'after_step']
    assert ast.dump(advance) == ast.dump(find(old, 'advance'))
    checkpoint = r['optimizer_checkpoints'][0]
    assert checkpoint['update'] == 8192 and checkpoint['roundtrip_exact']
    budget = read(ROOT/'cluster_packages/bridge_demo806_budget_v1/protocol.json')
    p = dict(date='2026-10-17', status='prepared_not_submitted',
        objective='On the current806-demo pool, does blocking context-module updates during the SAME7978-step continuation improve independent-development generalization compared with the completed joint continuation208586?',
        authorization='User approved the next fixed-context diffusion-head comparison after the806 context-readout diagnosis.',
        difference_from_old='Prior fixed-context runs were single-task/other architectures or small-sample fitting. This is the current806-episode generalization comparison, reusing the existing8192 full Adam checkpoint and completed joint control. No prefix retraining or repeated joint control.',
        single_intervention='Mask Adapter/history/state gradients toNone AFTER the unchanged global clipping and immediately BEFORE Adam.step. Only diffusion_decoder parameters and Adam moments advance. Context gradients are computed for clipping parity, but never used to update context parameters or moments.',
        fixed_context_definition='Fixed context-generating parameters, not a new deterministic evaluation-vector training dataset. Preserve train-mode context dropout, seeds, context calls, noise schedule, targets, loss and clipping. Eval contexts must replay8192 saved contexts on all3664 rows.',
        original_runner_sha256=sha(original), effective_runner_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        joint_report_sha256=sha(ROOT/'reports/bridge-budget-fit-208586/report.json'),
        original_budget_protocol_sha256=r['budget']['protocol_sha256'],
        prefix_file='runs/bridge-budget-fit-208586/training_state_u08192.pt',
        prefix_file_sha256=checkpoint['sha256'], prefix_parameters_sha256=checkpoint['parameters_sha256'],
        prefix_optimizer_sha256=checkpoint['optimizer_sha256'],
        old_development_sha256=budget['baseline_development_npz_sha256'],
        old_arrays_sha256={s: read(ROOT/'reports/bridge_demo806_analysis_208207/analysis.json')['downloaded_files'][f'reduced_u8192_{s}.npz']['sha256'] for s in ('train', 'development')},
        schedule_sha256=budget['schedule_sha256'], schedule_tensor_sha256=budget['schedule_tensor_sha256'],
        joint_extension_log_sha256=audit['downloaded_files']['extension_updates.jsonl']['sha256'],
        recipe=dict(train_episodes=806, train_windows=3624, development_episodes=10, development_windows=40,
                    batch=8, resume_update=8192, final_update=16170, new_updates=7978, learning_rate=1e-5,
                    sampling_steps=100, sampling_seeds=[1101,1102,1103], updated_parameters=77670023,
                    image_augmentation=False, gripper_updated=False, encoder_dropout_changed=False),
        preflight='Restore real8192 model+Adam; old4 replay;4 joint steps against existing saved logs; restore;4 masked steps; restore;repeat4 masked steps exactly. First masked decoder update must match joint. Context parameters,buffers and Adam unchanged. Total12 updates, no complete development scoring.',
        formal='Restore same8192 checkpoint directly. Exactly7978 masked updates8193..16170, evaluate full training pool and all40 development windows once at final endpoint. No prefix reconstruction, no complete joint retraining, no intermediate development selection.',
        tolerances=dict(context=1e-4, prediction=1e-4, loss=1e-5),
        primary_gate='Against208586 joint16170: BOTH window-weighted and episode-equal path position/rotation no higher; at least one strictly lower. No new gate based on training fit. Report endpoints and all10 episodes/5 tasks including failures.',
        secondary_comparators='Original8192 and405 references and static descriptive only; fixed readout is not a replacement policy. Separate position-only gains from overall position/rotation gains.',
        stopping='One finite endpoint, one preflight, no parameter/seed/budget sweep, no automatic rerun after integrity failure, no consumed13 or reserved25 test reads. No new independent test unless a later candidate/protocol is frozen.',
        interpretation_positive='Consistent development improvement supports context update dynamics participating in this setting; does not prove an Adapter implementation bug or successful robot operation.',
        interpretation_negative='This particular intervention does not improve generalization; stop this branch, do not infer information absence or automatically extend training.',
        limits=['Repeatedly used development set; not fresh-test evidence.', 'Context module gradients are computed to preserve global clipping, so compute cost need not fall.', 'Freezing after8192 does not isolate the representation learned before8192 or prove a unique module root cause.', 'Gripper frozen; no simulation or task success evaluation.'],
        evaluation_allowance_seconds=4000.0,
        storage=dict(new_token_cache=False, data_export=False, cluster_new_planning_gib=4,
                     final_model_and_adam='Save trainable model and full Adam/RNG checkpoint on cluster only; do not download PT/cache.',
                     project_planning_limit_gib=200, minimum_headroom_gib=15),
        controls=dict(joint16170=audit['summaries']['development'],
                      original8192=audit['baseline_summaries']['806_dev'], static=audit['baseline_summaries']['static_dev']),
        local_package_sha256={n:sha(PACK/n) for n in ('run_fixed_context.py','fixed_hooks.py','validate_preflight.py')})
    p['resource_estimate_note'] = 'Conservative4000s evaluation/loading allowance plus900s and1.3x measured7978-step updates; require total<4h, reserve4h.'
    write(PACK/'protocol.json', p)
    (PACK/'effective_runner.py.txt').write_bytes(effective.encode())
    for stage in ('preflight', 'fit'):
        guard = '' if stage == 'preflight' else ': "${FIXED_PREFLIGHT_JOB:?Supply verified preflight job ID}"\n/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py "/projects/Zeroshot/runs/bridge-fixed-preflight-${FIXED_PREFLIGHT_JOB}"\n'
        script = f'''#!/bin/bash
#SBATCH --job-name=bridge-fixed-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={'00:30:00' if stage == 'preflight' else '04:00:00'}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-fixed-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_fixed_context_v1
sha256sum --status -c SHA256SUMS
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 4*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient; not a live quota query"; exit 1; fi
{guard}nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_fixed_context.py \\
  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --old-cache /projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt \\
  --cache /projects/Zeroshot/bridge_demonstration_expansion_v1/full_tokens.pt \\
  --control-report /projects/Zeroshot/runs/bridge-state-rep-fit-202463/real/report.json \\
  --preflight-reference /projects/Zeroshot/runs/bridge-demo806-preflight-208194 \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-fixed-{stage}-${{SLURM_JOB_ID}}"
'''
        (PACK/(stage+'.sh')).write_bytes(script.encode())
    write(ROOT/'reports/bridge_fixed_context_preparation_20261017.json', dict(prepared=True, submitted=False,
        project_model_executed=False, gpu_used=False, prefix_retraining=False,
        source_ast_context_check_unchanged=True, source_ast_update_only_two_callbacks=True,
        protocol_sha256=sha(PACK/'protocol.json'), effective_runner_sha256=p['effective_runner_sha256']))
    print('FIXED-CONTEXT PROTOCOL PREPARED', sha(PACK/'protocol.json'))


if __name__ == '__main__': main()
