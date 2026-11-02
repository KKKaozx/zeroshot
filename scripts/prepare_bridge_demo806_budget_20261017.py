"""Prepare one fixed-pool budget experiment; CPU metadata/source checks only."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'cluster_packages/bridge_demo806_budget_v1'
BASE = ROOT/'cluster_packages/bridge_demonstration_expansion_v1'
ORIGIN = ROOT/'reports/bridge-demo806-fit-208207'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')


def main():
    rows = read(BASE/'rows.json')['train']
    assert len(rows) == 3624 and read(ROOT/'reports/bridge_demo806_analysis_208207/analysis.json')['verified']
    with np.load(BASE/'schedule.npz') as z:
        original = z['batches'].copy()
    tasks = np.array([r['task'] for r in rows])
    original_labels = tasks[original]
    requested = original_labels[np.arange(16170) % 8192]
    # Continue the same per-task shuffled streams, rather than restarting the row stream.
    streams = {}
    cursors = Counter()
    for ti, task in enumerate(sorted(set(tasks))):
        indices = np.flatnonzero(tasks == task).tolist()
        needed = int((requested == task).sum())+8
        sequence, cycle = [], 0
        while len(sequence) < needed:
            order = indices.copy()
            random.Random(460001+ti*100000+cycle).shuffle(order)
            sequence.extend(order)
            cycle += 1
        streams[task] = sequence
    batches = np.empty((16170, 8), dtype=np.int64)
    for update, labels in enumerate(requested):
        used = set()
        for slot, task in enumerate(labels):
            sequence, cursor = streams[task], cursors[task]
            if sequence[cursor] in used:
                at = next(i for i in range(cursor+1, len(sequence)) if sequence[i] not in used)
                sequence[cursor], sequence[at] = sequence[at], sequence[cursor]
            batches[update, slot] = sequence[cursor]
            used.add(sequence[cursor]); cursors[task] += 1
    assert np.array_equal(batches[:8192], original)
    assert np.array_equal(tasks[batches], requested)
    assert all(len(set(batch)) == 8 for batch in batches)
    exposure = np.bincount(batches.ravel(), minlength=3624)
    schedule = HERE/'schedule.npz'
    if schedule.exists():
        with np.load(schedule) as z:
            assert np.array_equal(z['batches'], batches)
    else:
        np.savez_compressed(schedule, batches=batches)
    spec = importlib.util.spec_from_file_location('budget_source_factory', HERE/'run_budget.py')
    factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
    source = (ORIGIN/'effective_runner_snapshot.py').read_text()
    effective = factory.adapted_source(source)
    compile(effective, 'budget_effective_runner', 'exec')
    original_tree, effective_tree = ast.parse(source), ast.parse(effective)
    for name in ('advance', 'context', 'check'):
        get = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(get(original_tree), include_attributes=False) == ast.dump(get(effective_tree), include_attributes=False)
    report = read(ORIGIN/'report.json')
    protocol = dict(date='2026-10-17', stage='prepared_not_submitted',
        objective='On the fixed806-demo pool, does continuing the unchanged low-LR trajectory from8192 to16170 updates improve same-task held-out performance?',
        authorization='User accepted one finite training-budget comparison after208207 closure; this is a separate experiment with a new frozen protocol.',
        baseline_job='208207', baseline_report_sha256=sha(ORIGIN/'report.json'),
        baseline_effective_runner_sha256=sha(ORIGIN/'effective_runner_snapshot.py'),
        baseline_checkpoint_sha256=report['checkpoints'][0]['sha256'],
        baseline_development_npz_sha256=sha(ORIGIN/'reduced_u8192_development.npz'),
        original_input_protocol_sha256=sha(BASE/'protocol.json'),
        budget_effective_runner_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        schedule_sha256=sha(schedule), schedule_tensor_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
        recipe=dict(train_episodes=806, train_windows=3624, batch_size=8, initialization_seed=42,
                    updates=16170, prefix_updates=8192, extension_updates=7978,
                    learning_rate_policy='1e-4 through4096,1e-5 from4097 through16170; prefix8192 unchanged.',
                    prefix_schedule_exact=True, tail_task_labels='Repeat existing8192-step task-label schedule for7978steps; continue the original per-task window streams.',
                    state_statistics='Unchanged original128 training observations', sampling_steps=100,
                    sampling_seeds=[1101, 1102, 1103], architecture_loss_augmentation_gripper='All unchanged; gripper frozen'),
        exposure=dict(mean_8192=65536/3624, mean_16170=float(exposure.mean()),
                      reference_405_mean=65536/1836, min_16170=int(exposure.min()), max_16170=int(exposure.max()),
                      interpretation='Approximately matched average sample exposure; individual/task exposure is not equal. The causal comparison is extra low-LR optimization on the same806 pool, not a pure data-size test.'),
        tolerances=dict(parameters=1e-5, loss=1e-5, context=1e-4, prediction=1e-4),
        replay='No Adam state was saved in208207. Reconstruct its8192 updates; require all losses, final parameters and old4 predictions within frozen tolerances before any7978-step extension. Stop if mismatched; never reset Adam from weights and call that a continuation.',
        preflight='Complete existing input/cache identity; frozen208207 old4 replay;4 boundary-index warm updates,4 continuation updates and4 exact-restored continuation updates. No new candidate development score.',
        stopping='One fixed endpoint at16170. No intermediate candidate development selection, extra endpoints, seed sweep or automatic further extension.',
        development_rule='For evidence of a generalization gain over208207, require both window-weighted and episode-equal mean path position/rotation no higher, at least one strictly lower. Training metrics diagnostic only. Report all10 episodes/5 tasks without excluding failures.',
        qualification_for_future_test='Require the same four means no worse than frozen405 baseline207558 and all four strictly below the static baseline, in addition to gain over208207. Then freeze candidate/protocol before any new independent test; do not score reserved25 in this runner.',
        outcomes=dict(train_better_dev_not='Extra budget did not resolve generalization in this finite test; stop this budget branch, without claiming every possible budget is ruled out.',
                      train_and_dev_better='Evidence supporting optimization budget as a contributor, not proof of a unique root cause; future independent assessment still required.',
                      train_not_better='Do not infer that data is bad or only generalization is responsible; report optimization remains unresolved at this endpoint.'),
        untouched=['Original E dataset', 'All earlier packages/checkpoints', 'Consumed13 independent-test records', 'Reserved25 targets/images'],
        storage=dict(new_token_cache=False, source_data_export=False, cluster_new_budget_gib=5,
                     minimum_planning_headroom_gib=15, project_planning_limit_gib=200,
                     optimizer_checkpoints='Save model+Adam+RNG at8192 and16170, each<2GiB, cluster only. Final model<1GiB; do not download PT/cache.'),
        resource_gate=dict(peak_reserved_gib=12, estimated_formal_seconds=12600),
        local_actual_project_model_executed=False, local_gpu_executed=False,
        original_8192_experiment_changed=False,
        control_means=read(ROOT/'reports/bridge_demo806_analysis_208207/analysis.json')['summaries'])
    # Keep only aggregate control means, not duplicate all training episodes in this small package.
    protocol['control_means'] = {k: {g: v[g] for g in ('window_weighted', 'episode_equal')}
                                 for k, v in protocol['control_means'].items() if k in ('405_dev', '806_dev', 'static_dev')}
    write(HERE/'protocol.json', protocol)
    for stage, limit, name in [('preflight', '00:30:00', 'preflight.sh'), ('fit', '04:00:00', 'fit.sh')]:
        guard = '' if stage == 'preflight' else (
            ': "${BUDGET_PREFLIGHT_JOB:?Supply successful budget preflight job ID}"\n'
            '/projects/Zeroshot/envs/multitask-preflight-v1/bin/python validate_preflight.py '
            '"/projects/Zeroshot/runs/bridge-budget-preflight-${BUDGET_PREFLIGHT_JOB}"\n')
        script = f'''#!/bin/bash
#SBATCH --job-name=bridge-budget-{stage}
#SBATCH --account=msc
#SBATCH --qos=msc
#SBATCH --partition=cluster02
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22528M
#SBATCH --time={limit}
#SBATCH --chdir=/projects/Zeroshot
#SBATCH --output=/projects/Zeroshot/logs/bridge-budget-{stage}-%j.out
set -eo pipefail
source /projects/Zeroshot/scripts/ssd_cache.sh
export TMPDIR="/projects/Zeroshot/.tmp/${{SLURM_JOB_ID}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 OMP_NUM_THREADS=4
mkdir -p "$TMPDIR" /projects/Zeroshot/runs
cd /projects/Zeroshot/bridge_demo806_budget_v1
sha256sum --status -c SHA256SUMS
PROJECT_USED_BYTES=$(du -s -B1 /projects/_ssd/Zeroshot | cut -f1)
if (( PROJECT_USED_BYTES + 5*1024*1024*1024 + 15*1024*1024*1024 > 200*1024*1024*1024 )); then echo "Storage planning headroom insufficient; this is not a live quota query"; exit 1; fi
{guard}nvidia-smi
/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u run_budget.py \\
  --data-root /projects/Zeroshot/bridge_expansion_training_v1/data \\
  --data-reference /projects/Zeroshot/bridge_batch_input_audit_v1/reference \\
  --clip /projects/Zeroshot/baseline_setup/clip-vit-large-patch14-v1 \\
  --old-cache /projects/Zeroshot/bridge_training_coverage_v1/full_tokens.pt \\
  --cache /projects/Zeroshot/bridge_demonstration_expansion_v1/full_tokens.pt \\
  --control-report /projects/Zeroshot/runs/bridge-state-rep-fit-202463/real/report.json \\
  --preflight-reference /projects/Zeroshot/runs/bridge-demo806-preflight-208194 \\
  --stage {stage} --output "/projects/Zeroshot/runs/bridge-budget-{stage}-${{SLURM_JOB_ID}}"
'''
        (HERE/name).write_text(script, encoding='utf-8', newline='\n')
    validation = dict(passed=True, schedule_shape=list(batches.shape), prefix8192_exact=True,
                      all_task_labels_match_predeclared_schedule=True, no_duplicate_window_in_batch=True,
                      unchanged_function_asts=['advance', 'context', 'check'], executable_source_compiles=True,
                      actual_project_model_executed=False, actual_gpu_executed=False,
                      requires_cluster_optimizer_and_replay_preflight=True,
                      mean_window_exposure=float(exposure.mean()), task_draw_counts=dict(Counter(tasks[batches].ravel())),
                      local_free_bytes=shutil.disk_usage(ROOT).free)
    write(HERE/'local_validation.json', validation)
    print(json.dumps(dict(prepared=True, mean_exposure=protocol['exposure'], local_validation=validation), indent=2))


if __name__ == '__main__':
    main()
