"""Two fixed CPU ridge diagnostics of archived 806-episode decoder contexts.

Prepare freezes the protocol and source hashes without fitting. Run verifies it.
No Torch, encoder, neural training, reserved test, or new policy inference.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_806_context_readout_20261017'
ROWS = ROOT/'cluster_packages/bridge_demonstration_expansion_v1/rows.json'
STAGES = {
    '8192': ('bridge-demo806-fit-208207', 'bridge_demo806_analysis_208207'),
    '16170': ('bridge-budget-fit-208586', 'bridge_budget_analysis_208586'),
}


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def identity(row): return row['shard']+'::'+str(row['record_index'])


def prepare():
    assert not OUT.exists(), 'Protocol already exists; do not repeat this experiment.'
    sources = [Path(__file__), ROWS,
               ROOT/'scripts/check_bridge_context_only_readout_20261016.py',
               ROOT/'reports/bridge_context_only_readout_20261016/report.json']
    for update, (folder, audit_name) in STAGES.items():
        ap = ROOT/'reports'/audit_name/'analysis.json'
        audit = read(ap)
        assert audit['verified']
        sources.append(ap)
        for split in ('train', 'development'):
            p = ROOT/'reports'/folder/f'reduced_u{update}_{split}.npz'
            assert sha(p) == audit['downloaded_files'][p.name]['sha256']
            sources.append(p)
    assert sha(ROWS) == '4dbcb6b7e89110950a2d301d6ce9f9250004fe916ffdf23db5dc067121b6ce59'
    plan = {
        'question': 'Do the saved decoder contexts of the 806-episode models permit lower held-out XYZ error under one fixed global linear readout, and how does this change after extra training?',
        'difference_from_prior': 'Prior context-only readout used 128 training windows of model202278. This uses the current3624 training windows/806 episodes and both8192/16170 archived training stages; no prior128 fit is rerun.',
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources},
        'algorithm': 'Exactly the prior global ridge objective: lambda1.0, train-only feature centering, ONE scalar sqrt(mean squared centered row norm), train-only target centering. Primal1024x1024 solve; independently reconstruct with symmetric eigenvectors. No per-axis feature scaling, task-specific map, extra raw inputs, or sweep.',
        'targets': '16 future XYZ targets (48 outputs), centimeters = saved target XYZ *10. Same3624 train and40 development windows for both stages.',
        'context_scope': '1024D actual decoder conditioning, including Adapter/history/state projection; not isolated Adapter output.',
        'evaluation': 'Position only. Window-weighted and episode-equal path/endpoint errors. Native DDPM comparator is mean error of3 saved samples, not error of averaged predictions. Retain all5 tasks and10 development episodes.',
        'positive_gate_per_stage': 'Both window-weighted AND episode-equal path position error below static AND corresponding native DDPM; at least6/10 episodes below static. Endpoint is descriptive, not an additional chosen gate.',
        'interpretation': ['Positive: a usable XYZ mapping exists under a different capacity/objective; no unique causal verdict about a faulty module.', 'Negative: this fixed linear map is insufficient; does not prove information is absent or Adapter is broken.', 'Two stages have different learned contexts; representation and readout changes are descriptive, not an isolated architecture intervention.'],
        'limitations': ['Reused development set, not a new final test or nested cross-validation.', 'No rotation, gripper, object localization, closed-loop simulation or grasp-success evaluation.', 'No deployment claim or selection between a linear policy and diffusion. Diffusion remains the project method.'],
        'stopping': 'Exactly two fixed fits, one per saved stage. No new neural model loads, image encoding, GPU/cluster work, additional datasets, parameter search or reserved-test reads.',
    }
    OUT.mkdir()
    write(OUT/'plan.json', plan)
    print('PROTOCOL FROZEN', sha(OUT/'plan.json'), flush=True)


def summarize(error, rows):
    episodes, tasks = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        episodes[identity(row)].append(i)
        tasks[row['task']].append(i)
    def subset(at):
        return dict(path_position_cm=float(error[at].mean()),
                    endpoint_position_cm=float(error[at, -1].mean()))
    ep = [dict(episode=k, task=rows[v[0]]['task'], windows=len(v), **subset(v))
          for k, v in episodes.items()]
    return dict(windows=len(rows), episodes=len(ep), window_weighted=subset(list(range(len(rows)))),
                episode_equal={key: float(np.mean([r[key] for r in ep]))
                               for key in ('path_position_cm', 'endpoint_position_cm')},
                per_episode=ep,
                per_task=[dict(task=k, windows=len(v), **subset(v)) for k, v in tasks.items()],
                per_target_position_cm=error.mean(0).tolist())


def run():
    assert (OUT/'plan.json').exists() and not (OUT/'report.json').exists(), 'Prepare first; completed fits must not be repeated.'
    start = time.monotonic()
    plan = read(OUT/'plan.json')
    for name, expected in plan['sources'].items():
        assert sha(ROOT/name) == expected, name
    rows = read(ROWS)
    assert len(rows['train']) == 3624 and len(rows['development']) == 40
    assert len(set(map(identity, rows['train']))) == 806
    assert len(set(map(identity, rows['development']))) == 10
    assert not set(map(identity, rows['train'])) & set(map(identity, rows['development']))
    results, saved, ep_csv, task_csv, win_csv = {}, {}, [], [], []
    first_targets = {}
    for update, (folder, _) in STAGES.items():
        arrays = {}
        for split, count, first in [('train', 3624, 0), ('development', 40, 3624)]:
            with np.load(ROOT/'reports'/folder/f'reduced_u{update}_{split}.npz', allow_pickle=False) as z:
                a = {k: z[k].copy() for k in ('contexts', 'targets', 'predictions', 'row_indices')}
            assert a['contexts'].shape == (count, 1024)
            assert a['targets'].shape == (count, 16, 8)
            assert a['predictions'].shape == (3, count, 16, 8)
            assert np.array_equal(a['row_indices'], np.arange(first, first+count))
            assert all(np.isfinite(v).all() for v in a.values())
            if split in first_targets: assert np.array_equal(a['targets'], first_targets[split])
            else: first_targets[split] = a['targets'].copy()
            arrays[split] = a
        x = arrays['train']['contexts'].astype(np.float64)
        center = x.mean(0)
        scale = float(np.sqrt(np.square(x-center).sum(1).mean()))
        assert scale > 1e-9
        x = (x-center)/scale
        y = arrays['train']['targets'][..., :3].astype(np.float64).reshape(3624, 48)*10
        ybar = y.mean(0)
        gram = x.T@x
        rhs = x.T@(y-ybar)
        weights = np.linalg.solve(gram+np.eye(1024), rhs)
        eigen, vectors = np.linalg.eigh(gram)
        weights_alt = vectors@((vectors.T@rhs)/(eigen[:, None]+1.))
        normal_residual = float(np.max(np.abs((gram+np.eye(1024))@weights-rhs)))
        assert normal_residual < 1e-8
        stage, max_solver = {}, 0.
        for split, count in [('train', 3624), ('development', 40)]:
            q = (arrays[split]['contexts'].astype(np.float64)-center)/scale
            p = (ybar+q@weights).reshape(count, 16, 3)
            alternate = (ybar+q@weights_alt).reshape(count, 16, 3)
            max_solver = max(max_solver, float(np.max(np.abs(p-alternate))))
            assert max_solver < 1e-8
            truth = arrays[split]['targets'][..., :3].astype(np.float64)*10
            metrics = {
                'linear': np.linalg.norm(p-truth, axis=-1),
                'static': np.linalg.norm(truth, axis=-1),
                'native_ddpm': np.linalg.norm(arrays[split]['predictions'][..., :3].astype(np.float64)*10-truth[None], axis=-1).mean(0),
            }
            stage[split] = {label: summarize(e, rows[split]) for label, e in metrics.items()}
            saved[f'u{update}_{split}_prediction_cm'] = p
            for label, s in stage[split].items():
                ep_csv.extend(dict(update=update, split=split, method=label, **r) for r in s['per_episode'])
                task_csv.extend(dict(update=update, split=split, method=label, **r) for r in s['per_task'])
            for i, row in enumerate(rows[split]):
                win_csv.append(dict(update=update, split=split, episode=identity(row), task=row['task'], start_index=row['start_index'],
                                    **{f'{k}_path_cm': float(v[i].mean()) for k, v in metrics.items()}))
        dev = stage['development']
        passes = {weight: all(dev['linear'][weight]['path_position_cm'] < dev[b][weight]['path_position_cm']
                              for b in ('static', 'native_ddpm'))
                  for weight in ('window_weighted', 'episode_equal')}
        comparison = {}
        for base in ('static', 'native_ddpm'):
            comparison[base] = sum(a['path_position_cm'] < b['path_position_cm']
                                   for a, b in zip(dev['linear']['per_episode'], dev[base]['per_episode']))
        results[update] = dict(summaries=stage, positive_position_boundary=all(passes.values()) and comparison['static'] >= 6,
                               mean_gates=passes, development_episodes_better_than=comparison,
                               eigen_prediction_max_difference_cm=max_solver, normal_equation_max_residual=normal_residual)
        saved[f'u{update}_weights'] = weights
        saved[f'u{update}_center'] = center
        saved[f'u{update}_scale'] = np.array(scale)
        saved[f'u{update}_target_center_cm'] = ybar
        print('READOUT_STAGE', update, json.dumps({k: v['episode_equal'] for k, v in dev.items()}), 'gate', results[update]['positive_position_boundary'], flush=True)
    np.savez_compressed(OUT/'predictions_and_coefficients.npz', **saved)
    for name, records in [('episodes.csv', ep_csv), ('tasks.csv', task_csv), ('windows.csv', win_csv)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    report = dict(verified=True, plan_sha256=sha(OUT/'plan.json'), source_sha256=sha(Path(__file__)),
                  results=results, identical_targets_between_stages=True,
                  elapsed_seconds=time.monotonic()-start, neural_training=False,
                  gpu_used=False, new_model_inference=False, reserved_test_read=False,
                  output_sha256={n: sha(OUT/n) for n in ('predictions_and_coefficients.npz', 'episodes.csv', 'tasks.csv', 'windows.csv')},
                  limitations=plan['limitations'], interpretation=plan['interpretation'])
    write(OUT/'report.json', report)
    print('READOUT COMPLETE', OUT/'report.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run()
