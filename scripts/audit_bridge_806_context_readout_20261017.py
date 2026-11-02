"""Reconstruct stored readout outputs and metrics without solving or fitting again."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bridge_806_context_readout_20261017'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert not (OUT/'independent_saved_array_audit.json').exists(), 'Audit already complete.'
    report, plan = read(OUT/'report.json'), read(OUT/'plan.json')
    rows = read(ROOT/'cluster_packages/bridge_demonstration_expansion_v1/rows.json')
    assert sha(OUT/'plan.json') == report['plan_sha256']
    for name, expected in plan['sources'].items(): assert sha(ROOT/name) == expected
    for name, expected in report['output_sha256'].items(): assert sha(OUT/name) == expected
    groups = defaultdict(list)
    for i, row in enumerate(rows['development']): groups[(row['shard'], row['record_index'])].append(i)
    audit = dict(verified=True, no_additional_fit=True, checks={}, posthoc_descriptive_sampling_mean={},
                 scope='Saved-array audit only. Sample mean comparator added after fixed readout results to check sampling variance; not a preregistered gate or proposed executable policy.')
    with np.load(OUT/'predictions_and_coefficients.npz', allow_pickle=False) as saved:
        for update, folder in [('8192', 'bridge-demo806-fit-208207'), ('16170', 'bridge-budget-fit-208586')]:
            arrays = {}
            for split in ('train', 'development'):
                with np.load(ROOT/'reports'/folder/f'reduced_u{update}_{split}.npz', allow_pickle=False) as z:
                    arrays[split] = {k: z[k] for k in ('contexts', 'targets', 'predictions')}
            x = arrays['train']['contexts'].astype(float)
            y = arrays['train']['targets'][..., :3].astype(float).reshape(len(x), -1)*10
            center, scale = saved[f'u{update}_center'], saved[f'u{update}_scale']
            w, yc = saved[f'u{update}_weights'], saved[f'u{update}_target_center_cm']
            assert np.max(abs(center-x.mean(0))) < 1e-12
            assert abs(float(scale)-np.sqrt(((x-center)**2).sum(1).mean())) < 1e-12
            assert np.max(abs(yc-y.mean(0))) < 1e-12
            a = (x-center)/scale
            stationarity = float(np.max(abs(a.T@(a@w+yc-y)+w)))
            assert stationarity < 1e-8
            for split in ('train', 'development'):
                query = (arrays[split]['contexts'].astype(float)-center)/scale
                reconstructed = (query@w+yc).reshape(-1, 16, 3)
                delta = float(np.max(abs(reconstructed-saved[f'u{update}_{split}_prediction_cm'])))
                assert delta < 1e-10
                error = np.sqrt(((reconstructed-arrays[split]['targets'][..., :3].astype(float)*10)**2).sum(-1))
                s = report['results'][update]['summaries'][split]['linear']
                assert abs(error.mean()-s['window_weighted']['path_position_cm']) < 1e-10
                if split == 'development':
                    assert abs(np.mean([error[ix].mean() for ix in groups.values()])-s['episode_equal']['path_position_cm']) < 1e-10
            truth = arrays['development']['targets'][..., :3].astype(float)*10
            native = arrays['development']['predictions'][..., :3].astype(float)*10
            mean_error = np.linalg.norm(native.mean(0)-truth, axis=-1)
            linear_error = np.linalg.norm(saved[f'u{update}_development_prediction_cm']-truth, axis=-1)
            audit['checks'][update] = dict(normal_equation_stationarity_max_abs=stationarity,
                stored_predictions_reconstruction_max_abs_cm=delta,
                train_only_centers_and_scalar_scale_recomputed=True, development_weighting_recomputed=True)
            audit['posthoc_descriptive_sampling_mean'][update] = dict(
                window_path_cm=float(mean_error.mean()),
                episode_equal_path_cm=float(np.mean([mean_error[ix].mean() for ix in groups.values()])),
                linear_episodes_better_than_native_sample_mean=int(sum(linear_error[ix].mean() < mean_error[ix].mean() for ix in groups.values())),
                per_task={task: dict(native_sample_mean_cm=float(mean_error[[i for i, row in enumerate(rows['development']) if row['task'] == task]].mean()),
                                    linear_cm=float(linear_error[[i for i, row in enumerate(rows['development']) if row['task'] == task]].mean()))
                          for task in sorted(set(row['task'] for row in rows['development']))})
    audit['report_sha256'] = sha(OUT/'report.json')
    audit['audit_script_sha256'] = sha(Path(__file__))
    (OUT/'independent_saved_array_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
