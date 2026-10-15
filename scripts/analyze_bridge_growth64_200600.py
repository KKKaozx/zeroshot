"""Audit the saved 64-window experiment without loading a model or training."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_bridge_training_trace_200039 import errors, FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/bridge-growth64-fit-200600'
PACK = ROOT / 'cluster_packages/bridge_growth64_v1'
OUT = ROOT / 'reports/bridge_growth64_analysis_200600_20261013'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(pred, target):
    p, r = errors(pred, target)
    good = ((p.mean(-1) <= .5) & (r.mean(-1) <= 5)).all(0)
    gate = bool(good.mean() >= .9 and (p.mean((1, 2)) <= .5).all()
                and (r.mean((1, 2)) <= 5).all() and (p[..., -1].mean(1) <= 1).all()
                and (r[..., -1].mean(1) <= 10).all())
    return dict(zip(FIELDS, map(float, (p.mean(), r.mean(), p[..., -1].mean(), r[..., -1].mean())))), good, gate, p, r

def main():
    global DATA, PACK, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--package', type=Path, default=PACK)
    ap.add_argument('--output', type=Path, default=OUT)
    ap.add_argument('--initialization-seed', type=int, default=43)
    ap.add_argument('--baseline-data', type=Path)
    args = ap.parse_args()
    DATA, PACK, OUT = args.data, args.package, args.output
    init = args.initialization_seed
    report = json.loads((DATA/'report.json').read_text())
    manifest = json.loads((DATA/'plan.json').read_text())['manifest']
    origin = json.loads((ROOT/'reports/bridge-trace-fit-200039/report.json').read_text())
    assert report['stage'] == 'fit' and report['initialization_seed'] == init
    assert manifest['initialization_seeds'] == [init] and manifest['late_lr']['initialization_seed'] == init
    assert report['manifest_sha256'] == sha(PACK/'manifest.json')
    assert manifest == json.loads((PACK/'manifest.json').read_text())
    assert report['source_sha256'] == manifest['source_sha256']
    assert all(sha(PACK/n) == h for n, h in report['source_sha256'].items())
    assert sha(DATA/'runner_snapshot.py') == report['source_sha256']['fit_bridge_growth64.py']
    assert all(report[k] for k in ('execution_checks_passed', 'input_contract_passed', 'clip_unchanged', 'other_frozen_unchanged'))
    assert not any(report[k] for k in ('gripper_trained', 'simulation_executed', 'downloads', 'reserved_test_targets_read'))
    assert len(report['data_checks']) == len(report['input_numeric_comparisons']) == 104
    assert report['data_checks'][:72] == origin['data_checks']
    assert all(row['raw_identity_exact'] for row in report['input_numeric_comparisons'])
    assert all(d['max_abs'] <= d['atol'] for row in report['input_numeric_comparisons'] for d in row['differences'].values())
    assert report['initial_shared_sha256'] == next(r['initial_shared_sha256'] for r in origin['runs'] if r['initialization_seed'] == init)
    assert report['sampling_steps'] == 100 and report['sampling_seeds'] == [1101, 1102, 1103]
    assert report['branch_update'] == 2048 and report['final_update'] == 4096
    assert len(report['runs']) == 1
    run = report['runs'][0]
    assert run['arm'] == 'reduced' and run['learning_rate'] == 1e-5
    assert run['common_weights_sha256'] == report['common_state_sha256']
    assert run['common_adam_sha256'] == report['common_adam_sha256']
    assert run['actual_draw_slots_per_window'] == [256]*64
    assert run['history'][-1]['train'] == run['final_train']
    identity = lambda row: (row['shard'], row['record_index'])
    old, added, dev = (set(map(identity, rows)) for rows in (manifest['train'][:32], manifest['train'][32:], manifest['development']))
    assert (len(old), len(added), len(dev)) == (16, 16, 10)
    assert not old & added and not (old | added) & dev
    for name, lo, hi, rate in [('common', 1, 2048, 1e-4), ('reduced', 2049, 4096, 1e-5)]:
        logs = [json.loads(s) for s in (DATA/f'{name}_updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in logs] == list(range(lo, hi+1))
        assert all(r['learning_rate'] == rate for r in logs)
        assert np.isfinite([r['loss'] for r in logs]).all()
        assert all(set(r['unclipped_module_gradient_norms']) == {'adapter.', 'diffusion_decoder.', 'history_projection.'} for r in logs)
        assert all(np.isfinite(v) and v > 0 for r in logs for v in r['unclipped_module_gradient_norms'].values())
        assert all(np.isfinite(r['global_unclipped_gradient_norm']) for r in logs)
        assert logs[0]['loss'] == (report['warmup_first_loss'] if name == 'common' else run['loss_first'])
        if name == 'reduced':
            assert logs[-1]['loss'] == run['loss_last'] and float(np.mean([r['loss'] for r in logs])) == run['mean_loss']
    cases = [('common', h['update'], 'train', h['train']) for h in report['common_history']]
    cases += [('reduced', h['update'], 'train', h['train']) for h in run['history']]
    cases += [('reduced', 4096, 'development', run['final_development'])]
    assert [h['update'] for h in report['common_history']] == [0, 1024, 2048]
    assert [h['update'] for h in run['history']] == [3072, 3584, 3840, 4096]
    expected = {'report.json', 'plan.json', 'runner_snapshot.py', 'common_updates.jsonl', 'reduced_updates.jsonl'}
    expected |= {f'{arm}_u{u:04d}_{s}.npz' for arm, u, s, _ in cases}
    assert expected <= {p.name for p in DATA.iterdir() if p.is_file()}
    labels, history, windows, final = {}, [], [], {}
    deltas = dict.fromkeys(FIELDS, 0.)
    def compare(metrics, recorded):
        for key, val in metrics.items():
            delta = abs(val-recorded[key]); deltas[key] = max(deltas[key], delta)
            assert delta < (.02 if 'rotation' in key else 1e-4), (key, val, recorded[key])
    for arm, update, split, recorded in cases:
        with np.load(DATA/f'{arm}_u{update:04d}_{split}.npz', allow_pickle=False) as z:
            pred, target, indices, contexts = (z[k] for k in ('predictions', 'targets', 'row_indices', 'contexts'))
        rows = manifest[split]; n = len(rows)
        assert pred.shape == (3, n, 16, 8) and target.shape == (n, 16, 8) and contexts.shape == (n, 1024)
        assert np.isfinite(contexts).all()
        assert np.array_equal(indices, np.arange(n) if split == 'train' else np.arange(64, 104))
        if split in labels: assert np.array_equal(target, labels[split])
        else: labels[split] = target.copy()
        static = np.zeros_like(target); static[..., 6] = 1
        sm, _, _, _, _ = summarize(static[None], target)
        for key in FIELDS:
            assert abs(sm[key]-recorded['static_'+key]) < (.02 if 'rotation' in key else 1e-4)
        groups = [('overall', np.arange(n))]
        if split == 'train': groups += [('retained_train', np.arange(32)), ('added_train', np.arange(32, 64))]
        for group, ix in groups:
            rec = recorded if group == 'overall' else recorded[group]
            metrics, good, gate, p, r = summarize(pred[:, ix], target[ix]); compare(metrics, rec)
            if split == 'train':
                assert int(good.sum()) == rec['fit_gate']['windows_all_three_seeds_pass']
                assert gate == rec['fit_gate']['pose_fit_passed']
            summary = dict(arm=arm, update=update, split=split, group=group, **metrics, passing_windows=int(good.sum()), windows=len(ix), fit_gate=gate)
            history.append(summary)
            if update == 4096:
                final[split+'/'+group] = dict(**summary, worst_window_mean_position_mm=float(p.mean((0, 2)).max()*10),
                                             largest_saved_target_position_mm=float(p.max()*10))
            if group == 'overall':
                for i, row in enumerate(rows):
                    windows.append(dict(arm=arm, update=update, split=split, window=i, episode=row['shard']+'::'+str(row['record_index']), task=row['task'],
                                        position_mm=float(p[:, i].mean()*10), rotation_deg=float(r[:, i].mean()), passing=bool(good[i])))
                for episode in rec['per_episode']:
                    ep_ix = [j for j, row in enumerate(rows) if row['shard']+'::'+str(row['record_index']) == episode['episode']]
                    em, _, _, _, _ = summarize(pred[:, ep_ix], target[ep_ix]); compare(em, episode)
    with np.load(ROOT/'reports/bridge-late-lr-fit-200150/reduced_u2048_train.npz') as z:
        assert np.array_equal(labels['train'][:32], z['targets'])
    with np.load(ROOT/'reports/bridge-late-lr-fit-200150/reduced_u2048_development.npz') as z:
        assert np.array_equal(labels['development'], z['targets'])
    with np.load(PACK/'added_reference/arrays.npz') as z:
        assert np.max(np.abs(labels['train'][32:]-z['targets'])) <= 1e-6
    dev_episodes = run['final_development']['per_episode']
    replication = None
    if args.baseline_data:
        baseline = json.loads((args.baseline_data/'report.json').read_text())
        baseline_manifest = json.loads((args.baseline_data/'plan.json').read_text())['manifest']
        assert sha(args.baseline_data/'report.json') == manifest['replication']['baseline_report_sha256']
        assert baseline['manifest_sha256'] == manifest['replication']['baseline_manifest_sha256']
        for key in ('train', 'development', 'config', 'schedule_sha256', 'batch_size', 'updates_per_arm',
                    'draw_slots_per_window', 'sampling_seeds', 'learning_rate_schedule', 'growth'):
            assert manifest[key] == baseline_manifest[key]
        entry = 'fit_bridge_growth64.py'
        assert (DATA/'runner_snapshot.py').read_text().replace('init = 42; seed(init)', 'init = 43; seed(init)') == (args.baseline_data/'runner_snapshot.py').read_text()
        assert {k:v for k,v in report['source_sha256'].items() if k!=entry} == {k:v for k,v in baseline['source_sha256'].items() if k!=entry}
        label_match = {}
        for split in ('train', 'development'):
            with np.load(args.baseline_data/f'reduced_u4096_{split}.npz') as z:
                assert np.array_equal(labels[split], z['targets'])
                label_match[split] = True
        replication = dict(baseline_seed=baseline['initialization_seed'], baseline_report_sha256=sha(args.baseline_data/'report.json'),
                           training_source_only_initialization_differs=True, manifest_recipe_and_labels_equal=True,
                           processed_input_hashes_exact=(report['data_checks']==baseline['data_checks']))
    result = dict(verified=True, initialization_seed=init, replication=replication,
                  downloaded_required_files=len(expected), downloaded_bytes=sum((DATA/n).stat().st_size for n in expected),
                  updates_verified=4096, train_episodes=32, development_episodes=10, final=final,
                  reduced_history_all_train_groups_pass=all(h['fit_gate'] for h in history if h['arm']=='reduced' and h['split']=='train'),
                  development_episodes_position_better_than_static=sum(e['path_position_cm'] < e['static_path_position_cm'] for e in dev_episodes),
                  max_float64_metric_difference=deltas, peak_reserved_gib=report['peak_reserved_gib'], elapsed_seconds=report['elapsed_seconds'],
                  pt_files_downloaded=bool(list(DATA.glob('*.pt'))), checkpoint_bytes_reported=sum(c['bytes'] for c in report['checkpoints']),
                  checkpoint_bytes_verified_locally=False, artifact_sha256={n:sha(DATA/n) for n in sorted(expected)})
    OUT.mkdir(exist_ok=True)
    for name, rows in [('history.csv', history), ('windows.csv', windows)]:
        with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for group, label, color in [('retained_train', 'Retained 32', '#1565c0'), ('added_train', 'Added 32', '#e0641d')]:
        points = [h for h in history if h['split']=='train' and h['group']==group]
        steps = [h['update'] for h in points]
        axes[0].plot(steps, [h['path_position_cm']*10 for h in points], 'o-', label=label, color=color)
        axes[1].plot(steps, [h['passing_windows'] for h in points], 'o-', color=color)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Mean training position error (mm)')
    axes[0].axhline(5, color='gray', linestyle='--', linewidth=1); axes[0].legend()
    axes[1].set_ylabel('Windows passing all 3 sampling draws / 32'); axes[1].set_ylim(-1, 33)
    axes[1].axhline(29, color='gray', linestyle='--', linewidth=1)
    for ax in axes:
        ax.axvline(2048, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Optimizer update'); ax.grid(alpha=.2)
    fig.suptitle('64-window fitting: one training run; LR reduced after update 2048')
    fig.savefig(OUT/'growth64_training.png', dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'}, indent=2))

if __name__ == '__main__': main()
