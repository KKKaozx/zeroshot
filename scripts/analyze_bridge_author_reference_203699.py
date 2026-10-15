"""CPU-only independent audit of the fixed-final author Bridge reference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/bridge-author-reference-203699'
PACK = ROOT / 'cluster_packages/bridge_author_reference_v1'
OUT = ROOT / 'reports/bridge_author_reference_analysis_203699'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def errors(pred, target):
    assert pred.ndim == 4 and pred.shape[1:] == target.shape
    assert np.isfinite(pred).all() and np.isfinite(target).all()
    delta = (pred[..., :3].astype(np.float64) - target[None, ..., :3]) * 10
    position = np.sqrt(np.sum(delta ** 2, axis=-1))
    a = pred[..., 3:7].astype(np.float64)
    b = target[None, ..., 3:7].astype(np.float64)
    assert np.linalg.norm(a, axis=-1).min() > 1e-8
    assert np.linalg.norm(b, axis=-1).min() > 1e-8
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    rotation = np.rad2deg(2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))
    return position, rotation, delta

def basic(position, rotation):
    good = ((position.mean(-1) <= .5) & (rotation.mean(-1) <= 5)).all(0)
    gate = bool(good.mean() >= .9 and (position.mean((1, 2)) <= .5).all()
                and (rotation.mean((1, 2)) <= 5).all()
                and (position[..., -1].mean(1) <= 1).all()
                and (rotation[..., -1].mean(1) <= 10).all())
    return dict(path_position_cm=float(position.mean()), path_rotation_deg=float(rotation.mean()),
                endpoint_position_cm=float(position[..., -1].mean()),
                endpoint_rotation_deg=float(rotation[..., -1].mean()),
                windows=position.shape[1], passing_windows=int(good.sum()), pose_fit_passed=gate)

def check_metric(computed, saved):
    for k, value in computed.items():
        if isinstance(value, float):
            assert abs(value - saved[k]) < 1e-7, (k, value, saved[k])
        else:
            assert value == saved[k], (k, value, saved[k])

def group_details(pred, target, rows, saved=None):
    pos, rot, delta = errors(pred, target)
    result = basic(pos, rot)
    if saved is not None:
        check_metric(result, saved)
    static = np.zeros_like(target)
    static[..., 6] = 1
    sp, sr, _ = errors(static[None], target)
    result['static'] = basic(sp, sr)
    if saved is not None:
        check_metric(result['static'], saved['static'])
    result['per_sampling_draw'] = [basic(pos[i:i+1], rot[i:i+1]) for i in range(len(pred))]
    result['per_horizon'] = [dict(target=i+1, position_cm=float(pos[..., i].mean()),
        rotation_deg=float(rot[..., i].mean()), static_position_cm=float(sp[..., i].mean()),
        static_rotation_deg=float(sr[..., i].mean())) for i in range(16)]
    result['position_bias_xyz_cm'] = delta.mean((0, 1, 2)).tolist()
    result['window_position_cm_quantiles'] = dict(zip(('min', 'p25', 'median', 'p75', 'p90', 'max'),
        map(float, np.quantile(pos.mean((0, 2)), [0, .25, .5, .75, .9, 1]))))
    result['failure_counts'] = dict(
        position=int((pos.mean(-1) > .5).any(0).sum()),
        rotation=int((rot.mean(-1) > 5).any(0).sum()),
        both=int(((pos.mean(-1) > .5).any(0) & (rot.mean(-1) > 5).any(0)).sum()))
    pairs = [np.linalg.norm((pred[a, ..., :3].astype(float)-pred[b, ..., :3])*10, axis=-1)
             for a, b in ((0, 1), (0, 2), (1, 2))]
    result['pairwise_sampling_position_difference_cm'] = float(np.mean(pairs))
    xyz = pred[..., :3].astype(float)*10
    truth = target[..., :3].astype(float)*10
    mean_xyz = xyz.mean(0)
    total_squared = float(np.square(xyz-truth).sum(-1).mean())
    mean_squared = float(np.square(mean_xyz-truth).sum(-1).mean())
    variation = float(np.square(xyz-mean_xyz).sum(-1).mean())
    assert abs(total_squared-mean_squared-variation) < 1e-10
    result['observed_three_draw_squared_position_decomposition'] = dict(
        total_cm2=total_squared, mean_prediction_residual_cm2=mean_squared,
        variation_between_draws_cm2=variation, variation_fraction=variation/total_squared,
        mean_prediction_path_error_cm=float(np.linalg.norm(mean_xyz-truth, axis=-1).mean()),
        used_for_model_selection_or_fitting_gate=False,
        limitation='Exact finite-three-draw identity; not an estimate of all model uncertainty or an evaluated deployment strategy.')
    result['predicted_quaternion_norm_range'] = [float(x) for x in
        (np.linalg.norm(pred[..., 3:7], axis=-1).min(), np.linalg.norm(pred[..., 3:7], axis=-1).max())]
    result['tasks'] = []
    for task in sorted({r['task'] for r in rows}):
        ix = [i for i, row in enumerate(rows) if row['task'] == task]
        result['tasks'].append(dict(task=task, **basic(pos[:, ix], rot[:, ix]),
                                   static=basic(sp[:, ix], sr[:, ix])))
    episodes = {}
    for i, row in enumerate(rows):
        episodes.setdefault(row['shard']+'::'+str(row['record_index']), []).append(i)
    saved_episodes = {x['episode']: x for x in saved['per_episode']} if saved else {}
    result['episodes'] = []
    for key, ix in episodes.items():
        item = dict(episode=key, task=rows[ix[0]]['task'], **basic(pos[:, ix], rot[:, ix]),
                    static=basic(sp[:, ix], sr[:, ix]))
        if saved is not None:
            check_metric(basic(pos[:, ix], rot[:, ix]), saved_episodes[key])
            check_metric(item['static'], saved_episodes[key]['static'])
        result['episodes'].append(item)
    for field in ('path_position_cm', 'path_rotation_deg'):
        result[field+'_better_than_static_demos'] = sum(x[field] < x['static'][field] for x in result['episodes'])
    if saved is not None:
        assert result['path_position_cm_better_than_static_demos'] == saved['position_better_than_static_demos']
    result['worst_windows_by_position'] = [dict(row=rows[i], position_cm=float(pos[:, i].mean()),
        rotation_deg=float(rot[:, i].mean())) for i in np.argsort(pos.mean((0, 2)))[-10:][::-1]]
    return result

def main():
    OUT.mkdir(exist_ok=True)
    r = read(DATA/'report.json')
    p = read(PACK/'training_protocol.json')
    assert r['execution_passed'] and r['trained_updates'] == 8192 and r['primary'] == 'ema'
    assert r['training_protocol_sha256'] == sha(PACK/'training_protocol.json')
    assert r['raw_input_report_sha256'] == p['input_export_report_sha256']
    assert r['author_commit'] == '5ba07ac6661db573af695b419a7947ecb704690f'
    assert r['resource_gate']['passed'] and r['peak_reserved_gib'] < 40
    assert r['training_statistics_only'] and r['validation_loss_is_four_training_rows']
    assert not r['simulation_executed'] and not r['reserved_test_used'] and 'error' not in r
    assert len(r['checkpoints']) == 2 and all(not x['includes_optimizer'] for x in r['checkpoints'])
    assert sum(x['bytes'] for x in r['checkpoints']) < 3 * 2**30
    for name, h in r['source_sha256'].items():
        assert sha(PACK/name) == h
    cfg = yaml.safe_load((DATA/'runtime_config.yaml').read_text())
    expected = yaml.safe_load((PACK/'training_config.yaml').read_text())
    for k in ('cache', 'protocol'):
        expected['task']['dataset'][k] = cfg['task']['dataset'][k]
    assert cfg == expected
    updates = [json.loads(x) for x in (DATA/'updates.jsonl').read_text().splitlines()]
    logs = [json.loads(x) for x in (DATA/'workspace/logs.json.txt').read_text().splitlines()]
    assert len(updates) == len(logs) == 8192
    epoch_means = []
    for i, (u, log) in enumerate(zip(updates, logs)):
        assert u['update'] == i+1 and u['epoch'] == log['epoch'] == i//16
        assert log['global_step'] == i and np.isfinite(u['loss']) and np.isfinite(log['train_loss'])
        t = i
        rate = 1e-4 * (t/500 if t < 500 else .5*(1+np.cos(np.pi*(t-500)/(8192-500))))
        assert abs(u['learning_rate']-rate) < 1e-15
        if i % 16 != 15:
            assert u['loss'] == log['train_loss']
        else:
            mean = np.mean([x['loss'] for x in updates[i-15:i+1]])
            assert abs(mean-log['train_loss']) < 1e-12
            epoch_means.append(float(mean))
    assert logs[-1]['lr'] == 0
    sys.path.insert(0, str(ROOT/'cluster_packages/bridge_author_preflight_v1'))
    from bridge_author_io import assemble_references
    refs = assemble_references(ROOT/'cluster_packages')['targets'][..., :7]
    assert len(p['rows']) == 168
    output = dict(audit_passed=True, independently_recomputed=True, local_model_executed=False,
        checkpoints_downloaded=False, full_weight_contents_verified=False, job=203699,
        trained_updates=8192, reported_peak_reserved_gib=r['peak_reserved_gib'],
        protocol_sha256=sha(PACK/'training_protocol.json'), input_export_report_sha256=r['raw_input_report_sha256'],
        epoch_mean_noise_loss=dict(first=epoch_means[0], last=epoch_means[-1], last32=float(np.mean(epoch_means[-32:]))),
        groups={}, project_comparison={})
    for name in ('raw', 'ema'):
        for split, indices in [('train', np.arange(128)), ('development', np.arange(128, 168))]:
            with np.load(DATA/(name+'_'+split+'.npz'), allow_pickle=False) as z:
                assert set(z.files) == {'predictions', 'targets', 'row_indices'}
                pred, target = z['predictions'].copy(), z['targets'].copy()
                assert pred.shape == (3, len(indices), 16, 7) and np.array_equal(z['row_indices'], indices)
            assert np.max(np.abs(target-refs[indices])) <= 1e-6
            rows = [p['rows'][i] for i in indices]
            output['groups'][name+'/'+split] = group_details(pred, target, rows, r['metrics'][name][split])
    for split, indices in [('train', np.arange(128)), ('development', np.arange(128, 168))]:
        with np.load(ROOT/'reports/bridge-growth128-fit-201663'/('reduced_u8192_'+split+'.npz'), allow_pickle=False) as z:
            assert np.array_equal(z['row_indices'], indices)
            assert np.max(np.abs(z['targets'][..., :7]-refs[indices])) <= 1e-6
            output['project_comparison'][split] = group_details(z['predictions'][..., :7], z['targets'][..., :7],
                [p['rows'][i] for i in indices])
    primary = output['groups']['ema/development']
    output['effective_author_reference'] = bool(output['groups']['ema/train']['pose_fit_passed']
        and primary['path_position_cm'] < primary['static']['path_position_cm']
        and primary['path_rotation_deg'] < primary['static']['path_rotation_deg']
        and primary['path_position_cm_better_than_static_demos'] >= 6)
    expected_files = ['report.json', 'runtime_config.yaml', 'updates.jsonl', 'workspace/logs.json.txt'] + [
        name+'_'+split+'.npz' for name in ('raw', 'ema') for split in ('train', 'development')]
    assert sorted(f.relative_to(DATA).as_posix() for f in DATA.rglob('*') if f.is_file()) == sorted(expected_files)
    output['files_sha256'] = {n:sha(DATA/n) for n in expected_files}
    (OUT/'analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for key, group in output['groups'].items():
        print(key, json.dumps({k:group[k] for k in ('path_position_cm','path_rotation_deg','passing_windows',
            'pose_fit_passed','failure_counts','window_position_cm_quantiles','pairwise_sampling_position_difference_cm')}, ensure_ascii=False))
    print('INDEPENDENT AUTHOR REFERENCE AUDIT PASSED; effective reference:', output['effective_author_reference'])

if __name__ == '__main__':
    main()
