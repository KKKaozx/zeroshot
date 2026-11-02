"""Independent saved-array audit of the completed fixed Bridge prior candidate."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_state_prior_v1'
BASE = ROOT/'cluster_packages/bridge_growth128_v1'
DATA = ROOT/'reports/bridge-prior-fit-205648'
PRE = ROOT/'reports/bridge-prior-preflight-205618'
OLD = ROOT/'reports/bridge-state-fit-202278/real'
OUT = ROOT/'reports/bridge_prior_analysis_205648'
sys.path.insert(0, str(PACK))
from run_prior import effective_source
from prior_contract import predict_prior
from analyze_bridge_growth64_200600 import summarize


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(name, records):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    assert not OUT.exists(), 'Preserve completed audit; do not overwrite.'
    protocol, manifest = read(PACK/'protocol.json'), read(BASE/'manifest.json')
    r, pre, old = read(DATA/'report.json'), read(PRE/'report.json'), read(OLD/'report.json')
    old_audit = read(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')
    assert old_audit['verified']
    for path, digest in protocol['inherited_sha256'].items():
        assert sha(PACK.parent/path) == digest
    assert sha(PACK/'prior.json') == protocol['prior_sha256']
    rows = manifest['train']+manifest['development']
    episodes = np.array([x['shard']+'::'+str(x['record_index']) for x in rows])
    assert len(set(episodes[:128])) == 64 and len(set(episodes[128:])) == 10
    assert not set(episodes[:128]) & set(episodes[128:])
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz') as z:
        prior = predict_prior(z['current_pose'], [x['task'] for x in rows], read(PACK/'prior.json'))
    assert hashlib.sha256(prior.tobytes()).hexdigest() == protocol['all_prior_float32_sha256']
    for directory, report, stage, end in [(DATA, r, 'fit', 8192), (PRE, pre, 'preflight', 8)]:
        assert report['stage'] == stage and report['state_arm'] == 'real'
        for k in ['execution_checks_passed', 'input_contract_passed', 'clip_unchanged', 'other_frozen_unchanged', 'position_prior_uses_current_observation_only']:
            assert report[k]
        for k in ['gripper_trained', 'simulation_executed', 'downloads', 'reserved_test_targets_read', 'original_control_retrained']:
            assert not report[k]
        assert report['manifest_sha256'] == sha(BASE/'manifest.json')
        assert report['source_sha256'] == manifest['source_sha256']
        assert report['data_checks'] == old['data_checks']
        assert len(report['data_checks']) == len(report['input_numeric_comparisons']) == 168
        assert all(c['raw_identity_exact'] and all(v['max_abs'] <= v['atol'] for v in c['differences'].values()) for c in report['input_numeric_comparisons'])
        assert report['initialization_seed'] == 43 and report['final_update'] == end
        assert report['sampling_steps'] == 100 and report['sampling_seeds'] == [1101, 1102, 1103]
        assert report['initial_shared_sha256'] == old['initial_shared_sha256']
        assert report['augmented_initial_sha256'] == old['augmented_initial_sha256']
        assert report['zero_prior_first_loss_difference'] == 0 and report['prior_fit_rows'] == list(range(128))
        for key, name in [('residual_protocol_sha256', 'protocol.json'), ('residual_entry_sha256', 'run_prior.py'), ('residual_helper_sha256', 'prior_contract.py')]:
            assert report[key] == sha(PACK/name)
        assert report['effective_runner_sha256'] == sha(directory/'effective_runner.py')
        # Text comparison deliberately normalizes OS line endings only.
        assert (directory/'effective_runner.py').read_text(encoding='utf-8') == effective_source()
        assert sha(directory/'runner_snapshot.py') == protocol['inherited_sha256']['bridge_state_conditioning_v1/fit_state_arm.py']
        assert read(directory/'plan.json')['manifest'] == manifest
        sc = report['state_checks']
        assert sc['raw_current_pose_exact'] and sc['train_only_statistics']
        assert not sc['target_or_future_state_used_as_context']
        assert sc['features_sha256'] == old['state_checks']['features_sha256']
    assert pre['branch_restore_passed'] and pre['branch_restore']['parameters_exact']
    assert not pre['full_token_cache_reused'] and r['full_token_cache_reused']
    assert r['warmup_first_loss'] == pre['warmup_first_loss']
    assert r['branch_update'] == 4096 and len(r['runs']) == 1
    assert r['runs'][0]['actual_draw_slots_per_window'] == [256]*128
    assert len(r['checkpoints']) == 1 and r['checkpoints'][0]['roundtrip_exact']
    assert not r['checkpoints'][0]['includes_optimizer'] and not pre['checkpoints']
    assert not list(DATA.glob('*.pt')) and not list(PRE.glob('*.pt'))
    for directory, logs in [(DATA, [('common', 1, 4096, 1e-4), ('reduced', 4097, 8192, 1e-5)]),
                            (PRE, [('common', 1, 4, 1e-4), ('constant', 5, 8, 1e-4), ('restore_same_lr', 5, 8, 1e-4), ('reduced', 5, 8, 1e-5)])]:
        for name, first, last, lr in logs:
            values = [json.loads(s) for s in (directory/f'{name}_updates.jsonl').read_text().splitlines()]
            assert [v['update'] for v in values] == list(range(first, last+1))
            assert all(v['learning_rate'] == lr and np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm']) for v in values)
            assert all(set(v['unclipped_module_gradient_norms']) == {'adapter.', 'diffusion_decoder.', 'history_projection.', 'state_projection.'} for v in values)
            assert all(np.isfinite(g) and g >= 0 for v in values for g in v['unclipped_module_gradient_norms'].values())
    for a, b in [('constant_updates.jsonl', 'restore_same_lr_updates.jsonl'), ('constant_u0008_train.npz', 'restore_same_lr_u0008_train.npz')]:
        if a.endswith('.npz'):
            with np.load(PRE/a) as x, np.load(PRE/b) as y:
                assert all(np.array_equal(x[k], y[k]) for k in x.files)
        else:
            assert (PRE/a).read_bytes() == (PRE/b).read_bytes()
    maximum = dict(position_cm=0., rotation_deg=0., reconstruction=0.)
    snapshots, metrics, final, hashes = [], [], {}, {}

    def compare(actual, recorded):
        for k, value in actual.items():
            difference = abs(value-recorded[k])
            maximum['rotation_deg' if 'rotation' in k else 'position_cm'] = max(maximum['rotation_deg' if 'rotation' in k else 'position_cm'], difference)
            assert difference < (.02 if 'rotation' in k else 1e-4), (k, value, recorded[k])

    for directory, report in [(DATA, r), (PRE, pre)]:
        expected = [('common', q['update'], 'train', q['train']) for q in report['common_history']]
        for run in report['runs']:
            expected += [(run['arm'], q['update'], 'train', q['train']) for q in run['history']]
            if directory == DATA:
                expected.append((run['arm'], 8192, 'development', run['final_development']))
        assert len(expected) == (8 if directory == DATA else 5)
        for arm, update, split, recorded in expected:
            file = directory/f'{arm}_u{update:04d}_{split}.npz'
            with np.load(file) as z:
                pred, target, ids, saved_prior, residual = [z[k].copy() for k in ['predictions', 'targets', 'row_indices', 'position_prior', 'residual_predictions']]
                clipped_count = int(z['reconstruction_clipped_components'])
            correct_ids = np.arange(128, 168) if split == 'development' else np.arange(128) if directory == DATA else np.array([0, 32, 64, 96])
            assert np.array_equal(ids, correct_ids) and np.array_equal(saved_prior, prior[ids])
            assert pred.shape == residual.shape == (3, len(ids), 16, 8)
            independently = residual.copy(); independently[..., :3] += saved_prior[None]
            before = independently[..., :3].copy(); independently[..., :3] = np.clip(before, -3., 3.)
            assert np.array_equal(independently, pred)
            assert int(np.count_nonzero(before != independently[..., :3])) == clipped_count
            assert np.array_equal(pred[..., 3:], residual[..., 3:])
            with np.load(OLD/f'reduced_u8192_{split}.npz') as original:
                reference = original['targets'] if split == 'development' else original['targets'][ids]
                assert np.max(np.abs(reference-target)) <= 1e-6
            summary, good, gate, pe, re = summarize(pred, target)
            compare(summary, recorded)
            if split == 'train':
                assert gate == recorded['fit_gate']['pose_fit_passed']
                assert int(good.sum()) == recorded['fit_gate']['windows_all_three_seeds_pass']
            snapshots.append(dict(stage=directory.name, arm=arm, update=update, split=split,
                **summary, passing_windows=int(good.sum()), fit_gate=gate, clipped_xyz_components=clipped_count))
            if directory == DATA and update == 8192:
                final[split] = dict(predictions=pred, targets=target, prior=saved_prior, pos=pe, rot=re, metrics=summary, gate=gate, passing=int(good.sum()), clipped=clipped_count)
    episodes_table, tasks_table, windows = [], [], []
    summaries, decompositions = {}, {}
    for split, q in final.items():
        with np.load(OLD/f'reduced_u8192_{split}.npz') as z:
            old_pred = z['predictions'].copy(); assert np.max(np.abs(z['targets']-q['targets'])) <= 1e-6
        old_m, _, _, old_p, old_r = summarize(old_pred, q['targets'])
        static = np.zeros_like(q['targets']); static[..., 6] = 1
        static_m, _, _, sp, sr = summarize(static[None], q['targets'])
        prior_p = np.linalg.norm(q['prior'].astype(float)-q['targets'][..., :3].astype(float), axis=-1)*10
        methods = {'original_ddpm':(old_p, old_r), 'new_prior_ddpm':(q['pos'], q['rot']), 'static':(sp,sr), 'state_prior_only':(prior_p[None],None)}
        summaries[split] = dict(original_ddpm=old_m, new_prior_ddpm=q['metrics'], static=static_m,
            state_prior_only=dict(path_position_cm=float(prior_p.mean()), endpoint_position_cm=float(prior_p[:, -1].mean())),
            train_fit_passed=q['gate'] if split=='train' else None, passing_windows=q['passing'], clipped_xyz_components=q['clipped'])
        local_rows = manifest[split]; grouping = defaultdict(list); task_grouping = defaultdict(list)
        for i, row in enumerate(local_rows):
            ep = row['shard']+'::'+str(row['record_index']); grouping[ep].append(i); task_grouping[row['task']].append(i)
            windows.append(dict(split=split, window=i, episode=ep, task=row['task'],
                new_position_cm=float(q['pos'][:,i].mean()), old_position_cm=float(old_p[:,i].mean()),
                static_position_cm=float(sp[:,i].mean()), prior_position_cm=float(prior_p[i].mean()),
                new_rotation_deg=float(q['rot'][:,i].mean()), old_rotation_deg=float(old_r[:,i].mean()), static_rotation_deg=float(sr[:,i].mean())))
        for by, destination, key in [(grouping, episodes_table, 'episode'), (task_grouping, tasks_table, 'task')]:
            for group, ix in by.items():
                record = dict(split=split, **{key:group}, windows=len(ix))
                if key=='episode': record['task']=local_rows[ix[0]]['task']
                for method, (position, rotation) in methods.items():
                    record[method+'_position_cm'] = float(position[:,ix].mean())
                    record[method+'_endpoint_position_cm'] = float(position[:,ix,-1].mean())
                    if rotation is not None:
                        record[method+'_rotation_deg'] = float(rotation[:,ix].mean())
                        record[method+'_endpoint_rotation_deg'] = float(rotation[:,ix,-1].mean())
                destination.append(record)
        # Error decomposition in physical XYZ, averaged over all saved draws.
        error = (q['prior'].astype(float)-q['targets'][..., :3].astype(float))*10
        correction = (q['predictions'][..., :3].astype(float)-q['prior'][None].astype(float))*10
        prior_sq = float(np.square(error).sum(-1).mean())
        correction_sq = float(np.square(correction).sum(-1).mean())
        cross = float(2*(error[None]*correction).sum(-1).mean())
        full_sq = float(np.square(error[None]+correction).sum(-1).mean())
        assert abs(full_sq-prior_sq-correction_sq-cross) < 1e-9
        decompositions[split] = dict(prior_mean_squared_distance_cm2=prior_sq, correction_mean_squared_distance_cm2=correction_sq,
            cross_term_cm2=cross, final_mean_squared_distance_cm2=full_sq, correction_mean_length_cm=float(np.linalg.norm(correction,axis=-1).mean()))
    dev = [e for e in episodes_table if e['split']=='development']; assert len(dev)==10
    counts = dict(position_better_than_old=sum(e['new_prior_ddpm_position_cm']<e['original_ddpm_position_cm'] for e in dev),
        rotation_better_than_old=sum(e['new_prior_ddpm_rotation_deg']<e['original_ddpm_rotation_deg'] for e in dev),
        position_better_than_static=sum(e['new_prior_ddpm_position_cm']<e['static_position_cm'] for e in dev),
        rotation_better_than_static=sum(e['new_prior_ddpm_rotation_deg']<e['static_rotation_deg'] for e in dev),
        position_better_than_prior=sum(e['new_prior_ddpm_position_cm']<e['state_prior_only_position_cm'] for e in dev))
    m = summaries['development']; thresholds = protocol['candidate_screen']
    screen = dict(train_fit=final['train']['gate'], position_better_than_static=m['new_prior_ddpm']['path_position_cm']<thresholds['position_below_static_cm'],
        position_better_than_old=m['new_prior_ddpm']['path_position_cm']<thresholds['position_below_original_cm'],
        rotation_no_worse_than_old=m['new_prior_ddpm']['path_rotation_deg']<=thresholds['rotation_no_worse_than_original_deg'],
        at_least6_episodes_position_better_than_old=counts['position_better_than_old']>=6)
    episode_equal = {method:{kind:float(np.mean([e[f'{method}_{kind}'] for e in dev])) for kind in ['position_cm']+([] if method=='state_prior_only' else ['rotation_deg'])} for method in ['original_ddpm','new_prior_ddpm','static','state_prior_only']}
    rng = np.random.default_rng(205648); resamples = rng.integers(0,10,size=(10000,10)); uncertainty={}
    for control in ['original_ddpm','static','state_prior_only']:
        delta = np.array([e[control+'_position_cm']-e['new_prior_ddpm_position_cm'] for e in dev])
        uncertainty[control] = dict(episode_equal_position_improvement_cm=float(delta.mean()),
            descriptive_paired_episode_bootstrap95_cm=np.quantile(delta[resamples].mean(1),[.025,.975]).tolist())
    archive = ROOT/'reports/bridge_prior_results_205648.tar.gz'
    with tarfile.open(archive, 'r:gz') as tar:
        members = [x for x in tar.getmembers() if x.isfile()]
        assert all(not Path(x.name).is_absolute() and '..' not in Path(x.name).parts and not x.name.endswith('.pt') for x in members)
        for member in members:
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(ROOT/'reports'/member.name)
    for directory in [DATA, PRE]:
        hashes.update({str(p.relative_to(ROOT)):sha(p) for p in directory.iterdir() if p.is_file()})
    result = dict(verified=True, final_job=205648, preflight_job=205618, formal_updates_verified=8192,
        snapshots_verified=len(snapshots), all_original_targets_and_split_verified=True,
        all_residual_plus_prior_reconstructions_exact=True, preflight_same_lr_restore_exact=True,
        original_control_retrained=False, token_cache_reuse_reported=True, token_cache_contents_downloaded=False,
        checkpoint_metadata=r['checkpoints'][0], checkpoint_downloaded=False, checkpoint_tensors_independently_verified=False,
        summaries=summaries, development_episode_counts=counts, episode_equal_development=episode_equal,
        candidate_checks=screen, candidate_passed=all(screen.values()), residual_error_decomposition=decompositions,
        descriptive_episode_uncertainty=uncertainty, max_recorded_metric_difference=maximum,
        peak_reserved_gib=r['peak_reserved_gib'], elapsed_seconds=r['elapsed_seconds'],
        artifact_sha256=hashes, archive_sha256=sha(archive), entry_sha256=sha(Path(__file__)),
        limits=['Same10 development demonstrations repeatedly used; exploratory, not fresh final test.',
                'Three sampling seeds are not independent neural training replications.',
                'Position/rotation offline only; gripper frozen; no physical grasp result.',
                'Checkpoint and token-cache identities are reported by runner; those tensors were not downloaded.',
                'Clipping reference and noisy-action origin follow declared residual parameterization; no unique module-causality proof.'],
        decision='Partial position benefit only; candidate fails frozen screen. Stop without extra training, seeds or mixture tuning.')
    OUT.mkdir(); (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_csv('history.csv', snapshots); write_csv('per_window.csv', windows)
    write_csv('paired_episodes.csv', episodes_table); write_csv('paired_tasks.csv', tasks_table)
    print(json.dumps({k:result[k] for k in ['verified','summaries','development_episode_counts','candidate_checks','candidate_passed','residual_error_decomposition']},indent=2))


if __name__=='__main__':
    main()
