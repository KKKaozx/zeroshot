"""Audit existing checkpoint-selection artifacts; no model loading or inference."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_bridge_growth64_200600 import summarize

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_inner_selection_v1'
DATA = ROOT/'reports/bridge-selection-fit-205997'
PRE = ROOT/'reports/bridge-selection-preflight-205825'
OUT = ROOT/'reports/bridge_inner_selection_analysis_205997'
sys.path.insert(0, str(PACK))
from selection_contract import verify_split, observed_statistics


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_write(name, records):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    assert not OUT.exists(), 'Preserve completed audit; do not rerun unchanged inputs.'
    protocol = read(PACK/'protocol.json')
    manifest = read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    rows = manifest['train']+manifest['development']
    indices = verify_split(rows, protocol['plan'])
    archive = ROOT/'reports/bridge_selection_results_205997.tar.gz'
    inventory = []
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isdir(): continue
            assert member.isfile() and not member.name.endswith(('.pt','.partial'))
            file = ROOT/'reports'/member.name
            assert tar.extractfile(member).read() == file.read_bytes()
            inventory.append(dict(path=member.name, bytes=member.size, sha256=sha(file)))
    assert len(inventory) == 32
    r, pre = read(DATA/'report.json'), read(PRE/'report.json')
    original = read(ROOT/'reports/bridge-state-fit-202278/real/report.json')
    with np.load(ROOT/'cluster_packages/bridge_state_conditioning_v1/current_pose.npz') as z:
        stats = observed_statistics(z['current_pose'], indices['optimization_indices'])
        v = z['current_pose'].astype(float)
    assert stats == protocol['state_statistics']
    # Independently assemble9 observed-state components, using96-row XYZ statistics only.
    roll,pitch,yaw = v[:,3],v[:,4],v[:,5]
    cr,cp,cy = np.cos(roll),np.cos(pitch),np.cos(yaw)
    sr,sp,sy = np.sin(roll),np.sin(pitch),np.sin(yaw)
    features = np.column_stack(((v[:,:3]-stats['xyz_mean'])/stats['xyz_std'],
                               cy*cp,sy*cp,-sp,cy*sp*sr-sy*cr,sy*sp*sr+cy*cr,cp*sr)).astype('<f4')
    feature_hash = hashlib.sha256(features.tobytes()).hexdigest()
    with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:
        targets_train = z['targets'].copy()
    with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_development.npz') as z:
        targets_outer = z['targets'].copy()
    reference_targets = np.concatenate([targets_train, targets_outer])
    for name, expected in protocol['inherited_sha256'].items(): assert sha(PACK.parent/name) == expected
    for directory, report, stage, end in [(PRE,pre,'preflight',8),(DATA,r,'fit',6144)]:
        assert report['stage'] == stage and report['final_update'] == end
        assert all(report[k] for k in ['execution_checks_passed','input_contract_passed','clip_unchanged',
                   'other_frozen_unchanged','selected_checkpoint_replay_exact','selection_locked','raw_audit_old128_scaling_discarded'])
        assert not any(report[k] for k in ['downloads','gripper_trained','simulation_executed',
                                         'reserved_test_targets_read','outer_scored_during_selection'])
        assert report['selected_checkpoint_replay_max_difference'] == 0
        assert report['inner_and_outer_gradient_draws'] == 0
        assert report['protocol_sha256'] == sha(PACK/'protocol.json') == sha(directory/'protocol.json')
        assert report['manifest_sha256'] == protocol['plan']['source_manifest_sha256']
        assert report['optimizer_indices'] == indices['optimization_indices']
        assert report['schedule_sha256'] == protocol['schedule_sha256']
        assert report['state_features_sha256'] == feature_hash and report['effective_state_statistics'] == stats
        assert report['data_checks'] == original['data_checks'] and len(report['data_checks']) == 168
        assert len(report['input_numeric_comparisons']) == 168
        assert all(v['raw_identity_exact'] and all(d['max_abs'] <= d['atol'] for d in v['differences'].values())
                   for v in report['input_numeric_comparisons'])
        assert report['raw_observed_state_checks']['raw_current_pose_exact']
        assert not report['raw_observed_state_checks']['target_or_future_state_used_as_context']
        assert report['initial_shared_sha256'] == original['initial_shared_sha256']
        assert report['initialization_seed'] == 43 and report['torch_version'] == original['torch_version']
        assert report['sampling_steps'] == 100 and report['sampling_seeds'] == [1101,1102,1103]
        cfg = json.loads(json.dumps(report['config'])); cfg['model']['name'] = manifest['config']['model']['name']
        assert cfg == manifest['config']
        assert not list(directory.glob('*.pt')) and not list(directory.glob('*.partial'))
        for name, expected in report['entry_sha256'].items(): assert sha(PACK/name) == expected
        assert sha(directory/'runner_snapshot.py') == sha(PACK/'run_selection.py')
        assert report['entry_sha256'] == r['entry_sha256'] and report['full_token_cache_reused']
        draws = np.asarray(report['actual_draw_slots_per_row'])
        withheld = indices['inner_selection_indices']+indices['outer_development_indices']
        assert draws[withheld].sum() == 0 and draws.sum() == end*8
        if stage == 'fit': assert np.array_equal(draws[indices['optimization_indices']],np.full(96,512))
        else: assert report['observer_replay_exact'] and report['actual_operational_updates_including_control'] == 16
        logs = [json.loads(s) for s in (directory/'updates.jsonl').read_text().splitlines()]
        assert [v['update'] for v in logs] == list(range(1,end+1))
        switch = 4 if stage == 'preflight' else 3072
        assert report['lr_switch_after_update'] == switch
        assert all(v['learning_rate'] == (1e-4 if v['update'] <= switch else 1e-5) for v in logs)
        assert all(np.isfinite(v['loss']) and np.isfinite(v['unclipped_global_norm']) and
                   all(np.isfinite(g) and g >= 0 for g in v['gradient_norms'].values()) for v in logs)
        assert all(set(v['gradient_norms']) == {'adapter.','diffusion_decoder.','history_projection.','state_projection.'} for v in logs)
    formal_logs = [json.loads(s) for s in (DATA/'updates.jsonl').read_text().splitlines()]
    pre_logs = [json.loads(s) for s in (PRE/'updates.jsonl').read_text().splitlines()]
    # The first four updates have identical LR/noise/rows; preflight switches LR at4.
    assert formal_logs[:4] == pre_logs[:4]
    snapshots, history, per_windows, final = [], [], [], {}
    max_deltas = dict(position_cm=0.,rotation_deg=0.,selection_value=0.)
    def check_file(directory, file_name, ix, recorded, label, update):
        with np.load(directory/file_name, allow_pickle=False) as z:
            prediction, target = z['predictions'].copy(), z['targets'].copy()
            assert z['row_indices'].tolist() == ix and z['sampling_seeds'].tolist() == [1101,1102,1103]
            assert prediction.shape == (3,len(ix),16,8) and target.shape == (len(ix),16,8)
            assert z['contexts'].shape == (len(ix),1024) and np.isfinite(z['contexts']).all()
        assert np.max(np.abs(target-reference_targets[ix])) <= 1e-6
        actual, good, gate, p, rot = summarize(prediction,target)
        static = np.zeros_like(target); static[...,6] = 1
        baseline,_,_,bp,br = summarize(static[None], target)
        for k,val in actual.items():
            kind = 'rotation_deg' if 'rotation' in k else 'position_cm'
            delta = abs(val-recorded[k]); max_deltas[kind] = max(max_deltas[kind],delta)
            assert delta < (.02 if 'rotation' in k else 1e-4)
            assert abs(baseline[k]-recorded['static_'+k]) < (.02 if 'rotation' in k else 1e-4)
        assert gate == recorded['fit_gate']['pose_fit_passed']
        assert int(good.sum()) == recorded['fit_gate']['windows_all_three_seeds_pass']
        episode_arrays = defaultdict(list)
        for j,i in enumerate(ix): episode_arrays[rows[i]['shard']+'::'+str(rows[i]['record_index'])].append(j)
        episodes = []
        for key, part in episode_arrays.items():
            value = dict(episode=key,task=rows[ix[part[0]]]['task'],windows=len(part),
                         path_position_cm=float(p[:,part].mean()),path_rotation_deg=float(rot[:,part].mean()),
                         endpoint_position_cm=float(p[:,part,-1].mean()),endpoint_rotation_deg=float(rot[:,part,-1].mean()),
                         static_path_position_cm=float(bp[:,part].mean()),static_path_rotation_deg=float(br[:,part].mean()))
            expected = next(e for e in recorded['per_episode'] if e['episode'] == key)
            for k in ['path_position_cm','path_rotation_deg','endpoint_position_cm','endpoint_rotation_deg']:
                assert abs(value[k]-expected[k]) < (.02 if 'rotation' in k else 1e-4)
            episodes.append(value)
        ep_pos = float(np.mean([e['path_position_cm'] for e in episodes]))
        ep_rot = float(np.mean([e['path_rotation_deg'] for e in episodes]))
        ep_static_pos = float(np.mean([e['static_path_position_cm'] for e in episodes]))
        ep_static_rot = float(np.mean([e['static_path_rotation_deg'] for e in episodes]))
        result = dict(label=label,update=update,windows=len(ix),**actual,fit_gate=gate,passing_windows=int(good.sum()),
                      episode_position_cm=ep_pos,episode_rotation_deg=ep_rot,
                      position_ratio=ep_pos/ep_static_pos,rotation_ratio=ep_rot/ep_static_rot,
                      selection_value=max(ep_pos/ep_static_pos,ep_rot/ep_static_rot),static=baseline,episodes=episodes)
        snapshots.append(dict(directory=directory.name,file=file_name,sha256=sha(directory/file_name)))
        if directory == DATA:
            for j,i in enumerate(ix):
                per_windows.append(dict(label=label,update=update,row_index=i,task=rows[i]['task'],
                    episode=rows[i]['shard']+'::'+str(rows[i]['record_index']),
                    path_position_cm=float(p[:,j].mean()),path_rotation_deg=float(rot[:,j].mean()),
                    endpoint_position_cm=float(p[:,j,-1].mean()),endpoint_rotation_deg=float(rot[:,j,-1].mean())))
        return result
    for directory, report, steps in [(PRE,pre,[4,8]),(DATA,r,protocol['plan']['selection_steps'])]:
        assert [v['update'] for v in report['selection_history']] == steps
        scores = []
        for entry in report['selection_history']:
            step = entry['update']
            calculated = check_file(directory,f'inner_u{step:04d}.npz',indices['inner_selection_indices'],entry['metrics'],'inner',step)
            delta = abs(calculated['selection_value']-entry['criterion']['value'])
            max_deltas['selection_value'] = max(max_deltas['selection_value'],delta); assert delta < 1e-4
            for key in ['position_ratio','rotation_ratio']:
                assert abs(calculated[key]-entry['criterion'][key]) < 1e-4
            scores.append((entry['criterion']['value'],step))
            if directory == DATA: history.append(calculated)
        lock = read(directory/'selection_lock.json')
        winner = min(scores)[1]
        assert report['selected_update'] == lock['selected_update'] == winner
        assert lock['checkpoint_sha256'] == report['selected_checkpoint_sha256']
        assert lock['rule'] == protocol['plan']['selection_rule']
        assert lock['criterion'] == next(e['criterion'] for e in report['selection_history'] if e['update'] == winner)
        # Independent float64 recomputation selects the same winner, despite tiny rounding differences.
        if directory == DATA: assert min(history,key=lambda e:(e['selection_value'],e['update']))['update'] == winner
    assert r['selected_update'] == 768
    for arm,step in [('selected',768),('final',6144)]:
        for split,key in [('train','optimization_indices'),('outer','outer_development_indices')]:
            final[arm+'/'+split] = check_file(DATA,f'{arm}_{split}_u{step:04d}.npz',indices[key],r[arm][split],arm+'/'+split,step)
    expected_files = {'protocol.json','report.json','runner_snapshot.py','selection_lock.json','updates.jsonl'}
    assert set(p.name for p in PRE.iterdir()) == expected_files|{'inner_u0004.npz','inner_u0008.npz'}
    expected_files |= {f'inner_u{u:04d}.npz' for u in protocol['plan']['selection_steps']}
    expected_files |= {f'{a}_{s}_u{u:04d}.npz' for a,u in [('selected',768),('final',6144)] for s in ['train','outer']}
    assert set(p.name for p in DATA.iterdir()) == expected_files
    selected, last = final['selected/outer'],final['final/outer']
    passed = all(selected[k]<last[k] for k in ['path_position_cm','path_rotation_deg'])
    assert not passed and r['both_outer_pose_errors_improved'] == passed
    paired = []
    for a,b in zip(last['episodes'],selected['episodes']):
        assert a['episode'] == b['episode']
        paired.append(dict(episode=a['episode'],task=a['task'],windows=a['windows'],
            final_position_cm=a['path_position_cm'],selected_position_cm=b['path_position_cm'],static_position_cm=a['static_path_position_cm'],
            final_rotation_deg=a['path_rotation_deg'],selected_rotation_deg=b['path_rotation_deg'],static_rotation_deg=a['static_path_rotation_deg']))
    analysis = dict(verified=True,job_id='205997',archive_sha256=sha(archive),inventory=inventory,
        saved_prediction_snapshots=len(snapshots),snapshots=snapshots,selected_update=768,final_update=6144,
        source_raw_input_targets_statistics_and_partition_verified=True,optimization_draws_per_window=512,
        internal_and_outer_optimizer_draws=0,first4_updates_replay_preflight=True,
        runtime_clip_and_frozen_invariants_passed=True,runtime_checkpoint_reload_exact=True,
        checkpoint_weights_downloaded=False,token_cache_downloaded=False,
        checkpoint_hashes_recorded_only={k:r[k] for k in ['selected_checkpoint_sha256','final_checkpoint_sha256']},
        max_report_metric_differences=max_deltas,inner_history=history,final=final,
        candidate_passed=False,no_new_training_encoding_inference_or_simulation=True,
        limitations=['Only one training initialization; original40 development windows repeatedly used, not untouched test.',
                     'Decoded outer labels entered identity checks, not optimization, scaling or checkpoint selection.',
                     'Parameter/Adam observer equality, CLIP invariance and checkpoint weight reload are runtime assertions; deleted/precluded PT tensors not independently reconstructed locally.',
                     'Verifies this fixed rule failed both primary metrics; does not exclude all overfitting or early-stopping mechanisms.'],
        stopping='Close fixed internal-selection candidate; no new seeds/rules/updates.')
    OUT.mkdir()
    (OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8')
    csv_write('inner_history.csv',[{k:v for k,v in h.items() if k not in ['static','episodes']} for h in history])
    csv_write('per_window.csv',per_windows); csv_write('paired_outer_episodes.csv',paired)
    csv_write('updates.csv', [dict(update=v['update'],loss=v['loss'],learning_rate=v['learning_rate']) for v in formal_logs])
    fig,axes = plt.subplots(1,2,figsize=(11.2,4),layout='constrained')
    ax=axes[0]
    for key,label in [('position_ratio','Position / static'),('rotation_ratio','Rotation / static')]:
        ax.plot([h['update'] for h in history],[h[key] for h in history],marker='.',label=label)
    ax.axhline(1,color='gray',linestyle='--',linewidth=1)
    ax.axvline(768,color='black',linestyle=':',label='Selected: 768')
    ax.set(title='Inner selection: whole-episode mean',xlabel='Training update',ylabel='Error relative to static')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1];labels=['Selected\ntrain','Final\ntrain','Selected\nouter','Final\nouter']
    cases=[final[k] for k in ['selected/train','final/train','selected/outer','final/outer']]
    x=np.arange(4)
    for offset,key,name in [(-.18,'path_position_cm','Position / static'),(.18,'path_rotation_deg','Rotation / static')]:
        heights=[v[key]/v['static'][key] for v in cases]
        bars=ax.bar(x+offset,heights,width=.36,label=name)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.axhline(1,color='gray',linestyle='--',linewidth=1)
    ax.set(xticks=x,xticklabels=labels,ylabel='Error relative to static',ylim=(0,1.23),title='Train fitting did not transfer to outer episodes')
    ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'selection_and_generalization.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(verified=True,files=len(inventory),snapshots=len(snapshots),selected_update=768,
        final_train={k:final['final/train'][k] for k in ['path_position_cm','path_rotation_deg','passing_windows','fit_gate']},
        selected_outer={k:selected[k] for k in ['path_position_cm','path_rotation_deg']},
        final_outer={k:last[k] for k in ['path_position_cm','path_rotation_deg']},max_differences=max_deltas),indent=2))


if __name__ == '__main__': main()
