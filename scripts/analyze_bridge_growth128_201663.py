"""Independently audit saved 128-window artifacts on CPU; never train."""
import csv
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_bridge_growth64_200600 import sha, summarize
from analyze_bridge_training_trace_200039 import FIELDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/bridge-growth128-fit-201663'
PACK = ROOT / 'cluster_packages/bridge_growth128_v1'
OUT = ROOT / 'reports/bridge_growth128_analysis_201663_20261013'


def main():
    global DATA, PACK, OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window-control',action='store_true')
    args=ap.parse_args()
    wc=args.window_control
    if wc:
        DATA=ROOT/'reports/bridge-window-fit-202069'
        PACK=ROOT/'cluster_packages/bridge_window_coverage_v1'
        OUT=ROOT/'reports/bridge_window_analysis_202069_20261013'
    report = json.loads((DATA/'report.json').read_text())
    manifest = json.loads((DATA/'plan.json').read_text())['manifest']
    baseline_dir = ROOT/('reports/bridge-growth128-fit-201663' if wc else 'reports/bridge-growth64-fit-200600')
    baseline = json.loads((baseline_dir/'report.json').read_text())
    base_pack=ROOT/('cluster_packages/bridge_growth128_v1' if wc else 'cluster_packages/bridge_growth64_v1')
    base_manifest = json.loads((base_pack/'manifest.json').read_text())
    assert manifest == json.loads((PACK/'manifest.json').read_text())
    assert report['stage'] == 'fit' and report['initialization_seed'] == 43
    assert report['manifest_sha256'] == sha(PACK/'manifest.json')
    assert report['source_sha256'] == manifest['source_sha256']
    assert all(sha(PACK/n) == h for n,h in report['source_sha256'].items())
    entry='fit_bridge_window_coverage.py' if wc else 'fit_bridge_growth128.py'
    assert sha(DATA/'runner_snapshot.py') == report['source_sha256'][entry]
    if wc:
        import ast
        assert json.loads((ROOT/'reports/bridge_growth128_analysis_201663_20261013/analysis.json').read_text())['verified']
        coverage=manifest['window_coverage'];changed=coverage['changed_row_indices']
        assert len(changed)==len(set(changed))==39
        assert changed==[i for i,(a,b) in enumerate(zip(manifest['train'],base_manifest['train'])) if a!=b]
        assert all((a['shard'],a['record_index'],a['task'])==(b['shard'],b['record_index'],b['task']) for a,b in zip(manifest['train'],base_manifest['train']))
        assert sha(baseline_dir/'report.json')==coverage['control_report_sha256']
        assert all(sha(base_pack/n)==h for n,h in coverage['base_files_sha256'].items())
        assert sha(PACK/'selection_plan.json')==coverage['selection_plan_sha256']
        assert all(sha(PACK/'changed_reference'/n)==h for n,h in coverage['new_reference_sha256'].items())
        for key in ('config','batch_size','schedule_sha256','updates_per_arm','draw_slots_per_window','sampling_seeds','learning_rate_schedule','trace'):
            assert manifest[key]==base_manifest[key],key
        assert all(h==base_manifest['source_sha256'][n] for n,h in manifest['source_sha256'].items() if n not in (entry,'window_coverage_input_contract.py'))
        trees=[ast.parse((p/n).read_text()) for p,n in [(PACK,entry),(base_pack,'fit_bridge_growth128.py')]]
        for name in ('context','advance','cpu_copy','state_sha','save_checkpoint'):
            nodes=[next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            assert ast.dump(nodes[0])==ast.dump(nodes[1]),name
        assert report['unchanged_input_raw_identity_passed'] and report['unchanged_input_cluster_tensors_exact']
        assert report['window_coverage_control']==coverage
        preflight_path=ROOT/'reports/bridge-window-preflight-202068/report.json'
        preflight=json.loads(preflight_path.read_text())
        assert all(preflight[k] for k in ('execution_checks_passed','branch_restore_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged','unchanged_input_raw_identity_passed'))
        assert preflight['stage']=='preflight' and preflight['manifest_sha256']==report['manifest_sha256']
        assert preflight['source_sha256']==report['source_sha256'] and preflight['data_checks']==report['data_checks']
        assert preflight['initial_shared_sha256']==report['initial_shared_sha256']
        assert preflight['branch_restore']['max_loss_difference']==preflight['branch_restore']['max_prediction_difference']==0
    else:
        assert sha(baseline_dir/'report.json') == manifest['growth']['retained64_report_sha256']
        assert manifest['train'][:64] == base_manifest['train']
    assert manifest['development'] == base_manifest['development']
    assert manifest['config'] == base_manifest['config']
    assert report['initial_shared_sha256'] == baseline['initial_shared_sha256']
    assert all(report[k] for k in ('execution_checks_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged'))
    assert not any(report[k] for k in ('gripper_trained','simulation_executed','downloads','reserved_test_targets_read'))
    assert len(report['data_checks']) == len(report['input_numeric_comparisons']) == 168
    if wc:
        def audit_index(i):return i if i<32 else (i+40 if i<128 else i-96)
        assert all(report['data_checks'][audit_index(i)]==baseline['data_checks'][audit_index(i)] for i in range(168) if i not in changed)
        metadata=json.loads((PACK/'changed_reference/metadata.json').read_text())
        assert metadata['rows']==[manifest['train'][i] for i in changed]
        assert all(report['data_checks'][audit_index(i)]['raw_identity']==metadata['raw_inputs'][j] for j,i in enumerate(changed))
    else: assert report['data_checks'][:104] == baseline['data_checks']
    assert all(c['raw_identity_exact'] and all(d['max_abs'] <= d['atol'] for d in c['differences'].values()) for c in report['input_numeric_comparisons'])
    assert report['sampling_steps'] == 100 and report['sampling_seeds'] == [1101,1102,1103]
    assert report['branch_update'] == 4096 and report['final_update'] == 8192
    identity = lambda r: (r['shard'],r['record_index'])
    old,new,dev = (set(map(identity,rows)) for rows in (manifest['train'][:64],manifest['train'][64:],manifest['development']))
    assert (len(old),len(new),len(dev)) == (32,32,10)
    assert not old & new and not (old|new) & dev
    assert len(report['runs']) == 1
    run = report['runs'][0]
    assert run['arm'] == 'reduced' and run['learning_rate'] == 1e-5
    assert run['common_weights_sha256'] == report['common_state_sha256']
    assert run['common_adam_sha256'] == report['common_adam_sha256']
    assert run['actual_draw_slots_per_window'] == [256]*128
    assert run['history'][-1]['train'] == run['final_train']
    assert [h['update'] for h in report['common_history']] == [0,2048,4096]
    assert [h['update'] for h in run['history']] == [6144,7168,7680,8192]
    for arm,lo,hi,lr in [('common',1,4096,1e-4),('reduced',4097,8192,1e-5)]:
        logs = [json.loads(s) for s in (DATA/f'{arm}_updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in logs] == list(range(lo,hi+1))
        assert all(r['learning_rate'] == lr for r in logs)
        assert np.isfinite([r['loss'] for r in logs]).all()
        assert all(set(r['unclipped_module_gradient_norms']) == {'adapter.','diffusion_decoder.','history_projection.'} for r in logs)
        assert all(np.isfinite(v) and v > 0 for r in logs for v in r['unclipped_module_gradient_norms'].values())
        assert all(np.isfinite(r['global_unclipped_gradient_norm']) for r in logs)
        assert logs[0]['loss'] == (report['warmup_first_loss'] if arm == 'common' else run['loss_first'])
        if arm == 'reduced':
            assert logs[-1]['loss'] == run['loss_last']
            assert float(np.mean([r['loss'] for r in logs])) == run['mean_loss']
    cases = [('common',h['update'],'train',h['train']) for h in report['common_history']]
    cases += [('reduced',h['update'],'train',h['train']) for h in run['history']]
    cases += [('reduced',8192,'development',run['final_development'])]
    required = ['report.json','plan.json','runner_snapshot.py','common_updates.jsonl','reduced_updates.jsonl']
    required += [f'{a}_u{u:04d}_{s}.npz' for a,u,s,_ in cases]
    missing = [n for n in required if not (DATA/n).is_file()]
    labels,history,windows,final,episodes = {},[],[],{},[]
    deltas = dict.fromkeys(FIELDS,0.)
    def compare(metrics,recorded):
        for k,v in metrics.items():
            deltas[k] = max(deltas[k],abs(v-recorded[k]))
            assert abs(v-recorded[k]) < (.02 if 'rotation' in k else 1e-4), (k,v,recorded[k])
    for arm,update,split,recorded in cases:
        file = DATA/f'{arm}_u{update:04d}_{split}.npz'
        if not file.is_file():
            continue  # Missing snapshots explicitly prevent the complete verification verdict.
        with np.load(file,allow_pickle=False) as z:
            pred,target,indices,context = (z[k] for k in ('predictions','targets','row_indices','contexts'))
        rows = manifest[split]; n = len(rows)
        assert pred.shape == (3,n,16,8) and target.shape == (n,16,8)
        assert context.shape == (n,1024) and np.isfinite(context).all()
        assert np.array_equal(indices,np.arange(n) if split == 'train' else np.arange(128,168))
        if split in labels: assert np.array_equal(target,labels[split])
        else: labels[split] = target.copy()
        static = np.zeros_like(target); static[...,6] = 1
        sm,_,_,_,_ = summarize(static[None],target)
        compare(sm,{k:recorded['static_'+k] for k in FIELDS})
        groups = [('overall',np.arange(n))]
        if split == 'train': groups += [('retained_train',np.arange(64)),('added_train',np.arange(64,128))]
        for group,ix in groups:
            rec = recorded if group == 'overall' else recorded[group]
            metrics,good,gate,p,r = summarize(pred[:,ix],target[ix]); compare(metrics,rec)
            if split == 'train':
                assert int(good.sum()) == rec['fit_gate']['windows_all_three_seeds_pass']
                assert gate == rec['fit_gate']['pose_fit_passed']
            summary = dict(arm=arm,update=update,split=split,group=group,**metrics,passing_windows=int(good.sum()),windows=len(ix),fit_gate=gate)
            history.append(summary)
            if update == 8192:
                final[split+'/'+group] = dict(**summary,worst_window_mean_position_mm=float(p.mean((0,2)).max()*10),largest_saved_target_position_mm=float(p.max()*10))
            if group == 'overall':
                for i,row in enumerate(rows):
                    windows.append(dict(arm=arm,update=update,split=split,window=i,episode=row['shard']+'::'+str(row['record_index']),task=row['task'],position_mm=float(p[:,i].mean()*10),rotation_deg=float(r[:,i].mean()),passing=bool(good[i])))
                for ep in rec['per_episode']:
                    ix = [j for j,row in enumerate(rows) if row['shard']+'::'+str(row['record_index']) == ep['episode']]
                    em,_,_,_,_ = summarize(pred[:,ix],target[ix]); compare(em,ep)
                    if update == 8192:
                        esm,_,_,_,_ = summarize(static[None,ix],target[ix])
                        episodes.append(dict(split=split,episode=ep['episode'],task=ep['task'],windows=len(ix),**em,
                            **{'static_'+k:v for k,v in esm.items()},position_better_than_static=em['path_position_cm']<esm['path_position_cm']))
    if wc:
        unchanged=[i for i in range(128) if i not in changed]
        with np.load(baseline_dir/'reduced_u8192_train.npz') as z:
            assert np.array_equal(labels['train'][unchanged],z['targets'][unchanged])
        with np.load(PACK/'changed_reference/arrays.npz') as z:
            assert np.max(np.abs(labels['train'][changed]-z['targets'])) <= 1e-6
            assert all([hashlib.sha256(v.tobytes()).hexdigest() for v in z[k]]==metadata['processed_sha256'][k] for k in z.files)
    else:
        with np.load(baseline_dir/'reduced_u4096_train.npz') as z:
            assert np.array_equal(labels['train'][:64],z['targets'])
        with np.load(PACK/'new_reference/arrays.npz') as z:
            assert np.max(np.abs(labels['train'][64:]-z['targets'])) <= 1e-6
    if 'development' in labels:
        with np.load(baseline_dir/('reduced_u8192_development.npz' if wc else 'reduced_u4096_development.npz')) as z:
            assert np.array_equal(labels['development'],z['targets'])
    dev_episodes = run['final_development']['per_episode']
    result = dict(verified=not missing,metadata_and_runner_verified=True,updates_verified=8192,
        predictions_verified=len(cases)-len(missing),missing_files=missing,final=final,
        reduced_history_all_train_groups_pass=(all(h['fit_gate'] for h in history if h['arm']=='reduced' and h['split']=='train') if not missing else None),
        max_float64_metric_difference=deltas,downloaded_required_files=len(required)-len(missing),downloaded_bytes=sum((DATA/n).stat().st_size for n in required if (DATA/n).is_file()),
        elapsed_seconds=report['elapsed_seconds'],peak_reserved_gib=report['peak_reserved_gib'],
        reported_development_episodes_position_better_than_static=sum(e['path_position_cm']<e['static_path_position_cm'] for e in dev_episodes),
        independently_recomputed_development_episodes_position_better_than_static=(sum(e['position_better_than_static'] for e in episodes if e['split']=='development') if 'development' in labels else None),
        pt_files_downloaded=bool(list(DATA.glob('*.pt'))),checkpoint_bytes_reported=sum(c['bytes'] for c in report['checkpoints']),checkpoint_bytes_verified_locally=False,
        artifact_sha256={n:sha(DATA/n) for n in required if (DATA/n).is_file()})
    if wc:
        treatment_groups=[]
        for h in run['history']:
            with np.load(DATA/f"reduced_u{h['update']:04d}_train.npz") as z:
                for group,ix in [('unchanged89',unchanged),('substituted39',changed)]:
                    m,good,gate,_,_=summarize(z['predictions'][:,ix],z['targets'][ix])
                    treatment_groups.append(dict(update=h['update'],group=group,windows=len(ix),passing_windows=int(good.sum()),fit_gate=gate,**m))
        result['coverage_train_groups']=treatment_groups
        result['coverage_all_late_group_gates_pass']=all(r['fit_gate'] for r in treatment_groups)
        result['preflight_verified']=True
        result['preflight_report_sha256']=sha(preflight_path)
        result['matched_control_initialization_and_unchanged_inputs_verified']=True
        result['slurm_final_state_not_supplied']=True
        paired=[];grouped=[]
        pool=json.loads((ROOT/'reports/bridge-xyz-eval-197349.json').read_text())['window_selection'];starts=defaultdict(list)
        for row in pool:starts[(row['shard'],row['record_index'])].append(row['start_index'])
        with np.load(DATA/'reduced_u8192_development.npz') as a,np.load(baseline_dir/'reduced_u8192_development.npz') as b:
            assert np.array_equal(a['targets'],b['targets'])
            _,_,_,pa,ra=summarize(a['predictions'],a['targets']);_,_,_,pb,rb=summarize(b['predictions'],b['targets'])
            static=np.zeros_like(a['targets']);static[...,6]=1
            _,_,_,ps,rs=summarize(static[None],a['targets'])
            for i,row in enumerate(manifest['development']):
                ss=starts[(row['shard'],row['record_index'])];t=row['start_index']
                phase='first' if t==min(ss) else ('last' if t==max(ss) else 'middle')
                paired.append(dict(window=i,episode=row['shard']+'::'+str(row['record_index']),task=row['task'],phase=phase,
                    control_position_cm=float(pb[:,i].mean()),treatment_position_cm=float(pa[:,i].mean()),static_position_cm=float(ps[:,i].mean()),
                    position_delta_cm=float(pa[:,i].mean()-pb[:,i].mean()),control_rotation_deg=float(rb[:,i].mean()),
                    treatment_rotation_deg=float(ra[:,i].mean()),static_rotation_deg=float(rs[:,i].mean()),rotation_delta_deg=float(ra[:,i].mean()-rb[:,i].mean())))
            for kind in ('overall','task','episode','phase'):
                keys=['overall'] if kind=='overall' else sorted({r[kind] for r in paired})
                for key in keys:
                    ix=[r['window'] for r in paired if kind=='overall' or r[kind]==key]
                    ma,_,_,_,_=summarize(a['predictions'][:,ix],a['targets'][ix]);mb,_,_,_,_=summarize(b['predictions'][:,ix],b['targets'][ix]);ms,_,_,_,_=summarize(static[None,ix],a['targets'][ix])
                    grouped.append(dict(kind=kind,key=key,windows=len(ix),**{'treatment_'+k:v for k,v in ma.items()},
                        **{'control_'+k:v for k,v in mb.items()},**{'static_'+k:v for k,v in ms.items()},
                        position_delta_cm=ma['path_position_cm']-mb['path_position_cm'],rotation_delta_deg=ma['path_rotation_deg']-mb['path_rotation_deg']))
        result['paired_development_groups']=grouped
        result['position_improved_vs_control_episodes']=sum(r['position_delta_cm']<0 for r in grouped if r['kind']=='episode')
        result['both_pose_metrics_improved_vs_control_episodes']=sum(r['position_delta_cm']<0 and r['rotation_delta_deg']<0 for r in grouped if r['kind']=='episode')
        result['verdict']='Both training subgroups fit; same-size window substitution did not improve aggregate held-out pose prediction. Do not adopt or expand this branch; no unique root cause established.' if result['coverage_all_late_group_gates_pass'] else 'Inspect training subgroup fitting before interpreting development result.'
        history+=treatment_groups
        OUT.mkdir(exist_ok=True)
        for name,records in [('paired_development_windows.csv',paired),('paired_development_groups.csv',grouped),('coverage_train_groups.csv',treatment_groups)]:
            with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    OUT.mkdir(exist_ok=True)
    for name,records in [('history.csv',history),('windows.csv',windows),('final_episodes.csv',episodes)]:
        if not records: continue
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    printed=result if not wc else {k:result[k] for k in ('verified','updates_verified','predictions_verified','missing_files','downloaded_bytes','final','coverage_train_groups','coverage_all_late_group_gates_pass','position_improved_vs_control_episodes','both_pose_metrics_improved_vs_control_episodes','verdict')}
    print(json.dumps({k:v for k,v in printed.items() if k!='artifact_sha256'},indent=2))


if __name__ == '__main__': main()
