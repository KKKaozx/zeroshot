"""Verify saved training-coverage evidence; no neural execution or weight loading."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np
from analyze_bridge_growth64_200600 import summarize

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_training_coverage_v1'
OLD = ROOT/'reports/bridge-state-fit-202278/real'
DATA = ROOT/'reports/bridge-coverage-fit-207348'
PRE = ROOT/'reports/bridge-coverage-preflight-207334'
OUT = ROOT/'reports/bridge_training_coverage_analysis_207348'
ARCHIVE = ROOT/'reports/bridge_training_coverage_results_207348.tar.gz'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(v) for v in p.read_text().splitlines()]
def identity(v): return v['shard'], v['record_index'], v['start_index']
def ep(v): return v['shard'], v['record_index']

def extract():
    archive = ARCHIVE
    inventory = []
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); seen = set(); total = 0
        for m in members:
            p = PurePosixPath(m.name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in m.name and ':' not in m.name
            assert p.parts[0] in {DATA.name, PRE.name} and (m.isdir() or m.isfile())
            if m.isdir(): continue
            assert len(p.parts) == 2 and p.suffix in {'.json','.jsonl','.npz','.py'}
            assert m.name.casefold() not in seen and m.size < 16*1024**2
            seen.add(m.name.casefold()); total += m.size
        assert len(seen) == (21 if PACK.name == 'bridge_rotation_weight_v1' else 17) and total < 100*1024**2
        for m in members:
            if not m.isfile(): continue
            dest = ROOT/'reports'/m.name
            assert dest.resolve().is_relative_to((ROOT/'reports').resolve())
            value = tar.extractfile(m).read(); assert len(value) == m.size
            if dest.exists(): assert dest.read_bytes() == value, 'Do not overwrite differing evidence'
            else:
                dest.parent.mkdir(exist_ok=True); dest.write_bytes(value)
            inventory.append(dict(path=m.name,bytes=m.size,sha256=sha(dest)))
    return archive,inventory

def main():
    global PACK, OLD, DATA, PRE, OUT, ARCHIVE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replication',action='store_true',help='Audit new seed42 replication207558 only; do not repeat207348.')
    ap.add_argument('--rotation-weight',action='store_true',help='Audit207746 only against already audited207558; no training.')
    args=ap.parse_args()
    assert not (args.replication and args.rotation_weight)
    if args.replication:
        PACK=ROOT/'cluster_packages/bridge_coverage_replication_v1'
        OLD=ROOT/'reports/bridge-state-rep-fit-202463/real'
        DATA=ROOT/'reports/bridge-coverage-rep-fit-207558'
        PRE=ROOT/'reports/bridge-coverage-rep-preflight-207521'
        OUT=ROOT/'reports/bridge_coverage_replication_analysis_207558'
        ARCHIVE=ROOT/'reports/bridge_coverage_replication_results_207558.tar.gz'
    if args.rotation_weight:
        PACK=ROOT/'cluster_packages/bridge_rotation_weight_v1'
        OLD=ROOT/'reports/bridge-state-rep-fit-202463/real'
        DATA=ROOT/'reports/bridge-rotation-weight-fit-207746'
        PRE=ROOT/'reports/bridge-rotation-weight-preflight-207722'
        OUT=ROOT/'reports/bridge_rotation_weight_analysis_207746'
        ARCHIVE=ROOT/'reports/bridge_rotation_weight_results_207746.tar.gz'
    assert not OUT.exists(), 'Completed audits must not be rerun'
    archive,inventory = extract()
    p=read(PACK/'protocol.json'); rows=read(PACK/'rows.json'); all_rows=rows['train']+rows['development']
    manifest=read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    old=read(OLD/'report.json'); r=read(DATA/'report.json'); pre=read(PRE/'report.json')
    assert sha(OLD/'report.json') == p['control_report_sha256']
    for path,h in p['inherited_sha256'].items(): assert sha(PACK.parent/path)==h, path
    for file,key in [('rows.json','rows_sha256'),('episode_reference.json','episode_reference_sha256'),('schedule.npz','schedule_file_sha256')]:
        assert sha(PACK/file)==p[key]
    audit=ROOT/'reports/bridge_expansion_source_audit_verified_v1.json'
    assert sha(audit)==p['source_audit_sha256'] and read(audit)['passed']
    reference=read(PACK/'episode_reference.json')
    assert reference['episodes']==read(audit)['episodes']
    refs={ep(v):v for v in reference['episodes']}
    assert len(rows['train'])==1836 and len(rows['development'])==40
    assert len(set(map(ep,rows['train'])))==405 and len(set(map(ep,rows['development'])))==10
    assert not set(map(ep,rows['train'])) & set(map(ep,rows['development']))
    assert list(map(identity,rows['train'][:128]))==list(map(identity,manifest['train']))
    assert list(map(identity,rows['development']))==list(map(identity,manifest['development']))
    assert len(set(map(identity,all_rows)))==1876
    for row in all_rows:
        e=refs[ep(row)]
        assert row['start_index'] in e['starts'] and row['start_index']+16<e['steps']
        assert row['partition']==e['partition'] and row['task']==e['instruction']
    with np.load(PACK/'schedule.npz',allow_pickle=False) as z: batches=z['batches'].copy()
    assert batches.shape==(8192,8) and batches.min()==0 and batches.max()==1835
    assert hashlib.sha256(batches.tobytes()).hexdigest()==p['schedule_tensor_sha256']
    assert all(len(set(v))==8 for v in batches)
    exposure=np.bincount(batches.ravel(),minlength=1836)
    assert (exposure.min(),exposure.max())==(23,72)
    assert Counter(all_rows[i]['task'] for i in batches.ravel())==p['schedule']['task_draw_counts']
    # The original Torch schedule was independently reconstructed in the prior local audit.
    prior=read(ROOT/'reports/bridge_training_coverage_local_preflight_v2_20261016/independent_audit.json')
    original_pack=ROOT/'cluster_packages/bridge_training_coverage_v1'
    assert prior['verified'] and prior['all65536_task_slots_exact'] and prior['protocol_sha256']==sha(original_pack/'protocol.json')
    if args.replication:
        proof=read(PACK/'reuse_control_proof.json')
        assert proof['verified'] and proof['control_report_sha256']==sha(OLD/'report.json')
        assert sha(PACK/'reuse_control_proof.json')==p['replication']['reuse_proof_sha256']
        assert p['cache_provenance_protocol_sha256']==sha(original_pack/'protocol.json')
        assert p['recipe']['initialization_seed']==42 and p['replication']['allowed_additional_initializations']==1
        for name in ['rows.json','schedule.npz','episode_reference.json']:
            assert (PACK/name).read_bytes()==(original_pack/name).read_bytes()
        origin_pre=ROOT/'reports/bridge-coverage-preflight-207334/report.json'
        origin_fit=ROOT/'reports/bridge-coverage-fit-207348/report.json'
        assert sha(origin_pre)==p['replication']['prior_preflight_report_sha256']
        assert sha(origin_fit)==p['replication']['prior_formal_report_sha256']
        for key in ['input_identity_sha256','token_cache_key_sha256','input_row_checks','episode_payload_checks']:
            assert r[key]==pre[key]==read(origin_pre)[key],key
        assert pre['full_token_cache_reused']
        prior_audit=ROOT/'reports/bridge_training_coverage_analysis_207348/analysis.json'
        assert sha(prior_audit)==p['replication']['previous_analysis_sha256'] and read(prior_audit)['verified']
        local=read(ROOT/'reports/bridge_coverage_replication_local_preflight_20261016/independent_audit.json')
        assert local['verified'] and local['only_two_initialization_expressions_changed_in_runner'] and local['cache_provenance_only_helper_changes']
    if args.rotation_weight:
        control=ROOT/'reports/bridge-coverage-rep-fit-207558'
        control_report=read(control/'report.json')
        control_audit=ROOT/'reports/bridge_coverage_replication_analysis_207558/analysis.json'
        assert sha(control/'report.json')==p['comparison']['control_report_sha256']
        assert sha(control_audit)==p['comparison']['control_analysis_sha256'] and read(control_audit)['verified']
        assert p['recipe']==read(ROOT/'cluster_packages/bridge_coverage_replication_v1/protocol.json')['recipe']
        for name in ['rows.json','schedule.npz','episode_reference.json']:
            assert (PACK/name).read_bytes()==(original_pack/name).read_bytes()
        assert p['cache_provenance_protocol_sha256']==sha(original_pack/'protocol.json')
        for key in ['input_identity_sha256','token_cache_key_sha256','input_row_checks','episode_payload_checks','initial_shared_sha256','augmented_initial_sha256','torch_version']:
            assert r[key]==pre[key]==control_report[key],key
        for stage,folder in [(r,DATA),(pre,PRE)]:
            assert stage['rotation_loss_weight']==p['rotation_loss']['weight']
            assert stage['loss_helper_sha256']==sha(PACK/'rotation_loss.py')==sha(folder/'rotation_loss_snapshot.py')==p['rotation_loss']['helper_sha256']
            assert sha(folder/'protocol_snapshot.json')==sha(PACK/'protocol.json')
            assert stage['complete_training_loss_scale_verified'] and stage['resident_token_rows']==1876 and not stage['local_compact_token_storage']
            assert all(stage['loss_math'][k] for k in ['weight1_forward_and_gradient_exact','xyz_loss_gradient_coefficient_preserved','quaternion_gradient_multiplied_by_fixed_weight'])
            assert stage['loss_math']['maximum_xyz_gradient_difference']==0
    retained_map=[*range(128),*range(1836,1876)]
    for folder,report,stage,operations in [(DATA,r,'fit',8192),(PRE,pre,'preflight',16)]:
        assert report['stage']==stage and report['optimization_operations']==operations and not report['local_bounded']
        for key in ['execution_checks_passed','clip_unchanged','other_frozen_unchanged','full_input_contract_passed',
                    'reused_complete_geometry_audit','retained_numeric_contract_passed','original_token_cache_reused',
                    'retained_visual_tokens_exact','retained_language_tokens_exact']: assert report[key],key
        for key in ['downloads','simulation_executed','gripper_trained','reserved_test_targets_read',
                    'old_control_retrained','state_statistics_changed','local_full_token_cache_written']: assert not report[key],key
        for actual,expected in [('retained_data_checks','data_checks'),('retained_state_checks','state_checks'),
            ('retained_state_features_sha256','state_features_sha256'),('initial_shared_sha256','initial_shared_sha256'),
            ('augmented_initial_sha256','augmented_initial_sha256'),('torch_version','torch_version')]:
            assert report[actual]==old[expected],actual
        assert report['entry_sha256']==sha(PACK/'run_training_coverage.py')==sha(folder/'runner_snapshot.py')
        assert report['input_helper_sha256']==sha(PACK/'coverage_inputs.py')
        assert report['protocol_sha256']==sha(PACK/'protocol.json')
        assert report['sampling_steps']==100 and report['sampling_seeds']==[1101,1102,1103] and report['initialization_seed']==p['recipe']['initialization_seed']
        assert report['token_rows']==report['active_token_rows']==1876
        assert report['verified_row_indices']==list(range(1876)) and len(report['input_row_checks'])==1876
        assert [v['index'] for v in report['input_row_checks']]==list(range(1876))
        assert len(report['episode_payload_checks'])==415
        assert {ep(v):v['payload_sha256'] for v in report['episode_payload_checks']}=={k:v['payload_sha256'] for k,v in refs.items()}
        assert hashlib.sha256(json.dumps(dict(rows=report['input_row_checks'],episodes=report['episode_payload_checks']),sort_keys=True).encode()).hexdigest()==report['input_identity_sha256']
        assert len(report['retained_numeric_comparisons'])==168
        assert all(v['raw_identity_exact'] and all(d['max_abs']<=d['atol'] for d in v['differences'].values()) for v in report['retained_numeric_comparisons'])
    for key in ['input_identity_sha256','token_cache_key_sha256','input_row_checks','episode_payload_checks']:
        assert r[key]==pre[key],key
    assert r['full_token_cache_reused'] and pre['branch_restore_passed']
    assert not pre['development_scored'] and r['development_scored']
    assert r['final_update']==8192 and r['branch_update']==4096 and pre['final_update']==8
    assert not pre['checkpoints'] and len(r['checkpoints'])==1
    assert r['checkpoints'][0]['roundtrip_exact'] and r['checkpoints'][0]['bytes']<1024**3
    assert pre['peak_reserved_gib']<7.5 and pre['estimated_formal_seconds']<9000
    logs={}
    for folder,groups in [(DATA,[('common',1,4096,1e-4),('reduced',4097,8192,1e-5)]),
        (PRE,[('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)])]:
        for arm,lo,hi,lr in groups:
            entries=lines(folder/f'{arm}_updates.jsonl');logs[(folder.name,arm)]=entries
            assert [v['update'] for v in entries]==list(range(lo,hi+1))
            assert all(v['learning_rate']==lr and np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm']) for v in entries)
            assert all(set(v['unclipped_module_gradient_norms'])=={'adapter.','diffusion_decoder.','history_projection.','state_projection.'} for v in entries)
            assert all(all(np.isfinite(g) and g>=0 for g in v['unclipped_module_gradient_norms'].values()) for v in entries)
            if args.rotation_weight:
                for v in entries:
                    base,xyz,q,weighted=[v[k] for k in ['unweighted_pose_mse','xyz_component_mse','quaternion_component_mse','weighted_rotation_contribution']]
                    w=p['rotation_loss']['weight']
                    assert np.isfinite([base,xyz,q,weighted]).all() and min(base,xyz,q,weighted)>=0
                    assert np.isclose(base,(3*xyz+4*q)/7,atol=1e-6,rtol=2e-6)
                    assert np.isclose(weighted,w*4/7*q,atol=1e-6,rtol=2e-6)
                    assert np.isclose(v['loss'],base+(w-1)*4/7*q,atol=1e-6,rtol=2e-6)
    assert logs[(DATA.name,'common')][:4]==logs[(PRE.name,'common')]
    assert logs[(PRE.name,'constant')]==logs[(PRE.name,'restore_same_lr')]
    snapshots=[]; final={};delta=dict(position_cm=0.,rotation_deg=0.)
    def compare(actual,recorded):
        for k,v in actual.items():
            kind='rotation_deg' if 'rotation' in k else 'position_cm';d=abs(v-recorded[k]);delta[kind]=max(delta[kind],d)
            assert d<(.02 if 'rotation' in k else 1e-4),(k,v,recorded[k])
    for folder,report in [(DATA,r),(PRE,pre)]:
        cases=[('reduced',8192,'train',r['runs'][0]['final_train']),('reduced',8192,'development',r['runs'][0]['final_development'])] if folder==DATA else [
            ('common',0,'train',None),('common',4,'train',pre['common_train'])]+[(v['arm'],8,'train',v['final_train']) for v in pre['runs']]
        expected={'report.json','runner_snapshot.py'}|{f'{name}_updates.jsonl' for name in (['common','reduced'] if folder==DATA else ['common','constant','restore_same_lr','reduced'])}
        if args.rotation_weight: expected.update(['rotation_loss_snapshot.py','protocol_snapshot.json'])
        for arm,u,split,recorded in cases:
            file=folder/f'{arm}_u{u:04d}_{split}.npz';expected.add(file.name)
            with np.load(file,allow_pickle=False) as z: pred,truth,ix,ctx=[z[k].copy() for k in ['predictions','targets','row_indices','contexts']]
            ids=np.arange(1836,1876) if split=='development' else np.arange(1836) if folder==DATA else np.array([0,32,64,96])
            assert np.array_equal(ix,ids) and pred.shape==(3,len(ids),16,8) and truth.shape==(len(ids),16,8) and ctx.shape==(len(ids),1024)
            assert all(np.isfinite(v).all() for v in [pred,truth,ctx])
            assert np.linalg.norm(pred[...,3:7],axis=-1).min()>1e-8
            for i,target in zip(ids,truth): assert hashlib.sha256(target.tobytes()).hexdigest()==r['input_row_checks'][i]['target_sha256']
            with np.load(OLD/f'reduced_u8192_{split}.npz',allow_pickle=False) as z:
                retained=truth if split=='development' else truth[:128] if folder==DATA else truth
                baseline=z['targets'] if split=='development' or folder==DATA else z['targets'][ids]
                assert np.max(np.abs(retained-baseline))<=1e-6
            metrics,good,gate,pe,re=summarize(pred,truth)
            static=np.zeros_like(truth);static[...,6]=1;sm,_,_,_,_=summarize(static[None],truth)
            if recorded:
                compare(metrics,recorded);compare(sm,{k:recorded['static_'+k] for k in sm})
                assert recorded['fit_gate']['pose_fit_passed']==gate
                assert recorded['fit_gate']['windows_all_three_seeds_pass']==int(good.sum())
            snapshots.append(dict(stage=report['stage'],arm=arm,update=u,split=split,**metrics,passing_windows=int(good.sum()),fit_gate=gate))
            if folder==DATA: final[split]=dict(pred=pred,truth=truth,metrics=metrics,position=pe,rotation=re)
        assert {v.name for v in folder.iterdir()}==expected
    with np.load(PRE/'constant_u0008_train.npz') as a,np.load(PRE/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    loss_statistics=None
    if args.rotation_weight:
        assert sha(PACK/'training_loss_scale.json')==p['rotation_loss']['statistics_sha256']
        raw=final['train']['truth'];t=raw.astype(np.float64);xyz=t[...,:3];q=t[...,3:7]
        assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-5 and (q[...,3]>=0).all()
        pe=float(np.square(xyz).sum(-1).mean());qe=float(np.square(q-np.array([0.,0.,0.,1.])).sum(-1).mean())
        loss_statistics=dict(train_windows=1836,train_targets=29376,training_targets_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),xyz_static_squared_sum_mean=pe,quaternion_static_squared_sum_mean=qe,weight=pe/qe,quaternion_negative_w_count=0,xyz_std=xyz.reshape(-1,3).std(0).tolist(),quaternion_std=q.reshape(-1,4).std(0).tolist())
        stored=read(PACK/'training_loss_scale.json')
        for k,v in loss_statistics.items(): assert v==stored[k],k
        assert loss_statistics['weight']==p['rotation_loss']['weight']
        for split in ['train','development']:
            with np.load(control/f'reduced_u8192_{split}.npz',allow_pickle=False) as z:
                assert np.array_equal(z['targets'],final[split]['truth'])
    episodes=[];tasks=[];horizons=[];summary={};counts={}
    for split,value in final.items():
        truth=value['truth'];static=np.zeros_like(truth);static[...,6]=1
        sm,_,_,se,sr=summarize(static[None],truth)
        methods={'coverage':(value['position'],value['rotation']),'static':(se,sr)}
        summary[split]=dict(coverage=value['metrics'],static=sm)
        if split=='development':
            with np.load((control if args.rotation_weight else OLD)/'reduced_u8192_development.npz') as z: op=z['predictions']
            om,_,_,oe,orr=summarize(op,truth);methods['original']=(oe,orr);summary[split]['original']=om
        else:
            summary[split]['retained128']=summarize(value['pred'][:,:128],truth[:128])[0]
        for by,destination in [('episode',episodes),('task',tasks)]:
            groups=defaultdict(list)
            for i,row in enumerate(rows[split]): groups[ep(row) if by=='episode' else row['task']].append(i)
            for key,ids in groups.items():
                v=dict(split=split,group='::'.join(map(str,key)) if by=='episode' else key,task=rows[split][ids[0]]['task'],windows=len(ids))
                for name,(pos,rot) in methods.items():
                    for field,arr in [('position_cm',pos),('rotation_deg',rot)]:
                        v[name+'_'+field]=float(arr[:,ids].mean());v[name+'_endpoint_'+field]=float(arr[:,ids,-1].mean())
                destination.append(v)
        local=[v for v in episodes if v['split']==split]
        counts[split]=dict(position_better_static=sum(v['coverage_position_cm']<v['static_position_cm'] for v in local),
                           rotation_better_static=sum(v['coverage_rotation_deg']<v['static_rotation_deg'] for v in local),episodes=len(local))
        if split=='development':
            counts[split].update(position_better_original=sum(v['coverage_position_cm']<v['original_position_cm'] for v in local),
                                rotation_better_original=sum(v['coverage_rotation_deg']<v['original_rotation_deg'] for v in local))
        summary[split]['episode_equal']={n:{field:float(np.mean([v[n+'_'+field] for v in local])) for field in ['position_cm','rotation_deg']} for n in methods}
        for k in range(16):
            v=dict(split=split,target=k+1)
            for name,(pos,rot) in methods.items(): v[name+'_position_cm']=float(pos[...,k].mean());v[name+'_rotation_deg']=float(rot[...,k].mean())
            horizons.append(v)
    dev=summary['development'];screen=p['screen']
    gates=dict(position_threshold=dev['coverage']['path_position_cm']<=screen['position_at_most_cm'],
        rotation_threshold=dev['coverage']['path_rotation_deg']<=screen['rotation_at_most_deg'],
        position_better_old=dev['coverage']['path_position_cm']<dev['original']['path_position_cm'],
        rotation_better_old=dev['coverage']['path_rotation_deg']<dev['original']['path_rotation_deg'],
        **({'episode_rotation_majority':counts['development']['rotation_better_original']>=screen['episodes_rotation_better_than_control_at_least']} if args.rotation_weight else {'episode_position_majority':counts['development']['position_better_original']>=screen['episodes_position_better_than_reused_control_at_least']}))
    result=dict(verified=True,job_id='207746' if args.rotation_weight else '207558' if args.replication else '207348',initialization_seed=p['recipe']['initialization_seed'],archive_sha256=sha(archive),inventory=inventory,metrics=summary,
        episode_counts=counts,gates=gates,candidate_passed=all(gates.values()),snapshots=snapshots,max_report_metric_difference=delta,
        fixed_schedule=dict(updates=8192,batch_size=8,total_draws=65536,minimum_window_draws=int(exposure.min()),maximum_window_draws=int(exposure.max()),mean_window_draws=float(exposure.mean())),
        initial_and_retained_contracts_match_old=True,preflight_first4_losses_and_gradients_exact=True,
        complete415_payload_identities_match_prior_geometry_audit=True,checkpoint_metadata=r['checkpoints'],checkpoint_downloaded=False,
        peak_reserved_gib=r['peak_reserved_gib'],elapsed_seconds=r['elapsed_seconds'],new_neural_execution=False,
        limitations=['One training initialization; three diffusion sampling seeds are not training replications.',
            'Repeated development set, not an untouched final test.',
            'More episodes, more within-episode windows and fewer repetitions per window change together at fixed update budget.',
            'Cache and frozen-parameter integrity use runtime hashes/assertions, not a downloaded model re-execution.',
            'Offline pose only; gripper frozen; no robotic grasp or unseen-task conclusion.'])
    if args.rotation_weight:
        result.update(control_job='207558',training_only_loss_statistics=loss_statistics,all8192_logged_loss_components_verified=True,only_intended_objective_changed=True)
        result['limitations'][2]='Quaternion component weighting changes shared optimization and global gradient clipping; does not identify a unique generalization root cause or validate the whole rotation representation.'
    OUT.mkdir()
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,values in [('episodes',episodes),('tasks',tasks),('horizons',horizons)]:
        fields=list(dict.fromkeys(k for v in values for k in v))
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
    print(json.dumps(dict(verified=True,development=dev,episode_counts=counts['development'],gates=gates,candidate_passed=all(gates.values()),
        tasks=[v for v in tasks if v['split']=='development'],max_report_metric_difference=delta),indent=2))

if __name__=='__main__': main()
