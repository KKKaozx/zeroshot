"""Audit saved current-state pair on CPU and apply its preregistered screen."""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_bridge_growth64_200600 import sha, summarize
from analyze_bridge_training_trace_200039 import FIELDS

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/bridge-state-fit-202278'
PRE=ROOT/'reports/bridge-state-preflight-202277'
PACK=ROOT/'cluster_packages/bridge_state_conditioning_v1'
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
ORIGINAL=ROOT/'reports/bridge-growth128-fit-201663'
OUT=ROOT/'reports/bridge_state_analysis_202278_20261013'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global DATA, PRE, PACK, OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replication',action='store_true',help='Audit the one fixed initialization42 pair202463')
    args=ap.parse_args()
    if args.replication:
        DATA=ROOT/'reports/bridge-state-rep-fit-202463'
        PRE=ROOT/'reports/bridge-state-rep-preflight-202462'
        PACK=ROOT/'cluster_packages/bridge_state_replication_v1'
        OUT=ROOT/'reports/bridge_state_rep_analysis_202463_20261014'
    protocol=read(PACK/'protocol.json');manifest=read(BASE/'manifest.json')
    if args.replication:
        first=read(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')
        assert first['verified'] and first['candidate_screen_passed']
        assert protocol['replication']['original_analysis_sha256']==sha(ROOT/'reports/bridge_state_analysis_202278_20261013/analysis.json')
        assert protocol['replication']['original_pair_report_sha256']==sha(ROOT/'reports/bridge-state-fit-202278/report.json')
        assert protocol['initialization_seed']==42 and protocol['replication']['additional_initializations_allowed']==1
    pair=read(DATA/'report.json');pre=read(PRE/'report.json');pv=read(PRE/'verification.json')
    assert pair['passed'] and pair['stage']=='fit' and pre['passed'] and pv['passed']
    assert pv['report_sha256']==sha(PRE/'report.json')
    assert pair['protocol_sha256']==pre['protocol_sha256']==sha(PACK/'protocol.json')
    assert pair['source_sha256']==pre['source_sha256']
    assert all(sha(PACK/n)==h for n,h in pair['source_sha256'].items())
    assert all(sha(BASE/n)==h for n,h in protocol['base_files_sha256'].items())
    assert pair['original_control_report_sha256']==sha(ORIGINAL/'report.json')==protocol['control_report_sha256']
    assert protocol['rows']==manifest['train']+manifest['development']
    assert sha(BASE/'manifest.json')==protocol['base_manifest_sha256']
    train_ep={(r['shard'],r['record_index']) for r in manifest['train']}
    dev_ep={(r['shard'],r['record_index']) for r in manifest['development']}
    assert len(train_ep)==64 and len(dev_ep)==10 and not train_ep&dev_ep
    origin=read(ORIGINAL/'report.json');reports={};values={};history=[];episodes=[];tasks=[];windows=[];deltas=dict.fromkeys(FIELDS,0.)
    trace_path=ROOT/'reports/bridge-trace-fit-200039/report.json'
    assert sha(trace_path)==manifest['late_lr']['origin_report_sha256']
    initial_reference=next(r for r in read(trace_path)['runs'] if r['initialization_seed']==protocol['initialization_seed'])
    hashes={};required_count=1
    def compare(values,recorded):
        for k,v in values.items():
            delta=abs(v-recorded[k]);deltas[k]=max(deltas[k],delta)
            assert delta<(.02 if 'rotation' in k else 1e-4),(k,v,recorded[k])
    for arm in ('zero','real'):
        d=DATA/arm;r=read(d/'report.json');reports[arm]=r
        assert pair['runs'][arm]['report_sha256']==sha(d/'report.json')
        assert r['stage']=='fit' and r['state_arm']==arm and r['initialization_seed']==protocol['initialization_seed']
        assert r['manifest_sha256']==sha(BASE/'manifest.json') and r['source_sha256']==manifest['source_sha256']
        assert sha(d/'runner_snapshot.py')==pair['source_sha256']['fit_state_arm.py']
        assert read(d/'plan.json')['manifest']==manifest
        for k in ('execution_checks_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged'):
            assert r[k]
        for k in ('gripper_trained','simulation_executed','downloads','reserved_test_targets_read'):assert not r[k]
        assert r['state_protocol_sha256']==sha(PACK/'protocol.json')
        sc=r['state_checks'];assert sc['raw_current_pose_exact'] and sc['train_only_statistics'] and sc['raw_windows']==168
        assert not sc['target_or_future_state_used_as_context']
        assert sc['features_sha256']==pv['inputs']['features_sha256']
        assert r['data_checks']==read(PRE/arm/'report.json')['data_checks']
        assert len(r['data_checks'])==len(r['input_numeric_comparisons'])==168
        assert all(a['raw_identity']==b['raw_identity'] for a,b in zip(r['data_checks'],origin['data_checks']))
        assert all(x['raw_identity_exact'] for x in r['input_numeric_comparisons'])
        assert all(v['max_abs']<=v['atol'] for x in r['input_numeric_comparisons'] for v in x['differences'].values())
        assert r['base_initial_shared_sha256']==r['initial_shared_sha256']==initial_reference['initial_shared_sha256']
        assert r['branch_update']==4096 and r['final_update']==8192 and r['sampling_steps']==100
        assert r['sampling_seeds']==manifest['sampling_seeds']==[1101,1102,1103]
        assert len(r['runs'])==1;run=r['runs'][0]
        assert run['arm']=='reduced' and run['learning_rate']==1e-5
        assert run['common_weights_sha256']==r['common_state_sha256'] and run['common_adam_sha256']==r['common_adam_sha256']
        assert run['actual_draw_slots_per_window']==[256]*128
        assert run['history'][-1]['train']==run['final_train']
        assert [x['update'] for x in r['common_history']]==[0,2048,4096]
        assert [x['update'] for x in run['history']]==[6144,7168,7680,8192]
        for name,lo,hi,lr in [('common',1,4096,1e-4),('reduced',4097,8192,1e-5)]:
            records=[json.loads(s) for s in (d/f'{name}_updates.jsonl').read_text().splitlines()]
            assert [x['update'] for x in records]==list(range(lo,hi+1))
            assert all(x['learning_rate']==lr and np.isfinite(x['loss']) and np.isfinite(x['global_unclipped_gradient_norm']) for x in records)
            expected={'adapter.','diffusion_decoder.','history_projection.','state_projection.'}
            assert all(set(x['unclipped_module_gradient_norms'])==expected for x in records)
            assert all(np.isfinite(v) and v>=0 for x in records for v in x['unclipped_module_gradient_norms'].values())
            state_grad=[x['unclipped_module_gradient_norms']['state_projection.'] for x in records]
            assert (all(v==0 for v in state_grad) if arm=='zero' else max(state_grad)>0)
            assert records[0]['loss']==(r['warmup_first_loss'] if name=='common' else run['loss_first'])
            if name=='reduced':assert records[-1]['loss']==run['loss_last'] and float(np.mean([x['loss'] for x in records]))==run['mean_loss']
        assert (run['state_port_weight_norm']==run['state_port_context_max_abs']==0 if arm=='zero' else run['state_port_weight_norm']>0 and run['state_port_context_max_abs']>0)
        cases=[('common',x['update'],'train',x['train']) for x in r['common_history']]
        cases += [('reduced',x['update'],'train',x['train']) for x in run['history']]
        cases += [('reduced',8192,'development',run['final_development'])]
        files=['report.json','plan.json','runner_snapshot.py','common_updates.jsonl','reduced_updates.jsonl']
        for name,u,split,recorded in cases:
            file=f'{name}_u{u:04d}_{split}.npz';files.append(file)
            with np.load(d/file,allow_pickle=False) as z:
                pred,target,ix,ctx=(z[k].copy() for k in ('predictions','targets','row_indices','contexts'))
            rows=manifest[split];n=len(rows)
            assert pred.shape==(3,n,16,8) and target.shape==(n,16,8)
            assert ctx.shape==(n,1024) and np.isfinite(ctx).all()
            assert ix.tolist()==list(range(n)) if split=='train' else ix.tolist()==list(range(128,168))
            with np.load(ORIGINAL/f'reduced_u8192_{split}.npz',allow_pickle=False) as z:assert np.array_equal(target,z['targets'])
            m,good,gate,p,q=summarize(pred,target);compare(m,recorded)
            static=np.zeros_like(target);static[...,6]=1
            sm,_,_,sp,sq=summarize(static[None],target);compare(sm,{k:recorded['static_'+k] for k in FIELDS})
            if split=='train':
                assert gate==recorded['fit_gate']['pose_fit_passed'] and int(good.sum())==recorded['fit_gate']['windows_all_three_seeds_pass']
                for group,j in [('retained_train',np.arange(64)),('added_train',np.arange(64,128))]:
                    gm,gg,gfit,_,_=summarize(pred[:,j],target[j]);compare(gm,recorded[group])
                    assert gfit==recorded[group]['fit_gate']['pose_fit_passed'] and int(gg.sum())==recorded[group]['fit_gate']['windows_all_three_seeds_pass']
            history.append(dict(arm=arm,update=u,split=split,**m,passing_windows=int(good.sum()),fit_gate=gate))
            if u==8192:
                values[arm+'/'+split]=dict(metrics=m,static=sm,gate=gate,passing_windows=int(good.sum()),predictions=pred,targets=target,position=p,rotation=q)
                by_ep=defaultdict(list);by_task=defaultdict(list)
                for i,row in enumerate(rows):
                    episode=row['shard']+'::'+str(row['record_index']);by_ep[episode].append(i);by_task[row['task']].append(i)
                    windows.append(dict(arm=arm,split=split,window=i,episode=episode,task=row['task'],path_position_cm=float(p[:,i].mean()),path_rotation_deg=float(q[:,i].mean()),endpoint_position_cm=float(p[:,i,-1].mean()),endpoint_rotation_deg=float(q[:,i,-1].mean())))
                recorded_ep={e['episode']:e for e in recorded['per_episode']}
                assert set(by_ep)==set(recorded_ep)
                for ep,j in by_ep.items():
                    em,_,_,_,_=summarize(pred[:,j],target[j]);compare(em,recorded_ep[ep])
                    esm,_,_,_,_=summarize(static[None,j],target[j]);compare(esm,{k:recorded_ep[ep]['static_'+k] for k in FIELDS})
                    episodes.append(dict(arm=arm,split=split,episode=ep,task=rows[j[0]]['task'],windows=len(j),**em,**{'static_'+k:v for k,v in esm.items()}))
                for task,j in by_task.items():
                    tm,_,_,_,_=summarize(pred[:,j],target[j]);tasks.append(dict(arm=arm,split=split,task=task,windows=len(j),**tm))
        assert len(files)==13 and all((d/f).is_file() for f in files)
        hashes.update({arm+'/'+f:sha(d/f) for f in files});required_count+=len(files)
        assert len(r['checkpoints'])==1 and r['checkpoints'][0]['roundtrip_exact'] and not r['checkpoints'][0]['includes_optimizer']
    assert required_count==27 and not list(DATA.rglob('*.pt'))
    assert reports['zero']['data_checks']==reports['real']['data_checks']
    assert reports['zero']['augmented_initial_sha256']==reports['real']['augmented_initial_sha256']
    assert reports['zero']['warmup_first_loss']==reports['real']['warmup_first_loss']
    for split in ('train','development'):
        assert np.array_equal(values['zero/'+split]['targets'],values['real/'+split]['targets'])
    with np.load(DATA/'zero/common_u0000_train.npz') as a,np.load(DATA/'real/common_u0000_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    dev_zero={e['episode']:e for e in episodes if e['arm']=='zero' and e['split']=='development'}
    dev_real={e['episode']:e for e in episodes if e['arm']=='real' and e['split']=='development'}
    paired=[]
    for ep,a in dev_zero.items():
        b=dev_real[ep]
        paired.append(dict(episode=ep,task=a['task'],windows=a['windows'],
            **{'zero_'+k:a[k] for k in FIELDS},**{'real_'+k:b[k] for k in FIELDS},
            **{'real_minus_zero_'+k:b[k]-a[k] for k in FIELDS},
            **{'static_'+k:a['static_'+k] for k in FIELDS},
            position_improved=b['path_position_cm']<a['path_position_cm'],rotation_improved=b['path_rotation_deg']<a['path_rotation_deg'],
            real_position_better_than_static=b['path_position_cm']<a['static_path_position_cm']))
    z=values['zero/development']['metrics'];b=values['real/development']['metrics'];static=values['real/development']['static']
    count=sum(e['position_improved'] for e in paired)
    both_fit=all(values[a+'/train']['gate'] for a in ('zero','real'));assert pair['both_training_fit']==both_fit
    screen=both_fit and count>=6 and all(b[k]<z[k] for k in FIELDS[:2])
    static_pass=all(b[k]<static[k] for k in FIELDS[:2])
    replay={}
    if args.replication:
        replay=dict(not_applicable=True,reason='Initialization42 differs from original201663 initialization43; exact replay is verified within same-initialization preflight.')
    else:
        for split in ('train','development'):
            with np.load(ORIGINAL/f'reduced_u8192_{split}.npz') as a:
                replay[split]=dict(max_prediction_difference=float(np.abs(a['predictions']-values['zero/'+split]['predictions']).max()),targets_exact=np.array_equal(a['targets'],values['zero/'+split]['targets']))
    result=dict(verified=True,formal_required_files=27,prediction_snapshots_verified=16,updates_per_arm_verified=8192,
        both_training_fit=both_fit,training_passing_windows={a:values[a+'/train']['passing_windows'] for a in ('zero','real')},
        train={a:values[a+'/train']['metrics'] for a in ('zero','real')},development=dict(zero=z,real=b,static=static),
        development_demos_position_improved=count,development_demos_rotation_improved=sum(e['rotation_improved'] for e in paired),
        development_demos_both_improved=sum(e['position_improved'] and e['rotation_improved'] for e in paired),
        real_demos_position_better_than_static=sum(e['real_position_better_than_static'] for e in paired),
        candidate_screen_passed=screen,development_static_protocol_passed=static_pass,
        episode_equal_weighted_development={a:{k:float(np.mean([e[k] for e in episodes if e['arm']==a and e['split']=='development'])) for k in FIELDS} for a in ('zero','real')},
        zero_original_prediction_comparison=replay,max_float64_metric_difference=deltas,
        artifact_sha256=hashes,preflight_verified=True,checkpoints_downloaded=False,
        checkpoint_contents_verified_locally=False,cluster_checkpoint_total_bytes=sum(reports[a]['checkpoints'][0]['bytes'] for a in ('zero','real')),
        action_context_state_checks_verified=True,initialization_seed=protocol['initialization_seed'],init_replication_not_executed=not args.replication,
        next_step=('One additional paired initialization allowed by prespecified candidate screen; no root-cause/generalization verdict.' if screen else 'Stop this state-injection candidate; do not sweep state layers or weights. Prepare same-Bridge author implementation reference per existing plan.'),
        limits=['One training initialization; three sampling seeds are not independent training replications.',
            'Development reused for diagnostics; no untouched test or closed-loop task success.'])
    if args.replication:
        first_csv=ROOT/'reports/bridge_state_analysis_202278_20261013/paired_development_episodes.csv'
        with first_csv.open(encoding='utf-8-sig',newline='') as f:old={r['episode']:r for r in csv.DictReader(f)}
        assert set(old)==set(dev_zero)
        result['replication']=dict(first_pair_job=202278,replication_pair_job=202463,
            initialization_seeds=[43,42],candidate_screen_repeated=bool(screen),
            position_improved_in_both_initializations=sum(e['position_improved'] and old[e['episode']]['position_improved']=='True' for e in paired),
            rotation_improved_in_both_initializations=sum(e['rotation_improved'] and old[e['episode']]['rotation_improved']=='True' for e in paired),
            paired_development_change_by_initialization={
                '43':{k:first['development']['real'][k]-first['development']['zero'][k] for k in FIELDS},
                '42':{k:b[k]-z[k] for k in FIELDS}},
            no_further_seed_selection=True,protected_test_evaluated=False)
        result['all_training_windows_fit']={a:values[a+'/train']['passing_windows']==128 for a in ('zero','real')}
        result['training_gate_definition']=dict(window_mean_position_cm_max=.5,window_mean_rotation_deg_max=5,
            fraction_all_three_sampling_seeds_pass_min=.9,
            per_sampling_seed_mean_position_cm_max=.5,per_sampling_seed_mean_rotation_deg_max=5,
            per_sampling_seed_endpoint_mean_position_cm_max=1,per_sampling_seed_endpoint_mean_rotation_deg_max=10)
        result['next_step']=('Candidate screen repeated and development beats static; fix recipe before any untouched-test evaluation, no further state/seed sweep.' if screen and static_pass else
            'Candidate screen repeated but development still worse than static: close this bounded state experiment as partial benefit; prepare the same-Bridge author implementation reference/interface plan, without further state or seed sweeps.' if screen else
            'Candidate screen did not replicate: stop this state-injection candidate without further seeds/layers/weight sweeps. Prepare same-Bridge author implementation reference per existing plan.')
        result['limits'][0]='Two fixed training initializations; three sampling seeds per initialization are not independent training replications.'
    OUT.mkdir(exist_ok=True)
    for name,records in [('history.csv',history),('windows.csv',windows),('episodes.csv',episodes),('tasks.csv',tasks),('paired_development_episodes.csv',paired)]:write_csv(name,records)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('verified','both_training_fit','training_passing_windows','development','development_demos_position_improved','development_demos_rotation_improved','development_demos_both_improved','candidate_screen_passed','development_static_protocol_passed','zero_original_prediction_comparison','next_step')},indent=2))

if __name__=='__main__':main()
