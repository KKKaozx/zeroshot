"""Independent CPU audit of the fixed trial, including saved physical trajectories."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_bridge_end_to_end_20261015 import qmatrix
from analyze_bridge_growth64_200600 import summarize

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/panda_spatial_fixed_trial_20261015'
DATA=ROOT/'reports/panda_spatial_control_v2_preparation_20261015'
OUT=ROOT/'reports/panda_spatial_fixed_trial_analysis_20261015'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def grip(pred,target):
    actual=target[...,7]>0;guess=pred[...,7]>0
    acc=(guess==actual[None]).mean((1,2))
    bacc=np.mean([(guess[:,actual==v]==v).mean(1) for v in [False,True]],axis=0)
    events={}
    for label,before,after in [('open_to_closed',True,False),('closed_to_open',False,True)]:
        event=(actual[:,:-1]==before)&(actual[:,1:]==after)
        correct=(guess[:,:,:-1]==before)&(guess[:,:,1:]==after)
        counts=correct[:,event].sum(1)
        events[label]=dict(target_pairs=int(event.sum()),correct_pairs=counts.tolist(),
            accuracy=float(counts.mean()/event.sum()) if event.any() else None)
    return dict(accuracy=float(acc.mean()),balanced_accuracy=float(bacc.mean()),seed_accuracy=acc.tolist(),seed_balanced_accuracy=bacc.tolist(),events=events)


def audit_offline(plan,record):
    protocol=read(DATA/'protocol.json');result={};horizon=[];phases=[]
    for split in ['train','development']:
        rows=[dict(episode=r['episode'],start=t) for r in protocol['episodes'] if r['split']==split for t in protocol['starts']]
        with np.load(RUN/f'offline_{split}.npz') as z:
            pred,target,indices=z['predictions'],z['targets'],z['row_indices']
            assert z['contexts'].shape==(len(rows),1024) and np.isfinite(z['contexts']).all()
        assert pred.shape==(3,len(rows),16,8) and target.shape==(len(rows),16,8)
        assert np.array_equal(indices,np.arange(len(rows)))
        m,passed,gate,p,r=summarize(pred,target)
        old=record['groups'][split]
        for field in m:
            source_name={'path_position_cm':'position_cm','path_rotation_deg':'rotation_deg'}.get(field,field)
            assert abs(m[field]-old[source_name]) < (.02 if 'rotation' in field else 1e-4)
        assert int(passed.sum())==old['windows_all_three_seeds_pass'] and gate==old['pose_fit_passed']
        gr=grip(pred,target)
        for k in ['accuracy','balanced_accuracy']:assert abs(gr[k]-old['gripper'][k])<1e-12
        for event,values in gr['events'].items():
            recorded=old['gripper'][event]
            assert values['target_pairs']==recorded['target_pairs'] and values['correct_pairs']==recorded['seed_correct_pairs']
        static=np.zeros_like(target);static[...,6]=1
        sm,_,_,_,_=summarize(static[None],target)
        assert abs(sm['path_position_cm']-old['static_position_cm'])<1e-4
        assert abs(sm['path_rotation_deg']-old['static_rotation_deg'])<.02
        # Compare saved target positions against original future measured states with an independent matrix.
        for i,row in enumerate(rows):
            with np.load(DATA/'episodes'/(row['episode']+'.npz')) as a:
                t=row['start'];R=qmatrix(a['rotations'][t])
                restored=a['positions'][t]+(R @ (target[i,:,:3].astype(float)*.1).T).T
                assert np.abs(restored-a['positions'][t+1:t+17]).max()<2e-7
                assert np.array_equal(target[i,:,7],a['commands'][t:t+16])
        result[split]=dict(metrics=m,gripper=gr,pose_fit_passed=gate,passing_windows=int(passed.sum()),windows=len(rows),
            static_position_cm=sm['path_position_cm'],static_rotation_deg=sm['path_rotation_deg'])
        for t in range(16):horizon.append(dict(split=split,recorded_future_step=t+1,position_cm=float(p[:,:,t].mean()),rotation_deg=float(r[:,:,t].mean())))
        for start in protocol['starts']:
            selected=np.array([row['start']==start for row in rows])
            phases.append(dict(split=split,observation_start=start,windows=int(selected.sum()),
                position_cm=float(p[:,selected].mean()),rotation_deg=float(r[:,selected].mean()),
                endpoint_position_cm=float(p[:,selected,-1].mean())))
    return result,horizon,phases


def audit_rollouts(plan,record):
    protocol=read(DATA/'protocol.json');rows={r['episode']:r for r in protocol['episodes'] if r['split']=='development'}
    assert record['complete'] and len(record['results'])==12
    assert {(r['episode'],r['sampling_seed']) for r in record['results']}=={(n,s) for n in rows for s in plan['sample_seeds']}
    table=[];max_decode=0.;max_reference=0.
    for r in record['results']:
        path=RUN/'rollouts'/f"{r['episode']}_seed{r['sampling_seed']}.npz"
        assert sha(path)==r['trajectory_sha256']
        with np.load(path) as z:
            a={k:z[k].copy() for k in z.files}
        assert a['actions'].shape==(112,8) and a['images'].shape==(29,224,224,3)
        assert a['replan_reference'].shape==(28,8) and a['finger_contacts'].shape==(112,2)
        assert all(np.isfinite(v).all() for v in a.values())
        expected=[]
        for k,ref in enumerate(a['replan_reference']):
            expected.extend(ref[:3]+(qmatrix(ref[3:7]) @ (a['actions'][k*4:k*4+4,:3].astype(float)*.1).T).T)
            if k:
                max_reference=max(max_reference,float(np.abs(ref[:3]-a['ee_positions'][k*4-1]).max()),
                    abs(float(ref[7]-a['measured_widths'][k*4-1])))
        delta=float(np.abs(np.asarray(expected)-a['unclipped_xyz']).max());max_decode=max(max_decode,delta)
        assert delta<2e-7 and max_reference<1e-10
        clipped=np.clip(a['unclipped_xyz'],plan['rollout']['position_min'],plan['rollout']['position_max'])
        assert np.array_equal(clipped,a['requested_poses'][:,:3])
        assert np.array_equal(a['actions'][:,7],a['requested_poses'][:,7])
        clipping=int((np.abs(clipped-a['unclipped_xyz'])>1e-12).any(1).sum())
        assert clipping==r['workspace_clipped_commands']
        with np.load(DATA/'episodes'/(r['episode']+'.npz')) as original:
            assert abs(float(original['object_heights'][0])-r['initial_height_m'])<1e-9
            assert np.abs(a['replan_reference'][0,:3]-original['positions'][0]).max()<1e-7
        h=a['object_heights'];contacts=a['finger_contacts']
        held=bool((h[-10:]>r['initial_height_m']+.10).all());success=held and bool(contacts[-1].all())
        assert held==r['last_ten_steps_held_above_10cm'] and success==r['success']
        assert int(contacts[-10:].all(1).sum())==r['two_finger_hold_steps']
        assert abs(float(h[-1])-r['final_height_m'])<1e-12
        assert not r['expert_handoff'] and not r['object_pose_given_to_model'] and not r['artificial_attachment']
        table.append(dict(episode=r['episode'],sampling_seed=r['sampling_seed'],success=success,lift_cm=r['lift_m']*100,
            final_two_finger_contact=bool(contacts[-1].all()),last_ten_two_finger=int(contacts[-10:].all(1).sum()),workspace_clipped_commands=clipping))
    counts={name:sum(r['success'] for r in table if r['episode']==name) for name in rows}
    assert counts==record['per_position_successes']
    assert sum(n>=2 for n in counts.values())==record['positions_with_at_least_two_successes']
    assert (sum(n>=2 for n in counts.values())>=3)==record['development_screen_passed']
    return table,dict(per_position_successes=counts,successes=sum(r['success'] for r in table),rollouts=12,
        screen_passed=record['development_screen_passed'],max_independent_decode_difference_m=max_decode,max_reference_difference=max_reference)


def main():
    plan,training,offline,rollout=(read(RUN/n) for n in ['plan.json','training_report.json','offline_report.json','rollout_report.json'])
    assert training['complete'] and training['updates_completed']==8192
    assert plan['source_sha256']==training['source_sha256'] and all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
    assert sha(RUN/'runner_snapshot.py')==sha(ROOT/'scripts/run_panda_spatial_fixed_trial_20261015.py')
    checkpoint=sha(RUN/'final.pt')
    assert checkpoint==training['checkpoint_sha256']==offline['checkpoint_sha256']==rollout['checkpoint_sha256']
    assert training['changed']==dict(clip=False,adapter=True,decoder=True,gripper=True)
    logs=[json.loads(line) for line in (RUN/'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in logs]==list(range(1,8193)) and all(np.isfinite(r['loss']) for r in logs)
    assert all(r['lr']==(1e-4 if r['update']<=4096 else 1e-5) for r in logs)
    assert training['actual_draws_per_window']==[512]*128
    assert not training['development_used_for_optimization'] and not training['test_used'] and not rollout['test_used']
    metrics,horizon,phases=audit_offline(plan,offline);table,physical=audit_rollouts(plan,rollout)
    assert not any((DATA/'episodes'/(r['episode']+'.npz')).exists() for r in read(DATA/'protocol.json')['episodes'] if r['split']=='test')
    OUT.mkdir(exist_ok=False)
    result=dict(verified=True,checkpoint_sha256=checkpoint,checkpoint_bytes=(RUN/'final.pt').stat().st_size,
        updates=8192,initialization=43,sampling_seeds=plan['sample_seeds'],offline=metrics,physical=physical,
        by_observation_start=phases,test_used=False,core_model_changed=False,peak_reserved_gib=training['peak_reserved_gib'],training_seconds=training['elapsed_seconds'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('rollouts.csv',table),('horizon.csv',horizon),('observation_start.csv',phases)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
