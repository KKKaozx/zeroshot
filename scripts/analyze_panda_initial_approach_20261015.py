"""Independent saved-array audit; no model loading or new inference."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_bridge_end_to_end_20261015 import errors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/panda_initial_approach_probe_20261015'
RUN=ROOT/'reports/panda_spatial_fixed_trial_20261015'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(a,b):
    p,r,_=errors(a,b)
    return dict(position_cm=float(p.mean()),rotation_deg=float(r.mean()),endpoint_cm=float(p[...,-1].mean()))


def main():
    assert not (OUT/'independent_audit.json').exists()
    plan,report,trace_plan,trace=(read(OUT/n) for n in ['plan.json','report.json','trace_plan.json','trace_report.json'])
    assert sha(OUT/'outputs.npz')==report['output_sha256'] and sha(OUT/'native_trace.npz')==trace['output_sha256']
    assert sha(ROOT/'scripts/probe_panda_initial_approach_20261015.py')==plan['probe_sha256']
    assert sha(ROOT/'scripts/trace_panda_initial_approach_20261015.py')==trace_plan['trace_sha256']
    assert all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
    assert sha(RUN/'final.pt')==plan['checkpoint_sha256']==report['checkpoint_sha256']==trace_plan['checkpoint_sha256']==trace['checkpoint_sha256']
    assert all(r['training_updates']==0 and not r['development_used'] and not r['test_used'] for r in [plan,report,trace_plan,trace])
    assert report['parameters_unchanged'] and trace['parameters_unchanged']
    with np.load(OUT/'outputs.npz') as z:data={k:z[k] for k in z.files}
    assert data['targets'].shape==(32,16,7) and all(np.isfinite(v).all() for v in data.values())
    for row in report['per_seed']:
        sl=slice(0,16) if row['observation_start']==0 else slice(16,32)
        a,b=(data[f"{row['seed']}_{row['level']}_{name}"][sl] for name in ['correct','image_swap'])
        for name,val in [('correct',measure(a,data['targets'][sl])),('image_swap',measure(b,data['targets'][sl])),('paired_output_change',measure(b,a))]:
            assert all(abs(row[name][k]-v)<1e-7 for k,v in val.items())
    with np.load(OUT/'native_trace.npz') as z:arr={k:z[k] for k in z.files}
    with np.load(RUN/'offline_train.npz') as z:truth=z['targets'];old=z['predictions']
    initial=np.array([r['start']==0 for r in trace_plan['selection']]);growth=[]
    for row in trace['per_seed']:
        select=initial if row['observation_start']==0 else ~initial
        value=measure(arr[f"{row['seed']}_{row['stage']}"][select],arr['targets'][select])
        assert all(abs(row[k]-v)<1e-7 for k,v in value.items())
    for si,seed in enumerate(trace_plan['sampling_seeds']):
        assert np.array_equal(arr[f'{seed}_native'],old[si,arr['indices']])
        p0,_,_=errors(arr[f'{seed}_t99'][initial],arr['targets'][initial])
        pf,_,_=errors(arr[f'{seed}_native'][initial],arr['targets'][initial])
        growth.append(dict(seed=seed,initial_windows_worsening=int((pf.mean(1)>p0.mean(1)).sum()),windows=16,
            mean_position_growth_cm=float((pf-p0).mean())))
    constants=[]
    for j,start in enumerate([0,12,24,36,44,52,68,86]):
        targets=truth[j::8,:,:7].astype(float)
        loo=(targets.sum(0)[None]-targets)/15
        constants.append(dict(observation_start=start,**measure(loo,targets)))
    with (OUT/'phase_constant_diagnostic.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(constants[0]));writer.writeheader();writer.writerows(constants)
    result=dict(verified=True,zero_new_training=True,train_only=True,same_native_predictions_exact=True,
        trace_growth=growth,phase_specific_other15_mean_diagnostic=constants,
        constant_baseline_limit='Uses expert phase and labels of other training demonstrations; diagnostic target variability, not a deployable policy or held-out performance',
        checked_outputs_sha256=report['output_sha256'],checked_trace_sha256=trace['output_sha256'])
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,start in zip(axes,[0,12]):
        for seed in trace_plan['sampling_seeds']:
            group=[r for r in trace['per_seed'] if r['seed']==seed and r['observation_start']==start]
            ax.plot(range(4),[r['position_cm'] for r in group],marker='o',label=str(seed))
        ax.set_xticks(range(4),['First x0 (t99)','Middle x0 (t49)','Last x0 (t0)','Final action'],rotation=20)
        ax.set(title=f'Training observation start {start}',ylabel='Path position error (cm)');ax.grid(alpha=.2);ax.legend(title='Sampling seed')
    fig.savefig(OUT/'matched_native_chain.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


def audit_initial_fit():
    folder=ROOT/'reports/panda_initial_only_fit_20261015'
    assert not (folder/'independent_audit.json').exists()
    plan,report=read(folder/'plan.json'),read(folder/'report.json')
    assert report['complete'] and report['updates']==plan['updates']==1024
    assert sha(folder/'entry_snapshot.py')==plan['entry_sha256']==sha(ROOT/'scripts/fit_panda_initial_only_20261015.py')
    assert all(sha(ROOT/n)==h for n,h in plan['source_sha256'].items())
    assert plan['config']==read(RUN/'plan.json')['config']
    assert report['initial_group_sha256']==read(RUN/'training_report.json')['initial_group_sha256']
    assert sha(folder/'final.pt')==report['checkpoint_sha256'] and (folder/'final.pt').stat().st_size==report['checkpoint_bytes']
    assert report['draws_per_window']==[512]*16 and report['clip_unchanged']
    assert report['initial_group_sha256']['clip']==report['final_group_sha256']['clip']
    assert not report['development_used'] and not report['test_used'] and not report['physical_rollout']
    log=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in log]==list(range(1,1025))
    assert all(np.isfinite(r['loss']) and r['lr']==(1e-4 if r['update']<=512 else 1e-5) for r in log)
    with np.load(folder/'final_train_predictions.npz') as z:pred,target=z['predictions'],z['targets']
    with np.load(RUN/'offline_train.npz') as z:
        assert np.array_equal(target,z['targets'][::8]);old=z['predictions'][:,::8]
    assert pred.shape==(3,16,16,8) and np.isfinite(pred).all() and np.all(target[...,7]==1)
    p,r,_=errors(pred,target)
    good=((p.mean(-1)<=.5)&(r.mean(-1)<=5)).all(0)
    gate=bool(good.mean()>=.9 and (p.mean((1,2))<=.5).all() and (r.mean((1,2))<=5).all()
        and (p[...,-1].mean(1)<=1).all() and (r[...,-1].mean(1)<=10).all())
    assert gate==report['metrics']['pose_fit_passed'] and int(good.sum())==report['metrics']['windows_all_three_seeds_pass']
    m=measure(pred,target)
    for k,key in [('position_cm','position_cm'),('rotation_deg','rotation_deg'),('endpoint_cm','endpoint_position_cm')]:
        assert abs(m[k]-report['metrics'][key])<(.02 if 'rotation' in k else 1e-4)
    result=dict(verified=True,updates=1024,training_windows=16,source_unchanged=True,clip_unchanged=True,
        original_initialization_exact=True,initial_only= m,original_mixed_initial=measure(old,target),pose_fit_passed=gate,
        passing_windows=int(good.sum()),development_used=False,test_used=False,physical_rollout=False,
        gripper_limit='All targets open; no evidence of closing, releasing or full grasping',checkpoint_sha256=report['checkpoint_sha256'])
    (folder/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.bar(['Mixed-stage model: initial windows','Initial-only capacity check'],[result['original_mixed_initial']['position_cm'],m['position_cm']])
    ax.set(ylabel='Train initial-approach path position error (cm)',title='Same16 initial observations;512 draws each');ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'initial_fit_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--initial-fit',action='store_true')
    if parser.parse_args().initial_fit:audit_initial_fit()
    else:main()
