"""Independent CPU review of state data, frozen recipe, launchers and GPU snapshots."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys
import numpy as np
from analyze_bridge_training_trace_200039 import errors

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_state_conditioning_v1'
BASE=ROOT/'cluster_packages/bridge_growth128_v1'
DATA=ROOT/'reports/bridge_state_local_preflight_v2_20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def function(text,name):
    found=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(found)==1
    return ast.dump(found[0],include_attributes=False)

def rotation_from_quaternion(rpy):
    r,p,y=np.asarray(rpy,dtype=float)/2
    a=np.array([np.sin(r)*np.cos(p)*np.cos(y)-np.cos(r)*np.sin(p)*np.sin(y),
        np.cos(r)*np.sin(p)*np.cos(y)+np.sin(r)*np.cos(p)*np.sin(y),
        np.cos(r)*np.cos(p)*np.sin(y)-np.sin(r)*np.sin(p)*np.cos(y),
        np.cos(r)*np.cos(p)*np.cos(y)+np.sin(r)*np.sin(p)*np.sin(y)])
    x,y,z,w=a/np.linalg.norm(a)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def audit_inputs():
    p=read(PACK/'protocol.json');m=read(BASE/'manifest.json')
    if 'replication' in p:
        prior=ROOT/'cluster_packages/bridge_state_conditioning_v1'
        assert p['initialization_seed']==42 and p['replication']['original_initialization']==43
        assert p['replication']['additional_initializations_allowed']==1
        assert p['replication']['original_protocol_sha256']==sha(prior/'protocol.json')
        previous=read(prior/'protocol.json')
        assert {k:v for k,v in p.items() if k not in ('initialization_seed','replication')}=={k:v for k,v in previous.items() if k!='initialization_seed'}
        for name in ('baseline_observed.py','fit_state_arm.py'):
            current=(PACK/name).read_text(encoding='utf-8')
            assert current.count('init = 42; seed(init)')==1
            assert current.replace('init = 42; seed(init)','init = 43; seed(init)')==(prior/name).read_text(encoding='utf-8')
        assert sha(PACK/'current_pose.npz')==sha(prior/'current_pose.npz')
        assert sha(PACK/'state_input_contract.py')==sha(prior/'state_input_contract.py')
        added="\n            assert report['initialization_seed']==protocol['initialization_seed']"
        runner=(PACK/'run_state_pair.py').read_text(encoding='utf-8')
        assert runner.count(added)==1
        assert runner.replace(added,'')==(prior/'run_state_pair.py').read_text(encoding='utf-8')
        for stage in ('preflight','fit'):
            shell=(PACK/f'{stage}.sh').read_text(encoding='utf-8')
            restored=shell.replace('bridge_state_replication_v1','bridge_state_conditioning_v1').replace('bridge-state-rep-','bridge-state-').replace('STATE_REP_PREFLIGHT_JOB','STATE_PREFLIGHT_JOB')
            assert restored==(prior/f'{stage}.sh').read_text(encoding='utf-8')
    assert p['rows']==m['train']+m['development']
    assert p['base_manifest_sha256']==sha(BASE/'manifest.json')
    assert all(sha(BASE/n)==h for n,h in p['base_files_sha256'].items())
    assert p['current_pose_sha256']==sha(PACK/'current_pose.npz')
    with np.load(PACK/'current_pose.npz',allow_pickle=False) as z:
        assert z.files==['current_pose'];v=z['current_pose'].copy()
    with np.load(ROOT/'reports/bridge_condition_input_audit_20261013/observed_states.npz',allow_pickle=False) as z:
        assert np.array_equal(v,z['current_state'][:,:6].astype('<f4'))
    assert v.shape==(168,6) and v.dtype==np.float32 and np.isfinite(v).all()
    stats=p['state_features'];train=v[:128,:3].astype(float)
    assert np.array_equal(train.mean(0),stats['xyz_mean']) and np.array_equal(train.std(0),stats['xyz_std'])
    assert stats['fit_rows']==list(range(128)) and not stats['clipping'] and not stats['future_state_used']
    # Execute only the pure NumPy feature function, with no Torch/model imports.
    tree=ast.parse((PACK/'state_input_contract.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='state_features')
    scope={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'features','exec'),scope)
    features=scope['state_features'](v,stats)
    independent=np.array([np.concatenate([(row[:3]-train.mean(0))/train.std(0),rotation_from_quaternion(row[3:])[:,:2].T.flatten()]) for row in v]).astype('<f4')
    assert np.max(np.abs(features-independent))<=1e-7
    synthetic=np.array([[0,0,0,0,0,0],[0,0,0,0,0,np.pi/2]],dtype=float)
    result=scope['state_features'](synthetic,dict(xyz_mean=[0]*3,xyz_std=[1]*3))
    assert np.allclose(result[0,3:],[1,0,0,0,1,0],atol=1e-7)
    assert np.allclose(result[1,3:],[0,1,0,-1,0,0],atol=1e-7)
    altered=v.copy();altered[128:,:3]+=1e4
    assert np.array_equal(altered[:128,:3].astype(float).mean(0),stats['xyz_mean'])
    assert np.array_equal(scope['state_features'](altered,stats)[:128],features[:128])
    # The ingress AST must select states[t,:6], and must never select a future slice.
    ingress=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_current_features')
    text=ast.unparse(ingress)
    assert 'states[t, :6]' in text and 'states[t +' not in text and "steps/action" not in text
    assert 'training = observed[:128, :3].astype(float)' in text
    old=(BASE/'fit_bridge_growth128.py').read_text(encoding='utf-8')
    zero=(PACK/'fit_state_arm.py').read_text(encoding='utf-8')
    legacy=(PACK/'baseline_observed.py').read_text(encoding='utf-8')
    for name in ('check','state_sha','cpu_copy'):
        assert function(old,name)==function(legacy,name)==function(zero,name),name
    assert function(old,'advance')==function(legacy,'advance')
    restored=zero.replace('clipping_parameters(model, trainables, args.state_arm)','trainables')
    assert function(old,'advance')==function(restored,'advance')
    assert function(old,'context')==function(legacy,'context')
    ctx=next(n for n in ast.walk(ast.parse(zero)) if isinstance(n,ast.FunctionDef) and n.name=='context')
    assert ast.dump(ctx.body[0])==ast.dump(next(n for n in ast.walk(ast.parse(old)) if isinstance(n,ast.FunctionDef) and n.name=='context').body[0])
    assert ast.unparse(ctx.body[-1])=='return now + model.history_projection(repeated - now) + model.state_projection(state_features[indices])'
    return dict(passed=True,windows=168,current_state_only=True,training_statistics_only=True,
        initialization_seed=p['initialization_seed'],replication_initialization_only='replication' in p,
        rotation_independently_verified=True,unchanged_evaluation_function=True,
        update_function_identical_except_verified_zero_gradient_clipping=True,
        max_feature_difference=float(np.max(np.abs(features-independent))),features_sha256=hashlib.sha256(features.tobytes()).hexdigest())

def launcher_check():
    code=ast.parse((PACK/'run_state_pair.py').read_text(encoding='utf-8'))
    main=next(n for n in code.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    statements=[]
    for node in main.body:
        statements.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
    outputs={}
    for stage in ('preflight','fit'):
        shell=(PACK/f'{stage}.sh').read_text(encoding='utf-8')
        assert shell.startswith('#!/bin/bash\n') and '\n+' not in shell
        start=shell.index('/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u')
        command=shlex.split(shell[start:].replace('\\\n',' '));argv=command[3:]
        argv=[x.replace('${SLURM_JOB_ID}','12345').replace('${STATE_PREFLIGHT_JOB}','12344').replace('${STATE_REP_PREFLIGHT_JOB}','12344') for x in argv]
        before=sys.argv;sys.argv=['run_state_pair.py']+argv
        scope={'argparse':argparse,'Path':Path,'__doc__':'CPU parser check'}
        try:exec(compile(ast.Module(body=statements,type_ignores=[]),'real_pair_parser','exec'),scope)
        finally:sys.argv=before
        parsed=scope['args'];assert parsed.stage==stage and str(parsed.output).endswith('-12345')
        assert parsed.control_report.as_posix()=='/projects/Zeroshot/runs/bridge-growth128-fit-201663/report.json'
        if stage=='fit':assert str(parsed.preflight_reference).endswith('-12344') and ('STATE_PREFLIGHT_JOB' in shell or 'STATE_REP_PREFLIGHT_JOB' in shell)
        outputs[stage]=dict(passed=True,argv=argv)
    return outputs

def main():
    global DATA, PACK
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--inputs-only',action='store_true')
    ap.add_argument('--data',type=Path,default=DATA)
    ap.add_argument('--pack',type=Path,default=PACK)
    args=ap.parse_args();DATA=args.data;PACK=args.pack
    result=dict(passed=False,inputs=audit_inputs(),launchers=launcher_check())
    if args.inputs_only:print(json.dumps(result['inputs']));return
    pair=read(DATA/'report.json');assert pair['passed'] and pair['zero_replay']['passed'] and pair['state_interface']['passed']
    assert pair['protocol_sha256']==sha(PACK/'protocol.json')
    assert all(sha(PACK/n)==h for n,h in pair['source_sha256'].items())
    snapshot_count=0;max_metric_delta=0.
    for arm in ('legacy','zero','real'):
        report=read(DATA/arm/'report.json')
        assert report['initialization_seed']==read(PACK/'protocol.json')['initialization_seed']
        assert pair['runs'][arm]['report_sha256']==sha(DATA/arm/'report.json')
        assert report['execution_checks_passed'] and report['branch_restore_passed'] and report['clip_unchanged'] and not report['checkpoints']
        assert report['branch_update']==16 and report['final_update']==32 and report['sampling_steps']==100
        records=[('common',0,report['common_history'][0]['train']),('common',16,report['common_history'][1]['train'])]
        records += [(r['arm'],32,r['final_train']) for r in report['runs']]
        for name,u,metric in records:
            file=DATA/arm/f'{name}_u{u:04d}_train.npz'
            with np.load(file,allow_pickle=False) as z:
                assert z['row_indices'].tolist()==list(range(16))+list(range(64,80))
                pred,target=z['predictions'],z['targets']
                assert pred.shape==(3,32,16,8) and np.isfinite(pred).all()
                position,rotation=errors(pred,target)
                recomputed=dict(path_position_cm=float(position.mean()),path_rotation_deg=float(rotation.mean()),
                    endpoint_position_cm=float(position[...,-1].mean()),endpoint_rotation_deg=float(rotation[...,-1].mean()))
                for k,v in recomputed.items():
                    delta=abs(v-metric[k]);max_metric_delta=max(max_metric_delta,delta)
                    # Match the existing float64 audit: acos of float32 unit quaternions
                    # has amplified rounding near zero. This is not a replay tolerance.
                    assert delta<(.02 if 'rotation' in k else 1e-4),(arm,file.name,k,delta)
            snapshot_count+=1
        for name in ('common','constant','restore_same_lr','reduced'):
            logs=[json.loads(x) for x in (DATA/arm/f'{name}_updates.jsonl').read_text().splitlines()]
            assert len(logs)==16
            assert all(np.isfinite(x['loss']) for x in logs)
        if arm!='legacy':
            assert report['state_checks']['features_sha256']==result['inputs']['features_sha256']
    assert snapshot_count==15 and not list(DATA.rglob('*.pt'))
    result.update(passed=True,report_sha256=sha(DATA/'report.json'),snapshots=15,max_recomputed_metric_difference=max_metric_delta,
        zero_replay=pair['zero_replay'],state_interface=pair['state_interface'],full_fit_or_generalization_executed=False)
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,snapshots=15,max_recomputed_metric_difference=max_metric_delta)))

if __name__=='__main__':main()
