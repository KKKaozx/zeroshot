"""CPU check of seed42 short preflight, including real launcher arguments."""
import ast
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import sys
import numpy as np
from analyze_bridge_training_trace_200039 import errors, FIELDS

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_growth64_replication_v1'
BASE = ROOT/'cluster_packages/bridge_growth64_v1'
DATA = ROOT/'reports/bridge_growth64_replication_local_preflight_20261013'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    global PACK, BASE, DATA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package',type=Path,default=PACK)
    ap.add_argument('--base',type=Path,default=BASE)
    ap.add_argument('--data',type=Path,default=DATA)
    ap.add_argument('--entry',default='fit_bridge_growth64.py')
    ap.add_argument('--initialization-seed',type=int,default=42)
    ap.add_argument('--train-count',type=int,choices=(64,128),default=64)
    ap.add_argument('--window-control',action='store_true',help='Same-size39-window substitution; base is frozen128 package')
    args=ap.parse_args();PACK,BASE,DATA=args.package,args.base,args.data
    init,n=args.initialization_seed,args.train_count
    window_control=args.window_control
    entry=args.entry
    warm,end=n//8,n//4
    report = json.loads((DATA/'report.json').read_text())
    assert all(report[k] for k in ('execution_checks_passed','branch_restore_passed','input_contract_passed','clip_unchanged','other_frozen_unchanged'))
    assert report['stage']=='preflight' and report['initialization_seed']==init and not report['checkpoints']
    assert report['manifest_sha256']==sha(PACK/'manifest.json')
    manifest=json.loads((PACK/'manifest.json').read_text()); old=json.loads((BASE/'manifest.json').read_text())
    assert manifest['initialization_seeds']==[init] and manifest['late_lr']['initialization_seed']==init
    preserved=('train','development','config','schedule_sha256','batch_size','updates_per_arm','draw_slots_per_window','sampling_seeds','learning_rate_schedule','growth') if n==64 else ('development','config','batch_size','draw_slots_per_window','sampling_seeds','learning_rate_schedule')
    for key in preserved:
        assert manifest[key]==old[key]
    if window_control:
        assert n==128 and init==43 and manifest['updates_per_arm']==old['updates_per_arm']==8192
        for key in ('growth','schedule_sha256','development','config','trace','learning_rate_schedule'):
            assert manifest[key]==old[key],key
        changed=manifest['window_coverage']['changed_row_indices']
        assert len(changed)==len(set(changed))==39
        assert changed==[i for i,(a,b) in enumerate(zip(manifest['train'],old['train'])) if a!=b]
        for a,b in zip(manifest['train'],old['train']):
            assert (a['shard'],a['record_index'],a['task'])==(b['shard'],b['record_index'],b['task'])
        assert manifest['window_coverage']['base_files_sha256']['manifest.json']==sha(BASE/'manifest.json')
        assert report['unchanged_input_raw_identity_passed']
        control=manifest['window_coverage']
        assert all(sha(BASE/name)==h for name,h in control['base_files_sha256'].items())
        assert sha(PACK/'selection_plan.json')==control['selection_plan_sha256']
        plan=json.loads((PACK/'selection_plan.json').read_text())
        assert plan['train']==manifest['train'] and plan['development']==manifest['development']
        pool_path=ROOT/'reports/bridge-xyz-eval-197349.json'
        assert sha(pool_path)==plan['selection_pool_sha256']
        pool=json.loads(pool_path.read_text())['window_selection'];by_ep=defaultdict(list)
        for row in pool:by_ep[(row['shard'],row['record_index'])].append(row['start_index'])
        phase_counts=Counter()
        for i,row in enumerate(manifest['train']):
            assert row in pool
            starts=sorted(set(by_ep[(row['shard'],row['record_index'])]))
            phase_counts['first' if row['start_index']==starts[0] else ('last' if row['start_index']==starts[-1] else 'middle')]+=1
            if i in changed:
                middle=[v for v in starts if starts[0]<v<starts[-1]]
                assert row['start_index']==min(middle,key=lambda v:(abs(v-(starts[0]+starts[-1])/2),v))
                assert old['train'][i]['start_index'] in (starts[0],starts[-1])
        assert dict(phase_counts)==control['indexed_phase_counts']=={'first':38,'middle':51,'last':39}
        # Independent reference audit in the inherited audit order, not model row order.
        references=[ROOT/'reports/bridge_batch_input_reference_20261012',BASE.parent/'bridge_growth64_v1/added_reference',BASE/'new_reference']
        expected=[]
        for directory in references:
            metadata=json.loads((directory/'metadata.json').read_text())
            expected.extend(metadata['raw_inputs'])
        directory=PACK/'changed_reference';metadata=json.loads((directory/'metadata.json').read_text())
        assert all(sha(directory/name)==h for name,h in control['new_reference_sha256'].items())
        assert metadata['rows']==[manifest['train'][i] for i in changed]
        def audit_index(i):return i if i<32 else (i+40 if i<128 else i-96)
        with np.load(directory/'arrays.npz',allow_pickle=False) as z:
            for j,i in enumerate(changed):
                expected[audit_index(i)]=metadata['raw_inputs'][j]
                for key,field in [('images','image_sha256'),('targets','target_sha256'),('currents','current_sha256')]:
                    h=hashlib.sha256(z[key][j].tobytes()).hexdigest()
                    assert np.isfinite(z[key][j]).all() and h==metadata['processed_sha256'][key][j]
                    assert report['data_checks'][audit_index(i)][field]==h
        assert len(expected)==168 and expected==[r['raw_identity'] for r in report['data_checks']]
    elif n==128:
        assert manifest['train'][:64]==old['train'] and manifest['updates_per_arm']==8192
        assert manifest['late_lr']['branch_update']==4096 and len(manifest['train'])==128
        assert manifest['growth']['retained64_manifest_sha256']==sha(BASE/'manifest.json')
        assert manifest['growth']['retained64_contract_sha256']==sha(BASE/'growth_input_contract.py')
    assert len(report['data_checks'])==len(report['input_numeric_comparisons'])==n+40
    assert all(r['raw_identity_exact'] for r in report['input_numeric_comparisons'])
    assert all(d['max_abs']<=d['atol'] for r in report['input_numeric_comparisons'] for d in r['differences'].values())
    assert report['source_sha256']==manifest['source_sha256']
    assert all(sha(PACK/n)==h for n,h in report['source_sha256'].items())
    assert sha(DATA/'runner_snapshot.py')==report['source_sha256'][args.entry]
    code=(PACK/args.entry).read_text()
    if n==64:
        assert code.replace('init = 42; seed(init)','init = 43; seed(init)')==(BASE/'fit_bridge_growth64.py').read_text()
    else:
        original=ast.parse((BASE/('fit_bridge_growth128.py' if window_control else 'fit_bridge_growth64.py')).read_text());changed=ast.parse(code)
        for name in ('context','advance','cpu_copy','state_sha','save_checkpoint'):
            a=next(node for node in ast.walk(original) if isinstance(node,ast.FunctionDef) and node.name==name)
            b=next(node for node in ast.walk(changed) if isinstance(node,ast.FunctionDef) and node.name==name)
            assert ast.dump(a)==ast.dump(b),name
    excluded=(entry,'window_coverage_input_contract.py' if window_control else 'growth128_input_contract.py')
    assert all(sha(PACK/name)==old['source_sha256'][name] for name in manifest['source_sha256'] if name not in excluded)
    tree=ast.parse(code); main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    setup=main_node.body[:5]; assert setup[-1].value.func.attr=='parse_args'
    parser=ast.fix_missing_locations(ast.Module(body=[ast.Import(names=[ast.alias(name='argparse')]),
        ast.ImportFrom(module='pathlib',names=[ast.alias(name='Path')],level=0)]+setup,type_ignores=[]))
    launcher_args={}
    for stage in ('preflight','fit'):
        shell=(PACK/f'{stage}.sh').read_text()
        tokens=shlex.split(shell.split('/projects/Zeroshot/envs/multitask-preflight-v1/bin/python -u ',1)[1].replace('\\\n',' '))
        assert '+' not in tokens
        saved=sys.argv;sys.argv=tokens
        try:
            namespace={'__doc__':'launcher verification'};exec(compile(parser,'launcher','exec'),namespace)
        finally:sys.argv=saved
        args=namespace['args'];assert args.stage==stage
        if stage=='fit':assert ('bridge-window-preflight-' if window_control else ('bridge-growth64-rep-preflight-' if n==64 else 'bridge-growth128-preflight-')) in str(args.preflight_reference)
        if window_control:
            assert args.control_report.as_posix()=='/projects/Zeroshot/runs/bridge-growth128-fit-201663/report.json'
        launcher_args[stage]=tokens
    arms={r['arm']:r for r in report['runs']};assert set(arms)=={'constant','restore_same_lr','reduced'}
    logs={name:[json.loads(s) for s in (DATA/f'{name}_updates.jsonl').read_text().splitlines()] for name in ('common',*arms)}
    for name,rows in logs.items():
        assert [r['update'] for r in rows]==list(range(1,warm+1) if name=='common' else range(warm+1,end+1))
        assert np.isfinite([r['loss'] for r in rows]).all()
        assert all(np.isfinite(v) and v>0 for r in rows for v in r['unclipped_module_gradient_norms'].values())
    assert logs['constant']==logs['restore_same_lr']
    assert len({arms[a]['loss_first'] for a in arms})==1
    for run in arms.values():
        assert run['common_weights_sha256']==report['common_state_sha256']
        assert run['common_adam_sha256']==report['common_adam_sha256']
        assert run['actual_draw_slots_per_window']==[1]*n
    cases=[('common',h['update'],h['train']) for h in report['common_history']]
    cases += [(a,h['update'],h['train']) for a,run in arms.items() for h in run['history']]
    max_delta=dict.fromkeys(FIELDS,0.)
    for arm,step,recorded in cases:
        with np.load(DATA/f'{arm}_u{step:04d}_train.npz') as z:
            pred,target,indices,ctx=(z[k] for k in ('predictions','targets','row_indices','contexts'))
        assert pred.shape==(3,32,16,8) and ctx.shape==(32,1024) and np.isfinite(ctx).all()
        assert np.array_equal(indices,np.r_[np.arange(16),np.arange(n//2,n//2+16)])
        for name,sl in [('overall',slice(None)),('retained_train',slice(0,16)),('added_train',slice(16,32))]:
            rec=recorded if name=='overall' else recorded[name];p,r=errors(pred[:,sl],target[sl])
            vals=dict(zip(FIELDS,map(float,(p.mean(),r.mean(),p[...,-1].mean(),r[...,-1].mean()))))
            for k,v in vals.items():
                delta=abs(v-rec[k]);max_delta[k]=max(max_delta[k],delta)
                assert delta<(.02 if 'rotation' in k else 1e-4)
            good=((p.mean(-1)<=.5)&(r.mean(-1)<=5)).all(0)
            gate=bool(good.mean()>=.9 and (p.mean((1,2))<=.5).all() and (r.mean((1,2))<=5).all()
                      and (p[...,-1].mean(1)<=1).all() and (r[...,-1].mean(1)<=10).all())
            assert int(good.sum())==rec['fit_gate']['windows_all_three_seeds_pass']
            assert gate==rec['fit_gate']['pose_fit_passed']
    with np.load(DATA/f'constant_u{end:04d}_train.npz') as a,np.load(DATA/f'restore_same_lr_u{end:04d}_train.npz') as b:
        assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
    assert not list(DATA.glob('*.pt'))
    result=dict(passed=True,report_sha256=sha(DATA/'report.json'),only_training_change=('39 window identities within same64 episodes/128 windows' if window_control else ('initialization43 ->42' if n==64 else '64 ->128 windows with matched per-window exposure')),
        input_checks=n+40,snapshots=len(cases),logged_updates=sum(map(len,logs.values())),
        restored_losses_and_saved_arrays_exact=True,max_float64_metric_difference=max_delta,
        launcher_arguments=launcher_args,full_fit_executed=False)
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='launcher_arguments'},indent=2))

if __name__=='__main__':main()
