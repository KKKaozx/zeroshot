"""Prepare the one seed42 coverage replication; preserve every original artifact."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'cluster_packages/bridge_training_coverage_v1'
PACK=ROOT/'cluster_packages/bridge_coverage_replication_v1'
LOCAL=ROOT/'reports/bridge_coverage_replication_local_preflight_20261016'
CONTROL=ROOT/'reports/bridge-state-rep-fit-202463/real'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
def replace_once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

def reuse_control():
    old=read(ROOT/'reports/bridge-state-fit-202278/real/report.json')
    r=read(CONTROL/'report.json')
    audited=ROOT/'reports/bridge_state_rep_analysis_202463_20261014/analysis.json'
    a=read(audited);assert a['verified'] and a['updates_per_arm_verified']==8192
    for name,h in a['artifact_sha256'].items():
        if name.startswith('real/'): assert sha(CONTROL.parent/name)==h,name
    base=ROOT/'cluster_packages/bridge_state_conditioning_v1/fit_state_arm.py'
    replica=ROOT/'cluster_packages/bridge_state_replication_v1/fit_state_arm.py'
    s=base.read_text(encoding='utf-8');t=replica.read_text(encoding='utf-8')
    assert t==replace_once(s,'init = 43; seed(init)','init = 42; seed(init)')
    assert sha(CONTROL/'runner_snapshot.py')==sha(replica)
    for key in ['data_checks','state_checks','state_features_sha256','source_sha256','manifest_sha256','torch_version','sampling_steps','sampling_seeds','branch_update','final_update']:
        assert r[key]==old[key],key
    assert r['initialization_seed']==42 and r['state_arm']=='real'
    assert r['execution_checks_passed'] and r['clip_unchanged'] and r['other_frozen_unchanged']
    assert read(CONTROL/'plan.json')['manifest']==read(ROOT/'reports/bridge-state-fit-202278/real/plan.json')['manifest']
    for split in ['train','development']:
        with np.load(CONTROL/f'reduced_u8192_{split}.npz',allow_pickle=False) as x, np.load(ROOT/f'reports/bridge-state-fit-202278/real/reduced_u8192_{split}.npz',allow_pickle=False) as y:
            assert np.array_equal(x['targets'],y['targets']) and np.array_equal(x['row_indices'],y['row_indices'])
    result=dict(verified=True,reused_job='202463',state_arm='real',initialization_seed=42,
        control_report_sha256=sha(CONTROL/'report.json'),prior_independent_audit_sha256=sha(audited),
        all_real_artifact_hashes_match_prior_audit=True,only_initialization_differs_from_original_small_pool_source=True,
        data_state_sources_schedule_noise_dropout_lr_budget_match=True,
        development=a['development']['real'],new_neural_execution=False)
    return result

def prepare():
    assert not PACK.exists(), 'Preserve frozen preparation'
    reuse=reuse_control();first=read(ROOT/'reports/bridge_training_coverage_analysis_207348/analysis.json')
    assert first['verified'] and first['candidate_passed']
    for line in (ORIGINAL/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert sha(ORIGINAL/n)==h
    PACK.mkdir()
    for name in ['rows.json','episode_reference.json','schedule.npz']:
        shutil.copyfile(ORIGINAL/name,PACK/name)
    p=read(ORIGINAL/'protocol.json');p['recipe']['initialization_seed']=42
    p['control_report_sha256']=sha(CONTROL/'report.json')
    p['objective']='One independent shared initialization42 replication of fixed-budget training coverage207348.'
    p['screen']['position_and_rotation_below_reused_control']=True
    p['resource_gate']['extra_storage_plan_gib']=2
    p['cache_provenance_protocol_sha256']=sha(ORIGINAL/'protocol.json')
    p['replication']=dict(previous_initialization=43,current_initialization=42,
        prior_formal_report_sha256=sha(ROOT/'reports/bridge-coverage-fit-207348/report.json'),
        prior_preflight_report_sha256=sha(ROOT/'reports/bridge-coverage-preflight-207334/report.json'),
        previous_analysis_sha256=sha(ROOT/'reports/bridge_training_coverage_analysis_207348/analysis.json'),
        original_entry_sha256=sha(ORIGINAL/'run_training_coverage.py'),original_helper_sha256=sha(ORIGINAL/'coverage_inputs.py'),
        original_preflight_job='207334',original_fit_job='207348',reused_small_pool_job='202463',
        reuse_proof_sha256=None,allowed_additional_initializations=1,
        changed_variable='Shared model initialization43 to42 only; identical large pool, batch/noise/dropout streams,8192 updates and final assessment.',
        cache_policy='Require and reuse existing complete1876-row cache from207334; original protocol fingerprint is its provenance, not this replication recipe.',
        decision='Apply unchanged absolute screen plus matched seed42 small-pool comparison; report all tasks/episodes even if failed. No additional seed or budget search.')
    write(PACK/'reuse_control_proof.json',reuse);p['replication']['reuse_proof_sha256']=sha(PACK/'reuse_control_proof.json')
    write(PACK/'protocol.json',p)
    source=(ORIGINAL/'run_training_coverage.py').read_text(encoding='utf-8')
    source=replace_once(source,'seed(43);model=RobotAdapterModel(cfg).cuda().eval()',"seed(p['recipe']['initialization_seed']);model=RobotAdapterModel(cfg).cuda().eval()")
    source=replace_once(source,'r.update(initialization_seed=43,',"r.update(initialization_seed=p['recipe']['initialization_seed'],")
    ast.parse(source);(PACK/'run_training_coverage.py').write_text(source,encoding='utf-8',newline='\n')
    helper=(ORIGINAL/'coverage_inputs.py').read_text(encoding='utf-8')
    helper=replace_once(helper,'protocol_sha256=sha(code/\'protocol.json\'),',"protocol_sha256=protocol['cache_provenance_protocol_sha256'],")
    at='    start=time.monotonic();small=json.loads'
    helper=replace_once(helper,at,"    assert sha(code.parent/'bridge_training_coverage_v1/protocol.json')==protocol['cache_provenance_protocol_sha256']\n    assert args.local_bounded or args.cache.is_file(), 'Existing complete cache required; no re-encoding allowed'\n"+at)
    ast.parse(helper);(PACK/'coverage_inputs.py').write_text(helper,encoding='utf-8',newline='\n')
    validate=(ORIGINAL/'validate_preflight.py').read_text(encoding='utf-8')
    validate=validate.replace('/bridge-state-fit-202278/real/','/bridge-state-rep-fit-202463/real/')
    validate=replace_once(validate,"default=CODE/'full_tokens.pt'","default=CODE.parent/'bridge_training_coverage_v1/full_tokens.pt'")
    validate=replace_once(validate,"    assert r['branch_restore_passed']", "    assert r['initialization_seed']==p['recipe']['initialization_seed']==42\n    assert r['full_token_cache_reused']\n    assert r['branch_restore_passed']")
    validate=replace_once(validate,"    assert r['full_token_cache_reused']\n",'''    assert r['full_token_cache_reused']
    original=CODE.parent/'bridge_training_coverage_v1'
    assert sha(original/'protocol.json')==p['cache_provenance_protocol_sha256']
    assert sha(original/'run_training_coverage.py')==p['replication']['original_entry_sha256']
    assert sha(original/'coverage_inputs.py')==p['replication']['original_helper_sha256']
    previous=args.directory.parent/'bridge-coverage-preflight-207334/report.json'
    previous_fit=args.directory.parent/'bridge-coverage-fit-207348/report.json'
    assert sha(previous)==p['replication']['prior_preflight_report_sha256']
    assert sha(previous_fit)==p['replication']['prior_formal_report_sha256']
    prior=read(previous)
    for key in ['input_identity_sha256','token_cache_key_sha256','input_row_checks','episode_payload_checks']:
        assert r[key]==prior[key],key
''')
    ast.parse(validate);(PACK/'validate_preflight.py').write_text(validate,encoding='utf-8',newline='\n')
    for stage in ['preflight','fit']:
        script=(ORIGINAL/f'{stage}.sh').read_text()
        script=script.replace('bridge-coverage-','bridge-coverage-rep-').replace('COVERAGE_PREFLIGHT_JOB','COVERAGE_REP_PREFLIGHT_JOB')
        script=script.replace('cd /projects/Zeroshot/bridge_training_coverage_v1','cd /projects/Zeroshot/bridge_coverage_replication_v1')
        script=script.replace('/bridge-state-fit-202278/real/','/bridge-state-rep-fit-202463/real/')
        script=script.replace('PROJECT_USED_BYTES + 5*','PROJECT_USED_BYTES + 2*')
        assert script.startswith('#!/bin/bash\n') and '\n+' not in script
        (PACK/f'{stage}.sh').write_text(script,encoding='utf-8',newline='\n')
    (PACK/'README.md').write_text('One initialization42 coverage replication. Reuse old seed42 small-pool202463; do not rerun small control. '
        'Same1836train/40development rows, identical8192x8 batches, LR/noise/dropout and state stats. FrozenCLIP/gripper. '
        'Require existing1876-token cache from207334, no new encoding/download or environment. New helper explicitly uses original protocol SHA only as cache provenance. '
        'Local bounded preflight is not cluster authorization. Complete preflight first; formal once after validation; no extra seeds or extended fitting. '
        'Original40 development repeatedly inspected, not untouched final test. Offline pose only.\n',encoding='utf-8')
    write(ROOT/'reports/bridge_coverage_replication_preparation_20261016.json',dict(stage='prepared_pending_local_preflight',reuse=reuse,
        protocol_sha256=sha(PACK/'protocol.json'),initialization_seed=42,old_control_retrained=False,new_job_submitted=False,
        cache_reencoding_allowed=False,full_local_cache_allowed=False))
    print('Prepared one seed42 replication; control202463 verified and reused.')

def package():
    r=read(LOCAL/'report.json');a=read(LOCAL/'independent_audit.json')
    assert r['local_bounded'] and r['execution_checks_passed'] and r['branch_restore_passed'] and a['verified']
    assert r['initialization_seed']==42 and not r['development_scored'] and r['peak_reserved_gib']<7.5
    assert r['protocol_sha256']==sha(PACK/'protocol.json') and r['entry_sha256']==sha(PACK/'run_training_coverage.py')
    write(PACK/'local_preflight_summary.json',dict(report_sha256=sha(LOCAL/'report.json'),audit_sha256=sha(LOCAL/'independent_audit.json'),
        elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib'],local_bounded=True,development_scored=False))
    files=sorted(f for f in PACK.iterdir() if f.is_file() and f.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in files),encoding='ascii',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[PACK/'SHA256SUMS']:z.write(f,f.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and all(z.read(n)==(PACK.parent/n).read_bytes() for n in z.namelist())
    assert archive.stat().st_size<1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print('PACKAGE',archive.stat().st_size,sha(archive))

def audit():
    from analyze_bridge_growth64_200600 import summarize
    import subprocess
    assert not (LOCAL/'independent_audit.json').exists(), 'Do not repeat completed audit'
    r=read(LOCAL/'report.json');p=read(PACK/'protocol.json')
    oldlocal=read(ROOT/'reports/bridge_state_replication_local_preflight_20261013/real/report.json')
    previous=read(ROOT/'reports/bridge_training_coverage_local_preflight_v2_20261016/report.json')
    assert r['local_bounded'] and r['execution_checks_passed'] and r['branch_restore_passed']
    assert r['initialization_seed']==42 and r['optimization_operations']==16 and not r['development_scored']
    assert not r['checkpoints'] and r['peak_reserved_gib']<7.5
    assert r['entry_sha256']==sha(PACK/'run_training_coverage.py')==sha(LOCAL/'runner_snapshot.py')
    assert r['input_helper_sha256']==sha(PACK/'coverage_inputs.py') and r['protocol_sha256']==sha(PACK/'protocol.json')
    for key in ['initial_shared_sha256','augmented_initial_sha256','torch_version']:
        assert r[key]==oldlocal[key],key
    for key in ['input_identity_sha256','retained_data_checks','retained_state_checks','retained_state_features_sha256',
                'input_row_checks','episode_payload_checks','verified_row_indices']:
        assert r[key]==previous[key],key
    assert not r['full_input_contract_passed'] and r['active_token_rows']==224
    for key in ['clip_unchanged','other_frozen_unchanged','retained_numeric_contract_passed','retained_visual_tokens_exact','retained_language_tokens_exact']:
        assert r[key],key
    for name in ['rows.json','episode_reference.json','schedule.npz']:
        assert (PACK/name).read_bytes()==(ORIGINAL/name).read_bytes()
    source=(PACK/'run_training_coverage.py').read_text(encoding='utf-8')
    source=replace_once(source,"seed(p['recipe']['initialization_seed']);model=RobotAdapterModel(cfg).cuda().eval()",'seed(43);model=RobotAdapterModel(cfg).cuda().eval()')
    source=replace_once(source,"r.update(initialization_seed=p['recipe']['initialization_seed'],",'r.update(initialization_seed=43,')
    assert source==(ORIGINAL/'run_training_coverage.py').read_text(encoding='utf-8')
    helper=(PACK/'coverage_inputs.py').read_text(encoding='utf-8')
    helper=replace_once(helper,"protocol_sha256=protocol['cache_provenance_protocol_sha256'],","protocol_sha256=sha(code/'protocol.json'),")
    helper=replace_once(helper,"    assert sha(code.parent/'bridge_training_coverage_v1/protocol.json')==protocol['cache_provenance_protocol_sha256']\n    assert args.local_bounded or args.cache.is_file(), 'Existing complete cache required; no re-encoding allowed'\n",'')
    assert helper==(ORIGINAL/'coverage_inputs.py').read_text(encoding='utf-8')
    oldp=read(ORIGINAL/'protocol.json')
    for k in ['config','counts','retained128_indices','state_statistics','schedule','assessment','stopping','inherited_sha256']:
        assert p[k]==oldp[k],k
    assert {k:v for k,v in p['recipe'].items() if k!='initialization_seed'}=={k:v for k,v in oldp['recipe'].items() if k!='initialization_seed'}
    with np.load(PACK/'schedule.npz') as z: schedule=z['batches']
    assert set(schedule[:8].ravel())<=set(r['verified_row_indices'])
    logs={}
    for name,lo,hi,lr in [('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)]:
        logs[name]=values=[json.loads(v) for v in (LOCAL/(name+'_updates.jsonl')).read_text().splitlines()]
        assert [v['update'] for v in values]==list(range(lo,hi+1)) and all(v['learning_rate']==lr and np.isfinite(v['loss']) for v in values)
    assert logs['constant']==logs['restore_same_lr']
    cases=[('common',0,None),('common',4,r['common_train'])]+[(v['arm'],8,v['final_train']) for v in r['runs']]
    delta={'position_cm':0.,'rotation_deg':0.}
    for arm,u,recorded in cases:
        with np.load(LOCAL/f'{arm}_u{u:04d}_train.npz',allow_pickle=False) as z:
            pred,target,ix,ctx=[z[k] for k in ['predictions','targets','row_indices','contexts']]
        assert np.array_equal(ix,[0,32,64,96]) and pred.shape==(3,4,16,8) and ctx.shape==(4,1024)
        with np.load(LOCAL.parent/'bridge_training_coverage_local_preflight_v2_20261016/common_u0000_train.npz') as z:
            assert np.array_equal(target,z['targets'])
        assert np.isfinite(ctx).all() and np.linalg.norm(pred[...,3:7],axis=-1).min()>1e-8
        m,_,_,_,_=summarize(pred,target)
        if recorded:
            for k,v in m.items():
                kind='rotation_deg' if 'rotation' in k else 'position_cm';d=abs(v-recorded[k]);delta[kind]=max(delta[kind],d)
                assert d<(.02 if 'rotation' in k else 1e-4)
    with np.load(LOCAL/'constant_u0008_train.npz') as a,np.load(LOCAL/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    assert not (ROOT/'training_cache/coverage_full_tokens.pt').exists()
    rejected=subprocess.run([sys.executable,str(PACK/'validate_preflight.py'),str(LOCAL),'--control',str(CONTROL/'report.json')],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'AssertionError' in rejected.stderr
    result=dict(verified=True,local_bounded=True,initialization_seed=42,report_sha256=sha(LOCAL/'report.json'),
        original_seed42_local_parameter_initialization_exact=True,original_bounded224_inputs_exact=True,
        only_two_initialization_expressions_changed_in_runner=True,cache_provenance_only_helper_changes=True,
        schedule_rows_state_statistics_unchanged=True,preflight_replay_exact=True,optimization_operations=16,
        prediction_snapshots_recomputed=5,max_saved_metric_difference=delta,validator_rejects_local_preflight=True,
        development_scored=False,full_cluster_cache_not_locally_loaded=True,new_neural_execution_by_audit=False,
        elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib'])
    write(LOCAL/'independent_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','package','audit']
    globals()[sys.argv[1]]()
