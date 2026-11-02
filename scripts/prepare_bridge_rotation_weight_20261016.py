"""One training-only scale-derived rotation-weight control on the existing large pool."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'cluster_packages/bridge_coverage_replication_v1'
PACK=ROOT/'cluster_packages/bridge_rotation_weight_v1'
LOCAL=ROOT/'reports/bridge_rotation_weight_local_preflight_v2_20261016'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')
def once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b)

COMPACT_LOOKUP="""        token_ids=torch.arange(1876,device='cuda')
        if args.local_bounded:
            token_ids=torch.full((1876,),-1,device='cuda',dtype=torch.long)
            token_ids[r['verified_row_indices']]=torch.arange(len(r['verified_row_indices']),device='cuda')
"""

def compact_helper(source):
    source=once(source,'        visual,language=visual_cpu.cuda(),language_cpu.cuda();reused_large=False',
        """        if args.local_bounded:
            active=torch.tensor(report['verified_row_indices'],dtype=torch.long)
            visual,language=visual_cpu[active].cuda(),language_cpu[active].cuda()
        else:visual,language=visual_cpu.cuda(),language_cpu.cuda()
        reused_large=False""")
    return once(source,"    return visual,language,texts['attention_mask'].cuda()", """    report['local_compact_token_storage']=bool(args.local_bounded)
    report['resident_token_rows']=len(report['verified_row_indices']) if args.local_bounded else 1876
    attention_mask=texts['attention_mask']
    if args.local_bounded:attention_mask=attention_mask[report['verified_row_indices']]
    return visual,language,attention_mask.cuda()""")

def compact_runner(source):
    source=once(source,'        def context(indices):',COMPACT_LOOKUP+'        def context(indices):')
    assert source.count('visual[indices],language[indices],mask[indices]')==2
    return source.replace('visual[indices],language[indices],mask[indices]',
        'visual[token_ids[indices]],language[token_ids[indices]],mask[token_ids[indices]]')

LOSS_SOURCE='''"""Change quaternion MSE weight only; diffusion coordinates and sampling stay fixed."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

def training_statistics(targets):
    raw=np.asarray(targets)
    assert raw.shape==(1836,16,8) and np.isfinite(raw).all()
    t=raw.astype(np.float64);q=t[...,3:7];xyz=t[...,:3]
    assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-5
    assert (q[...,3]>=0).all(), 'Original canonical quaternion labels required'
    identity=np.array([0.,0.,0.,1.])
    pe=float(np.square(xyz).sum(-1).mean())
    qe=float(np.square(q-identity).sum(-1).mean())
    assert pe>0 and qe>0
    return dict(train_windows=1836,train_targets=29376,
        training_targets_sha256=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
        xyz_static_squared_sum_mean=pe,quaternion_static_squared_sum_mean=qe,
        weight=pe/qe,quaternion_negative_w_count=int((q[...,3]<0).sum()),
        xyz_std=xyz.reshape(-1,3).std(0).tolist(),quaternion_std=q.reshape(-1,4).std(0).tolist())

def rotation_pose_loss(pred,target,weight):
    assert pred.shape==target.shape and pred.shape[-1]==7
    assert np.isfinite(weight) and weight>0
    base=F.mse_loss(pred,target)
    xyz=F.mse_loss(pred[...,:3],target[...,:3])
    quat=F.mse_loss(pred[...,3:7],target[...,3:7])
    # Preserve the original xyz coefficient1/7. w=1 is the exact old expression.
    loss=base if weight==1.0 else base+((float(weight)-1.)*4./7.)*quat
    return loss,dict(unweighted_pose_mse=float(base.detach()),xyz_component_mse=float(xyz.detach()),
        quaternion_component_mse=float(quat.detach()),weighted_rotation_contribution=float((float(weight)*4./7.*quat).detach()))

def verify_loss_math(weight):
    generator=torch.Generator().manual_seed(810031)
    pred=torch.randn(2,16,7,generator=generator,dtype=torch.float64,requires_grad=True)
    target=torch.randn(2,16,7,generator=generator,dtype=torch.float64)
    baseline=F.mse_loss(pred,target)
    identity,_=rotation_pose_loss(pred,target,1.)
    assert torch.equal(baseline,identity)
    g0=torch.autograd.grad(baseline,pred,retain_graph=True)[0]
    g1=torch.autograd.grad(identity,pred,retain_graph=True)[0]
    assert torch.equal(g0,g1)
    weighted,_=rotation_pose_loss(pred,target,weight)
    g=torch.autograd.grad(weighted,pred)[0]
    assert torch.allclose(g[...,:3],g0[...,:3],atol=1e-14,rtol=1e-12)
    assert torch.allclose(g[...,3:7],weight*g0[...,3:7],atol=1e-14,rtol=1e-12)
    assert torch.isfinite(g).all()
    return dict(weight1_forward_and_gradient_exact=True,xyz_loss_gradient_coefficient_preserved=True,
        quaternion_gradient_multiplied_by_fixed_weight=True,
        maximum_xyz_gradient_difference=float((g[...,:3]-g0[...,:3]).abs().max()),
        note='Shared global gradient clipping may still affect both parameter groups; no separate-gradient claim.')
'''

def prepare():
    assert not PACK.exists(), 'Preserve existing experiment packages'
    baseline_dir=ROOT/'reports/bridge-coverage-rep-fit-207558'
    analysis=ROOT/'reports/bridge_coverage_replication_analysis_207558/analysis.json'
    a=read(analysis);assert a['verified'] and not a['candidate_passed']
    for line in (BASE/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);assert sha(BASE/n)==h
    PACK.mkdir()
    for name in ['rows.json','schedule.npz','episode_reference.json','coverage_inputs.py','reuse_control_proof.json']:
        shutil.copyfile(BASE/name,PACK/name)
    (PACK/'coverage_inputs.py').write_text(compact_helper((BASE/'coverage_inputs.py').read_text()),encoding='utf-8',newline='\n')
    (PACK/'rotation_loss.py').write_text(LOSS_SOURCE,encoding='utf-8',newline='\n')
    # Read TRAIN ONLY for coefficient selection. No development prediction or target is used here.
    with np.load(baseline_dir/'reduced_u8192_train.npz',allow_pickle=False) as z:
        targets=z['targets'].copy();assert np.array_equal(z['row_indices'],np.arange(1836))
    namespace={};exec(compile(LOSS_SOURCE.split('def rotation_pose_loss')[0].replace('import torch\n','').replace('import torch.nn.functional as F\n',''),'<training_statistics>','exec'),namespace)
    stats=namespace['training_statistics'](targets)
    stats.update(source_train_npz_sha256=sha(baseline_dir/'reduced_u8192_train.npz'),
        coefficient_selection='Ratio of train-only static xyz squared-distance sum to quaternion identity-chord squared-distance sum.',
        development_targets_or_predictions_used_to_choose_weight=False)
    write(PACK/'training_loss_scale.json',stats)
    p=read(BASE/'protocol.json');history=p.pop('replication')
    p['cache_origin']={k:history[k] for k in ['original_entry_sha256','original_helper_sha256','prior_preflight_report_sha256','prior_formal_report_sha256']}
    p['objective']='One fixed train-label-derived rotation-MSE weighting test; preserve large-pool position benefit.'
    p['rotation_loss']=dict(weight=stats['weight'],statistics_sha256=sha(PACK/'training_loss_scale.json'),
        helper_sha256=sha(PACK/'rotation_loss.py'),formula='original_mean7_MSE + (w-1)*4/7*quaternion_component_MSE',
        normalization_changed=False,noise_changed=False,sampler_changed=False,rotation_representation_changed=False,
        interpretation='Loss-weighting hypothesis only, not proof of incorrect old formula or official-paper prescribed coefficient.')
    p['comparison']=dict(control_job='207558',control_report_sha256=sha(baseline_dir/'report.json'),
        control_analysis_sha256=sha(analysis),initialization_seed=42,control_not_retrained=True,
        control_development=a['metrics']['development']['coverage'])
    dev=a['metrics']['development']['coverage']
    p['screen']=dict(rotation_at_most_deg=min(15.098500432102608,.95*dev['path_rotation_deg']),
        position_at_most_cm=dev['path_position_cm'],episodes_rotation_better_than_control_at_least=6,
        all40windows_and_all10episodes_included=True,all5tasks_reported=True)
    p['stopping']='One coefficient from training labels, one initialization42,8192updates,final only. No coefficient sweep, extra seed or fitting extension after outcome.'
    p['local_preflight']='224 verified observations only, packed GPU token storage with explicit global-to-resident lookup; batch8 unchanged. Not representative full-cache memory; complete cluster preflight required.'
    p['limits']+=['Rotation weight changes shared optimization/gradient clipping; parameter gradients are not isolated.',
        'Quaternion component MSE is not angular geodesic loss. This test does not change or validate the whole rotation representation.']
    write(PACK/'protocol.json',p)
    source=(BASE/'run_training_coverage.py').read_text(encoding='utf-8')
    source=once(source,'from coverage_inputs import load_inputs,tokens',
        'from coverage_inputs import load_inputs,tokens\nfrom rotation_loss import rotation_pose_loss,training_statistics,verify_loss_math')
    source=once(source,"        for path,h in p['inherited_sha256'].items():",'''        assert sha(CODE/'rotation_loss.py')==p['rotation_loss']['helper_sha256']
        assert sha(CODE/'training_loss_scale.json')==p['rotation_loss']['statistics_sha256']
        r['rotation_loss_weight']=p['rotation_loss']['weight']
        r['loss_helper_sha256']=sha(CODE/'rotation_loss.py')
        r['loss_math']=verify_loss_math(r['rotation_loss_weight'])
        (args.output/'rotation_loss_snapshot.py').write_bytes((CODE/'rotation_loss.py').read_bytes())
        (args.output/'protocol_snapshot.json').write_bytes((CODE/'protocol.json').read_bytes())
        for path,h in p['inherited_sha256'].items():''')
    source=once(source,"        state=torch.from_numpy(features).cuda();valid=set(r['verified_row_indices'])",'''        frozen_stats=json.loads((CODE/'training_loss_scale.json').read_text())
        if not args.local_bounded:
            actual_stats=training_statistics(targets[:1836].cpu().numpy())
            assert all(actual_stats[k]==frozen_stats[k] for k in actual_stats), 'Training scale changed'
        r['complete_training_loss_scale_verified']=not args.local_bounded
        state=torch.from_numpy(features).cuda();valid=set(r['verified_row_indices'])''')
    source=once(source,'loss=torch.nn.functional.mse_loss(guess,truth);assert torch.isfinite(loss)',
        "loss,loss_parts=rotation_pose_loss(guess,truth,p['rotation_loss']['weight']);assert torch.isfinite(loss)")
    source=once(source,'global_unclipped_gradient_norm=float(norm),unclipped_module_gradient_norms=grads)',
        'global_unclipped_gradient_norm=float(norm),unclipped_module_gradient_norms=grads,**loss_parts)')
    source=compact_runner(source)
    ast.parse(source);(PACK/'run_training_coverage.py').write_text(source,encoding='utf-8',newline='\n')
    v=(BASE/'validate_preflight.py').read_text(encoding='utf-8').replace("p['replication']","p['cache_origin']")
    v=once(v,"    assert r['full_token_cache_reused']",'''    assert r['rotation_loss_weight']==p['rotation_loss']['weight']
    assert r['complete_training_loss_scale_verified']
    assert r['loss_math']['weight1_forward_and_gradient_exact']
    assert r['loss_helper_sha256']==sha(CODE/'rotation_loss.py')==p['rotation_loss']['helper_sha256']
    baseline=args.directory.parent/'bridge-coverage-rep-fit-207558/report.json'
    assert sha(baseline)==p['comparison']['control_report_sha256']
    assert r['full_token_cache_reused']''')
    v=once(v,"    assert r['complete_training_loss_scale_verified']", "    assert r['complete_training_loss_scale_verified']\n    assert not r['local_compact_token_storage'] and r['resident_token_rows']==1876")
    ast.parse(v);(PACK/'validate_preflight.py').write_text(v,encoding='utf-8',newline='\n')
    for stage in ['preflight','fit']:
        s=(BASE/f'{stage}.sh').read_text().replace('bridge-coverage-rep-','bridge-rotation-weight-')
        s=s.replace('bridge_coverage_replication_v1','bridge_rotation_weight_v1').replace('COVERAGE_REP_PREFLIGHT_JOB','ROTATION_WEIGHT_PREFLIGHT_JOB')
        assert s.startswith('#!/bin/bash\n') and '\n+' not in s
        (PACK/f'{stage}.sh').write_text(s,encoding='utf-8',newline='\n')
    (PACK/'README.md').write_text('One train-only label-scale-derived quaternion-MSE weight control. '
        'Same initialization42,1836train/40development,8192updates and frozenCLIP/gripper; compare existing207558 without retraining control. '
        'Do not change target coordinates, normalization, noise or sampling. Require existing complete coverage cache. '
        'Strict position guard and preset5percent rotation gain with6/10episode rotation improvement; failure ends this coefficient test. '
        'No sweep, extra seed, increased steps, new assets or environment. Local bounded preflight cannot authorize formal.\n',encoding='utf-8')
    write(ROOT/'reports/bridge_rotation_weight_preparation_20261016.json',dict(stage='prepared_pending_local_preflight',
        weight=stats['weight'],selection_train_only=True,baseline_reused='207558',baseline_retrained=False,new_job_submitted=False,
        protocol_sha256=sha(PACK/'protocol.json'),screen=p['screen']))
    print(json.dumps(dict(weight=stats['weight'],screen=p['screen'],new_job_submitted=False),indent=2))

def audit():
    from analyze_bridge_growth64_200600 import summarize
    import subprocess
    assert not (LOCAL/'independent_audit.json').exists()
    r=read(LOCAL/'report.json');p=read(PACK/'protocol.json')
    old=read(ROOT/'reports/bridge_coverage_replication_local_preflight_20261016/report.json')
    assert r['execution_checks_passed'] and r['branch_restore_passed'] and r['local_bounded']
    assert r['optimization_operations']==16 and not r['development_scored'] and not r['checkpoints']
    assert not r['complete_training_loss_scale_verified'] and r['rotation_loss_weight']==p['rotation_loss']['weight']
    for k in ['initial_shared_sha256','augmented_initial_sha256','input_identity_sha256','input_row_checks','episode_payload_checks','verified_row_indices','retained_state_features_sha256']:
        assert r[k]==old[k],k
    for k in ['clip_unchanged','other_frozen_unchanged','retained_numeric_contract_passed','retained_visual_tokens_exact','retained_language_tokens_exact']:
        assert r[k],k
    assert r['entry_sha256']==sha(PACK/'run_training_coverage.py')==sha(LOCAL/'runner_snapshot.py')
    assert r['loss_helper_sha256']==sha(PACK/'rotation_loss.py')==sha(LOCAL/'rotation_loss_snapshot.py')
    assert r['protocol_sha256']==sha(PACK/'protocol.json')==sha(LOCAL/'protocol_snapshot.json')
    assert all(r['loss_math'][k] for k in ['weight1_forward_and_gradient_exact','xyz_loss_gradient_coefficient_preserved','quaternion_gradient_multiplied_by_fixed_weight'])
    for name in ['rows.json','schedule.npz','episode_reference.json']:
        assert (PACK/name).read_bytes()==(BASE/name).read_bytes()
    assert (PACK/'coverage_inputs.py').read_text()==compact_helper((BASE/'coverage_inputs.py').read_text())
    assert r['local_compact_token_storage'] and r['resident_token_rows']==224
    source=(PACK/'run_training_coverage.py').read_text(encoding='utf-8')
    source=once(source,'\nfrom rotation_loss import rotation_pose_loss,training_statistics,verify_loss_math','')
    source=once(source,COMPACT_LOOKUP,'')
    assert source.count('visual[token_ids[indices]],language[token_ids[indices]],mask[token_ids[indices]]')==2
    source=source.replace('visual[token_ids[indices]],language[token_ids[indices]],mask[token_ids[indices]]',
        'visual[indices],language[indices],mask[indices]')
    start=source.index("        assert sha(CODE/'rotation_loss.py')")
    end=source.index("        for path,h in p['inherited_sha256'].items():",start)
    source=source[:start]+source[end:]
    start=source.index("        frozen_stats=json.loads((CODE/'training_loss_scale.json').read_text())")
    end=source.index('        state=torch.from_numpy(features).cuda();valid=set(r[\'verified_row_indices\'])',start)
    source=source[:start]+source[end:]
    source=once(source,"loss,loss_parts=rotation_pose_loss(guess,truth,p['rotation_loss']['weight']);assert torch.isfinite(loss)",
        'loss=torch.nn.functional.mse_loss(guess,truth);assert torch.isfinite(loss)')
    source=once(source,'global_unclipped_gradient_norm=float(norm),unclipped_module_gradient_norms=grads,**loss_parts)',
        'global_unclipped_gradient_norm=float(norm),unclipped_module_gradient_norms=grads)')
    assert source==(BASE/'run_training_coverage.py').read_text(encoding='utf-8')
    basep=read(BASE/'protocol.json')
    for key in ['recipe','config','schedule','state_statistics','control_report_sha256','inherited_sha256']:
        assert p[key]==basep[key],key
    # Independently recompute the coefficient from training targets with plain NumPy.
    with np.load(ROOT/'reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz') as z:t=z['targets'].astype(float)
    pos=float((t[...,:3]**2).sum(-1).mean());rot=float(((t[...,3:7]-[0,0,0,1])**2).sum(-1).mean())
    assert pos/rot==p['rotation_loss']['weight']
    logs={}
    for arm,lo,hi,lr in [('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)]:
        logs[arm]=values=[json.loads(v) for v in (LOCAL/(arm+'_updates.jsonl')).read_text().splitlines()]
        assert [v['update'] for v in values]==list(range(lo,hi+1))
        for v in values:
            assert v['learning_rate']==lr and np.isfinite(v['loss'])
            expected=v['unweighted_pose_mse']+(p['rotation_loss']['weight']-1)*4/7*v['quaternion_component_mse']
            assert abs(v['loss']-expected)<=max(1e-6,abs(expected)*2e-6)
    assert logs['constant']==logs['restore_same_lr']
    delta=dict(position_cm=0.,rotation_deg=0.)
    cases=[('common',0,None),('common',4,r['common_train'])]+[(v['arm'],8,v['final_train']) for v in r['runs']]
    for arm,u,recorded in cases:
        with np.load(LOCAL/f'{arm}_u{u:04d}_train.npz') as z:pred,truth,ix,ctx=[z[k] for k in ['predictions','targets','row_indices','contexts']]
        assert np.array_equal(ix,[0,32,64,96]) and pred.shape==(3,4,16,8) and ctx.shape==(4,1024)
        with np.load(ROOT/'reports/bridge_coverage_replication_local_preflight_20261016/common_u0000_train.npz') as z:
            assert np.array_equal(truth,z['targets'])
            if u==0:assert np.array_equal(pred,z['predictions']) and np.array_equal(ctx,z['contexts'])
        m,_,_,_,_=summarize(pred,truth)
        if recorded:
            for k,v in m.items():
                kind='rotation_deg' if 'rotation' in k else 'position_cm';d=abs(v-recorded[k]);delta[kind]=max(delta[kind],d)
                assert d<(.02 if 'rotation' in k else 1e-4)
    with np.load(LOCAL/'constant_u0008_train.npz') as a,np.load(LOCAL/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    rejection=subprocess.run([sys.executable,str(PACK/'validate_preflight.py'),str(LOCAL),'--control',str(ROOT/'reports/bridge-state-rep-fit-202463/real/report.json')],capture_output=True,text=True)
    assert rejection.returncode!=0 and 'AssertionError' in rejection.stderr
    assert not (ROOT/'training_cache/coverage_full_tokens.pt').exists()
    assert r['peak_reserved_gib']<7.5
    result=dict(verified=True,report_sha256=sha(LOCAL/'report.json'),selection_training_only=True,weight=pos/rot,
        initialization224inputs_and_pretraining_predictions_exact=True,loss_and_gradient_math_verified=True,
        neural_runner_only_loss_and_logging_changes=True,recipe_and_sampling_unchanged=True,
        local_only_compact_token_lookup=True,local_cache_memory_not_representative=True,
        preflight_restore_exact=True,optimization_operations=16,prediction_snapshots_recomputed=5,max_saved_metric_difference=delta,
        local_bounded=True,development_scored=False,validator_rejects_local_preflight=True,
        full_cluster_training_scale_pending=True,elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib'],
        no_neural_execution_by_audit=True)
    write(LOCAL/'independent_audit.json',result);print(json.dumps(result,indent=2))

def package():
    if (LOCAL/'independent_audit.json').exists():
        r=read(LOCAL/'report.json');a=read(LOCAL/'independent_audit.json');assert a['verified'] and r['execution_checks_passed']
        assert r['entry_sha256']==sha(PACK/'run_training_coverage.py') and r['protocol_sha256']==sha(PACK/'protocol.json')
        local=dict(report_sha256=sha(LOCAL/'report.json'),audit_sha256=sha(LOCAL/'independent_audit.json'),
            local_bounded=True,local_complete_preflight_passed=True,development_scored=False,
            elapsed_seconds=r['elapsed_seconds'],peak_reserved_gib=r['peak_reserved_gib'])
    else:
        a=read(LOCAL/'independent_smoke_audit.json')
        assert a['verified_smoke_only'] and not a['local_complete_preflight_passed']
        assert a['entry_sha256']==sha(PACK/'run_training_coverage.py') and a['protocol_sha256']==sha(PACK/'protocol.json')
        local=dict(smoke_audit_sha256=sha(LOCAL/'independent_smoke_audit.json'),local_bounded=True,
            local_complete_preflight_passed=False,development_scored=False,full_cluster_preflight_required=True,
            completed_logged_updates=a['completed_logged_updates'],reason_local_interrupted='Other application GPU contention; do not disrupt user activity.')
    write(PACK/'local_preflight_summary.json',local)
    files=sorted(f for f in PACK.iterdir() if f.is_file() and f.name!='SHA256SUMS')
    assert all(f.suffix in ['.py','.json','.npz','.sh','.md'] for f in files)
    (PACK/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in files),encoding='ascii',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files+[PACK/'SHA256SUMS']:z.write(f,f.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(z.read(n)==(PACK.parent/n).read_bytes() for n in z.namelist())
    assert archive.stat().st_size<1024**2
    PACK.with_suffix('.sha256').write_text(f'{sha(archive)}  {archive.name}\n',encoding='ascii',newline='\n')
    print('PACKAGE',archive.stat().st_size,sha(archive))

def smoke():
    """Audit the completed prefix and pure CPU math; do not certify missing restore checks."""
    import subprocess
    assert not (LOCAL/'independent_smoke_audit.json').exists()
    p=read(PACK/'protocol.json')
    assert sha(LOCAL/'runner_snapshot.py')==sha(PACK/'run_training_coverage.py')
    assert sha(LOCAL/'rotation_loss_snapshot.py')==sha(PACK/'rotation_loss.py')
    assert sha(LOCAL/'protocol_snapshot.json')==sha(PACK/'protocol.json')
    for file in PACK.glob('*.py'):ast.parse(file.read_text())
    # This subprocess uses CPU tensors only; it creates no CUDA device or neural model.
    program="""import importlib.util,json,numpy as np
from pathlib import Path
pack=Path('cluster_packages/bridge_rotation_weight_v1')
spec=importlib.util.spec_from_file_location('rotation_loss',pack/'rotation_loss.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=json.loads((pack/'protocol.json').read_text());frozen=json.loads((pack/'training_loss_scale.json').read_text())
with np.load('reports/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz') as z:stats=m.training_statistics(z['targets'])
assert all(stats[k]==frozen[k] for k in stats)
print(json.dumps(dict(math=m.verify_loss_math(p['rotation_loss']['weight']),complete_training_statistics_cpu_exact=True)))
"""
    checked=subprocess.run(['D:/miniconda/envs/zeroshot-gpu/python.exe','-c',program],cwd=ROOT,capture_output=True,text=True)
    assert checked.returncode==0,checked.stderr
    math= json.loads(checked.stdout)
    logs={}
    for arm in ['common','constant']:
        f=LOCAL/(arm+'_updates.jsonl');logs[arm]=[json.loads(v) for v in f.read_text().splitlines()] if f.exists() else []
    assert [v['update'] for v in logs['common']]==[1,2,3,4]
    for values in logs.values():
        for v in values:
            assert np.isfinite(v['loss']) and np.isfinite(v['global_unclipped_gradient_norm'])
            assert set(v['unclipped_module_gradient_norms'])=={'adapter.','diffusion_decoder.','history_projection.','state_projection.'}
            assert all(np.isfinite(g) and g>=0 for g in v['unclipped_module_gradient_norms'].values())
            expected=v['unweighted_pose_mse']+(p['rotation_loss']['weight']-1)*4/7*v['quaternion_component_mse']
            assert abs(v['loss']-expected)<=max(1e-6,abs(expected)*2e-6)
    with np.load(LOCAL/'common_u0000_train.npz') as a,np.load(ROOT/'reports/bridge_coverage_replication_local_preflight_20261016/common_u0000_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    metrics=[]
    for u in [0,4]:
        f=LOCAL/f'common_u{u:04d}_train.npz'
        with np.load(f) as z:pred,truth=z['predictions'].astype(float),z['targets'].astype(float)
        assert pred.shape==(3,4,16,8) and np.isfinite(pred).all()
        q=pred[...,3:7];t=truth[None,...,3:7]
        q=q/np.linalg.norm(q,axis=-1,keepdims=True);t=t/np.linalg.norm(t,axis=-1,keepdims=True)
        pe=np.linalg.norm(pred[...,:3]-truth[None,...,:3],axis=-1)*10
        re=np.degrees(2*np.arccos(np.clip(np.abs((q*t).sum(-1)),0,1)))
        metrics.append(dict(update=u,position_cm=float(pe.mean()),rotation_deg=float(re.mean())))
    assert not (ROOT/'training_cache/coverage_full_tokens.pt').exists()
    result=dict(verified_smoke_only=True,local_complete_preflight_passed=False,entry_sha256=sha(PACK/'run_training_coverage.py'),
        protocol_sha256=sha(PACK/'protocol.json'),loss_helper_sha256=sha(PACK/'rotation_loss.py'),
        completed_logged_updates=sum(len(v) for v in logs.values()),common_four_updates_finite_and_loss_formula_verified=True,
        pretraining_predictions_targets_contexts_exact=True,source_snapshots_exact=True,cpu_verification=math,
        training_probe_snapshots=metrics,full_cluster_preflight_required=True,restore_check_completed=False,
        full_cluster_input_cache_memory_not_yet_verified=True,development_scored=False,
        local_interrupted_for_user_gpu_contention=True,no_new_neural_execution_by_audit=True)
    write(LOCAL/'independent_smoke_audit.json',result);print(json.dumps(result,indent=2))

def compact():
    """Apply the local-only memory fix to the unshipped draft; preserve interrupted evidence."""
    assert not PACK.with_suffix('.zip').exists()
    path=PACK/'coverage_inputs.py'
    path.write_text(compact_helper(path.read_text()),encoding='utf-8',newline='\n')
    path=PACK/'run_training_coverage.py'
    path.write_text(compact_runner(path.read_text()),encoding='utf-8',newline='\n')
    p=read(PACK/'protocol.json')
    p['local_preflight']='224 verified observations only, packed GPU token storage with explicit global-to-resident lookup; batch8 unchanged. Not representative full-cache memory; complete cluster preflight required.'
    write(PACK/'protocol.json',p)
    path=PACK/'validate_preflight.py'
    v=once(path.read_text(),"    assert r['complete_training_loss_scale_verified']", "    assert r['complete_training_loss_scale_verified']\n    assert not r['local_compact_token_storage'] and r['resident_token_rows']==1876")
    path.write_text(v,encoding='utf-8',newline='\n')
    print('Only local token residency compacted; interrupted first attempt preserved, no formal changes to batch/data/recipe.')

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','audit','package','compact','smoke']
    globals()[sys.argv[1]]()
