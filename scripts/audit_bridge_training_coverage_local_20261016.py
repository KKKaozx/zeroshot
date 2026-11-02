"""Audit metadata and saved bounded local preflight, without a neural rerun."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_training_coverage_v1'
OUT=ROOT/'reports/bridge_training_coverage_local_preflight_v2_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert not (OUT/'independent_audit.json').exists()
    p=read(PACK/'protocol.json');r=read(OUT/'report.json')
    old=read(ROOT/'reports/bridge_visual_variation_local_preflight_20261016/report.json')
    manifest=read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    rows=read(PACK/'rows.json');all_rows=rows['train']+rows['development']
    assert r['execution_checks_passed'] and r['branch_restore_passed'] and r['local_bounded']
    assert not r['full_input_contract_passed'] and r['active_token_rows']<1876
    assert not r['development_scored'] and not r['checkpoints'] and r['optimization_operations']==16
    for k in ['clip_unchanged','other_frozen_unchanged','retained_numeric_contract_passed','retained_visual_tokens_exact','retained_language_tokens_exact']:
        assert r[k],k
    assert r['entry_sha256']==sha(PACK/'run_training_coverage.py')
    assert r['input_helper_sha256']==sha(PACK/'coverage_inputs.py')
    assert r['protocol_sha256']==sha(PACK/'protocol.json')
    assert sha(OUT/'runner_snapshot.py')==r['entry_sha256']
    for path,h in p['inherited_sha256'].items():assert sha(PACK.parent/path)==h
    for actual,expected in [('retained_data_checks','data_checks'),('retained_state_checks','state_checks'),
        ('retained_state_features_sha256','state_features_sha256'),('initial_shared_sha256','initial_shared_sha256'),
        ('augmented_initial_sha256','augmented_initial_sha256')]:assert r[actual]==old[expected],actual
    with np.load(PACK/'schedule.npz') as z:schedule=z['batches'].copy()
    original=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)]).numpy()
    tasks=np.array([v['task'] for v in all_rows])
    assert np.array_equal(tasks[schedule],np.array([v['task'] for v in manifest['train']])[original])
    assert all(len(set(v))==8 for v in schedule) and schedule.min()>=0 and schedule.max()<1836
    assert np.bincount(schedule.ravel(),minlength=1836).min()==23
    assert set(schedule[:8].ravel())<=set(r['verified_row_indices'])
    ep=lambda v:(v['shard'],v['record_index'])
    assert len(set(map(ep,rows['train'])))==405 and len(set(map(ep,rows['development'])))==10
    assert not set(map(ep,rows['train'])) & set(map(ep,rows['development']))
    assert [(v['shard'],v['record_index'],v['start_index']) for v in rows['development']]==[(v['shard'],v['record_index'],v['start_index']) for v in manifest['development']]
    log=lambda name:[json.loads(v) for v in (OUT/(name+'_updates.jsonl')).read_text().splitlines()]
    assert log('constant')==log('restore_same_lr')
    for name,first,last,lr in [('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)]:
        entries=log(name);assert [v['update'] for v in entries]==list(range(first,last+1))
        assert all(v['learning_rate']==lr and np.isfinite(v['loss']) for v in entries)
        assert all(set(v['unclipped_module_gradient_norms'])=={'adapter.','diffusion_decoder.','history_projection.','state_projection.'} for v in entries)
    cases=[('common',0,None),('common',4,r['common_train'])]
    cases += [(v['arm'],8,v['final_train']) for v in r['runs']]
    delta=dict(position_cm=0.,rotation_deg=0.)
    for arm,step,recorded in cases:
        with np.load(OUT/f'{arm}_u{step:04d}_train.npz') as z:
            pred,target,ix,ctx=[z[k] for k in ['predictions','targets','row_indices','contexts']]
        assert np.array_equal(ix,[0,32,64,96]) and pred.shape==(3,4,16,8) and ctx.shape==(4,1024)
        assert np.isfinite(pred).all() and np.isfinite(ctx).all()
        with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:
            assert np.max(np.abs(target-z['targets'][ix]))<=1e-6
        a,b=pred[...,3:7].astype(float),target[None,...,3:7].astype(float)
        assert np.linalg.norm(a,axis=-1).min()>1e-8
        a/=np.linalg.norm(a,axis=-1,keepdims=True);b/=np.linalg.norm(b,axis=-1,keepdims=True)
        pos=np.linalg.norm(pred[...,:3].astype(float)-target[None,...,:3],axis=-1)*10
        rot=np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)),0,1)))
        if recorded:
            for key,value in [('path_position_cm',pos.mean()),('path_rotation_deg',rot.mean()),
                ('endpoint_position_cm',pos[...,-1].mean()),('endpoint_rotation_deg',rot[...,-1].mean())]:
                kind='rotation_deg' if 'rotation' in key else 'position_cm'
                d=abs(float(value)-recorded[key]);delta[kind]=max(delta[kind],d)
                assert d<(.02 if 'rotation' in key else 1e-4)
    with np.load(OUT/'constant_u0008_train.npz') as a,np.load(OUT/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    assert not (ROOT/'training_cache/coverage_full_tokens.pt').exists()
    assert r['peak_reserved_gib']<7.5
    result=dict(verified=True,entry_sha256=r['entry_sha256'],input_helper_sha256=r['input_helper_sha256'],
        protocol_sha256=r['protocol_sha256'],report_sha256=sha(OUT/'report.json'),
        metadata_full405_and_original128_40_identity_verified=True,all65536_task_slots_exact=True,
        local_bounded=True,verified_input_rows=len(r['verified_row_indices']),
        full_pool_numerically_validated_locally=False,development_scored=False,
        original_local_initialization_state_and_retained_inputs_exact=True,
        branch_restore_saved_predictions_and_losses_exact=True,saved_prediction_snapshots=5,
        max_saved_metric_difference=delta,no_neural_execution_by_audit=True,no_local_full_token_cache=True,
        limits='Local bounded preflight cannot authorize formal. Cluster complete415-episode1876-window preflight and cache required.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
