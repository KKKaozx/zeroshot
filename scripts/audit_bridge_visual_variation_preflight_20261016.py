"""Verify the bounded visual-control preflight from saved data, without neural inference."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from analyze_bridge_growth64_200600 import summarize

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_visual_variation_v1'
DATA=ROOT/'reports/bridge_visual_variation_local_preflight_20261016'
sys.path.insert(0,str(PACK))
from run_visual_control import effective_source
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r=read(DATA/'report.json'); protocol=read(PACK/'protocol.json')
    helper=read(ROOT/'reports/bridge_visual_variation_plan_20261016/runtime_helper_cpu_check.json')
    assert helper['passed'] and helper['entry_sha256']==sha(PACK/'run_visual_control.py')
    assert r['stage']=='preflight' and r['state_arm']=='real' and r['final_update']==8
    for key in ['execution_checks_passed','input_contract_passed','branch_restore_passed','clip_unchanged',
                'other_frozen_unchanged','all_queries_same_visual_tokens','full_token_cache_reused']:
        assert r[key],key
    assert r['constant_visual_fit_rows']==list(range(128)) and not r['original_control_retrained']
    for key in ['language_inputs_changed','observed_state_inputs_changed','action_targets_changed',
                'gripper_trained','simulation_executed','downloads','reserved_test_targets_read']:
        assert not r[key],key
    assert r['visual_protocol_sha256']==sha(PACK/'protocol.json')
    assert r['visual_entry_sha256']==sha(PACK/'run_visual_control.py')
    assert r['encoder_cache_helper_sha256']==sha(ROOT/'cluster_packages/bridge_inner_selection_v1/selection_contract.py')
    for name,expected in protocol['inherited_sha256'].items():assert sha(PACK.parent/name)==expected
    assert (DATA/'effective_runner.py').read_text(encoding='utf-8')==effective_source()
    assert sha(DATA/'runner_snapshot.py')==protocol['inherited_sha256']['bridge_state_conditioning_v1/fit_state_arm.py']
    old=read(ROOT/'reports/bridge_state_prior_preflight_v2_20261015/report.json')
    assert r['initial_shared_sha256']==old['initial_shared_sha256']
    assert r['data_checks']==old['data_checks']
    assert r['state_features_sha256']==old['state_features_sha256']
    assert r['sampling_steps']==100 and r['sampling_seeds']==[1101,1102,1103]
    assert not list(DATA.glob('*.pt')) and not r['checkpoints']
    for name,first,last,lr in [('common',1,4,1e-4),('constant',5,8,1e-4),('restore_same_lr',5,8,1e-4),('reduced',5,8,1e-5)]:
        lines=[json.loads(v) for v in (DATA/f'{name}_updates.jsonl').read_text().splitlines()]
        assert [v['update'] for v in lines]==list(range(first,last+1))
        assert all(v['learning_rate']==lr and np.isfinite(v['loss']) for v in lines)
        assert all(all(np.isfinite(g) and g>=0 for g in v['unclipped_module_gradient_norms'].values()) for v in lines)
    assert (DATA/'constant_updates.jsonl').read_bytes()==(DATA/'restore_same_lr_updates.jsonl').read_bytes()
    snapshots=[('common',v['update'],v['train']) for v in r['common_history']]
    for arm in r['runs']:snapshots += [(arm['arm'],v['update'],v['train']) for v in arm['history']]
    assert len(snapshots)==5
    with np.load(ROOT/'reports/bridge-state-fit-202278/real/reduced_u8192_train.npz') as z:targets=z['targets'].copy()
    maximum=dict(position_cm=0.,rotation_deg=0.)
    for name,update,record in snapshots:
        with np.load(DATA/f'{name}_u{update:04d}_train.npz') as z:
            p,t,ix=z['predictions'],z['targets'],z['row_indices']
            assert ix.tolist()==[0,32,64,96] and p.shape==(3,4,16,8)
            assert np.max(np.abs(t-targets[ix]))<1e-6
        actual,_,_,_,_=summarize(p,t)
        for key,value in actual.items():
            delta=abs(value-record[key]);typ='rotation_deg' if 'rotation' in key else 'position_cm'
            maximum[typ]=max(maximum[typ],delta);assert delta<(.02 if 'rotation' in key else 1e-4)
    with np.load(DATA/'constant_u0008_train.npz') as a,np.load(DATA/'restore_same_lr_u0008_train.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    assert not list(DATA.glob('*development*'))
    result=dict(verified=True,entry_sha256=sha(PACK/'run_visual_control.py'),report_sha256=sha(DATA/'report.json'),
        protocol_sha256=sha(PACK/'protocol.json'),raw_inputs_and_targets_unchanged=True,
        actual_runtime_mean_helper_verified=True,original_local_initialization_and_state_features_exact=True,
        restored_parameter_loss_and_prediction_checks_passed=True,operational_updates=16,
        max_saved_metric_difference=maximum,neural_training_rerun_by_audit=False,development_scored=False,
        limits='Bounded execution proof only; no formal or generalization result. Parameter/Adam replay uses runtime assertions, no weight checkpoint saved locally.')
    (DATA/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
