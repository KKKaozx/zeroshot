"""Saved checkpoint/primitive RNG/metrics validation; no new inference or training."""
import argparse
import json
from pathlib import Path
import numpy as np
from pair_qualification import HERE,read,sha,attach


def verify_training(folder,role,contract,plans):
    import torch
    from pair_state import FORMAT,validate_trace
    from step_training_state import UpdateSchedule
    from completed_policy_export import contract_sha,model_sha
    from pose_runtime import digest
    r=read(folder/'receipt.json');p=torch.load(folder/'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
    assert r['training_passed'] and r['role']==role and r['updates_each']==contract['updates_each']
    assert p['format']==FORMAT and p['role']==role and p['contract']==contract
    assert p['contract_sha256']==contract_sha(contract)==r['contract_sha256']
    assert p['counter']==contract['updates_each'] and r['completed_source_schedule']==10240+contract['updates_each']
    assert model_sha(p['base'])==r['base_learned_sha256'] and digest(p['pose_port'])==r['port_sha256']
    assert r['frozen_sha256']==contract['frozen_sha256'] and r['source_file_size_mtime_unchanged']
    assert len(p['optimizer']['param_groups'])==3 and p['optimizer']['state']
    assert [g['lr'] for g in p['optimizer']['param_groups']]==UpdateSchedule(**p['schedule']).rates(10240+p['counter'])
    assert p['training_job_ids']==r['training_job_ids']
    assert r['prediction_file_sha256']==sha(folder/'predictions.npz')
    trace={'history':p['history'],'noise':p['noise']};assert len(trace['history'])==len(trace['noise'])==contract['updates_each']
    validate_trace(p,role,plans)
    return {'receipt':r,'trace':trace}


def verify_outputs(root,output):
    from sampling_metrics import summarize
    from pair_steps import check_paired_stochastic
    p=read(HERE/'protocol.json');r=read(output/'report.json');contract=r['contract']
    assert r['execution_complete'] and contract['protocol_sha256']==sha(HERE/'protocol.json')
    assert r['source_replay_exact'] and r['all_actual_noise_t_and_rng_matched'] and r['actual_paired_steps_matched']==p['updates_each']
    assert r['zero_update_guard']['passed'] and r['zero_update_guard']['optimizer_updates']==0
    assert r['zero_update_guard']==read(output/'paired_zero_update_guard.json')
    with np.load(HERE/'plans.npz') as z:plans={k:z[k] for k in z.files}
    arms={role:verify_training(output/role,role,contract,plans) for role in ['replacement','cycle']}
    assert arms['replacement']['receipt']['initial_identity']==arms['cycle']['receipt']['initial_identity']
    assert r['zero_update_guard']['initial_identity']==arms['replacement']['receipt']['initial_identity']
    assert r['training']=={role:v['receipt'] for role,v in arms.items()}
    jobs=sorted(set(j for arm in arms.values() for j in arm['receipt']['training_job_ids']))
    assert r['training_job_ids']==jobs and r['same_training_allocation']==(len(jobs)==1)
    for row,event,oldrow,oldevent in zip(arms['cycle']['trace']['history'],arms['cycle']['trace']['noise'],arms['replacement']['trace']['history'],arms['replacement']['trace']['noise']):
        check_paired_stochastic(row,event,oldrow,oldevent)
    selection=read(root/'oxe_target_pair_short_v1/selection.json')
    with np.load(output/'predictions_and_targets.npz') as z:
        target=z['targets'];mask=z['masks'];outputs={k:z[k] for k in ['replacement','cycle','source232587','static']}
        assert target.shape==mask.shape==(120,16,8) and all(v.shape==(3,120,16,8) and np.isfinite(v).all() for v in outputs.values())
        with np.load(root/'runs/oxe-pose-pair-232587/evaluation/predictions_and_targets.npz') as old:
            assert np.array_equal(target,old['targets']) and np.array_equal(mask,old['masks'])
            assert np.array_equal(outputs['source232587'],old['real_pose']) and np.array_equal(outputs['static'],old['static'])
        assert summarize(outputs,target,mask,selection)==r['metrics']
        for role in ['replacement','cycle']:
            with np.load(output/role/'predictions.npz') as arm:
                assert np.array_equal(arm['predictions'],outputs[role]) and np.array_equal(arm['targets'],target) and np.array_equal(arm['masks'],mask)
    m=r['metrics']
    return {'verified':True,'updates_each':p['updates_each'],'actual_noise_t_model_rng_matched':True,
        'primary_bridge64':{name:{k:v for k,v in values.items() if k!='per_window'} for name,values in m['primary_bridge64'].items()},
        'paired_comparisons':m['paired_comparisons'],'sampling_effect_gates':m['sampling_effect_gates'],
        'basic_competence_gates':m['basic_competence_gates'],'validation_source_regression_flags':m['validation_source_regression_flags'],
        'candidate_for_review_only':m['candidate_for_review_only'],'scope':m['scope'],'automatic_following_action':False}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'));ap.add_argument('output',type=Path)
    a=ap.parse_args();attach(a.root.resolve());result=verify_outputs(a.root.resolve(),a.output)
    print('SAMPLING PAIR OUTPUT VERIFIED; development results only')
    print(json.dumps(result,indent=2))
