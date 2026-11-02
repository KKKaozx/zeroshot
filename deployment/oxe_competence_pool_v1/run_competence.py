"""One moderate-pool arm; reuse the completed control, never train it again."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import competence_bind as binding


def make_contract(p,bound):
    return {'format':'oxe-moderate-pool-competence-contract-v1',
        'protocol_sha256':binding.sha(binding.HERE/'protocol.json'),'source_binding':bound,
        'plans_sha256':binding.sha(binding.HERE/'plans.npz'),'pool_rows_sha256':binding.sha(binding.HERE/'pool_rows.json'),
        'updates_each':p['updates_each'],'checkpoint_every':p['checkpoint_every'],
        'arm_training_cap_seconds':p['arm_training_cap_seconds'],
        'frozen_sha256':p['source_identity']['frozen_state_sha256'],
        'training_text_padding':77,'native_lr_horizon':65536,'retrain_control':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=a.root.resolve();bound=binding.admit(root)
    if a.check_only:
        print('MODERATE POOL CPU ADMISSION PASSED; no model construction, CUDA or submission')
        print(json.dumps(bound,indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    if a.resume:assert a.output.is_dir()
    else:a.output.mkdir(exist_ok=False)
    pq=binding.parent(root);q,pg=pq.attach(root);pg.initialize_runtime()
    from instrument_formal import prepare_source
    from pair_steps import initial_identity,train,sample,check_paired_stochastic
    from sampling_verify import verify_training
    from competence_metrics import summarize
    from competence_probe import run as probe
    from competence_verify import verify_outputs
    loader,loader_source=pq.install_loader(q)
    (a.output/'effective_source_loader.py').write_bytes(loader_source.encode())
    p=binding.read(binding.HERE/'protocol.json');plans=binding.load_plans();c=make_contract(p,bound)
    saved,old_report=binding.control(root);control=saved['trace'];old=root/'runs/oxe-sampling-pair-233531'
    old_state=old/'replacement/latest_state.pt';old_stamp=[old_state.stat().st_size,old_state.stat().st_mtime_ns]
    r={'execution_complete':False,'contract':c,'automatic_following_action':False,'main_model_replaced':False,
        'new_control_training_updates':0,'training':None,'metrics':None};tick=time.monotonic()
    try:
        b=loader(root);b['root']=root;fresh=initial_identity(b)
        assert fresh==saved['receipt']['initial_identity'],'Source learned state/Adam/RNG must match completed control'
        inputs_source=prepare_source(pg);exec(compile(inputs_source,'<qualified-original120-inputs>','exec'),pg.__dict__)
        (a.output/'effective_evaluation_inputs.py').write_bytes(inputs_source.encode())
        selection=binding.read(binding.HERE/'selection.json')
        cache=pg.prepare_inputs(root,b['data'],b['model'],b['port'],b['tokenizer'],b['api'],b['stats'],selection)
        prediction=sample(b,cache,p['sampling_seeds']);outputs={}
        with np.load(old/'predictions_and_targets.npz') as z:
            assert np.array_equal(prediction,z['source232587']),'Original source120 replay differs'
            assert np.array_equal(cache['targets'],z['targets']) and np.array_equal(cache['masks'],z['masks'])
            outputs={k:z[k].copy() for k in ['source232587','static','replacement']}
        with np.load(root/'runs/oxe-pose-pair-232587/evaluation/predictions_and_targets.npz') as z:
            assert np.array_equal(prediction,z['real_pose'])
            assert np.array_equal(cache['targets'],z['targets']) and np.array_equal(cache['masks'],z['masks'])
            assert np.array_equal(outputs['static'],z['static'])
        guard=probe(b,plans,control);guard_path=a.output/'zero_update_guard.json'
        if guard_path.exists():assert binding.read(guard_path)==guard
        else:guard_path.write_text(json.dumps(guard,indent=2)+'\n')
        r['zero_update_guard']=guard
        folder=a.output/'moderate_pool';folder.mkdir(exist_ok=a.resume)
        if (folder/'receipt.json').exists():
            assert a.resume
            verified=verify_training(folder,'moderate_pool',c,plans);receipt=verified['receipt'];trace=verified['trace']
            with np.load(folder/'predictions.npz') as z:outputs['moderate_pool']=z['predictions']
        else:
            receipt,trace=train(b,'moderate_pool',plans,folder,c,control=control,resume=a.resume)
            outputs['moderate_pool']=sample(b,cache,p['sampling_seeds'])
            np.savez_compressed(folder/'predictions.npz',predictions=outputs['moderate_pool'],targets=cache['targets'],masks=cache['masks'])
            receipt.update(initial_identity=fresh,prediction_file_sha256=binding.sha(folder/'predictions.npz'),
                source_file_size_mtime_unchanged=[b['source_path'].stat().st_size,b['source_path'].stat().st_mtime_ns]==bound['source_binding']['source_file_size_mtime'])
            assert receipt['source_file_size_mtime_unchanged']
            (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            verify_training(folder,'moderate_pool',c,plans)
        for row,event,oldrow,oldevent in zip(trace['history'],trace['noise'],control['history'],control['noise']):
            check_paired_stochastic(row,event,oldrow,oldevent)
        assert [old_state.stat().st_size,old_state.stat().st_mtime_ns]==old_stamp==bound['control_checkpoint_size_mtime']
        r['training']=receipt;r['metrics']=summarize(outputs,cache['targets'],cache['masks'],selection)
        np.savez_compressed(a.output/'predictions_and_targets.npz',targets=cache['targets'],masks=cache['masks'],**outputs)
        r.update(execution_complete=True,source_replay_exact=True,all_actual_noise_t_and_rng_matched=True,
            actual_matched_steps=p['updates_each'],control_checkpoint_unchanged=True,
            control_training_receipt=saved['receipt'],training_job_ids=receipt['training_job_ids'],
            old_control_training_job_ids=old_report['training_job_ids'],same_training_allocation=False,
            elapsed_seconds=time.monotonic()-tick)
    finally:
        (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
        print('MODERATE_POOL_REPORT',a.output/'report.json',flush=True)
    print('MODERATE_POOL_METRICS',json.dumps(verify_outputs(root,a.output),ensure_ascii=False),flush=True)
    print('MODERATE POOL COMPLETE; seen-training-anchor diagnostic only; no automatic extension or LIBERO',flush=True)


if __name__=='__main__':main()
