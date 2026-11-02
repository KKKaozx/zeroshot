"""One bounded A40 paired training/evaluation job, with committed exact resume states."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from pair_qualification import HERE,read,sha,admit,attach,install_loader


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=a.root.resolve();bound=admit(root)
    if a.check_only:
        print('SAMPLING PAIR ACTUAL CPU ADMISSION PASSED; no GPU/model/submission')
        print(json.dumps(bound,indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    if a.resume:assert a.output.is_dir()
    else:a.output.mkdir(exist_ok=False)
    q,pg=attach(root);pg.initialize_runtime()
    import torch
    from instrument_formal import prepare_source
    from pair_steps import preflight_pair_rng,initial_identity,train,sample
    from sampling_metrics import summarize
    from sampling_verify import verify_training,verify_outputs
    loader,loader_source=install_loader(q)
    (a.output/'effective_source_loader.py').write_bytes(loader_source.encode())
    with np.load(HERE/'plans.npz') as z:plans={k:z[k] for k in z.files}
    p=read(HERE/'protocol.json');contract={'format':'bounded-sampling-pair-contract-v1',
        'protocol_sha256':sha(HERE/'protocol.json'),'source_binding':bound,'plans_sha256':sha(HERE/'plans.npz'),
        'updates_each':p['updates_each'],'checkpoint_every':p['checkpoint_every'],
        'arm_training_cap_seconds':p['arm_training_cap_seconds'],'frozen_sha256':p['source_identity']['frozen_state_sha256'],
        'training_text_padding':77,'native_lr_horizon':65536,'original_source_plan_not_overwritten':True}
    report={'execution_complete':False,'contract':contract,'training':{},'metrics':None,
        'automatic_following_action':False,'main_model_replaced':False};tick=time.monotonic()
    cache=None;outputs={};control=None
    try:
        for role in ['replacement','cycle']:
            folder=a.output/role;folder.mkdir(exist_ok=a.resume)
            bundle=loader(root);bundle['root']=root;fresh=initial_identity(bundle)
            if cache is None:
                inputs_source=prepare_source(pg)
                exec(compile(inputs_source,'<qualified-original120evaluation-inputs>','exec'),pg.__dict__)
                selection=read(root/'oxe_target_pair_short_v1/selection.json')
                cache=pg.prepare_inputs(root,bundle['data'],bundle['model'],bundle['port'],bundle['tokenizer'],bundle['api'],bundle['stats'],selection)
                (a.output/'effective_evaluation_inputs.py').write_bytes(inputs_source.encode())
                prediction=sample(bundle,cache,p['sampling_seeds'])
                with np.load(root/'runs/oxe-pose-pair-232587/evaluation/predictions_and_targets.npz') as z:
                    assert np.array_equal(prediction,z['real_pose']),'Source232587 must replay before new training'
                    assert np.array_equal(cache['targets'],z['targets']) and np.array_equal(cache['masks'],z['masks'])
                    outputs['source232587']=z['real_pose'].copy();outputs['static']=z['static'].copy()
                guard=preflight_pair_rng(bundle,plans)
                guard_path=a.output/'paired_zero_update_guard.json'
                if guard_path.exists():assert read(guard_path)==guard
                else:guard_path.write_text(json.dumps(guard,indent=2)+'\n')
                report['zero_update_guard']=guard
            if (folder/'receipt.json').exists():
                assert a.resume
                saved=verify_training(folder,role,contract,plans)
                with np.load(folder/'predictions.npz') as z:outputs[role]=z['predictions']
                receipt=saved['receipt'];trace=saved['trace']
            else:
                receipt,trace=train(bundle,role,plans,folder,contract,control=control,resume=a.resume)
                outputs[role]=sample(bundle,cache,p['sampling_seeds'])
                np.savez_compressed(folder/'predictions.npz',predictions=outputs[role],targets=cache['targets'],masks=cache['masks'])
                receipt.update(initial_identity=fresh,prediction_file_sha256=sha(folder/'predictions.npz'),
                    source_file_size_mtime_unchanged=[bundle['source_path'].stat().st_size,bundle['source_path'].stat().st_mtime_ns]==bound['source_file_size_mtime'])
                assert receipt['source_file_size_mtime_unchanged']
                (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                verify_training(folder,role,contract,plans)
            report['training'][role]=receipt
            if role=='replacement':control=trace
            else:
                assert receipt['initial_identity']==report['training']['replacement']['initial_identity']
                from pair_steps import check_paired_stochastic
                for row,event,oldrow,oldevent in zip(trace['history'],trace['noise'],control['history'],control['noise']):
                    check_paired_stochastic(row,event,oldrow,oldevent)
            del bundle;gc.collect();torch.cuda.empty_cache()
        report['metrics']=summarize(outputs,cache['targets'],cache['masks'],selection)
        np.savez_compressed(a.output/'predictions_and_targets.npz',targets=cache['targets'],masks=cache['masks'],**outputs)
        jobs=sorted(set(j for r in report['training'].values() for j in r['training_job_ids']))
        report.update(execution_complete=True,source_replay_exact=True,all_actual_noise_t_and_rng_matched=True,
            actual_paired_steps_matched=p['updates_each'],
            training_job_ids=jobs,same_training_allocation=len(jobs)==1,elapsed_seconds=time.monotonic()-tick)
    finally:
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('SAMPLING_PAIR_REPORT',a.output/'report.json',flush=True)
    verified=verify_outputs(root,a.output)
    print('SAMPLING_PAIR_METRICS',json.dumps(verified,ensure_ascii=False),flush=True)
    print('SAMPLING PAIR COMPLETE; fixed development diagnostic, no automatic extra training or LIBERO',flush=True)


if __name__=='__main__':main()
