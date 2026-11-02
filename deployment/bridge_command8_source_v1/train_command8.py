"""One manually submitted8192-step source-only command8 candidate."""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

from command8_runtime import HERE,read,write,sha,admit,load_numeric,construct_training


def evaluate(b,a):
    import numpy as np
    import torch
    from command8_metrics import summarize
    from step_training_state import capture_rng,restore_rng
    rows=read(HERE/'vendor/selection.json')['rows'];saved={};results={}
    rng=capture_rng(include_cuda=True)
    try:
        b['model'].eval();b['port'].eval()
        for role in ('train','validation'):
            ids=[r['window_index'] for r in rows if r['role']==role];draws=[];target=[];mask=[]
            for seed in (1101,1102,1103):
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);parts=[]
                with torch.inference_mode():
                    for lo in range(0,len(ids),8):
                        x=b['pack'](role,ids[lo:lo+8])
                        context=b['model'].get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+b['port'](x['pose'])
                        parts.append(b['model'].sample(context,x['current']).cpu().numpy())
                        if seed==1101:target.append(x['actions'].cpu().numpy());mask.append(x['masks'].cpu().numpy())
                draws.append(np.concatenate(parts));print('COMMAND8_SOURCE_GENERATION',role,seed,len(ids),flush=True)
            pred=np.stack(draws);truth=np.concatenate(target);valid=np.concatenate(mask)
            for name,value in [('predictions',pred),('targets',truth),('masks',valid),('window_indices',np.array(ids,np.int64))]:
                saved[role+'_'+name]=value
            results[role]=summarize(pred,truth,valid,a['train_mean_command'])
    finally:restore_rng(rng)
    b['check_frozen']()
    return results,saved


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    root=args.root.resolve();admission=admit(root)
    if args.check_only:
        import json
        print('COMMAND8_SOURCE_ADMISSION',json.dumps(admission),flush=True);return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:raise ValueError('Manual Slurm allocation required')
    out=args.output.resolve()
    if out.parent!=root/'runs' or not out.name.startswith('bridge-command8-source-') or out.exists():
        raise ValueError('Fresh dedicated output required; no automatic retry')
    import numpy as np
    import torch
    import bcap_core
    from completed_policy_export import model_sha
    p=read(HERE/'protocol.json');out.mkdir();write(out/'admission.json',admission)
    started=time.monotonic();b=None
    report={'execution_complete':False,'target_supervised':False,'formal_matrix_admitted':False,
        'zero_shot_transfer_proven':False,'completed_updates':0,'encoder_updates':0,
        'source_checkpoint_sha256':p['source_checkpoint_sha256'],'effective_action_representation':p['action_representation'],
        'source_semantics_status':p['source_semantics_status'],'automatic_rollout':False,'automatic_following_action':False}
    def stop(signum,frame):
        if b is not None:b['request_stop']=True
    def hard(signum,frame):raise TimeoutError('5100s process cap; preserve last atomic checkpoint, no continuation')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,hard)
    signal.alarm(p['process_cap_seconds'])
    try:
        a,rows=load_numeric();b=construct_training(root,a,rows)
        torch.cuda.reset_peak_memory_stats()
        b['api'].set_seed(p['training_seed'])
        guard=bcap_core.zero_guard(b);write(out/'zero_update_guard.json',guard)
        extra={'history':[],'training_seconds':0.,'zero_update_guard':guard,'initial_identity':b['initial_identity']}
        # The inherited native loss uses NEW command targets supplied by b.batch.
        extra=bcap_core.train(b,out,extra)
        report['completed_updates']=b['plan'].completed_updates
        if report['completed_updates']!=8192:raise ValueError('Fixed endpoint not reached')
        result,saved=evaluate(b,a)
        np.savez_compressed(out/'predictions.npz',**saved)
        identity=bcap_core.identity(b)
        passed=all(v['source_engineering_review_gate_passed'] for v in result.values())
        payload={'format':'bridge-command8-source-policy-v1','completed_updates':8192,
            'source_completed_updates_before_label_change':65536,
            'state':bcap_core.state(b),'final_identity':identity,
            'source_constructor_config':b['source_constructor_config'],'source_stats':b['stats'],
            'contract':b['contract'],'effective_action_representation':p['action_representation'],
            'source_engineering_review_gate_passed':passed,'target_supervised':False,
            'formal_matrix_admitted':False,'zero_shot_transfer_proven':False}
        temporary=out/'policy.pt.tmp';torch.save(payload,temporary);temporary.replace(out/'policy.pt')
        received=torch.load(out/'policy.pt',map_location='cpu',weights_only=False,mmap=True)
        if {k:model_sha(v) for k,v in received['state'].items()}!=identity:raise ValueError('Policy state readback differs')
        if sha(root/p['source_checkpoint'])!=p['source_checkpoint_sha256']:raise ValueError('Protected original source file changed')
        report.update(execution_complete=True,metrics=result,initial_identity=b['initial_identity'],final_identity=identity,
            source_engineering_review_gate_passed=passed,training_seconds=extra['training_seconds'],
            policy_sha256=sha(out/'policy.pt'),original_source_file_sha_unchanged=True,
            max_current_pose_feature_abs_delta=b.get('max_pose_feature_abs_delta',0.),
            current_pose_feature_tolerance=1e-6,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            peak_scope='After construction, includes zero guard/training/evaluation',
            program_elapsed_seconds=time.monotonic()-started,
            elapsed_scope='After CPU admission, includes construction/guard/train/sample/policy save, before final verification/ZIP',
            constructor_config_scope='Old source architecture/frozen restoration template only; effective labels are the explicit new command8 contract')
        write(out/'report.json',report);write(out/'history.json',extra['history'])
        from verify_command8 import verify
        write(out/'verification.json',verify(out,full=True))
        import zipfile
        with zipfile.ZipFile(out/'small_results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('report.json','predictions.npz','history.json','admission.json','zero_update_guard.json','verification.json'):z.write(out/name,name)
        import json
        compact={k:v for k,v in report.items() if k!='metrics'}
        compact['metrics']={r:{'first_position_normalized':v['metrics']['model']['first_position_normalized'],
            'first_rotation_deg':v['metrics']['model']['first_rotation_deg'],
            'gate':v['source_engineering_review_gate_passed']} for r,v in result.items()}
        print('BRIDGE_COMMAND8_SOURCE_COMPLETE',json.dumps(compact),flush=True)
    except BaseException as e:
        report.update(error=type(e).__name__+': '+str(e),program_elapsed_seconds=time.monotonic()-started)
        if b is not None:report['completed_updates']=b['plan'].completed_updates
        # Never resave potentially partly applied optimizer steps on exception.
        write(out/'report.json',report);raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
