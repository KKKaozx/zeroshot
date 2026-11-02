"""One explicitly authorized single-source run; no follow-on submissions."""
import argparse
import os
from pathlib import Path
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'code'),str(HERE)]


def main():
    import numpy as np
    import torch
    from bcap_runtime import admit,setup,construct,attach_data,evaluate,read,write,sha
    from bcap_core import zero_guard,identity,save,resume,train,summarize
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--resume',action='store_true');a=ap.parse_args();root=a.root.resolve()
    admission=admit(root)
    if a.check_only:
        print('BRIDGE_CAPABILITY_CPU_ADMISSION',__import__('json').dumps(admission),flush=True);return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None
    out=a.output.resolve();runs=(root/'runs').resolve()
    assert out.parent==runs and out.name.startswith('bridge-full-capability-')
    if a.resume:assert (out/'latest_state.pt').is_file() and not (out/'report.json').exists()
    else:assert not out.exists();out.mkdir(parents=True)
    started=time.monotonic();setup();torch.cuda.reset_peak_memory_stats()
    b=construct(root,admission);attach_data(b)
    import signal
    def stop_requested(signum,frame):b['request_stop']=True
    signal.signal(signal.SIGUSR1,stop_requested);signal.signal(signal.SIGTERM,stop_requested)
    if a.resume:
        extra=resume(b,out/'latest_state.pt')
        with np.load(out/'initial_predictions.npz',allow_pickle=False) as z:
            before=z['before'].copy();targets=z['targets'].copy();masks=z['masks'].copy()
        assert extra['initial_predictions_sha256']==sha(out/'initial_predictions.npz')
    else:
        initial={'identity':identity(b),'empty_adam':not b['optimizer'].state,'pose_port_zero':not bool(b['port'].weight.count_nonzero()),
            'admission':admission,'frozen_restore_events':b['frozen_restore_events'],'source_learned_checkpoints_read':False}
        write(out/'initialization.json',initial)
        before,targets,masks=evaluate(b)
        with np.load(HERE/'evaluation_reference.npz',allow_pickle=False) as reference:
            assert np.array_equal(targets,reference['targets']) and np.array_equal(masks,reference['masks'])
        np.savez_compressed(out/'initial_predictions.npz',before=before,targets=targets,masks=masks)
        guard=zero_guard(b);write(out/'zero_update_guard.json',guard)
        extra={'history':[],'training_seconds':0.,'zero_update_guard':guard,'initial_identity':initial['identity'],
            'initial_predictions_sha256':sha(out/'initial_predictions.npz')}
        save(b,out/'latest_state.pt',extra)
    extra=train(b,out,extra)
    assert b['plan'].completed_updates==65536
    final_id=identity(b);after,t,m=evaluate(b)
    assert np.array_equal(t,targets) and np.array_equal(m,masks) and identity(b)==final_id
    # Pose-only static baseline follows the archived metrics convention. Unknown gripper is not scored as zero.
    static=np.zeros_like(targets);static[...,6]=1
    static=np.repeat(static[None],3,axis=0)
    np.savez_compressed(out/'predictions_and_targets.npz',before=before,after=after,static=static,targets=targets,masks=masks)
    calculated=summarize({'before':before,'after':after,'static':static},targets,masks,read(HERE/'selection.json'))
    policy={'format':'fresh-bridge-capability-policy-v1','contract':b['contract'],'completed_updates':65536,
        'initial_identity':extra['initial_identity'],'final_identity':final_id,'state':__import__('bcap_core').state(b)}
    tmp=out/'final_policy.pt.tmp';torch.save(policy,tmp);tmp.replace(out/'final_policy.pt')
    restored=torch.load(out/'final_policy.pt',map_location='cpu',weights_only=False,mmap=True)
    from completed_policy_export import model_sha
    assert {k:model_sha(v) for k,v in restored['state'].items()}==final_id
    assert restored['completed_updates']==65536 and restored['contract']==b['contract']
    report={'execution_complete':True,'completed_updates':65536,'training_seed':42,'protocol_sha256':sha(HERE/'protocol.json'),
        'initial_identity':extra['initial_identity'],'final_identity':final_id,'metrics':calculated,
        'training_seconds':extra['training_seconds'],'this_process_elapsed_seconds':time.monotonic()-started,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'full_optimizer_retained':True,
        'scope':'One fresh training seed; current finite-pool Bridge only;192reused evaluation windows; no LIBERO or causal mixing verdict',
        'automatic_following_action':False,'main_model_replaced':False}
    write(out/'report.json',report)
    from bcap_verify import verify
    checked=verify(out);write(out/'verification.json',checked)
    import zipfile
    small=['report.json','predictions_and_targets.npz','initial_predictions.npz','initialization.json','zero_update_guard.json','verification.json']
    with zipfile.ZipFile(out/'small_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in small:archive.write(out/name,name)
    print('BRIDGE_FULL_CAPABILITY_RESULT',__import__('json').dumps(checked),flush=True)
    print('BRIDGE_SMALL_RESULTS',out/'small_results.zip',sha(out/'small_results.zip'),flush=True)
    print('BRIDGE_FULL_CAPABILITY_COMPLETE',out/'report.json',flush=True)


if __name__=='__main__':main()
