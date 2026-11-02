"""Reviewable warm-start loss repair, one paired job, no automatic submission."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'code'), str(HERE / 'vendor/code'), str(HERE / 'vendor')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def admit(root):
    """Local files/state identity only; no model, rollout, or old scoring replay."""
    import numpy as np
    import torch
    from bcap_runtime import sha
    from completed_policy_export import model_sha
    p = read(HERE / 'pair_protocol.json')
    for line in (HERE / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ',1)
        if sha(HERE / name) != expected:
            raise ValueError('Package identity differs: ' + name)
    source = root / p['source_checkpoint']
    if not source.is_file() or sha(source) != p['source_checkpoint_sha256']:
        raise ValueError('Existing full234100 checkpoint absent or changed')
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    report = read(HERE / 'source_report.json')
    if (saved['format'] != 'fresh-bridge-capability-policy-v1' or saved['completed_updates'] != 65536
            or saved['final_identity'] != report['final_identity']
            or {k:model_sha(v) for k,v in saved['state'].items()} != report['final_identity']
            or saved['contract']['protocol_sha256'] != report['protocol_sha256']
            or saved['contract']['stats'] != read(HERE / 'vendor/state_stats.json')):
        raise ValueError('Incomplete or mismatched source base/pose-port state')
    plans = np.load(HERE / 'repair_batches.npy', allow_pickle=False)
    ids = np.load(HERE / 'vendor/train_indices.npy', allow_pickle=False)
    if plans.shape != (8192,8) or plans.dtype != np.int64 or not np.isin(plans,ids).all():
        raise ValueError('Expected fixed8192x8 plan from the unchanged full Bridge subpool')
    for role in ('train','validation'):
        if not (root / f'oxe_dino_main_preflight_v1/inputs/{role}/manifest.json').is_file():
            raise ValueError('Existing nominal input artifacts missing')
    if torch.cuda.is_initialized():
        raise RuntimeError('Admission unexpectedly initialized CUDA')
    return {'passed':True,'source_learned_state_identity_verified':True,'cuda_initialized':False,
            'model_constructed':False,'optimizer_updates':0,'paired_updates_per_arm':8192}


def run_arm(root, out, mode):
    import numpy as np
    import torch
    import bcap_core
    from bcap_runtime import setup, construct, attach_data, evaluate, sha
    from step_training_state import UpdateSchedule
    from bridge_timestep_repair import backward
    from completed_policy_export import model_sha
    p = read(HERE / 'pair_protocol.json')
    # Exact archived construction/frozen buffers; all learned values overwritten.
    setup()
    b = construct(root, {'plan_sha256':sha(HERE / 'vendor/batches.npy')})
    saved = torch.load(root / p['source_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    bcap_core.load_state(b, saved['state'])
    if bcap_core.identity(b) != read(HERE / 'source_report.json')['final_identity']:
        raise ValueError('Source weights incompletely restored')
    attach_data(b)
    b['plan'] = bcap_core.FixedPlan(np.load(HERE / 'repair_batches.npy', allow_pickle=False), sha(HERE / 'repair_batches.npy'))
    b['schedule'] = UpdateSchedule(total_updates=8192, base_lrs=[1e-5]*3, kind='constant', warmup_updates=0)
    b['schedule'].apply(b['optimizer'], 0)
    if b['optimizer'].state:
        raise ValueError('Weights-only fork requires empty Adam in BOTH arms')
    b['repair_mode'] = mode
    b['training_cap_seconds'] = 3000
    b['checkpoint_every'] = 2048
    b['contract'] = {'format':'bridge-timestep-repair-paired-arm-v1', 'pair_protocol':p,
        'mode':mode,'source_contract':saved['contract'],'source_identity':saved['final_identity'],
        'optimizer_initialization':'new_empty_adam_for_both_arms_not_source_optimizer_resume'}
    #Same sampled batches, dropout, native noise/t, and generation seeds in both arms.
    b['api'].set_seed(p['paired_training_seed'])
    bcap_core.backward = backward
    initial = bcap_core.identity(b)
    guard = bcap_core.zero_guard(b)
    write(out / 'zero_update_guard.json', guard)
    extra = {'history':[], 'training_seconds':0., 'zero_update_guard':guard, 'initial_identity':initial}
    def stop_requested(signum, frame):
        b['request_stop'] = True
    signal.signal(signal.SIGUSR1, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    extra = bcap_core.train(b, out, extra)
    final_identity = bcap_core.identity(b)
    after, targets, masks = evaluate(b)
    with np.load(HERE / 'source_arrays.npz', allow_pickle=False) as old:
        if not np.array_equal(targets,old['targets']) or not np.array_equal(masks,old['masks']):
            raise ValueError('Evaluation targets/masks changed from source evidence')
        before, static = old['after'].copy(), old['static'].copy()
    np.savez_compressed(out / 'predictions_and_targets.npz',after=after,targets=targets,masks=masks)
    calculated = bcap_core.summarize({'before':before,'after':after,'static':static}, targets,masks,read(HERE / 'vendor/selection.json'))
    payload = {'format':'bridge-timestep-repair-policy-v1','contract':b['contract'],
        'source_completed_updates':65536,'additional_updates':8192,'mode':mode,
        'initial_identity':initial,'final_identity':final_identity,'state':bcap_core.state(b)}
    temporary = out / 'final_policy.pt.tmp'
    torch.save(payload,temporary); temporary.replace(out / 'final_policy.pt')
    restored = torch.load(out / 'final_policy.pt',map_location='cpu',weights_only=False,mmap=True)
    if {k:model_sha(v) for k,v in restored['state'].items()} != final_identity:
        raise ValueError('Endpoint learned state readback differs')
    write(out / 'history.json',extra['history'])
    write(out / 'report.json',{'execution_complete':True,'mode':mode,'additional_updates':8192,
        'initial_identity':initial,'final_identity':final_identity,'metrics':calculated,
        'training_seconds':extra['training_seconds'],'train_and_evaluation_seconds':time.monotonic()-started,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'old_policy_rescored':False,'source_arrays_reused':True,'automatic_following_action':False})


def summarize_pair(out):
    import numpy as np
    from bridge_metrics import KEYS
    from bridge_timestep_repair import check_matched_random_stream
    names = ('uniform','min_snr5_grid_mean_one')
    arms = {name:read(out / name / 'report.json') for name in names}
    stream = check_matched_random_stream(*(read(out / name / 'history.json') for name in names))
    old = read(HERE / 'source_report.json')
    roles = {}; source_protection = True
    for role in ('train','validation'):
        control,candidate = [arms[name]['metrics'][role] for name in names]
        a,b = control['metrics']['after'],candidate['metrics']['after']
        n = len(a['per_window']); bootstrap = np.random.default_rng(20261030).integers(n,size=(10000,n))
        ci = {}
        for key in KEYS:
            delta = np.asarray([y[key]-x[key] for x,y in zip(a['per_window'],b['per_window'])])
            ci[key] = np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist()
        protection = all(b[k] <= old['metrics'][role]['metrics']['after'][k]*1.05
                         for k in ('path_position_normalized','endpoint_position_normalized'))
        source_protection &= protection
        roles[role] = {'candidate_basic_capability_passed':candidate['basic_capability_passed'],
            'candidate_minus_control_bootstrap95':ci,'source_position_within5percent':protection,
            'candidate_rotation_better_control_both_means':all(b[k]<a[k] for k in ('path_rotation_deg','endpoint_rotation_deg'))}
    accepted = (all(v['candidate_basic_capability_passed'] and v['candidate_rotation_better_control_both_means'] for v in roles.values())
        and roles['validation']['candidate_minus_control_bootstrap95']['path_rotation_deg'][1] < 0 and source_protection)
    result = {'execution_complete':True,'paired_updates':8192,'total_optimizer_updates':16384,
        'random_stream_check':stream,'roles':roles,'candidate_admitted_for_further_source_work':bool(accepted),
        'libero_or_main_matrix_admitted':False,'automatic_following_action':False,
        'scope':'One warm-start paired loss intervention; not fresh13-model results or a unique-cause proof'}
    write(out / 'pair_report.json',result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--arm',choices=('uniform','min_snr5_grid_mean_one'))
    args = ap.parse_args(); root = args.root.resolve()
    if args.arm:
        if not os.environ.get('SLURM_JOB_ID') or args.output is None:
            raise ValueError('Actual training only inside the manually submitted paired job')
        run_arm(root,args.output.resolve(),args.arm)
        return
    admission = admit(root)
    if args.check_only:
        print('BRIDGE_TIMESTEP_REPAIR_ADMISSION',json.dumps(admission),flush=True)
        return
    if not os.environ.get('SLURM_JOB_ID') or args.output is None:
        raise ValueError('No training without explicit manual job submission')
    out = args.output.resolve()
    if out.parent != (root / 'runs').resolve() or not out.name.startswith('bridge-timestep-repair-'):
        raise ValueError('A fresh dedicated run folder under project/runs is required')
    import shutil
    used = int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
    if used + 8*1024**3 >= 200_000_000_000 or shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError('Need8GiB state/atomic-save reserve; no automatic cleanup; du is only a quota estimate')
    out.mkdir(exist_ok=False)
    write(out / 'admission.json',admission)
    started = time.monotonic(); active = None
    def forward_stop(signum,frame):
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1,forward_stop); signal.signal(signal.SIGTERM,forward_stop)
    def hard_timeout(signum, frame):
        raise TimeoutError('Paired6900s process cap reached; no automatic continuation')
    signal.signal(signal.SIGALRM,hard_timeout)
    signal.alarm(6900)
    try:
        for name in ('uniform','min_snr5_grid_mean_one'):
            folder = out / name; folder.mkdir()
            remaining = 6900-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Paired job budget ended; no further arm or continuation')
            active = subprocess.Popen([sys.executable,'-B','-u',str(HERE / 'run_pair.py'),'--root',str(root),
                '--output',str(folder),'--arm',name])
            try:
                status = active.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                active.kill(); active.wait()
                raise TimeoutError('Paired6900s process cap reached; no automatic retry')
            if status:
                raise RuntimeError('Arm stopped before a valid endpoint: ' + name)
        result = summarize_pair(out)
        from verify_results import verify
        write(out / 'verification.json',verify(out))
        import zipfile
        small = ['pair_report.json','admission.json','verification.json']
        for name in ('uniform','min_snr5_grid_mean_one'):
            small += [name + '/' + f for f in ('report.json','predictions_and_targets.npz','zero_update_guard.json','history.json')]
        with zipfile.ZipFile(out / 'small_results.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in small: z.write(out / name,name)
        print('BRIDGE_TIMESTEP_REPAIR_COMPLETE',json.dumps(result),flush=True)
    except BaseException as e:
        if active is not None and active.poll() is None:
            active.kill(); active.wait()
        write(out / 'stopped.json',{'error':type(e).__name__+': '+str(e),'automatic_retry':False,
                                  'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__': main()
