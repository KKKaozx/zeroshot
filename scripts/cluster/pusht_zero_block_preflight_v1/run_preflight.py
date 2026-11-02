"""Eight fresh zero-block updates, full restore audit and resource timing only."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import numpy as np
import torch
import training_core as core
import zero_state
from native_data import TrainingData,sha,save_json
from token_binding import FrozenCache,project_update
from resource_probe import state_hash
from zero_blocks import remove_blocks,formula_check


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    root,out,bundle=a.root.resolve(),a.output.resolve(),Path(__file__).resolve().parent
    if root/'runs' not in out.parents: parser.error('New output must be under root/runs')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    r=dict(preflight_passed=False,errors=[],formal_training_submitted=False,development_scored=False,
        final50_accessed=False,trained_weights_loaded=False,clip_training_executed=False,
        unique_preflight_updates=0,recomputed_replay_updates=1,isolated_partial_probe_updates=1)
    def timeout(signum,frame): raise TimeoutError('Bounded resource check incomplete; no performance verdict')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(800)
    try:
        protocol=json.loads((bundle/'protocol.json').read_text());protocol_sha=sha(bundle/'protocol.json')
        base=json.loads((bundle/'base_protocol.json').read_text());base_sha=sha(bundle/'base_protocol.json')
        original=json.loads((bundle/'project_contract.json').read_text())
        r['protocol_sha256']=protocol_sha
        for name,digest in protocol['bundle_sources_sha256'].items():
            if sha(bundle/name)!=digest: raise ValueError('Changed bundle '+name)
        if not torch.cuda.is_available() or torch.__version__.split('+')[0]!='1.12.1':
            raise ValueError('Use established official CUDA environment')
        if shutil.disk_usage(out).free<8*1024**3: raise ValueError('Need8GiB free for atomic preflight state; no source deletion')
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        source=root/'paper_reproduction/official_pusht_v1/source'
        for name,digest in base['author_file_sha256'].items():
            if sha(source/name)!=digest: raise ValueError('Changed author source '+name)
        sys.path.insert(0,str(source))
        oldpath=root/'runs/pusht-project-part2-215999/report.json'
        if sha(oldpath)!=protocol['eight_block_report_sha256']: raise ValueError('Reference8 report changed')
        old=json.loads(oldpath.read_text())
        if not (old['training_completed'] and old['development_complete'] and not old['errors']
                and old['completed_updates']==84168 and old['project_contract_sha256']==sha(bundle/'project_contract.json')):
            raise ValueError('Eight-block reference is not the pinned endpoint')
        from omegaconf import OmegaConf
        import hydra
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.lr_scheduler import get_scheduler
        from project_frontend import ProjectObservationEncoder
        data=TrainingData(root,base,out)
        if data.evidence!=old['data']: raise ValueError('Data differ from reference8')
        cache=FrozenCache(root/original['cache_directory'],original,data)
        plan=core.BatchPlan();order_sha=plan.digest(84168)
        plan_sha=hashlib.sha256((order_sha+data.evidence['raw_training_images_sha256']).encode('ascii')).hexdigest()
        if order_sha!=original['sample_order_sha256'] or plan_sha!=original['sample_plan_sha256']:
            raise ValueError('Training sample plan changed')
        torch.manual_seed(42);np.random.seed(42);random.seed(42)
        cfg=OmegaConf.load(bundle/'checkpoint_config.yaml')
        policy=hydra.utils.instantiate(cfg.policy)
        if state_hash(policy.model)!=base['initial_backend_sha256']: raise ValueError('Fresh common backend mismatch')
        # Reconstruct the exact old RNG consumption, THEN remove the blocks.
        torch.manual_seed(43)
        policy.obs_encoder=ProjectObservationEncoder(cache.text,cache.mask,(64,65))
        initial=remove_blocks(policy.obs_encoder)
        normalizer=LinearNormalizer()
        normalizer.fit({'action':data.train_actions,'agent_pos':data.train_positions},last_n_dims=1,mode='limits')
        normalizer['image']=get_image_range_normalizer();policy.set_normalizer(normalizer)
        if state_hash(policy.normalizer)!=base['normalizer_sha256']: raise ValueError('Train normalizer changed')
        policy.cuda().train();averaged=copy.deepcopy(policy).eval()
        ema=hydra.utils.instantiate(cfg.ema,model=averaged)
        optimizer=hydra.utils.instantiate(cfg.optimizer,params=policy.parameters())
        scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=base['lr_horizon_updates'])
        fixed=core.tree_digest((policy.obs_encoder.text_tokens,policy.obs_encoder.text_mask))
        small_idx=plan.batch(0)[:2];smoke=data.batch(small_idx);smoke_tokens=cache.tokens(small_idx).cuda()
        normalized=policy.normalizer['agent_pos'].normalize(smoke['obs']['agent_pos'].cuda()).reshape(4,2)
        with torch.no_grad(): formula_check(policy.obs_encoder,smoke_tokens,normalized)
        r.update(initialization=initial,initial_backend_sha256=base['initial_backend_sha256'],normalizer_sha256=base['normalizer_sha256'],
            sample_plan_sha256=plan_sha,sample_order_sha256=order_sha,data=data.evidence,
            cache_report_sha256=original['cache_report_sha256'],eight_block_report_sha256=protocol['eight_block_report_sha256'],
            fresh_initialization_only=True,zero_block_formula_exact=True,losses=[],durations=[],warm_durations=[])
        current=0
        def snapshot(): return zero_state.capture(policy,averaged,optimizer,scheduler,ema,current,base_sha,plan_sha,protocol_sha)
        def update(index,audit=False):
            ids=plan.batch(index)
            torch.cuda.synchronize();tick=time.monotonic()
            loss,check=project_update(core,policy,optimizer,scheduler,ema,data.batch(ids),cache.tokens(ids),index,audit=audit)
            torch.cuda.synchronize()
            return loss,check,time.monotonic()-tick
        torch.cuda.reset_peak_memory_stats()
        for offset in range(8):
            if offset==2:
                replay=out/'replay.pt';core.atomic_checkpoint(replay,snapshot())
                old_loss,_,_=update(current);current+=1;expected=core.tree_digest(snapshot())
                payload=torch.load(replay,map_location='cpu')
                current=zero_state.restore(payload,policy,averaged,optimizer,scheduler,ema,base_sha,plan_sha,protocol_sha)
                del payload;gc.collect()
                loss,check,duration=update(current);current+=1
                actual=core.tree_digest(snapshot())
                if loss!=old_loss or expected!=actual: raise ValueError('Complete zero-block restore differs')
                r['disk_resume']=dict(exact=True,loss_before=old_loss,loss_after=loss,state_before_sha256=expected,
                    state_after_sha256=actual,replay_checkpoint_sha256=sha(replay),replay_completed_updates=2)
            else:
                loss,check,duration=update(current,audit=(offset==0));current+=1
            if offset==0:
                r['loss_and_gradient_audit']=dict(check,adapter_gradient_scope='Output LayerNorm only; zero active attention/FFN blocks')
            r['losses'].append(loss);r['durations'].append(duration)
            if offset>=3: r['warm_durations'].append(duration)
            r['unique_preflight_updates']=current
            print('ZERO_BLOCK_PREFLIGHT_UPDATE',current,'/8 loss',loss,flush=True)
        r['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        r['projection_updated']=state_hash(policy.obs_encoder.projection)!=initial['common_projection_sha256']
        r['output_norm_updated']=state_hash(policy.obs_encoder.adapter.output_norm)!=initial['common_norm_sha256']
        r['ema_projection_updated']=state_hash(averaged.obs_encoder.projection)!=initial['common_projection_sha256']
        if not all(r[k] for k in ('projection_updated','output_norm_updated','ema_projection_updated')):
            raise ValueError('Remaining frontend/EMA did not learn')
        r['warm_update_seconds']=float(np.mean(r['warm_durations']))
        r['estimated_formal_training_hours_excluding_io_and_rollout']=r['warm_update_seconds']*84168/3600
        # Independent scratch check of the real final38-item batch; not part of training8 state.
        if len(plan.batch(167))!=38: raise ValueError('Expected final partial batch38')
        partial_loss,_,_=update(167)
        if not np.isfinite(partial_loss): raise ValueError('Invalid partial batch')
        r['partial38_probe']=dict(logical_batch=38,microbatch_sizes=[8,8,8,8,6],finite_loss=True,
            loss=partial_loss,disposable=True,not_counted_as_formal_training=True)
        averaged.eval();averaged.obs_encoder.set_tokens(smoke_tokens)
        try:
            with torch.no_grad():
                torch.manual_seed(1001)
                sample=averaged.predict_action({k:v.cuda() for k,v in smoke['obs'].items()})
            if (sample['action'].shape!=(2,8,2) or sample['action_pred'].shape!=(2,16,2)
                    or not torch.isfinite(sample['action_pred']).all() or not torch.equal(sample['action'],sample['action_pred'][:,1:9])):
                raise ValueError('Bad zero-block sampler output or action slice')
            np.savez(out/'smoke_actions.npz',**{k:v.cpu().numpy() for k,v in sample.items()})
        finally: averaged.obs_encoder.cached_tokens=None
        if (core.tree_digest((policy.obs_encoder.text_tokens,policy.obs_encoder.text_mask))!=fixed
                or core.tree_digest((averaged.obs_encoder.text_tokens,averaged.obs_encoder.text_mask))!=fixed):
            raise ValueError('Frozen text changed')
        if any(p.requires_grad or p.grad is not None for p in averaged.parameters()): raise ValueError('EMA not frozen')
        r.update(preflight_passed=True,shared_text_unchanged=True,ema_frozen=True,active_cross_attention_blocks=0,
            smoke_actions_sha256=sha(out/'smoke_actions.npz'),gpu_name=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,ema_optimization_step_after_disposable_probe=ema.optimization_step,
            estimate_scope='Timing only on allocated GPU; excludes I/O, saves, rollout, queue; not an improvement verdict')
    except Exception as exc:
        r['errors'].append(type(exc).__name__+': '+str(exc));raise
    finally:
        signal.alarm(0);r['elapsed_seconds']=time.monotonic()-started
        save_json(out/'report.json',r);print('ZERO_BLOCK_PREFLIGHT_REPORT',out/'report.json',flush=True)
    print('ZERO BLOCK PREFLIGHT PASSED; no task-performance verdict or automatic formal submission',flush=True)


if __name__=='__main__': main()
