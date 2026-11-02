"""Fixed cached source contexts: shared pose-decoder gradients, no optimization."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
OUT=ROOT/'reports/oxe_source_gradient_alignment_20261027/attempt2'
BASE=ROOT/'reports/oxe_source_validation_decode_20261027/attempt1'
CACHE=ROOT/'reports/oxe_conditioning_sampling_audit_20261027/attempt1'
SOURCES=('Bridge','Language Table','BC-Z','Fractal')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mem,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
    if mem>1024 or util>20:raise RuntimeError('Existing GPU workload; no model loaded')
    OUT.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from completed_policy_export import model_sha
    from diffusion_decoder import MultiscaleConditionalUnet1D
    protocol=dict(scope='Shared pose decoder gradient alignment at one final checkpoint; fixed cached dev32, not a causal mixed-training ablation',
        sample_selection_unchanged=True,cached_contexts_sha256=sha(CACHE/'contexts.npy'),
        targets_and_masks_sha256=sha(BASE/'predictions_and_targets.npz'),
        cached_source_protocol_sha256=sha(CACHE/'protocol.json'),
        timesteps=[6,18,31,43,56,68,81,93],noise_seeds=[8401,8402,8403],
        noise_rule='Independent standard Gaussian noise by source using seed+1000*source_index; matched stratified timesteps across sources',
        gradient_scope='Shared diffusion_decoder pose epsilon loss only; detached actual cached context, no Adapter/gripper gradient claim',
        mixture_rule='Four groups of8, pose valid-element-normalized loss; report exact gradient sum weights for these32 windows',
        gradients_are_not_optimizer_steps=True,optimizer_updates=0,sampler_calls=0,
        encoders_reconstructed=False,simulation_steps=0,wall_cap_seconds=600,
        script_sha256=sha(__file__))
    write(OUT/'protocol.json',protocol)
    (OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    result=dict(execution_complete=False,errors=[],optimizer_updates=0,backward_calls=0,
        decoder_forward_calls=0,sampler_calls=0,profiles=[],scope=protocol['scope'])
    started=time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        selection=read(BASE/'selection_protocol.json')
        selected=selection['selected'];assert len(selected)==32
        assert all(r['source']==SOURCES[i//8] for i,r in enumerate(selected))
        old=read(CACHE/'report.json');assert old['execution_complete']
        assert sha(BASE/'predictions_and_targets.npz')==read(CACHE/'protocol.json')['reference_arrays_sha256']
        checkpoint=torch.load(ROOT/selection['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        learned=checkpoint['trainable_state_dict']
        assert model_sha(learned)==checkpoint['learned_state_sha256']==old['initial_learned_sha256']
        config=checkpoint['config']['model'];contract=checkpoint['training_contract']
        assert config['diffusion_prediction_type']=='epsilon' and config['separate_gripper_head']
        assert config['diffusion_architecture']=='multiscale' and config['num_diffusion_steps']==100
        for name in ('diffusion_decoder.py','models.py'):
            assert sha(ROOT/'code'/name)==contract['source_code_sha256'][name]
        saved=np.load(BASE/'predictions_and_targets.npz')
        contexts=np.load(CACHE/'contexts.npy');assert contexts.shape==(32,1024)
        decoder=MultiscaleConditionalUnet1D(action_dim=7,chunk_size=16,context_dim=1024,
            down_dims=tuple(config.get('diffusion_down_dims',(256,512,1024))),
            diffusion_step_embed_dim=int(config.get('diffusion_step_embed_dim',128)),
            kernel_size=int(config.get('diffusion_kernel_size',5)),num_steps=100)
        state={k.removeprefix('diffusion_decoder.'):v for k,v in learned.items() if k.startswith('diffusion_decoder.')}
        decoder.load_state_dict(state,strict=True)
        initial=model_sha(decoder.state_dict());assert initial==model_sha(state)
        del learned,state,checkpoint
        decoder.cuda().eval()
        assert not any(isinstance(m,(torch.nn.Dropout,torch.nn.modules.batchnorm._BatchNorm)) for m in decoder.modules())
        profile=read(ROOT/'reports/oxe_noaug_seed42_227223/cluster/preflight.json')
        alpha=torch.tensor(profile['schedule_values']['alpha_bars'],dtype=torch.float32)
        assert hashlib.sha256(alpha.numpy().tobytes()).hexdigest()==profile['tensors']['alpha_bars']['sha256']
        alpha=alpha.cuda()
        ctx=torch.from_numpy(contexts).cuda()
        clean=torch.from_numpy(saved['targets'][...,:7].copy()).cuda()
        masks=torch.from_numpy(saved['masks'][...,:7].copy()).cuda()
        # One saved scalar replay validates the direct decoder reconstruction.
        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
            torch.cuda.manual_seed(4126);eps=torch.randn_like(clean[:1]);a=alpha[25]
            pred=decoder(a.sqrt()*clean[:1]+(1-a).sqrt()*eps,torch.tensor([25],device='cuda'),ctx[:1])
            mse=float(((pred-eps).square()*masks[:1]).sum()/masks[:1].sum())
        reference=next(p for p in read(CACHE/'window_results.json')[0]['noise_probes'] if p['t']==25)['epsilon_correct_mse']
        result['decoder_forward_calls']=1;result['saved_scalar_replay_difference']=abs(mse-reference)
        assert abs(mse-reference)<1e-6,(mse,reference)
        params=list(decoder.named_parameters());parameters=[p for _,p in params]
        segments=[];offset=0
        for name,p in params:
            segments.append((name.split('.')[0],offset,offset+p.numel()));offset+=p.numel()
        weights=[float(masks[8*k:8*k+8].sum()/masks.sum()) for k in range(4)]
        result.update(decoder_parameters=offset,source_order=list(SOURCES),combined32_pose_weights=weights,
            loaded_complete_learned_sha256=old['initial_learned_sha256'],initial_decoder_sha256=initial,
            training_mode_equivalence='Decoder contains neither Dropout nor BatchNorm; cached context fixed, so decoder train/eval computation is identical')
        def matrices(vectors):
            gram={name:torch.zeros((4,4),dtype=torch.float64) for name,_,_ in segments}
            for group,lo,hi in segments:
                for start in range(lo,hi,262144):
                    tile=torch.stack([v[start:min(start+262144,hi)] for v in vectors]).double()
                    gram[group]+=tile@tile.T
            gram['all']=sum(gram.values())
            out={}
            w=torch.tensor(weights,dtype=torch.float64)
            for name,g in gram.items():
                norms=g.diag().clamp_min(0).sqrt();assert bool((norms>0).all())
                cosine=g/(norms[:,None]*norms[None,:])
                others=w.clone();others[0]=0
                def align(c):return float((g[0]@c)/(g[0,0].sqrt()*(c@g@c).clamp_min(1e-30).sqrt()))
                out[name]=dict(gram=g.tolist(),cosine=cosine.tolist(),gradient_norms=norms.tolist(),
                    bridge_vs_combined_cosine=align(w),bridge_vs_other_sources_cosine=align(others))
            return out
        means=[None]*4;torch.cuda.reset_peak_memory_stats();probe_started=time.monotonic()
        for seed in protocol['noise_seeds']:
            vectors=[];losses=[]
            for k,source in enumerate(SOURCES):
                if time.monotonic()-probe_started>600:raise RuntimeError('Fixed probe budget reached')
                ids=slice(8*k,8*k+8)
                g=torch.Generator().manual_seed(seed+1000*k)
                eps=torch.randn((8,16,7),generator=g).cuda()
                steps=torch.tensor(protocol['timesteps'],device='cuda',dtype=torch.long)
                a=alpha[steps,None,None]
                pred=decoder(a.sqrt()*clean[ids]+(1-a).sqrt()*eps,steps,ctx[ids])
                loss=((pred-eps).square()*masks[ids]).sum()/masks[ids].sum()
                gradients=torch.autograd.grad(loss,parameters)
                assert all(torch.isfinite(p).all() for p in gradients)
                vector=torch.cat([p.detach().flatten().cpu() for p in gradients])
                vectors.append(vector)
                if means[k] is None:means[k]=vector.clone()
                else:means[k].add_(vector)
                losses.append(float(loss.detach()))
                result['backward_calls']+=1;result['decoder_forward_calls']+=1
                del gradients,loss,pred
            matrix=matrices(vectors)
            result['profiles'].append(dict(noise_seed=seed,pose_losses=losses,modules=matrix))
            write(OUT/'report.json',result)
            print('GRADIENT_PROFILE',seed,'bridge_cosines',matrix['all']['cosine'][0],
                'bridge_vs_others',matrix['all']['bridge_vs_other_sources_cosine'],flush=True)
            del vectors
        for v in means:v.div_(len(protocol['noise_seeds']))
        result['noise_averaged_gradient_modules']=matrices(means)
        del means
        torch.cuda.synchronize()
        final=model_sha(decoder.state_dict());assert initial==final
        assert all(p.grad is None for p in parameters)
        result.update(execution_complete=True,final_decoder_sha256=final,decoder_weights_unchanged=True,
            diagnostic_seconds=time.monotonic()-probe_started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            scope_limits=['Only shared pose decoder; cached contexts detach Adapter and gripper',
                'Fixed dev8 per source and one final checkpoint, not gradient history during training',
                'Negative cosine is local gradient opposition, not causal evidence of long-run negative transfer',
                'Gradient sum first-order interpretation is not an AdamW update prediction'])
    except Exception as error:
        result['errors'].append(type(error).__name__+': '+str(error));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        write(OUT/'report.json',result)
        print('SOURCE_GRADIENT_ALIGNMENT_REPORT',str(OUT/'report.json'),flush=True)


if __name__=='__main__':main()
