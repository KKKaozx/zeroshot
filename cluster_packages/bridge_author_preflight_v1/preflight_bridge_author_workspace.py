"""Four actual updates through the author's unmodified workspace; no full training."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
from bridge_author_io import read,sha

def digest(module):
    h=hashlib.sha256()
    for n,p in module.named_parameters():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','cache','output'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();code=Path(__file__).resolve().parent;p=read(code/'protocol.json')
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    r=dict(stage='author_bridge_workspace_resource_preflight',passed=False,full_training=False,
           trained_updates=0,simulation_executed=False,development_policy_evaluated=False,
           protocol_sha256=sha(code/'protocol.json'),packages_installed=False,weights_downloaded=False)
    def persist():
        r['elapsed_seconds']=time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    try:
        assert all(sha(args.source/n)==h for n,h in p['author_source_sha256'].items())
        sys.path.insert(0,str(args.source))
        import torch
        import hydra
        from omegaconf import OmegaConf
        from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
        from bridge_author_dataset import BridgeAuthorDataset
        assert torch.cuda.is_available()
        cfg=OmegaConf.load(code/'preflight_config.yaml')
        cfg.task.dataset.cache=str(args.cache);cfg.task.dataset.protocol=str(code/'protocol.json')
        OmegaConf.save(cfg,args.output/'runtime_config.yaml')
        data=BridgeAuthorDataset(args.cache,code/'protocol.json');normalizer=data.get_normalizer()
        truth=data.get_all_actions();restored=normalizer['action'].unnormalize(normalizer['action'].normalize(truth))
        delta=float((truth-restored).abs().max());assert delta<2e-6
        stats=normalizer.get_input_stats()['action']
        flat=truth.reshape(-1,7)
        assert torch.equal(stats['min'],flat.min(0).values) and torch.equal(stats['max'],flat.max(0).values)
        for key in ('min','max'):assert np.allclose(stats[key].cpu().numpy(),data.report['training_action_'+key],atol=1e-6,rtol=0)
        (args.output/'workspace').mkdir()
        ws=TrainDiffusionUnetHybridWorkspace(cfg,output_dir=str(args.output/'workspace'))
        assert isinstance(ws.model,hydra.utils.get_class(cfg.policy._target_))
        assert ws.model.horizon==16 and ws.model.n_obs_steps==1 and ws.model.action_dim==7
        assert ws.model.noise_scheduler.config.prediction_type=='epsilon'
        before={k:digest(m) for k,m in {'encoder':ws.model.obs_encoder,'unet':ws.model.model,'ema_unet':ws.ema_model.model}.items()}
        gradients={k:[] for k in ('encoder','unet')};handles=[]
        for key,mod in [('encoder',ws.model.obs_encoder),('unet',ws.model.model)]:
            param=next(x for x in mod.parameters() if x.requires_grad)
            def capture(g,key=key):
                value=float(torch.linalg.vector_norm(g.detach()));assert np.isfinite(value)
                gradients[key].append(value)
            handles.append(param.register_hook(capture))
        torch.cuda.reset_peak_memory_stats();training_start=time.monotonic()
        ws.run() # Original training loop, AdamW, scheduler, EMA, compute_loss and train-batch sampling.
        torch.cuda.synchronize();workspace_seconds=time.monotonic()-training_start
        for h in handles:h.remove()
        steps={int(s['step']) for s in ws.optimizer.state.values() if 'step' in s}
        assert steps=={4};r['trained_updates']=4
        assert all(len(v)==4 and max(v)>0 for v in gradients.values())
        after={k:digest(m) for k,m in {'encoder':ws.model.obs_encoder,'unet':ws.model.model,'ema_unet':ws.ema_model.model}.items()}
        changed={k:before[k]!=after[k] for k in before};assert all(changed.values())
        batch=torch.utils.data.default_collate([data[0],data[1]])
        batch={k:{n:v.cuda() for n,v in x.items()} if isinstance(x,dict) else x.cuda() for k,x in batch.items()}
        outputs={}
        for name,policy in [('raw',ws.model),('ema',ws.ema_model)]:
            policy.eval();torch.manual_seed(1101);torch.cuda.manual_seed_all(1101)
            with torch.no_grad():outputs[name]=policy.predict_action(batch['obs'])['action_pred'].cpu().numpy()
            assert outputs[name].shape==(2,16,7) and np.isfinite(outputs[name]).all()
        np.savez_compressed(args.output/'train_contract_predictions.npz',targets=batch['action'].cpu().numpy(),**outputs)
        assert not list(args.output.rglob('*.ckpt')) and not list(args.output.rglob('*.pt'))
        normalizer_info={}
        for key in ('action','ee_state'):
            normalizer_info[key]={k:v.detach().cpu().numpy().tolist() for k,v in normalizer[key].params_dict.items() if k in ('scale','offset')}
        (args.output/'normalizer.json').write_text(json.dumps(normalizer_info,indent=2)+'\n',encoding='utf-8')
        peak=torch.cuda.max_memory_reserved()/2**30;assert peak<40
        r.update(passed=True,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),author_commit=p['author_commit'],
            author_source_verified=True,input_export_report_sha256=sha(args.cache/'export_report.json'),
            action_normalize_roundtrip_max_difference=delta,training_statistics_only=True,first_parameter_gradients=gradients,
            parameters_changed=changed,parameters=sum(x.numel() for x in ws.model.parameters()),
            workspace_seconds=workspace_seconds,peak_reserved_gib=peak,
            sampling_steps=100,action_prediction_shape=[2,16,7],checkpoint_saved=False,
            scientific_verdict='Execution and data-interface readiness only; not fitting or generalization success.')
    except Exception as e:r['error']=repr(e);raise
    finally:persist();print('AUTHOR BRIDGE WORKSPACE PREFLIGHT',r['passed'],args.output/'report.json',flush=True)
if __name__=='__main__':main()
