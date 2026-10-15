"""Bounded same-Bridge author reference: original Workspace, fixed final EMA primary."""
import argparse
from pathlib import Path
import sys
import time
import json
import hashlib
import numpy as np
from reference_metrics import evaluate_cached,budget_estimate

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('project','source','output'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();code=Path(__file__).resolve().parent;p=read(code/'training_protocol.json')
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    r=dict(stage='same_bridge_author_reference',execution_passed=False,training_protocol_sha256=sha(code/'training_protocol.json'),
           trained_updates=0,simulation_executed=False,reserved_test_used=False,checkpoint_selection='fixed final; EMA primary, raw secondary',
           source_sha256={n:sha(code/n) for n in ('run_bridge_author_reference.py','reference_metrics.py','training_config.yaml')})
    def persist():
        r['elapsed_seconds']=time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    try:
        precode=args.project/'bridge_author_preflight_v1'
        pre=args.project/'runs/bridge-author-preflight-203498'
        assert sha(pre/'author/report.json')==p['preflight_report_sha256'] and read(pre/'author/report.json')['passed']
        assert sha(pre/'inputs/export_report.json')==p['input_export_report_sha256']
        assert sha(precode/'protocol.json')==p['input_protocol_sha256']
        assert all(sha(precode/n)==h for n,h in p['reused_adapter_source_sha256'].items())
        author=read(precode/'protocol.json')
        assert all(sha(args.source/n)==h for n,h in author['author_source_sha256'].items())
        sys.path.insert(0,str(args.source));sys.path.insert(1,str(precode))
        import torch
        from omegaconf import OmegaConf
        from bridge_author_dataset import BridgeAuthorDataset
        from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
        assert torch.cuda.is_available()
        cfg=OmegaConf.load(code/'training_config.yaml')
        cfg.task.dataset.cache=str(pre/'inputs');cfg.task.dataset.protocol=str(precode/'protocol.json')
        OmegaConf.save(cfg,args.output/'runtime_config.yaml')
        data=BridgeAuthorDataset(pre/'inputs',precode/'protocol.json');normalizer=data.get_normalizer()
        assert len(data)==128 and len(data.get_validation_dataset())==4
        (args.output/'workspace').mkdir()
        ws=TrainDiffusionUnetHybridWorkspace(cfg,output_dir=str(args.output/'workspace'))
        ws.model.set_normalizer(normalizer);ws.ema_model.set_normalizer(normalizer)
        expected_pre=read(pre/'author/report.json')
        r.update(torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),parameters=sum(v.numel() for v in ws.model.parameters()),
                 author_commit=author['author_commit'],raw_input_report_sha256=sha(pre/'inputs/export_report.json'),
                 training_statistics_only=True,validation_loss_is_four_training_rows=True,primary='ema')
        assert r['parameters']==expected_pre['parameters']
        original_loss=ws.model.compute_loss;starts=[];training_losses=[]
        with (args.output/'updates.jsonl').open('w',encoding='utf-8') as log:
            def observed_loss(batch):
                #128/8 exact full batches; only the separate4-row training diagnostic has B=4.
                if int(batch['action'].shape[0])!=8:return original_loss(batch)
                before=[torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()]
                if len(starts)<8:
                    torch.cuda.synchronize();starts.append(time.monotonic())
                    if len(starts)==8:
                        measured=float(np.median(np.diff(starts)[2:]))
                        estimate=budget_estimate(measured,expected_pre['workspace_seconds'])
                        r['resource_gate']=dict(steady_update_seconds=measured,estimated_total_seconds=estimate,
                            limit_seconds=10800,passed=estimate<10200,first_updates_used_in_final_run=True)
                        persist()
                        assert estimate<10200,'Measured planning estimate exceeds bounded job budget; stop without more updates'
                        print('AUTHOR_RESOURCE_GATE',json.dumps(r['resource_gate']),flush=True)
                assert torch.equal(before[0],torch.get_rng_state()) and torch.equal(before[1],torch.cuda.get_rng_state())
                loss=original_loss(batch);value=float(loss.detach())
                assert np.isfinite(value);training_losses.append(value)
                update=len(training_losses)
                log.write(json.dumps(dict(update=update,epoch=int(ws.epoch),loss=value,learning_rate=ws.optimizer.param_groups[0]['lr']))+'\n')
                if update==1 or update%256==0:
                    log.flush();print('AUTHOR_UPDATE',update,'/8192 loss',value,'elapsed',round(time.monotonic()-start,1),flush=True)
                return loss
            ws.model.compute_loss=observed_loss # Observer delegates to the unchanged author loss; no mathematical update change.
            torch.cuda.reset_peak_memory_stats();ws.run();torch.cuda.synchronize()
        #Remove observer before serialization; do not pickle closures or the optimizer.
        del ws.model.compute_loss
        steps={int(s['step']) for s in ws.optimizer.state.values() if 'step' in s}
        assert steps=={8192} and len(training_losses)==8192 and ws.epoch==512
        r['trained_updates']=8192
        pre_logs=[json.loads(s) for s in (pre/'author/workspace/logs.json.txt').read_text().splitlines()][:4]
        r['prefix_loss_max_difference_from_preflight']=max(abs(a-b['train_loss']) for a,b in zip(training_losses[:4],pre_logs))
        assert not list(args.output.rglob('*.ckpt'))
        for name,policy in [('raw',ws.model),('ema',ws.ema_model)]:
            policy.eval()
            for k,v in normalizer.state_dict().items():assert torch.equal(v,policy.normalizer.state_dict()[k].cpu())
            payload=dict(stage='same_bridge_author_reference_final',policy_state={k:v.detach().cpu() for k,v in policy.state_dict().items()},
                         policy_config=OmegaConf.to_container(cfg.policy,resolve=True),training_protocol=p,
                         initialization_seed=43,updates=8192,primary=name=='ema')
            file=args.output/(name+'_final.pt');torch.save(payload,file)
            r.setdefault('checkpoints',[]).append(dict(name=file.name,bytes=file.stat().st_size,sha256=sha(file),includes_optimizer=False))
            assert sum(x['bytes'] for x in r['checkpoints'])<3*2**30
            for split,indices in [('train',np.arange(128)),('development',np.arange(128,168))]:
                metric,pred,target=evaluate_cached(policy,data.arrays,indices,p['rows'],p['sampling_seeds'])
                np.savez_compressed(args.output/(name+'_'+split+'.npz'),predictions=pred,targets=target,row_indices=indices)
                r.setdefault('metrics',{}).setdefault(name,{})[split]=metric
                print('AUTHOR_FINAL',name,split,json.dumps({k:metric[k] for k in ('path_position_cm','path_rotation_deg','passing_windows','pose_fit_passed')}),flush=True)
                persist()
        r.update(execution_passed=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                 scientific_verdict='Fixed final reference; independent physical-metric analysis required, no causal module verdict.')
        assert r['peak_reserved_gib']<40
    except Exception as e:
        r['error']=repr(e)
        if 'ws' in locals():r['trained_updates']=max([int(s['step']) for s in ws.optimizer.state.values() if 'step' in s]+[0])
        raise
    finally:persist();print('AUTHOR REFERENCE REPORT',args.output/'report.json',flush=True)
if __name__=='__main__':main()
