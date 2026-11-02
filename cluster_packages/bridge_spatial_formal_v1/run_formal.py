"""Two fixed4096 continuations, then one development comparison; frozenCLIP."""
import os
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from bridge_spatial_pair_metrics_v1 import summarize_pair,admission
from bridge_spatial_validation_v1 import load_validation

CODE=Path(__file__).resolve().parent


def check_sums(root,sha):
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);assert sha(root/name)==expected,(root,name)


def snapshot(model,original_keys,readout,P):
    return dict(base={n:v.detach().cpu().clone() for n,v in model.named_parameters() if n in original_keys},
                readout=P.cpu_copy(readout.state_dict()))


def restore(model,original_keys,readout,values):
    assert set(values)=={'base','readout'} and set(values['base'])==original_keys
    assert not model.load_state_dict(values['base'],strict=False).unexpected_keys
    readout.load_state_dict(values['readout'],strict=True)


def run_arm(model,readout,payload,original_keys,context,targets,current,batches,arm,output,P,
            reference_losses,updates=4096):
    candidate=arm=='candidate'
    assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
    P.reset_readout(readout,candidate)
    initial_base=P.fingerprint(snapshot(model,original_keys,readout,P)['base'])
    trainables=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.Adam(trainables,lr=1e-5,weight_decay=0.)
    assert not optimizer.state
    losses=[];seconds=[];score_gradient=None
    device=next(model.adapter.parameters()).device
    with (output/f'{arm}_updates.jsonl').open('x') as log:
        for update in range(1,updates+1):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            def calculate(update):
                ids=torch.from_numpy(batches[update-1]).long();model.train()
                global_update=8192+update
                torch.manual_seed(430000+global_update);torch.cuda.manual_seed_all(430000+global_update)
                ctx=context(ids)
                torch.manual_seed(420000+global_update);torch.cuda.manual_seed_all(420000+global_update)
                guess,truth,_=model.diffusion_loss(targets[ids],ctx,current[ids])
                return torch.nn.functional.mse_loss(guess,truth),guess
            loss,pred=calculate(update);assert torch.isfinite(loss)
            loss.backward()
            assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
            assert all(torch.isfinite(v.grad).all() for v in trainables if v.grad is not None)
            if candidate and update==1:
                score_gradient={n:float(v.grad.detach().norm()) for n,v in readout.named_parameters()}
                assert all(v>0 for v in score_gradient.values())
            torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
            optimizer.step()
            if device.type=='cuda':torch.cuda.synchronize()
            value=float(loss.detach());losses.append(value);seconds.append(time.monotonic()-tick)
            if update<=8 and reference_losses is not None:
                assert abs(value-reference_losses[update-1])<=1e-6,(arm,update,'preflight loss',value,reference_losses[update-1])
            log.write(json.dumps(dict(update=update,loss=value,learning_rate=1e-5))+'\n')
            if update<=2 or update%256==0:
                log.flush();print('SPATIAL_FORMAL_UPDATE',arm,update,'/',updates,'loss',value,flush=True)
            del loss,pred
    values=snapshot(model,original_keys,readout,P)
    assert P.fingerprint(values['base'])!=initial_base
    score_changed=any(torch.count_nonzero(v)>0 for v in readout.parameters())
    assert bool(score_changed)==candidate
    state_sha=P.fingerprint(values)
    checkpoint=output/f'{arm}_final.pt'
    assert not checkpoint.exists()
    torch.save(dict(format='bridge_spatial_readout_pose_v1',arm=arm,model=values,
        config=payload['config'],source_checkpoint_sha256=payload['source_checkpoint_sha256'],
        logical_optimizer_updates=updates),checkpoint)
    del values
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['format']=='bridge_spatial_readout_pose_v1' and P.fingerprint(saved['model'])==state_sha
    restore(model,original_keys,readout,saved['model'])
    assert P.fingerprint(snapshot(model,original_keys,readout,P))==state_sha
    del saved
    np.savez_compressed(output/f'{arm}_losses.npz',losses=np.asarray(losses))
    return dict(arm=arm,logical_optimizer_updates=updates,initial_base_sha256=initial_base,
        initial_optimizer_empty=True,trainable_parameters=sum(v.numel() for v in trainables),
        first8_losses=losses[:8],new_score_first_gradient_norms=score_gradient,
        new_score_parameters_changed=candidate,base_parameters_changed=True,
        checkpoint=checkpoint.name,checkpoint_sha256=P.sha(checkpoint),checkpoint_roundtrip_exact=True,
        final_state_sha256=state_sha,warm_update_seconds=float(np.mean(seconds[2:])))


def evaluate(model,validation,P):
    values=[];model.eval()
    # Match209481: compute contexts in no_grad (not inference_mode), batch2.
    with torch.no_grad():
        contexts=torch.cat([P.context_from_tokens(model,validation['visual'][i:i+2].cuda(),
            validation['language'][i:i+2].cuda(),validation['attention'][i:i+2].cuda(),
            validation['state'][i:i+2]) for i in range(0,50,2)])
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
        for seed in (1101,1102,1103):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rows=[]
            for start in range(0,50,2):
                prediction=model.sample(contexts[start:start+2],validation['current'][start:start+2])
                assert torch.isfinite(prediction).all()
                rows.append(prediction[:,:,:7].cpu().numpy())
            values.append(np.concatenate(rows))
            print('SPATIAL_DEV_SAMPLE',seed,'50/50',flush=True)
    return np.stack(values)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    pre=args.project/'bridge_spatial_readout_v1';prefix=args.project/'bridge_prefix_pair_v1'
    sys.path[:0]=[str(pre),str(pre/'frozen')]
    spec=importlib.util.spec_from_file_location('spatial_preflight_dependency',pre/'run_preflight.py')
    P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
    args.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    r=dict(stage='spatial_formal_pair',execution_checks_passed=False,generalization_verdict_available=False,
        preflight_job=210291,training_runs=[],final25_payloads_read=False,gripper_trained=False,
        old40_or_consumed13_rescored=False,downloads=False,weights_selected_by_development=False,
        both_weights_frozen_before_development_scoring=False,CLIP_training_allowed=False)
    def persist():
        r['elapsed_seconds']=time.monotonic()-begin
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        check_sums(CODE,P.sha);check_sums(pre,P.sha);check_sums(prefix,P.sha)
        contract=P.read(CODE/'formal_contract.json');p=P.read(pre/'protocol.json')
        assert P.sha(pre/'protocol.json')==contract['protocol_sha256']
        assert P.sha(pre/'SHA256SUMS')==contract['preflight_source_sums_sha256']
        for name,h in contract['prefix_validation_dependencies'].items():assert P.sha(prefix/name)==h,name
        old_result=args.project/'runs/bridge-prefix-formal-209481'
        for name,h in contract['baseline_result_files'].items():assert P.sha(old_result/name)==h,name
        preflight=args.project/'runs/bridge-spatial-preflight-210291'
        subprocess.run([sys.executable,str(pre/'verify_preflight.py'),str(preflight)],check=True)
        verified=P.read(preflight/'report.json')
        assert torch.cuda.is_available() and str(torch.__version__)==P.read(pre/'old_report.json')['torch_version']
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
        r.update(protocol_sha256=P.sha(pre/'protocol.json'),formal_source_sums_sha256=P.sha(CODE/'SHA256SUMS'),
            formal_contract_sha256=P.sha(CODE/'formal_contract.json'),preflight_report_sha256=P.sha(preflight/'report.json'))
        checkpoint=args.project/p['checkpoint']['relative_path']
        assert checkpoint.stat().st_size==p['checkpoint']['bytes'] and P.sha(checkpoint)==p['checkpoint']['sha256']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        payload['source_checkpoint_sha256']=p['checkpoint']['sha256']
        cfg=copy.deepcopy(payload['config']);a=copy.deepcopy(cfg);b=P.read(pre/'base_protocol.json')['config']
        a['model']['name']=b['model']['name']='<frozen_local_CLIP>';assert a==b
        train_reference=args.project/p['training_reference']['relative_path']
        assert P.sha(train_reference)==p['training_reference']['sha256']
        with np.load(train_reference,allow_pickle=False) as z:
            assert np.array_equal(z['row_indices'],np.arange(1836));targets=z['targets'].copy()
        state,current,texts,all_rows,checks=P.load_numeric_inputs(args.project,targets)
        cache=torch.load(args.project/'bridge_training_coverage_v1/full_tokens.pt',map_location='cpu',weights_only=True)
        old_report=P.read(pre/'old_report.json')
        assert hashlib.sha256(json.dumps(cache['key'],sort_keys=True).encode()).hexdigest()==old_report['token_cache_key_sha256']
        for key in ('visual','language'):assert P.tensor_sha(cache[key])==cache['hashes'][key]
        cfg['model']['name']=str(args.project/'baseline_setup/clip-vit-large-patch14-v1')
        torch.manual_seed(42);model=P.RobotAdapterModel(cfg).cpu().eval()
        clip_hash=[P.fingerprint(e.state_dict()) for e in (model.vision_encoder,model.text_encoder)]
        def original_digest(module):
            h=hashlib.sha256()
            for n,t in sorted(module.state_dict().items()):h.update(n.encode());h.update(memoryview(t.detach().cpu().contiguous().numpy()).cast('B'))
            return h.hexdigest()
        assert [original_digest(e) for e in (model.vision_encoder,model.text_encoder)]==cache['key']['clip']
        tokenizer=P.CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
        width=cache['language'].shape[1]
        canonical=tokenizer([v['task'] for v in all_rows],padding='max_length',max_length=width,truncation=False,return_tensors='pt')
        native=tokenizer(texts,padding='max_length',max_length=width,truncation=False,return_tensors='pt')
        for k in ('input_ids','attention_mask'):
            assert P.tensor_sha(canonical[k])==cache['key']['tokenizer_inputs'][k]
            assert torch.equal(native[k],canonical[k][:1836])
        mask=canonical['attention_mask'][:1836]
        model.history_projection=torch.nn.Linear(1024,1024,bias=False)
        model.state_projection=torch.nn.Linear(9,1024,bias=False)
        prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
        for name,v in model.named_parameters():v.requires_grad_(name.startswith(prefixes))
        original_keys={n for n,v in model.named_parameters() if v.requires_grad}
        assert original_keys==set(payload['trainable_state_dict'])
        assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
        frozen_hash=P.fingerprint({n:v for n,v in model.named_parameters() if not v.requires_grad})
        for name,child in model.named_children():
            if name not in ('vision_encoder','text_encoder'):child.cuda()
        for name,value in list(model.named_buffers(recurse=False)):setattr(model,name,value.cuda())
        readout=P.install_readout(model.adapter,candidate=True)
        targets=torch.from_numpy(targets).cuda();state=torch.from_numpy(state).cuda();current=torch.from_numpy(current).cuda()
        def context(ids):
            ids=torch.as_tensor(ids,dtype=torch.long)
            return P.context_from_tokens(model,cache['visual'][ids].cuda(),cache['language'][ids].cuda(),
                mask[ids].cuda(),state[ids.cuda()])
        with np.load(pre/'schedule.npz',allow_pickle=False) as z:batches=z['batches'].copy()
        assert P.sha(pre/'schedule.npz')==p['schedule_sha256'] and batches.shape==(4096,8)
        assert batches.min()>=0 and batches.max()<1836
        r.update(initial_checkpoint_sha256=p['checkpoint']['sha256'],numeric_inputs=checks,
            training_token_cache_reused=True,new_training_tokens_encoded=0,
            estimated_paired_formal_seconds=verified['estimated_paired_formal_seconds'])
        persist()
        for i,arm in enumerate(('control','candidate')):
            assert verified['runs'][i]['arm']==arm
            run=run_arm(model,readout,payload,original_keys,context,targets,current,batches,arm,args.output,P,
                        verified['runs'][i]['losses'],updates=4096)
            r['training_runs'].append(run)
            assert run['trainable_parameters']==old_report['trainable_parameters']+(1026 if arm=='candidate' else 0)
            frozen_now=P.fingerprint({n:v for n,v in model.named_parameters() if n not in original_keys and not n.startswith('adapter.spatial_readout.')})
            assert frozen_now==frozen_hash;persist()
        assert r['training_runs'][0]['initial_base_sha256']==r['training_runs'][1]['initial_base_sha256']
        assert r['training_runs'][0]['first8_losses'][0]==r['training_runs'][1]['first8_losses'][0]
        r['both_weights_frozen_before_development_scoring']=True;persist()
        validation=load_validation(prefix,model,tokenizer,P,args.output)
        baseline_report=P.read(old_result/'report.json')
        r['validation_tokens']={k:validation[k] for k in ('visual_sha256','language_sha256')}
        assert r['validation_tokens']==baseline_report['validation_tokens']
        assert validation['rows']==P.read(old_result/'validation_rows.json')
        with np.load(old_result/'validation_inputs.npz',allow_pickle=False) as z:
            assert np.array_equal(validation['targets'],z['targets'])
            assert np.array_equal(validation['state'].cpu().numpy(),z['state_features'])
        with np.load(old_result/'unchanged207558_full16.npz',allow_pickle=False) as z:
            assert np.array_equal(z['targets'],validation['targets']) and np.array_equal(z['sampling_seeds'],[1101,1102,1103])
            unchanged=z['predictions'].copy()
        results={'unchanged207558':summarize_pair(unchanged,validation['targets'],validation['rows'])}
        np.savez_compressed(args.output/'unchanged207558_full16.npz',predictions=unchanged,
            targets=validation['targets'],sampling_seeds=np.asarray([1101,1102,1103]))
        for run in r['training_runs']:
            path=args.output/run['checkpoint'];assert P.sha(path)==run['checkpoint_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            restore(model,original_keys,readout,saved['model']);del saved
            predictions=evaluate(model,validation,P)
            np.savez_compressed(args.output/f"{run['arm']}_full16.npz",predictions=predictions,
                targets=validation['targets'],sampling_seeds=np.asarray([1101,1102,1103]))
            results[run['arm']]=summarize_pair(predictions,validation['targets'],validation['rows'])
            print('SPATIAL_DEVELOPMENT_METRICS',run['arm'],json.dumps({part:results[run['arm']][part]['episode_equal'] for part in ('first','full16')}),flush=True)
        identity=np.zeros((1,50,16,7),np.float32);identity[...,6]=1.
        results['static']=summarize_pair(identity,validation['targets'],validation['rows'])
        assert clip_hash==[P.fingerprint(e.state_dict()) for e in (model.vision_encoder,model.text_encoder)]
        assert frozen_hash==P.fingerprint({n:v for n,v in model.named_parameters() if n not in original_keys and not n.startswith('adapter.spatial_readout.')})
        r.update(results=results,adoption=admission(results),clip_unchanged=True,gripper_unchanged=True,
            optimization_operations=8192,development_episodes=10,development_windows=50,
            generalization_verdict_available=True,execution_checks_passed=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            limits='Same-task offline pose only; first target is within original full16 predictions; no learned gripper or closed-loop verdict')
        assert r['peak_reserved_gib']<=7.5
        print('SPATIAL_DEVELOPMENT_ADMISSION',json.dumps(r['adoption']),flush=True)
    except BaseException as error:
        r['execution_checks_passed']=False;r['error']=repr(error);raise
    finally:
        persist();print('SPATIAL_FORMAL_REPORT',args.output/'report.json',flush=True)
    print('SPATIAL FORMAL PAIR COMPLETE; execution passing is not an improvement verdict',flush=True)


if __name__=='__main__':main()
