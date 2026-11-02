"""One cluster preflight: all inputs, real207558 replay,8 updates per arm.

No development scoring, final25 access, formal fit entry, or persistent weights.
"""
import os
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'frozen'))
from models import RobotAdapterModel
from state_input_contract import state_features
from transformers import CLIPTokenizer
from bridge_prefix_model_interface_v1 import PrefixModelInterface
from bridge_prefix_preflight_inputs_v1 import verify_inputs,build_delta_cache,tensor_sha


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(memoryview(t.detach().cpu().contiguous().numpy()).cast('B'))
    return h.hexdigest()
def cpu_copy(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_copy(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_copy(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_copy(x) for x in v)
    return copy.deepcopy(v)
def fingerprint(v):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(memoryview(x.detach().cpu().contiguous().numpy()).cast('B'))
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda a:(type(a).__name__,str(a))):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for t in x:visit(t)
        else:h.update(repr(x).encode())
    visit(v);return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic();r=dict(stage='preflight',execution_checks_passed=False,
        formal_training_executed=False,development_scored=False,final25_payloads_read=False,
        gripper_trained=False,downloads=False,runs=[],protocol_sha256=sha(CODE/'protocol.json'),
        package_sums_sha256=sha(CODE/'SHA256SUMS'))
    def persist():
        r['elapsed_seconds']=time.monotonic()-begin
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        for line in (CODE/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1);assert sha(CODE/name)==expected,name
        p=read(CODE/'protocol.json');old_report=read(CODE/'old_report.json')
        assert torch.cuda.is_available() and str(torch.__version__)==old_report['torch_version']
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.cuda.reset_peak_memory_stats()
        checkpoint=args.project/'runs/bridge-coverage-rep-fit-207558/final_trainable.pt'
        assert sha(checkpoint)==p['initialization']['sha256'],'Actual207558 checkpoint differs'
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(payload['config'])
        original=read(CODE/'base_protocol.json')['config']
        a,b=copy.deepcopy(cfg),copy.deepcopy(original)
        a['model']['name']=b['model']['name']='<pinned_local_CLIP>'
        assert a==b,'Do not infer old architecture from current defaults'
        old_predictions=args.project/'runs/bridge-coverage-rep-fit-207558/reduced_u8192_train.npz'
        with np.load(old_predictions,allow_pickle=False) as z:
            old_targets=z['targets'].copy();saved_contexts=z['contexts'].copy();saved_predictions=z['predictions'].copy()
        assert old_targets.shape==(1836,16,8) and saved_contexts.shape==(1836,1024)
        data=verify_inputs(CODE,args.project/'bridge_expansion_training_v1/data',old_targets)
        r['input_checks']=data['checks'];persist()
        old_cache_path=args.project/'bridge_training_coverage_v1/full_tokens.pt'
        old_cache=torch.load(old_cache_path,map_location='cpu',weights_only=True)
        assert hashlib.sha256(json.dumps(old_cache['key'],sort_keys=True).encode()).hexdigest()==old_report['token_cache_key_sha256']
        assert old_cache['visual'].shape==(1876,257,1024)
        for key in ('visual','language'):assert tensor_sha(old_cache[key])==old_cache['hashes'][key]
        cfg['model']['name']=str(args.project/'baseline_setup/clip-vit-large-patch14-v1')
        torch.manual_seed(42);model=RobotAdapterModel(cfg).cpu().eval()
        clip_hash=[digest(model.vision_encoder),digest(model.text_encoder)]
        assert clip_hash==old_cache['key']['clip'],'LocalCLIP differs from original frozen cache'
        model.history_projection=torch.nn.Linear(1024,1024,bias=False)
        model.state_projection=torch.nn.Linear(9,1024,bias=False)
        prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
        for name,value in model.named_parameters():value.requires_grad_(name.startswith(prefixes))
        trainable_names={n for n,v in model.named_parameters() if v.requires_grad}
        assert set(payload['trainable_state_dict'])==trainable_names
        loaded=model.load_state_dict(payload['trainable_state_dict'],strict=False)
        assert not loaded.unexpected_keys
        original_frozen={n:v._version for n,v in model.named_parameters() if not v.requires_grad}
        tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
        cache_path=CODE/'delta_visual_tokens.pt'
        delta_visual,representatives,attention=build_delta_cache(CODE,cache_path,old_cache,model,tokenizer,data,r)
        # CLIP remains onCPU after encoding. Only trainable head/Adapter and
        # per-batch cached tokens go toGPU; do not allocate all3394 tokens there.
        for name in ('adapter','diffusion_decoder','history_projection','state_projection'):
            getattr(model,name).cuda()
        for name,value in list(model.named_buffers(recurse=False)):
            setattr(model,name,value.cuda())
        assert model.alpha_bars.is_cuda, 'Diffusion table must follow CUDA timestep indices'
        interface=PrefixModelInterface(model).eval()
        trainables=[v for v in interface.parameters() if v.requires_grad]
        assert sum(v.numel() for v in trainables)==132326023
        language_indices=torch.arange(3394);language_indices[1836:]=representatives
        targets=torch.from_numpy(data['targets']).cuda();validity=torch.from_numpy(data['validity']).cuda()
        state=torch.from_numpy(data['state']).cuda()
        def context(ids):
            ids=torch.as_tensor(ids,dtype=torch.long)
            v=torch.empty((len(ids),257,1024),dtype=old_cache['visual'].dtype)
            old=ids<1836
            if old.any():v[old]=old_cache['visual'][ids[old]]
            if (~old).any():v[~old]=delta_visual[ids[~old]-1836]
            lang=old_cache['language'][language_indices[ids]]
            return interface.context_from_tokens(v.cuda(),lang.cuda(),attention[ids].cuda(),state[ids.cuda()])
        # Reproduce saved seed1101 with the ORIGINAL batch2 RNG consumption.
        # Skipped old batches burn noise only; no old1836-window rescoring.
        wanted=[0,32,64,96];guard=dict(context_max_abs=0.,prediction_max_abs=0.,windows=wanted,sampling_seed=1101)
        eval_seconds=[]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(1101);torch.cuda.manual_seed_all(1101)
            with torch.no_grad():
                for start in range(0,98,2):
                    if start not in wanted:
                        for _ in range(100):torch.randn((2,16,7),device='cuda')
                        continue
                    ids=[start,start+1];ctx=context(ids)
                    guard['context_max_abs']=max(guard['context_max_abs'],float(np.max(abs(ctx.cpu().numpy()-saved_contexts[ids]))))
                    tick=time.monotonic();pred=interface.sample_pose(ctx,16);torch.cuda.synchronize()
                    eval_seconds.append((time.monotonic()-tick)/2)
                    guard['prediction_max_abs']=max(guard['prediction_max_abs'],float(np.max(abs(pred[0].cpu().numpy()-saved_predictions[0,start,:,:7]))))
        assert guard['context_max_abs']<=1e-4 and guard['prediction_max_abs']<=1e-4,guard
        r['trained207558_replay']=dict(guard,passed=True,context_atol=1e-4,prediction_atol=1e-4,rtol=0.)
        print('PREFIX_REAL207558_REPLAY',json.dumps(r['trained207558_replay']),flush=True)
        with np.load(CODE/'matched_schedule.npz',allow_pickle=False) as z:
            schedules={k:z[k][:8].copy() for k in ('control','candidate')};horizons=z['requested_horizons'][:8].copy()
        all_seconds=[];source_hash=fingerprint(payload['trainable_state_dict'])
        for arm in ('control','candidate'):
            model.load_state_dict(payload['trainable_state_dict'],strict=False)
            torch.nn.init.zeros_(interface.prefix.request_projection.weight)
            assert fingerprint({n:v for n,v in model.named_parameters() if v.requires_grad})==source_hash
            optimizer=torch.optim.Adam(trainables,lr=1e-5,weight_decay=0.)
            losses=[];restore_passed=False;warm=[];first_state=fingerprint(interface.prefix.request_projection.state_dict())
            def calculate(update):
                ids=schedules[arm][update-1];K=int(horizons[update-1])
                interface.train();torch.manual_seed(620000+update);torch.cuda.manual_seed_all(620000+update)
                ctx=context(ids)
                torch.manual_seed(610000+update);torch.cuda.manual_seed_all(610000+update)
                loss,_,pred=interface.loss(targets[ids,:,:7],validity[ids],ctx,K)
                assert torch.isfinite(loss)
                return loss,pred
            for update in range(1,9):
                tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
                loss,_=calculate(update);loss.backward()
                assert all(v.grad is None for n,v in model.named_parameters() if not v.requires_grad)
                assert all(torch.isfinite(v.grad).all() for v in trainables if v.grad is not None)
                norm=torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();warm.append(time.monotonic()-tick);losses.append(float(loss.detach()))
                print('PREFIX_PREFLIGHT_UPDATE',arm,update,'/8','K',int(horizons[update-1]),'loss',losses[-1],flush=True)
                if update==4:
                    state_before=fingerprint(interface.trainable_payload());adam_before=fingerprint(optimizer.state_dict())
                    with torch.enable_grad():preview,pred=calculate(5);preview_value=float(preview.detach());preview_hash=tensor_sha(pred)
                    del preview,pred
                    temporary=args.output/f'{arm}_restore_check.pt';assert not temporary.exists()
                    torch.save(dict(model=interface.trainable_payload(),adam=cpu_copy(optimizer.state_dict()),
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
                    restored=torch.load(temporary,map_location='cpu',weights_only=True)
                    interface.load_trainable_payload(restored['model']);optimizer.load_state_dict(restored['adam'])
                    torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state(restored['cuda_rng'])
                    assert fingerprint(interface.trainable_payload())==state_before
                    assert fingerprint(optimizer.state_dict())==adam_before
                    with torch.enable_grad():again,pred=calculate(5)
                    assert float(again.detach())==preview_value and tensor_sha(pred)==preview_hash
                    del again,pred,restored
                    # Exact known file only, never recursive cleanup or a guessed path.
                    temporary.unlink();restore_passed=True
            assert fingerprint(interface.prefix.request_projection.state_dict())!=first_state
            assert fingerprint({n:v for n,v in model.named_parameters() if v.requires_grad})!=source_hash
            assert all(v._version==original_frozen[n] for n,v in model.named_parameters() if not v.requires_grad)
            r['runs'].append(dict(arm=arm,logical_optimizer_updates=8,losses=losses,
                restore_weights_and_Adam_exact=restore_passed,restored_next_prediction_and_loss_exact=True,
                request_projection_changed=True,base_trainables_changed=True,temporary_checkpoint_removed=True))
            all_seconds+=warm[2:]
            del optimizer
        assert r['runs'][0]['losses'][0]==r['runs'][1]['losses'][0], 'Common initialK16 update differs'
        assert [digest(model.vision_encoder),digest(model.text_encoder)]==clip_hash
        r.update(execution_checks_passed=True,actual207558_checkpoint_sha256=p['initialization']['sha256'],
            old_full_tokens_reused=True,old_cache_key_sha256=old_report['token_cache_key_sha256'],
            clip_unchanged=True,gripper_unchanged=True,optimization_operations=16,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            warm_update_seconds=float(np.mean(all_seconds)),
            estimated_paired_formal_seconds=120+1.2*(float(np.mean(all_seconds))*8192+float(np.mean(eval_seconds))*546),
            new_state_projection_statistics_fitted=False,final_weights_saved=False,
            verdict='Operational preflight only; no generalization/fit verdict.')
        assert r['peak_reserved_gib']<=7.5 and r['estimated_paired_formal_seconds']<=7200
    except Exception as error:r['error']=repr(error);r['execution_checks_passed']=False;raise
    finally:persist();print('PREFIX_PREFLIGHT_REPORT',args.output/'report.json',flush=True)
    print('PREFIX PAIR PREFLIGHT PASSED; inspect report before any formal training',flush=True)


if __name__=='__main__':main()
