"""One real207558 replay and eight updates per arm; no development scoring.

Uses original full16 diffusion, original1836 rows, and existing frozen tokens.
Does not import requested-prefix interfaces or encode any new observation.
"""
import os
os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1')
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'frozen'))
from models import RobotAdapterModel
from state_input_contract import state_features
from dataset import relative_pose_action, euler_xyz_to_quaternion, bridge_reverse_scan_gripper
from transformers import CLIPTokenizer
from bridge_prefix_input_contract_v1 import records_at, episode_arrays
from bridge_partial_future_contract_v1 import build_numeric_sample
from bridge_spatial_readout_interface_v1 import install_readout, context_from_tokens, reset_readout


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(memoryview(value.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def fingerprint(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((tuple(v.shape),str(v.dtype))).encode())
            h.update(memoryview(v.detach().cpu().contiguous().numpy()).cast('B'))
        elif isinstance(v,dict):
            for k in sorted(v, key=lambda k:(type(k).__name__,str(k))):
                h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def cpu_copy(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_copy(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_copy(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_copy(x) for x in v)
    return copy.deepcopy(v)


def check_sums():
    for line in (CODE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert sha(CODE/name)==digest, name


def load_numeric_inputs(project, saved_targets):
    """Rebuild numeric labels/state only from exactly the pinned405 records."""
    rows = read(CODE/'old_rows.json')
    all_rows = rows['train']+rows['development']
    assert len(rows['train'])==1836 and len(rows['development'])==40
    metadata = read(CODE/'train_manifest.json')['partitions']['train']
    audit = read(CODE/'input_audit.json')
    expected = {(e['shard'],e['record_index']):e for e in audit['episodes'] if e['role']=='train_delta'}
    stats = read(CODE/'state_protocol.json')['state_features']
    grouped = defaultdict(list)
    for i,row in enumerate(rows['train']):grouped[row['shard'],row['record_index']].append(i)
    by_shard = defaultdict(dict)
    for e in metadata:by_shard[e['shard']][e['record_index']]=e
    assert len(grouped)==len(metadata)==405 and set(grouped)==set(expected)
    states=np.empty((1836,9),np.float32);current=np.empty((1836,1),np.float32)
    raw_text=[None]*1836;maximum=0.;count=0
    for shard,episodes in sorted(by_shard.items()):
        for index,payload,values in records_at(project/'bridge_expansion_training_v1/data'/shard,episodes):
            identity=shard,index
            assert hashlib.sha256(payload).hexdigest()==expected[identity]['payload_sha256']
            raw,actions,grip=episode_arrays(values,episodes[index],bridge_reverse_scan_gripper)
            for i in grouped[identity]:
                t=rows['train'][i]['start_index']
                sample=build_numeric_sample(raw,grip,t,relative_pose_action,euler_xyz_to_quaternion)
                assert sample['valid'].all()
                delta=float(np.max(abs(sample['target']-saved_targets[i])))
                assert delta<=1e-6,(i,delta)
                assert np.array_equal(sample['target'][:,7],saved_targets[i,:,7])
                maximum=max(maximum,delta)
                states[i]=state_features(raw[t:t+1,:6],stats)[0]
                current[i]=sample['current_gripper']
                raw_text[i]=values['steps/language_instruction'][t].decode()
            count+=1
    assert count==405 and all(t is not None for t in raw_text)
    assert np.isfinite(states).all() and np.isfinite(current).all()
    return states,current,raw_text,all_rows,dict(episodes=count,windows=1836,
        target_max_abs=maximum,gripper_targets_exact=True,training_images_decoded=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    r=dict(stage='spatial_readout_preflight',execution_checks_passed=False,runs=[],
        development_scored=False,final25_payloads_read=False,formal_training_executed=False,
        gripper_trained=False,CLIP_training_allowed=False,downloads=False,
        final_weights_saved=False,images_encoded=0,validation_payloads_read=False)
    def persist():
        r['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        check_sums()
        p=read(CODE/'protocol.json');old_report=read(CODE/'old_report.json')
        r.update(protocol_sha256=sha(CODE/'protocol.json'),package_sums_sha256=sha(CODE/'SHA256SUMS'))
        assert torch.cuda.is_available() and str(torch.__version__)==old_report['torch_version']
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.cuda.reset_peak_memory_stats()
        checkpoint=args.project/p['checkpoint']['relative_path']
        assert checkpoint.stat().st_size==p['checkpoint']['bytes']
        assert sha(checkpoint)==p['checkpoint']['sha256']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(payload['config']);expected=read(CODE/'base_protocol.json')['config']
        a,b=copy.deepcopy(cfg),copy.deepcopy(expected)
        a['model']['name']=b['model']['name']='<frozen_local_CLIP>'
        assert a==b
        saved_train=args.project/p['training_reference']['relative_path']
        assert sha(saved_train)==p['training_reference']['sha256']
        with np.load(saved_train,allow_pickle=False) as z:
            assert np.array_equal(z['row_indices'],np.arange(1836))
            targets=z['targets'].copy();saved_context=z['contexts'].copy();saved_prediction=z['predictions'].copy()
        assert targets.shape==(1836,16,8) and saved_context.shape==(1836,1024)
        assert saved_prediction.shape==(3,1836,16,8)
        state,current,texts,all_rows,input_checks=load_numeric_inputs(args.project,targets)
        r['numeric_inputs']=input_checks;persist()
        cache=torch.load(args.project/'bridge_training_coverage_v1/full_tokens.pt',map_location='cpu',weights_only=True)
        assert hashlib.sha256(json.dumps(cache['key'],sort_keys=True).encode()).hexdigest()==old_report['token_cache_key_sha256']
        assert cache['visual'].shape==(1876,257,1024)
        for key in ('visual','language'):assert tensor_sha(cache[key])==cache['hashes'][key]
        cfg['model']['name']=str(args.project/'baseline_setup/clip-vit-large-patch14-v1')
        torch.manual_seed(42);model=RobotAdapterModel(cfg).cpu().eval()
        clip_hash=[fingerprint(v.state_dict()) for v in (model.vision_encoder,model.text_encoder)]
        # Original cache digests include the same sorted name/tensor byte layout.
        def original_digest(module):
            h=hashlib.sha256()
            for name,t in sorted(module.state_dict().items()):
                h.update(name.encode());h.update(memoryview(t.detach().cpu().contiguous().numpy()).cast('B'))
            return h.hexdigest()
        assert [original_digest(v) for v in (model.vision_encoder,model.text_encoder)]==cache['key']['clip']
        tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
        width=cache['language'].shape[1]
        canonical=tokenizer([v['task'] for v in all_rows],padding='max_length',max_length=width,
                            truncation=False,return_tensors='pt')
        native=tokenizer(texts,padding='max_length',max_length=width,truncation=False,return_tensors='pt')
        for k in ('input_ids','attention_mask'):
            assert tensor_sha(canonical[k])==cache['key']['tokenizer_inputs'][k]
            assert torch.equal(native[k],canonical[k][:1836])
        mask=canonical['attention_mask'][:1836]
        model.history_projection=torch.nn.Linear(1024,1024,bias=False)
        model.state_projection=torch.nn.Linear(9,1024,bias=False)
        prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
        for name,v in model.named_parameters():v.requires_grad_(name.startswith(prefixes))
        original_keys={n for n,v in model.named_parameters() if v.requires_grad}
        assert set(payload['trainable_state_dict'])==original_keys
        assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
        frozen_before=fingerprint({n:v for n,v in model.named_parameters() if not v.requires_grad})
        # Native loss/sample ALSO calls the frozen gripper layers. Keep their
        # device compatible without enabling their gradients or optimizing them.
        for name,child in model.named_children():
            if name not in ('vision_encoder','text_encoder'):child.cuda()
        for name,value in list(model.named_buffers(recurse=False)):setattr(model,name,value.cuda())
        targets=torch.from_numpy(targets).cuda();state=torch.from_numpy(state).cuda();current=torch.from_numpy(current).cuda()
        def context(ids):
            ids=torch.as_tensor(ids,dtype=torch.long)
            return context_from_tokens(model,cache['visual'][ids].cuda(),cache['language'][ids].cuda(),
                mask[ids].cuda(),state[ids.cuda()])
        wanted=[0,32,64,96];guard=dict(context_max_abs=0.,prediction_max_abs=0.)
        sample_seconds=[];captured={}
        model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            torch.manual_seed(1101);torch.cuda.manual_seed_all(1101)
            for start in range(0,98,2):
                if start not in wanted:
                    for _ in range(100):torch.randn((2,16,7),device='cuda')
                    continue
                ids=[start,start+1];ctx=context(ids);tick=time.monotonic()
                predicted=model.sample(ctx,current[ids]);torch.cuda.synchronize()
                sample_seconds.append((time.monotonic()-tick)/2)
                guard['context_max_abs']=max(guard['context_max_abs'],float(np.max(abs(ctx.cpu().numpy()-saved_context[ids]))))
                guard['prediction_max_abs']=max(guard['prediction_max_abs'],float(np.max(abs(predicted[0].cpu().numpy()-saved_prediction[0,start]))))
                captured[start]=(ctx.clone(),torch.cuda.get_rng_state().clone())
        assert guard['context_max_abs']<=1e-4 and guard['prediction_max_abs']<=1e-4,guard
        r['original207558_replay']=dict(guard,passed=True,windows=wanted,seed=1101)
        # Compare native sampler outputs under identical saved RNG and contexts;
        # full legacy replay above remains the authority for old-window identity.
        readout=install_readout(model.adapter,candidate=True)
        model.eval();zero_guard=dict(context_max_abs=0.,prediction_max_abs=0.)
        with torch.no_grad():
            for start in wanted:
                ids=[start,start+1];old_ctx,rng=captured[start];new_ctx=context(ids)
                zero_guard['context_max_abs']=max(zero_guard['context_max_abs'],float((old_ctx-new_ctx).abs().max()))
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.cuda.set_rng_state(rng);a=model.sample(old_ctx,current[ids])
                    torch.cuda.set_rng_state(rng);b=model.sample(new_ctx,current[ids])
                zero_guard['prediction_max_abs']=max(zero_guard['prediction_max_abs'],float((a-b).abs().max()))
        assert zero_guard['context_max_abs']<=1e-4 and zero_guard['prediction_max_abs']<=1e-4,zero_guard
        r['zero_readout_full_sampler_replay']=dict(zero_guard,passed=True)
        with np.load(CODE/'schedule.npz',allow_pickle=False) as z:batches=z['batches'].copy()
        assert batches.shape==(4096,8) and batches.min()>=0 and batches.max()<1836
        def snapshot():
            return dict(base={n:v.detach().cpu().clone() for n,v in model.named_parameters() if n in original_keys},
                        readout=cpu_copy(readout.state_dict()))
        def restore(values):
            assert set(values['base'])==original_keys
            assert not model.load_state_dict(values['base'],strict=False).unexpected_keys
            readout.load_state_dict(values['readout'],strict=True)
        timings=[]
        for arm,candidate in (('control',False),('candidate',True)):
            assert not model.load_state_dict(payload['trainable_state_dict'],strict=False).unexpected_keys
            reset_readout(readout,candidate)
            base_initial=fingerprint(snapshot()['base'])
            trainables=[v for v in model.parameters() if v.requires_grad]
            assert sum(v.numel() for v in trainables)==old_report['trainable_parameters']+(1026 if candidate else 0)
            optimizer=torch.optim.Adam(trainables,lr=1e-5,weight_decay=0.)
            assert not optimizer.state
            losses=[];restored=False;score_gradients=None
            def calculate(update):
                ids=torch.from_numpy(batches[update-1]).long();model.train()
                global_update=8192+update
                torch.manual_seed(430000+global_update);torch.cuda.manual_seed_all(430000+global_update)
                ctx=context(ids)
                torch.manual_seed(420000+global_update);torch.cuda.manual_seed_all(420000+global_update)
                guess,truth,_=model.diffusion_loss(targets[ids],ctx,current[ids])
                return torch.nn.functional.mse_loss(guess,truth),guess
            for update in range(1,9):
                tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
                loss,pred=calculate(update);assert torch.isfinite(loss)
                loss.backward()
                assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
                assert all(torch.isfinite(v.grad).all() for v in trainables if v.grad is not None)
                if update==1 and candidate:
                    score_gradients={n:float(v.grad.detach().norm()) for n,v in readout.named_parameters()}
                    assert all(v>0 for v in score_gradients.values())
                torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();timings.append(time.monotonic()-tick)
                losses.append(float(loss.detach()));del loss,pred
                print('SPATIAL_PREFLIGHT_UPDATE',arm,update,'/8','loss',losses[-1],flush=True)
                if update==4:
                    values=snapshot();adam=cpu_copy(optimizer.state_dict())
                    before=(fingerprint(values),fingerprint(adam))
                    preview,pred=calculate(5);comparison=(float(preview.detach()),tensor_sha(pred));del preview,pred
                    temporary=args.output/f'{arm}_restore_check.pt'
                    assert not temporary.exists()
                    torch.save(dict(model=values,adam=adam),temporary)
                    loaded=torch.load(temporary,map_location='cpu',weights_only=True)
                    restore(loaded['model']);optimizer.load_state_dict(loaded['adam'])
                    assert (fingerprint(snapshot()),fingerprint(optimizer.state_dict()))==before
                    again,pred=calculate(5)
                    assert (float(again.detach()),tensor_sha(pred))==comparison
                    del again,pred,loaded,values,adam
                    temporary.unlink();restored=True
            assert fingerprint(snapshot()['base'])!=base_initial
            if candidate:assert any(torch.count_nonzero(v)>0 for v in readout.parameters())
            else:assert all(torch.count_nonzero(v)==0 for v in readout.parameters())
            r['runs'].append(dict(arm=arm,logical_updates=8,losses=losses,initial_base_sha256=base_initial,
                base_parameters_changed=True,score_parameters_changed=candidate,
                new_score_gradient_norms=score_gradients,weights_Adam_and_next_calculation_restore_exact=restored,
                trainable_parameters=sum(v.numel() for v in trainables)))
            persist();del optimizer
        assert r['runs'][0]['initial_base_sha256']==r['runs'][1]['initial_base_sha256']
        assert r['runs'][0]['losses'][0]==r['runs'][1]['losses'][0]
        frozen_after=fingerprint({n:v for n,v in model.named_parameters() if n not in original_keys and not n.startswith('adapter.spatial_readout.')})
        assert frozen_before==frozen_after
        assert clip_hash==[fingerprint(v.state_dict()) for v in (model.vision_encoder,model.text_encoder)]
        peak=torch.cuda.max_memory_reserved()/1024**3
        warm=float(np.mean(timings[2:8]+timings[10:16]))
        estimate=120.+1.25*(warm*8192+float(np.mean(sample_seconds))*300)
        assert peak<=7.5 and estimate<=7200,(peak,estimate)
        r.update(execution_checks_passed=True,optimization_operations=16,clip_unchanged=True,
            other_frozen_unchanged=True,peak_reserved_gib=peak,warm_update_seconds=warm,
            estimated_paired_formal_seconds=estimate,actual_checkpoint_sha256=sha(checkpoint),
            training_token_cache_reused=True,new_token_cache_written=False)
        print('SPATIAL PREFLIGHT PASSED; no generalization verdict',args.output/'report.json',flush=True)
    except BaseException as error:
        r['error']=repr(error);raise
    finally:persist()


if __name__=='__main__':main()
