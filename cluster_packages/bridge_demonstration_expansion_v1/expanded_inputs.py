"""Append401 audited training records; reuse the complete1876-row frozen cache."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from dataset import UnifiedRobotDataset
from state_input_contract import state_features
from fit_state_arm import trainable_copy,state_sha
from replay_and_fit_bridge_positive import evaluate


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tsha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def jsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()


def load_inputs(args,code,protocol,rows,batches,report):
    assert not args.local_bounded, 'No incomplete or synthetic local model preflight'
    original=code.parent/'bridge_coverage_replication_v1'
    old_protocol=read(original/'protocol.json')
    assert sha(original/'protocol.json')==protocol['baseline_protocol_sha256']
    spec=importlib.util.spec_from_file_location('frozen405_input_helper',original/'coverage_inputs.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old_rows=read(original/'rows.json')
    with np.load(original/'schedule.npz',allow_pickle=False) as z:old_batches=z['batches'].copy()
    prior={}
    # Reuse the frozen input-integrity checks, not its old geometry experiment.
    old_items,old_features,small_items,checks,old_retained=helper.load_inputs(
        args,original,old_protocol,old_rows['train']+old_rows['development'],old_batches,prior)
    baseline_path=code.parent/'runs/bridge-coverage-rep-fit-207558/report.json'
    assert sha(baseline_path)==protocol['baseline_report_sha256']
    baseline=read(baseline_path)
    for k in ('input_identity_sha256','input_row_checks','episode_payload_checks'):
        assert prior[k]==baseline[k],('Old input contract changed',k)
    assert rows[:1836]==old_rows['train'] and rows[3624:]==old_rows['development']
    extra_rows=rows[1836:3624]
    references=read(code/'new_input_reference.json')
    audit=read(code/'new_source_audit.json')
    episodes={(e['shard'],e['record_index']):e for e in audit['episodes']}
    grouped={}
    for i,row in enumerate(extra_rows):grouped.setdefault((row['shard'],row['record_index']),[]).append(i)
    assert len(grouped)==401
    extra=[None]*1788;observed=np.zeros((1788,6),np.float32)
    extra_checks=[];payloads=[]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    with np.load(code/'new_numeric_reference.npz',allow_pickle=False) as ref:
        pixel_indices=ref['pixel_indices']
        for count,(key,indices) in enumerate(grouped.items()):
            example=UnifiedRobotDataset._load_tfrecord_example(str(code/'data'/key[0]),key[1])
            payload=hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()
            assert payload==episodes[key]['payload_sha256']
            states=np.asarray(example.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
            reader._load_tfrecord_example=lambda *_:example
            for i in indices:
                row=extra_rows[i];t=row['start_index'];expected=references['rows'][i]
                assert row==expected['row'] and t in episodes[key]['starts']
                item=reader._get_tfrecord_bridge_state_action(dict(file_path=row['shard'],record_index=row['record_index'],start_index=t))
                assert ' '.join(item[0].lower().split())==row['task'] and item[3].shape==(16,8)
                assert all(torch.isfinite(item[k]).all() for k in (1,2,3))
                assert item[1].shape==(3,224,224)
                image=item[1].numpy();target=item[3].numpy();current=item[2].numpy()
                assert np.max(np.abs(target.astype(float)-ref['targets'][i]))<=1e-6
                assert np.array_equal(current,ref['currents'][i])
                assert np.array_equal(states[t,:6],ref['observed'][i])
                assert np.max(np.abs(image.reshape(-1)[pixel_indices].astype(float)-ref['pixel_sketch'][i]))<=1e-6
                encoded=example.features.feature['steps/observation/image_0'].bytes_list.value[t]
                assert hashlib.sha256(encoded).hexdigest()==expected['encoded_image_sha256']
                extra[i]=item;observed[i]=states[t,:6]
                extra_checks.append(dict(index=1836+i,image_sha256=tsha(item[1]),target_sha256=tsha(item[3]),
                    current_sha256=tsha(item[2]),encoded_image_sha256=expected['encoded_image_sha256']))
            payloads.append(dict(shard=key[0],record_index=key[1],payload_sha256=payload))
            if (count+1)%50==0:print('ADDED_INPUT',count+1,'/401',flush=True)
    stats=read(code.parent/'bridge_state_conditioning_v1/protocol.json')['state_features']
    added_features=state_features(observed,stats)
    with np.load(code/'new_numeric_reference.npz',allow_pickle=False) as ref:
        assert np.max(np.abs(added_features.astype(float)-ref['state_features']))<=1e-6
    items=old_items[:1836]+extra+old_items[1836:]
    features=np.concatenate([old_features[:1836],added_features,old_features[1836:]])
    remap=list(range(1836))+list(range(3624,3664))
    old_checks=[dict(v,index=remap[v['index']]) for v in prior['input_row_checks']]
    row_checks=sorted(old_checks+extra_checks,key=lambda v:v['index'])
    all_payloads=prior['episode_payload_checks']+payloads
    retained=list(range(128))+list(range(3624,3664))
    report.update(retained_data_checks=checks,retained_state_checks=prior['retained_state_checks'],
        retained_state_features_sha256=prior['retained_state_features_sha256'],
        retained_numeric_comparisons=prior['retained_numeric_comparisons'],
        input_row_checks=row_checks,episode_payload_checks=all_payloads,
        verified_row_indices=list(range(3664)),full_input_contract_passed=True,
        reused_complete_geometry_audit=True,new401_geometry_audit_reused=True,
        retained_numeric_contract_passed=True,old_1876_input_contract_exact=True,
        old_input_identity_sha256=prior['input_identity_sha256'],old_1876_to_new_row_indices=remap,
        old_token_cache_key_sha256=baseline['token_cache_key_sha256'],
        input_identity_sha256=jsha(dict(rows=row_checks,episodes=all_payloads)))
    return items,features,small_items,checks,retained


def tokens(args,code,protocol,model,tokenizer,items,old_items,checks,retained_indices,report,digest):
    started=time.monotonic()
    old_map=report['old_1876_to_new_row_indices']
    texts=tokenizer([v[0] for v in items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    old_texts=tokenizer([items[i][0] for i in old_map],padding=True,truncation=True,max_length=77,return_tensors='pt')
    assert all(torch.equal(texts[k][old_map],v) for k,v in old_texts.items())
    clip=[digest(model.vision_encoder),digest(model.text_encoder)]
    old_key=dict(input_identity_sha256=report['old_input_identity_sha256'],
        protocol_sha256=protocol['old_cache_provenance_protocol_sha256'],torch_version=str(torch.__version__),
        clip=clip,tokenizer_inputs={k:tsha(v) for k,v in old_texts.items()})
    assert jsha(old_key)==report['old_token_cache_key_sha256']
    old=torch.load(args.old_cache,map_location='cpu',weights_only=True)
    assert old['key']==old_key and old['hashes']=={k:tsha(old[k]) for k in ('visual','language')}
    assert old['visual'].shape==(1876,257,1024)
    key=dict(input_identity_sha256=report['input_identity_sha256'],protocol_sha256=sha(code/'protocol.json'),
        torch_version=str(torch.__version__),clip=clip,tokenizer_inputs={k:tsha(v) for k,v in texts.items()})
    if args.cache.exists():
        cached=torch.load(args.cache,map_location='cpu',weights_only=True)
        assert cached['key']==key and cached['hashes']=={k:tsha(cached[k]) for k in ('visual','language')}
        reused=True
    else:
        assert args.stage=='preflight','Formal must reuse the complete preflight cache'
        cached=dict(key=key,visual=torch.zeros((3664,257,1024),dtype=old['visual'].dtype),
                    language=torch.zeros((3664,*old['language'].shape[1:]),dtype=old['language'].dtype))
        cached['visual'][old_map]=old['visual'];cached['language'][old_map]=old['language']
        with torch.no_grad():
            for lo in range(1836,3624,2):
                ix=list(range(lo,min(lo+2,3624)))
                cached['visual'][ix]=model.vision_encoder(pixel_values=torch.stack([items[i][1] for i in ix]).cuda()).last_hidden_state.cpu()
                if (lo-1836)%200==0:print('ADDED_ENCODE',lo-1836,'/1788',flush=True)
            for lo in range(1836,3624,16):
                ix=list(range(lo,min(lo+16,3624)))
                cached['language'][ix]=model.text_encoder(**{k:v[ix].cuda() for k,v in texts.items()}).last_hidden_state.cpu()
        cached['hashes']={k:tsha(cached[k]) for k in ('visual','language')}
        temporary=args.cache.with_suffix('.partial');assert not temporary.exists()
        torch.save(cached,temporary);assert temporary.stat().st_size<5*1024**3
        temporary.replace(args.cache);reused=False
    assert torch.equal(cached['visual'][old_map],old['visual'])
    assert torch.equal(cached['language'][old_map],old['language'])
    visual=cached['visual'].cuda();language=cached['language'].cuda()
    report.update(original_1876_token_cache_reused=True,retained_visual_tokens_exact=True,retained_language_tokens_exact=True,
        full_token_cache_reused=reused,local_full_token_cache_written=False,token_rows=3664,active_token_rows=3664,
        old_token_rows_reencoded=0,new_token_rows_encoded=(0 if reused else 1788),
        encoding_seconds=time.monotonic()-started,token_cache_key_sha256=jsha(key))
    return visual,language,texts['attention_mask'].cuda()


def old_replay_guard(args,code,protocol,model,visual,language,mask,features,items,report):
    checkpoint=code.parent/'runs/bridge-coverage-rep-fit-207558/final_trainable.pt'
    assert sha(checkpoint)==protocol['baseline_checkpoint_sha256']
    before=trainable_copy(model);before_hash=state_sha(before)
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    weights=value['trainable_state_dict'];assert set(weights)==set(before)
    model.load_state_dict(weights,strict=False);model.eval()
    ids=list(range(3624,3628));contexts=[]
    # Preserve the joint execution-mode/parameter-flag bundle proven on208034.
    with torch.inference_mode(False),torch.no_grad():
        for lo in range(0,4,2):
            ix=ids[lo:lo+2]
            now=model.adapter(visual[ix],language[ix],mask[ix])
            again=model.adapter(visual[ix],language[ix],mask[ix])
            contexts.append(now+model.history_projection(again-now)+model.state_projection(torch.from_numpy(features[ix]).cuda()))
        ctx=torch.cat(contexts)
    truth=np.stack([items[i][3].numpy() for i in ids])
    current=torch.stack([items[i][2] for i in ids]).cuda()
    pred,_=evaluate(model,ctx,current,truth,[1101,1102,1103],batch=2)
    np.savez_compressed(args.output/'old4_replay_check.npz',predictions=pred,contexts=ctx.cpu().numpy(),targets=truth)
    with np.load(code/'old4_reference.npz',allow_pickle=False) as old:
        assert np.array_equal(truth,old['targets'])
        cd=float(np.abs(ctx.cpu().numpy()-old['contexts']).max())
        pd=float(np.abs(pred-old['predictions']).max())
    report['old4_frozen_replay']=dict(context_max_abs=cd,prediction_max_abs=pd,passed=(cd<=1e-4 and pd<=1e-4))
    model.load_state_dict(before,strict=False)
    assert state_sha(trainable_copy(model))==before_hash
    assert all(p.grad is None for p in model.parameters())
    assert report['old4_frozen_replay']['passed'],('Old replay failed; inspect saved arrays',cd,pd)
    print('OLD4_FROZEN_REPLAY PASSED',cd,pd,flush=True)
