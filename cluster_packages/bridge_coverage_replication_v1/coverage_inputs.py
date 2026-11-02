"""Reuse the frozen415-episode audit and original168 numeric contracts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from growth128_input_contract import expanded_items
from bridge_input_contract import raw_identity
from dataset import UnifiedRobotDataset
from state_input_contract import load_current_features, state_features
from selection_contract import encode_or_reuse

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(v): return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def key(r): return r['shard'],r['record_index'],r['start_index']


def load_inputs(args, code, protocol, rows, batches, report):
    small_dir=code.parent/'bridge_growth128_v1'
    old=json.loads((small_dir/'manifest.json').read_text())
    old_rows=old['train']+old['development']
    checks=[];comparisons=[]
    old_items=expanded_items(old_rows,args.data_root,old,args.data_reference,checks,comparisons)
    old_features,state_check=load_current_features(code.parent/'bridge_state_conditioning_v1/protocol.json',
        old_rows,args.data_root,checks)
    state_protocol=json.loads((code.parent/'bridge_state_conditioning_v1/protocol.json').read_text())
    old_at={key(r):i for i,r in enumerate(old_rows)}
    retained_indices=[next(i for i,r in enumerate(rows) if key(r)==key(v)) for v in old_rows]
    assert retained_indices==list(range(128))+list(range(1836,1876))
    verified=set(retained_indices)
    verified.update(batches[:8].reshape(-1).tolist())
    if not args.local_bounded: verified=set(range(1876))
    reference=json.loads((code/'episode_reference.json').read_text())
    payload_reference={(e['shard'],e['record_index']):e for e in reference['episodes']}
    items=[None]*1876;observed=np.zeros((1876,6),np.float32)
    grouped={}
    for i in sorted(verified):grouped.setdefault((rows[i]['shard'],rows[i]['record_index']),[]).append(i)
    episode_checks=[];row_checks=[]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    for ei,(ep,indices) in enumerate(grouped.items()):
        example=UnifiedRobotDataset._load_tfrecord_example(str(args.data_root/ep[0]),ep[1])
        expected=payload_reference[ep]
        payload=hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()
        assert payload==expected['payload_sha256'], ('Payload changed',ep)
        states=np.asarray(example.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
        assert len(states)==expected['steps'] and np.isfinite(states).all()
        reader._load_tfrecord_example=lambda *_:example
        for i in indices:
            row=rows[i];t=row['start_index'];assert t in expected['starts'] and t+16<len(states)
            if key(row) in old_at: item=old_items[old_at[key(row)]]
            else:
                item=reader._get_tfrecord_bridge_state_action(dict(file_path=str(args.data_root/row['shard']),
                    record_index=row['record_index'],start_index=t))
            assert ' '.join(item[0].lower().split())==row['task']
            assert tuple(item[1].shape)==(3,224,224) and tuple(item[3].shape)==(16,8)
            assert len(item)==4 and all(torch.isfinite(item[j]).all() for j in [1,2,3])
            items[i]=item;observed[i]=states[t,:6]
            # The full deterministic payload and identical dataset.py/start were
            # geometrically audited previously; do not re-run that old experiment.
            row_checks.append(dict(index=i,image_sha256=tsha(item[1]),target_sha256=tsha(item[3]),
                current_sha256=tsha(item[2]),encoded_image_sha256=hashlib.sha256(
                    example.features.feature['steps/observation/image_0'].bytes_list.value[t]).hexdigest()))
        episode_checks.append(dict(shard=ep[0],record_index=ep[1],payload_sha256=payload))
        if (ei+1)%50==0: print('COVERAGE_INPUT',ei+1,'/',len(grouped),'episodes',flush=True)
    features=state_features(observed,state_protocol['state_features'])
    assert np.array_equal(features[retained_indices],old_features)
    report.update(retained_data_checks=checks,retained_numeric_comparisons=comparisons,
        retained_state_checks=state_check,retained_state_features_sha256=hashlib.sha256(old_features.tobytes()).hexdigest(),
        input_row_checks=sorted(row_checks,key=lambda v:v['index']),episode_payload_checks=episode_checks,
        verified_row_indices=sorted(verified),full_input_contract_passed=(len(verified)==1876),
        reused_complete_geometry_audit=True,retained_numeric_contract_passed=True)
    report['input_identity_sha256']=hashlib.sha256(json.dumps(dict(rows=report['input_row_checks'],episodes=episode_checks),sort_keys=True).encode()).hexdigest()
    return items,features,old_items,checks,retained_indices


def tokens(args,code,protocol,model,tokenizer,items,old_items,checks,retained_indices,report,digest):
    assert sha(code.parent/'bridge_training_coverage_v1/protocol.json')==protocol['cache_provenance_protocol_sha256']
    assert args.local_bounded or args.cache.is_file(), 'Existing complete cache required; no re-encoding allowed'
    start=time.monotonic();small=json.loads((code.parent/'bridge_growth128_v1/manifest.json').read_text())
    old_texts=tokenizer([v[0] for v in old_items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    identity=hashlib.sha256(json.dumps(dict(manifest=sha(code.parent/'bridge_growth128_v1/manifest.json'),
        checks=checks,sources=small['source_sha256']),sort_keys=True).encode()).hexdigest()
    assert args.old_cache.is_file(), 'Original cache must already exist'
    ov,ol,reused=encode_or_reuse(model,old_items,old_texts,args.old_cache,identity,digest);assert reused
    rows=json.loads((code/'rows.json').read_text());all_rows=rows['train']+rows['development']
    texts=tokenizer([v[0] if v is not None else r['task'] for v,r in zip(items,all_rows)],
        padding=True,truncation=True,max_length=77,return_tensors='pt')
    assert all(torch.equal(texts[k][retained_indices],v) for k,v in old_texts.items()), 'Old tokenized observations changed'
    cache_key=dict(input_identity_sha256=report['input_identity_sha256'],protocol_sha256=protocol['cache_provenance_protocol_sha256'],
        torch_version=str(torch.__version__),clip=[digest(model.vision_encoder),digest(model.text_encoder)],
        tokenizer_inputs={k:tsha(v) for k,v in texts.items()})
    if not args.local_bounded and args.cache.exists():
        cached=torch.load(args.cache,map_location='cpu',weights_only=True)
        assert cached['key']==cache_key and cached['hashes']=={k:tsha(cached[k]) for k in ['visual','language']}
        assert torch.equal(cached['visual'][retained_indices],ov.cpu()) and torch.equal(cached['language'][retained_indices],ol.cpu())
        visual,language=cached['visual'].cuda(),cached['language'].cuda();reused_large=True
    else:
        assert args.stage!='fit', 'Formal must reuse the complete preflight cache'
        # Zeros only for explicitly inactive LOCAL rows; every context access is
        # guarded against these rows, and this array is never saved locally.
        visual_cpu=torch.zeros((1876,257,1024),dtype=ov.dtype)
        language_cpu=torch.zeros((1876,ol.shape[1],ol.shape[2]),dtype=ol.dtype)
        visual_cpu[retained_indices]=ov.cpu();language_cpu[retained_indices]=ol.cpu()
        new=[i for i in report['verified_row_indices'] if i not in set(retained_indices)]
        with torch.no_grad():
            for lo in range(0,len(new),2):
                ix=new[lo:lo+2]
                visual_cpu[ix]=model.vision_encoder(pixel_values=torch.stack([items[i][1] for i in ix]).cuda()).last_hidden_state.cpu()
                if lo%200==0:print('COVERAGE_ENCODE',lo,'/',len(new),'new images',flush=True)
            for lo in range(0,len(new),16):
                ix=new[lo:lo+16]
                language_cpu[ix]=model.text_encoder(**{k:v[ix].cuda() for k,v in texts.items()}).last_hidden_state.cpu()
        if not args.local_bounded:
            assert report['full_input_contract_passed'] and len(new)==1708
            payload=dict(key=cache_key,visual=visual_cpu,language=language_cpu)
            payload['hashes']={k:tsha(payload[k]) for k in ['visual','language']}
            assert not args.cache.exists();args.cache.parent.mkdir(parents=True,exist_ok=True)
            temporary=args.cache.with_suffix('.partial');assert not temporary.exists()
            torch.save(payload,temporary)
            assert temporary.stat().st_size<protocol['resource_gate']['cache_budget_bytes']
            temporary.replace(args.cache)
        visual,language=visual_cpu.cuda(),language_cpu.cuda();reused_large=False
    report.update(original_token_cache_reused=True,retained_visual_tokens_exact=True,retained_language_tokens_exact=True,
        full_token_cache_reused=reused_large,local_full_token_cache_written=False,
        token_rows=1876,active_token_rows=len(report['verified_row_indices']),encoding_seconds=time.monotonic()-start,
        token_cache_key_sha256=hashlib.sha256(json.dumps(cache_key,sort_keys=True).encode()).hexdigest())
    return visual,language,texts['attention_mask'].cuda()
