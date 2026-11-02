"""Runtime input verification and delta-token caching for the single preflight."""
from collections import defaultdict
import hashlib
import json
import time
import numpy as np
import torch

from dataset import prepare_image,relative_pose_action,euler_xyz_to_quaternion,bridge_reverse_scan_gripper
from state_input_contract import state_features
from bridge_prefix_input_contract_v1 import sha,records_at,episode_arrays,image_contract
from bridge_partial_future_contract_v1 import build_numeric_sample


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def tensor_sha(t):return hashlib.sha256(memoryview(t.detach().cpu().contiguous().numpy()).cast('B')).hexdigest()


def verify_inputs(code,train_root,old_targets):
    audit=read(code/'input_reference/report.json');p=read(code/'protocol.json')
    rows=read(code/'candidate_rows.json')['train']
    pool=read(code/'train_manifest.json')
    stats=read(code/'state_protocol.json')['state_features']
    assert audit['passed'] and len(rows)==3394
    targets=np.full((3394,16,8),np.nan,dtype=np.float32);targets[:1836]=old_targets
    availability=np.zeros((3394,16),dtype=bool);availability[:1836]=True
    currents=np.empty((3394,1),np.float32);states=np.empty((3394,9),np.float32)
    images=[None]*1558;texts=[None]*3394
    old_expected=np.load(code/'input_reference/old_training_reference.npz',allow_pickle=False)
    assert np.array_equal(old_targets,old_expected['targets'])
    max_deltas=dict(targets=0.,state_features=0.,normalized_image_grid=0.,current=0.)
    grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[row['shard'],row['record_index']].append((i,row))
    counts=dict(train_delta=0,validation=0,train_episodes=0,validation_episodes=0)
    role_specs=[('train_delta',pool['partitions']['train'],train_root),
        ('validation',p['validation']['episodes'],code/'validation_data')]
    for role,episodes,data_root in role_specs:
        with np.load(code/f'input_reference/{role}_reference.npz',allow_pickle=False) as z:
            expected={k:z[k].copy() for k in z.files}
        checks=audit['rows'][role]
        by_ep=defaultdict(list)
        for j,check in enumerate(checks):by_ep[check['row']['shard'],check['row']['record_index']].append((j,check))
        by_shard=defaultdict(list)
        for e in episodes:by_shard[e['shard']].append(e)
        for shard,es in sorted(by_shard.items()):
            path=data_root/shard
            expected_file=(pool['shard_sha256'][shard] if role=='train_delta' else audit['validation_export_sha256'][shard])
            assert sha(path)==expected_file,('Source file changed',role,shard)
            meta={e['record_index']:e for e in es}
            for index,payload,values in records_at(path,meta):
                e=meta[index];identity=shard,index
                source=next(v for v in audit['episodes'] if (v['role'],v['shard'],v['record_index'])==(role,shard,index))
                assert hashlib.sha256(payload).hexdigest()==source['payload_sha256']
                raw_states,actions,grip=episode_arrays(values,e,bridge_reverse_scan_gripper)
                if role=='train_delta':
                    for i,row in grouped[identity]:
                        t=row['start_index'];texts[i]=values['steps/language_instruction'][t].decode()
                        states[i]=state_features(raw_states[t:t+1,:6],stats)[0]
                        currents[i]=2.*float(raw_states[t,6])-1.
                        if i<1836:
                            delta=float(np.max(abs(states[i]-old_expected['state_features'][i])))
                            assert delta<=1e-6
                for j,check in by_ep[identity]:
                    row=check['row'];t=row['start_index']
                    sample=build_numeric_sample(raw_states,grip,t,relative_pose_action,euler_xyz_to_quaternion)
                    image,actual,grid=image_contract(values,t,prepare_image)
                    for name in ('encoded_image_sha256','decoded_rgb_sha256','rgb_shape','instruction_sha256','instruction'):
                        assert actual[name]==check[name],(role,j,name)
                    assert np.array_equal(sample['valid'],expected['validity'][j])
                    assert np.array_equal(np.isnan(sample['target']),np.isnan(expected['targets'][j]))
                    observed_state=state_features(raw_states[t:t+1,:6],stats)[0]
                    arrays=dict(targets=(sample['target'][sample['valid']],expected['targets'][j][sample['valid']]),
                        current=(sample['current_gripper'],expected['current'][j]),
                        state_features=(observed_state,expected['state_features'][j]),
                        normalized_image_grid=(grid,expected['normalized_image_grid'][j]))
                    for name,(a,b) in arrays.items():
                        assert a.shape==b.shape and np.isfinite(a).all()
                        difference=float(np.max(abs(a.astype(float)-b.astype(float))))
                        assert difference<=audit['numeric_tolerances'][name],(role,j,name,difference)
                        max_deltas[name]=max(max_deltas[name],difference)
                    assert np.array_equal(sample['target'][sample['valid'],7],expected['targets'][j][sample['valid'],7])
                    if role=='train_delta':
                        i=1836+j
                        assert rows[i]==row
                        images[j]=image;targets[i]=sample['target'];availability[i]=sample['valid']
                    counts[role]+=1
                counts['train_episodes' if role=='train_delta' else 'validation_episodes']+=1
                if role=='train_delta' and counts['train_episodes']%50==0:
                    print('PREFIX_VERIFY',counts['train_episodes'],'/405',counts['train_delta'],'/1558',flush=True)
    old_expected.close()
    assert counts==dict(train_delta=1558,validation=50,train_episodes=405,validation_episodes=10)
    assert all(v is not None for v in images+texts)
    return dict(targets=targets,validity=availability,current=currents,state=states,
        images=images,texts=texts,rows=rows,checks=dict(counts=counts,max_numeric_deltas=max_deltas,
            final25_payloads_read=False,validation_scored=False,old1836_images_redecoded=False))


def build_delta_cache(code,path,old_cache,model,tokenizer,data,report):
    begin=time.monotonic();old_language=old_cache['language'];length=old_language.shape[1]
    raw=tokenizer(data['texts'],padding=False,truncation=False)['input_ids']
    assert max(map(len,raw))<=length,'New instruction exceeds pinned token width; no silent truncation'
    text=tokenizer(data['texts'],padding='max_length',max_length=length,return_tensors='pt')
    pairs={}
    for i in range(1836):pairs.setdefault(tuple(text['input_ids'][i].tolist()),i)
    representatives=[]
    for i in range(1836,3394):
        token=tuple(text['input_ids'][i].tolist())
        assert token in pairs,'Instruction token sequence not in verified old cache; stop instead of normalizing it'
        representatives.append(pairs[token])
        assert torch.equal(text['attention_mask'][i],text['attention_mask'][pairs[token]])
    cache_key=dict(input_audit_sha256=sha(code/'input_reference/report.json'),
        protocol_sha256=sha(code/'protocol.json'),torch_version=str(torch.__version__),
        old_cache_key=old_cache['key'],old_cache_hashes=old_cache['hashes'],
        representative_language_rows=representatives)
    if path.exists():
        cached=torch.load(path,map_location='cpu',weights_only=True)
        assert cached['key']==cache_key
        assert cached['visual'].shape==(1558,257,1024)
        assert tensor_sha(cached['visual'])==cached['visual_sha256']
        reused=True
    else:
        assert not path.with_suffix('.partial').exists()
        visual=torch.empty((1558,257,1024),dtype=old_cache['visual'].dtype)
        model.vision_encoder.cuda().eval()
        with torch.no_grad():
            for start in range(0,1558,2):
                value=model.vision_encoder(pixel_values=torch.stack(data['images'][start:start+2]).cuda()).last_hidden_state
                assert torch.isfinite(value).all()
                visual[start:start+len(value)]=value.cpu()
                if start%200==0:print('PREFIX_ENCODE',start,'/1558 new training images',flush=True)
        model.vision_encoder.cpu();torch.cuda.empty_cache()
        cached=dict(key=cache_key,visual=visual,visual_sha256=tensor_sha(visual))
        torch.save(cached,path.with_suffix('.partial'));path.with_suffix('.partial').replace(path)
        reused=False
    # Reuse language ONLY after exact tokenizer IDs and attention masks match,
    # preserving raw instruction variants instead of silently rewriting text.
    report.update(delta_token_cache_reused=reused,delta_visual_rows=1558,
        delta_cache_key_sha256=hashlib.sha256(json.dumps(cache_key,sort_keys=True).encode()).hexdigest(),
        new_language_reused_by_exact_token_ids=True,encoding_seconds=time.monotonic()-begin,
        delta_cache_bytes=path.stat().st_size)
    data['images']=None
    return cached['visual'],torch.tensor(representatives),text['attention_mask']
