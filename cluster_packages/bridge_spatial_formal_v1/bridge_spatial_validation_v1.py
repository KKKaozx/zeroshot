"""Read ONLY the fixed ten qualified development records after weights freeze."""
from collections import defaultdict
import json
import numpy as np
import torch


def load_validation(pre, model, tokenizer, P, output):
    from dataset import prepare_image,relative_pose_action,euler_xyz_to_quaternion,bridge_reverse_scan_gripper
    from state_input_contract import state_features
    from bridge_prefix_input_contract_v1 import records_at,image_contract,episode_arrays
    from bridge_partial_future_contract_v1 import build_numeric_sample
    audit=P.read(pre/'input_reference/report.json')
    rows=P.read(pre/'input_reference/validation_rows.json')
    checks=audit['rows']['validation']
    assert len(rows)==len(checks)==50 and [c['row'] for c in checks]==rows
    p=P.read(pre/'protocol.json')
    metadata={(e['shard'],e['record_index']):e for e in p['validation']['episodes']}
    protected={(e['shard'],e['record_index']) for e in p['final_assessment']['metadata_only_episodes']}
    assert len(metadata)==10 and not(set(metadata)&protected)
    stats=P.read(pre/'state_protocol.json')['state_features']
    grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[row['shard'],row['record_index']].append(i)
    assert set(grouped)==set(metadata)
    with np.load(pre/'input_reference/validation_reference.npz',allow_pickle=False) as z:
        targets=z['targets'].copy();state=z['state_features'].copy();current=z['current'].copy()
        grid_reference=z['normalized_image_grid'].copy();assert z['validity'].all()
    expected=[v.copy() for v in (targets,state,current)]
    images,texts=[None]*50,[None]*50
    for (shard,index),indices in grouped.items():
        path=pre/'validation_data'/shard
        assert P.sha(path)==audit['validation_export_sha256'][shard]
        record,payload,values=next(records_at(path,[index]))
        source=next(e for e in audit['episodes'] if (e['role'],e['shard'],e['record_index'])==('validation',shard,index))
        import hashlib
        assert record==index and hashlib.sha256(payload).hexdigest()==source['payload_sha256']
        raw,actions,grip=episode_arrays(values,metadata[shard,index],bridge_reverse_scan_gripper)
        for i in indices:
            t=rows[i]['start_index']
            item=build_numeric_sample(raw,grip,t,relative_pose_action,euler_xyz_to_quaternion)
            assert item['valid'].all();targets[i]=item['target'];current[i]=item['current_gripper']
            state[i]=state_features(raw[t:t+1,:6],stats)[0]
            image,actual,grid=image_contract(values,t,prepare_image)
            assert all(actual[k]==checks[i][k] for k in actual)
            assert float(abs(grid-grid_reference[i]).max())<=1e-6
            images[i],texts[i]=image,actual['instruction']
    assert all(v is not None for v in images+texts)
    for a,b in zip((targets,state,current),expected):assert float(abs(a-b).max())<=1e-6
    tokenized=tokenizer(texts,padding=True,max_length=77,truncation=False,return_tensors='pt')
    assert tokenized['input_ids'].shape[1]<=77
    for encoder in (model.vision_encoder,model.text_encoder):
        assert not any(v.requires_grad for v in encoder.parameters());encoder.cuda().eval()
    visual,language=[],[]
    with torch.no_grad():
        for start in range(0,50,2):
            visual.append(model.vision_encoder(pixel_values=torch.stack(images[start:start+2]).cuda()).last_hidden_state.cpu())
            tokens={k:v[start:start+2].cuda() for k,v in tokenized.items()}
            language.append(model.text_encoder(**tokens).last_hidden_state.cpu())
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    visual,language=torch.cat(visual),torch.cat(language)
    assert torch.isfinite(visual).all() and torch.isfinite(language).all()
    np.savez_compressed(output/'validation_inputs.npz',targets=targets,state_features=state,current=current)
    (output/'validation_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    return dict(rows=rows,targets=targets,state=torch.from_numpy(state).cuda(),
        current=torch.from_numpy(current).cuda(),visual=visual,language=language,
        attention=tokenized['attention_mask'],visual_sha256=P.tensor_sha(visual),language_sha256=P.tensor_sha(language))
