"""Verify raw168 inputs and cache same letterboxed image/pose for existing author env."""
import argparse
import ast
import os
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
from pathlib import Path
import sys
import json
import time
import numpy as np
from bridge_author_io import sha,read,rgb_from_clip,canonical

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--audit-only',action='store_true')
    args=ap.parse_args();start=time.monotonic();code=Path(__file__).resolve().parent
    protocol=read(code/'protocol.json');base=args.project/'bridge_growth128_v1'
    assert sha(base/'manifest.json')==protocol['base_manifest_sha256']
    manifest=read(base/'manifest.json');rows=manifest['train']+manifest['development']
    assert rows==protocol['rows']
    assert all(sha(base/n)==h for n,h in manifest['source_sha256'].items())
    sys.path.insert(0,str(base))
    from growth128_input_contract import expanded_items
    checks=[];comparisons=[]
    items=expanded_items(rows,args.data_root,manifest,args.project/'bridge_batch_input_audit_v1/reference',checks,comparisons)
    assert len(items)==168 and len(checks)==168
    images=np.stack([x[1].numpy() for x in items]);targets=np.stack([x[3].numpy() for x in items])
    assert targets.shape==(168,16,8) and np.isfinite(targets).all()
    rgb,inverse_delta=rgb_from_clip(images)
    statepack=args.project/'bridge_state_conditioning_v1'
    assert sha(statepack/'protocol.json')==protocol['state_protocol_sha256']
    assert sha(statepack/'current_pose.npz')==protocol['current_pose_sha256']
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('verified_observed_pose',statepack/'state_input_contract.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    features,state_checks=module.load_current_features(statepack/'protocol.json',rows,args.data_root,checks)
    assert sha(code/'language_cache.npz')==protocol['language_cache_sha256']
    with np.load(code/'language_cache.npz',allow_pickle=False) as z:
        texts=z['texts'].tolist();embeddings=z['embeddings'].copy()
    assert embeddings.shape==(len(texts),768) and set(texts)=={r['task'] for r in rows[:128]}
    language=np.stack([embeddings[texts.index(canonical(item[0]))] for item in items])
    assert all(canonical(item[0])==row['task'] for item,row in zip(items,rows))
    arrays=dict(image=rgb,ee_state=features,language=language.astype('<f4'),action=targets[...,:7].astype('<f4'))
    args.output.mkdir(parents=True,exist_ok=False)
    r=dict(passed=False,protocol_sha256=sha(code/'protocol.json'),rows=rows,train_windows=128,development_windows=40,
        raw_inputs_verified=True,data_checks=checks,input_numeric_comparisons=comparisons,
        state_checks=state_checks,image_inverse_clip_max_difference=inverse_delta,
        future_observations_used=False,history_length=1,action_first_target='source observation t+1 relative to current t',
        action_representation='tool_relative_pose_open_positive_v2: XYZ/0.1m then quaternion XYZW; gripper omitted',
        array_sha256={},audit_only=args.audit_only,trained=False,new_downloads=False)
    r['training_action_min']=arrays['action'][:128].reshape(-1,7).min(0).tolist()
    r['training_action_max']=arrays['action'][:128].reshape(-1,7).max(0).tolist()
    if not args.audit_only:
        for key,array in arrays.items():
            path=args.output/(key+'.npy');np.save(path,array,allow_pickle=False);r['array_sha256'][key]=sha(path)
    r.update(passed=True,elapsed_seconds=time.monotonic()-start)
    (args.output/'export_report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print('AUTHOR BRIDGE INPUT EXPORT: PASSED',args.output/'export_report.json',flush=True)
if __name__=='__main__':main()
