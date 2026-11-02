"""Reuse the frozen104-input contract and add64 pinned train inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from bridge_input_contract import raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def expanded_schedule():
    result=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)])
    assert result.shape==(8192,8)
    assert torch.equal(torch.bincount(result.flatten(),minlength=128),torch.full((128,),512))
    return result

def expanded_items(rows,data_root,manifest,reference,checks,comparisons):
    base=Path(__file__).resolve().parent.parent/'bridge_growth64_v1'
    assert digest_bytes((base/'manifest.json').read_bytes())==manifest['growth']['retained64_manifest_sha256']
    assert digest_bytes((base/'growth_input_contract.py').read_bytes())==manifest['growth']['retained64_contract_sha256']
    frozen=json.loads((base/'manifest.json').read_text())
    kept=rows[:64]+rows[128:]
    assert kept==frozen['train']+frozen['development']
    assert manifest['config']==frozen['config']
    spec=importlib.util.spec_from_file_location('frozen_growth64_inputs',base/'growth_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.expanded_items(kept,data_root,frozen,reference,checks,comparisons)
    directory=Path(__file__).resolve().parent/'new_reference'
    for name,h in manifest['growth']['new_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes())==h
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['rows']==rows[64:128]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    added=[]
    with np.load(directory/'arrays.npz',allow_pickle=False) as z:
        for i,row in enumerate(rows[64:128]):
            path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][i],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            hashes={k:digest_bytes(v.tobytes()) for k,v in arrays.items()}
            for k in arrays:assert digest_bytes(z[k][i].tobytes())==metadata['processed_sha256'][k][i]
            diffs={k:require_numeric_match(v,z[k][i],k,str(row)) for k,v in arrays.items()}
            checks.append(dict(image_sha256=hashes['images'],target_sha256=hashes['targets'],current_sha256=hashes['currents'],raw_identity=raw))
            comparisons.append(dict(index=104+i,raw_identity_exact=True,differences=diffs))
            added.append(item)
    return old[:64]+added+old[64:]
