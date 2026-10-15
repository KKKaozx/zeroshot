"""Extend the verified 72-input contract with 32 pinned train inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from bridge_input_contract import load_verified_items, raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def expanded_schedule():
    batches = [torch.randperm(64, generator=torch.Generator().manual_seed(440001+c)).reshape(8,8) for c in range(512)]
    result = torch.cat(batches)
    assert result.shape == (4096,8)
    assert torch.equal(torch.bincount(result.flatten(), minlength=64), torch.full((64,),512))
    return result

def expanded_items(rows, data_root, manifest, reference, checks, comparisons):
    old = load_verified_items(rows[:32]+rows[64:], data_root, manifest, reference, checks, comparisons)
    directory = Path(__file__).resolve().parent/'added_reference'
    for name,h in manifest['growth']['added_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes()) == h
    metadata = json.loads((directory/'metadata.json').read_text())
    assert metadata['rows'] == rows[32:64]
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16; reader.bridge_gripper_policy='reverse_scan_valid_steps_v2'; reader.bridge_current_gripper='continuous'
    added=[]
    with np.load(directory/'arrays.npz', allow_pickle=False) as z:
        for i,row in enumerate(rows[32:64]):
            path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][i],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split()) == ' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            hashes={k:digest_bytes(v.tobytes()) for k,v in arrays.items()}
            for k in arrays: assert digest_bytes(z[k][i].tobytes())==metadata['processed_sha256'][k][i]
            diffs={k:require_numeric_match(v,z[k][i],k,str(row)) for k,v in arrays.items()}
            checks.append(dict(image_sha256=hashes['images'],target_sha256=hashes['targets'],current_sha256=hashes['currents'],raw_identity=raw))
            comparisons.append(dict(index=72+i,raw_identity_exact=True,differences=diffs))
            added.append(item)
    return old[:32]+added+old[32:]
