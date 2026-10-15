"""Verify frozen128 inputs, then substitute39 individually pinned inputs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from bridge_input_contract import raw_identity, require_raw_identity, require_numeric_match, digest_bytes
from dataset import UnifiedRobotDataset

def audit_index(i):
    return i if i<32 else (i+40 if i<128 else i-96)

def frozen_base(manifest):
    base=Path(__file__).resolve().parent.parent/'bridge_growth128_v1'
    for name,h in manifest['window_coverage']['base_files_sha256'].items():
        assert digest_bytes((base/name).read_bytes())==h,name
    frozen=json.loads((base/'manifest.json').read_text())
    spec=importlib.util.spec_from_file_location('frozen_growth128_inputs',base/'growth128_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return frozen,module

def expanded_schedule():
    # Exactly the baseline row-index schedule, including512 draws/window.
    base=Path(__file__).resolve().parent.parent/'bridge_growth128_v1'
    spec=importlib.util.spec_from_file_location('frozen_growth128_schedule',base/'growth128_input_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.expanded_schedule()

def expanded_items(rows,data_root,manifest,reference,checks,comparisons):
    frozen,module=frozen_base(manifest)
    assert manifest['config']==frozen['config'] and manifest['development']==frozen['development']
    changed=manifest['window_coverage']['changed_row_indices']
    assert len(changed)==len(set(changed))==39
    assert [i for i,(a,b) in enumerate(zip(manifest['train'],frozen['train'])) if a!=b]==changed
    items=module.expanded_items(frozen['train']+frozen['development'],data_root,frozen,reference,checks,comparisons)
    directory=Path(__file__).resolve().parent/'changed_reference'
    for name,h in manifest['window_coverage']['new_reference_sha256'].items():
        assert digest_bytes((directory/name).read_bytes())==h
    metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['rows']==[rows[i] for i in changed]
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16;reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    with np.load(directory/'arrays.npz',allow_pickle=False) as z:
        for j,i in enumerate(changed):
            row=rows[i];path=str(Path(data_root)/row['shard'])
            example=UnifiedRobotDataset._load_tfrecord_example(path,row['record_index'])
            raw=raw_identity(example,row['start_index']);require_raw_identity(raw,metadata['raw_inputs'][j],str(row))
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=path,record_index=row['record_index'],start_index=row['start_index']))
            assert ' '.join(item[0].lower().split())==' '.join(row['task'].lower().split())
            arrays=dict(images=item[1].numpy(),targets=item[3].numpy(),currents=item[2].numpy())
            for k in arrays:assert digest_bytes(z[k][j].tobytes())==metadata['processed_sha256'][k][j]
            diffs={k:require_numeric_match(v,z[k][j],k,str(row)) for k,v in arrays.items()}
            h={k:digest_bytes(v.tobytes()) for k,v in arrays.items()};a=audit_index(i)
            checks[a]=dict(image_sha256=h['images'],target_sha256=h['targets'],current_sha256=h['currents'],raw_identity=raw)
            comparisons[a]=dict(index=a,raw_identity_exact=True,differences=diffs)
            items[i]=item
    return items
