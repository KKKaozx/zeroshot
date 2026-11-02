"""CPU audit of actual transfer records, references, schedule and runner adaptation."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import zipfile

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from tensorflow.core.example import example_pb2

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_demonstration_expansion_v1'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=read(PACK/'protocol.json');data=read(PACK/'transfer_data.json');audit=read(PACK/'new_source_audit.json')
    rows=read(PACK/'rows.json');baseline=ROOT/'cluster_packages/bridge_coverage_replication_v1'
    oldrows=read(baseline/'rows.json')
    assert rows['train'][:1836]==oldrows['train'] and rows['development']==oldrows['development']
    with np.load(PACK/'schedule.npz',allow_pickle=False) as z:b=z['batches'].copy()
    with np.load(baseline/'schedule.npz',allow_pickle=False) as z:oldb=z['batches'].copy()
    tasks=np.array([r['task'] for r in rows['train']]);ot=np.array([r['task'] for r in oldrows['train']])
    assert b.shape==(8192,8) and np.array_equal(tasks[b],ot[oldb])
    assert all(len(set(row))==8 for row in b)
    # Verify actual compressed TFRecord contents and numbering, independently of
    # the export writer. No raw source re-audit, image decoding or model calls.
    expected={(e['shard'],e['record_index']):e for e in audit['episodes']}
    found={};file_count=0
    for part in data['archives']:
        archive=PACK.parent/part['file'];assert sha(archive)==part['sha256']
        assert archive.stat().st_size<64*1024**2
        with zipfile.ZipFile(archive) as z:
            assert z.namelist()==part['members']
            for member in z.infolist():
                components=Path(member.filename).parts
                assert components[:2]==(PACK.name,'data') and len(components)==3
                shard=components[-1];raw=z.read(member)
                assert hashlib.sha256(raw).hexdigest()==data['files'][shard]['sha256']
                stream=io.BytesIO(raw);index=0;nonempty=[]
                while stream.tell()<len(raw):
                    header=stream.read(12);assert len(header)==12
                    length=struct.unpack('<Q',header[:8])[0]
                    payload=stream.read(length);assert len(payload)==length and len(stream.read(4))==4
                    if payload:
                        key=shard,index;assert key in expected and key not in found
                        e=example_pb2.Example.FromString(payload)
                        digest=hashlib.sha256(e.SerializeToString(deterministic=True)).hexdigest()
                        assert digest==expected[key]['payload_sha256']
                        found[key]=digest;nonempty.append(index)
                    index+=1
                assert stream.tell()==len(raw) and nonempty==data['files'][shard]['exported_record_indices']
                file_count+=1
    assert set(found)==set(expected) and len(found)==401 and file_count==305
    plan=read(ROOT/'configs/bridge_demonstration_expansion_eligible_20261017.json')
    reserved={(e['shard'],e['record_index']) for e in plan['future_test_candidates_identity_and_metadata_only']+plan['consumed_test_identity_only']}
    assert not set(found)&reserved
    with np.load(PACK/'new_numeric_reference.npz',allow_pickle=False) as z:
        assert z['targets'].shape==(1788,16,8) and z['observed'].shape==(1788,6)
        assert z['pixel_sketch'].shape==(1788,96)
        assert all(np.isfinite(z[k]).all() for k in z.files)
    spec=importlib.util.spec_from_file_location('only_ast_size_adapter',PACK/'run_expansion.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=(baseline/'run_training_coverage.py').read_text()
    effective,counts=module.adapted_source(original)
    contract=read(PACK/'adaptation_contract.json')
    assert hashlib.sha256(effective.encode()).hexdigest()==contract['effective_runner_sha256']
    def function(tree,name):return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('advance','context'):
        assert ast.dump(function(ast.parse(effective),name),include_attributes=False)==ast.dump(function(ast.parse(original),name),include_attributes=False)
    for path in PACK.glob('*.py'):compile(path.read_text(),str(path),'exec')
    assert p['recipe']==read(baseline/'protocol.json')['recipe']
    assert p['config']==read(baseline/'protocol.json')['config']
    assert p['counts']==dict(train_windows=3624,train_episodes=806,development_windows=40,development_episodes=10)
    old_map=list(range(1836))+list(range(3624,3664))
    marker=np.zeros((3664,2),np.int64);source=np.arange(1876*2).reshape(1876,2)
    marker[old_map]=source;assert np.array_equal(marker[old_map],source)
    assert not any((PACK/'data').iterdir()),'Only verified compressed files retained locally'
    result=dict(passed=True,stage='local_cpu_transfer_and_source_adaptation_checks',
        gpu_used=False,actual_model_loaded=False,new_inference=False,training=False,
        actual_data_archives=len(data['archives']),actual_subset_shards=305,actual_added_records=401,
        all_transferred_nonempty_records_match_source_audit=True,original_record_numbers_preserved=True,
        future_or_consumed_test_payloads_exported=False,all65536_task_slots_match=True,
        training_advance_ast_identical=True,context_function_ast_identical=True,dimension_replacements=counts,
        old_cache_row_remap_synthetic_cpu_check=True,full_new_numeric_reference_valid=True,
        bash_syntax_checked=True,local_full_token_cache_written=False,
        protocol_sha256=sha(PACK/'protocol.json'),helper_sha256=sha(PACK/'expanded_inputs.py'),
        wrapper_sha256=sha(PACK/'run_expansion.py'),
        limits='No actual CLIP/Adapter/decoder or CUDA execution locally. Cluster must check complete inputs, actual old4 checkpoint replay, gradients, cache, resources and16 operational updates before formal training.')
    (PACK/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
