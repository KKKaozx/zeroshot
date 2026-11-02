"""CPU-only independent checks on the new evaluation package and saved input proof."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_heldout_evaluation_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    path=PACK/'run_evaluation.py';tree=ast.parse(path.read_text())
    # No optimization API, loss call or training-mode activation in this new entry point.
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'backward','step','zero_grad','diffusion_loss','train'},node.func.attr
    local=read(ROOT/'reports/bridge_heldout_input_check_20261016/report.json')
    assert local['execution_checks_passed'] and local['input_contract_passed'] and not local['test_predictions_generated']
    assert not local['gpu_used'] and not local['model_loaded']
    source_alignment_only=False
    if local['entry_sha256']!=sha(path):
        # The post-check review aligned unexecuted GPU text batching16 with the old cache.
        # Reuse only the actually unchanged CPU input/metric paths; cluster replay remains required.
        prior=ROOT/'reports/bridge_heldout_input_check_20261016/runner_snapshot.py'
        assert sha(prior)==local['entry_sha256']
        old_tree=ast.parse(prior.read_text());new_tree=ast.parse(path.read_text())
        old_functions={n.name:ast.dump(n,include_attributes=False) for n in old_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        new_functions={n.name:ast.dump(n,include_attributes=False) for n in new_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert old_functions.keys()==new_functions.keys() and [k for k in old_functions if old_functions[k]!=new_functions[k]]==['encode']
        assert 'range(0,len(items),16)' in path.read_text() and 'lo:lo+16' in path.read_text()
        source_alignment_only=True
    assert len(local['input_checks'])==64 and all(v['raw_exact'] for v in local['input_checks'])
    assert all(max(v['max_absolute_differences'].values())==0 for v in local['input_checks'])
    p=read(PACK/'protocol.json');selection=read(PACK/'selection.json');assets=read(PACK/'assets.json')
    assert local['protocol_sha256']==sha(PACK/'protocol.json')==assets['protocol_frozen_before_decoding_sha256']
    assert local['assets_sha256']==sha(PACK/'assets.json')
    assert p['counts']==dict(episodes=13,windows=60,replay_windows=4,tasks=5)
    assert len(selection['rows'])==60 and len(selection['episodes'])==13
    previous=read(ROOT/'reports/bridge_expansion_source_audit_verified_v1.json')
    old_payloads={e['payload_sha256'] for e in previous['episodes']}
    new_payloads={e['payload_sha256'] for e in assets['episodes']}
    assert len(new_payloads)==13 and not old_payloads&new_payloads
    spec=importlib.util.spec_from_file_location('heldout_metrics_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=selection['rows']
    with np.load(PACK/'input_reference.npz',allow_pickle=False) as z:
        target=z['targets'][4:].copy()
    assert target.shape==(60,16,8)
    # Unit invariants test physical units, endpoint indexing and quaternion sign invariance.
    m=module.summary(target[None],target,rows)
    assert m['episodes']==13 and len(m['per_task'])==5 and len(m['per_target'])==16
    assert m['window_weighted']['path_position_cm']==0 and m['window_weighted']['path_rotation_deg']<1e-5
    changed=target.copy();changed[...,:3]+=np.array([.1,0,0]);changed[...,3:7]*=-1
    m=module.summary(changed[None],target,rows)
    assert np.isclose(m['window_weighted']['path_position_cm'],1.,atol=1e-6)
    assert m['window_weighted']['path_rotation_deg']<1e-5
    endpoint=target.copy();endpoint[:,-1,0]+=.1
    m=module.summary(endpoint[None],target,rows)
    assert np.isclose(m['window_weighted']['endpoint_position_cm'],1.,atol=1e-6)
    assert np.isclose(m['window_weighted']['path_position_cm'],1/16,atol=1e-6)
    # Both controls' original4-window arrays bind replay order and the physical target values.
    for fixed in p['models']:
        with np.load(PACK/(fixed['label']+'_replay.npz'),allow_pickle=False) as z, np.load(ROOT/'reports'/fixed['report_directory']/'reduced_u8192_development.npz',allow_pickle=False) as old:
            assert np.array_equal(z['predictions'],old['predictions'][:,:4]) and np.array_equal(z['contexts'],old['contexts'][:4])
    record=dict(passed=True,protocol_sha256=sha(PACK/'protocol.json'),assets_sha256=sha(PACK/'assets.json'),entry_sha256=sha(path),
        actual_cpu_input_check_report_sha256=sha(ROOT/'reports/bridge_heldout_input_check_20261016/report.json'),
        all64_inputs_raw_and_processed_exact_locally=True,new13_payloads_distinct_from_all415_training_and_development_payloads=True,
        metrics_unit_tests_passed=True,metrics_test_predictions='Synthetic arrays only, never neural predictions.',no_optimizer_or_training_calls_in_entry=True,
        frozen_replay_references_exact=True,gpu_used=False,neural_model_executed=False,checkpoints_downloaded=False,
        post_input_check_change_only_unexecuted_gpu_encoder_text_batch2_to_original16=source_alignment_only,
        unchanged_cpu_input_and_metric_paths_verified=True,
        cluster_checkpoint_replay_still_required=True)
    (PACK/'local_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
