"""Package only the demonstrated context-execution restoration, keeping the test frozen."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_heldout_context_repair_v1'
OLD=ROOT/'cluster_packages/bridge_heldout_evaluation_v1'
DIAG=ROOT/'cluster_packages/bridge_heldout_replay_probe_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (PACK/'repair_contract.json').exists(),'Keep submitted repair immutable'
    original=read(OLD/'protocol.json')
    contract=dict(date='2026-10-17',failure_job='208005',diagnostic_job='208034',
        original_protocol_sha256=sha(OLD/'protocol.json'),original_entry_sha256=sha(OLD/'run_evaluation.py'),original_assets_sha256=sha(OLD/'assets.json'),
        diagnostic_entry_sha256=sha(DIAG/'run_probe.py'),diagnostic_contract_sha256=sha(DIAG/'probe_contract.json'),
        unchanged_replay_guard=original['replay_guard'],
        change='Wrap original context body with inference_mode(False)+no_grad and original prefix parameter requires_grad flags. No changes to model values, source files, CLIP encoding, batch sizes, DDPM, RNG rules, metrics, test rows or tolerances.',
        evidence_required='Independently check all16 saved diagnostic arrays against old reference arrays and recorded differences. Original-style cases exact0 on both models; failed-style cases reproduce >1e-4 prediction mismatch. Failed208005 report must confirm no new test scores.',
        actual_gpu_diagnostic_report_downloaded_locally=False,
        automatic_runtime_diagnostic_array_check_required=True,
        no_training=True,new_data_or_weights=False,original_package_modified=False,original_atomic_claim_preserved=True,
        limits='Identifies and restores a joint context-execution mode difference in the NEW evaluation entry. Does not isolate individual backend kernel or explain historical generalization error. New13episode scores still unknown.')
    write(PACK/'repair_contract.json',contract)
    for name in ['run_fixed.py','evaluate_fixed.sh']:assert (PACK/name).is_file()
    tree=ast.parse((PACK/'run_fixed.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in {'backward','step','zero_grad','diffusion_loss','train'}
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    spec=importlib.util.spec_from_file_location('context_repair_under_test',PACK/'run_fixed.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=torch.nn.Module()
    for name in ['adapter','diffusion_decoder','history_projection','state_projection','frozen_head']:
        model.add_module(name,torch.nn.Linear(2,2))
    model.eval();before={n:v.clone() for n,v in model.state_dict().items()}
    prefixes=('adapter.','diffusion_decoder.','history_projection.','state_projection.')
    def original_body(m,visual,language,mask,state):
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        assert all(v.requires_grad==n.startswith(prefixes) for n,v in m.named_parameters())
        return m.adapter(visual)
    wrapped=module.original_context_wrapper(original_body,prefixes)
    x=torch.tensor([[.2,-.8],[.3,.7]],dtype=torch.float32)
    for n,v in model.named_parameters():v.requires_grad_(n.startswith(prefixes))
    with torch.no_grad():expected=original_body(model,x,None,None,None)
    for v in model.parameters():v.requires_grad_(False)
    with torch.inference_mode():
        inference_input=x+0
        assert torch.is_inference(inference_input)
        actual=wrapped(model,inference_input,None,None,None)
        assert torch.is_inference_mode_enabled(), 'Caller scope must be restored'
    assert torch.equal(actual,expected) and not torch.is_inference(actual) and not actual.requires_grad
    assert all(torch.equal(v,before[n]) for n,v in model.state_dict().items()) and all(v.grad is None for v in model.parameters())
    # Actual original helper exposes the same unchanged body; imports only, no model/CLIP load.
    old_spec=importlib.util.spec_from_file_location('original_heldout_helper_under_test',OLD/'run_evaluation.py')
    old_module=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_module)
    assert callable(old_module.contexts.__wrapped__)
    write(PACK/'local_validation.json',dict(passed=True,entry_sha256=sha(PACK/'run_fixed.py'),repair_contract_sha256=sha(PACK/'repair_contract.json'),
        ast_no_training_calls=True,outer_inference_mode_explicitly_disabled_inside_context=True,original_prefix_flags_restored=True,
        caller_inference_scope_restored=True,ordinary_no_grad_context_tensor_returned=True,toy_cpu_output_exact_and_parameters_unchanged=True,
        original_context_helper_unwrap_available=True,actual_model_loaded=False,actual_neural_inference=False,gpu_used=False,
        original_test_protocol_and_entry_unchanged=True,cluster_diagnostic_array_audit_and_both_strict_replays_still_required=True))
    files=sorted(p for p in PACK.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (PACK/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in files),encoding='utf-8',newline='\n')
    archive=PACK.with_suffix('.zip');assert not archive.exists()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[PACK/'SHA256SUMS']:z.write(path,path.relative_to(PACK.parent).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in files+[PACK/'SHA256SUMS']:assert z.read(path.relative_to(PACK.parent).as_posix())==path.read_bytes()
    assert archive.stat().st_size<100*1024
    PACK.with_suffix('.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(dict(bytes=archive.stat().st_size,sha256=sha(archive),protocol_sha256=contract['original_protocol_sha256'],original_replay_guard_unchanged=True),indent=2))

if __name__=='__main__':main()
