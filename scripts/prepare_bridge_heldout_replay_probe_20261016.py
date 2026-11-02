"""Package a bounded old-window diagnostic after208005, without changing the frozen test."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_heldout_replay_probe_v1'
OLD=ROOT/'cluster_packages/bridge_heldout_evaluation_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (PACK/'probe_contract.json').exists(),'Preserve a submitted diagnostic package'
    cases=[dict(name='saved_context_sampler',tokens='saved_context',context_style='original',state_style='original'),
           dict(name='cached_original_state',tokens='cached',context_style='original',state_style='original'),
           dict(name='cached_failed_state',tokens='cached',context_style='original',state_style='failed'),
           dict(name='cached_failed_mode',tokens='cached',context_style='failed',state_style='failed')]
    for encoder in ['original','failed']:
        for context in ['original','failed']:
            cases.append(dict(name=f'fresh_{encoder}_encoder_{context}_context',tokens=encoder,context_style=context,state_style='failed'))
    assert len(cases)==8
    contract=dict(date='2026-10-17',purpose='Diagnose new evaluation entry mismatch on old4windows only; no new test inference.',
        original_protocol_sha256=sha(OLD/'protocol.json'),failed_entry_sha256=sha(OLD/'run_evaluation.py'),failure_job='208005',
        observed_terminal_max_abs_difference=0.00011682510375976562,terminal_threshold=0.0001,
        tolerance_changed=False,original_test_protocol_changed=False,original_models_or_weights_changed=False,
        cases=cases,case_definitions=dict(original='Context under torch.no_grad; original requires_grad flags on Adapter/decoder/history/state only.',
            failed='Context under torch.inference_mode; all parameter requires_grad=False, matching failed new entry.',
            encoder_original='CLIP encoding under torch.no_grad, same image2/text16 shapes.',encoder_failed='CLIP encoding under torch.inference_mode, same image2/text16 shapes.'),
        sampling=dict(steps=100,seeds=[1101,1102,1103],batch_size=2),optimizer=False,training=False,
        exact_duplicate_sampling='Reuse only when exact context tensor hash AND original/failed parameter-flag mode match; all other inputs/weights/seeds held fixed.',
        output='Save each case before comparing to unchanged old thresholds, even on mismatch; no automatic continuation or threshold relaxation.',
        limits='Sources establish two entry differences, not a proven root cause. Cached tokens, context computation mode, and state array shape are isolated here. This cannot explain the historical centimetre-scale generalization error by itself.')
    write(PACK/'probe_contract.json',contract)
    source=(PACK/'run_probe.py').read_text();tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in {'backward','step','zero_grad','diffusion_loss','train'},node.func.attr
    # Independently test the new component-wise difference utility without Torch, models or CUDA.
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='delta')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<delta unit>','exec'),env)
    reference=np.zeros((3,4,16,8),dtype=np.float64);reference[...,6]=1
    changed=reference.copy();changed[...,0]=.1;changed[...,3:7]*=-1
    d=env['delta'](changed,reference)
    assert np.isclose(d['xyz_max_component_difference_cm'],1.) and np.isclose(d['position_vector_max_difference_cm'],1.)
    assert d['rotation_max_difference_deg']==0 and d['gripper_max_abs']==0
    assert 'selection[\'rows\']' in source and "rows=selection['replay_rows']" in source
    assert 'new_holdout_scored=False' in source and 'evaluation_claim.json' in source
    assert not any(p.suffix in {'.pt','.npz'} for p in PACK.iterdir())
    write(PACK/'local_validation.json',dict(passed=True,entry_sha256=sha(PACK/'run_probe.py'),contract_sha256=sha(PACK/'probe_contract.json'),
        ast_and_no_training_calls_passed=True,component_units_and_quaternion_sign_test_passed=True,
        new_data_or_weights_in_package=False,new_neural_execution=False,gpu_used=False,cluster_replay_diagnostic_pending=True))
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
    print(json.dumps(dict(package=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))

if __name__=='__main__':main()
