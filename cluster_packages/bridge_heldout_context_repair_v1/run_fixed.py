"""Restore original context execution only; reuse the untouched frozen heldout evaluator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'bridge_heldout_evaluation_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def original_context_wrapper(function,prefixes):
    def restored(model,visual,language,mask,state):
        # The unchanged caller still has an outer inference_mode. Explicitly leave it.
        with torch.inference_mode(False),torch.no_grad():
            for name,param in model.named_parameters():param.requires_grad_(name.startswith(prefixes))
            assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            assert not model.training and all(v.grad is None for v in model.parameters())
            result=function(model,visual,language,mask,state)
            assert not result.requires_grad and not torch.is_inference(result)
            return result
    return restored

def verify_diagnostic(project,contract):
    assert sha(ORIGINAL/'protocol.json')==contract['original_protocol_sha256']
    assert sha(ORIGINAL/'run_evaluation.py')==contract['original_entry_sha256']
    assert sha(ORIGINAL/'assets.json')==contract['original_assets_sha256']
    assert not (ORIGINAL/'evaluation_claim.json').exists(),'New test was already claimed; do not repeat scoring'
    protocol=read(ORIGINAL/'protocol.json');assets=read(ORIGINAL/'assets.json')
    diagnostic=project/'runs/bridge-replay-diag-208034';report=read(diagnostic/'report.json')
    assert report['execution_checks_passed'] and report['parameters_unchanged'] and report['clip_unchanged']
    assert not report['new_holdout_scored'] and not report['new_holdout_predictions_generated'] and not report['training']
    assert report['old_windows']==4 and report['new_windows']==0 and not report['original_numeric_tolerances_changed']
    assert report['entry_sha256']==contract['diagnostic_entry_sha256']==sha(diagnostic/'runner_snapshot.py')
    assert report['probe_contract_sha256']==contract['diagnostic_contract_sha256']==sha(diagnostic/'probe_contract_snapshot.json')
    failed=read(diagnostic/'failed_208005_report.json')
    assert sha(diagnostic/'failed_208005_report.json')==report['failed_report_sha256']
    assert not failed['test_predictions_generated'] and not failed['models'] and not failed['execution_checks_passed']
    assert failed['entry_sha256']==contract['original_entry_sha256']
    assert protocol['replay_guard']==contract['unchanged_replay_guard']
    proof={};files=[]
    cases=read(diagnostic/'probe_contract_snapshot.json')['cases']
    for fixed in protocol['models']:
        label=fixed['label'];reference=ORIGINAL/(label+'_replay.npz')
        assert sha(reference)==assets['files'][reference.name]
        with np.load(reference,allow_pickle=False) as z:old_pred=z['predictions'].copy();old_ctx=z['contexts'].copy()
        values={}
        for case in cases:
            name=case['name'];path=diagnostic/(label+'_'+name+'.npz')
            with np.load(path,allow_pickle=False) as z:
                pred=z['predictions'].copy();ctx=z['contexts'].copy()
            assert pred.shape==(3,4,16,8) and ctx.shape==(4,1024) and np.isfinite(pred).all() and np.isfinite(ctx).all()
            pd=float(np.abs(pred.astype(float)-old_pred.astype(float)).max());cd=float(np.abs(ctx.astype(float)-old_ctx.astype(float)).max())
            recorded=report['models'][label][name]
            assert np.isclose(pd,recorded['prediction_difference']['max_abs'],rtol=0,atol=1e-12)
            assert np.isclose(cd,recorded['context_max_abs'],rtol=0,atol=1e-12)
            values[name]=dict(context_max_abs=cd,prediction_max_abs=pd)
            if case['context_style']=='original':assert pd==0 and cd==0,(label,name,'Original-style equality required')
            else:assert pd>protocol['replay_guard']['prediction_atol'] and cd<=protocol['replay_guard']['context_atol'],(label,name)
            files.append(dict(file=path.name,sha256=sha(path)))
        proof[label]=values
    return dict(verified=True,diagnostic_job='208034',report_sha256=sha(diagnostic/'report.json'),
        checked_prediction_files=files,comparisons=proof,old_tolerances_unchanged=True,
        conclusion='Original context-execution bundle reproduces exactly on both frozen checkpoints; failed bundle reproduces discrepancy. CLIP encoding styles and tested state array shapes do not explain this replay difference.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True);parser.add_argument('--clip',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['evaluate'],default='evaluate')
    args=parser.parse_args();assert not args.output.exists(),'Preserve previous execution evidence'
    evidence=dict(repair_entry_sha256=sha(Path(__file__)),repair_contract_sha256=sha(HERE/'repair_contract.json'),
        execution_passed=False,new_training=False,original_protocol_changed=False,numeric_tolerance_changed=False)
    try:
        for line in (HERE/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1);assert sha(HERE/name)==expected
        contract=read(HERE/'repair_contract.json')
        evidence['diagnostic_evidence']=verify_diagnostic(args.project,contract)
        spec=importlib.util.spec_from_file_location('frozen_heldout_evaluator',ORIGINAL/'run_evaluation.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        assert callable(module.contexts.__wrapped__)
        module.contexts=original_context_wrapper(module.contexts.__wrapped__,module.PREFIXES)
        evidence['only_runtime_override']='contexts: restore original parameter flags and no_grad, explicitly disabling outer inference_mode'
        # Unchanged original main keeps input checks, restore checks, sampling, metrics,
        # both strict replay guards and the original atomic one-time test claim.
        module.main()
        evidence['execution_passed']=True
    except Exception as error:evidence['error']=repr(error);raise
    finally:
        args.output.mkdir(parents=True,exist_ok=True)
        write(args.output/'repair_entry.json',evidence)
        (args.output/'repair_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (args.output/'repair_contract_snapshot.json').write_bytes((HERE/'repair_contract.json').read_bytes())
        if (args.output/'report.json').exists():
            report=read(args.output/'report.json')
            report['context_execution_repair']=dict(entry_sha256=evidence['repair_entry_sha256'],contract_sha256=evidence['repair_contract_sha256'],
                diagnostic_job='208034',diagnostic_verified=evidence.get('diagnostic_evidence',{}).get('verified',False),
                underlying_entry_unchanged_sha256=contract['original_entry_sha256'],numeric_tolerance_changed=False)
            write(args.output/'report.json',report)
        print('CONTEXT REPAIR EVIDENCE',args.output/'repair_entry.json',flush=True)

if __name__=='__main__':main()
