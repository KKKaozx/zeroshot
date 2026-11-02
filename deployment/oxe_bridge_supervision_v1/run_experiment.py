"""Preserve the archived main training path, instrument one supervised term."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BUNDLE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_source():
    path=BUNDLE/'code/step_training_runner.py';source=path.read_text()
    changes=[
        ('    end = validate_options(args, api)\n','    end = SUPERVISION.limit_end(validate_options(args, api))\n'),
        ('    json.dumps(contract, sort_keys=True, allow_nan=False)\n',
         '    SUPERVISION.bind(model,api,contract,plan,dataset)\n    json.dumps(contract, sort_keys=True, allow_nan=False)\n'),
        ("        history, validation = extra['updates'], extra['validation']\n",
         "        history, validation = extra['updates'], extra['validation']\n        SUPERVISION.restore_state(extra['source_supervision'],plan.completed_updates)\n"),
        ("'language_counts': dict(language_counts)})", "'language_counts': dict(language_counts),'source_supervision':SUPERVISION.state_dict()})"),
        ('        indices = plan.begin_batch()\n','        indices = plan.begin_batch()\n        SUPERVISION.begin(indices,plan.completed_updates)\n'),
        ('loss, pose, gripper = api.policy_loss(model, output, actions, criterion,',
         'loss, pose, gripper = SUPERVISION.loss(model, output, actions, criterion,'),
        ('        optimizer.zero_grad(set_to_none=True)\n        loss.backward()\n',
         '        optimizer.zero_grad(set_to_none=True)\n        SUPERVISION.audit_initial(model,optimizer)\n        loss.backward()\n'),
        ('        plan.commit_batch()\n','        plan.commit_batch()\n        SUPERVISION.commit(plan.completed_updates)\n'),
        ("'gradient_norm': float(norm), 'valid_dimensions':", "'gradient_norm': float(norm), 'kept_bridge_windows': SUPERVISION.last_committed_keep, 'valid_dimensions':"),
        ("    (out / 'step_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')",
         "    report['supervision_counterfactual']=SUPERVISION.state_dict()\n    (out / 'step_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')"),
    ]
    for before,after in changes:
        assert source.count(before)==1,('Archived source hook differs',before)
        source=source.replace(before,after)
    return path,source


def options_for(entry,root,output,end,resume):
    reference=read(BUNDLE/'reference_contract.json')
    options=entry.parse_args([])
    for key,value in reference['arguments'].items():setattr(options,key,value)
    inputs=root/'oxe_dino_main_preflight_v1/inputs'
    values=dict(output_dir=str(output),step_resume=str(resume) if resume else None,stop_after_updates=end,
        checkpoint_every_updates=2048,log_interval=256,cache_dir=None,
        dataset_dir=str(root/'data/oxe_nominal_pool_v1'),split_manifest=str(inputs/'split_manifest.json'),
        original_instruction_catalog=str(inputs/'instruction_catalog.json'),language_bank=None,
        language_table_groups=None,oxe_nominal_source_map=None,
        oxe_nominal_targets=str(inputs/'train'),oxe_nominal_validation_targets=str(inputs/'validation'),
        oxe_formal_protocol=str(BUNDLE/'source_protocol.json'),oxe_preflight_binding=str(BUNDLE/'source_binding.json'),
        model_name=str(root/'baseline_setup/clip-vit-large-patch14-v1'),
        dino_assets=str(root/'dinov2_large_frozen_assets_20261023'))
    for key,value in values.items():setattr(options,key,value)
    return options


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    p.add_argument('--mode',choices=['preflight','fit'],required=True)
    p.add_argument('--stage',type=int,choices=[1,2],default=1)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    reference=read(BUNDLE/'reference_contract.json');protocol=read(BUNDLE/'protocol.json')
    for name,digest in protocol['bundle_inputs'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in reference['source_code_sha256'].items():assert sha(BUNDLE/'code'/name)==digest,name
    assert sha(BUNDLE/'source_protocol.json')==reference['offline_training_protocol']['sha256']
    assert sha(BUNDLE/'source_binding.json')==reference['offline_training_protocol']['cluster_preflight_binding_sha256']
    if args.mode=='fit' and args.preflight is None:raise ValueError('Successful new engineering preflight required')
    if args.mode=='preflight':
        output=root/'runs'/('oxe-bridge-supervision-preflight-'+os.environ.get('SLURM_JOB_ID','CHECK_ONLY'))
        assert not output.exists();end=32768
    else:
        output=root/'runs/oxe-bridge-supervision-s42';end=32768 if args.stage==1 else 65536
        assert not (output/'final_policy.pt').exists(),'Completed diagnostic policy already exists'
    import torch
    resume=None;completed=0
    if args.mode=='fit' and (output/'latest_step.pt').exists():
        latest=torch.load(output/'latest_step.pt',map_location='cpu',weights_only=False,mmap=True)
        assert latest['contract']['supervision_counterfactual']['protocol_sha256']==sha(BUNDLE/'protocol.json')
        completed=latest['plan']['completed_updates'];resume=output/'latest_step.pt';del latest
    assert completed<end,'Requested endpoint already completed'
    if args.mode=='fit' and args.stage==2:assert completed>=32768,'Stage1 must complete first'
    if os.name=='posix':
        used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
        assert used+8*1024**3<200_000_000_000 and shutil.disk_usage(root).free>8*1024**3,'8GiB storage reserve required'
    sys.path.insert(0,str(BUNDLE/'code'))
    import train as entry
    options=options_for(entry,root,output,end,resume)
    from oxe_nominal_dataset import validate_options
    validate_options(options)
    if args.check_only:
        print(json.dumps(dict(passed=True,mode=args.mode,stage=args.stage,restored_update=completed,
            actual_segment_end=2 if args.mode=='preflight' else end,total_horizon=65536,
            model_loaded=False,training_submitted=False),indent=2));return
    assert torch.cuda.is_available(),'Allocated CUDA required'
    bound=reference['offline_training_protocol']['cluster_preflight']
    assert torch.cuda.get_device_name()==bound['gpu']=='NVIDIA A40'
    assert str(torch.__version__)==bound['torch_version'] and torch.version.cuda==bound['cuda_runtime']
    for name,version in bound['packages'].items():assert importlib.metadata.version(name)==version,name
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(reference['execution']['intraop_threads'])
    torch.set_num_interop_threads(reference['execution']['interop_threads'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==reference['execution']['cublas_workspace_config']
    path,effective=effective_source()
    from supervision import Supervisor
    hook=Supervisor(BUNDLE,args.mode,hashlib.sha256(effective.encode()).hexdigest(),args.preflight)
    if args.mode=='preflight':hook.engineering_end=1
    import step_training_runner
    step_training_runner.__dict__['SUPERVISION']=hook
    exec(compile(effective,str(path),'exec'),step_training_runner.__dict__)
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        entry.train(options)
        resumed_from=None
        if args.mode=='preflight':
            assert hook.done==1 and hook.first_replay['passed']
            first_initial=hook.initial_sha
            shutil.copyfile(output/'step_report.json',output/'engineering_update1_report.json')
            # A real checkpoint boundary: reconstruct through the same original
            # entry, restore AdamW/plan/RNG and commit exactly one more update.
            # Neither model nor optimizer from this probe enters formal fit.
            hook=Supervisor(BUNDLE,args.mode,hashlib.sha256(effective.encode()).hexdigest())
            hook.engineering_end=2
            step_training_runner.__dict__['SUPERVISION']=hook
            options.step_resume=str(output/'latest_step.pt')
            entry.train(options)
            assert hook.initial_sha==first_initial
            resumed_from=1
        expected=2 if args.mode=='preflight' else end
        assert hook.done==expected and hook.first_replay['passed']
        report=read(output/'step_report.json')
        assert report['completed_updates']==expected and report['frozen_state_unchanged']
        receipt=dict(passed=True,mode=args.mode,stage=args.stage,completed_updates=expected,total_horizon=65536,
            protocol_sha256=sha(BUNDLE/'protocol.json'),initial_learned_sha256=hook.initial_sha,
            engineering_resumed_from_update=resumed_from,
            first_reference_replay=hook.first_replay,source_supervision_state=hook.state_dict(),
            gpu=torch.cuda.get_device_name(),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            elapsed_seconds=time.monotonic()-tick,extra_training_or_libero_submitted=False,
            scope='Diagnostic supervision intervention only; not a main thesis model or task-success verdict')
        name='diagnostic_receipt.json' if args.mode=='preflight' else f'stage{args.stage}_receipt.json'
        if args.mode=='fit' and args.stage==2:
            assert hook.bridge_draws==131244 and hook.all_draws==524276 and hook.no_bridge==6617
            from completed_policy_export import export_completed
            receipt['policy_export']=export_completed(output/'latest_step.pt',discard_completed_optimizer=True)
        (output/name).write_text(json.dumps(receipt,indent=2)+'\n')
        print('BRIDGE_SUPERVISION_DIAGNOSTIC_COMPLETE',json.dumps(receipt,indent=2),flush=True)
    finally:
        if output.exists():(output/'effective_step_training_runner.py').write_text(effective)


if __name__=='__main__':main()
