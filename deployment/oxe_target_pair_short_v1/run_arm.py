"""Native-target original-main prefix. Fresh model for each fixed role."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from protocols import HERE,read,sha,install


def options(entry,root,output,target):
    args=entry.parse_args([])
    for key,value in read(HERE/'reference_contract.json')['arguments'].items():setattr(args,key,value)
    inputs=root/'oxe_dino_main_preflight_v1/inputs'
    values={'diffusion_prediction_type':target,'output_dir':str(output),'stop_after_updates':8192,'step_resume':None,
        'checkpoint_every_updates':8192,'log_interval':256,'cache_dir':None,
        'dataset_dir':str(root/'data/oxe_nominal_pool_v1'),'split_manifest':str(inputs/'split_manifest.json'),
        'original_instruction_catalog':str(inputs/'instruction_catalog.json'),'oxe_nominal_targets':str(inputs/'train'),
        'oxe_nominal_validation_targets':str(inputs/'validation'),'oxe_nominal_source_map':None,
        'oxe_formal_protocol':str(HERE/f'protocol_{target}.json'),'oxe_preflight_binding':str(HERE/'source_binding.json'),
        'model_name':str(root/'baseline_setup/clip-vit-large-patch14-v1'),
        'dino_assets':str(root/'dinov2_large_frozen_assets_20261023')}
    for key,value in values.items():setattr(args,key,value)
    return args


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--target',choices=['epsilon','sample'],required=True);a=ap.parse_args();root=a.root.resolve()
    assert not a.output.exists() and os.environ.get('SLURM_JOB_ID')
    sys.path.insert(0,str(HERE/'code'))
    import torch
    import tensorflow as tf
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
    reference=read(HERE/'reference_contract.json');bound=reference['offline_training_protocol']['cluster_preflight']
    assert str(torch.__version__)==bound['torch_version'] and torch.version.cuda==bound['cuda_runtime']
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.set_num_interop_threads(48);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    import train as entry
    install()
    from oxe_nominal_dataset import validate_options
    args=options(entry,root,a.output,a.target);validate_options(args)
    # Check the archived LR prefix before loading encoders or executing updates.
    # A host/libm variant must fail early rather than after a full short arm.
    import numpy as np
    from step_training_state import UpdateSchedule
    lr=UpdateSchedule(total_updates=65536,base_lrs=[args.learning_rate,args.gripper_learning_rate or args.learning_rate],
        kind=args.lr_schedule,warmup_updates=args.warmup_updates)
    with np.load(HERE/'reference_prefix.npz') as z:
        assert all(lr.rates(i)==z['lrs'][i].tolist() for i in range(8192)),'Archived LR prefix differs before model construction'
    from instrument import loop_source,method_source
    path,loop=loop_source();mpath,method=method_source()
    from hook import Hook
    h=Hook(a.target,hashlib.sha256(loop.encode()).hexdigest(),hashlib.sha256(method.encode()).hexdigest())
    import models,step_training_runner
    models.__dict__['DIAG_HOOK']=h
    exec(compile(method,str(mpath),'exec'),models.__dict__)
    models.RobotAdapterModel.diffusion_loss=models.__dict__['diffusion_loss']
    step_training_runner.__dict__['DIAG_HOOK']=h
    exec(compile(loop,str(path),'exec'),step_training_runner.__dict__)
    from resume_constants import training_constant_constructor
    from completed_policy_export import model_sha,contract_sha
    template=json.loads(json.dumps(reference));template['config']['model']['diffusion_prediction_type']=a.target
    events=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def record(event):
        events.append(event);print('EXACT_DIAGNOSTIC_FROZEN_IDENTITY',a.target,json.dumps(event),flush=True)
    try:
        with training_constant_constructor(models.RobotAdapterModel,contract=template,profile_path=HERE/f'profile_{a.target}.json',
                learned_state=entry.trainable_state_dict,on_restored=record):
            entry.train(args)
        assert h.done==h.captures==8192 and h.first_replay['passed'] and len(events)==1
        report=read(a.output/'step_report.json');assert report['diagnostic_prefix_completed'] and report['frozen_state_unchanged']
        assert report['validation']==[],'No extra validation losses or model selection in this prefix'
        state=torch.load(a.output/'latest_step.pt',map_location='cpu',weights_only=False,mmap=True)
        assert state['contract_sha256']==contract_sha(state['contract'])
        assert state['contract']['config']['model']['diffusion_prediction_type']==a.target
        receipt={'passed':True,'target':a.target,'prefix_updates':8192,'original_lr_horizon':65536,
            'slurm_job_id':os.environ['SLURM_JOB_ID'],'node':os.environ.get('SLURMD_NODENAME'),
            'gpu':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),'cuda_runtime':torch.version.cuda,
            'pair_protocol_sha256':sha(HERE/'pair_protocol.json'),'hook_state':h.state(),
            'final_learned_sha256':model_sha(state['model']),'checkpoint_contract_sha256':state['contract_sha256'],
            'frozen_restore_events':events,'elapsed_seconds':time.monotonic()-start,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
            'full_optimizer_state_retained':True,'not_full_horizon_training':True,'automatic_following_action':False}
        (a.output/'arm_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('MAIN_TARGET_PREFIX_ARM_COMPLETE',json.dumps(receipt,indent=2),flush=True)
    finally:
        if a.output.exists():
            (a.output/'effective_runner.py').write_text(loop)
            (a.output/'effective_diffusion_loss.py').write_text(method)


if __name__=='__main__':main()
