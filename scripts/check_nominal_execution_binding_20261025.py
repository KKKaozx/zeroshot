"""New fixed-update loader and nominal-time guard; no training or physics."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))


def main():
    out=ROOT/'reports/nominal_execution_binding_20261025';out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from libero_waypoint_execution import WaypointExecutionContract
    from libero_project_policy import checkpoint_components,load_project_policy,validate_execution_target_binding
    torch.set_num_threads(2)
    kwargs=json.loads((ROOT/'configs/libero_nominal_execution_candidate_20261025.json').read_text())['constructor_kwargs']
    kwargs['target_times_seconds']=tuple(kwargs['target_times_seconds']);kwargs['output_scales']=tuple(kwargs['output_scales'])
    contract=WaypointExecutionContract(**kwargs)
    ticks=contract.ticks();assert ticks.tolist()==list(range(4,65,4)) and contract.execute_waypoints==1
    path=ROOT/'reports/nominal_main_entry_20261025/attempt2/full/latest_step.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    config,source,saved_contract=checkpoint_components(checkpoint)
    assert config==saved_contract['config'] and source.keys()==checkpoint['model'].keys()
    validate_execution_target_binding(config,contract)
    wrong=copy.copy(kwargs);wrong['target_times_seconds']=tuple(.1*j for j in range(1,17))
    try:validate_execution_target_binding(config,WaypointExecutionContract(**wrong))
    except ValueError:pass
    else:raise AssertionError('Same horizon but doubled execution speed admitted')
    bad=copy.copy(checkpoint);bad['contract_sha256']='0'*64
    try:checkpoint_components(bad)
    except ValueError:pass
    else:raise AssertionError('Changed fixed-update contract digest admitted')
    usage=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    used,util=map(int,usage.split(','))
    if used>1024 or util>20:raise ValueError('GPU busy; leave existing workload alone')
    clip=Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    policy=load_project_policy(path,clip,contract,device='cuda',sampling_seed=1101,
        local_dino_path=ROOT/'baselines/dinov2_large_frozen_assets_20261023')
    assert policy.checkpoint_metadata['fixed_update_state_loaded']
    assert policy.checkpoint_metadata['frozen_encoders_and_tokenizer_match_training']
    assert policy.checkpoint_metadata['input_preflight_checkpoint']
    request={'rgb_upright':np.full((224,224,3),127,np.uint8),'instruction':'move the blue block left',
        'current_open_command':None,'action_representation':config['action']['representation']}
    result=policy(request)
    assert result.shape==(16,8) and np.isfinite(result).all()
    assert policy.last_inference['gripper_input_features']==[0.,0.]
    report={'passed':True,'actual_waypoint_contract_ticks':ticks.tolist(),'executed_prefix_native_ticks':int(ticks[0]),
        'changed_clock_same_horizon_rejected':True,'changed_checkpoint_contract_checksum_rejected':True,
        'fixed_update_checkpoint_loaded_without_parameter_migration':True,
        'actual_frozen_encoder_and_tokenizer_identity_verified':True,
        'actual_saved_small_adapter_multiscale_callback_shape':list(result.shape),
        'actual_callback_calls':1,'checkpoint_scope':'Earlier2-update engineering checkpoint, never formal policy',
        'input_image_scope':'Synthetic gray image for interface only','physics_steps':0,'optimizer_updates':0,
        'old_fitting_resume_or_resource_suites_repeated':False,'new_cluster_jobs':0,
        'source_units_or_cross_robot_tcp_calibration_proven':False,'generalization_verdict':None}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
