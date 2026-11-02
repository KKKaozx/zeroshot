"""Explicit command8 endpoint loader and persistent-goal executor.

This is an uncalibrated engineering interface, never formal LIBERO scoring.
No expert actions, source desired-state reconstruction or learned target-domain
controller enters inference. Keep this separate from the old pose8 executor.
"""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'code'),str(HERE/'vendor/code'),str(HERE/'vendor')]
from command8_runtime import read,sha
from bridge_controller_command_contract import (COMMAND_FORMAT,map_increment,advance_control_goal,native_feedback_for_goal)
from libero_pose_transfer import SourceTargetPoseMapping
from libero_execution_binding import snapshot_execution_state,require_execution_state,step_bound_native_command


def restore_endpoint(model,port,payload,protocol):
    from train import trainable_state_dict
    from completed_policy_export import model_sha
    if (payload.get('format')!='bridge-command8-fresh-scaled-policy-v1' or payload.get('completed_updates')!=8192
            or payload.get('effective_action_representation')!=COMMAND_FORMAT
            or payload.get('target_supervised') or payload.get('formal_matrix_admitted')
            or payload.get('source_learned_checkpoint_loaded') is not False or payload.get('fresh_learned_initialization') is not True
            or payload['contract']['protocol_sha256']!=sha(HERE/'protocol.json')
            or payload['contract']['protocol']!=protocol
            or payload['source_stats']!=read(HERE/'vendor/state_stats.json')):
        raise ValueError('Exact completed command8 endpoint required; old reached-pose checkpoint cannot load here')
    expected={'base':trainable_state_dict(model),'pose_port':port.state_dict()}
    state=payload['state']
    if set(state)!=set(expected):raise ValueError('Both complete learned groups required')
    for group,values in expected.items():
        if (set(values)!=set(state[group]) or any(values[k].shape!=v.shape for k,v in state[group].items())
                or model_sha(state[group])!=payload['final_identity'][group]
                or any(not torch.isfinite(v).all() for v in state[group].values())):raise ValueError('Incomplete/corrupt endpoint group: '+group)
    result=model.load_state_dict(state['base'],strict=False)
    if result.unexpected_keys or set(expected['base']).intersection(result.missing_keys):raise ValueError('Base restore incomplete')
    port.load_state_dict(state['pose_port'],strict=True)
    model.max_normalized_position=protocol['sampler_position_limit']
    from motion_scaling import install_standardized_sampler
    if (payload['contract']['motion_normalizer_sha256']!=sha(HERE/'motion_normalizer.json')
            or payload['contract']['motion_normalizer']!=read(HERE/'motion_normalizer.json')):raise ValueError('Exact source-only training normalizer required')
    install_standardized_sampler(model,payload['contract']['motion_normalizer'])


def load_policy(checkpoint,*,clip_assets,dino_assets,mapping,device='cuda',sampling_seed=1101):
    import train,models
    from transformers import CLIPTokenizer
    from step_training_runner import frozen_digest,tokenizer_identity
    from resume_constants import training_constant_constructor
    from frozen_profile_restore import normalized_config
    if Path(models.__file__).resolve()!=HERE/'vendor/code/models.py':raise RuntimeError('Use an isolated command8 process; wrong model module loaded')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if saved.get('format')!='bridge-command8-fresh-scaled-policy-v1' or saved.get('completed_updates')!=8192:
        raise ValueError('Completed command8 endpoint required; no reached-pose checkpoint reinterpretation')
    p=read(HERE/'protocol.json');ref=read(HERE/'vendor/reference.json')
    cfg=copy.deepcopy(saved['source_constructor_config'])
    expected=copy.deepcopy(ref['config']);expected['model']['diffusion_prediction_type']='sample'
    if normalized_config(cfg)!=normalized_config(expected):raise ValueError('Source architecture restore template differs')
    for assets in (Path(clip_assets),Path(dino_assets)):
        if not (assets/'config.json').is_file():raise FileNotFoundError('Existing local encoder assets required')
    cfg['model']['name']=str(Path(clip_assets).resolve());cfg['model']['dino_assets']=str(Path(dino_assets).resolve())
    template=copy.deepcopy(ref);template['config']=saved['source_constructor_config']
    events=[]
    with training_constant_constructor(models.RobotAdapterModel,contract=template,
            profile_path=HERE/'vendor/profile_sample.json',learned_state=train.trainable_state_dict,on_restored=events.append):
        model=models.RobotAdapterModel(cfg)
    learned=train.trainable_state_dict(model)
    if frozen_digest(model,learned)!=ref['frozen_state_sha256']:raise ValueError('Frozen assets differ from source training')
    width=saved['state']['pose_port']['weight'].shape[0]
    with torch.random.fork_rng():port=torch.nn.Linear(18,width,bias=False)
    restore_endpoint(model,port,saved,p)
    tokenizer=CLIPTokenizer.from_pretrained(str(Path(clip_assets)),local_files_only=True)
    if tokenizer_identity(tokenizer)!=ref['tokenizer']:raise ValueError('Source tokenizer differs')
    policy=CommandPolicy(model,port,tokenizer,saved['source_stats'],mapping,device=device,sampling_seed=sampling_seed)
    policy.metadata={'checkpoint_sha256':sha(checkpoint),'effective_action_representation':COMMAND_FORMAT,
        'source_engineering_review_gate_passed':saved['source_engineering_review_gate_passed'],
        'frozen_assets_and_tokenizer_verified':True,'base_and18D_port_complete':True,
        'target_supervised':False,'physical_mapping_calibrated':False,'formal_scoring_admitted':False}
    return policy


class CommandPolicy:
    def __init__(self,model,port,tokenizer,stats,mapping,*,device,sampling_seed):
        self.mapping=SourceTargetPoseMapping(mapping)
        self.model=model.to(device).requires_grad_(False).eval()
        self.port=port.to(device).requires_grad_(False).eval()
        self.tokenizer,self.stats=tokenizer,copy.deepcopy(stats)
        self.device=torch.device(device);self.seed=int(sampling_seed);self.calls=0;self.last_inference=None
        self.action_representation=COMMAND_FORMAT

    def __call__(self,observation,instruction,current_open_command):
        from dataset import prepare_image
        from pose_features import encode
        source_p,source_q=self.mapping.source_reference(observation['robot0_eef_pos'],observation['robot0_eef_quat'])
        feature=torch.from_numpy(encode(np.r_[source_p,source_q][None],['Bridge'],self.stats,'real_pose')).to(self.device)
        rgb=np.asarray(observation['agentview_image'])[::-1].copy()
        pixels=prepare_image(rgb)[None].to(self.device)
        tokens=self.tokenizer([instruction],padding='max_length',max_length=77,truncation=True,return_tensors='pt')
        tokens={k:v.to(self.device) for k,v in tokens.items()}
        current=[0.,0.] if current_open_command is None else [float(current_open_command),1.]
        current=torch.tensor([current],dtype=torch.float32,device=self.device)
        devices=[] if self.device.type=='cpu' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        seed=self.seed+self.calls
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if devices:torch.cuda.manual_seed(seed)
            context=self.model.get_context_vector(pixels,tokens['input_ids'],tokens['attention_mask'])+self.port(feature)
            values=self.model.sample(context,current)[0].cpu().numpy()
        if values.shape!=(16,8) or not np.isfinite(values).all() or not np.isin(values[:,7],[-1.,1.]).all():raise ValueError('Invalid decoded command8 sample')
        self.last_inference={'call':self.calls,'sampling_seed':seed,'output_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
            'source_current_pose_features':feature.cpu().numpy().tolist(),'source_command_history_input':current.cpu().numpy().tolist()}
        self.calls+=1
        return values


class IncrementExecutor:
    """One source increment per4 native ticks; previous desired target persists."""
    def __init__(self,mapping,protocol):
        self.mapping=SourceTargetPoseMapping(mapping);self.spec=copy.deepcopy(mapping);self.protocol=protocol
        if (protocol['action_representation']!=COMMAND_FORMAT or protocol['native_ticks_per_command']!=4
                or protocol['execute_commands_per_replan']!=1 or protocol['native_hz']!=20):
            raise ValueError('Fixed command8/one-increment/four-native-tick contract required')
        if any(np.any(np.asarray(mapping[k])!=0) for k in ('source_eef_to_point_source_units','target_eef_to_point_m')):
            raise ValueError('This candidate supports only the explicitly declared corresponding-point/zero-offset convention')
        self.goal=None

    def execute(self,env,policy,observation,instruction,current_open_command,*,native_step_budget):
        controller=env.robots[0].controller;scales=np.array([.05]*3+[.5]*3)
        if (native_step_budget<4 or env.env.control_freq!=20 or controller.control_dim!=6 or not controller.use_delta
                or not np.array_equal(controller.output_max,scales) or not np.array_equal(controller.output_min,-scales)
                or not np.array_equal(np.broadcast_to(controller.input_max,(6,)),np.ones(6))
                or not np.array_equal(np.broadcast_to(controller.input_min,(6,)),-np.ones(6))):raise ValueError('Fixed20Hz native controller/budget mismatch')
        if getattr(policy,'action_representation',None)!=COMMAND_FORMAT:raise ValueError('Reached-pose policy cannot execute as command increments')
        bound=snapshot_execution_state(env)
        raw=np.asarray(policy(observation,instruction,current_open_command),dtype=np.float64)
        require_execution_state(env,bound)
        if raw.shape!=(16,8):raise ValueError('Command8 policy must return H16x8')
        original=raw[0].copy();limited=original.copy()
        limited[:3]=np.clip(limited[:3],-.5,.5)
        angles=Rotation.from_quat(limited[3:7]).as_euler('xyz')
        clipped_angles=np.clip(angles,-.25,.25)
        limited[3:7]=Rotation.from_euler('xyz',clipped_angles).as_quat()
        packet={'format':COMMAND_FORMAT,'values':limited,'supervision_mask':np.ones(8),'position_divisor':.1}
        packet=map_increment(packet,source_axes_to_target=self.spec['source_axes_to_target'],
            target_meters_per_source_unit=self.spec['target_meters_per_source_unit'],target_position_divisor=.1,
            common_control_point_convention=self.spec['evidence'])
        initialized=self.goal is None
        prior_p=np.asarray(observation['robot0_eef_pos']) if initialized else self.goal['position']
        prior_q=np.asarray(observation['robot0_eef_quat']) if initialized else self.goal['quaternion_xyzw']
        self.goal=advance_control_goal(packet,previous_goal_position=prior_p,previous_goal_quaternion=prior_q,
            measured_pivot_position=observation['robot0_eef_pos'],workspace_bounds=None)
        trace=[];obs=observation;success=False;done=False
        for tick in range(4):
            feedback=native_feedback_for_goal(self.goal,measured_position=obs['robot0_eef_pos'],
                measured_quaternion=obs['robot0_eef_quat'],output_scales=scales,saturation_policy='bounded_feedback')
            obs,_,done,_=step_bound_native_command(env,feedback['native_action'],bound)
            bound=snapshot_execution_state(env);success=bool(env.check_success())
            trace.append({'native_tick_in_command':tick+1,'native_action':feedback['native_action'].tolist(),
                'desired_position':self.goal['position'].tolist(),'desired_quaternion':self.goal['quaternion_xyzw'].tolist(),
                'measured_position':np.asarray(obs['robot0_eef_pos']).tolist(),'native_success':success,
                'native_command_saturated':feedback['native_command_saturated']})
            current_open_command=float(limited[7])
            if success or done:break
        return {'observation':obs,'current_open_command':current_open_command,'trace':trace,
            'success':success,'done':bool(done),'raw_first_command8':original.tolist(),'bounded_first_command8':limited.tolist(),
            'source_translation_or_euler_clipped':bool(np.any(original[:3]!=limited[:3]) or np.any(angles!=clipped_angles)),
            'goal_initialized_from_measured_once':initialized,'desired_goal_updates_this_call':1,
            'previous_goal_position':np.asarray(prior_p).tolist(),'previous_goal_quaternion':np.asarray(prior_q).tolist()}
