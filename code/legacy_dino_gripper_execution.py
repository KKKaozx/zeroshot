"""229116 source-policy callback with a separate gripper and explicit geometry.

Original source output is TOOL-relative/right-relative. The executor receives
world/left-relative deltas AFTER an explicit supplied spatial mapping. Original
9D state features and measured continuous opening are retained, not replaced by
the newer18D port/command availability input.
"""
import hashlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dino_gripper_sidecar import replace_gripper, source_state_features, measured_feature, SOURCE_SHA
from libero_pose_transfer import SourceTargetPoseMapping


def tool_waypoints_to_target(values, mapping, target_position, target_quaternion,
                             source_scale=.1, target_scale=.1):
    if not isinstance(mapping,SourceTargetPoseMapping):
        raise ValueError('An explicit spatial convention is required')
    values = np.asarray(values,dtype=np.float64)
    if values.shape != (16,8):
        raise ValueError('Original H16x8 tool-relative source trajectory required')
    _,source_q = mapping.source_reference(target_position,target_quaternion)
    source_r = Rotation.from_quat(source_q)
    world = values.copy()
    world[:,:3] = source_r.apply(values[:,:3])
    world[:,3:7] = (source_r * Rotation.from_quat(values[:,3:7]) * source_r.inv()).as_quat()
    return mapping.target_waypoints(world,target_position,target_quaternion,source_scale,target_scale)


class LegacyDinoGripperPolicy:
    def __init__(self, source, tokenizer, head, state_stats, mapping, *, device, sampling_seed=1101,
                 state_feature_builder=None):
        if getattr(source,'source_checkpoint_sha256',None) != SOURCE_SHA:
            raise ValueError('Restored exact229116 source model required')
        self.source = source.to(device).requires_grad_(False).eval()
        self.head = head.to(device).requires_grad_(False).eval()
        self.tokenizer,self.state_stats = tokenizer,state_stats
        self.mapping = SourceTargetPoseMapping(mapping)
        self.device,self.seed = torch.device(device),int(sampling_seed)
        self.calls,self.last_inference = 0,None
        self.state_feature_builder = state_feature_builder or source_state_features

    def __call__(self, request):
        from dataset import prepare_image  #The pinned229116 module in the bundle.
        if request['action_representation'] != 'source_cartesian_delta_pose_open_positive_v1':
            raise ValueError('Mapped target output requires world/left-relative executor contract')
        image = np.asarray(request['rgb_upright'])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Upright uint8 RGB required')
        source_p,source_q = self.mapping.source_reference(request['reference_position_m'],request['reference_quaternion_xyzw'])
        pose9 = torch.from_numpy(self.state_feature_builder(source_p,source_q,self.state_stats)[None]).to(self.device)
        current = torch.from_numpy(measured_feature(request['measured_open_fraction'])[None]).to(self.device)
        tokens = self.tokenizer([request['instruction']],padding=True,truncation=True,max_length=77,return_tensors='pt')
        tokens = {k:v.to(self.device) for k,v in tokens.items()}
        pixels = prepare_image(image)[None].to(self.device)
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == 'cuda' else []
        seed = self.seed + self.calls
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if devices: torch.cuda.manual_seed(seed)
            visual = self.source.vision_encoder(pixel_values=pixels).last_hidden_state
            language = self.source.text_encoder(**tokens).last_hidden_state
            now = self.source.adapter(visual,language,tokens['attention_mask'])
            repeated = self.source.adapter(visual,language,tokens['attention_mask'])
            context = now + self.source.history_projection(repeated-now) + self.source.state_projection(pose9)
            values = replace_gripper(self.source,self.head,context,current,source_checkpoint_sha256=SOURCE_SHA)[0].cpu().numpy()
        result = tool_waypoints_to_target(values,self.mapping,request['reference_position_m'],request['reference_quaternion_xyzw'])
        self.last_inference = {'call_index':self.calls,'sampling_seed':seed,'source_pose_parameters_updated':False,
            'source_checkpoint_sha256':SOURCE_SHA,'gripper_is_separate':True,'state_port_width':9,
            'measured_current_feature':current.cpu().numpy().tolist(),
            'input_semantics':'measured continuous opening, not command history',
            'source_output':'tool/right-relative','target_output':'world/left-relative',
            'output_sha256':hashlib.sha256(result.tobytes()).hexdigest(),
            'physical_mapping_calibrated':False,'formal_success_scored':False}
        self.calls += 1
        return result


def bind_panda_measured_observation(policy, observation, *, fully_open_width_m=.08):
    """Bind the same causal observation as execute_policy_chunk; no core change.

    .08 is the declared Panda XML finger-travel engineering convention. It
    does not establish physical equality to the source robot opening fraction.
    """
    qpos = np.asarray(observation['robot0_gripper_qpos'],dtype=np.float64)
    if qpos.shape != (2,) or not np.isfinite(qpos).all() or fully_open_width_m <= 0:
        raise ValueError('Two measured Panda finger joints and explicit open width required')
    fraction = float(np.abs(qpos).sum()/fully_open_width_m)
    def callback(request):
        if (not np.array_equal(request['reference_position_m'],observation['robot0_eef_pos'])
                or not np.array_equal(request['reference_quaternion_xyzw'],observation['robot0_eef_quat'])):
            raise ValueError('Gripper measurement came from a different observation')
        augmented = dict(request); augmented['measured_open_fraction'] = fraction
        return policy(augmented)
    return callback
