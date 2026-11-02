"""Bridge pose-port inference adapter; calibrated inputs and outputs are paired."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from libero_project_policy import ProjectWaypointPolicy, validate_execution_target_binding
from libero_pose_transfer import SourceTargetPoseMapping
from cartesian_action_contract import SOURCE_REPRESENTATION

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'deployment/bridge_full_capability_v1'


def _pose_codec():
    path = FROZEN / 'pose_features.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != '596a980947c9f8cbbc7b4380fd4585d19960abc36075813684907b2c43e52df5':
        raise ValueError('Frozen training pose encoder changed')
    spec = importlib.util.spec_from_file_location('_bridge_training_pose_features', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class PoseConditionedWaypointPolicy(ProjectWaypointPolicy):
    def __init__(self, model, tokenizer, config, execution_contract, *, device, sampling_seed,
                 pose_port, training_stats, spatial_mapping):
        if not isinstance(spatial_mapping, SourceTargetPoseMapping):
            raise ValueError('A supplied spatial mapping is mandatory; no world-pose passthrough')
        if (not isinstance(pose_port, torch.nn.Linear) or pose_port.in_features != 18
                or pose_port.bias is not None):
            raise ValueError('Restored bias-free trained Linear(18,context_width) pose-port required')
        if config['action']['representation'] != SOURCE_REPRESENTATION:
            raise ValueError('Pose-port wrapper requires the saved source-Cartesian representation')
        if (training_stats.get('fit_partition') != 'train' or training_stats.get('fit_source') != 'Bridge'
                or np.asarray(training_stats.get('xyz_mean')).shape != (3,)
                or np.asarray(training_stats.get('xyz_std')).shape != (3,)
                or not np.isfinite(training_stats['xyz_mean']).all()
                or not np.isfinite(training_stats['xyz_std']).all()
                or (np.asarray(training_stats['xyz_std']) <= 0).any()):
            raise ValueError('Frozen Bridge training pose statistics required; do not fit target statistics')
        self.pose_codec = _pose_codec()
        self.pose_port = pose_port.to(device).requires_grad_(False).eval()
        self.training_stats = copy.deepcopy(training_stats)
        self.spatial_mapping = spatial_mapping
        self.execution_position_scale = execution_contract.position_scale_meters
        super().__init__(model, tokenizer, config, execution_contract, device=device, sampling_seed=sampling_seed)

    def _tokenize(self, instruction):
        # Match the deployed Bridge batch path, including its explicit77-token padding.
        tokens = self.tokenizer([instruction], padding='max_length', max_length=77,
                                truncation=True, return_tensors='pt')
        return {key: value.to(self.device) for key, value in tokens.items()}

    def _inference_context(self, pixels, tokens, request):
        source_p, source_q = self.spatial_mapping.source_reference(
            request['reference_position_m'], request['reference_quaternion_xyzw'])
        features = self.pose_codec.encode(np.r_[source_p, source_q][None], ['Bridge'],
                                          self.training_stats, 'real_pose')
        pose = torch.from_numpy(features).to(self.device)
        self.pose_port.eval()
        context = super()._inference_context(pixels, tokens, request)
        if context.shape != (1, self.pose_port.out_features):
            raise ValueError('Pose-port width differs from the loaded visual-language context')
        return context + self.pose_port(pose)

    def _execution_values(self, values, request):
        if values.shape != (1, self.config['model']['chunk_size'], 8):
            raise ValueError('Unexpected native sampler output shape')
        mapped = self.spatial_mapping.target_waypoints(values[0], request['reference_position_m'],
            request['reference_quaternion_xyzw'], self.config['action']['position_scale_meters'],
            self.execution_position_scale)
        return mapped[None]

    def __call__(self, request):
        result = super().__call__(request)
        self.last_inference.update(pose_port_applied=True, measured_pose_feature_width=18,
            input_and_output_mapping_applied=True, source_pose_stats_refitted=False,
            spatial_mapping_evidence=self.spatial_mapping.evidence,
            physical_calibration_verified=False, formal_libero_scoring_admitted=False)
        return result


def restore_pose_state(model, pose_port, checkpoint, expected_identity):
    """Restore both learned groups; no zero-port fallback or partial load."""
    from train import trainable_state_dict
    from completed_policy_export import model_sha
    if (checkpoint.get('format') != 'fresh-bridge-capability-policy-v1'
            or checkpoint.get('completed_updates') != 65536
            or checkpoint.get('final_identity') != expected_identity
            or set(checkpoint.get('state', {})) != {'base', 'pose_port'}):
        raise ValueError('Incomplete or unbound Bridge pose-policy checkpoint')
    states = checkpoint['state']; expected = {'base': trainable_state_dict(model), 'pose_port': pose_port.state_dict()}
    for group in expected:
        if (set(states[group]) != set(expected[group]) or model_sha(states[group]) != expected_identity[group]
                or any(states[group][k].shape != v.shape or states[group][k].dtype != v.dtype
                       for k, v in expected[group].items())):
            raise ValueError('Missing, changed or incompatible learned state: ' + group)
    incompatible = model.load_state_dict(states['base'], strict=False)
    if incompatible.unexpected_keys or set(expected['base']).intersection(incompatible.missing_keys):
        raise ValueError('Base learned modules not completely restored')
    pose_port.load_state_dict(states['pose_port'], strict=True)


def load_bridge_pose_policy(checkpoint_path, local_clip_path, execution_contract, *,
                            spatial_mapping, source_report_path, local_dino_path,
                            frozen_profile_path, device, sampling_seed):
    """Load trusted local234100-format state; never fetch weights or fit mapping."""
    mapping = SourceTargetPoseMapping(spatial_mapping)  # Fail before model/file loading if facts are absent.
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError('Full local final_policy.pt is missing; the six-file small result archive has no learned weights')
    from models import RobotAdapterModel
    from transformers import CLIPTokenizer
    from train import trainable_state_dict
    from step_training_runner import frozen_digest, tokenizer_identity
    from frozen_profile_restore import restore_training_constants, normalized_config
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'fresh-bridge-capability-policy-v1' or set(saved.get('state', {})) != {'base', 'pose_port'}:
        raise ValueError('Full Bridge base plus pose-port policy payload required')
    contract = saved['contract']; ref = json.loads((FROZEN / 'reference.json').read_bytes())
    if (contract.get('format') != 'fresh-bridge-full-capability-v1'
            or contract.get('protocol_sha256') != hashlib.sha256((FROZEN / 'protocol.json').read_bytes()).hexdigest()
            or contract.get('stats') != json.loads((FROZEN / 'state_stats.json').read_bytes())
            or contract.get('source_frozen_sha256') != ref['frozen_state_sha256']
            or contract.get('tokenizer') != ref['tokenizer']):
        raise ValueError('Checkpoint Bridge recipe or frozen statistics/provenance differs')
    expected_config = copy.deepcopy(ref['config'])
    expected_config['model']['diffusion_prediction_type'] = 'sample'
    if (normalized_config(contract['config']) != normalized_config(expected_config)
            or contract.get('single_source_supervision') != 'Bridge'
            or contract.get('no_augmentation') is not True):
        raise ValueError('Saved sampler/action/config differs from the trained Bridge recipe')
    report = json.loads(Path(source_report_path).read_bytes())
    if (not report.get('execution_complete') or report.get('completed_updates') != 65536
            or report.get('protocol_sha256') != contract['protocol_sha256']):
        raise ValueError('Completed source report is required to bind both learned identities')
    config = copy.deepcopy(contract['config'])
    validate_execution_target_binding(config, execution_contract)
    for name in ('models.py', 'adapter.py', 'diffusion_decoder.py', 'frozen_visual_backbone.py', 'dataset.py', 'cartesian_action_contract.py'):
        if hashlib.sha256((ROOT / 'code' / name).read_bytes()).hexdigest() != ref['source_code_sha256'][name]:
            raise ValueError('Runtime differs from trained model/geometry implementation: ' + name)
    clip, dino = Path(local_clip_path).resolve(), Path(local_dino_path).resolve()
    if not (clip / 'config.json').is_file() or not (dino / 'config.json').is_file():
        raise ValueError('Existing local CLIP and DINO directories required; no remote fallback')
    config['model'].update(name=str(clip), dino_assets=str(dino))
    model = RobotAdapterModel(config)
    width = saved['state']['pose_port']['weight'].shape[0]
    # No optimizer is constructed. Initialization is overwritten by complete learned state.
    with torch.random.fork_rng(): port = torch.nn.Linear(18, width, bias=False)
    restore_pose_state(model, port, saved, report['final_identity'])
    tokenizer = CLIPTokenizer.from_pretrained(str(clip), local_files_only=True)
    frozen_contract = copy.deepcopy(ref); frozen_contract['config'] = contract['config']
    if tokenizer_identity(tokenizer) != contract['tokenizer']:
        raise ValueError('Tokenizer differs from source training')
    if frozen_digest(model, trainable_state_dict(model)) != contract['source_frozen_sha256']:
        restore_training_constants(model, trainable_state_dict(model), frozen_contract, frozen_profile_path)
    if frozen_digest(model, trainable_state_dict(model)) != contract['source_frozen_sha256']:
        raise ValueError('Frozen encoder/runtime identity differs from training')
    policy = PoseConditionedWaypointPolicy(model, tokenizer, contract['config'], execution_contract,
        device=device, sampling_seed=sampling_seed, pose_port=port, training_stats=contract['stats'], spatial_mapping=mapping)
    policy.checkpoint_metadata = {'checkpoint': str(path.resolve()), 'base_and_pose_port_restored': True,
        'source_training_stats_restored': True, 'source_report': str(Path(source_report_path).resolve()),
        'supplied_mapping_evidence': mapping.evidence, 'physical_calibration_verified_by_loader': False,
        'formal_libero_scoring_admitted': False, 'source_capability_verified_by_loader': False}
    return policy
