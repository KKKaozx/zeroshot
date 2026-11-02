"""Main 8-D policy inference callback for the explicit-time LIBERO executor.

Uses the main training image/tokenizer functions and the checkpoint's unchanged
diffusion settings. Loading or matching an interface does not establish learned
competence, trained gripper behavior, source timing, or cross-robot calibration.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataset import ACTION_REPRESENTATION, prepare_image
from train import tokenise, trainable_state_dict
from cartesian_action_contract import TOOL_REPRESENTATION, SOURCE_REPRESENTATION


def validate_execution_target_binding(config,execution_contract):
    """A matching16-waypoint shape is not evidence of matching physical times."""
    binding=config['action'].get('nominal_target_binding')
    if binding is None:return # Preserve legacy fixtures with their explicit external contract.
    if (binding.get('format') not in ('oxe-nominal-target-binding-v1','oxe-nominal-target-set-v1')
            or binding.get('label_family')!='nominal_resampled_observed_reached_pose'
            or config['action']['representation']!=SOURCE_REPRESENTATION
            or binding.get('target_times_seconds')!=list(execution_contract.target_times_seconds)):
        raise ValueError('Checkpoint nominal target times/label family differ from execution; matching horizon alone is insufficient')
    if any(row.get('target_times_seconds')!=binding['target_times_seconds']
            or row.get('protocol_sha256')!=binding['protocol_sha256']
            for row in binding.get('partition_bindings',{}).values()):
        raise ValueError('Training/validation target clocks differ in the checkpoint binding')


def checkpoint_components(checkpoint):
    """Read trusted local epoch or full fixed-update state, never migrate modules."""
    if checkpoint.get('format')=='main-completed-fixed-policy-v1':
        from completed_policy_export import contract_sha,model_sha
        contract=checkpoint['training_contract']
        if (contract_sha(contract)!=checkpoint['training_contract_sha256']
                or checkpoint['config']!=contract['config']
                or checkpoint['completed_updates']!=contract['arguments']['total_updates']
                or model_sha(checkpoint['trainable_state_dict'])!=checkpoint['learned_state_sha256']):
            raise ValueError('Completed policy contract, update count or learned tensors differ')
        return checkpoint['config'],checkpoint['trainable_state_dict'],contract
    if checkpoint.get('format')=='fixed-update-training-state-v1':
        contract=checkpoint['contract']
        digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=checkpoint.get('contract_sha256'):raise ValueError('Fixed-update checkpoint contract checksum differs')
        return contract['config'],checkpoint['model'],contract
    return checkpoint['config'],checkpoint['trainable_state_dict'],None


class ProjectWaypointPolicy:
    def __init__(self, model, tokenizer, config, execution_contract, *, device, sampling_seed):
        self.config = copy.deepcopy(config)
        action, settings = self.config['action'], self.config['model']
        pair_input = settings.get('current_gripper_encoding') == 'last_command_with_availability_v1'
        expected_semantics = ('last_command_open_positive_with_availability_v1' if pair_input
                              else 'last_command_open_positive_v1')
        execution_contract.ticks()
        validate_execution_target_binding(self.config,execution_contract)
        if (action.get('representation') not in (TOOL_REPRESENTATION, SOURCE_REPRESENTATION)
                or action.get('representation') != execution_contract.action_representation
                or action.get('position_scale_meters') != execution_contract.position_scale_meters
                or settings.get('action_dim') != 8
                or settings.get('chunk_size') != len(execution_contract.target_times_seconds)
                or settings.get('decoder_type') != 'diffusion'
                or not settings.get('separate_gripper_head')
                or not settings.get('trajectory_conditioned_gripper')
                or not settings.get('condition_on_current_gripper')
                or settings.get('current_gripper_encoding') not in ('binary', 'last_command_with_availability_v1')
                or settings.get('current_gripper_semantics') != expected_semantics
                or getattr(model, 'current_gripper_feature_dim', 1) != (2 if pair_input else 1)):
            raise ValueError('Need matching 8-D diffusion config and explicit last-command semantics; binary alone is insufficient')
        if (model.action_dim != 8 or model.chunk_size != settings['chunk_size']
                or model.decoder_type != 'diffusion'
                or model.num_diffusion_steps != settings['num_diffusion_steps']
                or model.beta_schedule != settings['beta_schedule']
                or model.diffusion_prediction_type != settings['diffusion_prediction_type']):
            raise ValueError('Loaded model and checkpoint inference settings differ')
        if type(sampling_seed) is not int or not 0 <= sampling_seed < 2 ** 63 - 1:
            raise ValueError('Explicit nonnegative integer sampling seed required')
        self.model, self.tokenizer = model, tokenizer
        self.action_representation = action['representation']
        self.command_pair_input = pair_input
        self.device = torch.device(device)
        self.sampling_seed, self.calls, self.last_inference = sampling_seed, 0, None
        if settings.get('visual_backbone','clip') != getattr(model,'visual_backbone','clip'):
            raise ValueError('Runtime visual backbone differs from checkpoint config')
        self._check_frozen_clip()
        self.model.to(self.device).eval()

    def _check_frozen_clip(self):
        for encoder in (self.model.vision_encoder, self.model.text_encoder):
            if any(p.requires_grad for p in encoder.parameters()):
                raise ValueError('Visual backbone and CLIP text parameters must remain frozen')

    def __call__(self, request):
        if request.get('action_representation', self.action_representation) != self.action_representation:
            raise ValueError('Runtime request frame differs from the checkpoint action contract')
        image = np.asarray(request['rgb_upright'])
        instruction = request['instruction']
        command = request['current_open_command']
        if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                or not isinstance(instruction, str) or not instruction.strip()
                or not ((type(command) in (int, float) and command in (-1, 1))
                        or (command is None and self.command_pair_input))):
            raise ValueError('Need upright uint8 RGB, language and known binary command; no measured-opening substitution')
        self._check_frozen_clip()
        self.model.eval()
        pixels = prepare_image(image).unsqueeze(0).to(self.device)
        tokens = tokenise(self.tokenizer, [instruction], self.device)
        features = ([0., 0.] if command is None else [float(command), 1.]) if self.command_pair_input else [command]
        gripper = torch.tensor([features], dtype=torch.float32, device=self.device)
        seed = (self.sampling_seed + self.calls) % (2 ** 63 - 1)
        cuda_devices = [] if self.device.type != 'cuda' else [
            self.device.index if self.device.index is not None else torch.cuda.current_device()]
        # Preserve the caller's CPU RNG and only the CUDA device actually used.
        with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            context = self.model.get_context_vector(
                pixels, tokens['input_ids'], attention_mask=tokens.get('attention_mask'))
            result = self.model.sample(context, current_gripper=gripper)
            values = result.detach().cpu().numpy().copy()
        expected = (1, self.config['model']['chunk_size'], 8)
        if (values.shape != expected or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values[0, :, 3:7], axis=1), 1, rtol=0, atol=1e-3)
                or not np.isin(values[0, :, 7], [-1., 1.]).all()):
            raise ValueError('Main sampler did not return valid decoded 8-D waypoints')
        self.last_inference = {'call_index': self.calls, 'sampling_seed': seed,
                               'input_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                               'output_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                               'model_eval': not self.model.training,
                               'gripper_input_semantics': self.config['model']['current_gripper_semantics'],
                               'gripper_input_features': features,
                               'trained_competence_or_tcp_calibration_proven': False}
        self.calls += 1
        return values[0]


def load_project_policy(checkpoint_path, local_clip_path, execution_contract, *, device, sampling_seed,
                        local_dino_path=None):
    """Load a trusted project checkpoint with all main learned-module keys.

    local_clip_path must already exist. No download or architecture conversion.
    Frozen CLIP is not included in main checkpoints, so the caller must bind this
    local backbone's identity to training provenance before formal evaluation.
    """
    from models import RobotAdapterModel
    from transformers import CLIPTokenizer

    clip = Path(local_clip_path).resolve()
    if not clip.is_dir() or not (clip / 'config.json').is_file():
        raise ValueError('Existing local CLIP directory required; no implicit remote download')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_config,source,fixed_contract=checkpoint_components(checkpoint)
    config = copy.deepcopy(saved_config)
    # Reject a changed clock before loading the frozen assets or learned tensors.
    validate_execution_target_binding(config,execution_contract)
    original_name = config['model']['name']
    config['model']['name'] = str(clip)
    visual_backbone=config['model'].get('visual_backbone','clip')
    if visual_backbone=='dinov2':
        from frozen_visual_backbone import validate_dino_config
        if local_dino_path is None:
            raise ValueError('DINO checkpoint requires explicit local_dino_path; no CLIP fallback')
        config['model']['dino_assets']=str(Path(local_dino_path).resolve())
        validate_dino_config(config['model'])
    elif local_dino_path is not None:
        raise ValueError('Cannot attach DINO assets to a CLIP checkpoint')
    model = RobotAdapterModel(config)
    expected = trainable_state_dict(model)
    if set(source) != set(expected):
        raise ValueError('Checkpoint must contain all and only main learned modules, including the gripper head')
    if any(source[key].shape != expected[key].shape for key in expected):
        raise ValueError('Checkpoint tensor shapes differ; no partial migration for inference')
    incompatible = model.load_state_dict(source, strict=False)
    if incompatible.unexpected_keys or any(key in expected for key in incompatible.missing_keys):
        raise ValueError('Learned checkpoint state was not completely restored')
    tokenizer = CLIPTokenizer.from_pretrained(str(clip), local_files_only=True)
    frozen_identity_verified=False
    if fixed_contract is not None:
        from step_training_runner import frozen_digest,tokenizer_identity,file_sha
        for name in ('models.py','adapter.py','diffusion_decoder.py','cartesian_action_contract.py'):
            if fixed_contract['source_code_sha256'].get(name)!=file_sha(Path(__file__).with_name(name)):
                raise ValueError('Fixed-update runtime model/geometry implementation changed: '+name)
        if (frozen_digest(model,expected)!=fixed_contract['frozen_state_sha256']
                or tokenizer_identity(tokenizer)!=fixed_contract['tokenizer']):
            raise ValueError('Inference frozen encoder/tokenizer identity differs from fixed-update training')
        frozen_identity_verified=True
    policy = ProjectWaypointPolicy(model, tokenizer, config, execution_contract,
                                   device=device, sampling_seed=sampling_seed)
    policy.checkpoint_metadata = {
        'checkpoint': str(Path(checkpoint_path).resolve()), 'original_backbone_name': original_name,
        'local_backbone_path': str(clip), 'experiment_kind': checkpoint.get('experiment_kind'),
        'all_learned_state_keys_restored': True, 'local_backbone_matches_training_identity_verified': frozen_identity_verified,
        'visual_backbone':visual_backbone,
        'dino_asset_identity_verified':visual_backbone=='dinov2',
        'visual_input_contract':config['model'].get('visual_input_contract','legacy_clip_prepare_image'),
        'fixed_update_state_loaded':fixed_contract is not None,
        'frozen_encoders_and_tokenizer_match_training':frozen_identity_verified,
        'nominal_target_binding':config['action'].get('nominal_target_binding'),
        'input_preflight_checkpoint':bool(fixed_contract and fixed_contract['arguments'].get('oxe_input_preflight')),
        'gripper_training_or_task_success_verified': False,
    }
    return policy
