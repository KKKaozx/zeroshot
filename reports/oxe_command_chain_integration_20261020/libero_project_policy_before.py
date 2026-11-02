"""Main 8-D policy inference callback for the explicit-time LIBERO executor.

Uses the main training image/tokenizer functions and the checkpoint's unchanged
diffusion settings. Loading or matching an interface does not establish learned
competence, trained gripper behavior, source timing, or cross-robot calibration.
"""
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from dataset import ACTION_REPRESENTATION, prepare_image
from train import tokenise, trainable_state_dict


class ProjectWaypointPolicy:
    def __init__(self, model, tokenizer, config, execution_contract, *, device, sampling_seed):
        self.config = copy.deepcopy(config)
        action, settings = self.config['action'], self.config['model']
        execution_contract.ticks()
        if (action.get('representation') != ACTION_REPRESENTATION
                or action.get('position_scale_meters') != execution_contract.position_scale_meters
                or settings.get('action_dim') != 8
                or settings.get('chunk_size') != len(execution_contract.target_times_seconds)
                or settings.get('decoder_type') != 'diffusion'
                or not settings.get('separate_gripper_head')
                or not settings.get('trajectory_conditioned_gripper')
                or not settings.get('condition_on_current_gripper')
                or settings.get('current_gripper_encoding') != 'binary'
                or settings.get('current_gripper_semantics') != 'last_command_open_positive_v1'):
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
        self.device = torch.device(device)
        self.sampling_seed, self.calls, self.last_inference = sampling_seed, 0, None
        self._check_frozen_clip()
        self.model.to(self.device).eval()

    def _check_frozen_clip(self):
        for encoder in (self.model.vision_encoder, self.model.text_encoder):
            if any(p.requires_grad for p in encoder.parameters()):
                raise ValueError('CLIP vision and text parameters must remain frozen')

    def __call__(self, request):
        image = np.asarray(request['rgb_upright'])
        instruction = request['instruction']
        command = request['current_open_command']
        if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                or not isinstance(instruction, str) or not instruction.strip()
                or type(command) not in (int, float) or command not in (-1, 1)):
            raise ValueError('Need upright uint8 RGB, language and known binary command; no measured-opening substitution')
        self._check_frozen_clip()
        self.model.eval()
        pixels = prepare_image(image).unsqueeze(0).to(self.device)
        tokens = tokenise(self.tokenizer, [instruction], self.device)
        gripper = torch.tensor([[command]], dtype=torch.float32, device=self.device)
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
                               'gripper_input_semantics': 'last explicit open-positive binary command',
                               'trained_competence_or_tcp_calibration_proven': False}
        self.calls += 1
        return values[0]


def load_project_policy(checkpoint_path, local_clip_path, execution_contract, *, device, sampling_seed):
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
    config = copy.deepcopy(checkpoint['config'])
    original_name = config['model']['name']
    config['model']['name'] = str(clip)
    model = RobotAdapterModel(config)
    expected = trainable_state_dict(model)
    source = checkpoint['trainable_state_dict']
    if set(source) != set(expected):
        raise ValueError('Checkpoint must contain all and only main learned modules, including the gripper head')
    if any(source[key].shape != expected[key].shape for key in expected):
        raise ValueError('Checkpoint tensor shapes differ; no partial migration for inference')
    incompatible = model.load_state_dict(source, strict=False)
    if incompatible.unexpected_keys or any(key in expected for key in incompatible.missing_keys):
        raise ValueError('Learned checkpoint state was not completely restored')
    tokenizer = CLIPTokenizer.from_pretrained(str(clip), local_files_only=True)
    policy = ProjectWaypointPolicy(model, tokenizer, config, execution_contract,
                                   device=device, sampling_seed=sampling_seed)
    policy.checkpoint_metadata = {
        'checkpoint': str(Path(checkpoint_path).resolve()), 'original_backbone_name': original_name,
        'local_backbone_path': str(clip), 'experiment_kind': checkpoint.get('experiment_kind'),
        'all_learned_state_keys_restored': True, 'local_backbone_matches_training_identity_verified': False,
        'gripper_training_or_task_success_verified': False,
    }
    return policy
