"""Actual main model + local random tiny CLIP, no task data or training.

Checks checkpoint reload, shared preprocessing, DDPM replay and executor request
binding. This is an engineering fixture, not the trained ViT-L/14 policy.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from transformers import CLIPConfig, CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from dataset import ACTION_REPRESENTATION, prepare_image
from libero_project_policy import ProjectWaypointPolicy, load_project_policy
from libero_waypoint_execution import WaypointExecutionContract
from models import RobotAdapterModel
from train import tokenise, trainable_state_dict


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Incompatible inference accepted')


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    output = ROOT / 'reports/libero_project_policy_20261020.json'
    contract = WaypointExecutionContract(tuple(.1 * (i + 1) for i in range(4)), 1, .1,
                                         'Synthetic .1s fixture, not a source-rate claim',
                                         'No cross-robot calibration in this fixture')
    with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as temp:
        folder = Path(temp)
        clip_path = folder / 'random_tiny_clip'
        clip_path.mkdir()
        # A complete offline CLIP/tokenizer pair generated in this temporary
        # fixture. No pretrained downloads, real images, or cached task features.
        vocabulary = {f'{letter}{suffix}': i for i, (letter, suffix) in enumerate(
            (letter, suffix) for letter in 'abcdefghijklmnopqrstuvwxyz' for suffix in ('', '</w>'))}
        vocabulary.update({'<|startoftext|>': 52, '<|endoftext|>': 53})
        tokenizer = CLIPTokenizer(vocab=vocabulary, merges=[])
        tokenizer.save_pretrained(clip_path)
        clip_config = CLIPConfig(
            text_config={'vocab_size': 54, 'hidden_size': 32, 'intermediate_size': 64,
                         'num_hidden_layers': 1, 'num_attention_heads': 4, 'max_position_embeddings': 77,
                         'bos_token_id': 52, 'eos_token_id': 53, 'pad_token_id': 53},
            vision_config={'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
                           'num_attention_heads': 4, 'image_size': 224, 'patch_size': 32},
            projection_dim=16)
        CLIPModel(clip_config).save_pretrained(clip_path)
        config = {
            'action': {'representation': ACTION_REPRESENTATION, 'position_scale_meters': .1,
                       'max_normalized_position': 3.},
            'model': {'name': str(clip_path), 'action_dim': 8, 'chunk_size': 4,
                      'num_adapter_layers': 1, 'attention_dim': 32, 'num_attention_heads': 4,
                      'dropout': .2, 'adapter_pooling': 'cls_patch_mean', 'decoder_hidden_dim': 32,
                      'decoder_type': 'diffusion', 'num_diffusion_steps': 4,
                      'beta_schedule': 'squaredcos_cap_v2', 'diffusion_prediction_type': 'sample',
                      'clip_denoised': True, 'separate_gripper_head': True,
                      'trajectory_conditioned_gripper': True, 'condition_on_current_gripper': True,
                      'current_gripper_encoding': 'binary',
                      'current_gripper_semantics': 'last_command_open_positive_v1'}}
        model = RobotAdapterModel(config)
        checkpoint = {'config': config, 'trainable_state_dict': trainable_state_dict(model),
                      'experiment_kind': 'random_engineering_fixture_not_trained'}
        checkpoint_path = folder / 'random_fixture.pt'
        torch.save(checkpoint, checkpoint_path)
        runtime = load_project_policy(checkpoint_path, clip_path, contract, device='cpu', sampling_seed=701)
        before_state = state_digest(runtime.model)
        rgb = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
        request = {'rgb_upright': rgb, 'instruction': 'move left', 'current_open_command': 1.,
                   'reference_position_m': np.zeros(3), 'reference_quaternion_xyzw': np.array([0, 0, 0, 1.])}
        torch.manual_seed(99)
        before_rng = torch.get_rng_state().clone()
        result = runtime(request)
        assert torch.equal(before_rng, torch.get_rng_state())
        assert state_digest(runtime.model) == before_state
        assert runtime.last_inference['sampling_seed'] == 701
        assert not runtime.model.training
        assert all(not encoder.training and all(not p.requires_grad for p in encoder.parameters())
                   for encoder in (runtime.model.vision_encoder, runtime.model.text_encoder))
        with torch.random.fork_rng(), torch.inference_mode():
            torch.manual_seed(701)
            text = tokenise(runtime.tokenizer, [request['instruction']], torch.device('cpu'))
            context = runtime.model.get_context_vector(prepare_image(rgb)[None], text['input_ids'],
                                                       attention_mask=text['attention_mask'])
            direct = runtime.model.sample(context, current_gripper=torch.tensor([[1.]]))[0].numpy()
        assert np.array_equal(result, direct)
        assert result.shape == (4, 8) and np.isin(result[:, -1], [-1, 1]).all()
        # Check another independent reload and the next deterministic draw.
        reloaded = load_project_policy(checkpoint_path, clip_path, contract, device='cpu', sampling_seed=701)
        assert np.array_equal(result, reloaded(request))
        assert np.array_equal(runtime(request), reloaded(request))
        assert runtime.last_inference['sampling_seed'] == 702
        assert state_digest(runtime.model) == before_state

        for mutation in ('measured_gripper', 'unspecified_binary_semantics', 'position_scale', 'horizon', 'regression'):
            bad = copy.deepcopy(config)
            if mutation == 'measured_gripper':
                bad['model']['current_gripper_encoding'] = 'bridge_measured_affine_unbounded_v1'
            elif mutation == 'unspecified_binary_semantics':
                del bad['model']['current_gripper_semantics']
            elif mutation == 'position_scale':
                bad['action']['position_scale_meters'] = 1.
            elif mutation == 'horizon':
                bad['model']['chunk_size'] = 16
            else:
                bad['model']['decoder_type'] = 'regression'
            rejected(lambda: ProjectWaypointPolicy(runtime.model, runtime.tokenizer, bad, contract,
                                                     device='cpu', sampling_seed=701))
        rejected(lambda: runtime(dict(request, current_open_command=.3)))
        rejected(lambda: runtime(dict(request, rgb_upright=rgb.astype(np.float32) / 255)))
        runtime.model.vision_encoder.embeddings.class_embedding.requires_grad_(True)
        rejected(lambda: runtime(request))
        runtime.model.vision_encoder.embeddings.class_embedding.requires_grad_(False)
        partial = copy.deepcopy(checkpoint)
        partial['trainable_state_dict'].pop(next(k for k in partial['trainable_state_dict'] if k.startswith('gripper_head.')))
        partial_path = folder / 'partial.pt'
        torch.save(partial, partial_path)
        rejected(lambda: load_project_policy(partial_path, clip_path, contract, device='cpu', sampling_seed=701))

    report = {'checks_passed': True, 'actual_RobotAdapterModel_used': True,
              'backbone': 'Local randomly initialized tiny CLIP, not pretrained ViT-L/14',
              'actual_main_adapter_decoder_and_gripper_code_used': True,
              'main_image_preprocessing_and_tokenise_reused': True,
              'direct_vs_callback_max_absolute_difference': float(np.max(np.abs(result - direct))),
              'complete_learned_state_restore_and_repeat_inference_exact': True,
              'caller_cpu_rng_preserved': True, 'state_dict_unchanged_during_inference': True,
              'frozen_clip_and_eval_checked': True, 'mismatched_scale_horizon_gripper_partial_state_rejected': True,
              'binary_without_explicit_command_semantics_rejected': True,
              'training_updates': 0, 'physics_steps': 0, 'pretrained_downloads': 0,
              'trained_checkpoint_or_real_task_competence_verified': False,
              'cuda_path_tested': False, 'benchmark_success_claimed': False}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
