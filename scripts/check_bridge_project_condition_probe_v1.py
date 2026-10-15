"""CPU verification of observer/RNG pairing against the actual frozen sampler.

No checkpoint/CLIP loaded, no GPU execution, no training. Reduced-width random
decoder tests the observer machinery, not the scientific diagnostic outcome.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import types

os.environ.update(CUDA_VISIBLE_DEVICES='-1', USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'cluster_packages/bridge_project_condition_probe_v1'
SOURCE = ROOT / 'cluster_packages/bridge_growth128_v1'
sys.path.insert(0, str(SOURCE)); sys.path.insert(0, str(PACK))
from models import RobotAdapterModel, diffusion_betas
from diffusion_decoder import MultiscaleConditionalUnet1D
from evaluate_bridge_endpoint_metrics import pose_errors
from probe_helpers import STEPS, sha, rng_state, restore_rng, rng_equal, replace_rows, sample_with_trace, pose_metrics


def main():
    p = json.loads((PACK / 'protocol.json').read_text())
    for file in PACK.glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'), feature_version=(3, 10))
    # Same methods as production; avoid CLIP construction. Tiny random decoder
    # is only for checking non-invasive recording and row/RNG pairing on CPU.
    torch.set_num_threads(2); torch.manual_seed(17)
    m = RobotAdapterModel.__new__(RobotAdapterModel); torch.nn.Module.__init__(m)
    m.vision_encoder = torch.nn.Identity(); m.text_encoder = torch.nn.Identity()
    m.num_diffusion_steps = 100; m.diffusion_prediction_type = 'sample'; m.clip_denoised = True
    m.max_normalized_position = 3.0; m.action_dim = 8; m.chunk_size = 16
    m.separate_gripper_head = True; m.decoder_type = 'diffusion'; m.gripper_target_mode = 'state'
    m.diffusion_decoder = MultiscaleConditionalUnet1D(context_dim=4, down_dims=(16, 32), diffusion_step_embed_dim=16)
    b = diffusion_betas(100, 'squaredcos_cap_v2'); a = 1 - b; ab = torch.cumprod(a, 0)
    previous = torch.cat([torch.ones(1), ab[:-1]])
    for name, value in dict(betas=b, alphas=a, alpha_bars=ab,
        posterior_variance=(b * (1 - previous) / (1 - ab)).clamp(min=1e-20),
        posterior_mean_x0=b * previous.sqrt() / (1 - ab),
        posterior_mean_xt=(1 - previous) * a.sqrt() / (1 - ab)).items():
        m.register_buffer(name, value)
    m.predict_gripper_logits = types.MethodType(lambda self, c, pose, current: torch.zeros_like(pose[..., 0]), m)
    m.eval(); m.requires_grad_(False)
    initial = {n: v.clone() for n, v in m.state_dict().items()}
    contexts, current = torch.randn(4, 4), torch.randn(4, 1)
    device = torch.device('cpu'); state = rng_state(device)
    plain = m.sample(contexts[:2], current[:2]); after = rng_state(device)
    restore_rng(state, device)
    observed, trace = sample_with_trace(m, contexts[:2], current[:2])
    assert torch.equal(plain, observed) and rng_equal(after, rng_state(device))
    changed_c, changed_g = replace_rows(contexts[:2], current[:2], [1], [3], contexts, current)
    assert torch.equal(changed_c[0], contexts[0]) and torch.equal(changed_c[1], contexts[3])
    assert torch.equal(changed_g[0], current[0]) and torch.equal(changed_g[1], current[3])
    restore_rng(state, device)
    changed, other_trace = sample_with_trace(m, changed_c, changed_g)
    assert rng_equal(after, rng_state(device))
    assert np.array_equal(trace['initial_noise'], other_trace['initial_noise'])
    assert torch.equal(changed[0], plain[0])
    assert not torch.equal(changed[1], plain[1])
    assert all(torch.equal(v, m.state_dict()[n]) for n, v in initial.items())
    for step in STEPS:
        assert np.max(np.abs(trace[step]['clean'][..., :3])) <= 3
        assert np.max(np.abs(trace[step]['clean'][..., 3:])) <= 1
    pred = observed.numpy()[..., :7].astype(np.float64)
    truth = changed.numpy()[..., :7].astype(np.float64)
    pos, rot = pose_errors(pred, truth); metrics = pose_metrics(pred, truth)
    assert abs(metrics['position_cm'] - pos.mean()) < 1e-12
    assert abs(metrics['rotation_deg'] - rot.mean()) < 1e-10
    qflip = pred.copy(); qflip[..., 3:7] *= -1
    assert pose_metrics(pred, qflip)['rotation_deg'] < 1e-5
    shifted = pred.copy(); shifted[..., 0] += 1
    assert abs(pose_metrics(shifted, pred)['position_cm'] - 10) < 1e-12
    # Verify cached conditions/targets used by the planned pairs, original row
    # identity, and hash checks before making a cluster package.
    run = ROOT / 'reports/bridge-state-fit-202278/real'
    for split, count, offset in [('train', 128, 0), ('development', 40, 128)]:
        with np.load(run / ('reduced_u8192_' + split + '.npz')) as z:
            assert z['contexts'].shape == (count, 1024) and z['predictions'].shape == (3, count, 16, 8)
            assert np.array_equal(z['row_indices'], np.arange(offset, offset + count))
    with np.load(PACK / 'currents.npz') as z:
        assert [hashlib.sha256(v.tobytes()).hexdigest() for v in z['currents']] == p['current_checks']
    assert p['sampling_seeds'] == [1101, 1102, 1103] and p['trace_steps'] == list(STEPS)
    assert p['pairs'] == [dict(window=w, donor_window=d) for w, d in
                          zip(range(138, 147), [111, 111, 54, 111, 113, 106, 29, 115, 114])]
    result = dict(passed=True, bash_syntax_passed=False,
                  protocol_sha256=sha(PACK / 'protocol.json'),
                  source_sha256={f.name: sha(f) for f in PACK.iterdir() if f.is_file() and f.name != 'SHA256SUMS'},
                  actual_frozen_native_sampler_used_on_cpu=True, observer_prediction_and_rng_exact=True,
                  paired_initial_noise_and_rng_exact=True, untouched_batch_row_exact=True,
                  parameters_and_buffers_unchanged=True, pose_units_and_sign_checked=True,
                  fixed_inputs_verified=True, cpu_fixture_down_dims=[16, 32],
                  real_checkpoint_loaded=False, clip_loaded=False, gpu_replay_executed=False,
                  training_executed=False, scientific_result_available=False)
    path = ROOT / 'reports/bridge_project_condition_probe_preparation.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
