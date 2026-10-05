"""Fixed-budget learned diffusion fit on audited frozen training contexts."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from cached_diffusion import RobotAdapterModel, policy_loss, metrics


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def decision(m):
    return dict(position=m['position_error_cm'] <= .5,
        rotation=m['rotation_error_deg'] <= 5.,
        open_recall=m['open_recall'] is not None and m['open_recall'] >= .95,
        closed_recall=m['closed_recall'] is not None and m['closed_recall'] >= .95,
        open_to_closed=m['open_to_closed_correct'] == m['open_to_closed_pairs'],
        closed_to_open=m['closed_to_open_correct'] == m['closed_to_open_pairs'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prediction-type', choices=['epsilon', 'sample'], required=True)
    parser.add_argument('--initial-weights', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='5 updates; does not make a diagnostic pass claim')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'integrity.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    assert torch.cuda.is_available(), 'GPU required; submit the Slurm GPU job.'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA linear interpolation backward has no deterministic kernel in this path.
    torch.use_deterministic_algorithms(True, warn_only=True)
    seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle = torch.load(root/'fixed_context.pt', map_location='cpu', weights_only=False)
    protocol = json.loads((root/'provenance.json').read_text())
    device = torch.device('cuda')
    print('Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), flush=True)
    bundle['config']['model']['diffusion_prediction_type'] = args.prediction_type
    model = RobotAdapterModel(bundle['config']).to(device)
    initial = torch.load(args.initial_weights, map_location='cpu', weights_only=False)
    model.load_state_dict(initial, strict=True)
    initial_hash = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        initial_hash.update(key.encode())
        initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    protocol['initial_state_sha256'] = initial_hash.hexdigest()
    protocol['prediction_type'] = args.prediction_type
    protocol['training_noise_seed'] = 42
    training_draw_hash = hashlib.sha256()
    draw_calls = 0
    def capture_inputs(module, inputs):
        nonlocal draw_calls
        if module.training:
            for tensor in inputs[:2]:
                training_draw_hash.update(tensor.detach().cpu().numpy().tobytes())
            draw_calls += 1
    hook = model.diffusion_decoder.register_forward_pre_hook(capture_inputs)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('diffusion_decoder.'))
    frozen = {k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
    data = {k:bundle[k].to(device) for k in ['context','current','targets','masks']}
    assert data['targets'].shape == (14,16,8) and data['masks'].eq(1).all()
    optimizer = torch.optim.Adam(model.diffusion_decoder.parameters(), lr=3e-4, weight_decay=0)
    target = data['targets'].cpu().numpy().astype(np.float64)
    baseline = np.zeros_like(target)
    baseline[:,:,6] = 1
    baseline[:,:,7] = -1
    protocol['baseline'] = metrics(baseline, target)
    protocol['actual_steps'] = 5 if args.smoke else 2000
    protocol['smoke_only'] = args.smoke
    protocol['reproducibility_limit'] = 'CUDA linear interpolation backward may be nondeterministic; fixed seeds do not guarantee bitwise identical retraining.'
    save(args.output_dir/'experiment.json', protocol)
    history = []

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        predictions, reports = [], []
        # Evaluation must not change the training timestep/noise RNG sequence.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for value in protocol['sample_seeds']:
                seed(value)
                p = model.sample(data['context'], data['current']).cpu().numpy().astype(np.float64)
                teacher = model.predict_gripper_logits(data['context'], data['targets'][:,:,:7],
                                                      data['current']).cpu().numpy()
                score = metrics(p, target, teacher)
                checks = decision(score)
                reports.append(dict(seed=value, metrics=score, checks=checks, passed=all(checks.values())))
                predictions.append(p)
                print(f"step={step} seed={value} position={score['position_error_cm']:.4f}cm "
                      f"rotation={score['rotation_error_deg']:.3f}deg balance={score['balanced_accuracy']:.3f} "
                      f"switches={score['open_to_closed_correct']}/{score['open_to_closed_pairs']},"
                      f"{score['closed_to_open_correct']}/{score['closed_to_open_pairs']}", flush=True)
        # random and numpy RNG are not used by the optimizer/noise draws.
        report = dict(step=step, samples=reports)
        history.append(report)
        save(args.output_dir/'history.json', history)
        if step == protocol['actual_steps']:
            save(args.output_dir/'final_metrics.json', report)
            save(args.output_dir/'diagnostic_decision.json', dict(
                passed=all(r['passed'] for r in reports) if not args.smoke else None,
                smoke_only=args.smoke, checkpoint_step=step, checks_by_seed=reports,
                scope=protocol['scope']))
            np.savez_compressed(args.output_dir/'final_predictions.npz', predictions=np.stack(predictions),
                targets=target, indices=np.array(bundle['indices']), seeds=np.array(protocol['sample_seeds']))
            torch.save(dict(config=bundle['config'], state_dict=model.state_dict(), steps=step,
                            indices=bundle['indices'], not_a_deployable_checkpoint=True),
                       args.output_dir/'final_head.pt')
        model.train()

    evaluate(0)
    started = time.monotonic()
    for step in range(1, protocol['actual_steps']+1):
        optimizer.zero_grad(set_to_none=True)
        output = model.diffusion_loss(data['targets'], data['context'], data['current'])
        total, pose, grip = policy_loss(model, output, data['targets'], torch.nn.MSELoss(),
            current_grippers=data['current'], supervision_masks=data['masks'])
        assert torch.isfinite(total)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.diffusion_decoder.parameters(), 1.)
        assert torch.isfinite(gradient_norm)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f'update={step} pose_loss={float(pose.detach()):.6f} teacher_grip_loss={float(grip.detach()):.6f} '
                  f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        if step % 250 == 0 or step == protocol['actual_steps']:
            evaluate(step)
    assert all(torch.equal(p, frozen[k]) for k,p in model.named_parameters() if k in frozen)
    assert draw_calls == protocol['actual_steps']
    hook.remove()
    save(args.output_dir/'training_coverage.json', dict(windows=14, action_targets=224,
        training_input_sha256=training_draw_hash.hexdigest(), training_input_calls=draw_calls,
        initial_state_sha256=protocol['initial_state_sha256'],
        updates=protocol['actual_steps'], draws_by_window=[protocol['actual_steps']]*14,
        frozen_gripper_unchanged=True, validation_targets_used=False, test_targets_used=False))
    print('SMOKE COMPLETED' if args.smoke else 'FIXED BUDGET DIFFUSION FIT COMPLETED', flush=True)


if __name__ == '__main__':
    main()
