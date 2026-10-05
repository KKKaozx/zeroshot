"""Run the two prediction targets with shared initialization and verified noise draws."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
from cached_diffusion import RobotAdapterModel
from run_fit import seed, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, digest in json.loads((root/'integrity.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    assert torch.cuda.is_available()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    seed(42)
    bundle = torch.load(root/'fixed_context.pt', map_location='cpu', weights_only=False)
    model = RobotAdapterModel(bundle['config'])
    missing = model.load_state_dict(bundle['gripper_state_dict'], strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('diffusion_decoder.') or k in
        ['betas','alphas','alpha_bars','posterior_variance','posterior_mean_x0','posterior_mean_xt']
        for k in missing.missing_keys)
    initial = args.output_dir/'shared_initial_weights.pt'
    torch.save(model.state_dict(), initial)
    initial_file_hash = hashlib.sha256(initial.read_bytes()).hexdigest()
    del model
    for arm in ['epsilon', 'sample']:
        print('START ARM:', arm, 'updates:', 5 if args.smoke else 2000, flush=True)
        command = [sys.executable, '-u', str(root/'run_fit.py'),
            '--output-dir', str(args.output_dir/arm), '--prediction-type', arm,
            '--initial-weights', str(initial)]
        if args.smoke:
            command.append('--smoke')
        subprocess.run(command, check=True)
    assert hashlib.sha256(initial.read_bytes()).hexdigest() == initial_file_hash
    coverage = {arm:json.loads((args.output_dir/arm/'training_coverage.json').read_text())
                for arm in ['epsilon','sample']}
    checks = dict(
        identical_initial_state=coverage['epsilon']['initial_state_sha256'] == coverage['sample']['initial_state_sha256'],
        identical_all_training_noisy_inputs_and_timesteps=coverage['epsilon']['training_input_sha256'] == coverage['sample']['training_input_sha256'],
        identical_updates_and_coverage=coverage['epsilon']['updates'] == coverage['sample']['updates'] == (5 if args.smoke else 2000),
        frozen_gripper_unchanged=all(c['frozen_gripper_unchanged'] for c in coverage.values()))
    assert all(checks.values()), checks
    final = {arm:json.loads((args.output_dir/arm/'final_metrics.json').read_text())
             for arm in ['epsilon','sample']}
    decisions = {arm:json.loads((args.output_dir/arm/'diagnostic_decision.json').read_text())
                 for arm in ['epsilon','sample']}
    comparison = dict(paired_checks=checks, smoke_only=args.smoke,
        epsilon_passed=decisions['epsilon']['passed'], x0_passed=decisions['sample']['passed'],
        actual_updates_per_arm=5 if args.smoke else 2000, final_metrics=final,
        initial_state_sha256=coverage['epsilon']['initial_state_sha256'],
        all_training_inputs_sha256=coverage['epsilon']['training_input_sha256'],
        shared_initial_file_sha256=initial_file_hash,
        scope='Single initialization seed, 14 training windows; same-data fitting, not held-out generalization.',
        reproducibility_limit='CUDA interpolation backward may be nondeterministic; input draws and initialization are explicitly checked.')
    save(args.output_dir/'pair_comparison.json', comparison)
    for arm in ['epsilon','sample']:
        for row in final[arm]['samples']:
            m = row['metrics']
            print(f"FINAL arm={arm} seed={row['seed']} position={m['position_error_cm']:.4f}cm "
                  f"rotation={m['rotation_error_deg']:.3f}deg passed={row['passed']}", flush=True)
    print('PAIRED CONTROLS:', checks, flush=True)
    print('PAIRED SMOKE COMPLETED' if args.smoke else 'MATCHED EPSILON/X0 FIT COMPLETED', flush=True)
    print('Results:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
