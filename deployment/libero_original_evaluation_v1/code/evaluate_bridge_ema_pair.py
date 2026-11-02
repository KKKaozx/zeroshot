"""Evaluate raw and EMA weights from one optimization trajectory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--window-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    bundle = Path(__file__).resolve().parent
    sys.path.insert(0, str(args.pack.resolve()))
    sys.path.insert(0, str(bundle))
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import bridge_plan_selection, bridge_plan_splits, collate_batch, set_seed, trainable_state_dict
    from run_bridge_official_unet_training import digest, select_windows, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors
    from evaluate_bridge_official_unet_one_step import extended_metrics, groups_for
    from evaluate_bridge_one_step_sampler import finish_pose, decode_state_gripper

    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    training_path = args.training_run / 'report.json'
    report = json.loads(training_path.read_text())
    if not (report.get('passed') and report.get('stage') == 'multiscale_x0_training_with_ema'
            and report['ema_updates'] == 18360 and not report['reserved_test_targets_read']):
        raise ValueError('Training integrity report is not eligible')
    for name, expected in report['protocol']['module_sha256'].items():
        path = bundle / name if name in ('models.py', 'diffusion_decoder.py') else args.pack / name
        if sha256(path) != expected:
            raise ValueError(f'Core module changed: {name}')
    if sha256(bundle / 'run_bridge_official_unet_training.py') != report['protocol']['runner_sha256']:
        raise ValueError('Training runner changed')
    if sha256(args.window_reference) != report['protocol']['window_reference_sha256']:
        raise ValueError('Window reference changed')
    selection = bridge_plan_selection(args.pack / 'manifest.json')
    dataset = UnifiedRobotDataset(
        data_dir=str(args.pack / 'data'), chunk_size=16, stride=4,
        sources=['tfrecord'], min_trajectory_steps=17, exclude_path_parts=[],
        exclude_schemas=[], tfrecord_splits=['train'],
        bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous', bridge_episode_selection=selection)
    rows = select_windows(dataset, bridge_plan_splits(dataset), selection)
    reference = json.loads(args.window_reference.read_text())
    if rows != reference['window_selection']:
        raise ValueError('Evaluation population differs from the frozen protocol')
    groups = groups_for(rows)
    if (len(groups['train/overall']), len(groups['validation/overall'])) != (1836, 40):
        raise ValueError('Unexpected complete window counts')
    items = [dataset[row['dataset_index']] for row in rows]
    targets = np.stack([item[3].numpy() for item in items]).astype(np.float32)
    paths = {'raw': args.training_run / 'final.pt', 'ema': args.training_run / 'ema.pt'}
    hashes = {key: sha256(path) for key, path in paths.items()}
    if hashes['raw'] != report['checkpoint']['sha256'] or hashes['ema'] != report['ema_checkpoint']['sha256']:
        raise ValueError('Checkpoint bytes changed')
    payloads = {key: torch.load(path, map_location='cpu', weights_only=False) for key, path in paths.items()}
    if payloads['raw']['config'] != payloads['ema']['config']:
        raise ValueError('EMA model configuration differs from raw')
    if any(p['protocol'] != report['protocol'] for p in payloads.values()):
        raise ValueError('Checkpoint protocol differs from training report')
    model = RobotAdapterModel(payloads['raw']['config'])
    if not (model.decoder_type == 'diffusion' and model.num_diffusion_steps == 100
            and model.diffusion_prediction_type == 'sample'
            and model.separate_gripper_head and model.gripper_target_mode == 'state'):
        raise ValueError('Expected archived 100-step x0 diffusion with state gripper')
    expected_keys = set(trainable_state_dict(model))
    if any(set(p['model']) != expected_keys for p in payloads.values()):
        raise ValueError('EMA parameter set differs from raw')
    frozen = lambda: {k: v for k, v in model.state_dict().items()
                      if k.startswith(('vision_encoder.', 'text_encoder.'))}
    clip_hash = digest(frozen())
    if clip_hash != report['protocol']['clip_sha256']:
        raise ValueError('Frozen CLIP differs from training')
    model = model.cuda().eval()
    tokenizer = CLIPTokenizer.from_pretrained(payloads['raw']['config']['model']['name'], local_files_only=True)
    outputs = {}
    rng_hashes = {}
    timings = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for branch in ('raw', 'ema'):
            model.load_state_dict(payloads[branch]['model'], strict=False)
            model.eval()
            outputs[branch] = {name: np.empty((3, len(rows), 16, 8), np.float32)
                               for name in ('native_ddpm100', 'one_step_x0')}
            streams = {mode: [hashlib.sha256() for _ in range(3)] for mode in outputs[branch]}
            began = time.monotonic()
            for start in range(0, len(rows), 2):
                language, images, current, _, _ = collate_batch(items[start:start + 2])
                tokens = tokenizer(language, padding=True, truncation=True, return_tensors='pt')
                context = model.get_context_vector(images.cuda(), tokens['input_ids'].cuda(), tokens['attention_mask'].cuda())
                current = current.cuda()
                for mode in outputs[branch]:
                    for seed in range(3):
                        set_seed(seed * 10000 + start)
                        streams[mode][seed].update(torch.cuda.get_rng_state().numpy().tobytes())
                        if mode == 'native_ddpm100':
                            value = model.sample(context, current)
                        else:
                            noise = torch.randn(len(language), 16, 7, device='cuda')
                            timestep = torch.full((len(language),), 99, device='cuda', dtype=torch.long)
                            pose = finish_pose(torch, model.diffusion_decoder(noise, timestep, context),
                                               float(model.max_normalized_position or 3.0))
                            gripper = decode_state_gripper(torch, model, context, pose, current)
                            value = torch.cat((pose, gripper.unsqueeze(-1)), dim=-1)
                        streams[mode][seed].update(torch.cuda.get_rng_state().numpy().tobytes())
                        if not torch.isfinite(value).all():
                            raise ValueError(f'Non-finite {branch}/{mode} output')
                        outputs[branch][mode][seed, start:start + len(language)] = value.cpu().numpy()
                if start == 0 or (start + 2) % 200 == 0:
                    print('EMA_PAIR_PROGRESS', branch, min(start + 2, len(rows)), '/', len(rows),
                          'seconds', round(time.monotonic() - began, 1), flush=True)
            rng_hashes[branch] = {mode: [h.hexdigest() for h in values] for mode, values in streams.items()}
            timings[branch] = time.monotonic() - began
    if rng_hashes['raw'] != rng_hashes['ema']:
        raise ValueError('Raw/EMA CUDA noise streams are not paired')
    if digest(frozen()) != clip_hash or any(sha256(paths[key]) != hashes[key] for key in paths):
        raise ValueError('Frozen CLIP or saved checkpoints changed during evaluation')
    static = np.zeros_like(targets)
    static[..., 6] = 1
    static[..., 7] = -1
    result_groups = {}
    for name, indices in groups.items():
        result_groups[name] = {'static_always_closed': extended_metrics(static[indices], targets[indices], pose_errors)}
        for branch in outputs:
            result_groups[name][branch] = {
                mode: [extended_metrics(pred[seed, indices], targets[indices], pose_errors) for seed in range(3)]
                for mode, pred in outputs[branch].items()}
    prediction_path = args.output / 'paired-predictions.npz'
    np.savez_compressed(prediction_path, targets=targets,
                        **{f'{branch}_{mode}': value for branch, modes in outputs.items() for mode, value in modes.items()})
    result = {'stage': 'same_training_raw_vs_ema', 'execution_checks_passed': True,
              'ema_improvement_established': None, 'weights_updated': False,
              'reserved_test_targets_read': False, 'training_report_sha256': sha256(training_path),
              'checkpoint_sha256': hashes, 'sampling_seeds': [0, 1, 2],
              'paired_cuda_rng_streams': True, 'rng_stream_sha256': rng_hashes,
              'counts': {'train': 1836, 'validation': 40}, 'window_selection': rows,
              'groups': result_groups, 'evaluation_seconds': timings,
              'elapsed_seconds': time.monotonic() - started,
              'predictions_sha256': sha256(prediction_path),
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
              'limits': ['One training seed; three sampling seeds are not independent training repeats.',
                         'Full 100-step DDPM and one-step x0 are separate protocols.',
                         'Overlapping windows are not independent demonstrations.',
                         'Offline pose/event errors are not simulated task success.',
                         'Static gripper comparison is always closed, not persistence.']}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    for mode in ('native_ddpm100', 'one_step_x0'):
        for branch in ('raw', 'ema'):
            metrics = result_groups['validation/overall'][branch][mode]
            keys = ('path_position_cm', 'path_rotation_deg', 'endpoint_position_cm', 'endpoint_rotation_deg',
                    'gripper_balanced_accuracy', 'grasp_exact_timing_accuracy', 'release_exact_timing_accuracy')
            print('EMA_PAIR_VALIDATION', mode, branch,
                  json.dumps({key: float(np.mean([m[key] for m in metrics])) for key in keys}), flush=True)
    print('EMA PAIR EVALUATION COMPLETE: execution passing is not an improvement verdict', args.output / 'report.json')


if __name__ == '__main__':
    main()
