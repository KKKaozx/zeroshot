"""Read-only latest-checkpoint probe: target noising versus native generation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_windows(dataset, splits, selection, per_task, all_train=False):
    """Select a source-order diagnostic subset or every fixed split window."""
    lookup = {(r['shard'], r['record_index']): r for r in selection}
    counts, seen, rows = defaultdict(int), set(), []
    for part in ('train', 'validation'):
        for index in sorted(splits[part], key=lambda i: (
                Path(dataset.samples[i]['file_path']).name,
                dataset.samples[i]['record_index'], dataset.samples[i]['start_index'])):
            sample = dataset.samples[index]
            key = Path(sample['file_path']).name, sample['record_index']
            task = lookup[key]['instruction']
            if part == 'train' and not all_train:
                if key in seen or counts[task] >= per_task:
                    continue
                seen.add(key)
                counts[task] += 1
            rows.append(dict(dataset_index=index, partition=part, task=task,
                shard=key[0], record_index=key[1], start_index=sample['start_index']))
    if all_train:
        assert {r['dataset_index'] for r in rows} == set(splits['train']) | set(splits['validation'])
    elif len(counts) != 5 or any(v != per_task for v in counts.values()):
        raise ValueError('Training subset does not cover five tasks at requested count')
    return rows


def donor_indices(rows):
    """Explicit swaps; development window counts need not permit a bijection."""
    images, texts = [], []
    for row in rows:
        pool = [(i, r) for i, r in enumerate(rows) if r['partition'] == row['partition']]
        tasks = sorted({r['task'] for _, r in pool})
        next_task = tasks[(tasks.index(row['task']) + 1) % len(tasks)]
        image = next(i for i, r in pool if r['task'] == row['task']
            and (r['shard'], r['record_index']) != (row['shard'], row['record_index']))
        text = next(i for i, r in pool if r['task'] == next_task)
        images.append(image)
        texts.append(text)
    return images, texts


def sample_with_trace(model, context, current, steps):
    """Observe the actual production decoder without replacing its sampler."""
    trace, calls = {}, []
    def observe(module, inputs, output):
        step = int(inputs[1][0])
        calls.append(step)
        if step in steps:
            trace[step] = (inputs[0].detach().cpu().numpy().copy(),
                output.detach().cpu().numpy().copy())
    hook = model.diffusion_decoder.register_forward_hook(observe)
    try:
        output = model.sample(context, current)
    finally:
        hook.remove()
    assert calls == list(reversed(range(model.num_diffusion_steps)))
    assert set(trace) == set(steps)
    return output[..., :7], trace


def run_chain_probe(model, contexts, items, target, groups, report, reference_path, pose_metrics):
    import numpy as np
    import torch
    from train import collate_batch, set_seed
    reference = json.loads(reference_path.read_text())
    full = report.get('all_train_windows',False)
    for key in ('checkpoint_sha256','module_sha256','sampling_seeds') + (() if full else ('window_selection',)):
        assert report[key] == reference[key], f'Prior probe mismatch: {key}'
    steps = (99,89,74,49,24,9,0)
    variance = model.posterior_variance.clone()
    originals = {}
    report.update(experiment='paired_production_ddpm_with_and_without_step_noise',
        prior_probe_sha256=sha256(reference_path), trace_steps=list(steps),
        trace_semantics='Raw predicted x0 before each reverse update; target used for scoring only',
        clip_denoised=model.clip_denoised, max_normalized_position=model.max_normalized_position,
        chains={}, prior_native_replay_metric_differences={})
    report['limits'] = ['Full fixed training/development pool; reserved test unused.' if full else '40 fixed training windows, not full training evaluation.',
        'Same initial noise and same production mean update; no training or hyperparameter selection.',
        'Suppressing posterior step noise changes the sampling distribution, not only numerical precision.',
        'Better target error does not imply better robot-task performance.',
        'Initial noise still varies across seeds; no-step-noise chain is deterministic only conditional on it.',
        'Traced x0 errors are not errors of completed robot actions or proofs of a unique root cause.']
    try:
        for variant in ('native_stochastic','no_step_noise'):
            model.posterior_variance.copy_(variance if variant == 'native_stochastic' else torch.zeros_like(variance))
            final_results, trace_results = defaultdict(list), {t:defaultdict(list) for t in steps}
            for seed in range(3):
                prediction = np.empty((len(items),16,7),np.float32)
                predicted_x0 = {t:np.empty_like(prediction) for t in steps}
                noisy_states = {t:np.empty_like(prediction) for t in steps}
                for start in range(0,len(items),2):
                    context = contexts['correct'][start:start+2].cuda()
                    _,_,current,_,_ = collate_batch(items[start:start+2])
                    set_seed(seed*10000+start)
                    output,trace = sample_with_trace(model,context,current.cuda(),steps)
                    if full and (start+len(output)) % 200 == 0:
                        print('CHAIN_PROGRESS',variant,'seed',seed,'windows',start+len(output),'/',len(items),flush=True)
                    assert torch.isfinite(output).all()
                    prediction[start:start+len(output)] = output.cpu().numpy()
                    for t,(noisy,clean) in trace.items():
                        assert np.isfinite(noisy).all() and np.isfinite(clean).all()
                        noisy_states[t][start:start+len(output)] = noisy
                        predicted_x0[t][start:start+len(output)] = clean
                if variant == 'native_stochastic':
                    originals[seed] = (noisy_states[99].copy(),predicted_x0[99].copy(),prediction.copy())
                else:
                    assert np.array_equal(noisy_states[99],originals[seed][0]), 'Initial noise differs'
                    assert np.array_equal(predicted_x0[99],originals[seed][1]), 'Initial x0 prediction differs'
                for group,indices in groups.items():
                    values = pose_metrics(prediction[indices],target[indices,:,:7])
                    if variant == 'no_step_noise':
                        change = pose_metrics(prediction[indices],originals[seed][2][indices])
                        values.update(paired_output_change_cm=change['position_cm'],
                            paired_output_change_deg=change['rotation_deg'])
                    final_results[group].append(values)
                    for t in steps:
                        score = pose_metrics(predicted_x0[t][indices],target[indices,:,:7])
                        score['noisy_state_component_mse'] = float(((noisy_states[t][indices]-target[indices,:,:7])**2).mean())
                        trace_results[t][group].append(score)
            report['chains'][variant] = dict(final=dict(final_results),
                predicted_x0_trace={str(t):dict(v) for t,v in trace_results.items()})
            print('CHAIN',variant,'train',final_results['train/overall'],
                'development',final_results['validation/overall'],flush=True)
        if full:
            report['one_step_pure_noise'] = report['chains']['native_stochastic']['predicted_x0_trace']['99']
            report['prior_replay_not_directly_comparable'] = 'Population/order and batch seeds changed; compare the three current paired outputs instead.'
        for group in (() if full else ('train/overall','validation/overall')):
            old = reference['cases']['native_sample']['correct'][group]
            new = report['chains']['native_stochastic']['final'][group]
            report['prior_native_replay_metric_differences'][group] = [
                {k:new[i][k]-old[i][k] for k in ('position_cm','rotation_deg','pose_component_mse')}
                for i in range(3)]
        report['identical_initial_noise_and_initial_x0'] = True
    finally:
        model.posterior_variance.copy_(variance)
    report['posterior_variance_restored'] = bool(torch.equal(model.posterior_variance,variance))
    assert report['posterior_variance_restored']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-episodes-per-task', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--chain-probe', action='store_true')
    parser.add_argument('--reference-report', type=Path)
    parser.add_argument('--all-train-windows', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output path')
    if args.train_episodes_per_task < 2:
        raise ValueError('At least two training episodes per task are required')
    if args.chain_probe and args.reference_report is None:
        raise ValueError('Chain probe requires previous complete conditioning report')
    if args.all_train_windows and not args.chain_probe:
        raise ValueError('Full-window mode is limited to the paired chain probe')
    # Import the archived training package, rather than mutable workspace models.
    sys.path.insert(0, str(args.pack.resolve()))
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    from transformers import CLIPTokenizer
    from train import bridge_plan_selection, bridge_plan_splits, collate_batch, set_seed, trainable_state_dict
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from evaluate_multitask_pilot import metrics

    def state_digest(model):
        digest = hashlib.sha256()
        for name, value in sorted(trainable_state_dict(model).items()):
            digest.update(name.encode('utf-8'))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def pose_metrics(prediction, target):
        # Reuse the production physical metric; artificial gripper columns are
        # omitted from reported metrics, so no teacher label becomes accuracy.
        prediction = torch.as_tensor(prediction)
        target = torch.as_tensor(target)
        mse = float(((prediction-target)**2).mean())
        norm = prediction[..., 3:7].norm(dim=-1)
        invalid = norm <= 1e-6
        prediction = prediction.clone()
        prediction[..., 3:7][invalid] = torch.tensor([0., 0., 0., 1.])
        extra = torch.zeros_like(target[..., :1])
        values = metrics(torch.cat([prediction, extra], -1), torch.cat([target, extra], -1))
        return dict(windows=len(target), position_cm=values['position_cm'],
            rotation_deg=values['rotation_deg'], pose_component_mse=mse,
            quaternion_degenerate_count=int(invalid.sum()))

    manifest = args.pack / 'manifest.json'
    selected = bridge_plan_selection(manifest)
    dataset = UnifiedRobotDataset(data_dir=str(args.pack / 'data'), chunk_size=16, stride=4,
        sources=['tfrecord'], min_trajectory_steps=17, exclude_path_parts=[], exclude_schemas=[],
        tfrecord_splits=['train'], bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous', bridge_episode_selection=selected)
    splits = bridge_plan_splits(dataset)
    assert {k: len(v) for k, v in splits.items()} == json.loads(manifest.read_text())['expected_windows']
    rows = select_windows(dataset, splits, selected, args.train_episodes_per_task,args.all_train_windows)
    images, texts = donor_indices(rows)
    items = [dataset[r['dataset_index']] for r in rows]
    target = np.stack([item[3].numpy() for item in items])
    assert np.isfinite(target).all() and all(torch.all(item[4] == 1) for item in items)
    if args.prepare_only:
        if args.chain_probe:
            reference = json.loads(args.reference_report.read_text())
            if not args.all_train_windows:
                assert rows == reference['window_selection']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(stage='subset_and_control_preparation_only',
            model_loaded=False, trained=False, reserved_test_targets_read=False,
            window_selection=rows, image_donors=images, language_donors=texts,
            counts={part: sum(r['partition'] == part for r in rows) for part in ('train','validation')}),indent=2)+'\n')
        print('PROBE SUBSET AND CONTROLS: PASSED',args.output,flush=True)
        return
    checkpoint = torch.load(args.run / 'latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['split_indices'] == splits
    model = RobotAdapterModel(checkpoint['config']).cuda().eval()
    assert model.decoder_type == 'diffusion' and model.diffusion_prediction_type == 'sample'
    assert model.separate_gripper_head and model.num_diffusion_steps == 100
    assert set(checkpoint['trainable_state_dict']) == set(trainable_state_dict(model))
    model.load_state_dict(checkpoint['trainable_state_dict'], strict=False)
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint['config']['model']['name'], local_files_only=True)
    before = state_digest(model)
    conditions = ('correct',) if args.chain_probe else ('correct', 'image_swap_same_task', 'language_swap_other_task')
    contexts = defaultdict(list)
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(rows), 2):
            batch = items[start:start+2]
            language, pixels, _, _, _ = collate_batch(batch)
            for condition in conditions:
                text = language if condition != 'language_swap_other_task' else [items[texts[i]][0] for i in range(start, start+len(batch))]
                image = pixels if condition != 'image_swap_same_task' else torch.stack([items[images[i]][1] for i in range(start, start+len(batch))])
                tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
                contexts[condition].append(model.get_context_vector(image.cuda(),
                    tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()).cpu())
        contexts = {k: torch.cat(v) for k, v in contexts.items()}
        groups = {}
        for part in ('train', 'validation'):
            groups[part+'/overall'] = [i for i, r in enumerate(rows) if r['partition'] == part]
            for task in sorted({r['task'] for r in rows}):
                groups[part+'/task/'+task] = [i for i, r in enumerate(rows) if r['partition'] == part and r['task'] == task]
        report = dict(trained=False, weights_updated=False, checkpoint='latest', checkpoint_epoch=checkpoint['epoch'],
            checkpoint_sha256=sha256(args.run/'latest.pt'), reserved_test_targets_read=False,
            training_subset_policy='All fixed train/development windows' if args.all_train_windows else f'First full window of first {args.train_episodes_per_task} source-order episodes per task; no prediction-based selection',
            all_train_windows=args.all_train_windows,
            module_sha256={name: sha256(args.pack/name) for name in ('models.py','adapter.py','diffusion_decoder.py','dataset.py','train.py')},
            window_selection=rows, image_donors=images, language_donors=texts, sampling_seeds=[0,1,2],
            control_input_checks=dict(image_pixel_mse_mean=float(np.mean([
                float(((items[i][1]-items[j][1])**2).mean()) for i,j in enumerate(images)])),
                identical_image_pairs=sum(torch.equal(items[i][1],items[j][1]) for i,j in enumerate(images)),
                different_language_pairs=sum(items[i][0] != items[j][0] for i,j in enumerate(texts))),
            context_relative_change={k: float((contexts[k]-contexts['correct']).norm()/contexts['correct'].norm().clamp(min=1e-8))
                for k in conditions if k != 'correct'}, cases={},
            limits=['Training subset 40 windows is not full-training evaluation.',
                'Corrupted conditioning is diagnostic, not a valid robotic-task benchmark.',
                'One-step noised-target reconstruction has access to target information.',
                'Native generation has final clipping/quaternion normalization; one-step position is raw.',
                'Sensitivity does not identify a unique module bug or prove correct semantic grounding.'])
        if args.chain_probe:
            assert before == json.loads(args.reference_report.read_text())['trainable_state_sha256_after']
            run_chain_probe(model,contexts,items,target,groups,report,args.reference_report,pose_metrics)
        for case in (() if args.chain_probe else ('target_t0','target_t24','target_t49','target_t74','target_t99','pure_noise_t99','native_sample')):
            report['cases'][case] = {}
            correct_predictions = {}
            for condition in conditions:
                results = defaultdict(list)
                for seed in range(3):
                    prediction = np.empty((len(rows),16,7), np.float32)
                    for start in range(0, len(rows), 2):
                        context = contexts[condition][start:start+2].cuda()
                        _, _, current, action, _ = collate_batch(items[start:start+2])
                        clean = action[..., :7].cuda()
                        set_seed(seed*10000 + start)
                        if case == 'native_sample':
                            output = model.sample(context, current.cuda())[...,:7]
                        else:
                            step = int(case.rsplit('t',1)[1])
                            noise = torch.randn_like(clean)
                            alpha = model.alpha_bars[step]
                            noisy = noise if case == 'pure_noise_t99' else alpha.sqrt()*clean + (1-alpha).sqrt()*noise
                            timestep = torch.full((len(clean),),step,device='cuda',dtype=torch.long)
                            output = model.diffusion_decoder(noisy,timestep,context)
                        assert torch.isfinite(output).all()
                        prediction[start:start+len(clean)] = output.cpu().numpy()
                    for group, indices in groups.items():
                        values = pose_metrics(prediction[indices],target[indices,:,:7])
                        if condition != 'correct':
                            changed = pose_metrics(prediction[indices],correct_predictions[seed][indices])
                            values.update(paired_prediction_change_cm=changed['position_cm'],
                                paired_prediction_change_deg=changed['rotation_deg'],
                                paired_prediction_component_mse=changed['pose_component_mse'])
                        results[group].append(values)
                    if condition == 'correct':
                        correct_predictions[seed] = prediction
                report['cases'][case][condition] = dict(results)
                print('PROBE',case,condition,'train',results['train/overall'],
                    'development',results['validation/overall'],flush=True)
        report['alpha_bars'] = {str(t): float(model.alpha_bars[t]) for t in (0,24,49,74,99)}
        report['static_baselines'] = {}
        static = np.zeros_like(target[:,:,:7]); static[:,:,6] = 1
        for group, indices in groups.items():
            report['static_baselines'][group] = pose_metrics(static[indices],target[indices,:,:7])
        after = state_digest(model)
        assert before == after and all(p.grad is None for p in model.parameters())
        report.update(trainable_state_sha256_before=before, trainable_state_sha256_after=after,
            elapsed_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('FROZEN CHAIN PROBE: COMPLETE' if args.chain_probe else 'FROZEN CONDITIONING PROBE: COMPLETE',args.output,flush=True)


if __name__ == '__main__':
    main()
