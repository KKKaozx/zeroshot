"""Frozen training-window probe of noise level and image conditioning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def select_training(rows, per_task=8):
    selected, seen, counts = [], set(), {}
    for row in rows:
        if row["partition"] != "train":
            continue
        episode = (row["shard"], row["record_index"])
        task = row["task"]
        if episode in seen or counts.get(task, 0) >= per_task:
            continue
        selected.append(row)
        seen.add(episode)
        counts[task] = counts.get(task, 0) + 1
    if len(counts) != 5 or any(value != per_task for value in counts.values()):
        raise ValueError("Expected eight independent training demonstrations per task")
    donors = []
    for index, row in enumerate(selected):
        donors.append(next(i for i, other in enumerate(selected)
                           if other["task"] == row["task"] and i != index))
    return selected, donors


def restoration_comparison(noisy_raw, noisy_pose, output_pose, target, pose_errors):
    """Compare the same noisy input before and after the learned restoration."""
    input_position, input_rotation = pose_errors(noisy_pose, target)
    output_position, output_rotation = pose_errors(output_pose, target)
    return {
        "input_position_cm": float(input_position.mean()),
        "input_rotation_deg": float(input_rotation.mean()),
        "position_improvement_cm": float((input_position-output_position).mean()),
        "rotation_improvement_deg": float((input_rotation-output_rotation).mean()),
        "position_improved_target_fraction": float((output_position < input_position).mean()),
        "rotation_improved_target_fraction": float((output_rotation < input_rotation).mean()),
        "input_raw_xyz_mse": float(((noisy_raw[..., :3]-target[..., :3])**2).mean()),
        "input_raw_quaternion_mse": float(((noisy_raw[..., 3:]-target[..., 3:])**2).mean()),
    }


def traced_ddim(torch, model, context, noise, ddim_sample):
    """Observe one unchanged deterministic chain; remove hook on all exits."""
    trace, calls = {}, []
    def observe(module, inputs, output):
        step = int(inputs[1][0])
        calls.append(step)
        if step in (99, 49, 24, 9, 0):
            trace[step] = (inputs[0].detach().cpu().clone(), output.detach().cpu().clone())
    hook = model.diffusion_decoder.register_forward_hook(observe)
    try:
        final = ddim_sample(torch, model, context, noise, 100)
    finally:
        hook.remove()
    if calls != list(range(99, -1, -1)):
        raise ValueError("Expected a full 100-step DDIM chain")
    return final, trace


def traced_native(model, context, current, steps):
    """Observe the production sampler without reimplementing its equations."""
    trace, calls = {}, []
    def observe(module, inputs, output):
        step = int(inputs[1][0])
        calls.append(step)
        if step in steps:
            trace[step] = (inputs[0].detach().cpu().clone(), output.detach().cpu().clone())
    hook = model.diffusion_decoder.register_forward_hook(observe)
    try:
        final = model.sample(context, current)
    finally:
        hook.remove()
    if calls != list(range(model.num_diffusion_steps - 1, -1, -1)) or set(trace) != set(steps):
        raise ValueError('Production sampler trace does not match the requested steps')
    return final, trace


def capture_native_noise(torch, model, context, current):
    """Record actual production draws; pass each draw through unchanged."""
    from unittest.mock import patch
    trace, noises, active = {}, {}, []
    original = torch.randn_like
    def observe(module, inputs, output):
        step = int(inputs[1][0])
        active.append(step)
        trace[step] = (inputs[0].detach().cpu().clone(),output.detach().cpu().clone())
    def record(tensor, *args, **kwargs):
        value = original(tensor, *args, **kwargs)
        if not active or active[-1] in noises:
            raise ValueError('Unexpected noise draw in production sampler')
        noises[active[-1]] = value.detach().cpu().clone()
        return value
    hook = model.diffusion_decoder.register_forward_hook(observe)
    try:
        with patch.object(torch,'randn_like',side_effect=record):
            final = model.sample(context,current)
    finally:
        hook.remove()
    if active != list(range(model.num_diffusion_steps-1,-1,-1)) or set(noises)!=set(range(1,model.num_diffusion_steps)):
        raise ValueError('Sampler decoder/noise call contract changed')
    return final,trace,noises


def export_scheduler_inputs(args,model,contexts,current,selected,source):
    import torch
    from train import set_seed,trainable_state_dict
    from run_bridge_official_unet_training import digest
    from evaluate_bridge_one_step_sampler import sha256
    if not model.clip_denoised or model.num_diffusion_steps!=100:
        raise ValueError('Expected clipped archived DDPM100 x0 sampler')
    before=digest(trainable_state_dict(model))
    before_clip=digest({k:v for k,v in model.state_dict().items() if k.startswith(('vision_encoder.','text_encoder.'))})
    if before_clip!=source['protocol']['clip_sha256']:
        raise ValueError('Frozen CLIP hash differs')
    shape=(100,len(selected),16,7)
    inputs=np.empty(shape,np.float32); estimates=np.empty_like(inputs); noises=np.zeros_like(inputs)
    final_poses=np.empty(shape[1:],np.float32)
    with torch.inference_mode():
        for start in range(0,len(selected),2):
            context=contexts['correct'][start:start+2].cuda(); measured=current[start:start+2].cuda()
            set_seed(start)
            final,trace,draws=capture_native_noise(torch,model,context,measured)
            after_rng=torch.cuda.get_rng_state().clone()
            set_seed(start)
            replay=model.sample(context,measured)
            if not torch.equal(final,replay) or not torch.equal(after_rng,torch.cuda.get_rng_state()):
                raise ValueError('Noise capture changed production sampling')
            final_poses[start:start+len(context)]=final[...,:7].cpu().numpy()
            for step,(latent,x0) in trace.items():
                inputs[step,start:start+len(context)]=latent.numpy()
                estimates[step,start:start+len(context)]=x0.numpy()
                if step: noises[step,start:start+len(context)]=draws[step].numpy()
    if digest(trainable_state_dict(model))!=before or digest({k:v for k,v in model.state_dict().items() if k.startswith(('vision_encoder.','text_encoder.'))})!=before_clip:
        raise ValueError('Live model changed')
    if sha256(args.training_run/'final.pt')!=source['checkpoint']['sha256']:
        raise ValueError('Saved checkpoint changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tensors=args.output.with_suffix('.npz')
    np.savez_compressed(tensors,inputs=inputs,estimates=estimates,noises=noises,final_poses=final_poses,
                        betas=model.betas.cpu().numpy(),alpha_bars=model.alpha_bars.cpu().numpy(),
                        variance=model.posterior_variance.cpu().numpy(),
                        coeff_x0=model.posterior_mean_x0.cpu().numpy(),coeff_xt=model.posterior_mean_xt.cpu().numpy())
    report={'stage':'production_scheduler_tensor_export','execution_checks_passed':True,
            'trained':False,'weights_updated':False,'reserved_test_targets_read':False,
            'checkpoint_sha256':source['checkpoint']['sha256'],'training_report_sha256':sha256(args.training_report),
            'window_selection':selected,'sampling_seed':0,'tensors_path':str(tensors),'tensors_sha256':sha256(tensors),
            'position_limit':float(model.max_normalized_position or 3),'capture_replay_exact':True,
            'clip_unchanged':True,'parameters_unchanged':True,'core_module_sha256':source['protocol']['module_sha256']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PRODUCTION SCHEDULER TENSORS EXPORTED',tensors,flush=True)


def run_input_origin_probe(args, model, contexts, current, targets, selected, source, module_hashes, initial_hash):
    import torch
    from train import set_seed, trainable_state_dict
    from run_bridge_official_unet_training import digest
    from evaluate_bridge_one_step_sampler import finish_pose, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors

    if model.num_diffusion_steps != 100:
        raise ValueError('This diagnostic expects the archived 100-step model')
    steps = (99, 74, 49, 24, 9, 0)
    truth = targets[..., :7].numpy()
    limit = float(model.max_normalized_position or 3)
    state_hash = digest(trainable_state_dict(model))
    frozen = lambda: {k:v for k,v in model.state_dict().items()
                      if k.startswith(('vision_encoder.', 'text_encoder.'))}
    clip_hash = digest(frozen())
    if clip_hash != source['protocol']['clip_sha256']:
        raise ValueError('Frozen CLIP differs from the archived training protocol')
    groups = {'overall': list(range(len(selected)))}
    groups.update({'task/'+task: [i for i,r in enumerate(selected) if r['task']==task]
                   for task in sorted({r['task'] for r in selected})})
    records = {name: [] for name in groups}
    per_window = []
    replay_max = 0.0
    began = time.monotonic()
    with torch.inference_mode():
        for seed in range(3):
            arrays = {kind: {step: {'input':np.empty_like(truth), 'estimate':np.empty_like(truth)}
                            for step in steps} for kind in ('forward_noised_truth', 'native_chain')}
            finals = np.empty_like(truth)
            for start in range(0, len(selected), 2):
                context = contexts['correct'][start:start+2].cuda()
                measured = current[start:start+2].cuda()
                target = targets[start:start+2, :, :7].cuda()
                set_seed(seed*10000+start)
                final, trace = traced_native(model, context, measured, steps)
                finals[start:start+len(context)] = final[..., :7].cpu().numpy()
                # Unobserved replay must match the hooked sampler and RNG consumption.
                rng_after = torch.cuda.get_rng_state().clone()
                set_seed(seed*10000+start)
                replay = model.sample(context, measured)
                replay_max = max(replay_max, float((replay-final).abs().max()))
                if not torch.equal(rng_after, torch.cuda.get_rng_state()) or replay_max != 0:
                    raise ValueError('Observing the sampler changed its result or RNG stream')
                # Independent generator: scoring inputs cannot enter or perturb generation.
                generator = torch.Generator(device='cuda').manual_seed(700000+seed*10000+start)
                noise = torch.randn(target.shape, device='cuda', generator=generator)
                for step in steps:
                    alpha = model.alpha_bars[step]
                    noisy = alpha.sqrt()*target+(1-alpha).sqrt()*noise
                    timestep = torch.full((len(context),),step,device='cuda',dtype=torch.long)
                    estimate = model.diffusion_decoder(noisy,timestep,context)
                    for kind, values in (('forward_noised_truth',(noisy,estimate)),('native_chain',trace[step])):
                        for name,value in zip(('input','estimate'), values):
                            if not torch.isfinite(value).all():
                                raise ValueError(f'Nonfinite {kind}/{step}/{name}')
                            arrays[kind][step][name][start:start+len(context)] = value.cpu().numpy()
            for group, ids in groups.items():
                result = {'seed':seed,'steps':{}}
                for step in steps:
                    result['steps'][str(step)] = {}
                    for kind in arrays:
                        raw = arrays[kind][step]['estimate'][ids]
                        latent = arrays[kind][step]['input'][ids]
                        clean = finish_pose(torch,torch.from_numpy(raw),limit).numpy()
                        position,rotation = pose_errors(clean,truth[ids])
                        result['steps'][str(step)][kind] = {
                            'position_cm':float(position.mean()),'rotation_deg':float(rotation.mean()),
                            'raw_xyz_mse':float(((raw[...,:3]-truth[ids,...,:3])**2).mean()),
                            'raw_quaternion_mse':float(((raw[...,3:]-truth[ids,...,3:])**2).mean()),
                            'estimate_xyz_clip_fraction':float((np.abs(raw[...,:3])>limit).mean()),
                            'input_xyz_mean':latent[...,:3].mean(axis=(0,1)).tolist(),
                            'input_xyz_std':latent[...,:3].std(axis=(0,1)).tolist(),
                            'input_quaternion_norm_mean':float(np.linalg.norm(latent[...,3:],axis=-1).mean()),
                        }
                pos,rot = pose_errors(finals[ids],truth[ids])
                result['native_final']={'position_cm':float(pos.mean()),'rotation_deg':float(rot.mean())}
                records[group].append(result)
            for i,row in enumerate(selected):
                window={'seed':seed,'window':row,'steps':{}}
                for step in steps:
                    window['steps'][str(step)]={}
                    for kind in arrays:
                        clean=finish_pose(torch,torch.from_numpy(arrays[kind][step]['estimate'][i:i+1]),limit).numpy()
                        pos,rot=pose_errors(clean,truth[i:i+1])
                        window['steps'][str(step)][kind]={'position_cm':float(pos.mean()),'rotation_deg':float(rot.mean())}
                per_window.append(window)
            print('INPUT_ORIGIN seed',seed,'complete',len(selected),'training windows',flush=True)
    if (digest(trainable_state_dict(model)) != state_hash or digest(frozen()) != clip_hash
            or sha256(args.training_run/'final.pt') != initial_hash):
        raise ValueError('Model parameters or checkpoint changed')
    summary = {}
    for step in steps:
        summary[str(step)] = {kind:{key:float(np.mean([r['steps'][str(step)][kind][key] for r in records['overall']]))
                                   for key in ('position_cm','rotation_deg')}
                              for kind in arrays}
        print('INPUT_ORIGIN t'+str(step),json.dumps(summary[str(step)]),flush=True)
    report={'stage':'frozen_native_ddpm_input_origin','execution_checks_passed':True,
            'trained':False,'weights_updated':False,'validation_payload_read':False,'reserved_test_targets_read':False,
            'training_report_sha256':sha256(args.training_report),'checkpoint_sha256':initial_hash,
            'probe_sha256':sha256(Path(__file__)),
            'module_sha256':module_hashes,'window_selection':selected,'sampling_seeds':[0,1,2],
            'trace_steps':list(steps),'sampler':'unchanged production model.sample, DDPM100',
            'hook_replay_max_difference':replay_max,'parameters_unchanged':True,'clip_unchanged':True,
            'groups':records,'per_window':per_window,'overall_seed_mean':summary,
            'target_xyz_outside_sampling_bound_fraction':float((np.abs(truth[...,:3])>limit).mean()),
            'target_window_any_xyz_outside_bound_fraction':float((np.abs(truth[...,:3])>limit).any(axis=(1,2)).mean()),
            'elapsed_seconds':time.monotonic()-began,
            'limits':['Forty training windows, one per demonstration, eight per task; not the full training population.',
                      'Forward-noised truth uses target actions: reconstruction diagnostic, never a deployable policy.',
                      'Native chain generation receives only observation context and current gripper, never target actions.',
                      'Different input origins can explain an error gap; this does not establish a unique causal defect.',
                      'Statistics describe sampled latents; physical error scores apply to predicted x0 after existing pose constraints.',
                      'Matched demonstration error cannot adjudicate alternative valid trajectories or task success.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('FROZEN INPUT ORIGIN CHECK COMPLETE',args.output,flush=True)


def run_tail_probe(args, model, contexts, targets, selected, donors, source, module_hashes, initial_hash):
    import torch
    from train import set_seed
    from evaluate_bridge_ddim_steps import ddim_sample
    from evaluate_bridge_one_step_sampler import finish_pose, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors

    truth = targets[..., :7].numpy()
    started = time.monotonic()
    steps = (99, 49, 24, 9, 0)
    groups = {"overall": list(range(len(selected)))}
    groups.update({"task/" + task: [i for i,r in enumerate(selected) if r["task"] == task]
                   for task in sorted({r["task"] for r in selected})})
    results = {group: [] for group in groups}
    limit = float(model.max_normalized_position or 3)
    with torch.inference_mode():
        for seed in range(3):
            inputs = {step: np.empty_like(truth) for step in steps}
            x0s = {step: np.empty_like(truth) for step in steps}
            final = np.empty_like(truth)
            for start in range(0, len(selected), 2):
                context = contexts["correct"][start:start + 2].cuda()
                set_seed(seed * 10000 + start)
                noise = torch.randn(len(context), 16, 7, device="cuda")
                completed, trace = traced_ddim(torch, model, context, noise, ddim_sample)
                final[start:start + len(context)] = finish_pose(torch, completed, limit).cpu().numpy()
                for step in steps:
                    noisy, estimate = trace[step]
                    inputs[step][start:start + len(context)] = finish_pose(torch, noisy, limit).numpy()
                    x0s[step][start:start + len(context)] = finish_pose(torch, estimate, limit).numpy()
            for group, ids in groups.items():
                def score(value):
                    pos, rot = pose_errors(value[ids], truth[ids])
                    return {"position_cm": float(pos.mean()), "rotation_deg": float(rot.mean())}
                final_score = score(final)
                stage_results = {}
                for step in steps:
                    input_score, x0_score = score(inputs[step]), score(x0s[step])
                    input_pos, input_rot = pose_errors(inputs[step][ids], truth[ids])
                    final_pos, final_rot = pose_errors(final[ids], truth[ids])
                    stage_results[str(step)] = {
                        "input_before_step": input_score,
                        "clean_estimate_at_step": x0_score,
                        "final_minus_input_position_cm": final_score["position_cm"]-input_score["position_cm"],
                        "final_minus_input_rotation_deg": final_score["rotation_deg"]-input_score["rotation_deg"],
                        "final_minus_estimate_position_cm": final_score["position_cm"]-x0_score["position_cm"],
                        "final_minus_estimate_rotation_deg": final_score["rotation_deg"]-x0_score["rotation_deg"],
                        "final_worse_than_input_position_fraction": float((final_pos > input_pos).mean()),
                        "final_worse_than_input_rotation_fraction": float((final_rot > input_rot).mean()),
                    }
                results[group].append({"seed": seed, "final": final_score, "steps": stage_results})
            print("TAIL_SEED", seed, json.dumps(results["overall"][-1]), flush=True)
    checkpoint_path = args.training_run / "final.pt"
    if sha256(checkpoint_path) != initial_hash:
        raise ValueError("Checkpoint changed")
    report = {
        "stage": "frozen_multiscale_ddim_tail_probe", "trained": False, "weights_updated": False,
        "reserved_test_targets_read": False, "validation_payload_read": False,
        "training_report_sha256": sha256(args.training_report), "checkpoint_sha256": initial_hash,
        "reference_probe_sha256": sha256(args.reference_probe), "module_sha256": module_hashes,
        "window_selection": selected, "image_donor_indices": donors, "sampling_seeds": [0,1,2],
        "sampler": "100-step DDIM eta=0, unchanged existing ddim_sample implementation",
        "trace_steps": list(steps), "groups": results, "elapsed_seconds": time.monotonic()-started,
        "difference_sign": "Positive final-minus-input/estimate means continuing increases target error.",
        "limits": [
            "Same forty training windows; no development or test payloads.",
            "Scoring uses target data, but generation never receives target actions.",
            "Input snapshot is still a noisy latent with final pose constraints; estimate snapshot is predicted x0.",
            "These snapshots are offline ablations, not executable early-stop policies.",
            "Tail results concern this deterministic DDIM chain, not every sampler or training parameterization.",
            "No timing or checkpoint is chosen as best on these diagnostic results.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN MULTISCALE DDIM TAIL PROBE: PASSED", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "training-run", "training-report", "window-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--chain-tail", action="store_true")
    modes.add_argument("--input-origin", action="store_true")
    modes.add_argument("--scheduler-export", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.chain_tail and args.reference_probe is None:
        raise ValueError("Chain-tail probe requires a prior reconstruction report")
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(args.pack.resolve()))
    sys.path.insert(0, str(script_dir))
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import torch
    from transformers import CLIPTokenizer
    from models import RobotAdapterModel
    from dataset import UnifiedRobotDataset
    from train import bridge_plan_selection, bridge_plan_splits, collate_batch, set_seed, trainable_state_dict
    from evaluate_bridge_one_step_sampler import finish_pose, select_windows, sha256
    from evaluate_bridge_endpoint_metrics import pose_errors

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = json.loads(args.training_report.read_text())
    reference = json.loads(args.window_reference.read_text())
    if not source["passed"] or source["stage"] != "multiscale_x0_training":
        raise ValueError("Invalid source training report")
    if source["protocol"]["window_reference_sha256"] != sha256(args.window_reference):
        raise ValueError("Window reference differs")
    module_hashes = {name: sha256((script_dir if name in {"models.py", "diffusion_decoder.py"}
                                   else args.pack) / name)
                     for name in source["protocol"]["module_sha256"]}
    if module_hashes != source["protocol"]["module_sha256"]:
        raise ValueError("Model/data modules differ from training")
    selection = bridge_plan_selection(args.pack / "manifest.json")
    dataset = UnifiedRobotDataset(
        data_dir=str(args.pack / "data"), chunk_size=16, stride=4,
        sources=["tfrecord"], min_trajectory_steps=17, exclude_path_parts=[],
        exclude_schemas=[], tfrecord_splits=["train"],
        bridge_gripper_policy="reverse_scan_valid_steps_v2",
        bridge_current_gripper="continuous", bridge_episode_selection=selection)
    rows = select_windows(dataset, bridge_plan_splits(dataset), selection)
    if rows != reference["window_selection"]:
        raise ValueError("Population differs from frozen reference")
    selected, donors = select_training(rows)
    prior_probe = None
    if args.reference_probe is not None:
        prior_probe = json.loads(args.reference_probe.read_text())
        if (prior_probe["window_selection"] != selected
                or prior_probe["image_donor_indices"] != donors
                or prior_probe["checkpoint_sha256"] != source["checkpoint"]["sha256"]):
            raise ValueError("Probe subset or checkpoint differs from 194940")
    items = [dataset[row["dataset_index"]] for row in selected]
    language, images, current, targets, masks = collate_batch(items)
    if not (masks[..., :7] > .5).all():
        raise ValueError("Probe requires fully supervised pose targets")
    if not (args.input_origin or args.scheduler_export) and any(torch.equal(images[i], images[j]) for i, j in enumerate(donors)):
        raise ValueError("A donor image is identical after production preprocessing")
    checkpoint_path = args.training_run / "final.pt"
    initial_hash = sha256(checkpoint_path)
    if initial_hash != source["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != source["protocol"]:
        raise ValueError("Checkpoint protocol differs")
    model = RobotAdapterModel(checkpoint["config"])
    if model.diffusion_architecture != "multiscale" or model.diffusion_prediction_type != "sample":
        raise ValueError("Expected multiscale x0 model")
    if set(checkpoint["model"]) != set(trainable_state_dict(model)):
        raise ValueError("Checkpoint keys differ")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.cuda().eval()
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint["config"]["model"]["name"], local_files_only=True)
    contexts = {"correct": []} if args.chain_tail or args.input_origin or args.scheduler_export else {"correct": [], "image_swap": []}
    with torch.inference_mode():
        for start in range(0, len(items), 2):
            ids = list(range(start, min(start + 2, len(items))))
            tokens = tokenizer([language[i] for i in ids], padding=True, truncation=True, return_tensors="pt")
            for kind in contexts:
                image_ids = ids if kind == "correct" else [donors[i] for i in ids]
                contexts[kind].append(model.get_context_vector(
                    images[image_ids].cuda(), tokens["input_ids"].cuda(), tokens["attention_mask"].cuda()).cpu())
    contexts = {key: torch.cat(value) for key, value in contexts.items()}
    if args.scheduler_export:
        export_scheduler_inputs(args,model,contexts,current,selected,source)
        return
    if args.input_origin:
        run_input_origin_probe(args, model, contexts, current, targets, selected, source, module_hashes, initial_hash)
        return
    if args.chain_tail:
        run_tail_probe(args, model, contexts, targets, selected, donors, source, module_hashes, initial_hash)
        return
    truth = targets[..., :7].numpy()
    cases = {}
    began = time.monotonic()
    with torch.inference_mode():
        for label, step in [("t0", 0), ("t24", 24), ("t49", 49), ("t74", 74), ("t99", 99), ("pure_noise_t99", 99)]:
            kinds = ("correct", "image_swap") if label == "pure_noise_t99" else ("correct",)
            cases[label] = {}
            outputs = {}
            for kind in kinds:
                seeds = []
                for seed in range(3):
                    raw = np.empty_like(truth)
                    clean = np.empty_like(truth)
                    input_raw = np.empty_like(truth)
                    input_pose = np.empty_like(truth)
                    for start in range(0, len(items), 2):
                        target = targets[start:start + 2, :, :7].cuda()
                        set_seed(seed * 10000 + start)
                        noise = torch.randn_like(target)
                        alpha = model.alpha_bars[step]
                        noisy = noise if label == "pure_noise_t99" else alpha.sqrt() * target + (1-alpha).sqrt() * noise
                        timestep = torch.full((len(target),), step, device="cuda", dtype=torch.long)
                        value = model.diffusion_decoder(noisy, timestep, contexts[kind][start:start + 2].cuda())
                        if not torch.isfinite(value).all():
                            raise ValueError("Nonfinite output")
                        raw[start:start + len(target)] = value.cpu().numpy()
                        clean[start:start + len(target)] = finish_pose(torch, value, float(model.max_normalized_position or 3)).cpu().numpy()
                        input_raw[start:start + len(target)] = noisy.cpu().numpy()
                        input_pose[start:start + len(target)] = finish_pose(torch, noisy, float(model.max_normalized_position or 3)).cpu().numpy()
                    seeds.append((raw, clean, input_raw, input_pose))
                outputs[kind] = seeds
                groups = {"overall": list(range(len(items)))}
                groups.update({"task/" + task: [i for i,r in enumerate(selected) if r["task"] == task]
                               for task in sorted({r["task"] for r in selected})})
                cases[label][kind] = {}
                for group, ids in groups.items():
                    scores = []
                    for seed, (raw, clean, input_raw, input_pose) in enumerate(seeds):
                        pos, rot = pose_errors(clean[ids], truth[ids])
                        value = {"seed": seed, "position_cm": float(pos.mean()), "rotation_deg": float(rot.mean()),
                                 "raw_component_mse": float(((raw[ids]-truth[ids])**2).mean()),
                                 "raw_xyz_mse": float(((raw[ids,:,:3]-truth[ids,:,:3])**2).mean()),
                                 "raw_quaternion_mse": float(((raw[ids,:,3:]-truth[ids,:,3:])**2).mean())}
                        value.update(restoration_comparison(
                            input_raw[ids], input_pose[ids], clean[ids], truth[ids], pose_errors))
                        norms = np.linalg.norm(raw[ids, :, 3:], axis=-1)
                        value["output_raw_quaternion_norm_mean"] = float(norms.mean())
                        value["output_degenerate_quaternions"] = int((norms <= 1e-6).sum())
                        if kind == "image_swap":
                            change = clean[ids] - outputs["correct"][seed][1][ids]
                            value["paired_position_change_cm"] = float(np.linalg.norm(change[..., :3], axis=-1).mean()*10)
                        scores.append(value)
                    cases[label][kind][group] = scores
                overall = cases[label][kind]["overall"]
                mean = lambda key: float(np.mean([row[key] for row in overall]))
                print("RESTORATION", label, kind,
                      "input=", round(mean("input_position_cm"), 4), round(mean("input_rotation_deg"), 4),
                      "output=", round(mean("position_cm"), 4), round(mean("rotation_deg"), 4),
                      "improvement=", round(mean("position_improvement_cm"), 4), round(mean("rotation_improvement_deg"), 4),
                      flush=True)
    if sha256(checkpoint_path) != initial_hash:
        raise ValueError("Checkpoint changed")
    static = np.zeros_like(truth)
    static[..., 6] = 1
    pos, rot = pose_errors(static, truth)
    report = {"stage": "frozen_multiscale_noise_image_probe", "trained": False, "weights_updated": False,
              "schema_version": 2,
              "input_baseline": "Same noisy input with production position clipping and quaternion normalization; positive improvement means output error is lower.",
              "reference_probe_sha256": sha256(args.reference_probe) if args.reference_probe is not None else None,
              "reserved_test_targets_read": False, "validation_payload_read": False,
              "training_report_sha256": sha256(args.training_report), "checkpoint_sha256": initial_hash,
              "module_sha256": module_hashes, "window_selection": selected, "image_donor_indices": donors,
              "training_windows": len(selected), "sampling_seeds": [0,1,2],
              "alpha_bars": {str(t): float(model.alpha_bars[t]) for t in [0,24,49,74,99]},
              "static": {"position_cm": float(pos.mean()), "rotation_deg": float(rot.mean())},
              "cases": cases, "elapsed_seconds": time.monotonic()-began,
              "limits": ["Forty fixed training windows, one per demonstration; not full training evaluation.",
                         "Noisy-target cases contain ground truth and are reconstruction diagnostics only.",
                         "Image swaps test the use of the observed scene, not semantic correctness.",
                         "No automatic threshold can by itself establish a unique root cause."]}
    if prior_probe is not None:
        report["prior_overall_metric_differences"] = {
            label: {
                kind: [{key: row[key]-prior_probe["cases"][label][kind]["overall"][seed][key]
                        for key in ("position_cm", "rotation_deg")}
                       for seed, row in enumerate(groups["overall"])]
                for kind, groups in variants.items()
            }
            for label, variants in cases.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FROZEN MULTISCALE NOISE/IMAGE PROBE: PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()
