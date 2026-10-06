"""Matched frozen-context decoder comparison; cluster preflight precedes training."""
import argparse
import gc
import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from cached_diffusion import RobotAdapterModel
from run_full_fit import Sampler, seed, summary
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class AuthorDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ConditionalUnet1D(input_dim=7, global_cond_dim=1024,
            diffusion_step_embed_dim=128, down_dims=[512, 1024, 2048],
            kernel_size=5, n_groups=8, cond_predict_scale=True)

    def forward(self, x, t, context):
        return self.network(x, t, global_cond=context)


def construct(bundle, initial, arm, protocol):
    seed(protocol["initialization_seed"])
    model = RobotAdapterModel(bundle["config"])
    model.load_state_dict(initial, strict=True)
    if arm == "author":
        seed(protocol["initialization_seed"])
        model.diffusion_decoder = AuthorDecoder()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("diffusion_decoder."))
    return model


def noisy_batch(model, data, selected, generator):
    target = data["targets"][selected, :, :7]
    t = torch.randint(0, 100, (len(selected),), generator=generator)
    noise = torch.randn(target.shape, generator=generator)
    a = model.alpha_bars.detach().cpu()[t].reshape(-1, 1, 1)
    return a.sqrt()*target + (1-a).sqrt()*noise, t, target


def update(model, optimizer, noisy, t, target, context, microbatch):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, len(target), microbatch):
        s = slice(start, start+microbatch)
        prediction = model.diffusion_decoder(noisy[s].cuda(), t[s].cuda(), context[s].cuda())
        loss = (prediction-target[s].cuda()).square().mean()
        assert torch.isfinite(loss)
        weighted = loss * (len(target[s])/len(target))
        weighted.backward()
        total += float(weighted.detach())
    norm = nn.utils.clip_grad_norm_(model.diffusion_decoder.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    return total, float(norm)


@contextmanager
def evaluation_state(model):
    """Sampling must not alter training modes or any global RNG stream."""
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            model.eval()
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in modes:
            module.training = training


@torch.no_grad()
def evaluate(model, data, protocol, label="FINAL"):
    device = next(model.parameters()).device
    samples = []
    payload = {"sample_seeds": np.asarray(protocol["sample_seeds"])}
    with evaluation_state(model):
        for sample_seed in protocol["sample_seeds"]:
            result = {}
            for part, d in data.items():
                seed(sample_seed)
                predictions = []
                for start in range(0, len(d["targets"]), protocol["eval_batch"]):
                    s = slice(start, start+protocol["eval_batch"])
                    p = model.sample(d["context"][s].to(device), d["current"][s].to(device))
                    assert torch.isfinite(p).all()
                    predictions.append(p.cpu().numpy().astype(np.float64))
                pred = np.concatenate(predictions)
                target = d["targets"].numpy().astype(np.float64)
                payload.setdefault(part+"_predictions", []).append(pred)
                payload[part+"_targets"] = target
                payload[part+"_indices"] = np.asarray(d["indices"])
                payload[part+"_episode_keys"] = np.asarray(d["episode_keys"])
                payload[part+"_starts"] = np.asarray(d["starts"])
                result[part] = summary(pred, target, np.asarray(d["episode_keys"]), np.asarray(d["starts"]))
                m = result[part]
                print(f"{label} seed={sample_seed} {part} windows={len(pred)} "
                      f"position={m['position_error_cm']:.4f}cm rotation={m['rotation_error_deg']:.3f}deg", flush=True)
            samples.append(dict(sample_seed=sample_seed, partitions=result))
    for part in data:
        payload[part+"_predictions"] = np.stack(payload[part+"_predictions"])
    return samples, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "train"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--arm", choices=["both", "author"], default="both")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--monitor-train", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for name, expected in json.loads((root/"integrity.json").read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    assert torch.cuda.is_available(), "GPU job required; do not run on login node."
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root/"protocol.json").read_text())
    assert protocol["effective_batch"] == 64 and protocol["steps"] == 9875
    assert 64 % protocol["microbatch"] == 0
    protocol_hash = hashlib.sha256((root/"protocol.json").read_bytes()).hexdigest()
    arms = ["project", "author"] if args.arm == "both" else ["author"]
    if args.learning_rate is not None:
        assert args.learning_rate > 0 and np.isfinite(args.learning_rate)
        protocol["learning_rate"] = args.learning_rate
    monitor_steps = [0, 2500, 5000, 7500, 9000, 9500, 9875] if args.monitor_train else []
    if args.arm == "author":
        protocol["walltime_seconds"] = 7200
    settings = dict(arms=arms, learning_rate=protocol["learning_rate"], monitor_steps=monitor_steps,
                    walltime_seconds=protocol["walltime_seconds"])
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    baseline = None
    if args.arm == "author":
        assert args.learning_rate == 3e-5 and args.monitor_train, "Only the planned one-off configuration is supported."
        baseline = json.loads((root/"baseline_reference.json").read_text())
        assert baseline["protocol_sha256"] == protocol_hash
        assert baseline["identical_training_inputs"] and baseline["identical_frozen_gripper"]
    bundle = torch.load(root/"full_context.pt", map_location="cpu", weights_only=False)
    initial = torch.load(root/"shared_initial_weights.pt", map_location="cpu", weights_only=False)
    data = bundle["data"]
    assert bundle["config"]["model"]["diffusion_prediction_type"] == "sample"
    assert bundle["config"]["model"]["beta_schedule"] == "squaredcos_cap_v2"
    assert data["train"]["targets"].shape == (316, 16, 8)
    assert data["validation"]["targets"].shape == (55, 16, 8)
    assert all(d["masks"].eq(1).all() for d in data.values())
    assert not set(data["train"]["indices"]) & set(data["validation"]["indices"])
    assert not set(data["train"]["episode_keys"]) & set(data["validation"]["episode_keys"])
    if args.mode == "train":
        assert args.preflight_report, "A passed GPU preflight report is required."
        pre = json.loads(args.preflight_report.read_text())
        assert pre["passed"] and pre["protocol_sha256"] == protocol_hash
        assert pre.get("run_settings_sha256") == settings_hash, "Preflight settings changed."
        assert pre["gpu_name"] == torch.cuda.get_device_name(), "GPU type changed; repeat preflight."
    reports = []
    for arm in arms:
        print(f"Starting {args.mode}: {arm}", flush=True)
        arm_dir = args.output_dir/arm
        arm_dir.mkdir()
        model = construct(bundle, initial, arm, protocol)
        initial_hash = state_hash(model.state_dict())
        frozen = {k:v.detach().clone() for k,v in model.state_dict().items()
                  if not k.startswith("diffusion_decoder.")}
        frozen_hash = state_hash(frozen)
        if baseline:
            assert initial_hash == baseline["initial_state_sha256"]
            assert frozen_hash == baseline["frozen_state_sha256"]
            # Check the ENTIRE original training input stream before any update.
            check_sampler = Sampler()
            check_generator = torch.Generator().manual_seed(protocol["training_noise_seed"])
            expected_inputs = hashlib.sha256()
            for _ in range(protocol["steps"]):
                selection = check_sampler.take()
                x, ts, y = noisy_batch(model, data["train"], selection, check_generator)
                for value in [selection, ts.numpy(), x.numpy(), y.numpy()]:
                    expected_inputs.update(np.ascontiguousarray(value).tobytes())
            assert expected_inputs.hexdigest() == baseline["paired_inputs_sha256"]
            assert np.all(check_sampler.frequency == 2000)
            print("BASELINE INITIALIZATION, FROZEN PARAMETERS AND FULL INPUT STREAM: PASSED", flush=True)
        parameters = sum(p.numel() for p in model.diffusion_decoder.parameters())
        model.cuda().train()
        optimizer = torch.optim.Adam(model.diffusion_decoder.parameters(), lr=protocol["learning_rate"], weight_decay=0)
        sampler = Sampler()
        generator = torch.Generator().manual_seed(protocol["training_noise_seed"])
        paired_hash = hashlib.sha256()
        history = []
        monitoring = []
        recent_losses, recent_norms = [], []
        monitor_protocol = dict(protocol, sample_seeds=[1101])
        def monitor(step):
            measurements, _ = evaluate(model, {"train": data["train"]}, monitor_protocol, label=f"MONITOR step={step}")
            monitoring.append(dict(step=step, samples=measurements, validation_evaluated=False))
            save(arm_dir/"training_monitor.json", monitoring)
        steps = protocol["steps"] if args.mode == "train" else 4
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        timed = []
        if args.mode == "train" and 0 in monitor_steps:
            monitor(0)
        for step in range(1, steps+1):
            selected = sampler.take()
            noisy, t, target = noisy_batch(model, data["train"], selected, generator)
            for value in [selected, t.numpy(), noisy.numpy(), target.numpy()]:
                paired_hash.update(np.ascontiguousarray(value).tobytes())
            context = data["train"]["context"][selected]
            torch.cuda.synchronize()
            tick = time.monotonic()
            loss, grad_norm = update(model, optimizer, noisy, t, target, context, protocol["microbatch"])
            torch.cuda.synchronize()
            timed.append(time.monotonic()-tick)
            recent_losses.append(loss)
            recent_norms.append(grad_norm)
            if step==1 or step%250==0 or step==steps:
                history.append(dict(step=step, pose_x0_loss=loss, gradient_norm_before_clip=grad_norm,
                    recent_loss_median=float(np.median(recent_losses)), recent_gradient_norm_max=max(recent_norms),
                    recent_batches=len(recent_losses), elapsed_s=time.monotonic()-started))
                save(arm_dir/"history.json", history)
                recent_losses.clear()
                recent_norms.clear()
                print(f"{arm} update={step}/{steps} loss={loss:.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            if args.mode == "train" and step in monitor_steps:
                monitor(step)
        if args.mode == "preflight":
            model.eval()
            # Measure actual 100-step inference on the frozen training inputs.
            torch.cuda.synchronize()
            tick = time.monotonic()
            with torch.no_grad(), evaluation_state(model):
                p = model.sample(data["train"]["context"][:protocol["eval_batch"]].cuda(),
                                 data["train"]["current"][:protocol["eval_batch"]].cuda())
                assert torch.isfinite(p).all()
            torch.cuda.synchronize()
            eval_time = time.monotonic()-tick
            mean_step = float(np.mean(timed[1:]))
            batches = sum((len(d["targets"])+protocol["eval_batch"]-1)//protocol["eval_batch"] for d in data.values())
            estimate = mean_step*protocol["steps"] + eval_time*batches*len(protocol["sample_seeds"])
            train_batches = (316+protocol["eval_batch"]-1)//protocol["eval_batch"]
            estimate += eval_time*train_batches*len(monitor_steps)
            if args.monitor_train:
                # Exercise the real full train-only monitoring path with Adam resident.
                model.train()
                tick = time.monotonic()
                monitor(steps)
                torch.cuda.synchronize()
                measured_monitor = time.monotonic()-tick
                estimate += max(0, measured_monitor-eval_time*train_batches)*len(monitor_steps)
            row = dict(arm=arm, parameters=parameters, peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
                       timed_seconds_per_update=mean_step, timed_seconds_per_sampling_batch=eval_time,
                       estimated_training_and_final_eval_seconds=estimate,
                       run_settings=settings, full_training_stream_matches_baseline=bool(baseline),
                       preflight_optimizer_steps=steps, paired_inputs_sha256=paired_hash.hexdigest(),
                       initial_state_sha256=initial_hash, frozen_state_sha256=frozen_hash)
            for k,v in frozen.items():
                assert torch.equal(model.state_dict()[k].cpu(),v), k
            print(f"RESOURCE {arm}: peak={row['peak_reserved_GiB']:.2f}GiB "
                  f"update={mean_step:.3f}s estimated={estimate/3600:.2f}h", flush=True)
        else:
            assert np.all(sampler.frequency==2000)
            if baseline:
                assert paired_hash.hexdigest() == baseline["paired_inputs_sha256"]
            # Release Adam moments before sampling/serialization to avoid needless memory.
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            samples, payload = evaluate(model, data, protocol)
            np.savez_compressed(arm_dir/"final_predictions.npz", **payload)
            for k,v in frozen.items():
                assert torch.equal(model.state_dict()[k].cpu(),v), k
            torch.save(dict(config=bundle["config"], arm=arm, state_dict=model.state_dict(),
                protocol=protocol, dataset_identity=bundle["dataset_identity"],
                not_a_deployable_checkpoint=True), arm_dir/"final_head.pt")
            row = dict(arm=arm, parameters=parameters, updates=steps, samples=samples,
                initial_state_sha256=initial_hash, frozen_state_sha256=frozen_hash,
                paired_inputs_sha256=paired_hash.hexdigest(), frozen_parameters_unchanged=True,
                min_draws=int(sampler.frequency.min()), max_draws=int(sampler.frequency.max()),
                validation_used_for_optimizer=False, test_targets_used=False,
                elapsed_seconds=time.monotonic()-started)
            row["run_settings"] = settings
            row["full_training_stream_matches_baseline"] = bool(baseline)
            if baseline:
                train_pass = all(s["partitions"]["train"]["position_error_cm"]<=.5 and
                                 s["partitions"]["train"]["rotation_error_deg"]<=5 for s in samples)
                dev_pass = all(m["position_error_cm"]<m["zero_motion_position_error_cm"] and
                               m["rotation_error_deg"]<m["identity_rotation_error_deg"]
                               for s in samples for m in [s["partitions"]["validation"],
                                   *s["partitions"]["validation"]["by_episode"].values()])
                row["decision"] = dict(final_training_pose_threshold_passed=train_pass,
                    development_pose_beats_baseline_every_seed_and_episode=dev_pass,
                    stop_further_learning_rate_search=True, gripper_requires_separate_assessment=True)
            save(arm_dir/"final_metrics.json", row)
        assert all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite final parameters"
        reports.append(row)
        save(arm_dir/"run_record.json", row)
        del model
        if args.mode=="preflight": del optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if len(reports) == 2:
        assert reports[0]["paired_inputs_sha256"]==reports[1]["paired_inputs_sha256"]
        assert reports[0]["frozen_state_sha256"]==reports[1]["frozen_state_sha256"]
    if args.mode=="preflight":
        total_gib=torch.cuda.get_device_properties(0).total_memory/1024**3
        estimated=sum(r["estimated_training_and_final_eval_seconds"] for r in reports)
        passed=all(r["peak_reserved_GiB"]<total_gib*.90 for r in reports) and estimated<protocol["walltime_seconds"]*.75
        result=dict(passed=passed, gpu_name=torch.cuda.get_device_name(), gpu_total_GiB=total_gib,
            protocol_sha256=protocol_hash, estimated_pair_seconds=estimated, arms=reports,
            run_settings_sha256=settings_hash, run_settings=settings,
            scope="Disposable resource check; four optimizer steps per arm, no learned conclusions or reused weights.")
        save(args.output_dir/"preflight.json", result)
        print(f"PREFLIGHT passed={passed} estimated_pair={estimated/3600:.2f}h", flush=True)
        assert passed, "Resource headroom/estimated walltime check failed; do not submit training."
    else:
        result=dict(protocol=protocol, protocol_sha256=protocol_hash, arms=reports,
            identical_training_inputs=True, identical_frozen_gripper=True,
            independent_initial_parameters_expected=True, independent_training_runs=1,
            scope="Matched decoder-architecture diagnostic; not full author-policy reproduction or zero-shot success.")
        result["run_settings"] = settings
        result["run_settings_sha256"] = settings_hash
        if baseline:
            result["baseline_reference"] = baseline
            result.pop("independent_initial_parameters_expected")
            result["scope"] = "One author learning-rate follow-up; no architecture comparison or zero-shot success claim."
        save(args.output_dir/"pair_results.json", result)
    print(f"{args.mode.upper()} COMPLETE: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
