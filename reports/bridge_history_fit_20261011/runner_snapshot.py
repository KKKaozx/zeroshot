"""Bounded two-observation control; production modules remain unchanged.

Both arms have the same shared Adapter and zero-initialized history projection.
Only the older image changes: repeated current image versus actual frame t-1.
Preflight is an interface/resource check, never an improvement verdict.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
from PIL import Image
import torch
from transformers import CLIPTokenizer
from dataset import UnifiedRobotDataset, prepare_image
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest, evaluate, seed

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def select_training(rows, count):
    """Task round-robin, one middle eligible window per episode; no target ranking."""
    episodes = {}
    for row in rows:
        if row["partition"] == "train" and row["start_index"] > 0:
            episodes.setdefault((row["shard"], row["record_index"]), []).append(row)
    pools = {}
    for key in sorted(episodes):
        windows = sorted(episodes[key], key=lambda r: r["start_index"])
        row = copy.deepcopy(windows[len(windows) // 2])
        task = " ".join(row["task"].lower().split())
        pools.setdefault(task, []).append(row)
    selected = []
    while len(selected) < count:
        progressed = False
        for task in sorted(pools):
            if pools[task] and len(selected) < count:
                selected.append(pools[task].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("Insufficient distinct training episodes with real history")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "fit"], default="preflight")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--data", type=Path, default=Path("E:/dataset/bridge_v2_0.0.1/0.0.1"))
    parser.add_argument("--clip", type=Path, default=Path("D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"))
    args = parser.parse_args()
    if args.windows < 5 or args.batch_size < 2 or args.batch_size > args.windows or args.updates < 1:
        parser.error("Require windows>=5, 2<=batch-size<=windows, updates>=1")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(stage=args.stage, execution_checks_passed=False, effect_verdict=None,
                  validation_evaluated=False, reserved_test_targets_used=False,
                  simulation_executed=False, downloads=False, checkpoints_saved=False)
    sources = {n: sha(ROOT / "code" / n) for n in
               ["dataset.py", "models.py", "adapter.py", "diffusion_decoder.py",
                "replay_and_fit_bridge_positive.py", "evaluate_bridge_endpoint_metrics.py",
                "check_bridge_history_pair.py"]}
    report["source_sha256"] = sources
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        manifest_path = ROOT / "reports/bridge-xyz-eval-197349.json"
        reference = json.loads(manifest_path.read_text())
        rows = select_training(reference["window_selection"], args.windows)
        validation_episodes = {(r["shard"], r["record_index"]) for r in reference["window_selection"]
                               if r["partition"] == "validation"}
        assert not validation_episodes.intersection((r["shard"], r["record_index"]) for r in rows)
        config_path = ROOT / "reports/bridge_joint_both_events_1000_local_20261011/plan.json"
        config = copy.deepcopy(json.loads(config_path.read_text())["config"])
        config["model"]["name"] = str(args.clip)
        updates = 6 if args.stage == "preflight" else args.updates
        plan = dict(selection=rows, manifest_sha256=sha(manifest_path),
                    config_source_sha256=sha(config_path), config=config,
                    arms=["repeat_current", "real_previous"], updates_per_arm=updates,
                    batch_size=args.batch_size, optimizer="Adam", lr=1e-4,
                    weight_decay=0, schedule="constant", gradient_clip=1.,
                    train_seed=42, adapter_dropout=.1, loss="Pose x0 MSE only; gripper frozen",
                    history="Same episode image at t-1; current observation t; targets t+1..t+16",
                    context="current_adapter_context + W(previous_adapter_context-current_adapter_context)",
                    history_projection_initialization="W=0; both initial contexts exactly equal current-only",
                    changing_variable="Older image: current image repeated or actual previous frame",
                    sample_steps=100, sample_seeds=[1101, 1102, 1103],
                    purpose="Resource/interface preflight" if args.stage == "preflight" else
                    "16-window multi-task train-only fitting pilot; no generalization verdict",
                    frame_interval_seconds="Unknown; one recorded step, not an assumed camera rate",
                    output_policy="Reports/predictions only; no new large weights or token cache")
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        data = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        data.chunk_size = 16
        data.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
        data.bridge_current_gripper = "continuous"
        images, previous, targets, currents, instructions, lineage = [], [], [], [], [], []
        load = UnifiedRobotDataset._load_tfrecord_example
        for row in rows:
            path = str(args.data / row["shard"])
            example = load(path, row["record_index"])
            data._load_tfrecord_example = lambda *_: example
            sample = dict(file_path=path, record_index=row["record_index"], start_index=row["start_index"])
            instruction, image, current, target = data._get_tfrecord_bridge_state_action(sample)
            assert " ".join(instruction.lower().split()) == " ".join(row["task"].lower().split())
            raw = example.features.feature["steps/observation/image_0"].bytes_list.value
            t = int(row["start_index"])
            assert 0 <= t - 1 < t and t + 16 < len(raw)
            def frame(index):
                return prepare_image(np.asarray(Image.open(io.BytesIO(raw[index])).convert("RGB")))
            assert torch.equal(frame(t), image)
            past = frame(t - 1)
            images.append(image); previous.append(past); targets.append(target); currents.append(current)
            instructions.append(instruction)
            lineage.append(dict(shard=row["shard"], record_index=row["record_index"],
                                previous_index=t-1, current_index=t, target_indices=list(range(t+1,t+17)),
                                current_image_sha256=tensor_sha(image), previous_image_sha256=tensor_sha(past)))
        images, previous = torch.stack(images), torch.stack(previous)
        targets, currents = torch.stack(targets), torch.stack(currents)
        report["data_checks"] = dict(train_windows=len(rows), train_episodes=len(rows),
            task_counts={task: sum(" ".join(s.lower().split()) == task for s in instructions)
                         for task in sorted(set(" ".join(s.lower().split()) for s in instructions))},
            lineage=lineage, no_future_observations=True, no_heldout_episode_overlap=True,
            targets_shared_sha256=tensor_sha(targets), current_shared_sha256=tensor_sha(currents),
            history_pixel_mean_absolute_difference=float((images-previous).abs().mean()))
        seed(42)
        model = RobotAdapterModel(config).cuda().eval()
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        text = {k: v.cuda() for k,v in tokenizer(instructions, padding=True, truncation=True,
                                               max_length=77, return_tensors="pt").items()}
        with torch.no_grad():
            language = model.text_encoder(**text).last_hidden_state.detach()
            def encode(values):
                return torch.cat([model.vision_encoder(pixel_values=values[i:i+2].cuda()).last_hidden_state.detach()
                                  for i in range(0,len(values),2)])
            visual, older = encode(images), encode(previous)
        model.vision_encoder.cpu(); model.text_encoder.cpu()
        torch.cuda.empty_cache()
        clip_hashes = [digest(model.vision_encoder), digest(model.text_encoder)]
        model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        prefixes = ("adapter.", "diffusion_decoder.", "history_projection.")
        for name,p in model.named_parameters():
            p.requires_grad_(name.startswith(prefixes))
        initial = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                   if not n.startswith(("vision_encoder.", "text_encoder."))}
        initial_hash = hashlib.sha256("".join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
        targets_gpu, currents_gpu = targets.cuda(), currents.cuda()
        schedule = [torch.randperm(len(rows), generator=torch.Generator().manual_seed(440000+k))[:args.batch_size]
                    for k in range(1, updates+1)]
        report["common_batch_order_sha256"] = tensor_sha(torch.stack(schedule))
        report["initial_shared_sha256"] = initial_hash
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["added_history_parameters"] = model.history_projection.weight.numel()
        report["gpu"] = torch.cuda.get_device_name()
        report["groups"] = {}
        initial_predictions = None
        for arm in plan["arms"]:
            model.load_state_dict(initial, strict=False)
            assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items())
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            previous_tokens = visual if arm == "repeat_current" else older
            def context(index):
                now = model.adapter(visual[index], language[index], text["attention_mask"][index])
                before = model.adapter(previous_tokens[index], language[index], text["attention_mask"][index])
                return now + model.history_projection(before-now)
            def check(step, preflight=False):
                model.eval()
                with torch.no_grad():
                    idx = torch.arange(min(4,len(rows)) if preflight else len(rows),device="cuda")
                    ctx = torch.cat([context(idx[i:i+2]) for i in range(0,len(idx),2)])
                    predictions, metrics = evaluate(model, ctx, currents_gpu[idx], targets[idx.cpu()].numpy(),
                                                    [1101] if preflight else plan["sample_seeds"], batch=2)
                return predictions, dict(update=step, **metrics)
            pred, start_metrics = check(0, preflight=True)
            if initial_predictions is None:
                initial_predictions = pred.copy()
            else:
                difference = float(np.max(np.abs(pred-initial_predictions)))
                assert difference == 0., difference
                report["initial_native_prediction_max_absolute_difference"] = difference
            history, losses, durations, first_grad = [start_metrics], [], [], None
            torch.cuda.reset_peak_memory_stats()
            for update,index in enumerate(schedule,1):
                model.train()
                index = index.cuda()
                seed(430000+update)
                torch.cuda.synchronize(); tick = time.monotonic()
                ctx = context(index)
                seed(420000+update)
                prediction, truth, _ = model.diffusion_loss(targets_gpu[index], ctx, currents_gpu[index])
                loss = torch.nn.functional.mse_loss(prediction,truth)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                if first_grad is None:
                    first_grad = {name: any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in module.parameters())
                                  for name,module in [("adapter",model.adapter),("decoder",model.diffusion_decoder),
                                                      ("history_projection",model.history_projection),
                                                      ("clip_vision",model.vision_encoder),("clip_text",model.text_encoder)]}
                    assert first_grad["adapter"] and first_grad["decoder"] and first_grad["history_projection"]
                    assert not first_grad["clip_vision"] and not first_grad["clip_text"]
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                durations.append(time.monotonic()-tick); losses.append(float(loss.detach()))
                if update == 1 or update % 100 == 0:
                    print("HISTORY_UPDATE",arm,update,"/",updates,"loss",losses[-1],flush=True)
                if args.stage == "fit" and (update == updates or update == 100):
                    pred, metrics = check(update)
                    history.append(metrics)
                    print("HISTORY_METRICS",arm,json.dumps(metrics),flush=True)
            pred, metrics = check(updates, preflight=True) if args.stage == "preflight" else (pred,history[-1])
            if args.stage == "preflight":
                history.append(metrics)
                replay,_ = check(updates, preflight=True)
                assert np.array_equal(replay,pred)
            changed = {prefix: any(not torch.equal(model.state_dict()[n].cpu(),v)
                                   for n,v in initial.items() if n.startswith(prefix)) for prefix in prefixes}
            assert all(changed.values())
            assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items() if not n.startswith(prefixes))
            report["groups"][arm] = dict(updates=updates, first_gradients=first_grad, parameter_changed=changed,
                frozen_other_modules_unchanged=True, peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                mean_warm_update_seconds=float(np.mean(durations[1:])) if len(durations)>1 else durations[0],
                loss_first=losses[0],loss_last=losses[-1],history=history,
                final_model_sha256={name:digest(module) for name,module in
                                   [("adapter",model.adapter),("decoder",model.diffusion_decoder),
                                    ("history_projection",model.history_projection)]})
            np.savez_compressed(args.output/(arm+"_predictions.npz"), predictions=pred,
                targets=targets[:4].numpy() if args.stage == "preflight" else targets.numpy())
            if args.stage == "fit" and arm == "real_previous":
                # Freeze learned weights and current observations. Replace only
                # previous frames by another training episode of the same task.
                permutation = list(range(len(rows)))
                for task in report["data_checks"]["task_counts"]:
                    members = [i for i,s in enumerate(instructions) if " ".join(s.lower().split()) == task]
                    if len(members) < 2:
                        raise ValueError("Same-task history swap requires >=2 episodes per task")
                    for source, destination in zip(members, members[1:]+members[:1]):
                        permutation[source] = destination
                frozen_before = [digest(model.adapter), digest(model.diffusion_decoder), digest(model.history_projection)]
                previous_tokens = older[permutation]
                swapped_predictions, swapped_metrics = check(updates)
                report["frozen_same_task_history_swap"] = dict(permutation=permutation, metrics=swapped_metrics,
                    current_frames_targets_language_unchanged=True, only_previous_frame_replaced=True,
                    paired_prediction_max_absolute_change=float(np.abs(swapped_predictions-pred).max()))
                assert frozen_before == [digest(model.adapter), digest(model.diffusion_decoder), digest(model.history_projection)]
                np.savez_compressed(args.output/"real_previous_same_task_swap_predictions.npz",
                                    predictions=swapped_predictions,targets=targets.numpy())
                previous_tokens = older
            del optimizer, ctx, prediction, truth, loss
            for p in model.parameters(): p.grad=None
            torch.cuda.empty_cache()
        assert clip_hashes == [digest(model.vision_encoder),digest(model.text_encoder)]
        assert all(sha(ROOT/"code"/n)==h for n,h in sources.items())
        report.update(execution_checks_passed=True, clip_unchanged=True, production_source_unchanged=True,
                      limits=["16 distinct training episodes; no validation or simulation result.",
                              "One training initialization, three sampling seeds only in fit mode.",
                              "Extra history projection is shared in both arms; this is a new experimental architecture.",
                              "Pose loss only; gripper frozen and not assessed.",
                              "Fit results cannot establish a unique root cause; preflight metrics are untrained checks."])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("HISTORY_PAIR_REPORT",args.output/"report.json",flush=True)


if __name__ == "__main__":
    main()
