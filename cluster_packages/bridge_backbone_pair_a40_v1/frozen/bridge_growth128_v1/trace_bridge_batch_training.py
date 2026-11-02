"""Observe the frozen 32-window mixed-batch recipe, without changing its updates."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import torch
from transformers import CLIPTokenizer
from bridge_input_contract import load_verified_items
from check_bridge_batch_control import schedule, metrics, tensor_sha, sha
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest, evaluate, score, seed

PREFIXES = ("adapter.", "diffusion_decoder.", "history_projection.")


@contextmanager
def observation_guard(model):
    """Restore NumPy/Torch RNG and every module mode; reject tensor mutations."""
    numpy_state = np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    modes = [(module, module.training) for module in model.modules()]
    versions = {name: value._version for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with torch.no_grad():
                model.eval()
                yield
    finally:
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode
        assert torch.equal(cpu_rng, torch.get_rng_state()), "Observer changed CPU RNG"
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state()), "Observer changed CUDA RNG"
        assert all(versions[n] == p._version for n, p in model.named_parameters()), "Observer changed parameters"
        assert all(torch.equal(buffers[n], b) for n, b in model.named_buffers()), "Observer changed buffers"


def trainable_copy(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.startswith(PREFIXES)}


def gradient_norms(model):
    """Read unclipped gradients; no backward, RNG draws, or gradient mutation."""
    values = []
    for prefix in PREFIXES:
        norms = [torch.linalg.vector_norm(p.grad.detach()) for n, p in model.named_parameters()
                 if n.startswith(prefix) and p.grad is not None]
        assert norms, "Missing gradients: " + prefix
        values.append(torch.linalg.vector_norm(torch.stack(norms)))
    result = torch.stack(values).cpu().tolist()
    assert all(np.isfinite(result))
    return dict(zip(PREFIXES, result))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "data-reference", "clip", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--stage", choices=("preflight", "fit"), default="preflight")
    ap.add_argument("--preflight-reference", type=Path)
    ap.add_argument("--origin-report", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    code = Path(__file__).resolve().parent
    sources = {name: sha(code/name) for name in manifest["source_sha256"]}
    report = dict(stage=args.stage, execution_checks_passed=False, manifest_sha256=sha(args.manifest),
        source_sha256=sources, downloads=False, gripper_trained=False, simulation_executed=False,
        reserved_test_targets_read=False, observer_changes_training_recipe=False, runs=[], data_checks=[],
        input_numeric_comparisons=[], checkpoints_saved=False)
    (args.output/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cfg = copy.deepcopy(manifest["config"]); cfg["model"]["name"] = str(args.clip)
    (args.output/"plan.json").write_text(json.dumps(dict(manifest=manifest, runtime_config=cfg), indent=2)+"\n")
    def persist():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        assert sources == manifest["source_sha256"]
        assert torch.cuda.is_available(), "GPU allocation required"
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert manifest["arms"] == ["mixed"] and manifest["updates_per_arm"] == 2048
        batches = schedule("mixed")
        assert tensor_sha(batches) == manifest["schedule_sha256"]["mixed"]
        rows = manifest["train"] + manifest["development"]
        identity = lambda row: (row["shard"], row["record_index"])
        assert len(manifest["train"]) == 32 and len(manifest["development"]) == 40
        assert len(set(map(identity, rows[:32]))) == 16 and len(set(map(identity, rows[32:]))) == 10
        assert not set(map(identity, rows[:32])) & set(map(identity, rows[32:]))
        items = load_verified_items(rows, args.data_root, manifest, args.data_reference,
            report["data_checks"], report["input_numeric_comparisons"])
        report["input_contract_passed"] = True
        targets = torch.stack([v[3] for v in items]).cuda()
        current = torch.stack([v[2] for v in items]).cuda()
        previous = None
        if args.stage == "fit":
            assert args.preflight_reference and args.origin_report
            previous = json.loads((args.preflight_reference/"report.json").read_text())
            assert previous["stage"] == "preflight" and previous["execution_checks_passed"]
            assert previous["observer_replay_passed"]
            assert previous["manifest_sha256"] == report["manifest_sha256"] and previous["source_sha256"] == sources
            assert previous["data_checks"] == report["data_checks"]
        origin = json.loads(args.origin_report.read_text()) if args.origin_report else None
        if origin:
            assert sha(args.origin_report) == manifest["trace"]["origin_report_sha256"]
            assert origin["execution_checks_passed"] and origin["manifest_sha256"] == manifest["trace"]["origin_manifest_sha256"]
            if args.stage == "fit":
                assert origin["data_checks"] == report["data_checks"], "Same-cluster processed input identity changed"
        report.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, sampling_steps=100,
                      sampling_seeds=manifest["sampling_seeds"], observer_rng_and_mode_guard=True)
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        updates = 8 if args.stage == "preflight" else 2048
        seeds = [42] if args.stage == "preflight" else manifest["initialization_seeds"]
        points = [0, 4, 8] if args.stage == "preflight" else manifest["trace"]["train_monitor_updates"]
        for init in seeds:
            seed(init)
            model = RobotAdapterModel(cfg).cuda().eval()
            texts = tokenizer([v[0] for v in items], padding=True, truncation=True, max_length=77, return_tensors="pt")
            with torch.no_grad():
                language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                    for i in range(0, len(items), 16)]).detach()
                visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                    for i in range(0, len(items), 2)]).detach()
            mask = texts["attention_mask"].cuda()
            model.vision_encoder.cpu(); model.text_encoder.cpu(); torch.cuda.empty_cache()
            clip_hash = [digest(model.vision_encoder), digest(model.text_encoder)]
            model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
            torch.nn.init.zeros_(model.history_projection.weight)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(PREFIXES))
            initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
                       if not n.startswith(("vision_encoder.", "text_encoder."))}
            initial_hash = hashlib.sha256("".join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
            if origin and args.stage == "fit":
                old = next(r for r in origin["runs"] if r["initialization_seed"] == init and r["arm"] == "mixed")
                assert initial_hash == old["initial_shared_sha256"], "Initial weights changed relative to 198871"
            parameter_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
            report["checkpoint_payload_bytes_upper_estimate"] = parameter_bytes*5 + 32*1024**2
            assert report["checkpoint_payload_bytes_upper_estimate"] < manifest["trace"]["checkpoint_budget_bytes"]
            trainables = [p for p in model.parameters() if p.requires_grad]
            def context(indices):
                now = model.adapter(visual[indices], language[indices], mask[indices])
                repeated = model.adapter(visual[indices], language[indices], mask[indices])
                return now + model.history_projection(repeated-now)
            def check(update, split, save):
                indices = list(range(32)) if split == "train" else list(range(32, 72))
                with observation_guard(model):
                    contexts = torch.cat([context(torch.tensor(indices[i:i+2], device="cuda"))
                        for i in range(0, len(indices), 2)])
                    pred, gate = evaluate(model, contexts, current[indices], targets[indices].cpu().numpy(),
                        manifest["sampling_seeds"], batch=2)
                    ctx = contexts.cpu().numpy()
                    truth = targets[indices].cpu().numpy()
                    result = metrics(pred, truth, [rows[i] for i in indices])
                    if split == "train": result["fit_gate"] = gate
                if save:
                    np.savez_compressed(args.output/f"seed{init}_u{update:04d}_{split}.npz",
                        predictions=pred, targets=truth, contexts=ctx, row_indices=np.asarray(indices))
                return result, pred
            control = None
            phases = [False, True] if args.stage == "preflight" else [True]
            row = dict(initialization_seed=init, initial_shared_sha256=initial_hash, history=[], checkpoints=[])
            report["runs"].append(row)
            for observed in phases:
                model.load_state_dict(initial, strict=False)
                assert all(torch.equal(model.state_dict()[n].cpu(), v) for n, v in initial.items())
                optimizer = torch.optim.Adam(trainables, lr=1e-4)
                torch.cuda.reset_peak_memory_stats()
                loss_values, durations, gradient_rows = [], [], []
                if observed:
                    m, _ = check(0, "train", True)
                    row["history"].append(dict(update=0, train=m))
                    persist()
                log_path = args.output/f"seed{init}_{'observed' if observed else 'control'}_updates.jsonl"
                with log_path.open("w") as log:
                    for update in range(1, updates+1):
                        model.train(); indices = batches[update-1].cuda()
                        seed(430000+update)
                        torch.cuda.synchronize(); tick = time.monotonic()
                        ctx = context(indices)
                        seed(420000+update)
                        prediction, truth, _ = model.diffusion_loss(targets[indices], ctx, current[indices])
                        loss = torch.nn.functional.mse_loss(prediction, truth)
                        assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        grads = gradient_norms(model) if observed else None
                        if update == 1:
                            assert all(any(p.grad is not None and bool(torch.any(p.grad != 0))
                                for n,p in model.named_parameters() if n.startswith(prefix)) for prefix in PREFIXES)
                            assert all(p.grad is None for p in model.vision_encoder.parameters())
                            assert all(p.grad is None for p in model.text_encoder.parameters())
                        global_norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
                        optimizer.step()
                        torch.cuda.synchronize(); durations.append(time.monotonic()-tick)
                        loss_values.append(float(loss.detach()))
                        entry = dict(update=update, loss=loss_values[-1])
                        if observed:
                            entry.update(unclipped_module_gradient_norms=grads, global_unclipped_gradient_norm=float(global_norm),
                                         gradient_clip_scale=min(1., 1./(float(global_norm)+1e-6)))
                            gradient_rows.append(entry)
                        log.write(json.dumps(entry)+"\n")
                        if update == 1 or update % 256 == 0:
                            log.flush(); print("TRACE_UPDATE", init, "observed" if observed else "control", update,
                                "/", updates, "loss", loss_values[-1], flush=True)
                        if observed and update in points:
                            m, _ = check(update, "train", True)
                            row["history"].append(dict(update=update, train=m))
                            persist()
                            print("TRACE_MONITOR", init, update, "train_mm", m["path_position_cm"]*10,
                                "rotation", m["path_rotation_deg"], "windows", m["fit_gate"]["windows_all_three_seeds_pass"], flush=True)
                train_result, pred = check(updates, "train", observed)
                dev_result, dev_pred = check(updates, "development", observed)
                if not observed:
                    control = dict(state=trainable_copy(model), losses=loss_values, train=pred, development=dev_pred)
                else:
                    row.update(updates=updates, actual_draw_slots_per_window=torch.bincount(batches[:updates].flatten(), minlength=32).tolist(),
                        loss_first=loss_values[0], loss_last=loss_values[-1], final=dict(train=train_result, development=dev_result),
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, mean_warm_update_seconds=float(np.mean(durations[1:])),
                        gradient_summary={prefix:dict(min=float(np.min([g["unclipped_module_gradient_norms"][prefix] for g in gradient_rows])),
                            max=float(np.max([g["unclipped_module_gradient_norms"][prefix] for g in gradient_rows])),
                            zero_updates=sum(g["unclipped_module_gradient_norms"][prefix] == 0 for g in gradient_rows)) for prefix in PREFIXES},
                        changed={prefix:any(not torch.equal(model.state_dict()[n].cpu(), v) for n,v in initial.items() if n.startswith(prefix)) for prefix in PREFIXES})
                    assert all(row["changed"].values())
                    assert all(torch.equal(model.state_dict()[n].cpu(), v) for n,v in initial.items() if not n.startswith(PREFIXES))
                    if control:
                        max_parameter_delta = max(float((p.detach().cpu()-control["state"][n]).abs().max())
                            for n,p in model.named_parameters() if n.startswith(PREFIXES))
                        max_prediction_delta = max(float(np.abs(pred-control["train"]).max()), float(np.abs(dev_pred-control["development"]).max()))
                        max_loss_delta = float(np.max(np.abs(np.asarray(loss_values)-control["losses"])))
                        assert max_parameter_delta <= 1e-5 and max_prediction_delta <= 1e-3 and max_loss_delta <= 1e-5
                        report["observer_replay_passed"] = True
                        report["observer_replay"] = dict(max_parameter_difference=max_parameter_delta,
                            max_prediction_difference=max_prediction_delta, max_loss_difference=max_loss_delta,
                            tolerances=dict(parameters=1e-5, predictions=1e-3, losses=1e-5),
                            exact=bool(max_parameter_delta == max_prediction_delta == max_loss_delta == 0))
                    if previous and init == 42:
                        old = previous["runs"][0]
                        assert old["initial_shared_sha256"] == initial_hash and abs(old["loss_first"]-loss_values[0]) < 1e-6
                        row["same_package_preflight_initialization_first_loss_match"] = True
                    if origin and args.stage == "fit":
                        old = next(r for r in origin["runs"] if r["initialization_seed"] == init and r["arm"] == "mixed")
                        assert abs(old["loss_first"]-loss_values[0]) < 1e-6
                        row["origin_first_loss_match"] = True
                        row["origin_final_metric_differences"] = {key:train_result[key]-old["final"]["train"][key]
                            for key in ("path_position_cm", "path_rotation_deg", "endpoint_position_cm", "endpoint_rotation_deg")}
                        row["origin_final_close"] = all(abs(delta) <= (.5 if "rotation" in key else .1)
                            for key, delta in row["origin_final_metric_differences"].items())
                    if args.stage == "fit":
                        # Only trainable weights; include Adam moments for seed43 continuation.
                        checkpoint = dict(stage="pose_trace_trainable_only", initialization_seed=init, update=updates,
                            manifest_sha256=report["manifest_sha256"], source_sha256=sources, config=cfg,
                            trainable_state_dict=trainable_copy(model), frozen_nonclip_initial_sha256=initial_hash,
                            optimizer_state_dict=optimizer.state_dict() if init == 43 else None,
                            reconstruction="Recreate same initialized frozen model, add zero-initialized history_projection; load these named trainable tensors.",
                            next_update_seed_recipe=dict(adapter=430000, diffusion=420000))
                        file = args.output/f"seed{init}_u2048_trainable.pt"
                        torch.save(checkpoint, file)
                        restored = torch.load(file, map_location="cpu", weights_only=False)
                        assert restored["manifest_sha256"] == report["manifest_sha256"]
                        assert restored["trainable_state_dict"].keys() == checkpoint["trainable_state_dict"].keys()
                        assert all(torch.equal(v, restored["trainable_state_dict"][n]) for n,v in checkpoint["trainable_state_dict"].items())
                        assert (restored["optimizer_state_dict"] is not None) == (init == 43)
                        row["checkpoints"].append(dict(path=file.name, bytes=file.stat().st_size, sha256=sha(file), includes_optimizer=init == 43))
                        row["checkpoints"][-1]["weight_roundtrip_exact"] = True
                        del checkpoint, restored
                        report["checkpoints_saved"] = True
                        assert sum(f.stat().st_size for f in args.output.glob("*.pt")) < manifest["trace"]["checkpoint_budget_bytes"]
                    persist()
                    print("TRACE_FINAL", init, "train_mm", train_result["path_position_cm"]*10,
                        "gate", train_result["fit_gate"]["pose_fit_passed"], "dev_cm", dev_result["path_position_cm"], flush=True)
                del optimizer, ctx, prediction, truth, loss
                for p in model.parameters(): p.grad = None
                torch.cuda.empty_cache()
            assert clip_hash == [digest(model.vision_encoder), digest(model.text_encoder)]
            row["frozen_clip_and_other_unchanged"] = True
            del model, initial, trainables, language, visual, mask, control
            torch.cuda.empty_cache()
        assert sources == {name:sha(code/name) for name in sources}
        report.update(execution_checks_passed=True, clip_unchanged=True, production_source_unchanged=True,
            scientific_improvement_verdict="Observe convergence and late deterioration; execution passing is not fitting or generalization.")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        persist()
        print("TRAINING_TRACE_REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
