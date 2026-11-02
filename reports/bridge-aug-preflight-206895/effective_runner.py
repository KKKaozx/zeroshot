"""Check 128-window fitting with unchanged modules and matched per-window exposure."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import sys
BASE = Path(__file__).resolve().parent.parent/"bridge_growth128_v1"
sys.path.insert(0,str(BASE))

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import torch
from transformers import CLIPTokenizer
from growth128_input_contract import expanded_items as load_verified_items, expanded_schedule
from check_bridge_batch_control import schedule, metrics, tensor_sha, sha
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest, evaluate, score, seed
from trace_bridge_batch_training import PREFIXES as BASE_PREFIXES, observation_guard, trainable_copy as base_copy, gradient_norms as base_gradients
from state_input_contract import load_current_features, add_zero_port, clipping_parameters
PREFIXES=BASE_PREFIXES+("state_projection.",)
def trainable_copy(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith(PREFIXES)}
def gradient_norms(model):
    result=base_gradients(model)
    result["state_projection."]=float(torch.linalg.vector_norm(model.state_projection.weight.grad.detach()))
    assert np.isfinite(result["state_projection."])
    return result


def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_sha(value):
    """Fingerprint tensor values and Adam state, independent of device location."""
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            h.update(str((tuple(v.shape), str(v.dtype))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda x:(type(x).__name__, str(x))):
                h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v: visit(item)
        else: h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "data-root", "data-reference", "clip", "origin-report", "output", "retained-report"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--stage", choices=("preflight", "fit"), default="preflight")
    ap.add_argument("--preflight-reference", type=Path)
    ap.add_argument("--state-arm",choices=("zero","real"),required=True)
    ap.add_argument("--state-protocol",type=Path,required=True)
    args = ap.parse_args()
    assert args.state_arm == "real"
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    code = BASE
    sources = {n:sha(code/n) for n in manifest["source_sha256"]}
    report = dict(stage=args.stage, execution_checks_passed=False, manifest_sha256=sha(args.manifest),
        source_sha256=sources, data_checks=[], input_numeric_comparisons=[], runs=[], checkpoints=[],
        gripper_trained=False, simulation_executed=False, downloads=False, reserved_test_targets_read=False)
    report.update(augmentation_protocol_sha256=sha(AUG_CODE/"protocol.json"),
        augmentation_entry_sha256=sha(AUG_CODE/"run_image_augmentation.py"),
        augmentation_helper_sha256=sha(AUG_CODE/"image_views.py"),
        original_control_retrained=False, candidate="training_only_replicate_pad8_random_shift")
    cfg = copy.deepcopy(manifest["config"]); cfg["model"]["name"] = str(args.clip)
    (args.output/"plan.json").write_text(json.dumps(dict(manifest=manifest, runtime_config=cfg), indent=2)+"\n")
    (args.output/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"effective_runner.py").write_text(EFFECTIVE_SOURCE,encoding="utf-8")
    def persist():
        report["elapsed_seconds"] = time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    try:
        assert sources == manifest["source_sha256"]
        assert torch.cuda.is_available(), "GPU allocation required"
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        assert sha(args.origin_report) == manifest["late_lr"]["origin_report_sha256"]
        origin = json.loads(args.origin_report.read_text())
        assert origin["execution_checks_passed"] and origin["manifest_sha256"] == manifest["late_lr"]["origin_manifest_sha256"]
        rows = manifest["train"]+manifest["development"]
        identity = lambda r:(r["shard"], r["record_index"])
        assert len(rows) == 168 and len(set(map(identity, rows[:128]))) == 64
        assert not set(map(identity, rows[:128])) & set(map(identity, rows[128:]))
        batches = expanded_schedule()
        assert tensor_sha(batches) == manifest["schedule_sha256"]["mixed"]
        items = load_verified_items(rows, args.data_root, manifest, args.data_reference,
                                   report["data_checks"], report["input_numeric_comparisons"])
        report["input_contract_passed"] = True
        if args.stage == "fit":
            assert args.preflight_reference
            pre = json.loads((args.preflight_reference/"report.json").read_text())
            assert pre["execution_checks_passed"] and pre["branch_restore_passed"] and pre["stage"] == "preflight"
            assert pre["manifest_sha256"] == report["manifest_sha256"] and pre["source_sha256"] == sources
            for key in ("augmentation_protocol_sha256","augmentation_entry_sha256","augmentation_helper_sha256"):
                assert pre[key] == report[key]
            assert pre["evaluation_uses_original_tokens"] and pre["center_view_input_exact"]
            assert pre["augmentation_calls"] == 16 and pre["global_rng_preserved_by_augmentation"]
            assert pre["peak_reserved_gib"] < 7.5
            assert pre["estimated_formal_seconds"] < 9000
            assert pre["data_checks"] == report["data_checks"]
            assert report["data_checks"][:72] == origin["data_checks"]
            assert sha(args.retained_report) == manifest["growth"]["retained64_report_sha256"]
            retained = json.loads(args.retained_report.read_text())
            assert retained["execution_checks_passed"] and report["data_checks"][:104] == retained["data_checks"]
        targets = torch.stack([v[3] for v in items]).cuda()
        current = torch.stack([v[2] for v in items]).cuda()
        init = 43; seed(init)
        model = RobotAdapterModel(cfg).cuda().eval()
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        texts = tokenizer([v[0] for v in items], padding=True, truncation=True, max_length=77, return_tensors="pt")
        identity=hashlib.sha256(json.dumps(dict(manifest=report["manifest_sha256"],checks=report["data_checks"],sources=sources),sort_keys=True).encode()).hexdigest()
        assert TOKEN_CACHE.is_file(), "Reuse the existing original-view cache; do not encode a new full cache"
        visual,language,reused=encode_or_reuse(model,items,texts,TOKEN_CACHE,identity,digest)
        assert reused
        report.update(full_token_cache_reused=True, original_visual_token_sha256=tensor_sha(visual),
            language_inputs_changed=False, observed_state_inputs_changed=False, action_targets_changed=False)
        mask = texts["attention_mask"].cuda()
        model.vision_encoder.cpu(); model.text_encoder.cpu(); torch.cuda.empty_cache()
        clip_hash = [digest(model.vision_encoder), digest(model.text_encoder)]
        model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        for n,p in model.named_parameters(): p.requires_grad_(n.startswith(PREFIXES))
        initial = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                   if not n.startswith(("vision_encoder.", "text_encoder."))}
        initial_hash = hashlib.sha256("".join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
        reference = next(r for r in origin["runs"] if r["initialization_seed"] == init)
        if args.stage == "fit": assert initial_hash == reference["initial_shared_sha256"]
        report["base_initial_shared_sha256"] = initial_hash
        features, state_checks = load_current_features(args.state_protocol, rows, args.data_root, report["data_checks"])
        state_features = torch.from_numpy(features).cuda()
        if args.state_arm == "zero": state_features = torch.zeros_like(state_features)
        add_zero_port(model)
        report.update(state_arm=args.state_arm, state_checks=state_checks, state_features_sha256=tensor_sha(state_features),
            state_protocol_sha256=sha(args.state_protocol),
            augmented_initial_sha256=state_sha(trainable_copy(model)), state_port_initial_zero=True)
        initial["state_projection.weight"] = model.state_projection.weight.detach().cpu().clone()
        trainables = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainables, lr=1e-4)
        warm_end, end = (4, 8) if args.stage == "preflight" else (4096, 8192)
        report.update(initialization_seed=init, initial_shared_sha256=initial_hash, branch_update=warm_end,
            final_update=end, gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            sampling_steps=100, sampling_seeds=manifest["sampling_seeds"])
        views=OnlineViews(model,items,visual,args.output,report)
        optimization_seconds=[]
        torch.cuda.reset_peak_memory_stats()
        def context(indices, augmentation_seed=None, arm=None):
            tokens=visual[indices] if augmentation_seed is None else views(indices,augmentation_seed,arm)
            now = model.adapter(tokens, language[indices], mask[indices])
            repeated = model.adapter(tokens, language[indices], mask[indices])
            return now+model.history_projection(repeated-now)+model.state_projection(state_features[indices])
        def check(arm, update, split="train"):
            indices = list(range(128)) if split == "train" else list(range(128, 168))
            if args.stage == "preflight" and split == "train": indices = [0,32,64,96]
            with observation_guard(model):
                ctx = torch.cat([context(torch.tensor(indices[i:i+2], device="cuda")) for i in range(0, len(indices), 2)])
                truth = targets[indices].cpu().numpy()
                pred, gate = evaluate(model, ctx, current[indices], truth, manifest["sampling_seeds"], batch=2)
                result = metrics(pred, truth, [rows[i] for i in indices])
                if split == "train":
                    result["fit_gate"] = gate
                    for label,is_old in (("retained_train",True),("added_train",False)):
                        part = [j for j,i in enumerate(indices) if (i<64)==is_old]
                        result[label] = metrics(pred[:,part],truth[part],[rows[indices[j]] for j in part])
                        result[label]["fit_gate"] = score(pred[:,part],truth[part])
            np.savez_compressed(args.output/f"{arm}_u{update:04d}_{split}.npz", predictions=pred,
                                targets=truth, contexts=ctx.cpu().numpy(), row_indices=np.asarray(indices))
            print("LATE_LR_MONITOR", arm, update, split, "position_mm", result["path_position_cm"]*10,
                  "rotation", result["path_rotation_deg"], flush=True)
            return result, pred
        def advance(arm, first, last, points):
            history, losses = [], []
            with (args.output/f"{arm}_updates.jsonl").open("w") as log:
                for update in range(first, last+1):
                    model.train(); indices = batches[update-1].cuda()
                    update_started=time.monotonic()
                    seed(430000+update); ctx = context(indices,510000+update,arm)
                    seed(420000+update)
                    prediction, truth, _ = model.diffusion_loss(targets[indices], ctx, current[indices])
                    loss = torch.nn.functional.mse_loss(prediction, truth)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    grads = gradient_norms(model)
                    assert all(p.grad is None for p in model.vision_encoder.parameters())
                    assert all(p.grad is None for p in model.text_encoder.parameters())
                    norm = torch.nn.utils.clip_grad_norm_(clipping_parameters(model, trainables, args.state_arm), 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    optimization_seconds.append(time.monotonic()-update_started)
                    value = float(loss.detach()); losses.append(value)
                    log.write(json.dumps(dict(update=update, loss=value, learning_rate=optimizer.param_groups[0]["lr"],
                        unclipped_module_gradient_norms=grads, global_unclipped_gradient_norm=float(norm)))+"\n")
                    if update == first or update % 256 == 0:
                        log.flush(); print("LATE_LR_UPDATE", arm, update, "/", last, "loss", value, flush=True)
                    if update in points:
                        m,_ = check(arm, update); history.append(dict(update=update, train=m)); persist()
            return history, losses
        def save_checkpoint(name, include_optimizer):
            if args.stage != "fit" or include_optimizer: return
            payload = dict(stage="late_lr_trainable_only", initialization_seed=init,
                manifest_sha256=report["manifest_sha256"], source_sha256=sources, config=cfg,
                trainable_state_dict=warm_weights if include_optimizer else trainable_copy(model),
                optimizer_state_dict=warm_adam if include_optimizer else None,
                frozen_nonclip_initial_sha256=initial_hash, update=warm_end if include_optimizer else end,
                reconstruction="Recreate frozen model; add zero history_projection and bias-free9-to1024 state_projection; load trainable state; use checkpoint state protocol to construct current observed-pose context.")
            payload["image_augmentation_protocol"]=AUG_PROTOCOL
            payload["state_arm"]=args.state_arm
            payload["state_protocol"]=json.loads(args.state_protocol.read_text(encoding="utf-8"))
            file = args.output/name; torch.save(payload, file)
            restored = torch.load(file, map_location="cpu", weights_only=False)
            assert state_sha(restored["trainable_state_dict"]) == state_sha(payload["trainable_state_dict"])
            assert state_sha(restored["optimizer_state_dict"]) == state_sha(payload["optimizer_state_dict"])
            report["checkpoints"].append(dict(path=file.name, sha256=sha(file), bytes=file.stat().st_size,
                                              includes_optimizer=include_optimizer, roundtrip_exact=True))
            assert sum(p.stat().st_size for p in args.output.glob("*.pt")) < manifest["late_lr"]["checkpoint_budget_bytes"]
        m,_ = check("common", 0)
        warm_history, warm_losses = advance("common", 1, warm_end, [warm_end] if args.stage == "preflight" else [2048, 4096])
        common = warm_history[-1]["train"]
        report["common_history"] = [dict(update=0, train=m)]+warm_history
        report["warmup_first_loss"] = warm_losses[0]
        if args.stage == "fit":
            report["new_data_first_loss"] = warm_losses[0]
            report["warmup_fit_passed"] = common["fit_gate"]["pose_fit_passed"]
        warm_weights = trainable_copy(model); warm_adam = cpu_copy(optimizer.state_dict())
        warm_weights_hash, warm_adam_hash = state_sha(warm_weights), state_sha(warm_adam)
        report["common_base_state_sha256"] = state_sha(base_copy(model))
        report["common_state_sha256"] = warm_weights_hash; report["common_adam_sha256"] = warm_adam_hash
        save_checkpoint("common_u1024_with_adam.pt", True)
        # The constant arm continues in place; replaying it checks the restore path.
        points = [end] if args.stage == "preflight" else [6144, 7168, 7680, 8192]
        baseline_hash = None; baseline_losses = None; baseline_pred = None
        arms = [("constant", 1e-4), ("restore_same_lr", 1e-4), ("reduced", 1e-5)] if args.stage == "preflight" else [("reduced", 1e-5)]
        for arm, lr in arms:
            if arm != "constant":
                model.load_state_dict(warm_weights, strict=False)
                optimizer.load_state_dict(cpu_copy(warm_adam))
            assert state_sha(trainable_copy(model)) == warm_weights_hash
            assert state_sha(optimizer.state_dict()) == warm_adam_hash
            for group in optimizer.param_groups: group["lr"] = lr
            row = dict(arm=arm, learning_rate=lr, common_weights_sha256=warm_weights_hash,
                       common_adam_sha256=warm_adam_hash, history=[])
            report["runs"].append(row)
            row["history"], losses = advance(arm, warm_end+1, end, points)
            final = row["history"][-1]["train"]
            with np.load(args.output/f"{arm}_u{end:04d}_train.npz") as z: pred = z["predictions"]
            row.update(final_train=final, mean_loss=float(np.mean(losses)), loss_first=losses[0], loss_last=losses[-1],
                actual_draw_slots_per_window=torch.bincount(batches[warm_end:end].flatten(), minlength=128).tolist())
            if args.stage == "fit": assert row["actual_draw_slots_per_window"] == [256]*128
            else: assert sum(row["actual_draw_slots_per_window"]) == 32
            row["final_base_state_sha256"] = state_sha(base_copy(model))
            row["state_port_weight_norm"] = float(torch.linalg.vector_norm(model.state_projection.weight.detach()))
            row["state_port_context_max_abs"] = float(model.state_projection(state_features).detach().abs().max())
            row["state_port_sha256"] = tensor_sha(model.state_projection.weight)
            if args.stage == "preflight":
                current_hash = state_sha(trainable_copy(model))
                if arm == "constant": baseline_hash, baseline_losses, baseline_pred = current_hash, losses, pred.copy()
                elif arm == "restore_same_lr":
                    delta = float(np.abs(pred-baseline_pred).max())
                    loss_delta = float(np.max(np.abs(np.asarray(losses)-baseline_losses)))
                    assert current_hash == baseline_hash and delta == loss_delta == 0
                    report["branch_restore_passed"] = True
                    report["branch_restore"] = dict(parameters_exact=True, max_loss_difference=loss_delta,
                        max_prediction_difference=delta, includes_adam_moments_and_step=True)
                else:
                    assert losses[0] == baseline_losses[0]
                    assert current_hash != baseline_hash, "Different learning rate must change updates"
                    report["lr_intervention_changes_parameters"] = True
            else:
                dev,_ = check(arm, end, "development"); row["final_development"] = dev
                save_checkpoint(f"{arm}_u{end}_trainable.pt", False)
                if arm == "constant":
                    row["origin_final_metric_differences"] = {k:final[k]-reference["final"]["train"][k] for k in
                        ("path_position_cm", "path_rotation_deg", "endpoint_position_cm", "endpoint_rotation_deg")}
                    report["control_reproduces_origin"] = all(abs(v) <= (.5 if "rotation" in k else .1)
                        for k,v in row["origin_final_metric_differences"].items()) and not final["fit_gate"]["pose_fit_passed"]
            assert all(torch.equal(model.state_dict()[n].cpu(), v) for n,v in initial.items() if not n.startswith(PREFIXES))
            assert state_sha(warm_weights) == warm_weights_hash and state_sha(warm_adam) == warm_adam_hash
            print("LATE_LR_FINAL", arm, "train_mm", final["path_position_cm"]*10,
                  "gate", final["fit_gate"]["pose_fit_passed"], flush=True)
            persist()
        assert clip_hash == [digest(model.vision_encoder), digest(model.text_encoder)]
        assert sources == {n:sha(code/n) for n in sources}
        report.update(warm_update_seconds=float(np.mean(optimization_seconds[1:])),
            estimated_formal_seconds=float(np.mean(optimization_seconds[1:]))*8192+1200,
            original_evaluation_tokens_unchanged=(tensor_sha(visual)==report["original_visual_token_sha256"]))
        assert report["original_evaluation_tokens_unchanged"]
        assert report["augmentation_calls"] == (16 if args.stage=="preflight" else 8192)
        report.update(execution_checks_passed=True, clip_unchanged=True, other_frozen_unchanged=True,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      scientific_verdict="Training image-shift intervention; judge original40 held-out path position AND rotation, not fit or execution success.")
    except Exception as error:
        report["error"] = repr(error); raise
    finally:
        persist(); print("LATE_LR_REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__": main()
