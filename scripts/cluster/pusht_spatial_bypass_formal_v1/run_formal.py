"""Fixed paired continuation from215999; one endpoint EMA rollout per mode."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

import training_core as core
import bypass_state
from native_data import sha, save_json
from spatial_bypass import MODES, attach
from token_binding import project_update
from preflight_support import prepare, restore_original
from resource_probe import state_hash
from base_verifier import validate_metrics
from pair_metrics import admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    root, output, bundle = args.root.resolve(), args.output.resolve(), Path(__file__).resolve().parent
    if root / "runs" not in output.parents:
        parser.error("Use a new output under root/runs")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"formal_complete": False, "errors": [], "groups": {}, "final_set_accessed": False,
              "automatic_following_submission": False, "clip_training_executed": False,
              "scope": "One bounded paired continuation, reused dev20 overlap reward only; not final confirmation or grasp success"}
    client = None
    try:
        protocol_path, contract_path = bundle / "protocol.json", bundle / "formal_contract.json"
        protocol, contract = json.loads(protocol_path.read_text()), json.loads(contract_path.read_text())
        protocol_sha, contract_sha = sha(protocol_path), sha(contract_path)
        for name, digest in contract["bundle_sources_sha256"].items():
            if sha(bundle / name) != digest:
                raise ValueError("Formal bundle file changed: " + name)
        if protocol_sha != contract["preflight_protocol_sha256"]:
            raise ValueError("Preflight protocol changed")
        binding = json.loads(args.binding.read_text())
        binding_sha = sha(args.binding)
        if (binding.get("format") != "pusht-bypass-preflight-binding-v1" or binding["preflight_job"] != 217778
                or binding["formal_contract_sha256"] != contract_sha or binding["preflight_protocol_sha256"] != protocol_sha
                or not binding["verified"]["preflight_verified"] or binding["preflight_updates_used_as_training_start"]):
            raise ValueError("Wrong admitted preflight binding")
        preflight = root / contract["preflight_directory"]
        for path, digest in ((preflight / "report.json", binding["preflight_report_sha256"]),
                             (preflight / "bypass_statistics.npz", binding["statistics_sha256"])):
            if sha(path) != digest:
                raise ValueError("Admitted preflight artifact changed")
        if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "1.12.1":
            raise ValueError("Use the established official CUDA environment")
        if shutil.disk_usage(output).free < 18 * 1024 ** 3:
            raise ValueError("Need18GiB free for two complete atomic endpoints and transient save; no source deletion")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        shutil.copyfile(args.binding, output / "preflight_binding.json")
        shutil.copyfile(preflight / "bypass_statistics.npz", output / "bypass_statistics.npz")
        base, original, data, cache, plan, source_payload = prepare(root, output, protocol)
        if source_payload["completed_updates"] != contract["source_updates"]:
            raise ValueError("Formal start must be source215999, never preflight-trained weights")
        with np.load(output / "bypass_statistics.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["frame_ids"], cache.ids):
                raise ValueError("Training-only statistic frames changed")
            statistics = {mode: {name: arrays[mode + "_" + name].copy() for name in ("center", "scale")} for mode in MODES}
        baseline = json.loads((bundle / "unchanged_development.json").read_text())
        native = json.loads((bundle / "native_development.json").read_text())
        result.update(formal_contract_sha256=contract_sha, preflight_protocol_sha256=protocol_sha,
                      preflight_binding_sha256=binding_sha, preflight_report_sha256=binding["preflight_report_sha256"],
                      statistics_sha256=binding["statistics_sha256"], source_checkpoint_sha256=protocol["source_checkpoint_sha256"],
                      source_updates=contract["source_updates"], endpoint_updates=contract["endpoint_updates"],
                      new_updates_per_mode=contract["new_updates_per_mode"], data=data.evidence,
                      source_sample_plan_sha256=original["sample_plan_sha256"],
                      unchanged_reference=validate_metrics(baseline, base), native_reference=validate_metrics(native, base))
        first_losses, lr_hashes = [], []
        developments = {}
        for mode in MODES:
            group = output / mode
            group.mkdir()
            policy, averaged, optimizer, scheduler, ema = restore_original(cache, data, source_payload, base, original)
            attached = attach(policy, averaged, optimizer, mode, statistics)
            current = contract["source_updates"]
            fixed = core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask,
                policy.obs_encoder.bypass_center, policy.obs_encoder.bypass_scale, policy.obs_encoder.spatial_weights))
            row = {"mode": mode, "training_completed": False, "development_complete": False,
                   "source_updates": current, "completed_updates": current, "new_updates": 0, "attach": attached,
                   "old_complete_restore_exact": True, "preflight_weights_used": False,
                   "training_seconds": 0., "source_lr": optimizer.param_groups[0]["lr"]}
            result["groups"][mode] = row
            order_digest, lr_digest = hashlib.sha256(), hashlib.sha256()
            train_start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()

            def save_checkpoint():
                payload = bypass_state.capture(policy, averaged, optimizer, scheduler, ema, current,
                    source_payload["protocol_sha256"], original["sample_plan_sha256"], protocol_sha, mode)
                payload.update(formal_contract_sha256=contract_sha, preflight_binding_sha256=binding_sha,
                               source_checkpoint_sha256=protocol["source_checkpoint_sha256"],
                               statistics_sha256=binding["statistics_sha256"])
                path = group / "latest.pt"
                core.atomic_checkpoint(path, payload)
                digest = sha(path)
                sidecar = {"format": payload["format"], "mode": mode, "sha256": digest,
                    "completed_updates": current, "source_updates": contract["source_updates"],
                    "new_updates": current - contract["source_updates"], "formal_contract_sha256": contract_sha,
                    "bypass_protocol_sha256": protocol_sha, "preflight_binding_sha256": binding_sha,
                    "statistics_sha256": binding["statistics_sha256"],
                    "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
                    "sample_plan_sha256": original["sample_plan_sha256"]}
                save_json(group / "latest.json", sidecar)
                row.update(checkpoint=str(path), checkpoint_sha256=digest, completed_updates=current,
                           new_updates=current - contract["source_updates"])
                save_json(group / "report.json", row)
                save_json(output / "report.json", result)
                print("SPATIAL_BYPASS_SAVED", mode, current, path, flush=True)

            with (group / "updates.jsonl").open("x", encoding="utf-8") as log:
                while current < contract["endpoint_updates"]:
                    index = current
                    indices = plan.batch(index)
                    ids = indices.astype("<i8")
                    order_digest.update(np.asarray([len(ids)], dtype="<i8").tobytes())
                    order_digest.update(ids.tobytes())
                    loss, audit = project_update(core, policy, optimizer, scheduler, ema,
                        data.batch(indices), cache.tokens(indices), index, audit=(index == contract["source_updates"]))
                    gradient = policy.obs_encoder.bypass_head.weight.grad
                    if gradient is None or not torch.isfinite(gradient).all() or not (gradient.abs().max() > 0):
                        raise ValueError("Invalid new bypass gradient")
                    current += 1
                    lr_next = scheduler.get_last_lr()[0]
                    lr_digest.update(np.asarray([current], dtype="<i8").tobytes())
                    lr_digest.update(np.asarray([lr_next], dtype="<f8").tobytes())
                    log.write(json.dumps({"completed_updates": current, "loss": loss,
                                          "logical_batch_size": len(ids), "lr_for_next_update": lr_next}) + "\n")
                    if index == contract["source_updates"]:
                        audit["bypass_gradient_finite_nonzero"] = True
                        row["first_loss_and_gradient_audit"] = audit
                        first_losses.append(loss)
                    if (current - contract["source_updates"]) % 250 == 0:
                        log.flush()
                        print("SPATIAL_BYPASS_FORMAL_UPDATE", mode, current - contract["source_updates"],
                              "/", contract["new_updates_per_mode"], "loss", loss,
                              "seconds", round(time.monotonic() - train_start, 1), flush=True)
                    if (current - contract["source_updates"]) % contract["checkpoint_interval"] == 0 or current == contract["endpoint_updates"]:
                        log.flush()
                        save_checkpoint()
            if order_digest.hexdigest() != contract["continuation_order_sha256"]:
                raise ValueError("Continuation batch sequence changed")
            if core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask,
                policy.obs_encoder.bypass_center, policy.obs_encoder.bypass_scale, policy.obs_encoder.spatial_weights)) != fixed:
                raise ValueError("Frozen text/statistic buffers changed")
            if not (torch.equal(policy.obs_encoder.text_tokens, averaged.obs_encoder.text_tokens)
                    and torch.equal(policy.obs_encoder.text_mask, averaged.obs_encoder.text_mask)
                    and torch.equal(policy.obs_encoder.bypass_center, averaged.obs_encoder.bypass_center)
                    and torch.equal(policy.obs_encoder.bypass_scale, averaged.obs_encoder.bypass_scale)):
                raise ValueError("EMA fixed input buffers changed")
            head_steps = int(optimizer.state[policy.obs_encoder.bypass_head.weight]["step"])
            if head_steps != contract["new_updates_per_mode"] or ema.optimization_step != current:
                raise ValueError("Incorrect new Adam or continued EMA counters")
            lr_hashes.append(lr_digest.hexdigest())
            row.update(training_completed=True, completed_updates=current,
                       new_updates=current - contract["source_updates"],
                       continuation_order_sha256=order_digest.hexdigest(), lr_trace_sha256=lr_digest.hexdigest(),
                       new_head_optimizer_step=head_steps, ema_optimization_step=ema.optimization_step,
                       fixed_text_and_statistics_unchanged=True, old_learning_rate_horizon_unchanged=True,
                       peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                       training_seconds=time.monotonic() - train_start)
            save_json(group / "report.json", row)
            del policy, optimizer, scheduler, ema
            gc.collect()
            torch.cuda.empty_cache()
            from live_client import CLIPClient, LiveProjectPolicy
            from evaluate_bypass import evaluate
            client = CLIPClient(root, group)
            if not (np.array_equal(client.text, cache.text.numpy()) and np.array_equal(client.text_mask, cache.mask.numpy())):
                raise ValueError("Live text differs from frozen training text")
            before_evaluation = state_hash(averaged)
            development = evaluate(LiveProjectPolicy(averaged, client), group, base)
            closed = client.close()
            client = None
            if not (closed["all_frozen"] and closed["weights_unchanged"] and state_hash(averaged) == before_evaluation
                    and all(not p.requires_grad and p.grad is None for p in averaged.parameters())):
                raise ValueError("Frozen evaluation model or CLIP state changed")
            row.update(development_complete=True, development=development, evaluation_worker=closed,
                       ema_state_sha256=before_evaluation, ema_state_unchanged_during_evaluation=True)
            developments[mode] = development
            save_json(group / "report.json", row)
            save_json(output / "report.json", result)
            print("SPATIAL_BYPASS_ENDPOINT", mode, development["mean_score"], development["full_reward_fraction"], flush=True)
            del averaged, development
            gc.collect()
            torch.cuda.empty_cache()
        if first_losses[0] != first_losses[1] or lr_hashes[0] != lr_hashes[1]:
            raise ValueError("Paired zero-head initial loss or learning-rate sequence changed")
        outcome = admission(developments["global_control"], developments["spatial_candidate"], baseline, base, contract)
        save_json(output / "admission.json", outcome)
        result.update(formal_complete=True, matched_initial_loss_exact=True, matched_lr_trace_exact=True, admission=outcome)
        print("SPATIAL_BYPASS_DEVELOPMENT_ADMISSION", json.dumps(outcome), flush=True)
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if client is not None:
            client.abort()
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(output / "report.json", result)
        print("SPATIAL_BYPASS_FORMAL_REPORT", output / "report.json", flush=True)
    print("SPATIAL BYPASS PAIR COMPLETE; execution passing is not an improvement verdict", flush=True)


if __name__ == "__main__":
    main()
