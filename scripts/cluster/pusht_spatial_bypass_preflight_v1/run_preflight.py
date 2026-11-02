"""New bypass integration/resources only; no task score or formal training entry."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

import training_core as core
import bypass_state
from native_data import sha, save_json
from token_binding import project_update
from spatial_bypass import attach, MODES
from preflight_support import prepare, fit_statistics, restore_original, encoder_context, sample_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Use new root/runs output")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"preflight_passed": False, "errors": [], "formal_training_executed": False,
              "policy_rollouts": 0, "development_scored": False, "sealed_final_accessed": False,
              "clip_model_loaded_or_updated": False, "automatic_submission": False, "groups": {}}
    try:
        bundle = Path(__file__).resolve().parent
        protocol_path = bundle / "protocol.json"
        protocol_sha = sha(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        result["protocol_sha256"] = protocol_sha
        if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "1.12.1":
            raise ValueError("Use established official CUDA environment")
        if shutil.disk_usage(out).free < 12 * 1024 ** 3:
            raise ValueError("Need12GiB free for two complete replay checkpoints; no source files will be removed")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        base, original, data, cache, plan, payload = prepare(root, out, protocol)
        statistics, stat_evidence = fit_statistics(cache, out)
        result["statistics"] = stat_evidence
        result["source_checkpoint_sha256"] = protocol["source_checkpoint_sha256"]
        result["training_data"] = data.evidence
        result["cache_report_sha256"] = original["cache_report_sha256"]
        start = int(payload["completed_updates"])
        batch_indices = plan.batch(start)[:2]
        smoke_batch, smoke_tokens = data.batch(batch_indices), cache.tokens(batch_indices)
        first_losses = []
        for mode in MODES:
            group = out / mode
            group.mkdir()
            policy, averaged, optimizer, scheduler, ema = restore_original(cache, data, payload, base, original)
            before_raw = encoder_context(policy, smoke_batch, smoke_tokens)
            before_ema = encoder_context(averaged, smoke_batch, smoke_tokens)
            before_actions = sample_actions(averaged, smoke_batch, smoke_tokens)
            before_rng = core.tree_digest(core.rng_state())
            attached = attach(policy, averaged, optimizer, mode, statistics)
            if core.tree_digest(core.rng_state()) != before_rng:
                raise ValueError("Attaching zero bypass consumed original RNG")
            after_raw = encoder_context(policy, smoke_batch, smoke_tokens)
            after_ema = encoder_context(averaged, smoke_batch, smoke_tokens)
            after_actions = sample_actions(averaged, smoke_batch, smoke_tokens)
            if not (torch.equal(before_raw, after_raw) and torch.equal(before_ema, after_ema)
                    and core.tree_digest(before_actions) == core.tree_digest(after_actions)):
                raise ValueError("Zero-initialized bypass changed source endpoint predictions")
            fixed_text_before = core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask))
            statistic_before = core.tree_digest((policy.obs_encoder.bypass_center, policy.obs_encoder.bypass_scale,
                                                policy.obs_encoder.spatial_weights))
            row = {"mode": mode, "original_complete_restore_exact": True, "attach": attached,
                   "zero_raw_context_exact": True, "zero_ema_context_exact": True, "zero_ddpm100_actions_exact": True,
                   "rng_unchanged_at_attach": True, "updates": [], "warm_update_seconds": [],
                   "source_updates": start, "preflight_unique_updates": protocol["preflight_updates_per_mode"],
                   "replay_recomputed_steps": 1}
            result["groups"][mode] = row
            current = start
            torch.cuda.reset_peak_memory_stats()

            def snapshot():
                return bypass_state.capture(policy, averaged, optimizer, scheduler, ema, current,
                    payload["protocol_sha256"], original["sample_plan_sha256"], protocol_sha, mode)

            def one_update(index, audit=False):
                indices = plan.batch(index)
                torch.cuda.synchronize()
                tick = time.monotonic()
                loss, check = project_update(core, policy, optimizer, scheduler, ema,
                    data.batch(indices), cache.tokens(indices), index, audit=audit)
                torch.cuda.synchronize()
                duration = time.monotonic() - tick
                gradient = policy.obs_encoder.bypass_head.weight.grad
                if gradient is None or not torch.isfinite(gradient).all() or not (gradient.abs().max() > 0):
                    raise ValueError("Missing/invalid/nonlearning new bypass gradient")
                if audit:
                    check["bypass_gradient_finite_nonzero"] = True
                    check["clip_input_detached"] = True
                return loss, check, duration

            for offset in range(protocol["preflight_updates_per_mode"]):
                if offset == 2:
                    path = group / "replay.pt"
                    core.atomic_checkpoint(path, snapshot())
                    loss_a, _, _ = one_update(current)
                    current += 1
                    expected = core.tree_digest(snapshot())
                    restored = torch.load(path, map_location="cpu")
                    current = bypass_state.restore(restored, policy, averaged, optimizer, scheduler, ema,
                        payload["protocol_sha256"], original["sample_plan_sha256"], protocol_sha, mode)
                    del restored
                    gc.collect()
                    loss, check, duration = one_update(current)
                    current += 1
                    actual = core.tree_digest(snapshot())
                    if loss != loss_a or actual != expected:
                        raise ValueError("New bypass complete disk resume differs")
                    row["disk_resume"] = {"exact": True, "loss_before": loss_a, "loss_after": loss,
                        "state_before_sha256": expected, "state_after_sha256": actual,
                        "checkpoint": str(path), "checkpoint_sha256": sha(path),
                        "scope": "Raw/EMA, old and new Adam moments, original LR, EMA counter, all RNG, fixed spatial buffers/mode"}
                else:
                    loss, check, duration = one_update(current, audit=(offset == 0))
                    current += 1
                if offset == 0:
                    row["loss_and_gradient_audit"] = check
                    first_losses.append(loss)
                if offset >= 3:
                    row["warm_update_seconds"].append(duration)
                row["updates"].append({"completed_updates": current, "loss": loss, "seconds": duration,
                                        "lr_for_next_update": scheduler.get_last_lr()[0]})
                print("SPATIAL_BYPASS_PREFLIGHT_UPDATE", mode, offset + 1, "/", protocol["preflight_updates_per_mode"], "loss", loss, flush=True)
            if core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask)) != fixed_text_before:
                raise ValueError("Frozen text changed")
            if core.tree_digest((policy.obs_encoder.bypass_center, policy.obs_encoder.bypass_scale,
                                  policy.obs_encoder.spatial_weights)) != statistic_before:
                raise ValueError("Fixed train-only spatial statistics changed")
            if any(p.requires_grad for p in averaged.parameters()):
                raise ValueError("EMA parameters must remain frozen")
            sampled = sample_actions(averaged, smoke_batch, smoke_tokens)
            row.update(completed_updates=current, fixed_text_unchanged=True, fixed_spatial_buffers_unchanged=True,
                ema_frozen=True, new_raw_head_max_abs=float(policy.obs_encoder.bypass_head.weight.abs().max()),
                new_ema_head_max_abs=float(averaged.obs_encoder.bypass_head.weight.abs().max()),
                new_head_optimizer_step=int(optimizer.state[policy.obs_encoder.bypass_head.weight]["step"]),
                ema_optimization_step=ema.optimization_step,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                warm_mean_seconds=float(np.mean(row["warm_update_seconds"])),
                post_update_sample_shapes={k: list(v.shape) for k, v in sampled.items()})
            if (row["new_raw_head_max_abs"] <= 0 or row["new_ema_head_max_abs"] <= 0
                    or row["new_head_optimizer_step"] != protocol["preflight_updates_per_mode"]
                    or row["ema_optimization_step"] != current):
                raise ValueError("New trainable/EMA/Adam update counters incorrect")
            save_json(group / "report.json", row)
            del policy, averaged, optimizer, scheduler, ema, before_raw, before_ema, after_raw, after_ema, before_actions, after_actions, sampled
            gc.collect()
            torch.cuda.empty_cache()
        if first_losses[0] != first_losses[1]:
            raise ValueError("Matched groups differ at zero-head first loss")
        # The immutable source was read only, but also check identity again after both groups.
        if sha(root / protocol["source_checkpoint"]) != protocol["source_checkpoint_sha256"]:
            raise ValueError("Source checkpoint changed")
        result.update(preflight_passed=True, matched_first_loss_exact=True, source_checkpoint_unchanged=True,
                      mean_seconds_per_group={k: v["warm_mean_seconds"] for k, v in result["groups"].items()},
                      estimated_paired_training_seconds=sum(v["warm_mean_seconds"] for v in result["groups"].values()) * protocol["planned_formal_updates_per_mode"],
                      estimate_excludes="Restore, statistics, checkpoints, development rollouts, scheduler waiting; no exact runtime promise")
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(out / "report.json", result)
        print("SPATIAL_BYPASS_PREFLIGHT_REPORT", out / "report.json", flush=True)
    print("SPATIAL BYPASS PREFLIGHT PASSED; no generalization verdict or automatic formal submission", flush=True)


if __name__ == "__main__":
    main()
