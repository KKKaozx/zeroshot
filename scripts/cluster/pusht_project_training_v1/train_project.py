"""Matched frozen-CLIP frontend training, exactly two continuous budget segments."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

import training_core as core
from native_data import TrainingData, sha, save_json
from token_binding import FrozenCache, bind_tokens, project_update


def validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, segment):
    if sidecar.get("project_contract_sha256") != contract_sha:
        raise ValueError("Only this project's complete checkpoints may resume; native learned weights forbidden")
    if sidecar["protocol_sha256"] != protocol_sha or sidecar["sample_plan_sha256"] != plan_sha:
        raise ValueError("Resume base protocol or sample plan changed")
    completed = sidecar["completed_updates"]
    lower = 0 if segment == 1 else 42084
    if not lower <= completed < target or (segment == 1 and completed == 0):
        raise ValueError("Resume counter outside this continuous training segment")
    if sidecar["segment"] not in ((1,) if segment == 1 else (1, 2)):
        raise ValueError("Wrong source segment")
    if not (sidecar.get("startup_checks") or {}).get("disk_resume_replay_exact"):
        raise ValueError("Resume requires the committed startup audit (update2 or later)")
    return completed


def snapshot(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha, contract_sha):
    payload = core.capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha)
    payload["project_contract_sha256"] = contract_sha
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment", type=int, choices=(1, 2), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root / "runs" not in output.parents:
        parser.error("Output must be a new directory under root/runs")
    if args.segment == 2 and args.resume is None:
        parser.error("Segment2 requires segment1's complete checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"segment": args.segment, "segment_completed": False, "training_completed": False,
              "development_complete": False, "errors": [], "completed_updates": 0,
              "final_set_accessed": False, "clip_training_executed": False,
              "trained_native_weights_loaded": False, "automatic_submission": False}
    client = None
    try:
        bundle = Path(__file__).resolve().parent
        contract_path = bundle / "project_contract.json"
        contract = json.loads(contract_path.read_text())
        contract_sha = sha(contract_path)
        protocol_path = bundle / "base_protocol.json"
        protocol_sha = sha(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        for name, digest in contract["bundle_sources_sha256"].items():
            if sha(bundle / name) != digest:
                raise ValueError("Project bundle file changed: " + name)
        if protocol_sha != contract["base_protocol_sha256"]:
            raise ValueError("Native base protocol changed")
        for relative, digest in dict(protocol["required_reports_sha256"], **contract["required_reports_sha256"]).items():
            if sha(root / relative) != digest:
                raise ValueError("Required reference report changed: " + relative)
        native = json.loads((root / "runs/pusht-native84k-212158/report.json").read_text())
        if not (native["training_completed"] and native["development_complete"] and not native["errors"]
                and native["completed_updates"] == 84168
                and native["development"]["reference_competence_gate_passed"]):
            raise ValueError("Native reference is not admitted")
        if not (torch.__version__.split("+")[0] == "1.12.1" and torch.cuda.is_available()):
            raise ValueError("Use the established official CUDA environment")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        if shutil.disk_usage(output).free < 12 * 1024 ** 3:
            raise ValueError("Need12GiB free for atomic complete checkpoints; preserve cache and source checkpoint")
        source = root / "paper_reproduction/official_pusht_v1/source"
        for name, digest in protocol["author_file_sha256"].items():
            if sha(source / name) != digest:
                raise ValueError("Author source changed: " + name)
        sys.path.insert(0, str(source))
        from omegaconf import OmegaConf
        import hydra
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.lr_scheduler import get_scheduler
        from project_frontend import ProjectObservationEncoder
        from resource_probe import state_hash
        data = TrainingData(root, protocol, output)
        cache = FrozenCache(root / contract["cache_directory"], contract, data)
        plan = core.BatchPlan()
        order_sha = plan.digest(84168)
        import hashlib
        plan_sha = hashlib.sha256((order_sha + data.evidence["raw_training_images_sha256"]).encode("ascii")).hexdigest()
        if order_sha != contract["sample_order_sha256"] or plan_sha != contract["sample_plan_sha256"]:
            raise ValueError("Native sample/data plan differs")
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        cfg = OmegaConf.load(bundle / "checkpoint_config.yaml")
        policy = hydra.utils.instantiate(cfg.policy)
        if state_hash(policy.model) != protocol["initial_backend_sha256"]:
            raise ValueError("Fresh backend differs from native initial state")
        # Retain the common fresh backend, replace only the frontend family.
        torch.manual_seed(43)
        policy.obs_encoder = ProjectObservationEncoder(cache.text, cache.mask, (64, 65))
        if sum(p.numel() for p in policy.parameters() if p.requires_grad) != 305159362:
            raise ValueError("Project parameter count differs from resource check")
        normalizer = LinearNormalizer()
        normalizer.fit({"action": data.train_actions, "agent_pos": data.train_positions}, last_n_dims=1, mode="limits")
        normalizer["image"] = get_image_range_normalizer()
        policy.set_normalizer(normalizer)
        if state_hash(policy.normalizer) != protocol["normalizer_sha256"]:
            raise ValueError("Train-only normalizer differs")
        policy.cuda().train()
        averaged = copy.deepcopy(policy)
        averaged.eval()
        ema = hydra.utils.instantiate(cfg.ema, model=averaged)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
        scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500,
                                  num_training_steps=protocol["lr_horizon_updates"])
        fixed_text_digest = core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask))
        completed = 0
        target = contract["segment_endpoints"][args.segment - 1]
        latest = output / "latest.pt"
        startup = None

        def capture_now():
            return snapshot(policy, averaged, optimizer, scheduler, ema, completed,
                            protocol_sha, plan_sha, contract_sha)

        def save_checkpoint():
            core.atomic_checkpoint(latest, capture_now())
            sidecar = {"sha256": sha(latest), "completed_updates": completed,
                       "protocol_sha256": protocol_sha, "sample_plan_sha256": plan_sha,
                       "project_contract_sha256": contract_sha, "segment": args.segment,
                       "startup_checks": startup}
            save_json(latest.with_suffix(".json"), sidecar)
            result.update(checkpoint=str(latest), checkpoint_sha256=sidecar["sha256"])
            save_json(output / "report.json", result)
            print("PUSHT_PROJECT_SAVED", completed, latest, flush=True)

        if args.resume:
            resume = args.resume.resolve()
            if root / "runs" not in resume.parents or resume.name != "latest.pt":
                raise ValueError("Resume must be an explicit project latest.pt under root/runs")
            sidecar = json.loads(resume.with_suffix(".json").read_text())
            validate_resume(sidecar, contract_sha, protocol_sha, plan_sha, target, args.segment)
            if sha(resume) != sidecar["sha256"]:
                raise ValueError("Resume checkpoint checksum differs")
            payload = torch.load(resume, map_location="cpu")
            if payload.get("project_contract_sha256") != contract_sha:
                raise ValueError("Checkpoint payload is not this project run")
            completed = core.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)
            if completed != sidecar["completed_updates"] or core.tree_digest(payload) != core.tree_digest(capture_now()):
                raise ValueError("Restored raw/EMA/Adam/LR/EMA counter/all RNG state differs")
            startup = sidecar["startup_checks"]
            if args.segment == 2 and not startup:
                raise ValueError("Segment1 startup audit missing")
            result.update(complete_resume_state_exact=True, resumed_from=str(resume),
                          resume_source_sha256=sidecar["sha256"])
            del payload
            gc.collect()
            print("PUSHT_PROJECT_RESUME PASSED", completed, "/", target, flush=True)
        elif args.segment != 1:
            raise ValueError("Only segment1 may initialize fresh")
        result.update(project_contract_sha256=contract_sha, protocol_sha256=protocol_sha,
                      sample_order_sha256=order_sha, sample_plan_sha256=plan_sha,
                      initial_backend_sha256=protocol["initial_backend_sha256"],
                      normalizer_sha256=protocol["normalizer_sha256"], data=data.evidence,
                      initial_completed_updates=completed, completed_updates=completed,
                      target_updates=target, total_project_budget=84168,
                      cache_report_sha256=contract["cache_report_sha256"],
                      frozen_clip_cache_verified=True, lr_horizon_updates=512400,
                      fixed_text_digest=fixed_text_digest)

        def one_update(index, audit=False):
            indices = plan.batch(index)
            return project_update(core, policy, optimizer, scheduler, ema, data.batch(indices),
                                  cache.tokens(indices), index, audit=audit)

        with (output / "updates.jsonl").open("x", encoding="utf-8") as log:
            while completed < target:
                loss, check = one_update(completed, audit=(completed == 0))
                completed += 1
                result["completed_updates"] = completed
                log.write(json.dumps({"completed_updates": completed, "loss": loss,
                                      "lr_for_next_update": scheduler.get_last_lr()[0]}) + "\n")
                if completed == 1:
                    # The same mandatory full disk resume audit used by native.
                    save_checkpoint()
                    loss_a, _ = one_update(1)
                    completed = 2
                    expected = core.tree_digest(capture_now())
                    payload = torch.load(latest, map_location="cpu")
                    completed = core.restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)
                    del payload
                    gc.collect()
                    loss_b, _ = one_update(1)
                    completed = 2
                    actual = core.tree_digest(capture_now())
                    if loss_a != loss_b or expected != actual:
                        raise ValueError("Project complete disk resume and update2 differ")
                    startup = {"loss_audit": check, "disk_resume_replay_exact": True,
                               "loss_before_restore": loss_a, "loss_after_restore": loss_b,
                               "state_digest_before_restore": expected, "state_digest_after_restore": actual,
                               "audit_recomputed_updates_excluded_from_training_budget": 1,
                               "scope": "Raw,EMA,Adam,LR,EMA counter,Python/NumPy/CPU/CUDA RNG"}
                    save_json(output / "startup_checks.json", startup)
                    result["completed_updates"] = 2
                    log.write(json.dumps({"completed_updates": 2, "loss": loss_b,
                                          "lr_for_next_update": scheduler.get_last_lr()[0]}) + "\n")
                    print("PUSHT_PROJECT_STARTUP PASSED; original loss, features and disk resume", flush=True)
                    # Commit audited state before a possible interruption.
                    save_checkpoint()
                if completed % 250 == 0:
                    log.flush()
                    print("PUSHT_PROJECT_UPDATE", completed, "/", target, "loss", loss,
                          "seconds", round(time.monotonic() - started, 1), flush=True)
                if completed % 2000 == 0 or completed == target:
                    save_checkpoint()
        if core.tree_digest((policy.obs_encoder.text_tokens, policy.obs_encoder.text_mask)) != fixed_text_digest:
            raise ValueError("Frozen text features changed")
        if core.tree_digest((averaged.obs_encoder.text_tokens, averaged.obs_encoder.text_mask)) != fixed_text_digest:
            raise ValueError("EMA frozen text features changed")
        result.update(segment_completed=True, training_completed=completed == 84168,
                      startup_checks=startup, frozen_text_unchanged=True,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                      training_seconds=time.monotonic() - started,
                      restored_device_checks=core.RESTORE_DEVICE_CHECKS)
        save_json(output / "report.json", result)
        if args.segment == 2:
            del optimizer, scheduler, policy
            gc.collect()
            torch.cuda.empty_cache()
            from live_client import CLIPClient, LiveProjectPolicy
            from evaluate_project import evaluate
            client = CLIPClient(root, output)
            if not (np.array_equal(client.text, cache.text.numpy())
                    and np.array_equal(client.text_mask, cache.mask.numpy())):
                raise ValueError("Live evaluation text differs from training")
            development = evaluate(LiveProjectPolicy(averaged, client), output, protocol)
            close = client.close()
            client = None
            if not (close["all_frozen"] and close["weights_unchanged"]):
                raise ValueError("Evaluation CLIP freeze/hash check failed")
            result.update(development_complete=True, development=development, evaluation_worker=close)
        else:
            result["next_action"] = "Verify this checkpoint, then manually submit segment2; no development scoring or selection"
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if client is not None:
            client.abort()
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(output / "report.json", result)
        print("PUSHT_PROJECT_REPORT", output / "report.json", flush=True)
    print("PUSHT PROJECT SEGMENT COMPLETE", args.segment,
          "; only a completed endpoint rollout can support a performance comparison", flush=True)


if __name__ == "__main__":
    main()
