"""Native Push-T reference: 20k updates then fixed EMA development rollouts.

No project training, pretrained weights, final-set access, downloads or installs.
"""
import argparse
from collections import deque
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

from training_core import (BatchPlan, atomic_checkpoint, audited_original_loss,
                           capture, restore, tree_digest, update)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temporary = Path(str(path) + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


class TrainingData:
    def __init__(self, root, protocol, output):
        import zarr
        from diffusion_policy.common.sampler import create_indices
        data_path = root / "runs/pusht-float-data-210690/report.json"
        if sha(data_path) != protocol["data_report_sha256"]:
            raise ValueError("Data report identity changed")
        report = json.loads(data_path.read_text())
        assert report["data_integrity_passed"]
        group = zarr.open_group(report["dataset_path"], mode="r")
        ends = group["meta/episode_ends"][:]
        self.actions, states = group["data/action"][:], group["data/state"][:]
        for name, array in (("meta/episode_ends", ends), ("data/action", self.actions), ("data/state", states)):
            if hashlib.sha256(array.tobytes()).hexdigest() != report["arrays"][name]["decoded_sha256"]:
                raise ValueError("Numeric dataset changed: " + name)
        mask = np.zeros(len(ends), dtype=bool)
        mask[report["author_recipe_counts"]["train_episode_ids"]] = True
        self.rows = create_indices(ends, 16, mask, 1, 7)
        assert self.rows.shape == (10726, 4) and int(mask.sum()) == 90
        frame_mask = np.zeros(int(ends[-1]), dtype=bool)
        begin = 0
        for i, end in enumerate(ends):
            if mask[i]:
                frame_mask[begin:int(end)] = True
            begin = int(end)
        ids = np.flatnonzero(frame_mask)
        assert len(ids) == 11356
        self.train_actions = self.actions[ids]
        self.train_positions = states[ids, :2]
        self.positions = states[:, :2]
        self.lookup = np.full(len(frame_mask), -1, dtype=np.int64)
        self.lookup[ids] = np.arange(len(ids))
        self.images = np.empty((len(ids), 96, 96, 3), dtype=np.float32)
        image_array = group["data/img"]
        for start in range(0, len(frame_mask), int(image_array.chunks[0])):
            stop = min(start + int(image_array.chunks[0]), len(frame_mask))
            selected = ids[(ids >= start) & (ids < stop)]
            if len(selected):
                raw = image_array[start:stop][selected - start]
                if raw.dtype != np.float32 or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 255:
                    raise ValueError("Training pixels violate the verified raw contract")
                self.images[self.lookup[selected]] = raw
            if start % (int(image_array.chunks[0]) * 20) == 0:
                print("PUSHT_NATIVE_LOAD", start, "/", len(frame_mask), flush=True)
        reference = root / "runs/pusht-front-resource-210763/inputs.npz"
        if sha(reference) != protocol["resource_inputs_sha256"]:
            raise ValueError("Resource input identity changed")
        chosen = np.random.default_rng(20261018).choice(len(self.rows), 64, replace=False)
        actual = self.batch(chosen)
        with np.load(reference, allow_pickle=False) as ref:
            assert np.array_equal(self.train_actions, ref["train_actions"])
            assert np.array_equal(self.train_positions, ref["train_agent_pos"])
            assert np.array_equal(actual["action"].numpy(), ref["actions"])
            assert np.array_equal(actual["obs"]["agent_pos"].numpy(), ref["agent_pos"])
            expected_images = ref["raw_images"].transpose(0, 1, 4, 2, 3) / np.float32(255)
            assert np.array_equal(actual["obs"]["image"].numpy(), expected_images)
        self.evidence = {"train_episodes": 90, "train_windows": len(self.rows), "train_frames": len(ids),
                         "raw_training_images_sha256": hashlib.sha256(self.images.tobytes()).hexdigest(),
                         "resource_64_windows_exact": True, "normalizer_inputs_exact": True,
                         "normalizer_scope": "90 train episodes only", "data_changed": False}
        save_json(output / "training_data.json", self.evidence)

    def batch(self, indices):
        frames = []
        for row in self.rows[indices]:
            start, stop, sample_start, sample_stop = map(int, row)
            frames.append(np.pad(np.arange(start, stop), (sample_start, 16 - sample_stop), mode="edge"))
        frames = np.stack(frames)
        local = self.lookup[frames[:, :2]]
        if (local < 0).any():
            raise ValueError("A training window escaped the training split")
        return {"obs": {"image": torch.from_numpy(self.images[local].transpose(0, 1, 4, 2, 3) / np.float32(255)),
                        "agent_pos": torch.from_numpy(self.positions[frames[:, :2]].copy())},
                "action": torch.from_numpy(self.actions[frames].copy())}


def evaluate(policy, output, protocol):
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
    policy.eval()
    records = []
    for index in range(protocol["development"]["count"]):
        seed = protocol["development"]["start_seed"] + index
        filename = output / ("development_seed_%d.mp4" % seed) if index < 2 else None
        recorder = VideoRecorder.create_h264(fps=10, codec="h264", input_pix_fmt="rgb24",
                                            crf=22, thread_type="FRAME", thread_count=1)
        video_env = VideoRecordingWrapper(PushTImageEnv(legacy=True, render_size=96),
                                          video_recoder=recorder, file_path=str(filename) if filename else None,
                                          steps_per_render=1)
        env = MultiStepWrapper(video_env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)
        try:
            env.seed(seed)
            observation = env.reset()
            policy.reset()
            done, decision = False, 0
            while not done:
                if decision >= 38:
                    raise ValueError("Author multistep max-episode limit did not terminate")
                obs = {k: torch.from_numpy(v[None]).to(policy.device) for k, v in observation.items()}
                torch.manual_seed(protocol["development"]["diffusion_seed_base"] + index * 1000 + decision)
                with torch.no_grad():
                    actions = policy.predict_action(obs)["action"][0].cpu().numpy()
                assert actions.shape == (8, 2) and np.isfinite(actions).all()
                observation, _, done, _ = env.step(actions)
                decision += 1
            rewards = np.asarray(env.reward, dtype=np.float64)
            assert len(rewards) and np.isfinite(rewards).all()
            score = float(rewards.max())
            records.append({"environment_seed": seed, "max_overlap_reward": score,
                            "full_reward": score == 1.0, "steps": len(rewards),
                            "reward_series": rewards.tolist(),
                            "video": filename.name if filename else None})
        finally:
            recorder.stop()
            env.close()
        save_json(output / "development_partial.json", {"episodes": records, "complete": False})
        print("PUSHT_NATIVE_DEV", index + 1, "/20", seed, "score", score, flush=True)
    scores = np.asarray([row["max_overlap_reward"] for row in records])
    fraction = float(np.mean([row["full_reward"] for row in records]))
    means = np.random.RandomState(90210).choice(scores, (20000, len(scores)), replace=True).mean(1)
    n, z = len(scores), 1.959963984540054
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    mean = float(scores.mean())
    passed = mean >= protocol["development"]["competence_mean_score"] and fraction >= protocol["development"]["competence_full_reward_fraction"]
    result = {"complete": True, "split": "development_only", "episodes": records,
              "mean_score": mean, "full_reward_fraction": fraction,
              "mean_score_episode_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
              "full_reward_wilson_95_interval": [center - half, center + half],
              "reference_competence_gate_passed": passed,
              "checkpoint_selection": "EMA at exactly20000 updates; no best-of-development selection",
              "not_robot_grasp_success": True, "final_set_accessed": False,
              "next_action": "Review native result before preparing a matched project run" if passed else
                             "Native reference not established at20k; inspect evidence before any predefined-cap continuation; no project attribution"}
    save_json(output / "development.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root / "runs" not in output.parents:
        parser.error("Output must be a new directory under root/runs")
    output.mkdir(parents=True, exist_ok=False)
    result = {"training_completed": False, "development_complete": False, "errors": [],
              "project_training_executed": False, "final_set_accessed": False,
              "completed_updates": 0}
    started = time.monotonic()
    try:
        bundle = Path(__file__).resolve().parent
        protocol_path = bundle / "training_protocol.json"
        protocol_sha = sha(protocol_path)
        protocol = json.loads(protocol_path.read_text())
        assert torch.__version__.split("+")[0] == "1.12.1" and torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        if shutil.disk_usage(output).free < 15 * 1024 ** 3:
            raise ValueError("Need15GiB free for atomic raw/EMA/Adam checkpoints and videos")
        for relative, digest in protocol["required_reports_sha256"].items():
            if sha(root / relative) != digest:
                raise ValueError("Required report changed: " + relative)
        live = json.loads((root / "runs/pusht-online-211193/report.json").read_text())
        assert live["bridge_execution_passed"] and not live["errors"] and live["worker"]["all_frozen"] and live["worker"]["weights_unchanged"]
        source = root / "paper_reproduction/official_pusht_v1/source"
        for name, digest in protocol["author_file_sha256"].items():
            if sha(source / name) != digest:
                raise ValueError("Author file changed: " + name)
        sys.path.insert(0, str(source))
        from omegaconf import OmegaConf
        import hydra
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.lr_scheduler import get_scheduler
        from diffusion_policy.model.diffusion.ema_model import EMAModel
        from resource_probe import state_hash
        data = TrainingData(root, protocol, output)
        plan = BatchPlan()
        order_sha = plan.digest(protocol["native_upper_cap_updates"])
        # Resume must also reject changed training pixels outside the old64 probe.
        plan_sha = hashlib.sha256((order_sha + data.evidence["raw_training_images_sha256"]).encode("ascii")).hexdigest()
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        cfg = OmegaConf.load(bundle / "checkpoint_config.yaml")
        policy = hydra.utils.instantiate(cfg.policy)
        if state_hash(policy.model) != protocol["initial_backend_sha256"]:
            raise ValueError("Native initial backend differs from the common reference")
        normalizer = LinearNormalizer()
        normalizer.fit({"action": data.train_actions, "agent_pos": data.train_positions}, last_n_dims=1, mode="limits")
        normalizer["image"] = get_image_range_normalizer()
        policy.set_normalizer(normalizer)
        if state_hash(policy.normalizer) != protocol["normalizer_sha256"]:
            raise ValueError("Train-only normalizer differs from established reference")
        policy.cuda().train()
        averaged = copy.deepcopy(policy)
        ema = hydra.utils.instantiate(cfg.ema, model=averaged)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
        scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500,
                                  num_training_steps=protocol["lr_horizon_updates"])
        completed = 0
        latest = output / "latest.pt"
        audit_loss = None
        if args.resume:
            resume = args.resume.resolve()
            if root / "runs" not in resume.parents or resume.name != "latest.pt":
                raise ValueError("Resume only an explicit latest.pt under root/runs")
            sidecar = json.loads(resume.with_suffix(".json").read_text())
            if sha(resume) != sidecar["sha256"]:
                raise ValueError("Resume checkpoint checksum differs")
            payload = torch.load(resume, map_location="cpu")
            completed = restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)
            del payload
            gc.collect()
            result["resumed_from"] = str(resume)
        assert 0 <= completed <= protocol["first_checkpoint_updates"]
        result.update(protocol_sha256=protocol_sha, sample_plan_sha256=plan_sha,
                      sample_order_sha256=order_sha,
                      initial_backend_sha256=protocol["initial_backend_sha256"],
                      normalizer_sha256=protocol["normalizer_sha256"], data=data.evidence,
                      lr_horizon_updates=protocol["lr_horizon_updates"],
                      deterministic_algorithms=True, initial_completed_updates=completed)
        audit_path = output / "startup_checks.json"
        # On resume inspect one microbatch objective without changing parameters.
        if args.resume:
            audit_batch = data.batch(plan.batch(completed if completed < 20000 else 19999)[:8])
            device_batch = {"obs": {k: v.cuda() for k, v in audit_batch["obs"].items()},
                            "action": audit_batch["action"].cuda()}
            loss, check = audited_original_loss(policy, device_batch, completed, 0)
            del loss, device_batch, audit_batch
            save_json(audit_path, {"loss_audit": check, "resume_checksum_verified": True,
                                   "resume_origin_sidecar": sidecar,
                                   "note": "Restored complete state; original fresh-run replay audit is retained in checkpoint sidecar"})
        log = (output / "updates.jsonl").open("x", encoding="utf-8")
        try:
            while completed < protocol["first_checkpoint_updates"]:
                loss, check = update(policy, optimizer, scheduler, ema,
                                     data.batch(plan.batch(completed)), completed, audit=(completed == 0))
                completed += 1
                result["completed_updates"] = completed
                row = {"completed_updates": completed, "loss": loss, "lr_for_next_update": scheduler.get_last_lr()[0]}
                log.write(json.dumps(row) + "\n")
                if completed == 1:
                    audit_loss = check
                    payload = capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha)
                    atomic_checkpoint(latest, payload)
                    del payload
                    # Compute update2, reload the on-disk update1 state, replay2.
                    second_batch = data.batch(plan.batch(1))
                    loss_a, _ = update(policy, optimizer, scheduler, ema, second_batch, 1)
                    expected = tree_digest(capture(policy, averaged, optimizer, scheduler, ema, 2, protocol_sha, plan_sha))
                    payload = torch.load(latest, map_location="cpu")
                    assert restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha) == 1
                    del payload
                    gc.collect()
                    loss_b, _ = update(policy, optimizer, scheduler, ema, second_batch, 1)
                    actual = tree_digest(capture(policy, averaged, optimizer, scheduler, ema, 2, protocol_sha, plan_sha))
                    if expected != actual or loss_a != loss_b:
                        raise ValueError("Complete disk resume replay differs; training stopped at startup")
                    completed = 2
                    result["completed_updates"] = completed
                    result["resume_replay_exact"] = True
                    save_json(audit_path, {"loss_audit": audit_loss, "disk_resume_replay_exact": True,
                               "loss_before_restore": loss_a, "loss_after_restore": loss_b,
                               "state_digest_before_restore": expected, "state_digest_after_restore": actual,
                               "state_scope": "raw,EMA,Adam,LR,EMA counter,Python/NumPy/CPU/CUDA RNG",
                               "audit_recomputed_updates_excluded_from_training_budget": 1})
                    save_json(latest.with_suffix(".json"), {"sha256": sha(latest), "completed_updates": 1,
                               "protocol_sha256": protocol_sha, "sample_plan_sha256": plan_sha,
                               "startup_checks": json.loads(audit_path.read_text())})
                    print("PUSHT_NATIVE_STARTUP PASSED; original loss and complete resume exact", flush=True)
                    log.write(json.dumps({"completed_updates": 2, "loss": loss_b, "lr_for_next_update": scheduler.get_last_lr()[0]}) + "\n")
                if completed % 250 == 0:
                    log.flush()
                    print("PUSHT_NATIVE_UPDATE", completed, "/20000 loss", loss, "seconds", round(time.monotonic() - started, 1), flush=True)
                if completed % 2000 == 0 or completed == 20000:
                    atomic_checkpoint(latest, capture(policy, averaged, optimizer, scheduler, ema, completed, protocol_sha, plan_sha))
                    save_json(latest.with_suffix(".json"), {"sha256": sha(latest), "completed_updates": completed,
                               "protocol_sha256": protocol_sha, "sample_plan_sha256": plan_sha,
                               "startup_checks": json.loads(audit_path.read_text())})
                    save_json(output / "report.json", result)
                    print("PUSHT_NATIVE_SAVED", completed, latest, flush=True)
        finally:
            log.close()
        # Resume-at20k has no new checkpoint; record its immutable input identity.
        checkpoint = latest if latest.exists() else args.resume.resolve()
        result.update(training_completed=True, completed_updates=completed,
                      checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                      training_seconds=time.monotonic() - started)
        save_json(output / "report.json", result)
        del optimizer, scheduler, policy
        gc.collect()
        torch.cuda.empty_cache()
        development = evaluate(averaged, output, protocol)
        result.update(development_complete=True, development=development)
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save_json(output / "report.json", result)
        print("PUSHT_NATIVE_REPORT", output / "report.json", flush=True)
    print("PUSHT_NATIVE_20K COMPLETE; inspect development competence before any project training", flush=True)
    print("PUSHT_NATIVE_COMPETENCE", development["reference_competence_gate_passed"],
          "mean", development["mean_score"], "full_reward_fraction", development["full_reward_fraction"], flush=True)


if __name__ == "__main__":
    main()
