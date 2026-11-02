"""Bounded real-data resource probe only; never score a rollout or submit training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_root(root, protocol):
    source = root / "paper_reproduction/official_pusht_v1/source"
    for name, expected in protocol["author_file_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("Author source fingerprint changed: " + name)
    sys.path.insert(0, str(source))
    return source


def prepare(root, out, protocol):
    import numpy as np
    import zarr
    source_root(root, protocol)
    if shutil.disk_usage(str(out)).free < 512 * 1024 ** 2:
        raise ValueError("Need 512 MiB free for the small resource cache")
    from diffusion_policy.common.sampler import create_indices
    path = root / "runs/pusht-float-data-210690/report.json"
    if sha(path) != protocol["data_report_sha256"]:
        raise ValueError("Data integrity report changed")
    data = json.loads(path.read_text())
    if not data["data_integrity_passed"]:
        raise ValueError("Data integrity has not passed")
    group = zarr.open_group(data["dataset_path"], mode="r")
    ends = group["meta/episode_ends"][:]
    train_ids = data["author_recipe_counts"]["train_episode_ids"]
    train_mask = np.zeros(len(ends), dtype=bool)
    train_mask[train_ids] = True
    rows = create_indices(ends, 16, train_mask, 1, 7)
    if len(rows) != 10726:
        raise ValueError("Training window pool differs from integrity report")
    chosen = np.random.default_rng(20261018).choice(len(rows), size=64, replace=False)
    image_ids, actions, positions = [], [], []
    for index in chosen:
        start, stop, sample_start, sample_stop = map(int, rows[index])
        ids = np.pad(np.arange(start, stop), (sample_start, 16 - sample_stop), mode="edge")
        image_ids.extend(ids[:2].tolist())
        actions.append(np.stack([group["data/action"][int(i)] for i in ids]))
        positions.append(np.stack([group["data/state"][int(i), :2] for i in ids[:2]]))
    images = np.stack([group["data/img"][int(i)] for i in image_ids])
    if images.dtype != np.float32 or not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError("Unexpected raw image values in probe inputs")
    frame_mask = np.zeros(int(ends[-1]), dtype=bool)
    begin = 0
    for i, end in enumerate(ends):
        if train_mask[i]:
            frame_mask[begin:int(end)] = True
        begin = int(end)
    # Small numeric arrays only; fit data restricted to the 90 training episodes.
    np.savez(out / "inputs.npz", raw_images=images.reshape(64, 2, 96, 96, 3),
             actions=np.stack(actions), agent_pos=np.stack(positions),
             train_actions=group["data/action"][:][frame_mask],
             train_agent_pos=group["data/state"][:, :2][frame_mask])
    result = {"train_episodes": len(train_ids), "train_windows": len(rows),
              "train_frames": int(frame_mask.sum()), "sampled_windows": chosen.tolist(),
              "observation_frame_ids": image_ids, "unique_frames_in_probe": len(set(image_ids)),
              "input_sha256": sha(out / "inputs.npz"), "normalizer_scope": "90 train episodes only",
              "full_cache_estimated_bytes_float32": int(frame_mask.sum()) * 257 * 1024 * 4,
              "dev_or_final_scored": False}
    save(out / "prepare.json", result)
    print("PUSHT_RESOURCE_INPUT_READY", result["train_frames"], "training frames", flush=True)


def encode(root, out, protocol):
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    assets = root / "baseline_setup/clip-vit-large-patch14-v1"
    model = CLIPModel.from_pretrained(str(assets), local_files_only=True)
    model.requires_grad_(False).eval()
    before = state_hash(model)
    tokenizer = CLIPTokenizer.from_pretrained(str(assets), local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(str(assets), local_files_only=True)
    text = tokenizer(protocol["fixed_text"], return_tensors="pt", truncation=True, max_length=77)
    with np.load(out / "inputs.npz") as inputs:
        images = inputs["raw_images"].reshape(128, 96, 96, 3)
    model.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    tokens = []
    seconds = []
    with torch.inference_mode():
        text_tokens = model.text_model(input_ids=text["input_ids"].cuda(),
                                        attention_mask=text["attention_mask"].cuda()).last_hidden_state.cpu().numpy()
        for start in range(0, 128, 8):
            torch.cuda.synchronize()
            tick = time.monotonic()
            pixels = processor(images=list(images[start:start + 8]), return_tensors="pt",
                               do_rescale=True)["pixel_values"].cuda()
            if tuple(pixels.shape[1:]) != (3, 224, 224) or not torch.isfinite(pixels).all():
                raise ValueError("Unexpected CLIP preprocessing")
            tokens.append(model.vision_model(pixel_values=pixels).last_hidden_state.cpu().numpy())
            torch.cuda.synchronize()
            seconds.append(time.monotonic() - tick)
            print("PUSHT_FROZEN_CLIP", start + 8, "/128", flush=True)
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    unchanged = state_hash(model) == before
    assert unchanged and all(not p.requires_grad and p.grad is None for p in model.parameters())
    visual = np.concatenate(tokens)
    assert visual.shape == (128, 257, 1024) and np.isfinite(visual).all()
    np.savez(out / "tokens.npz", visual=visual.reshape(64, 2, 257, 1024),
             text=text_tokens, text_mask=text["attention_mask"].numpy())
    save(out / "clip.json", {"weights_unchanged": unchanged, "all_frozen": True,
                             "visual_shape": list(visual.shape), "cache_dtype": str(visual.dtype),
                             "tokens_sha256": sha(out / "tokens.npz"),
                             "peak_reserved_gib": peak,
                             "warm_seconds_per_8_frames": sum(seconds[2:]) / len(seconds[2:]),
                             "preprocessing": "Local CLIPImageProcessor; raw float pixels /255 once, resize/normalize per CLIP config",
                             "observations": 2, "backbone_trained": False})


def gradient_ok(module):
    import torch
    gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients) and any(bool(g.abs().max() > 0) for g in gradients)


def native_state_columns(encoder, image):
    import torch
    encoder.eval()
    images = image[None].repeat(3, 1, 1, 1)
    positions = torch.tensor([[0., 0.], [0.25, 0.], [0., 0.25]])
    with torch.no_grad():
        features = encoder({"image": images, "agent_pos": positions})
    columns = []
    for axis in range(2):
        delta = features[axis + 1] - features[0]
        changed = torch.nonzero(delta.abs() > 1e-6).reshape(-1)
        if len(changed) != 1 or abs(float(delta[changed[0]]) - 0.25) > 1e-6:
            raise ValueError("Native state branch is not the assumed identity; stop")
        columns.append(int(changed[0]))
    if len(set(columns)) != 2:
        raise ValueError("Native position axes overlap")
    return columns


def measure(root, out, protocol, name):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import hydra
    source_root(root, protocol)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from project_frontend import ProjectObservationEncoder
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    cfg = OmegaConf.load(Path(__file__).parent / "checkpoint_config.yaml")
    policy = hydra.utils.instantiate(cfg.policy)
    assert policy.obs_feature_dim == 66 and policy.horizon == 16 and policy.n_obs_steps == 2
    initial_backend = state_hash(policy.model)
    with np.load(out / "inputs.npz") as file:
        arrays = {k: file[k] for k in file.files}
    normalizer = LinearNormalizer()
    normalizer.fit({"action": arrays["train_actions"], "agent_pos": arrays["train_agent_pos"]}, last_n_dims=1, mode="limits")
    normalizer["image"] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    # Do not assume whether robomimic concatenates state before or after RGB.
    image = torch.from_numpy(arrays["raw_images"][0, 0].transpose(2, 0, 1) / 255)
    image = normalizer["image"].normalize(image)
    state_columns = native_state_columns(policy.obs_encoder, image)
    tokens = None
    if name == "project":
        with np.load(out / "tokens.npz") as file:
            tokens = torch.from_numpy(file["visual"])
            torch.manual_seed(43)
            policy.obs_encoder = ProjectObservationEncoder(torch.from_numpy(file["text"]), torch.from_numpy(file["text_mask"]), state_columns)
    normalizer_hash = state_hash(policy.normalizer)
    policy.cuda().train()
    averaged = copy.deepcopy(policy)
    ema = EMAModel(averaged, update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4,
                                 betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    images = torch.from_numpy(arrays["raw_images"].transpose(0, 1, 4, 2, 3) / 255)
    actions, positions = torch.from_numpy(arrays["actions"]), torch.from_numpy(arrays["agent_pos"])
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    gradients = {}
    for update in range(5):
        torch.cuda.synchronize()
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, 64, 8):
            if tokens is not None:
                policy.obs_encoder.set_tokens(tokens[start:start + 8].reshape(16, 257, 1024).cuda())
            batch = {"obs": {"image": images[start:start + 8].cuda(), "agent_pos": positions[start:start + 8].cuda()},
                     "action": actions[start:start + 8].cuda()}
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            (loss / 8).backward()
            total += float(loss.detach()) / 8
            if tokens is not None:
                policy.obs_encoder.cached_tokens = None
        if update == 0:
            gradients = {"frontend": gradient_ok(policy.obs_encoder), "backend": gradient_ok(policy.model)}
            if tokens is not None:
                gradients.update(adapter=gradient_ok(policy.obs_encoder.adapter), projection=gradient_ok(policy.obs_encoder.projection))
            assert all(gradients.values()), gradients
        optimizer.step()
        ema.step(policy)
        torch.cuda.synchronize()
        times.append(time.monotonic() - tick)
        losses.append(total)
        print("PUSHT_RESOURCE_UPDATE", name, update + 1, "/5 loss", total, "seconds", times[-1], flush=True)
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    averaged.eval()
    if tokens is not None:
        averaged.obs_encoder.set_tokens(tokens[:2].reshape(4, 257, 1024).cuda())
    torch.cuda.synchronize()
    tick = time.monotonic()
    with torch.no_grad():
        prediction = averaged.predict_action({"image": images[:2].cuda(), "agent_pos": positions[:2].cuda()})
    torch.cuda.synchronize()
    sample_seconds = time.monotonic() - tick
    assert tuple(prediction["action"].shape) == (2, 8, 2) and torch.isfinite(prediction["action_pred"]).all()
    result = {"group": name, "initial_backend_sha256": initial_backend, "normalizer_sha256": normalizer_hash,
              "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0), "precision": "float32",
              "native_agent_pos_columns": state_columns,
              "logical_batch": 64, "microbatch": 8, "accumulation": 8, "updates": 5, "warm_updates": 3,
              "warm_update_seconds": sum(times[2:]) / 3, "peak_reserved_gib": peak,
              "update_seconds": times,
              "overall_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
              "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
              "first_gradients_finite_and_nonzero": gradients, "losses_resource_only": losses,
              "ema_included_in_update_timing": True, "sampling_ddpm100_seconds_batch2": sample_seconds,
              "sampling_action_shape": list(prediction["action"].shape), "rollout_executed": False,
              "cache_frontend_state": "fresh random init; no pretrained Push-T or Bridge backend loaded",
              "note": "Timing/gradient probe, same 64 train windows repeated. Diffusion draws not paired across frontends; no efficacy comparison."}
    save(out / (name + ".json"), result)


def summarize(out, protocol):
    records = {name: json.loads((out / (name + ".json")).read_text()) for name in ("prepare", "clip", "native", "project")}
    assert records["native"]["initial_backend_sha256"] == records["project"]["initial_backend_sha256"]
    assert records["native"]["normalizer_sha256"] == records["project"]["normalizer_sha256"]
    assert records["native"]["native_agent_pos_columns"] == records["project"]["native_agent_pos_columns"]
    assert records["clip"]["weights_unchanged"] and records["clip"]["all_frozen"]
    estimates = {name: records[name]["warm_update_seconds"] * 84168 / 3600 for name in ("native", "project")}
    result = {"resource_execution_passed": True, "records": records,
              "reference_budget_projection_only": {"updates_each": 84168, "training_gpu_hours": estimates,
                  "excludes": "full token cache preparation, logging/checkpoints, closed-loop evaluation, live CLIP bridge and I/O variation"},
              "formal_training_ready": False, "automatic_formal_submission": False,
              "remaining": ["Final matched protocol/budget review", "Cached-vs-live feature identity and closed-loop IPC verification", "Lock training/dev/final boundaries before scoring"],
              "generalization_verdict": None}
    save(out / "report.json", result)
    print("PUSHT_FRONTEND_RESOURCE_REPORT", out / "report.json", flush=True)
    print("PUSHT FRONTEND RESOURCE CHECK COMPLETE; no generalization verdict or automatic training", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "clip", "native", "project", "summary"))
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Output must be under root/runs")
    if args.stage == "prepare":
        out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    try:
        if args.stage == "prepare":
            prepare(root, out, protocol)
        elif args.stage == "clip":
            encode(root, out, protocol)
        elif args.stage in ("native", "project"):
            measure(root, out, protocol, args.stage)
        else:
            summarize(out, protocol)
    except Exception as exc:
        save(out / (args.stage + "_error.json"), {"stage": args.stage, "error": type(exc).__name__ + ": " + str(exc),
             "formal_training_ready": False, "no_fallback_or_install": True})
        raise


if __name__ == "__main__":
    main()
