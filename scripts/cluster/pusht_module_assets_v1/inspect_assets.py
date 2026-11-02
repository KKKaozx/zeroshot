"""Read existing Push-T assets on a CPU node; never construct a policy/dataset."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

COMMIT = "5ba07ac6661db573af695b419a7947ecb704690f"
SOURCE_FILES = (
    "diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py",
    "diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py",
    "diffusion_policy/model/diffusion/conditional_unet1d.py",
    "diffusion_policy/dataset/pusht_image_dataset.py",
    "diffusion_policy/common/sampler.py",
)
PACKAGES = ("torch", "torchvision", "numpy", "zarr", "numcodecs", "dill",
            "omegaconf", "diffusers", "transformers", "huggingface-hub", "Pillow")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()[:2000]}


def environment(interpreter):
    if not interpreter.is_file():
        return {"interpreter": str(interpreter), "exists": False}
    # Distribution metadata only: no torch/transformers import in the other env.
    code = """import importlib.metadata as m, json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': versions}))
"""
    result = command([str(interpreter), "-I", "-c", code, json.dumps(PACKAGES)])
    if result["returncode"] == 0:
        return json.loads(result["stdout"])
    return {"interpreter": str(interpreter), "error": result}


def author_window_counts(ends, config, np):
    """Fixed-commit sampler formula, using only episode_ends, never frame arrays."""
    lengths = np.diff(np.r_[0, ends])
    n = len(lengths)
    ratio = float(config["task.dataset.val_ratio"])
    seed = int(config["task.dataset.seed"])
    val = np.zeros(n, dtype=bool)
    if ratio > 0:
        n_val = min(max(1, round(n * ratio)), n - 1)
        val[np.random.default_rng(seed).choice(n, n_val, replace=False)] = True
    train = ~val
    maximum = config["task.dataset.max_train_episodes"]
    if maximum is not None and int(train.sum()) > int(maximum):
        current = np.flatnonzero(train)
        chosen = np.random.default_rng(seed).choice(len(current), int(maximum), replace=False)
        train = np.zeros(n, dtype=bool)
        train[current[chosen]] = True
    horizon = int(config["task.dataset.horizon"])
    before = min(max(int(config["task.dataset.pad_before"]), 0), horizon - 1)
    after = min(max(int(config["task.dataset.pad_after"]), 0), horizon - 1)
    counts = np.maximum(0, lengths - horizon + before + after + 1)
    return {"episodes": n, "frames": int(ends[-1]),
            "length_min": int(lengths.min()), "length_max": int(lengths.max()),
            "train_episode_ids": np.flatnonzero(train).tolist(),
            "train_episodes": int(train.sum()), "train_windows": int(counts[train].sum()),
            "original_val_mask_episodes": int(val.sum()),
            "author_validation_complement_episodes": int((~train).sum()),
            "author_validation_complement_windows": int(counts[~train].sum()),
            "scope": "Existing checkpoint recipe only; not a new experiment split or budget."}


def inspect_dataset(path, config):
    result = {"path": str(path), "exists": path.exists(), "frames_read": False}
    if not path.exists():
        return result
    import numpy as np
    import zarr
    store = None
    try:
        if path.is_file():
            store = zarr.ZipStore(str(path), mode="r")
            group = zarr.open_group(store=store, mode="r")
        else:
            group = zarr.open_group(str(path), mode="r")
        arrays = {}
        for key in ("data/img", "data/state", "data/action", "meta/episode_ends"):
            array = group[key]
            arrays[key] = {"shape": list(array.shape), "chunks": list(array.chunks),
                           "dtype": str(array.dtype),
                           "uncompressed_bytes": math.prod(array.shape) * array.dtype.itemsize}
        result["arrays"] = arrays
        meta = group["meta/episode_ends"]
        if meta.ndim != 1 or meta.size == 0 or meta.size > 1000000 or meta.dtype.kind not in "iu":
            raise ValueError("Unexpected episode_ends schema/size; no data read")
        ends = meta[:]  # The only array values read, at most 8 MB for int64.
        if ends[0] <= 0 or not np.all(ends[1:] > ends[:-1]):
            raise ValueError("Episode boundaries are not strictly increasing")
        for key in ("data/img", "data/state", "data/action"):
            if arrays[key]["shape"][0] != int(ends[-1]):
                raise ValueError("Frame count inconsistent with episode_ends: " + key)
        if arrays["data/img"]["shape"][1:] != [96, 96, 3]:
            raise ValueError("Expected author Push-T image shape [N,96,96,3]")
        if arrays["data/action"]["shape"][1:] != [2]:
            raise ValueError("Expected author Push-T action shape [N,2]")
        if len(arrays["data/state"]["shape"]) != 2 or arrays["data/state"]["shape"][1] < 2:
            raise ValueError("Expected state with at least 2 coordinates")
        result["episode_ends_sha256"] = hashlib.sha256(ends.tobytes()).hexdigest()
        result["author_recipe_counts"] = author_window_counts(ends, config, np)
        result["status"] = "metadata_consistent; frame_chunks_not_verified"
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        if store is not None:
            store.close()
    return result


def inspect_checkpoint(path, source):
    import dill
    import torch
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    sys.path.insert(0, str(source))
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    paths = (
        "training.num_epochs", "training.seed", "training.gradient_accumulate_every",
        "training.max_train_steps", "training.use_ema", "dataloader.batch_size",
        "policy.horizon", "policy.n_obs_steps", "policy.n_action_steps",
        "policy.obs_as_global_cond", "policy.diffusion_step_embed_dim",
        "policy.down_dims", "policy.shape_meta", "policy.noise_scheduler",
        "task.dataset.zarr_path", "task.dataset.horizon", "task.dataset.pad_before",
        "task.dataset.pad_after", "task.dataset.seed", "task.dataset.val_ratio",
        "task.dataset.max_train_episodes",
    )
    config = {}
    for key in paths:
        value = OmegaConf.select(cfg, key)
        config[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    progress = {}
    for key in ("global_step", "epoch"):
        value = payload.get("pickles", {}).get(key)
        if isinstance(value, (bytes, bytearray)):
            value = dill.loads(value)
        if value is None:
            value = payload.get(key)
        progress[key] = int(value) if isinstance(value, int) else {"unresolved_type": type(value).__name__}
    states = payload["state_dicts"]
    state = states["model"]
    key = "model.down_modules.0.0.cond_encoder.1.weight"
    interface = {"weight_key": key, "resolved": False}
    if key in state and config["policy.obs_as_global_cond"]:
        shape = list(state[key].shape)
        embedding = int(config["policy.diffusion_step_embed_dim"])
        n_obs = int(config["policy.n_obs_steps"])
        global_dim = shape[1] - embedding
        if global_dim <= 0 or global_dim % n_obs:
            raise ValueError("Cannot infer observation width from conditional weight")
        obs = config["policy.shape_meta"]["obs"]
        if set(obs) != {"image", "agent_pos"} or obs["agent_pos"]["shape"] != [2]:
            raise ValueError("Unexpected observation schema; do not infer RGB width")
        interface.update(resolved=True, weight_shape=shape, global_condition_dim=global_dim,
                         per_observation_feature_dim=global_dim // n_obs,
                         rgb_feature_dim=global_dim // n_obs - 2,
                         preserved_agent_pos_dim=2)
    optimizer = states.get("optimizer", {})
    steps = []
    for item in optimizer.get("state", {}).values():
        step = item.get("step")
        if isinstance(step, torch.Tensor) and step.numel() == 1:
            steps.append(int(step.item()))
        elif isinstance(step, (int, float)):
            steps.append(int(step))
    return {"config": config, "saved_progress": progress, "interface": interface,
            "optimizer_step_summary": {"parameters_with_counter": len(steps),
                                       "min": min(steps) if steps else None,
                                       "max": max(steps) if steps else None},
            "progress_caveat": "Saved counters, not proof every planned epoch ran; exact update semantics require workspace review.",
            "device": "cpu", "model_instantiated": False, "model_forward_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Optional explicit existing Zarr or zip path")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Output must be a NEW directory beneath root/runs")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    started = time.monotonic()
    task = root / "paper_reproduction/official_pusht_v1"
    source = task / "source"
    report = {"stage": "read_only_cpu_asset_inventory", "training_updates": 0,
              "gpu_requested": False, "installs_or_downloads": False,
              "errors": [], "unresolved": [
                  "CLIP/Adapter runtime compatibility (metadata is not an import/forward test)",
                  "Matched training protocol and actual GPU memory/speed/total budget",
                  "Training dataset frame-chunk completeness/content integrity",
              ]}
    try:
        report["environments"] = {name: environment(root / "envs" / name / "bin/python")
                                  for name in ("official-pusht-v1", "bridge-diffusion")}
        report["source"] = {"path": str(source), "expected_commit": COMMIT,
                            "head": command(["git", "-C", str(source), "rev-parse", "HEAD"]),
                            "tracked_changes": command(["git", "-C", str(source), "diff", "HEAD", "--name-only"]),
                            "file_sha256": {p: sha(source / p) if (source / p).is_file() else None for p in SOURCE_FILES}}
        checkpoint = task / "pusht_image.ckpt"
        record = task / "checkpoint.sha256"
        report["checkpoint_file"] = {"path": str(checkpoint), "exists": checkpoint.is_file(),
                                     "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                                     "previous_sha_record": record.read_text().strip()[:1000] if record.is_file() else None,
                                     "sha_freshly_verified": False}
        print("ASSET_INVENTORY reading existing checkpoint on CPU", flush=True)
        report["checkpoint"] = inspect_checkpoint(checkpoint, source)
        config = report["checkpoint"]["config"]
        dataset = Path(config["task.dataset.zarr_path"])
        candidates = [dataset] if dataset.is_absolute() else [source / dataset, task / dataset, root / dataset]
        candidates += [Path(str(p) + ".zip") for p in list(candidates)]
        if args.dataset is not None:
            candidates.insert(0, args.dataset.resolve())
        candidates = list(dict.fromkeys(candidates))
        report["datasets"] = [inspect_dataset(path, config) for path in candidates]
        report["dataset_search_scope"] = "Only listed paths; absence here does not establish absence elsewhere. No downloads."
        clip = root / "baseline_setup/clip-vit-large-patch14-v1"
        report["clip_files"] = {"path": str(clip), "exists": clip.is_dir(),
                                "top_level_files": [{"name": p.name, "bytes": p.stat().st_size}
                                                    for p in sorted(clip.iterdir()) if p.is_file()] if clip.is_dir() else []}
        for name in ("config.json", "preprocessor_config.json"):
            path = clip / name
            if path.is_file():
                report["clip_files"][name] = {"sha256": sha(path), "content": json.loads(path.read_text())}
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    report["elapsed_seconds"] = time.monotonic() - started
    report["inventory_execution_complete"] = not report["errors"]
    with (out / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print("PUSHT_ASSET_REPORT", out / "report.json", flush=True)
    print("Inventory complete:", report["inventory_execution_complete"], flush=True)
    print("No training, inference, installation, download, or GPU use; no generalization verdict.", flush=True)
    if report["errors"]:
        print("ERRORS", json.dumps(report["errors"]), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
