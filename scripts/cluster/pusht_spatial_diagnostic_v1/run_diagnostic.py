"""Bounded frozen spatial readout, fixed protocol; never train or execute a policy."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from probe_math import choose_frames, labels_from_state, fit_readout, summarize, paired_interval
from extract_features import project_features


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verified_json(path, digest):
    if sha(path) != digest:
        raise ValueError("Changed input: " + str(path))
    return json.loads(Path(path).read_text())


def endpoint_frontend(root, bundle, spec, native=False):
    """Read complete immutable snapshots, retain only EMA observation modules."""
    from resource_probe import state_hash
    ckpt = root / spec["path"]
    sidecar = json.loads(ckpt.with_suffix(".json").read_text())
    if sidecar["sha256"] != spec["sha256"] or sha(ckpt) != spec["sha256"]:
        raise ValueError("Endpoint checkpoint SHA differs")
    if sidecar["completed_updates"] != 84168:
        raise ValueError("Not the fixed endpoint")
    payload = torch.load(ckpt, map_location="cpu")
    base = json.loads((bundle / "base_protocol.json").read_text())
    if not (payload["format"] == "pusht-native-complete-v1" and payload["completed_updates"] == 84168
            and payload["protocol_sha256"] == spec["protocol_sha256"]
            and payload["plan_sha256"] == spec["sample_plan_sha256"]):
        raise ValueError("Checkpoint contract differs")
    prefix = "obs_encoder."
    state = {k[len(prefix):]: v for k, v in payload["ema"].items() if k.startswith(prefix)}
    normalizer = {k[len("normalizer."):]: v for k, v in payload["ema"].items() if k.startswith("normalizer.")}
    if native:
        import hydra
        from omegaconf import OmegaConf
        # Original constructor, no checkpoint backend loading or decoder forward.
        torch.manual_seed(42)
        policy = hydra.utils.instantiate(OmegaConf.load(bundle / "checkpoint_config.yaml").policy)
        model = policy.obs_encoder
        policy.normalizer.load_state_dict(normalizer)
        normalizer_model = policy.normalizer
        del policy
        if state_hash(normalizer_model) != base["normalizer_sha256"]:
            raise ValueError("Native train-only normalizer differs")
    else:
        from project_frontend import ProjectObservationEncoder
        if payload.get("project_contract_sha256") != spec["project_contract_sha256"]:
            raise ValueError("Not the verified project family checkpoint")
        model = ProjectObservationEncoder(state["text_tokens"], state["text_mask"], (64, 65))
        normalizer_model = None
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().cuda()
    if normalizer_model is not None:
        normalizer_model.requires_grad_(False).eval().cuda()
    del payload, state, normalizer
    gc.collect()
    return model, normalizer_model, state_hash(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("New output must be under root/runs")
    out.mkdir(parents=True, exist_ok=False)
    bundle = Path(__file__).resolve().parent
    started = time.monotonic()
    result = {"diagnostic_complete": False, "errors": [], "policy_training_updates": 0,
              "policy_rollouts": 0, "sealed_final_accessed": False,
              "policy_development_scores_used": False, "automatic_submission": False,
              "scope": "Frozen single-frame representation readouts; not policy improvement or unique root-cause proof"}
    client = None
    try:
        protocol = json.loads((bundle / "protocol.json").read_text())
        result["protocol_sha256"] = sha(bundle / "protocol.json")
        for name, digest in protocol["reused_sources_sha256"].items():
            if sha(bundle / name) != digest:
                raise ValueError("Changed pinned helper " + name)
        if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "1.12.1":
            raise ValueError("Use the established official CUDA environment")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        base = json.loads((bundle / "base_protocol.json").read_text())
        source = root / "paper_reproduction/official_pusht_v1/source"
        for name, digest in dict(base["author_file_sha256"], **protocol["state_source_sha256"]).items():
            if sha(source / name) != digest:
                raise ValueError("Author source changed: " + name)
        sys.path.insert(0, str(source))
        data_report = verified_json(root / "runs/pusht-float-data-210690/report.json", base["data_report_sha256"])
        if not data_report["data_integrity_passed"]:
            raise ValueError("Original data integrity has not passed")
        import zarr
        group = zarr.open_group(data_report["dataset_path"], mode="r")
        ends, state = group["meta/episode_ends"][:], group["data/state"][:]
        for name, values in (("meta/episode_ends", ends), ("data/state", state)):
            if array_sha(values) != data_report["arrays"][name]["decoded_sha256"]:
                raise ValueError("Source array changed: " + name)
        image_digest = hashlib.sha256()
        for start in range(0, len(state), 512):
            image_digest.update(np.ascontiguousarray(group["data/img"][start:start + 512]).tobytes())
        if image_digest.hexdigest() != data_report["arrays"]["data/img"]["decoded_sha256"]:
            raise ValueError("Source image array changed since frozen cache construction")
        split = protocol["split"]
        if split["fit_episode_ids"] != data_report["author_recipe_counts"]["train_episode_ids"]:
            raise ValueError("Probe fit must use precisely original train90")
        ids, episodes, roles = choose_frames(ends, split["fit_episode_ids"], split["heldout_episode_ids"], split["frames_per_episode"])
        fit = roles == "fit"
        labels = labels_from_state(state[ids])
        images = np.stack([group["data/img"][int(i)] for i in ids])
        if images.shape != (len(ids), 96, 96, 3) or images.dtype != np.float32 or not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
            raise ValueError("Raw float image contract differs")
        save(out / "frame_manifest.json", {"frame_ids": ids.tolist(), "episode_ids": episodes.tolist(),
             "roles": roles.tolist(), "raw_images_sha256": array_sha(images), "states_sha256": array_sha(state[ids]),
             "label_order": ["agent_x/512", "agent_y/512", "block_x/512", "block_y/512", "sin(angle_rad)", "cos(angle_rad)"]})
        # First fit episode and first heldout episode only, fixed before outcomes, for alignment QA.
        audit_ids = np.r_[np.arange(12), np.flatnonzero(~fit)[:12]]
        np.savez(out / "alignment_samples.npz", raw_images=images[audit_ids], state=state[ids[audit_ids]],
                 frame_ids=ids[audit_ids], episode_ids=episodes[audit_ids], roles=roles[audit_ids])
        result["inputs"] = {"fit_episodes": len(split["fit_episode_ids"]), "heldout_episodes": len(split["heldout_episode_ids"]),
                            "fit_frames": int(fit.sum()), "heldout_frames": int((~fit).sum()),
                            "manifest_sha256": sha(out / "frame_manifest.json"), "same_frame_labels": True}
        image_hashes = [array_sha(image) for image in images]
        result["inputs"]["identical_selected_images_across_roles"] = len(
            set(h for h, flag in zip(image_hashes, fit) if flag) &
            set(h for h, flag in zip(image_hashes, fit) if not flag))
        if result["inputs"]["identical_selected_images_across_roles"]:
            raise ValueError("Exact selected image overlap across episode roles; review before scoring readouts")
        print("SPATIAL_DIAGNOSTIC_INPUT", result["inputs"], flush=True)
        cache_dir = root / "runs/pusht-project-cache-213439"
        cache_report = verified_json(cache_dir / "report.json", protocol["cache_report_sha256"])
        if not (cache_report["cache_complete"] and not cache_report["errors"] and cache_report["worker"]["all_frozen"]):
            raise ValueError("Original cache is not qualified")
        for name, info in cache_report["files"].items():
            if (cache_dir / name).stat().st_size != info["bytes"] or sha(cache_dir / name) != info["sha256"]:
                raise ValueError("Cache identity changed: " + name)
        cached_ids = np.load(cache_dir / "frame_ids.npy", allow_pickle=False)
        cached_tokens = np.load(cache_dir / "visual.npy", mmap_mode="r", allow_pickle=False)
        expected_cache_ids = np.concatenate([np.arange(0 if ep == 0 else ends[ep - 1], ends[ep]) for ep in sorted(split["fit_episode_ids"])])
        if (not np.array_equal(cached_ids, expected_cache_ids) or cached_tokens.shape != (11356, 257, 1024)
                or cached_tokens.dtype != np.float32 or np.intersect1d(ids[~fit], cached_ids).size):
            raise ValueError("Cache/split frame identity differs")
        project, _, before = endpoint_frontend(root, bundle, protocol["project_checkpoint"])
        from resource_probe import state_hash
        from live_client import CLIPClient
        client = CLIPClient(root, out)
        if not (np.array_equal(client.text, project.text_tokens.cpu().numpy())
                and np.array_equal(client.text_mask, project.text_mask.cpu().numpy())):
            raise ValueError("Fixed text differs from endpoint")
        features = {}
        for begin in range(0, len(ids), 8):
            stop = min(len(ids), begin + 8)
            if fit[begin:stop].all():
                where = np.searchsorted(cached_ids, ids[begin:stop])
                if not np.array_equal(cached_ids[where], ids[begin:stop]):
                    raise ValueError("Train tokens not aligned")
                tokens = np.asarray(cached_tokens[where]).copy()
            elif (~fit[begin:stop]).all():
                tokens = client.encode(images[begin:stop])
            else:
                raise ValueError("Batch crossed frozen split boundary")
            found = project_features(project, torch.from_numpy(tokens).cuda())
            for name, value in found.items():
                if name not in features:
                    features[name] = np.empty((len(ids), value.shape[1]), dtype=np.float32)
                features[name][begin:stop] = value
            if begin % 128 == 0:
                print("SPATIAL_DIAGNOSTIC_FEATURES", stop, "/", len(ids), flush=True)
        close = client.close()
        client = None
        if not (close["all_frozen"] and close["weights_unchanged"] and state_hash(project) == before
                and all(not p.requires_grad and p.grad is None for p in project.parameters())):
            raise ValueError("Frozen model changed")
        result["project_frozen"] = {"frontend_state_sha256": before, "unchanged": True, "clip_worker": close}
        del project, cached_tokens
        gc.collect()
        torch.cuda.empty_cache()
        native, normalizer, before = endpoint_frontend(root, bundle, protocol["native_checkpoint"], native=True)
        features["native_rgb64"] = np.empty((len(ids), 64), dtype=np.float32)
        for begin in range(0, len(ids), 8):
            raw = torch.from_numpy(np.moveaxis(images[begin:begin + 8], -1, 1) / 255).cuda()
            obs = {"image": raw, "agent_pos": raw.new_zeros((len(raw), 2))}
            with torch.no_grad():
                encoded = native(normalizer.normalize(obs))
                if encoded.shape != (len(raw), 66):
                    raise ValueError("Native observation columns differ")
                if begin == 0:
                    other = native(normalizer.normalize(dict(obs, agent_pos=torch.ones_like(obs["agent_pos"]) * 400)))
                    if not torch.equal(encoded[:, :64], other[:, :64]):
                        raise ValueError("Native RGB probe leaked supplied agent coordinates")
            features["native_rgb64"][begin:begin + len(raw)] = encoded[:, :64].cpu().numpy()
        if not (state_hash(native) == before and all(not p.requires_grad and p.grad is None for p in native.parameters())):
            raise ValueError("Native encoder changed")
        result["native_frozen"] = {"frontend_state_sha256": before, "unchanged": True,
                                    "agent_position_not_appended_to_rgb_probe": True}
        del native, normalizer
        gc.collect()
        torch.cuda.empty_cache()
        # Calibrators are separate readout families, never mixed into vision features.
        features["raw_pixels12"] = (images.reshape(len(ids), 12, 8, 12, 8, 3).mean((2, 4)) / 255).reshape(len(ids), -1)
        features["agent_state_only"] = state[ids, :2].copy()
        if sorted(features) != sorted(protocol["readout_stages"]):
            raise ValueError("Unexpected readout stages")
        records, prediction_arrays = {}, {"labels": labels, "source_state": state[ids], "frame_ids": ids,
                                           "episode_ids": episodes, "fit_mask": fit}
        for name in protocol["readout_stages"]:
            p, shuffled, model = fit_readout(features[name], labels, fit, protocol["ridge"], protocol["shuffle_seed"])
            np.savez(out / ("readout_" + name + ".npz"), **model)
            prediction_arrays[name] = p
            prediction_arrays[name + "_shuffled"] = shuffled
            records[name] = {"dimensions": int(features[name].shape[1]),
                "fit": summarize(p, labels, episodes, fit), "heldout": summarize(p, labels, episodes, ~fit),
                "shuffled_heldout": summarize(shuffled, labels, episodes, ~fit)}
            print("SPATIAL_DIAGNOSTIC_READOUT", name, records[name]["heldout"]["episode_equal"], flush=True)
        constant = np.repeat(labels[fit].mean(0)[None], len(labels), axis=0)
        records["train_mean"] = {"fit": summarize(constant, labels, episodes, fit),
                                 "heldout": summarize(constant, labels, episodes, ~fit)}
        prediction_arrays["train_mean"] = constant
        np.savez(out / "predictions.npz", **prediction_arrays)
        comparisons = {a + "__to__" + b: paired_interval(records[a]["heldout"], records[b]["heldout"])
                       for a, b in protocol["paired_comparisons"]}
        result.update(diagnostic_complete=True, readouts=records, comparisons=comparisons,
                      policy_or_clip_weights_updated=False, diagnostic_heads_fitted=True,
                      diagnostic_readout_hyperparameter_search=False,
                      metric_units="512x512 simulation pixels and circular degrees; not cm or reward",
                      limitations=protocol["interpretation_limits"])
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if client is not None:
            client.abort()
        result["elapsed_seconds"] = time.monotonic() - started
        save(out / "report.json", result)
        print("SPATIAL_DIAGNOSTIC_REPORT", out / "report.json", flush=True)
    print("FROZEN SPATIAL DIAGNOSTIC COMPLETE; no policy improvement verdict", flush=True)


if __name__ == "__main__":
    main()
