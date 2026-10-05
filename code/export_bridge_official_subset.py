"""Export fixed RLDS windows for the author's single-step LCBC loader.

Output records group sparse observation/next-observation pairs by source episode.
They are NOT contiguous trajectories and must not be used for multi-step/RL jobs.
The original training/evaluation code is not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
import tensorflow as tf


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(stream)


def hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def load_author_reader(root):
    """Import unmodified author data code without requiring its unused augmenter.

    This audit disables augmentation. The import guard raises if augmentation is
    ever called; it does not claim to validate the author's augmentation module
    or complete JAX dependency environment.
    """
    root = Path(root).resolve()
    provenance = read_json(root / "provenance.json")
    for name, details in provenance["files"].items():
        if sha256(root / name) != details["sha256"]:
            raise ValueError(f"Reference changed: {name}")
    for name in ["jaxrl_m", "jaxrl_m.data"]:
        if name in sys.modules:
            raise RuntimeError("Run this audit in a fresh Python process")
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = package
    guard = types.ModuleType("jaxrl_m.data.tf_augmentations")
    def unavailable(*args, **kwargs):
        raise RuntimeError("Augmentation is outside this read-only audit")
    guard.augment = unavailable
    sys.modules[guard.__name__] = guard
    return importlib.import_module("jaxrl_m.data.bridge_dataset"), provenance


def read_record(path, record_index):
    """Seek over preceding payloads, reading only the selected record's data."""
    with Path(path).open("rb") as stream:
        for index in range(record_index + 1):
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"Missing record {record_index}: {path}")
            length = struct.unpack("<Q", header[:8])[0]
            if index != record_index:
                stream.seek(length + 4, io.SEEK_CUR)
                continue
            payload = stream.read(length)
            if len(payload) != length or len(stream.read(4)) != 4:
                raise ValueError("Truncated source record")
    example = tf.train.Example()
    example.ParseFromString(payload)
    return example.features.feature


def image_rgb(encoded):
    with Image.open(io.BytesIO(encoded)) as image:
        return np.array(image.convert("RGB"))


def tensor_feature(array):
    value = tf.io.serialize_tensor(tf.convert_to_tensor(array)).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--window-audit", type=Path, required=True)
    parser.add_argument("--saved-targets", type=Path, required=True)
    parser.add_argument("--author-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output must be a new directory; previous exports are never overwritten")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    author, provenance = load_author_reader(args.author_reference)
    manifest = read_json(args.split_manifest)
    audit = read_json(args.window_audit)
    if manifest["chunk_size"] != 1:
        raise ValueError("Only the existing single-step manifest is supported")
    import torch
    from dataset import relative_pose_action, euler_xyz_to_quaternion
    saved = torch.load(args.saved_targets, map_location="cpu", weights_only=False)
    if saved["dataset_identity"] != manifest["dataset_identity"]:
        raise ValueError("Saved targets and split manifest identities differ")
    targets = {int(i): saved["target"][j, 0].numpy() for j, i in enumerate(saved["indices"])}
    rows = {int(r["index"]): r for r in audit["rows"]}
    if len(rows) != len(audit["rows"]):
        raise ValueError("Duplicate source window indices")
    partitions = {name: list(map(int, manifest["split_indices"][name])) for name in ["train", "validation"]}
    wanted = set(partitions["train"]) | set(partitions["validation"])
    if (set(partitions["train"]) & set(partitions["validation"]) or wanted & set(manifest["split_indices"]["test"])):
        raise ValueError("Partition overlap")
    if set(targets) != wanted or set(rows) != wanted:
        raise ValueError("Training/validation target coverage differs")
    selection = {(r["shard"], int(r["record_index"])): r for r in manifest["bridge_episode_selection"]}
    used_episodes = set()
    args.output_dir.mkdir(parents=True)
    expected = {}; sample_order = {}; episode_metadata = []
    for partition, indices in partitions.items():
        groups = {}
        for index in indices:
            row = rows[index]
            key = (row["shard"], int(row["record_index"]))
            group = selection[key]
            if group["partition"] != partition:
                raise ValueError("Source episode partition differs")
            groups.setdefault(key, []).append(row)
        path = args.output_dir / "data" / "sweep_into_pile" / ("train" if partition == "train" else "val") / "out.tfrecord"
        path.parent.mkdir(parents=True)
        expected[partition] = []; sample_order[partition] = []
        with tf.io.TFRecordWriter(str(path)) as writer:
            for key, group_rows in groups.items():
                source = selection[key]
                features = read_record(args.dataset_dir / key[0], key[1])
                language = list(features["steps/language_instruction"].bytes_list.value)
                n = len(language)
                state = np.array(features["steps/observation/state"].float_list.value, np.float32).reshape(n, 7)
                commands = np.array(features["steps/action"].float_list.value, np.float32).reshape(n, 7)
                images = list(features["steps/observation/image_0"].bytes_list.value)
                first = np.array(features["steps/is_first"].int64_list.value, dtype=bool)
                last = np.array(features["steps/is_last"].int64_list.value, dtype=bool)
                terminal = np.array(features["steps/is_terminal"].int64_list.value, dtype=bool)
                origin = features["episode_metadata/file_path"].bytes_list.value[0].decode()
                eid = int(features["episode_metadata/episode_id"].int64_list.value[0])
                if (n != source["steps"] or len(images) != n or origin != source["origin_file_path"] or eid != source["episode_id"]):
                    raise ValueError("Source episode identity differs")
                if first.shape != (n,) or last.shape != (n,) or terminal.shape != (n,) or np.flatnonzero(first).tolist() != [0] or np.flatnonzero(last).tolist() != [n-1]:
                    raise ValueError("Unexpected RLDS boundary flags")
                if {" ".join(x.decode().lower().split()) for x in language} != {source["instruction"]}:
                    raise ValueError("Source language differs")
                # Author scan runs on ALL valid actions before sparse selection.
                binary = author._binarize_gripper_actions(tf.constant(commands[:-1, 6])).numpy()
                if not np.isin(binary, [0., 1.]).all():
                    raise ValueError("Unresolved gripper tail must not be thresholded silently")
                starts = np.array([int(r["start_index"]) for r in group_rows])
                if np.any(starts < 0) or np.any(starts + 1 >= n):
                    raise ValueError("Selected target has no next observation")
                decoded = {int(t): image_rgb(images[int(t)]) for t in np.unique(np.r_[starts, starts+1])}
                obs = np.stack([decoded[int(t)] for t in starts])
                nxt = np.stack([decoded[int(t+1)] for t in starts])
                actions = commands[starts].copy(); actions[:, 6] = binary[starts]
                truncates = np.zeros(len(starts), bool); truncates[-1] = True
                arrays = {
                    "observations/images0": obs, "observations/state": state[starts],
                    "next_observations/images0": nxt, "next_observations/state": state[starts+1],
                    "actions": actions, "terminals": terminal[starts+1], "truncates": truncates,
                    "language": np.array([source["instruction"].encode()]),
                }
                example = tf.train.Example(features=tf.train.Features(feature={k:tensor_feature(v) for k,v in arrays.items()}))
                writer.write(example.SerializeToString())
                for j, row in enumerate(group_rows):
                    index = int(row["index"]); t = int(starts[j])
                    expected[partition].append({"index":index,"state":state[t].copy(),"next":state[t+1].copy(),
                        "image_hash":hashlib.sha256(obs[j].tobytes()).hexdigest(),
                        "next_image_hash":hashlib.sha256(nxt[j].tobytes()).hexdigest(),
                        "language":language[t],"terminal":bool(terminal[t+1])})
                    sample_order[partition].append(index)
                used_episodes.add(key)
                episode_metadata.append({**source,"starts":starts.tolist(),"indices":[int(r["index"]) for r in group_rows]})
        print(f"Exported {partition}: {len(groups)} episodes, {len(indices)} windows", flush=True)
    metrics = {}; training_actions = []
    for partition in partitions:
        path = args.output_dir / "data" / "sweep_into_pile" / ("train" if partition == "train" else "val") / "out.tfrecord"
        # batch=1 preserves all pairs through the unmodified author's batching.
        # Rebatch only for verification, explicitly retaining the final batch.
        dataset = author.BridgeDataset([str(path)], seed=42, batch_size=1, train=False,
            augment=False, load_language=True, skip_unlabeled=True, relabel_actions=True,
            action_proprio_metadata=None, goal_relabeling_strategy="uniform",
            goal_relabeling_kwargs={"reached_proportion":0.})
        batches = dataset.tf_dataset.unbatch().batch(8, drop_remainder=False).as_numpy_iterator()
        count = 0; max_xyz = 0.; max_q = 0.; action_rows = []; batch_sizes = []
        for batch in batches:
            batch_sizes.append(len(batch["actions"]))
            for j in range(len(batch["actions"])):
                exp = expected[partition][count]
                state = batch["observations"]["proprio"][j]
                nxt = batch["next_observations"]["proprio"][j]
                action = batch["actions"][j]
                np.testing.assert_array_equal(state, exp["state"])
                np.testing.assert_array_equal(nxt, exp["next"])
                np.testing.assert_array_equal(action[:6], exp["next"][:6]-exp["state"][:6])
                if hashlib.sha256(batch["observations"]["image"][j].tobytes()).hexdigest() != exp["image_hash"] or hashlib.sha256(batch["next_observations"]["image"][j].tobytes()).hexdigest() != exp["next_image_hash"]:
                    raise ValueError("Decoded RGB image differs")
                if batch["goals"]["language"][j] != exp["language"] or bool(batch["terminals"][j]) != exp["terminal"]:
                    raise ValueError("Language or terminal flag differs")
                mapped = relative_pose_action(state[:3],euler_xyz_to_quaternion(state[3:6]),
                    state[:3]+action[:3],euler_xyz_to_quaternion(state[3:6]+action[3:6]),2*float(action[6])-1)
                target = targets[exp["index"]]
                max_xyz = max(max_xyz,float(np.max(np.abs(mapped[:3]-target[:3]))))
                max_q = max(max_q,float(min(np.max(np.abs(mapped[3:7]-target[3:7])),np.max(np.abs(mapped[3:7]+target[3:7])))))
                np.testing.assert_allclose(mapped, target, rtol=0, atol=1e-5)
                action_rows.append(action); count += 1
        if count != len(partitions[partition]) or set(sample_order[partition]) != set(partitions[partition]):
            raise ValueError("Missing or additional windows")
        metrics[partition] = {"windows":count,"batch_sizes":batch_sizes,"max_normalized_xyz_difference":max_xyz,
            "max_quaternion_difference_up_to_sign":max_q,"gripper_mismatches":0,"rgb_state_language_checks_passed":count}
        if partition == "train": training_actions = np.stack(action_rows)
    mean = training_actions.mean(axis=0); std = training_actions.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError("Training action dimension has effectively zero variance")
    for key in used_episodes:
        if selection[key]["partition"] not in partitions: raise ValueError("Test target read")
    files = {p.relative_to(args.output_dir).as_posix():{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in args.output_dir.rglob("*.tfrecord")}
    export = {"purpose":"fixed_window_single_step_lcbc_subset", "dataset_identity":manifest["dataset_identity"],
        "source_manifest_sha256":sha256(args.split_manifest),"official_commit":provenance["commit"],
        "sample_order":sample_order,"episodes":episode_metadata,"files":files,"test_targets_used":False,
        "constraints":["Sparse pairs are not contiguous trajectories; single-step LCBC only.","Gripper binarized on complete valid sequence before selecting starts.","Do not set act_pred_horizon or obs_horizon on this export."]}
    stats = {"action":{"mean":mean.tolist(),"std":std.tolist()},
        "proprio":{"mean":[0.]*7,"std":[1.]*7},"computed_from":"316 selected training windows only"}
    report = {"passed":True,"official_commit":provenance["commit"],"tensorflow_version":tf.__version__,"episodes":len(used_episodes),
        "partitions":metrics,"test_targets_used":False,"normalization_during_read_check":None,
        "limitations":["Author loader and gripper function executed, without augmentation or JAX model.",
          "Unused augmentation import replaced by fail-fast guard; full author dependency environment not verified.",
          "Sparse transition groups are only suitable for single-step language-conditioned BC.",
          "Saved target conversion shares project's pose encoder; not independent physical calibration."]}
    for name, value in [("export_manifest.json",export),("train_action_stats.json",stats),("read_check.json",report)]:
        (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
