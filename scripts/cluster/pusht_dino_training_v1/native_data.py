"""Unchanged native training-data reader extracted without a training main."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


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
