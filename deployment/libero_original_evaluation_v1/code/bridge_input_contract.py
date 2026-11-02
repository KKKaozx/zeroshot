"""Exact raw identity plus bounded absolute float comparison across environments."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import UnifiedRobotDataset

ABS_TOLERANCE = {"images": 1e-6, "targets": 1e-6, "currents": 0.0}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def raw_identity(example, index):
    fields = example.features.feature
    encoded = fields["steps/observation/image_0"].bytes_list.value[index]
    rgb = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
    return dict(encoded_image_sha256=digest_bytes(encoded), decoded_rgb_sha256=digest_bytes(rgb.tobytes()),
        rgb_shape=list(rgb.shape), state_sha256=digest_bytes(np.asarray(fields["steps/observation/state"].float_list.value, dtype="<f4").tobytes()),
        action_sha256=digest_bytes(np.asarray(fields["steps/action"].float_list.value, dtype="<f4").tobytes()),
        instruction_sha256=digest_bytes(fields["steps/language_instruction"].bytes_list.value[index]))


def require_raw_identity(actual, expected, label):
    if actual != expected:
        fields = [k for k in expected if actual.get(k) != expected[k]]
        raise ValueError(f"Raw input identity changed at {label}: {fields}")


def require_numeric_match(actual, expected, name, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Shape/dtype changed for {name} at {label}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"Nonfinite {name} at {label}")
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    maximum = float(delta.max())
    limit = ABS_TOLERANCE[name]
    if maximum > limit:
        raise ValueError(f"Numeric {name} mismatch at {label}: max_abs={maximum:.9g} > atol={limit}; rtol=0")
    if name == "targets" and not np.array_equal(actual[...,7], expected[...,7]):
        raise ValueError(f"Gripper label changed at {label}")
    return dict(exact=bool(np.array_equal(actual, expected)), max_abs=maximum,
        mean_abs=float(delta.mean()), atol=limit, rtol=0)


def load_verified_items(rows, data_root, manifest, reference_directory, checks, comparisons):
    """Lists supplied by caller retain diagnostic evidence even if a later row fails."""
    directory = Path(reference_directory)
    for name, expected in manifest["input_reference_files_sha256"].items():
        if digest_bytes((directory/name).read_bytes()) != expected:
            raise ValueError("Reference integrity mismatch: "+name)
    metadata = json.loads((directory/"metadata.json").read_text())
    assert metadata["manifest_sha256"] == manifest["input_reference_manifest_sha256"]
    assert metadata["dataset_sha256"] == manifest["source_sha256"]["dataset.py"]
    assert len(rows) == len(metadata["raw_inputs"]) == 72
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size = 16
    reader.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
    reader.bridge_current_gripper = "continuous"
    items = []
    with np.load(directory/"arrays.npz") as reference:
        for i, row in enumerate(rows):
            label = f"row {i} {row['shard']}::{row['record_index']} t{row['start_index']}"
            path = str(Path(data_root)/row["shard"])
            example = UnifiedRobotDataset._load_tfrecord_example(path, row["record_index"])
            raw = raw_identity(example, row["start_index"])
            require_raw_identity(raw, metadata["raw_inputs"][i], label)
            reader._load_tfrecord_example = lambda *_: example
            item = reader._get_tfrecord_bridge_state_action(dict(file_path=path,
                record_index=row["record_index"], start_index=row["start_index"]))
            assert " ".join(item[0].lower().split()) == " ".join(row["task"].lower().split())
            arrays = dict(images=item[1].numpy(), targets=item[3].numpy(), currents=item[2].numpy())
            hashes = {k:digest_bytes(v.tobytes()) for k,v in arrays.items()}
            for name in arrays:
                assert digest_bytes(reference[name][i].tobytes()) == metadata["processed_sha256"][name][i]
            if i < 32:
                assert metadata["processed_sha256"]["images"][i] == manifest["retained_train_checks"][i]["image_sha256"]
                assert metadata["processed_sha256"]["targets"][i] == manifest["retained_train_checks"][i]["target_sha256"]
            checks.append(dict(image_sha256=hashes["images"], target_sha256=hashes["targets"],
                current_sha256=hashes["currents"], raw_identity=raw))
            differences = {k:require_numeric_match(v, reference[k][i], k, label) for k,v in arrays.items()}
            comparisons.append(dict(index=i, raw_identity_exact=True, differences=differences))
            items.append(item)
    return items
