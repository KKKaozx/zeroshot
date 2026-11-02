"""CPU-only raw-input and numeric comparison for failed batch preflight.

Does not train, load CLIP, install packages, or bypass the old assertion.
Export local references, then measure the same 72 windows on the cluster.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time

os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TF="0", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import PIL
from PIL import Image
import torch
from dataset import UnifiedRobotDataset


def sha(value):
    return hashlib.sha256(value).hexdigest()


def versions():
    return dict(python=platform.python_version(), system=platform.system(),
        torch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__)


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    return dict(exact=bool(np.array_equal(actual, expected)), max_abs=float(diff.max()),
        mean_abs=float(diff.mean()), unequal_elements=int(np.count_nonzero(diff)),
        above_1e_5=int(np.count_nonzero(diff > 1e-5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--export-reference", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    dataset_sha = sha(Path(__file__).with_name("dataset.py").read_bytes())
    assert dataset_sha == manifest["source_sha256"]["dataset.py"], "Dataset source changed"
    rows = manifest["train"] + manifest["development"]
    assert len(rows) == 72
    reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size = 16
    reader.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
    reader.bridge_current_gripper = "continuous"
    inputs, images, targets, currents = [], [], [], []
    for row in rows:
        path = str(args.data_root/row["shard"])
        example = UnifiedRobotDataset._load_tfrecord_example(path, row["record_index"])
        fields = example.features.feature
        index = row["start_index"]
        encoded = fields["steps/observation/image_0"].bytes_list.value[index]
        rgb = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
        state = np.asarray(fields["steps/observation/state"].float_list.value, dtype="<f4")
        action = np.asarray(fields["steps/action"].float_list.value, dtype="<f4")
        language = fields["steps/language_instruction"].bytes_list.value[index]
        inputs.append(dict(encoded_image_sha256=sha(encoded), decoded_rgb_sha256=sha(rgb.tobytes()),
            rgb_shape=list(rgb.shape), state_sha256=sha(state.tobytes()), action_sha256=sha(action.tobytes()),
            instruction_sha256=sha(language)))
        reader._load_tfrecord_example = lambda *_: example
        item = reader._get_tfrecord_bridge_state_action(dict(file_path=path, record_index=row["record_index"], start_index=index))
        assert " ".join(item[0].lower().split()) == " ".join(row["task"].lower().split())
        images.append(item[1].numpy())
        targets.append(item[3].numpy())
        currents.append(item[2].numpy())
    arrays = dict(images=np.stack(images), targets=np.stack(targets), currents=np.stack(currents))
    assert all(np.isfinite(v).all() for v in arrays.values())
    metadata = dict(manifest_sha256=sha(args.manifest.read_bytes()), dataset_sha256=dataset_sha,
        versions=versions(), raw_inputs=inputs, processed_sha256={k:[sha(v.tobytes()) for v in values] for k,values in arrays.items()})
    if args.export_reference:
        args.reference.mkdir(parents=True, exist_ok=False)
        for i in range(32):
            old = manifest["retained_train_checks"][i]
            assert metadata["processed_sha256"]["images"][i] == old["image_sha256"]
            assert metadata["processed_sha256"]["targets"][i] == old["target_sha256"]
        np.savez_compressed(args.reference/"arrays.npz", **arrays)
        (args.reference/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print("LOCAL_REFERENCE_EXPORTED", len(rows), "windows", versions(), flush=True)
        return
    assert args.output
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="cpu_input_comparison_only", trained=False, clip_loaded=False,
        error=None, comparison_completed=False, cluster_versions=versions(), windows=[], original_preflight_job=198833)
    try:
        reference = json.loads((args.reference/"metadata.json").read_text())
        assert reference["manifest_sha256"] == metadata["manifest_sha256"]
        assert reference["dataset_sha256"] == dataset_sha
        report["local_versions"] = reference["versions"]
        with np.load(args.reference/"arrays.npz") as expected:
            for i, row in enumerate(rows):
                raw_equal = {k:inputs[i][k] == reference["raw_inputs"][i][k] for k in inputs[i]}
                delta = {k:compare(values[i], expected[k][i]) for k,values in arrays.items()}
                old_hash_equal = metadata["processed_sha256"]["images"][i] == reference["processed_sha256"]["images"][i]
                report["windows"].append(dict(index=i, selection=row, raw_equal=raw_equal,
                    normalized_image_hash_equal=old_hash_equal, differences=delta,
                    max_target_xyz_component_difference_cm=float(np.abs(arrays["targets"][i][:,:3].astype(np.float64)-expected["targets"][i][:,:3]).max()*10)))
        report["summary"] = dict(windows=len(rows), all_raw_and_decoded_inputs_equal=all(all(r["raw_equal"].values()) for r in report["windows"]),
            normalized_images_exact_count=sum(r["normalized_image_hash_equal"] for r in report["windows"]),
            max_normalized_image_difference=max(r["differences"]["images"]["max_abs"] for r in report["windows"]),
            max_target_component_difference=max(r["differences"]["targets"]["max_abs"] for r in report["windows"]),
            max_current_gripper_difference=max(r["differences"]["currents"]["max_abs"] for r in report["windows"]))
        report["first_image_hash_mismatch"] = next((r for r in report["windows"] if not r["normalized_image_hash_equal"]), None)
        report["comparison_completed"] = True
        print("INPUT_COMPARISON", json.dumps(report["summary"]), flush=True)
        if report["first_image_hash_mismatch"]:
            r = report["first_image_hash_mismatch"]
            print("FIRST_IMAGE_HASH_MISMATCH", r["index"], r["selection"]["shard"], r["selection"]["record_index"],
                "raw_equal", r["raw_equal"], "image_delta", r["differences"]["images"], flush=True)
        print("VERSIONS", json.dumps(dict(local=report["local_versions"], cluster=report["cluster_versions"])), flush=True)
        print("Diagnosis requires raw identity and numeric difference review; no training assertion bypassed.", flush=True)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("INPUT_AUDIT_REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
