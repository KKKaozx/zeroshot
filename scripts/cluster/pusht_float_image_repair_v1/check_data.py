"""Read the already extracted 210631 data. No download, extraction, cast, or models."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_array(array, key, np):
    separator = getattr(array, "_dimension_separator", ".")
    grid = [range((size + chunk - 1) // chunk) for size, chunk in zip(array.shape, array.chunks)]
    expected = math.prod(len(axis) for axis in grid)
    if expected > 500000:
        raise ValueError("Unexpected chunk count: " + key)
    for coordinate in itertools.product(*grid):
        chunk_key = array.path + "/" + separator.join(str(x) for x in coordinate)
        if chunk_key not in array.store:
            raise ValueError("Missing stored chunk: " + chunk_key)
    result = hashlib.sha256()
    minimum, maximum = None, None
    started = time.monotonic()
    for start in range(0, array.shape[0], 256):
        data = array[start:start + 256]
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite data: " + key)
        low, high = float(data.min()), float(data.max())
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
        result.update(np.ascontiguousarray(data).tobytes())
        if key == "data/img" and start % 4096 == 0:
            print("PUSHT_IMAGE_DECODE", start, "/", array.shape[0], "seconds", round(time.monotonic() - started, 1), flush=True)
    return {"shape": list(array.shape), "chunks": list(array.chunks), "dtype": str(array.dtype),
            "stored_chunks_checked": expected, "decoded_sha256": result.hexdigest(),
            "raw_min": minimum, "raw_max": maximum}


def image_contract(stats):
    low, high = stats["raw_min"], stats["raw_max"]
    if low < 0 or high > 255:
        raise ValueError("Image values outside author's raw [0,255] contract; no clipping/casting")
    if high <= 1:
        raise ValueError("Image values look already normalized/degenerate; do not silently divide by 255")
    return {"storage_dtype_preserved": stats["dtype"], "raw_range": [low, high],
            "author_operation": "np.moveaxis(image, -1, 1) / 255",
            "range_after_author_divide_255": [low / 255, high / 255],
            "uint8_conversion_performed": False, "normalization_statistics_fitted": False}


def verify_dataset(group, schema, np, counts, recipe, report):
    # Match the actual observed archive schema, rather than accepting arbitrary floats.
    for key, expected in schema.items():
        array = group[key]
        if (list(array.shape) != expected["shape"] or array.dtype != np.dtype(expected["dtype"])
                or list(array.chunks) != expected["chunks"]):
            raise ValueError("Schema differs from reported cached archive: " + key)
    ends = group["meta/episode_ends"][:]
    if ends[0] <= 0 or not np.all(ends[1:] > ends[:-1]):
        raise ValueError("Invalid episode boundaries")
    if any(group[key].shape[0] != int(ends[-1]) for key in ("data/action", "data/state", "data/img")):
        raise ValueError("Array lengths differ from episode boundaries")
    arrays = report.setdefault("arrays", {})
    for key in ("meta/episode_ends", "data/action", "data/state", "data/img"):
        arrays[key] = check_array(group[key], key, np)
        print("PUSHT_ARRAY_VERIFIED", key, arrays[key]["shape"], arrays[key]["dtype"], flush=True)
    # Preserve range evidence in report even if the subsequent semantic guard fails.
    report["image_processing_contract"] = image_contract(arrays["data/img"])
    report["author_recipe_counts"] = counts(ends, recipe, np)
    report["all_expected_chunks_present_and_decoded"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Output must be a new directory under root/runs")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    started = time.monotonic()
    report = {"stage": "cached_float32_image_integrity_repair", "origin_job": 210631,
              "downloaded": False, "extracted_again": False, "data_modified": False,
              "model_executed": False, "training_updates": 0, "gpu_used": False,
              "runtime_compatibility_verified": False, "formal_training_ready": False, "errors": []}
    try:
        import numpy as np
        import zarr
        from reference_helpers import author_window_counts
        contract = json.loads((Path(__file__).parent / "repair_contract.json").read_text())
        origin_path = root / "runs/pusht-data-prepare-210631/report.json"
        origin = json.loads(origin_path.read_text())
        report["origin_report_sha256"] = sha(origin_path)
        if origin["errors"] != ["ValueError: Expected uint8 [N,96,96,3] images"]:
            raise ValueError("Unexpected origin failure; do not apply this repair")
        archive = root / "paper_reproduction/pusht_module_data_v1/pusht.zip"
        report["archive_sha256"] = sha(archive)
        if not origin["archive"]["zip_crc_passed"] or report["archive_sha256"] != origin["archive"]["sha256"]:
            raise ValueError("Cached archive differs from 210631; stop without download")
        relative = PurePosixPath(origin["archive"]["zarr_relative_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Invalid recorded Zarr path")
        stage = (origin_path.parent / "extraction_staging").resolve()
        dataset = (stage / str(relative)).resolve()
        if stage not in dataset.parents or not dataset.is_dir():
            raise ValueError("Previously extracted dataset absent; no automatic extraction")
        report["dataset_path"] = str(dataset)
        report["preserve_origin_run_data"] = True
        report["active_environment_from_origin_210631"] = origin["active_environment"]
        # Tiny metadata reads only; full ZIP CRC already passed and SHA is unchanged.
        with zipfile.ZipFile(archive) as stream:
            for key, expected in contract["observed_schema"].items():
                meta = json.loads(stream.read(str(relative) + "/" + key + "/.zarray"))
                if any(meta[field] != expected[field] for field in ("shape", "chunks", "dtype")):
                    raise ValueError("ZIP metadata differs from user observation: " + key)
        group = zarr.open_group(str(dataset), mode="r")
        verify_dataset(group, contract["observed_schema"], np, author_window_counts,
                       contract["author_dataset_config"], report)
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    report["elapsed_seconds"] = time.monotonic() - started
    report["data_integrity_passed"] = not report["errors"]
    write_json(out / "report.json", report)
    print("PUSHT_FLOAT_DATA_REPORT", out / "report.json", flush=True)
    print("PUSHT CACHED DATA", "PASSED" if report["data_integrity_passed"] else "FAILED", flush=True)
    if report["errors"]:
        print("ERRORS", json.dumps(report["errors"]), flush=True)
    print("No download, extraction, cast, installation, training, or model evaluation.", flush=True)
    return 0 if report["data_integrity_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
