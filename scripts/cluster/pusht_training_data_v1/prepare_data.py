"""Prepare the existing author's small Push-T dataset, without models or installs."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile

URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
MAX_DOWNLOAD = 256 * 1024 ** 2
MAX_EXTRACTED = 4 * 1024 ** 3
PACKAGES = ("torch", "torchvision", "numpy", "transformers", "huggingface-hub",
            "safetensors", "Pillow", "zarr", "numcodecs", "diffusers", "dill", "omegaconf")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def active_environment(root):
    interpreter = root / "envs/multitask-preflight-v1/bin/python"
    code = """import importlib.metadata as m, json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
print(json.dumps({'python': sys.version, 'executable': sys.executable, 'packages': versions}))
"""
    result = subprocess.run([str(interpreter), "-I", "-B", "-c", code, json.dumps(PACKAGES)],
                            text=True, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout)


def download(destination, out):
    if destination.exists():
        return {"downloaded_this_run": False, "bytes": destination.stat().st_size}
    started = time.monotonic()
    errors = []
    for attempt in range(1, 4):
        part = out / ("download-attempt-%d.zip.part" % attempt)
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "PushT-data-preparation/1"})
            total = 0
            with urllib.request.urlopen(request, timeout=30) as response, part.open("xb") as stream:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_DOWNLOAD:
                    raise ValueError("Download exceeds the declared 256 MiB bound")
                announced = 0
                while True:
                    if time.monotonic() - started > 600:
                        raise TimeoutError("Download exceeded 10 minutes; preserve partial files and stop")
                    block = response.read(256 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_DOWNLOAD:
                        raise ValueError("Download exceeds 256 MiB bound")
                    stream.write(block)
                    if total // (4 * 1024 ** 2) > announced:
                        announced = total // (4 * 1024 ** 2)
                        print("PUSHT_DOWNLOAD", round(total / 1024 ** 2, 1), "MiB", flush=True)
                if declared and total != int(declared):
                    raise ValueError("Downloaded length differs from HTTP Content-Length")
            # Exclusive publish: no overwrite/hardlink permission requirement.
            # An interrupted copy is preserved; CRC checking will reject it.
            with part.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            part.unlink()
            return {"downloaded_this_run": True, "bytes": total, "attempt": attempt,
                    "seconds": time.monotonic() - started, "url": URL}
        except Exception as exc:
            errors.append(type(exc).__name__ + ": " + str(exc))
            print("PUSHT_DOWNLOAD_ATTEMPT_FAILED", attempt, errors[-1], flush=True)
            if destination.exists():
                raise RuntimeError("Archive appeared during publication; preserve and inspect it") from exc
            if isinstance(exc, ValueError) or time.monotonic() - started > 600:
                break
    raise RuntimeError("Download failed (no TLS verification bypass): " + json.dumps(errors))


def archive_manifest(archive):
    names = set()
    total = 0
    with zipfile.ZipFile(archive) as stream:
        if len(stream.infolist()) > 200000:
            raise ValueError("Unexpectedly many archive members")
        for member in stream.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (not member.filename or "\\" in member.filename or path.is_absolute()
                    or ".." in path.parts or ":" in member.filename or stat.S_ISLNK(mode)):
                raise ValueError("Unsafe archive member: " + member.filename)
            normalized = str(path)
            if normalized in names:
                raise ValueError("Duplicate archive member: " + normalized)
            names.add(normalized)
            total += member.file_size
            if total > MAX_EXTRACTED:
                raise ValueError("Archive expansion exceeds 4 GiB bound")
        bad = stream.testzip()
        if bad is not None:
            raise ValueError("Archive CRC failed: " + bad)
    suffix = "/data/img/.zarray"
    roots = [name[:-len(suffix)] for name in names if name.endswith(suffix)]
    roots = [name for name in roots if all(name + "/" + key + "/.zarray" in names
                                         for key in ("data/action", "data/state", "meta/episode_ends"))]
    if len(roots) != 1:
        raise ValueError("Expected exactly one complete Push-T Zarr metadata root: " + repr(roots))
    return {"members": len(names), "extracted_file_bytes": total, "zarr_relative_path": roots[0],
            "zip_crc_passed": True}


def check_array(array, key, np):
    separator = getattr(array, "_dimension_separator", ".")
    grid = [range((size + chunk - 1) // chunk) for size, chunk in zip(array.shape, array.chunks)]
    expected = math.prod(len(axis) for axis in grid)
    if expected > 500000:
        raise ValueError("Unexpected chunk count: " + key)
    for coordinate in itertools.product(*grid):
        chunk_key = array.path + "/" + separator.join(str(x) for x in coordinate)
        if chunk_key not in array.store:
            raise ValueError("Missing stored chunk (Zarr would silently fill): " + chunk_key)
    h = hashlib.sha256()
    for start in range(0, array.shape[0], 256):
        data = array[start:start + 256]
        if data.dtype.kind in "fc" and not np.isfinite(data).all():
            raise ValueError("Nonfinite values in " + key)
        h.update(np.ascontiguousarray(data).tobytes())
    return {"shape": list(array.shape), "chunks": list(array.chunks), "dtype": str(array.dtype),
            "stored_chunks_checked": expected, "decoded_sha256": h.hexdigest()}


def validate_zarr(path, contract):
    import numpy as np
    import zarr
    from reference_helpers import author_window_counts
    group = zarr.open_group(str(path), mode="r")
    ends_array = group["meta/episode_ends"]
    if ends_array.ndim != 1 or not 0 < ends_array.size <= 1000000 or ends_array.dtype.kind not in "iu":
        raise ValueError("Unexpected episode boundary metadata")
    ends = ends_array[:]
    if ends[0] <= 0 or not np.all(ends[1:] > ends[:-1]):
        raise ValueError("Episode boundaries must increase")
    for key in ("data/img", "data/state", "data/action"):
        if group[key].shape[0] != int(ends[-1]):
            raise ValueError("Frame count differs from episode boundaries: " + key)
    if group["data/img"].shape[1:] != (96, 96, 3) or group["data/img"].dtype != np.dtype("uint8"):
        raise ValueError("Expected uint8 [N,96,96,3] images")
    if group["data/action"].shape[1:] != (2,) or len(group["data/state"].shape) != 2 or group["data/state"].shape[1] < 2:
        raise ValueError("Unexpected action/state shape")
    arrays = {}
    for key in ("meta/episode_ends", "data/action", "data/state", "data/img"):
        arrays[key] = check_array(group[key], key, np)
        print("PUSHT_DATA_ARRAY_VERIFIED", key, arrays[key]["shape"], flush=True)
    return {"path": str(path), "arrays": arrays, "all_expected_chunks_present_and_decoded": True,
            "author_recipe_counts": author_window_counts(ends, contract["author_dataset_config"], np),
            "scope": "Integrity read only; no normalization statistics fitted or model data sampled."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--existing-zarr", type=Path, help="Use this existing Zarr directory without downloading")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    if root / "runs" not in out.parents:
        parser.error("Output must be a new directory beneath root/runs")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    report = {"stage": "cpu_data_preparation", "training_updates": 0, "gpu_used": False,
              "environments_modified": False, "checkpoint_loaded": False, "errors": []}
    started = time.monotonic()
    try:
        contract = json.loads((Path(__file__).parent / "data_contract.json").read_text())
        report["active_environment"] = active_environment(root)
        write_json(out / "active_environment.json", report["active_environment"])
        print("ACTIVE_PROJECT_ENVIRONMENT", json.dumps(report["active_environment"]), flush=True)
        # Reuse a known location if it has appeared since 210557. Never overwrite it.
        candidates = ([args.existing_zarr.resolve()] if args.existing_zarr else [])
        candidates += [Path(p) for p in contract["previously_checked_zarr_directories"]]
        existing = next((p for p in candidates if p.is_dir()), None)
        if args.existing_zarr and existing != args.existing_zarr.resolve():
            raise FileNotFoundError("Explicit existing Zarr directory is absent")
        if existing is not None:
            report["dataset"] = validate_zarr(existing, contract)
            report["download"] = {"downloaded_this_run": False, "existing_directory_used": True}
        else:
            assets = root / "paper_reproduction/pusht_module_data_v1"
            assets.mkdir(parents=True, exist_ok=True)
            archive = assets / "pusht.zip"
            report["download"] = download(archive, out)
            info = archive_manifest(archive)
            info["sha256"] = digest(archive)
            info["sha_scope"] = "Locally computed identity, not a publisher-provided checksum"
            report["archive"] = info
            extracted = assets / "extracted"
            marker = assets / "prepared.json"
            if extracted.exists():
                if not marker.is_file() or json.loads(marker.read_text())["archive_sha256"] != info["sha256"]:
                    raise ValueError("Existing extraction has no matching completion marker; preserve and stop")
                dataset = extracted / info["zarr_relative_path"]
                report["dataset"] = validate_zarr(dataset, contract)
            else:
                if shutil.disk_usage(str(assets)).free < info["extracted_file_bytes"] + 512 * 1024 ** 2:
                    raise ValueError("Not enough free space for extraction and 512 MiB headroom")
                stage = out / "extraction_staging"
                stage.mkdir()
                with zipfile.ZipFile(archive) as stream:
                    stream.extractall(str(stage))  # Paths and symlinks already checked above.
                report["dataset"] = validate_zarr(stage / info["zarr_relative_path"], contract)
                stage.rename(extracted)
                report["dataset"]["path"] = str(extracted / info["zarr_relative_path"])
                write_json(marker, {"archive_sha256": info["sha256"], "dataset": report["dataset"]["path"]})
        report["runtime_compatibility_verified"] = False
        report["formal_training_ready"] = False
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    report["elapsed_seconds"] = time.monotonic() - started
    report["data_preparation_passed"] = not report["errors"]
    write_json(out / "report.json", report)
    print("PUSHT_DATA_REPORT", out / "report.json", flush=True)
    print("PUSHT DATA PREPARATION", "PASSED" if report["data_preparation_passed"] else "FAILED", flush=True)
    if report["errors"]:
        print("ERRORS", json.dumps(report["errors"]), flush=True)
    print("No model execution, training, package installation, or generalization verdict.", flush=True)
    return 0 if report["data_preparation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
