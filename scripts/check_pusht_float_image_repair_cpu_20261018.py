"""Synthetic float32 range/chunk checks, no real dataset, codec, network or model."""
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_float_image_repair_v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected rejection")


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    module = load("repair", BUNDLE / "check_data.py")
    helper = load("helper", BUNDLE / "reference_helpers.py")

    class Array:
        _dimension_separator = "."

        def __init__(self, key, data, chunks):
            self.path, self.data, self.chunks = key, data, chunks
            self.shape, self.dtype = data.shape, data.dtype
            grid = [range((s + c - 1) // c) for s, c in zip(self.shape, chunks)]
            self.store = {key + "/" + ".".join(map(str, p)): b"present"
                          for p in itertools.product(*grid)}

        def __getitem__(self, item):
            return self.data[item]

    image = np.linspace(0, 255, 20 * 96 * 96 * 3, dtype=np.float32).reshape(20, 96, 96, 3)
    original_hash = hashlib.sha256(image.tobytes()).hexdigest()
    group = {
        "data/img": Array("data/img", image, (7, 96, 96, 3)),
        "data/action": Array("data/action", np.arange(40, dtype=np.float32).reshape(20, 2), (7, 2)),
        "data/state": Array("data/state", np.arange(100, dtype=np.float32).reshape(20, 5), (7, 5)),
        "meta/episode_ends": Array("meta/episode_ends", np.array([10, 20], dtype=np.int64), (3,)),
    }
    schema = {key: {"shape": list(a.shape), "chunks": list(a.chunks), "dtype": a.dtype.str}
              for key, a in group.items()}
    contract = json.loads((BUNDLE / "repair_contract.json").read_text())
    report = {}
    module.verify_dataset(group, schema, np, helper.author_window_counts,
                          contract["author_dataset_config"], report)
    assert report["all_expected_chunks_present_and_decoded"]
    assert report["image_processing_contract"]["range_after_author_divide_255"] == [0, 1]
    assert image.dtype == np.float32 and hashlib.sha256(image.tobytes()).hexdigest() == original_hash
    assert report["arrays"]["data/img"]["decoded_sha256"] == original_hash
    reject(lambda: module.image_contract({"raw_min": 0., "raw_max": 1., "dtype": "float32"}))
    reject(lambda: module.image_contract({"raw_min": -1., "raw_max": 255., "dtype": "float32"}))
    reject(lambda: module.image_contract({"raw_min": 0., "raw_max": 256., "dtype": "float32"}))
    image.flat[0] = np.nan
    reject(lambda: module.check_array(group["data/img"], "data/img", np))
    image.flat[0] = 0
    del group["data/img"].store["data/img/2.0.0.0"]
    reject(lambda: module.check_array(group["data/img"], "data/img", np))
    result = {"python39_syntax": True, "float32_fractional_pixels_preserved": True,
              "decoded_hash_and_author_scale_verified": True, "five_range_finite_chunk_rejections": True,
              "real_zarr_codec_verified": False, "network_or_model_executed": False}
    out = ROOT / "reports/pusht_float_repair_cpu_checks_20261018.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
