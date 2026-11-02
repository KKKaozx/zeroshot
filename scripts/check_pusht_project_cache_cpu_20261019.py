"""Cache indexing/publication tests with numeric stand-ins; no model execution."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_project_cache_v1"
sys.path.insert(0, str(BUNDLE))
from cache_tools import encode_frames, compare_reference, sha


class Client:
    def __init__(self, dtype=np.float32, fail=False):
        self.calls, self.dtype, self.fail = [], dtype, fail

    def encode(self, images):
        self.calls.append(len(images))
        if self.fail and len(self.calls) == 2:
            raise RuntimeError("Test injected encoder failure")
        values = images[:, 0, 0, 0, None, None]
        return np.broadcast_to(values, (len(images), 3, 4)).astype(self.dtype).copy()


def reject(fn, kind=ValueError):
    try:
        fn()
    except kind:
        return
    raise AssertionError("Invalid cache accepted")


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    images = np.zeros((35, 96, 96, 3), dtype=np.float32)
    images[:, 0, 0, 0] = np.arange(35)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        folder = Path(temp)
        writing = folder / "visual.writing.npy"
        writer = np.lib.format.open_memmap(writing, mode="w+", dtype=np.float32, shape=(35, 3, 4))
        client = Client()
        assert encode_frames(images, writer, client) == 35
        assert client.calls == [32, 3]
        writer.flush()
        del writer
        published = folder / "visual.npy"
        writing.rename(published)
        cache = np.load(published, mmap_mode="r", allow_pickle=False)
        ids = np.arange(100, 135)
        reference_ids = np.asarray([120, 101, 134, 100, 101])
        expected = np.broadcast_to((reference_ids - 100)[:, None, None], (5, 3, 4)).astype(np.float32)
        result = compare_reference(cache, ids, reference_ids, expected)
        assert result["passed"] and result["max_absolute_difference"] == 0
        reject(lambda: compare_reference(cache, ids, [99], expected[:1]))
        reject(lambda: compare_reference(cache, ids[::-1], reference_ids, expected))
        reject(lambda: compare_reference(cache, ids, reference_ids, expected + 1))
        reject(lambda: encode_frames(images, np.zeros((35, 3, 4), np.float32), Client(dtype=np.float64)))
        reject(lambda: encode_frames(images, np.zeros((35, 3, 4), np.float32), Client(fail=True)), RuntimeError)
        assert len(sha(published)) == 64
        del cache
    old = ROOT / "scripts/cluster/pusht_live_bridge_v1"
    assert all((BUNDLE / name).read_bytes() == (old / name).read_bytes()
               for name in ("clip_worker.py", "live_client.py", "wire.py", "resource_probe.py", "live_contract.json"))
    result = {"cpu_cache_mechanics_passed": True, "last_short_request32_plus3": True,
              "memory_mapped_publication_and_reference_mapping_exact": True,
              "missing_unsorted_wrong_features_wrong_dtype_and_encoder_failure_rejected": True,
              "existing_live_worker_files_byte_identical": True, "python39_AST": True,
              "actual_CLIP_or_training_data_or_GPU_executed": False,
              "full_cache_already_created": False, "formal_training_executed": False}
    (ROOT / "reports/pusht_project_cache_cpu_checks_20261019.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
