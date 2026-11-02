"""Check archive handling and chunk guards with synthetic inputs; no network/model."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_training_data_v1"


def must_fail(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected rejection")


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    spec = importlib.util.spec_from_file_location("pusht_prepare", BUNDLE / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work = ROOT / "reports/pusht_data_package_cpu_checks_20261018"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        directory = Path(directory)
        valid = directory / "valid.zip"
        with zipfile.ZipFile(valid, "w") as stream:
            for key in ("data/img", "data/action", "data/state", "meta/episode_ends"):
                stream.writestr("pusht/example.zarr/" + key + "/.zarray", "{}")
        assert module.archive_manifest(valid)["zarr_relative_path"] == "pusht/example.zarr"
        rejected = 0
        for index, name in enumerate(("../escape", "/absolute", "folder\\escape", "C:/drive")):
            path = directory / ("bad%d.zip" % index)
            with zipfile.ZipFile(path, "w") as stream:
                stream.writestr(name, "x")
            must_fail(lambda: module.archive_manifest(path))
            rejected += 1
        link = directory / "link.zip"
        with zipfile.ZipFile(link, "w") as stream:
            member = zipfile.ZipInfo("symbolic")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(member, "../outside")
        must_fail(lambda: module.archive_manifest(link))
        rejected += 1
        # Fake Zarr surface tests chunk-address construction and decoding guards.
        class Array:
            path = "data/action"
            _dimension_separator = "."
            chunks = (2, 2)

            def __init__(self, values):
                self.values = values
                self.shape = values.shape
                self.dtype = values.dtype
                self.store = {"data/action/0.0": b"x", "data/action/1.0": b"x"}

            def __getitem__(self, key):
                return self.values[key]

        array = Array(np.arange(6, dtype=np.float32).reshape(3, 2))
        assert module.check_array(array, "data/action", np)["stored_chunks_checked"] == 2
        del array.store["data/action/1.0"]
        must_fail(lambda: module.check_array(array, "data/action", np))
        array = Array(np.array([[np.nan, 1], [2, 3], [4, 5]], dtype=np.float32))
        must_fail(lambda: module.check_array(array, "data/action", np))
        # HTTPS body mocked locally: exercise publication, reuse and length guard.
        class Response(io.BytesIO):
            headers = {"Content-Length": "3"}
        original = module.urllib.request.urlopen
        try:
            module.urllib.request.urlopen = lambda *args, **kwargs: Response(b"abc")
            archive = directory / "download.zip"
            first = directory / "first"
            first.mkdir()
            assert module.download(archive, first)["downloaded_this_run"]
            assert archive.read_bytes() == b"abc"
            assert not module.download(archive, first)["downloaded_this_run"]
            Response.headers = {"Content-Length": str(module.MAX_DOWNLOAD + 1)}
            second = directory / "second"
            second.mkdir()
            try:
                module.download(directory / "oversized.zip", second)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Oversized HTTP response accepted")
        finally:
            module.urllib.request.urlopen = original
    result = {"python39_syntax": True, "unsafe_archive_cases_rejected": rejected,
              "valid_manifest": True, "chunk_presence_and_nonfinite_guards": True,
              "mock_download_publish_reuse_and_size_guard": True,
              "real_zarr_decode_tested": False, "real_download_executed": False,
              "models_loaded": False}
    (work / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
