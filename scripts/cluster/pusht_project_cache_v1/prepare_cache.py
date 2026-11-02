"""Encode established90 training demos once with strictly frozen CLIP."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from cache_tools import sha, encode_frames, compare_reference


def save(path, value):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/projects/Zeroshot"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if root / "runs" not in output.parents:
        parser.error("Output must be a new directory under root/runs")
    output.mkdir(parents=True, exist_ok=False)
    result = {"cache_complete": False, "errors": [], "training_updates": 0,
              "development_or_final_scored": False, "automatic_training_submission": False}
    client, writer = None, None
    started = time.monotonic()
    try:
        bundle = Path(__file__).resolve().parent
        contract_path = bundle / "cache_contract.json"
        contract = json.loads(contract_path.read_text())
        protocol = json.loads((bundle / "base_protocol.json").read_text())
        if shutil.disk_usage(output).free < contract["minimum_free_bytes"]:
            raise ValueError("Need32GiB free before the11.13GiB cache, reserving space for later complete checkpoints")
        for relative, digest in contract["required_reference_files_sha256"].items():
            if sha(root / relative) != digest:
                raise ValueError("Established reference changed: " + relative)
        source = root / "paper_reproduction/official_pusht_v1/source"
        for relative, digest in protocol["author_file_sha256"].items():
            if sha(source / relative) != digest:
                raise ValueError("Author source changed: " + relative)
        sys.path.insert(0, str(source))
        from native_data import TrainingData
        from live_client import CLIPClient
        data = TrainingData(root, protocol, output)
        if data.evidence["raw_training_images_sha256"] != contract["raw_training_images_sha256"]:
            raise ValueError("Training raw pixels differ from the native reference")
        frame_ids = np.flatnonzero(data.lookup >= 0).astype("<i8")
        assert np.array_equal(data.lookup[frame_ids], np.arange(11356))
        assert len(frame_ids) == len(data.images) == 11356
        resource = json.loads((root / "runs/pusht-front-resource-210763/report.json").read_text())
        token_path = root / "runs/pusht-front-resource-210763/tokens.npz"
        if sha(token_path) != contract["resource_tokens_sha256"]:
            raise ValueError("Established resource token file changed")
        client = CLIPClient(root, output)
        with np.load(token_path, allow_pickle=False) as reference:
            assert np.array_equal(client.text_mask, reference["text_mask"])
            assert np.allclose(client.text, reference["text"], atol=1e-4, rtol=1e-5)
            text_delta = float(np.max(np.abs(client.text - reference["text"])))
        np.savez(output / "text.npz", text=client.text, text_mask=client.text_mask)
        np.save(output / "frame_ids.npy", frame_ids, allow_pickle=False)
        writing = output / "visual.writing.npy"
        writer = np.lib.format.open_memmap(writing, mode="w+", dtype="<f4", shape=(11356, 257, 1024))
        written = encode_frames(data.images, writer, client,
            lambda stop, count: print("PUSHT_CLIP_CACHE", stop, "/", count, flush=True))
        assert written == 11356
        # Close before a long disk flush/hash so the idle worker cannot time out.
        worker = client.close()
        if not worker["all_frozen"] or not worker["weights_unchanged"]:
            raise ValueError("Frozen CLIP invariant failed")
        request_seconds = list(client.latencies)
        client = None
        writer.flush()
        del writer
        writer = None
        gc.collect()
        os.replace(writing, output / "visual.npy")
        visual = np.load(output / "visual.npy", mmap_mode="r", allow_pickle=False)
        reference_ids = resource["records"]["prepare"]["observation_frame_ids"]
        with np.load(token_path, allow_pickle=False) as reference:
            comparison = compare_reference(visual, frame_ids, reference_ids,
                                            reference["visual"].reshape(128, 257, 1024))
        files = {name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size}
                 for name in ("visual.npy", "frame_ids.npy", "text.npz")}
        result.update(cache_complete=True, cache_contract_sha256=sha(contract_path),
                      files=files, visual_shape=list(visual.shape), visual_dtype=str(visual.dtype),
                      training_data=data.evidence, reference128=comparison,
                      text_max_absolute_difference=text_delta, worker=worker,
                      requests=len(request_seconds), clip_request_seconds=sum(request_seconds),
                      cache_data_gib=visual.nbytes / 1024 ** 3,
                      scope="Frozen train-frame features only; no Adapter/native training or task evaluation")
    except Exception as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        if client is not None:
            try:
                client.abort()
            except Exception as cleanup:
                result["errors"].append("Worker cleanup: " + str(cleanup))
        if writer is not None:
            try:
                writer.flush()
            except Exception as cleanup:
                result["errors"].append("Cache flush: " + str(cleanup))
        result["elapsed_seconds"] = time.monotonic() - started
        save(output / "report.json", result)
        print("PUSHT_PROJECT_CACHE_REPORT", output / "report.json", flush=True)
    print("PUSHT FROZEN TRAIN CACHE COMPLETE; no model training or generalization verdict", flush=True)


if __name__ == "__main__":
    main()
