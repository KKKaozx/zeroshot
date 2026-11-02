"""CPU-only integrity review of prepared frozen training features."""
import argparse
import json
from pathlib import Path
import numpy as np
from cache_tools import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    contract = Path(__file__).with_name("cache_contract.json")
    assert report["cache_complete"] and not report["errors"]
    assert report["training_updates"] == 0 and not report["development_or_final_scored"]
    assert report["cache_contract_sha256"] == sha(contract)
    assert report["worker"]["all_frozen"] and report["worker"]["weights_unchanged"]
    assert report["reference128"]["passed"] and report["reference128"]["frames"] == 128
    for name, info in report["files"].items():
        assert name in ("visual.npy", "frame_ids.npy", "text.npz")
        path = args.report.parent / name
        assert path.stat().st_size == info["bytes"] and sha(path) == info["sha256"]
    assert set(report["files"]) == {"visual.npy", "frame_ids.npy", "text.npz"}
    visual = np.load(args.report.parent / "visual.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(args.report.parent / "frame_ids.npy", allow_pickle=False)
    assert visual.shape == (11356, 257, 1024) and visual.dtype == np.float32
    assert ids.shape == (11356,) and ids.dtype == np.int64 and (np.diff(ids) > 0).all()
    assert ids.min() >= 0 and ids.max() < 25650
    with np.load(args.report.parent / "text.npz", allow_pickle=False) as text:
        assert text["text"].shape[-1] == 768 and np.isfinite(text["text"]).all()
        assert text["text_mask"].shape == text["text"].shape[:2]
    print("PUSHT FROZEN TRAIN CACHE VERIFIED; no model performance verdict")
    print(json.dumps({"frames": len(ids), "cache_data_gib": report["cache_data_gib"],
                      "reference128": report["reference128"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
