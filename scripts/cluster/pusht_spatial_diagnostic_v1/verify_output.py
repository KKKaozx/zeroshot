"""Offline numerical output audit only; never loads a policy or selects a variant."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_math import labels_from_state, summarize, paired_interval


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)):
        return bool(np.isclose(a, b, atol=1e-9, rtol=1e-10))
    return a == b


def validate(output, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    r = json.loads((output / "report.json").read_text())
    if (not r["diagnostic_complete"] or r["errors"] or r["policy_training_updates"] != 0
            or r["policy_rollouts"] != 0 or r["sealed_final_accessed"] or r["policy_development_scores_used"]
            or r["policy_or_clip_weights_updated"] or not r["diagnostic_heads_fitted"]
            or r["diagnostic_readout_hyperparameter_search"]):
        raise ValueError("Diagnostic incomplete or scope violation")
    if r["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("Diagnostic protocol changed")
    for group in ("project_frozen", "native_frozen"):
        if not r[group]["unchanged"]:
            raise ValueError("Frozen encoder changed")
    if not r["project_frozen"]["clip_worker"]["all_frozen"] or not r["project_frozen"]["clip_worker"]["weights_unchanged"]:
        raise ValueError("CLIP freeze check failed")
    manifest_path = output / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != r["inputs"]["manifest_sha256"]:
        raise ValueError("Frame manifest changed")
    with np.load(output / "predictions.npz", allow_pickle=False) as a:
        labels, fit, episodes, ids = a["labels"], a["fit_mask"], a["episode_ids"], a["frame_ids"]
        if not (len(fit) == 1560 and fit.sum() == 1080 and np.array_equal(labels, labels_from_state(a["source_state"]))
                and np.array_equal(ids, manifest["frame_ids"]) and len(np.unique(ids)) == len(ids)
                and np.array_equal(episodes, manifest["episode_ids"])
                and np.array_equal(fit, np.asarray(manifest["roles"]) == "fit")):
            raise ValueError("Source-label/split/frame identity differs")
        for role, mask in (("fit", fit), ("heldout", ~fit)):
            expected = protocol["split"][role + "_episode_ids"]
            if sorted(set(episodes[mask].tolist())) != sorted(expected):
                raise ValueError("Episode role differs")
            if any(np.sum(episodes[mask] == ep) != 12 for ep in expected):
                raise ValueError("Episode weighting differs")
        for stage in protocol["readout_stages"] + ["train_mean"]:
            for role, mask in (("fit", fit), ("heldout", ~fit)):
                if not same(summarize(a[stage], labels, episodes, mask), r["readouts"][stage][role]):
                    raise ValueError("Recomputed metrics differ: " + stage + "/" + role)
            if stage != "train_mean" and not same(summarize(a[stage + "_shuffled"], labels, episodes, ~fit),
                                                 r["readouts"][stage]["shuffled_heldout"]):
                raise ValueError("Shuffled control differs")
        constant = np.repeat(labels[fit].mean(0)[None], len(labels), axis=0)
        if not np.array_equal(constant, a["train_mean"]):
            raise ValueError("Mean baseline used wrong split")
    for left, right in protocol["paired_comparisons"]:
        key = left + "__to__" + right
        if not same(paired_interval(r["readouts"][left]["heldout"], r["readouts"][right]["heldout"]), r["comparisons"][key]):
            raise ValueError("Paired interval differs")
    return {"output_verified": True, "scope": "Frozen representation readout only; no policy improvement verdict",
            "heldout": {k: v["heldout"]["episode_equal"] for k, v in r["readouts"].items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    verified = validate(args.output, Path(__file__).with_name("protocol.json"))
    print("FROZEN SPATIAL DIAGNOSTIC OUTPUT VERIFIED; not a policy/generalization improvement verdict")
    print(json.dumps(verified, indent=2))
