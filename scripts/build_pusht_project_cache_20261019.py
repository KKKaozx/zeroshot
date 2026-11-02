"""Package the once-only frozen-feature cache preparation; no training entry."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_project_cache_v1"
LIVE = ROOT / "scripts/cluster/pusht_live_bridge_v1"
NATIVE = ROOT / "scripts/cluster/pusht_native_training_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ("clip_worker.py", "live_client.py", "wire.py", "resource_probe.py", "live_contract.json"):
        shutil.copyfile(LIVE / name, BUNDLE / name)
    shutil.copyfile(NATIVE / "training_protocol.json", BUNDLE / "base_protocol.json")
    native_source = (NATIVE / "train_native.py").read_text(encoding="utf-8")
    tree = ast.parse(native_source)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ("sha", "save_json", "TrainingData")]
    extracted = "\n\n\n".join(ast.get_source_segment(native_source, node) for node in selected)
    prefix = '"""Unchanged native training-data reader extracted without a training main."""\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport torch\n\n\n'
    (BUNDLE / "native_data.py").write_text(prefix + extracted + "\n", encoding="utf-8")
    old_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingData")
    new_cls = next(n for n in ast.parse(prefix + extracted).body if isinstance(n, ast.ClassDef))
    assert ast.dump(old_cls) == ast.dump(new_cls)
    protocol = json.loads((NATIVE / "training_protocol.json").read_text())
    resource = json.loads((ROOT / "reports/pusht_front_resource_210763.json").read_text())
    native20 = json.loads((ROOT / "reports/pusht_native_211309/report.json").read_text())
    cap = json.loads((ROOT / "reports/pusht_native_212158/development.json").read_text())
    assert cap["reference_competence_gate_passed"]
    required = dict(protocol["required_reports_sha256"])
    for path in (ROOT / "reports/pusht_native_212158").glob("*.json"):
        required["runs/pusht-native84k-212158/" + path.name] = sha(path)
    contract = {"stage": "once_only_frozen_train_feature_preparation",
                "native_reference_job": 212158, "matched_future_budget_updates": 84168,
                "required_reference_files_sha256": required,
                "resource_tokens_sha256": resource["records"]["clip"]["tokens_sha256"],
                "raw_training_images_sha256": native20["data"]["raw_training_images_sha256"],
                "train_episodes": 90, "windows": 10726, "frames": 11356,
                "visual_shape": [11356, 257, 1024], "visual_dtype": "float32",
                "visual_data_bytes": 11356 * 257 * 1024 * 4,
                "minimum_free_bytes": 32 * 1024 ** 3,
                "feature_reference_tolerance": {"atol": 1e-4, "rtol": 1e-5},
                "frozen_clip": "existing211193 worker unchanged; both vision/text frozen, full model hash unchanged",
                "data_reader": "Exact native TrainingData class, same raw-pixel hash and windows/normalizer input checks",
                "train_features_only": True, "training_updates": 0,
                "new_data_download_or_package_install": False,
                "development_or_final_scoring": False,
                "automatic_training_submission": False,
                "formal_training_entry_included": False}
    (BUNDLE / "cache_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    for path in BUNDLE.iterdir():
        if path.is_file():
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (BUNDLE / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files), encoding="ascii")
    archive = ROOT / "cluster_packages/pusht_project_cache_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in sorted(BUNDLE.iterdir()):
            if path.is_file():
                target.write(path, BUNDLE.name + "/" + path.name)
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        for row in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, name = row.split("  ", 1)
            assert hashlib.sha256(target.read(BUNDLE.name + "/" + name)).hexdigest() == digest
    release = {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
               "file_count": len(files) + 1, "cache_contract_sha256": sha(BUNDLE / "cache_contract.json"),
               "cache_data_gib": contract["visual_data_bytes"] / 1024 ** 3,
               "original_live_worker_unchanged": True, "native_training_data_AST_unchanged": True,
               "formal_training_entry_included": False, "submitted": False}
    (ROOT / "reports/pusht_project_cache_release_20261019.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
