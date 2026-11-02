"""Build a versioned restore-device overlay; never edit the submitted bundle."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts/cluster/pusht_native_training_v1"
BUNDLE = ROOT / "scripts/cluster/pusht_native_device_repair_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = (ORIGINAL / "training_core.py").read_text(encoding="utf-8")
    before = '''    policy.load_state_dict(payload["raw"], strict=True)
    averaged.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    # Torch 1.12 normally casts optimizer tensors when loading, enforce device.
    device = next(policy.parameters()).device
'''
    after = '''    # The author's normalizer rebuilds ParameterDict from the CPU payload.
    # Capture the destination BEFORE load, then move newly rebuilt tensors too.
    device = next(policy.parameters()).device
    policy.load_state_dict(payload["raw"], strict=True)
    averaged.load_state_dict(payload["ema"], strict=True)
    policy.to(device)
    averaged.to(device)
    for label, model in (("raw", policy), ("ema", averaged)):
        misplaced = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                     if value.device != device]
        if misplaced:
            raise ValueError("Restored tensors on wrong device: " + label + ": " + str(misplaced))
    RESTORE_DEVICE_CHECKS.append({"destination": str(device), "raw_and_ema_all_tensors_on_destination": True})
    print("PUSHT_RESTORE_DEVICE PASSED", str(device), "raw and EMA", flush=True)
    optimizer.load_state_dict(payload["optimizer"])
    # Torch 1.12 normally casts optimizer tensors when loading, enforce device.
'''
    assert source.count(before) == 1
    patched = source.replace(before, after).replace("import torch\n", "import torch\n\nRESTORE_DEVICE_CHECKS = []\n", 1)
    old_functions = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_functions = {n.name: ast.dump(n) for n in ast.parse(patched).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert set(old_functions) == set(new_functions)
    changed = [k for k in old_functions if old_functions[k] != new_functions[k]]
    assert changed == ["restore"]
    (BUNDLE / "training_core_repaired.py").write_text(patched, encoding="utf-8")
    original_hashes = {p.name: sha(p) for p in ORIGINAL.iterdir() if p.is_file()}
    contract = {"origin_failed_job": 211279, "scope": "restore tensor placement only",
                "original_file_sha256": original_hashes,
                "unchanged_training_protocol_sha256": sha(ORIGINAL / "training_protocol.json"),
                "changed_functions": changed, "training_protocol_or_loss_changed": False,
                "original_directory_modified": False, "failed_audit_checkpoint_used": False,
                "startup_audit_and_full_training_budget_kept": True}
    (BUNDLE / "repair_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    for path in BUNDLE.iterdir():
        if path.is_file():
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (BUNDLE / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files), encoding="ascii")
    archive = ROOT / "cluster_packages/pusht_native_device_repair_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in sorted(BUNDLE.iterdir()):
            if path.is_file():
                target.write(path, BUNDLE.name + "/" + path.name)
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        for row in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, name = row.split("  ", 1)
            assert hashlib.sha256(target.read(BUNDLE.name + "/" + name)).hexdigest() == digest
    release = {"archive": str(archive), "bytes": archive.stat().st_size,
               "sha256": sha(archive), "changed_functions": changed,
               "original_protocol_unchanged": True, "submitted": False}
    (ROOT / "reports/pusht_native_device_repair_release_20261018.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
