"""Pin verified20k evidence and generate one bounded continuation, not new tuning."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts/cluster/pusht_native_training_v1"
REPAIR = ROOT / "scripts/cluster/pusht_native_device_repair_v1"
BUNDLE = ROOT / "scripts/cluster/pusht_native_extension_v1"
EVIDENCE = ROOT / "reports/pusht_native_211309"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis = json.loads((ROOT / "reports/pusht_native_analysis_211309_20261019.json").read_text())
    assert analysis["full_small_artifacts_verified_locally"] and not analysis["reference_competence_gate_passed"]
    report = json.loads((EVIDENCE / "report.json").read_text())
    for name in ("checkpoint_config.yaml", "resource_probe.py", "training_protocol.json"):
        shutil.copyfile(ORIGINAL / name, BUNDLE / name)
    shutil.copyfile(REPAIR / "training_core_repaired.py", BUNDLE / "training_core.py")
    shutil.copyfile(ORIGINAL / "verify_training.py", BUNDLE / "base_verifier.py")
    protocol = json.loads((ORIGINAL / "training_protocol.json").read_text())
    source = (ORIGINAL / "train_native.py").read_text(encoding="utf-8")
    replacements = [
        ('"""Native Push-T reference: 20k updates then fixed EMA development rollouts.',
         '"""Native Push-T continuation: verified20k to predeclared84168, fixed endpoint EMA rollouts.'),
        ('        protocol = json.loads(protocol_path.read_text())',
         '''        protocol = json.loads(protocol_path.read_text())
        extension_path = bundle / "extension_contract.json"
        extension = json.loads(extension_path.read_text())
        target_updates = extension["target_updates"]
        assert target_updates == protocol["native_upper_cap_updates"] == 84168
        if args.resume is None:
            raise ValueError("Extension requires an explicit complete-state checkpoint")
        result["extension_contract_sha256"] = sha(extension_path)
        result["source20k_checkpoint_sha256"] = extension["source_checkpoint_sha256"]
        result["stage"] = "native_continuation_to_predeclared_cap"'''),
        ('            completed = restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)',
         '''            completed = restore(payload, policy, averaged, optimizer, scheduler, ema, protocol_sha, plan_sha)
            expected_resume_digest = tree_digest(payload)
            restored_resume_digest = tree_digest(capture(policy, averaged, optimizer, scheduler, ema,
                                                        completed, protocol_sha, plan_sha))
            if expected_resume_digest != restored_resume_digest:
                raise ValueError("Full restored state differs from source checkpoint")
            result["complete_resume_state_exact"] = True
            result["resume_state_digest"] = restored_resume_digest
            print("PUSHT_NATIVE_EXTENSION_RESUME PASSED", completed, "/", target_updates, flush=True)'''),
        ('        assert 0 <= completed <= protocol["first_checkpoint_updates"]',
         '        assert 20000 <= completed <= target_updates\n        result["completed_updates"] = completed'),
        ('plan.batch(completed if completed < 20000 else 19999)', 'plan.batch(min(completed, target_updates - 1))'),
        ('            while completed < protocol["first_checkpoint_updates"]:', '            while completed < target_updates:'),
        ('"/20000 loss"', '"/84168 loss"'),
        ('                if completed % 2000 == 0 or completed == 20000:',
         '                if completed % 2000 == 0 or completed == target_updates:'),
        ('"protocol_sha256": protocol_sha, "sample_plan_sha256": plan_sha,\n                               "startup_checks"',
         '"protocol_sha256": protocol_sha, "sample_plan_sha256": plan_sha,\n                               "extension_contract_sha256": sha(extension_path),\n                               "startup_checks"'),
        ('"EMA at exactly20000 updates; no best-of-development selection"',
         '"EMA at exactly84168 updates; no best-of-development selection"'),
        ('"Review native result before preparing a matched project run"',
         '"Native cap competent; review before preparing matched84168 frozen-CLIP project group"'),
        ('"Native reference not established at20k; inspect evidence before any predefined-cap continuation; no project attribution"',
         '"Native cap not competent; stop this module-comparison branch, no extra budget or project attribution"'),
        ('PUSHT_NATIVE_20K COMPLETE', 'PUSHT_NATIVE_84168 COMPLETE'),
        ('# Resume-at20k has no new checkpoint;', '# Resume-at-cap has no new checkpoint;')
    ]
    edits = []
    for before, after in replacements:
        count = source.count(before)
        expected_count = 2 if before.startswith('"protocol_sha256"') else 1
        assert count == expected_count, (before, count, expected_count)
        source = source.replace(before, after)
        edits.append({"before": before, "after": after, "occurrences": count})
    old_tree = ast.parse((ORIGINAL / "train_native.py").read_text())
    new_tree = ast.parse(source)
    get = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = get(old_tree), get(new_tree)
    changed = [name for name in old if old[name] != new[name]]
    assert changed == ["evaluate", "main"]
    (BUNDLE / "effective_train_native.py").write_text(source, encoding="utf-8")
    required = {"runs/pusht-native20k-211309/" + p.name: sha(p) for p in EVIDENCE.glob("*.json")}
    contract = {"stage": "native_predeclared_cap_continuation", "source_job": 211309,
                "source_checkpoint": "runs/pusht-native20k-211309/latest.pt",
                "source_checkpoint_sha256": report["checkpoint_sha256"],
                "source_completed_updates": 20000, "target_updates": 84168, "new_updates": 64168,
                "base_protocol_sha256": report["protocol_sha256"],
                "sample_plan_sha256": report["sample_plan_sha256"],
                "sample_order_sha256": report["sample_order_sha256"],
                "raw_training_images_sha256": report["data"]["raw_training_images_sha256"],
                "required_reference_files_sha256": required,
                "original_runner_sha256": sha(ORIGINAL / "train_native.py"),
                "repaired_training_core_sha256": sha(REPAIR / "training_core_repaired.py"),
                "effective_runner_changed_functions": changed,
                "algorithm_or_hyperparameter_change": False,
                "resume": "Restore raw/EMA/Adam/LR/EMA counter/all RNG and compare complete state digest; continue sample/noise plan at saved counter",
                "lr_horizon_updates": 512400, "lr_restarted": False,
                "evaluation": "Only endpoint84168 EMA; same20 dev seeds/sampling streams and same competence gate",
                "cap_failure": "Stop this module-comparison branch, no extra native budget or project attribution",
                "automatic_project_training_or_final_access": False,
                "clip_unfreeze_allowed": False}
    (BUNDLE / "extension_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/pusht_native_extension_source_edits_20261019.json").write_text(json.dumps(edits, indent=2) + "\n")
    for path in BUNDLE.iterdir():
        if path.is_file():
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    assert sha(BUNDLE / "training_protocol.json") == report["protocol_sha256"]
    assert sha(BUNDLE / "training_core.py") == sha(REPAIR / "training_core_repaired.py")
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (BUNDLE / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files), encoding="ascii")
    archive = ROOT / "cluster_packages/pusht_native_extension_v1.zip"
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
               "file_count": len(files) + 1, "extension_contract_sha256": sha(BUNDLE / "extension_contract.json"),
               "start_updates": 20000, "end_updates": 84168, "new_updates": 64168,
               "estimated_pure_training_hours_from_observed_20k": report["training_seconds"] / 20000 * 64168 / 3600,
               "no_weights_data_or_tokens_in_archive": True, "submitted": False}
    (ROOT / "reports/pusht_native_extension_release_20261019.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
