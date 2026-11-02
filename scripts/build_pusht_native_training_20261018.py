"""Build the native20k release from pinned local evidence; no model execution."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_native_training_v1"
RESOURCE = ROOT / "scripts/cluster/pusht_frontend_resource_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for name in ("checkpoint_config.yaml", "resource_probe.py"):
        shutil.copyfile(RESOURCE / name, BUNDLE / name)
    resource_protocol = json.loads((RESOURCE / "protocol.json").read_text())
    resource_protocol["author_file_sha256"]["diffusion_policy/model/common/lr_scheduler.py"] = sha(
        ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/model/common/lr_scheduler.py")
    resource = json.loads((ROOT / "reports/pusht_front_resource_210763.json").read_text())
    live_path = ROOT / "reports/pusht_online_211193.json"
    live = json.loads(live_path.read_text())
    assert live["bridge_execution_passed"] and not live["errors"]
    assert live["worker"]["all_frozen"] and live["worker"]["weights_unchanged"]
    assert all(v["passed"] and v["max_absolute_difference"] == 0 for v in live["identity_checks"].values())
    protocol = {
        "stage": "native_reference_first20k_only",
        "goal": "Establish a competent trained native Push-T reference before matched frozen-CLIP frontend training",
        "data_report_sha256": resource_protocol["data_report_sha256"],
        "required_reports_sha256": {
            "runs/pusht-float-data-210690/report.json": sha(ROOT / "reports/pusht_float_data_210690.json"),
            "runs/pusht-front-resource-210763/report.json": sha(ROOT / "reports/pusht_front_resource_210763.json"),
            "runs/pusht-online-211193/report.json": sha(live_path)},
        "resource_inputs_sha256": resource["records"]["prepare"]["input_sha256"],
        "author_file_sha256": resource_protocol["author_file_sha256"],
        "initial_backend_sha256": resource["records"]["native"]["initial_backend_sha256"],
        "normalizer_sha256": resource["records"]["native"]["normalizer_sha256"],
        "first_checkpoint_updates": 20000,
        "native_upper_cap_updates": 84168,
        "upper_cap_extension_exposed_by_this_entry": False,
        "lr_horizon_updates": 168 * 3050,
        "lr_schedule": "Author cosine, warmup500, original planned3050 epochs *168 logical batches; no20k cosine compression",
        "batch_plan": {"algorithm": "RandomState(42+epoch).permutation(10726);168 batches per epoch, final38",
                       "logical_batch": 64, "microbatch": 8, "partial_micro_weight": "actual micro_size / actual logical_size",
                       "checkpoint_identity": "SHA256(sample order digest + raw training image digest), rejects changed pixels at resume"},
        "diffusion_rng": "SHA256-derived seed by update/micro, reset after original mask; independent from frontend stream",
        "loss": "Unmodified pinned author compute_loss, epsilon objective, cosineDDPM100; original target and add_noise share local noise",
        "training_differences_from_original_workspace": [
            "Train90 frames only for normalizer, shared between future frontend groups",
            "Explicit deterministic shuffle and separate per-micro frontend/diffusion random streams",
            "Memory-equivalent sample-weighted microbatch accumulation,64=8x8 except final38",
            "Deterministic kernels requested and full EMA optimization counter restored",
            "Fixed20k development endpoint, no original online W&B or repeated validation checkpoint search"],
        "startup_checks": ["First original loss/noise/t and representative gradient", "Complete disk restore and update2 exact replay"],
        "development": {"start_seed": 5000000, "count": 20, "max_steps": 300,
                        "diffusion_seed_base": 6000000, "serial_batch": 1,
                        "checkpoint": "EMA at20000 only", "competence_mean_score": 0.8,
                        "competence_full_reward_fraction": 0.5, "videos": "first2 seeds, no outcome selection"},
        "sealed_final": {"start_seed": 5100000, "count": 50, "access_allowed_by_this_entry": False},
        "if_native20k_passes": "Prepare project family with same20k budget and common plan, fresh matched backend, frozen CLIP",
        "if_native20k_fails": "No frontend attribution or project run; review bounded native continuation separately",
        "automatic_other_submission": False,
        "clip_must_remain_frozen": True,
        "bridge207558_or_final25_access": False
    }
    (BUNDLE / "training_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for path in BUNDLE.iterdir():
        if path.is_file() and path.suffix in (".py", ".sh", ".md", ".json", ".yaml"):
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (BUNDLE / "SHA256SUMS").write_text("".join(sha(p) + "  " + p.name + "\n" for p in files), encoding="ascii")
    archive = ROOT / "cluster_packages/pusht_native_training_v1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in sorted(BUNDLE.iterdir()):
            if path.is_file():
                target.write(path, "pusht_native_training_v1/" + path.name)
    with zipfile.ZipFile(archive) as target:
        assert target.testzip() is None
        for row in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, name = row.split("  ", 1)
            assert hashlib.sha256(target.read("pusht_native_training_v1/" + name)).hexdigest() == digest
    release = {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
               "file_count": len(files) + 1, "protocol_sha256": sha(BUNDLE / "training_protocol.json"),
               "formal_native_updates": 20000, "automatic_job_submitted": False,
               "contains_weights_data_or_token_cache": False}
    (ROOT / "reports/pusht_native_training_release_20261018.json").write_text(json.dumps(release, indent=2) + "\n")
    analysis = {"online_report_sha256": sha(live_path), "full_report_verified_locally": True,
                "clip_all_frozen_and_unchanged": True, "all_five_identity_differences_zero": True,
                "two_decisions_16_steps_completed": True, "training_updates": 0,
                "task_score_or_generalization_verdict": None}
    (ROOT / "reports/pusht_online_analysis_211193.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
