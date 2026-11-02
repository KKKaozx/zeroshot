"""New formal-entry trace/binding/gate/verifier checks; no policy or rollout execution."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_bypass_formal_v1"
sys.path.insert(0, str(BUNDLE))
from pair_metrics import admission
from verify_formal import validate
from native_data import sha
import training_core as core
import bind_preflight


def development(scores, selection, base):
    scores = list(map(float, scores))
    mean, full = float(np.mean(scores)), sum(s == 1 for s in scores) / 20
    return {"episodes": [{"environment_seed": 5000000 + i, "reward_series": [s], "steps": 1,
                          "max_overlap_reward": s, "full_reward": s == 1.,
                          "video": "development_seed_%d.mp4" % (5000000 + i) if i < 2 else None} for i, s in enumerate(scores)],
            "mean_score": mean, "full_reward_fraction": full, "complete": True, "final_set_accessed": False,
            "reference_competence_gate_passed": mean >= base["development"]["competence_mean_score"] and full >= base["development"]["competence_full_reward_fraction"],
            "checkpoint_selection": selection}


def main():
    torch.set_num_threads(1)
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, f):
        try:
            f()
        except (ValueError, AssertionError):
            checks.append(name)
        else:
            raise AssertionError(name)
    contract = json.loads((BUNDLE / "formal_contract.json").read_text())
    protocol = json.loads((BUNDLE / "protocol.json").read_text())
    base = json.loads((BUNDLE / "base_protocol.json").read_text())
    make_dev = lambda values: development(values, contract["endpoint_selection"], base)
    baseline = development([.3] * 20, "old fixed endpoint", base)
    control, candidate = make_dev([.2] * 20), make_dev([.5] * 20)
    outcome = admission(control, candidate, baseline, base, contract)
    check("meaningful_matched_gain_admitted", outcome["development_admission_passed"])
    small = admission(control, make_dev([.21] * 20), baseline, base, contract)
    check("small_or_baseline_worse_gain_not_admitted", not small["development_admission_passed"])
    mixed_control = make_dev([1.] + [.1] * 19)
    mixed_candidate = make_dev([.7] * 20)
    losing_full = admission(mixed_control, mixed_candidate, baseline, base, contract)
    check("mean_gain_cannot_hide_lower_full_reward_fraction", not losing_full["gates"]["full_reward_fraction_no_worse_than_both"] and not losing_full["development_admission_passed"])
    uncertain = admission(make_dev([.1] * 20), make_dev([1.] * 4 + [0.] * 16), development([.05] * 20, "old", base), base, contract)
    check("paired_uncertainty_gate_blocks_unstable_mean", uncertain["gates"]["mean_gain_at_least0_05_vs_global_control"] and not uncertain["gates"]["paired_mean_gain_bootstrap95_lower_positive"])
    check("paired_bootstrap_deterministic", outcome == admission(control, candidate, baseline, base, contract))
    malformed = copy.deepcopy(candidate)
    malformed["episodes"][0]["environment_seed"] += 1
    reject("reject_unpaired_scene_seeds", lambda: admission(control, malformed, baseline, base, contract))
    malformed = copy.deepcopy(candidate)
    malformed["checkpoint_selection"] = "best of several checkpoints"
    reject("reject_best_checkpoint_selection", lambda: admission(control, malformed, baseline, base, contract))
    malformed = copy.deepcopy(candidate)
    malformed["episodes"][0]["max_overlap_reward"] = .3
    reject("reject_inconsistent_reward_series", lambda: admission(control, malformed, baseline, base, contract))
    # The evaluator's executable environment/noise/reward path is identical.
    original = ast.parse((ROOT / "scripts/cluster/pusht_project_training_v1/evaluate_project.py").read_text())
    changed = ast.parse((BUNDLE / "evaluate_bypass.py").read_text())
    selection_old = "EMA at exactly84168 updates; no best-of-development selection"
    mapping = {"SPATIAL_BYPASS_DEV": "PUSHT_PROJECT_DEV", contract["endpoint_selection"]: selection_old,
        "Compare matched bypass endpoints under the predeclared development gate; keep final50 sealed": "Compare fixed endpoint project frontend with admitted native reference; keep final50 sealed",
        "Inspect the predeclared paired development gate; no automatic further training or final access": "Project below reference competence threshold; inspect matched frontend evidence before new experiments"}
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in mapping:
                return ast.copy_location(ast.Constant(mapping[node.value]), node)
            return node
    check("evaluator_only_logging_and_metadata_changed", ast.dump(original) == ast.dump(Normalize().visit(changed)))
    # Run the actual formal while-loop with cheap stand-ins, exactly8192 steps.
    tree = ast.parse((BUNDLE / "run_formal.py").read_text())
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main_fn) if isinstance(n, ast.While))
    namespace = {"current": 84168, "contract": contract, "plan": core.BatchPlan(), "np": np, "torch": torch,
        "mode": "spatial_candidate", "row": {}, "first_losses": [], "log": io.StringIO(), "json": json,
        "order_digest": hashlib.sha256(), "lr_digest": hashlib.sha256(), "train_start": 0,
        "time": SimpleNamespace(monotonic=lambda: 1.), "print": lambda *a, **kw: None,
        "scheduler": SimpleNamespace(get_last_lr=lambda: [1e-4]),
        "policy": SimpleNamespace(obs_encoder=SimpleNamespace(bypass_head=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(1))))) }
    namespace["policy"].obs_encoder.bypass_head.weight.grad = torch.ones(1)
    namespace["data"] = SimpleNamespace(batch=lambda indices: indices)
    namespace["cache"] = SimpleNamespace(tokens=lambda indices: indices)
    calls, checkpoints = [], []
    def fake_update(core_arg, p, opt, scheduler, ema, batch, tokens, index, audit=False):
        calls.append((index, len(batch), audit))
        return .2, {"original_compute_loss_used": True} if audit else None
    namespace.update(core=core, optimizer=None, ema=None, project_update=fake_update,
                     save_checkpoint=lambda: checkpoints.append(namespace["current"]))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "actual_formal_loop", "exec"), namespace)
    check("actual_loop_exact84168_to92360_no_preflight_steps", [c[0] for c in calls] == list(range(84168, 92360)) and namespace["current"] == 92360)
    check("first_update_only_objective_audit", [c[0] for c in calls if c[2]] == [84168])
    check("partial_logical_batches38_retained", all(n in (38, 64) for _, n, _ in calls) and any(n == 38 for _, n, _ in calls))
    check("actual_continuation_order_matches_frozen_contract", namespace["order_digest"].hexdigest() == contract["continuation_order_sha256"])
    check("complete_checkpoints_every1024_and_endpoint", checkpoints == list(range(85192, 92361, 1024)))
    lines = [json.loads(line) for line in namespace["log"].getvalue().splitlines()]
    check("one_unique_log_per_formal_update", len(lines) == 8192 and len({x["completed_updates"] for x in lines}) == 8192)
    # Binding interface uses the exact admitted report/statistics SHA, not8-step weights.
    with tempfile.TemporaryDirectory(prefix="bypass_binding_", dir=ROOT / "reports") as temp:
        temp = Path(temp)
        copied = temp / "bundle"
        shutil.copytree(BUNDLE, copied, ignore=shutil.ignore_patterns("__pycache__"))
        preflight = temp / contract["preflight_directory"]
        preflight.mkdir(parents=True)
        (preflight / "report.json").write_text(json.dumps({"source_checkpoint_sha256": protocol["source_checkpoint_sha256"]}))
        (preflight / "bypass_statistics.npz").write_bytes(b"explicitly synthetic statistic bytes")
        binding_path = temp / "preflight_binding.json"
        verified = {"preflight_verified": True}
        with patch.object(bind_preflight, "__file__", str(copied / "bind_preflight.py")), patch.object(bind_preflight, "validate", return_value=verified) as validator:
            with patch.object(sys, "argv", ["bind_preflight.py", "--root", str(temp), "--binding", str(binding_path)]):
                bind_preflight.main()
            binding = json.loads(binding_path.read_text())
            check("binding_invokes_original_preflight_verifier_once", validator.call_count == 1)
            check("binding_pins_small_artifact_sha_and_original_source", binding["preflight_report_sha256"] == sha(preflight / "report.json")
                  and binding["statistics_sha256"] == sha(preflight / "bypass_statistics.npz") and binding["source_checkpoint_sha256"] == protocol["source_checkpoint_sha256"]
                  and not binding["preflight_updates_used_as_training_start"])
            with patch.object(sys, "argv", ["bind_preflight.py", "--root", str(temp), "--binding", str(binding_path)]):
                reject("reject_replacing_existing_submission_binding", bind_preflight.main)
    # Roundtrip the actual full output verifier with a synthetic paired artifact set.
    with tempfile.TemporaryDirectory(prefix="bypass_output_", dir=ROOT / "reports") as temp:
        output = Path(temp)
        (output / "bypass_statistics.npz").write_bytes(b"synthetic statistics only")
        binding = {"preflight_job": 217778, "verified": {"preflight_verified": True},
                   "formal_contract_sha256": sha(BUNDLE / "formal_contract.json"), "preflight_report_sha256": "synthetic-report-sha",
                   "statistics_sha256": sha(output / "bypass_statistics.npz")}
        (output / "preflight_binding.json").write_text(json.dumps(binding))
        report = {"formal_complete": True, "errors": [], "final_set_accessed": False, "clip_training_executed": False,
            "automatic_following_submission": False, "matched_initial_loss_exact": True, "matched_lr_trace_exact": True,
            "formal_contract_sha256": sha(BUNDLE / "formal_contract.json"), "preflight_protocol_sha256": sha(BUNDLE / "protocol.json"),
            "source_checkpoint_sha256": protocol["source_checkpoint_sha256"], "source_updates": 84168, "endpoint_updates": 92360, "new_updates_per_mode": 8192,
            "preflight_binding_sha256": sha(output / "preflight_binding.json"), "preflight_report_sha256": "synthetic-report-sha",
            "statistics_sha256": sha(output / "bypass_statistics.npz"), "data": {"raw_training_images_sha256": protocol["raw_training_images_sha256"]},
            "source_sample_plan_sha256": "synthetic-plan", "groups": {}}
        devs = {"global_control": make_dev([.2] * 20), "spatial_candidate": make_dev([1.] * 10 + [.7] * 10)}
        for mode, dev in devs.items():
            directory = output / mode
            directory.mkdir()
            (directory / "latest.pt").write_bytes(b"synthetic complete checkpoint placeholder")
            side = {"sha256": sha(directory / "latest.pt"), "mode": mode, "completed_updates": 92360, "new_updates": 8192,
                "formal_contract_sha256": report["formal_contract_sha256"], "bypass_protocol_sha256": report["preflight_protocol_sha256"],
                "preflight_binding_sha256": report["preflight_binding_sha256"], "statistics_sha256": report["statistics_sha256"],
                "source_checkpoint_sha256": report["source_checkpoint_sha256"], "sample_plan_sha256": "synthetic-plan"}
            (directory / "latest.json").write_text(json.dumps(side))
            (directory / "updates.jsonl").write_text(namespace["log"].getvalue())
            (directory / "development.json").write_text(json.dumps(dev))
            for item in dev["episodes"][:2]:
                (directory / item["video"]).write_bytes(b"synthetic metadata-only video placeholder")
            row = {"training_completed": True, "development_complete": True, "old_complete_restore_exact": True,
                "preflight_weights_used": False, "fixed_text_and_statistics_unchanged": True, "old_learning_rate_horizon_unchanged": True,
                "ema_state_unchanged_during_evaluation": True, "source_updates": 84168, "completed_updates": 92360, "new_updates": 8192,
                "new_head_optimizer_step": 8192, "ema_optimization_step": 92360, "first_loss_and_gradient_audit": {"ok": True},
                "attach": {"new_parameters": 262144}, "evaluation_worker": {"all_frozen": True, "weights_unchanged": True},
                "checkpoint": str(directory / "latest.pt"), "checkpoint_sha256": side["sha256"],
                "lr_trace_sha256": namespace["lr_digest"].hexdigest(), "continuation_order_sha256": contract["continuation_order_sha256"], "development": dev}
            report["groups"][mode] = row
            (directory / "report.json").write_text(json.dumps(row))
        actual_baseline = json.loads((BUNDLE / "unchanged_development.json").read_text())
        report["admission"] = admission(devs["global_control"], devs["spatial_candidate"], actual_baseline, base, contract)
        (output / "admission.json").write_text(json.dumps(report["admission"]))
        (output / "report.json").write_text(json.dumps(report))
        check("full_output_verifier_accepts_consistent_synthetic_artifacts", validate(output, BUNDLE)["output_verified"])
        altered = copy.deepcopy(report)
        altered["admission"]["development_admission_passed"] = not report["admission"]["development_admission_passed"]
        (output / "report.json").write_text(json.dumps(altered))
        reject("verifier_rejects_edited_admission", lambda: validate(output, BUNDLE))
        (output / "report.json").write_text(json.dumps(report))
        (output / "global_control/latest.pt").write_bytes(b"changed synthetic bytes")
        reject("verifier_rejects_changed_checkpoint_bytes", lambda: validate(output, BUNDLE))
    for name, digest in contract["bundle_sources_sha256"].items():
        check("pinned_" + name, sha(BUNDLE / name) == digest)
    for file in BUNDLE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), feature_version=(3, 9))
    checks.append("python39_syntax_all_modules")
    for name in ("formal.sh", "submit_once.sh"):
        subprocess.run(["C:/Program Files/Git/bin/bash.exe", "-n", str(BUNDLE / name)], check=True)
        checks.append("bash_syntax_" + name)
    archive = ROOT / "cluster_packages/pusht_spatial_bypass_formal_v1.zip"
    with zipfile.ZipFile(archive) as z:
        check("archive_members_byte_exact", all(z.read(BUNDLE.name + "/" + p.name) == p.read_bytes() for p in BUNDLE.iterdir() if p.is_file()))
    report = {"all_passed": True, "count": len(checks), "checks": checks, "archive_sha256": sha(archive),
              "scope": "Only NEW formal loop/binding/gates/output-verifier and metadata checks on synthetic stand-ins; no repeat of existing67 model-interface checks",
              "actual_model_checkpoints_loaded": False, "actual_policy_training": False, "actual_policy_rollouts": 0,
              "preflight_resource_evidence": "User reported original server verifier passed217778", "cluster_job_submitted": False}
    (ROOT / "reports/pusht_spatial_bypass_formal_checks_20261021.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
