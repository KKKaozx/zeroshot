"""Small CUDA continuation + actual generated loop boundary tests, no task data."""
import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_native_extension_v1"
sys.path.insert(0, str(ROOT / "scripts"))
import check_pusht_native_device_repair_gpu_20261018 as fixture


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_loop(core):
    tree = ast.parse((BUNDLE / "effective_train_native.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.While))
    updates, saved = [], []

    def fake_update(policy, optimizer, scheduler, ema, batch, completed, audit=False):
        assert not audit and completed >= 20000
        updates.append(completed)
        return .123, None

    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        folder = Path(temp)
        audit = folder / "startup_checks.json"
        audit.write_text('{"test": true}')
        namespace = {"completed": 20000, "target_updates": 84168,
                     "policy": None, "optimizer": None, "ema": None,
                     "scheduler": type("Scheduler", (), {"get_last_lr": lambda _: [1e-4]})(),
                     "averaged": None, "result": {}, "protocol_sha": "p", "plan_sha": "s",
                     "data": type("Data", (), {"batch": lambda _, ids: ids})(), "plan": core.BatchPlan(),
                     "update": fake_update, "capture": lambda *args: {"completed_updates": args[5]},
                     "atomic_checkpoint": lambda path, payload: saved.append(payload["completed_updates"]),
                     "save_json": lambda path, value: None, "sha": lambda path: "pinned",
                     "latest": folder / "latest.pt", "output": folder, "audit_path": audit,
                     "extension_path": BUNDLE / "extension_contract.json", "json": json,
                     "log": io.StringIO(), "started": 0., "time": type("Clock", (), {"monotonic": staticmethod(lambda: 1.)})}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[loop], type_ignores=[]), str(BUNDLE / "effective_train_native.py"), "exec"), namespace)
        assert updates == list(range(20000, 84168))
        assert namespace["completed"] == namespace["result"]["completed_updates"] == 84168
        assert saved == list(range(22000, 84001, 2000)) + [84168]
        assert len(namespace["log"].getvalue().splitlines()) == 64168
    return {"actual_generated_loop_start_index": updates[0], "actual_generated_loop_last_index": updates[-1],
            "new_updates_only": len(updates), "last_checkpoint": saved[-1], "checkpoint_count": len(saved)}


def check_wrapper():
    wrapper = load("native_extension_wrapper", BUNDLE / "run_extension.py")
    import runpy
    real_run_path = runpy.run_path
    original_argv, original_path = sys.argv[:], sys.path[:]
    original_core = sys.modules.get("training_core")
    calls = []
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
            root = Path(temp)
            source = ROOT / "reports/pusht_native_211309"
            target = root / "runs/pusht-native20k-211309"
            target.mkdir(parents=True)
            for path in source.glob("*.json"):
                shutil.copyfile(path, target / path.name)
            output = root / "runs/extension-binding"

            def bind_and_stop(path, **kwargs):
                ns = real_run_path(path, **kwargs)
                assert ns["restore"] is sys.modules["training_core"].restore
                assert "RESTORE_DEVICE_CHECKS" in ns["restore"].__globals__
                calls.append(True)
                ns["main"] = lambda: output.mkdir(parents=True)
                return ns

            sys.argv = [str(wrapper.__file__), "--root", str(root), "--output", str(output)]
            with patch("runpy.run_path", bind_and_stop):
                wrapper.main()
            assert (output / "extension_provenance.json").is_file()
            # An unrelated extension resume cannot skip the reference/state guards.
            bad = root / "runs/unrelated/latest.pt"
            bad.parent.mkdir(parents=True)
            bad.with_suffix(".json").write_text('{"extension_contract_sha256": "wrong"}')
            sys.argv = [str(wrapper.__file__), "--root", str(root), "--output", str(root / "runs/rejected"), "--resume", str(bad)]
            try:
                wrapper.main()
            except ValueError as exc:
                assert "different extension" in str(exc)
            else:
                raise AssertionError("Unrelated checkpoint accepted")
    finally:
        sys.argv, sys.path = original_argv, original_path
        if original_core is not None:
            sys.modules["training_core"] = original_core
    assert calls == [True]


def main():
    core = load("extension_core_test", BUNDLE / "training_core.py")
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    LinearNormalizer, Single = fixture.normalizer_classes()
    torch.manual_seed(42)
    policy = fixture.cpu_fixture.Policy()
    policy.obs_encoder = fixture.ConvFrontend()
    policy.mask_generator = fixture.DeviceMask()
    stats = LinearNormalizer()
    stats.fit({"action": np.random.RandomState(42).randn(100, 2).astype(np.float32),
               "agent_pos": np.random.RandomState(43).randn(100, 2).astype(np.float32)})
    stats["image"] = Single.create_identity()
    policy.normalizer = LinearNormalizer()
    policy.normalizer.load_state_dict(stats.state_dict())
    policy.cuda()
    averaged, ema = fixture.cpu_fixture.make_ema(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    factor = lambda n: n / 500 if n < 500 else .5 * (1 + np.cos(np.pi * (n - 500) / (512400 - 500)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    # Place small fixtures at the counter boundary; do not pretend to run20k updates.
    scheduler.last_epoch = 19999
    scheduler._last_lr = [1e-4 * factor(19999)]
    optimizer.param_groups[0]["lr"] = scheduler._last_lr[0]
    ema.optimization_step = 19999
    batch = {"obs": {"image": torch.randn(8, 2, 3, 8, 8), "agent_pos": torch.randn(8, 2, 2)},
             "action": torch.randn(8, 16, 2)}
    core.update(policy, optimizer, scheduler, ema, batch, 19999)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        path = Path(temp) / "latest.pt"
        core.atomic_checkpoint(path, core.capture(policy, averaged, optimizer, scheduler, ema, 20000, "p", "s"))
        uninterrupted_loss, _ = core.update(policy, optimizer, scheduler, ema, batch, 20000)
        expected = core.tree_digest(core.capture(policy, averaged, optimizer, scheduler, ema, 20001, "p", "s"))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert core.restore(payload, policy, averaged, optimizer, scheduler, ema, "p", "s") == 20000
        assert core.tree_digest(payload) == core.tree_digest(core.capture(policy, averaged, optimizer, scheduler, ema, 20000, "p", "s"))
        continued_loss, _ = core.update(policy, optimizer, scheduler, ema, batch, 20000, audit=True)
        assert uninterrupted_loss == continued_loss
        assert expected == core.tree_digest(core.capture(policy, averaged, optimizer, scheduler, ema, 20001, "p", "s"))
        assert scheduler.last_epoch == 20001 and ema.optimization_step == 20001
        assert scheduler.get_last_lr()[0] == 1e-4 * factor(20001) > 0
    loop = check_loop(core)
    check_wrapper()
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    protocol = json.loads((BUNDLE / "training_protocol.json").read_text())
    reference = json.loads((ROOT / "reports/pusht_native_211309/report.json").read_text())
    assert core.BatchPlan().digest(84168) == reference["sample_order_sha256"]
    assert hashlib.sha256((BUNDLE / "training_protocol.json").read_bytes()).hexdigest() == reference["protocol_sha256"]
    assert (BUNDLE / "training_core.py").read_bytes() == (ROOT / "scripts/cluster/pusht_native_device_repair_v1/training_core_repaired.py").read_bytes()
    result = {"small_real_cuda_continuation_passed": True, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "full_source_state_digest_matches_after_restore": True,
              "next_update_loss_and_full_state_match_uninterrupted": True,
              "lr_and_ema_counters_continue_without_reset": True,
              "sample_order_matches_211309": True, "base_protocol_and_repaired_core_byte_identical": True,
              "generated_runner_binds_repaired_core": True, "unrelated_resume_rejected": True,
              "generated_loop": loop, "python39_AST": True,
              "full_native_policy_or_training_dataset_executed": False,
              "development_or_final_scored": False, "actual_cluster_torch112_entry_executed_locally": False}
    (ROOT / "reports/pusht_native_extension_checks_20261019.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
