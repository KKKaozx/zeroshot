"""Bounded implementation checks: new token binding/resume/evaluator, no task run."""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_project_training_v1"
sys.path.insert(0, str(BUNDLE))
import training_core as core
from project_frontend import ProjectObservationEncoder
from token_binding import bind_tokens, observation_ids, project_update, FrozenCache
from train_project import validate_resume, snapshot
sys.path.insert(0, str(ROOT / "scripts"))
import check_pusht_native_training_cpu_20261018 as fixture


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 6)
        self.dropout = nn.Dropout(.1)

    def forward(self, visual, text, mask):
        return self.dropout(self.linear(visual[:, 0]))


class DirectEncoder(ProjectObservationEncoder):
    def forward(self, obs):
        self.set_tokens(obs["image"][:, None, :].detach())
        try:
            return super().forward(obs)
        finally:
            self.cached_tokens = None


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(134, 2)

    def forward(self, trajectory, timesteps, local_cond=None, global_cond=None):
        return self.linear(torch.cat((trajectory, global_cond[:, None].expand(-1, 16, -1)), -1))


class Policy(nn.Module):
    compute_loss = fixture.original_loss()

    def __init__(self, direct=False):
        super().__init__()
        encoder = DirectEncoder if direct else ProjectObservationEncoder
        self.obs_encoder = encoder(torch.ones(1, 2, 3), torch.ones(1, 2), (64, 65), adapter=Adapter(), width=6)
        self.model = Backend()
        self.mask_generator = fixture.Mask()
        self.noise_scheduler = fixture.Scheduler()
        self.normalizer = fixture.Normalizer()
        self.obs_as_global_cond, self.n_obs_steps = True, 2


def setup(direct=False):
    torch.manual_seed(123)
    policy = Policy(direct)
    averaged, ema = fixture.make_ema(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 500 if step < 500 else
                                                .5 * (1 + np.cos(np.pi * (step - 500) / (512400 - 500))))
    return policy, averaged, optimizer, scheduler, ema


def rejected(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return True
    raise AssertionError("Invalid input accepted")


def main():
    torch.set_num_threads(1)
    checks = {}
    rows = np.asarray([[0, 15, 1, 16], [12, 21, 0, 9], [4, 20, 0, 16]], dtype=np.int64)
    assert np.array_equal(observation_ids(rows), [[0, 0], [12, 13], [4, 5]])
    assert rejected(lambda: observation_ids(rows.astype(float)))
    assert rejected(lambda: observation_ids(np.asarray([[0, 15, 0, 16]])))
    from native_data import TrainingData
    data = TrainingData.__new__(TrainingData)
    data.rows = rows
    data.lookup = np.arange(21)
    data.images = np.arange(21 * 96 * 96 * 3, dtype=np.float32).reshape(21, 96, 96, 3) % 256
    data.actions = np.arange(42, dtype=np.float32).reshape(21, 2)
    data.positions = data.actions.copy()
    actual = data.batch(np.asarray([2, 0, 1, 0]))
    ids = observation_ids(rows[[2, 0, 1, 0]])
    assert np.array_equal(actual["obs"]["image"].numpy(), data.images[ids].transpose(0, 1, 4, 2, 3) / np.float32(255))
    cache = FrozenCache.__new__(FrozenCache)
    cache.rows, cache.ids = rows, np.arange(21)
    cache.visual = np.repeat(np.arange(21, dtype=np.float32)[:, None, None], 6, axis=-1)
    got = cache.tokens([2, 0, 1, 0]).numpy()
    assert np.array_equal(got[:, 0, 0], ids.reshape(-1))
    checks["padded_duplicate_unsorted_global_frame_lookup_matches_native_batch"] = True

    for size in (8, 38, 64):
        torch.manual_seed(2026)
        batch = {"obs": {"image": torch.randn(size, 2, 6), "agent_pos": torch.randn(size, 2, 2)},
                 "action": torch.randn(size, 16, 2)}
        tokens = batch["obs"]["image"].reshape(-1, 1, 6).detach()
        policy, averaged, optimizer, scheduler, ema = setup()
        reference, ref_avg, ref_opt, ref_lr, ref_ema = setup(direct=True)
        loss, audit = project_update(core, policy, optimizer, scheduler, ema, batch, tokens, 0, audit=True)
        direct_loss, _ = core.update(reference, ref_opt, ref_lr, ref_ema, batch, 0, audit=True)
        assert loss == direct_loss and all(audit.values())
        assert core.tree_digest(policy.state_dict()) == core.tree_digest(reference.state_dict())
        assert core.tree_digest(averaged.state_dict()) == core.tree_digest(ref_avg.state_dict())
        assert policy.obs_encoder.cached_tokens is None and len(policy.obs_encoder._forward_pre_hooks) == 0
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
            path = Path(temp) / "latest.pt"
            core.atomic_checkpoint(path, snapshot(policy, averaged, optimizer, scheduler, ema, 1, "p", "s", "c"))
            a, _ = project_update(core, policy, optimizer, scheduler, ema, batch, tokens, 1)
            expected = core.tree_digest(snapshot(policy, averaged, optimizer, scheduler, ema, 2, "p", "s", "c"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert core.restore(payload, policy, averaged, optimizer, scheduler, ema, "p", "s") == 1
            b, _ = project_update(core, policy, optimizer, scheduler, ema, batch, tokens, 1)
            assert a == b and expected == core.tree_digest(snapshot(policy, averaged, optimizer, scheduler, ema, 2, "p", "s", "c"))
    checks["real_author_loss_and_EMA_binding_equal_direct_input_batch8_38_64"] = True
    checks["all_new_frontend_and_backend_gradients_finite_nonzero"] = True
    checks["complete_disk_restore_next_update_exact"] = True
    policy = setup()[0]
    def fail_inside_binding():
        with bind_tokens(policy.obs_encoder, torch.zeros(16, 1, 6), 8):
            policy.obs_encoder({"agent_pos": torch.zeros(15, 2)})
    assert rejected(fail_inside_binding)
    assert policy.obs_encoder.cached_tokens is None and not policy.obs_encoder._forward_pre_hooks
    assert rejected(lambda: bind_once_trainable(policy))
    checks["bad_micro_count_and_trainable_features_rejected_hooks_cleaned"] = True

    sidecar = {"project_contract_sha256": "c", "protocol_sha256": "p", "sample_plan_sha256": "s",
               "completed_updates": 42084, "segment": 1, "startup_checks": {"disk_resume_replay_exact": True}}
    assert validate_resume(sidecar, "c", "p", "s", 84168, 2) == 42084
    for key, bad in (("project_contract_sha256", "native"), ("completed_updates", 42083),
                     ("completed_updates", 84168), ("startup_checks", None), ("sample_plan_sha256", "other")):
        altered = dict(sidecar, **{key: bad})
        assert rejected(lambda: validate_resume(altered, "c", "p", "s", 84168, 2))
    checks["native_checkpoint_wrong_plan_wrong_boundary_missing_audit_rejected"] = True

    # Execute the real continuation loop with cheap updates, not a second loop.
    runner_tree = ast.parse((BUNDLE / "train_project.py").read_text())
    main_node = next(n for n in runner_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main_node) if isinstance(n, ast.While))
    for begin, end in ((2, 42084), (42084, 84168)):
        updates, saved = [], []
        namespace = {"completed": begin, "target": end, "result": {}, "log": io.StringIO(),
                     "json": json, "started": 0., "time": type("Clock", (), {"monotonic": staticmethod(lambda: 1.)}),
                     "scheduler": type("LR", (), {"get_last_lr": lambda _: [1e-4]})()}
        def cheap_update(index, audit=False):
            assert not audit
            updates.append(index)
            return .123, None
        namespace["one_update"] = cheap_update
        namespace["save_checkpoint"] = lambda: saved.append(namespace["completed"])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[copy.deepcopy(loop)], type_ignores=[]), "real_continuation_loop", "exec"), namespace)
        assert updates == list(range(begin, end))
        assert namespace["completed"] == end
        assert saved[-1] == end and all(x % 2000 == 0 for x in saved[:-1])
        rows_logged = [json.loads(s)["completed_updates"] for s in namespace["log"].getvalue().splitlines()]
        assert rows_logged == list(range(begin + 1, end + 1))
    checks["actual_segment_loop_no_reset_skip_or_extra_updates_42084_84168"] = True

    original_path = ROOT / "scripts/cluster/pusht_native_extension_v1/effective_train_native.py"
    original = ast.parse(original_path.read_text())
    old = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    new = next(n for n in ast.parse((BUNDLE / "evaluate_project.py").read_text()).body if isinstance(n, ast.FunctionDef))
    class IgnoreLabels(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and (node.value.startswith("PUSHT_") or "Native cap" in node.value
                    or "Compare fixed endpoint" in node.value or "Project below reference" in node.value):
                return ast.copy_location(ast.Constant("label"), node)
            return node
    assert ast.dump(IgnoreLabels().visit(old)) == ast.dump(IgnoreLabels().visit(new))
    checks["endpoint_evaluator_native_AST_identical_except_labels"] = True
    contract = json.loads((BUNDLE / "project_contract.json").read_text())
    for name, relative in contract["reused_files_byte_identical"].items():
        assert (BUNDLE / name).read_bytes() == (ROOT / "scripts/cluster" / relative).read_bytes()
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    with zipfile.ZipFile(ROOT / "cluster_packages/pusht_project_training_v1.zip") as archive:
        assert archive.testzip() is None
        for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read("pusht_project_training_v1/" + name)).hexdigest() == digest
    checks["reused_sources_identical_python39_AST_and_archive_SHA"] = True
    checks.update(all_checks_passed=True, real_full_project_training_tested_locally=False,
                  new_resource_or_wiring_run=False, task_rollout_executed=False, cuda_used=False)
    (ROOT / "reports/pusht_project_training_checks_20261020.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def bind_once_trainable(policy):
    with bind_tokens(policy.obs_encoder, torch.zeros(16, 1, 6, requires_grad=True), 8):
        pass


if __name__ == "__main__":
    main()
