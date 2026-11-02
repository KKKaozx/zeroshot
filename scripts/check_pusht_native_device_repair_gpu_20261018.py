"""Small real CUDA regression using the author's actual normalizer/load logic.

No original large policy, CLIP, dataset, simulator or task scoring is executed.
"""
import ast
import contextlib
import copy
import importlib.util
import json
import io
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
from typing import Dict, Union
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/cluster/pusht_native_training_v1"))
import check_pusht_native_training_cpu_20261018 as cpu_fixture
import training_core as old


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalizer_classes():
    fixture = load_module("dict_tensor_fixture", ROOT / "scripts/fixtures/pusht_author_dict_of_tensor_mixin.py")
    source = ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/model/common/normalizer.py"
    tree = ast.parse(source.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

    def dict_apply(value, fn):
        return {k: dict_apply(v, fn) for k, v in value.items()} if isinstance(value, dict) else fn(value)

    namespace = {"torch": torch, "nn": nn, "np": np, "zarr": SimpleNamespace(Array=type("UnusedZarr", (), {})),
                 "Dict": Dict, "Union": Union, "unittest": unittest, "dict_apply": dict_apply,
                 "DictOfTensorMixin": fixture.DictOfTensorMixin}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["LinearNormalizer"], namespace["SingleFieldLinearNormalizer"]


class ConvFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3, padding=1)
        self.linear = nn.Linear(4, 8)

    def forward(self, obs):
        features = self.conv(obs["image"]).mean((-1, -2))
        return self.linear(torch.cat([features, obs["agent_pos"]], -1))


class DeviceMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.empty(0))

    def forward(self, shape):
        return torch.zeros(shape, dtype=torch.bool, device=self.anchor.device)


def main():
    assert torch.cuda.is_available(), "Real CUDA required; do not report a CPU fallback as GPU validation"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    repaired = load_module("device_repaired_core", ROOT / "scripts/cluster/pusht_native_device_repair_v1/training_core_repaired.py")
    LinearNormalizer, SingleNormalizer = normalizer_classes()
    torch.manual_seed(42)
    policy = cpu_fixture.Policy()
    policy.obs_encoder = ConvFrontend()
    policy.mask_generator = DeviceMask()
    normalizer = LinearNormalizer()
    normalizer.fit({"action": np.random.RandomState(42).randn(100, 2).astype(np.float32),
                    "agent_pos": np.random.RandomState(43).randn(100, 2).astype(np.float32)})
    normalizer["image"] = SingleNormalizer.create_identity()
    policy.normalizer = LinearNormalizer()
    policy.normalizer.load_state_dict(normalizer.state_dict())
    policy.cuda()
    averaged, ema = cpu_fixture.make_ema(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: min(n / 500, 1))
    batch = {"obs": {"image": torch.randn(8, 2, 3, 8, 8), "agent_pos": torch.randn(8, 2, 2)},
             "action": torch.randn(8, 16, 2)}
    old.update(policy, optimizer, scheduler, ema, batch, 0, audit=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        checkpoint = Path(temp) / "latest.pt"
        old.atomic_checkpoint(checkpoint, old.capture(policy, averaged, optimizer, scheduler, ema, 1, "p", "s"))
        loss_a, _ = old.update(policy, optimizer, scheduler, ema, batch, 1)
        expected = old.tree_digest(old.capture(policy, averaged, optimizer, scheduler, ema, 2, "p", "s"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        old.restore(payload, policy, averaged, optimizer, scheduler, ema, "p", "s")
        assert next(policy.obs_encoder.parameters()).is_cuda
        assert policy.normalizer.params_dict["image"]["scale"].device.type == "cpu"
        try:
            old.update(policy, optimizer, scheduler, ema, batch, 1)
        except RuntimeError as exc:
            error = str(exc)
            assert "Input type" in error and ("cuda" in error or "CUDA" in error)
        else:
            raise AssertionError("Old CPU-normalizer/CUDA-convolution failure was not reproduced")
        assert repaired.restore(payload, policy, averaged, optimizer, scheduler, ema, "p", "s") == 1
        assert policy.normalizer.params_dict["image"]["scale"].is_cuda
        assert averaged.normalizer.params_dict["image"]["scale"].is_cuda
        loss_b, _ = repaired.update(policy, optimizer, scheduler, ema, batch, 1)
        actual = repaired.tree_digest(repaired.capture(policy, averaged, optimizer, scheduler, ema, 2, "p", "s"))
        assert loss_a == loss_b and expected == actual
    # Run the real unmodified runner's import path, stop before its main work.
    wrapper = load_module("device_repair_wrapper_test", ROOT / "scripts/cluster/pusht_native_device_repair_v1/run_repaired.py")
    import runpy
    original_run_path = runpy.run_path
    argv, search_path = sys.argv[:], sys.path[:]
    prior_core = sys.modules.get("training_core")
    binding_checked = []
    try:
        with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temporary:
            fake_root = Path(temporary)
            original_bundle = fake_root / "pusht_native_training_v1"
            shutil.copytree(ROOT / "scripts/cluster/pusht_native_training_v1", original_bundle)
            output = fake_root / "runs/wrapper-binding"

            def bind_then_stop(path, **kwargs):
                namespace = original_run_path(path, **kwargs)
                assert namespace["restore"] is sys.modules["training_core"].restore
                assert "RESTORE_DEVICE_CHECKS" in namespace["restore"].__globals__
                binding_checked.append(True)
                namespace["main"] = lambda: output.mkdir(parents=True)
                return namespace

            sys.argv = [str(wrapper.__file__), "--root", str(fake_root), "--output", str(output)]
            with patch("runpy.run_path", bind_then_stop):
                wrapper.main()
            assert (output / "device_repair.json").is_file()
            with (original_bundle / "training_protocol.json").open("a") as stream:
                stream.write(" ")
            sys.argv = [str(wrapper.__file__), "--root", str(fake_root), "--output", str(fake_root / "runs/rejected")]
            try:
                wrapper.main()
            except ValueError as exc:
                assert "Original bundle changed" in str(exc)
            else:
                raise AssertionError("Modified original bundle accepted")
    finally:
        sys.argv, sys.path = argv, search_path
        if prior_core is not None:
            sys.modules["training_core"] = prior_core
    assert binding_checked == [True]
    for path in (ROOT / "scripts/cluster/pusht_native_device_repair_v1").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    record = {"gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
              "real_cuda_regression_passed": True,
              "author_normalizer_and_published_dict_load_behavior_used": True,
              "old_cpu_input_cuda_conv_error_reproduced": error,
              "raw_and_ema_normalizer_back_on_gpu": True,
              "next_training_update_loss_and_full_state_replay_exact": True,
              "actual_original_runner_binds_repaired_restore": True,
              "modified_original_bundle_rejected": True,
              "changed_production_functions": ["restore"], "original_bundle_or_protocol_modified": False,
              "same_cluster_torch112_tested_locally": False,
              "CLIP_or_dataset_or_simulation_executed": False,
              "development_or_final_scored": False}
    (ROOT / "reports/pusht_native_device_repair_gpu_checks_20261018.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
