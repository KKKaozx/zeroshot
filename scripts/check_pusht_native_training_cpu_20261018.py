"""CPU checks of real author loss method, RNG pairing and disk resume mechanics.

Uses small substitute modules, no real native weights, dataset, CUDA or env.
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_native_training_v1"
sys.path.insert(0, str(BUNDLE))
from training_core import (BatchPlan, atomic_checkpoint, audited_original_loss, capture,
                           planned_loss_rng, restore, seed_for, tree_digest, update)


def original_loss():
    source = ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiffusionUnetHybridImagePolicy")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_loss")
    namespace = {"torch": torch, "F": F,
                 "dict_apply": lambda d, fn: {k: fn(v) for k, v in d.items()},
                 "reduce": reduce}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["compute_loss"]


class Normalizer:
    def normalize(self, value):
        return value

    def __getitem__(self, key):
        return self


class Mask(nn.Module):
    def forward(self, shape):
        return torch.zeros(shape, dtype=torch.bool)


class Front(nn.Module):
    def __init__(self, extra_draws=0):
        super().__init__()
        self.linear = nn.Linear(6, 8)
        self.dropout = nn.Dropout(.1)
        self.extra_draws = extra_draws

    def forward(self, obs):
        if self.extra_draws:
            torch.rand(self.extra_draws)
        return self.dropout(self.linear(torch.cat([obs["image"], obs["agent_pos"]], -1)))


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(18, 2)

    def forward(self, trajectory, timesteps, local_cond=None, global_cond=None):
        context = global_cond[:, None].expand(-1, 16, -1)
        return self.linear(torch.cat([trajectory, context], -1)) + timesteps[:, None, None] / 1000


class Scheduler:
    class Config:
        num_train_timesteps = 100
        prediction_type = "epsilon"
    config = Config()

    def __init__(self):
        self.draws = []

    def add_noise(self, trajectory, noise, timesteps):
        self.draws.append((noise.detach().clone(), timesteps.detach().clone()))
        return .4 * trajectory + .6 * noise


class Policy(nn.Module):
    compute_loss = original_loss()

    def __init__(self, extra_draws=0):
        super().__init__()
        self.obs_encoder = Front(extra_draws)
        self.model = Backend()
        self.mask_generator = Mask()
        self.noise_scheduler = Scheduler()
        self.normalizer = Normalizer()
        self.obs_as_global_cond, self.n_obs_steps = True, 2


def make_ema(policy):
    path = ROOT / "baselines/diffusion_policy_data_reference/diffusion_policy/model/diffusion/ema_model.py"
    spec = importlib.util.spec_from_file_location("author_ema_native_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    averaged = copy.deepcopy(policy)
    return averaged, module.EMAModel(averaged, power=.75)


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    plan = BatchPlan()
    all_ids = np.concatenate([plan.batch(i) for i in range(168)])
    assert np.array_equal(np.sort(all_ids), np.arange(10726))
    assert len(plan.batch(167)) == 38 and len(plan.batch(168)) == 64
    assert BatchPlan().digest(84168) == BatchPlan().digest(84168)
    torch.manual_seed(42)
    policy = Policy()
    averaged, ema = make_ema(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 500 if step < 500 else
                                                 .5 * (1 + np.cos(np.pi * (step - 500) / (512400 - 500))))
    batch = {"obs": {"image": torch.randn(38, 2, 4), "agent_pos": torch.randn(38, 2, 2)},
             "action": torch.randn(38, 16, 2)}
    loss, check = update(policy, optimizer, scheduler, ema, batch, 0, audit=True)
    assert all(check.values())
    clone = copy.deepcopy(policy)
    clone.obs_encoder.extra_draws = 100
    clone.noise_scheduler.draws.clear()
    small = {"obs": {k: v[:8] for k, v in batch["obs"].items()}, "action": batch["action"][:8]}
    with planned_loss_rng(policy, 7, 0):
        policy.compute_loss(small)
    with planned_loss_rng(clone, 7, 0):
        clone.compute_loss(small)
    assert all(torch.equal(a, b) for a, b in zip(policy.noise_scheduler.draws[-1], clone.noise_scheduler.draws[-1]))
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as temp:
        checkpoint = Path(temp) / "latest.pt"
        atomic_checkpoint(checkpoint, capture(policy, averaged, optimizer, scheduler, ema, 1, "protocol", "plan"))
        loss_a, _ = update(policy, optimizer, scheduler, ema, batch, 1)
        expected = tree_digest(capture(policy, averaged, optimizer, scheduler, ema, 2, "protocol", "plan"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert restore(payload, policy, averaged, optimizer, scheduler, ema, "protocol", "plan") == 1
        loss_b, _ = update(policy, optimizer, scheduler, ema, batch, 1)
        assert loss_a == loss_b
        assert tree_digest(capture(policy, averaged, optimizer, scheduler, ema, 2, "protocol", "plan")) == expected
        try:
            restore(payload, policy, averaged, optimizer, scheduler, ema, "changed", "plan")
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched protocol accepted")
    # Accumulation weighting must include a short6-sample final microbatch.
    linear = nn.Linear(3, 2)
    x, y = torch.randn(38, 3), torch.randn(38, 2)
    F.mse_loss(linear(x), y).backward()
    full_grad = [p.grad.clone() for p in linear.parameters()]
    linear.zero_grad()
    for start in range(0, 38, 8):
        stop = min(start + 8, 38)
        (F.mse_loss(linear(x[start:stop]), y[start:stop]) * (stop - start) / 38).backward()
    for a, p in zip(full_grad, linear.parameters()):
        torch.testing.assert_close(a, p.grad, rtol=1e-5, atol=1e-7)
    # Actual first2 observation/pad path against independent generated rows.
    from train_native import TrainingData
    data = TrainingData.__new__(TrainingData)
    data.rows = np.asarray([[0, 15, 1, 16], [12, 21, 0, 9]])
    data.images = np.arange(21 * 96 * 96 * 3, dtype=np.float32).reshape(21, 96, 96, 3) % 256
    data.actions = np.arange(42, dtype=np.float32).reshape(21, 2)
    data.positions = data.actions.copy()
    data.lookup = np.arange(21)
    result = data.batch(np.asarray([0, 1]))
    assert np.array_equal(result["obs"]["agent_pos"][0].numpy(), data.positions[[0, 0]])
    assert np.array_equal(result["action"][1, -7:].numpy(), np.repeat(data.actions[20][None], 7, axis=0))
    from verify_training import validate_metrics
    protocol = {"development": {"start_seed": 5000000, "count": 20,
                               "competence_mean_score": .8, "competence_full_reward_fraction": .5}}
    rows = [{"environment_seed": 5000000 + i, "reward_series": [0., 1.], "max_overlap_reward": 1.,
             "full_reward": True, "steps": 2} for i in range(20)]
    development = {"episodes": rows, "mean_score": 1., "full_reward_fraction": 1.,
                   "reference_competence_gate_passed": True, "complete": True, "final_set_accessed": False}
    assert validate_metrics(development, protocol)["reference_competence_gate_passed"]
    broken = copy.deepcopy(development)
    broken["episodes"][0]["max_overlap_reward"] = .5
    try:
        validate_metrics(broken, protocol)
    except AssertionError:
        pass
    else:
        raise AssertionError("Reward/summary inconsistency accepted")
    record = {"cpu_checks_passed": True, "original_author_loss_AST_used": True,
              "original_author_EMA_used": True, "independent_diffusion_draws_despite_frontend_rng": True,
              "loss_and_representative_gradient_exact": True, "complete_disk_resume_exact": True,
              "changed_resume_protocol_rejected": True, "all_windows_once_per_epoch": True,
              "last_batch_38_and_microbatch_6_weighted": True, "padded_observation_and_action_indexing": True,
              "python39_AST": True, "actual_native_GPU_training_tested_locally": False,
              "development_reward_reduction_and_inconsistent_summary_rejection": True,
              "actual_development_or_final_scored": False}
    (ROOT / "reports/pusht_native_training_cpu_checks_20261018.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
