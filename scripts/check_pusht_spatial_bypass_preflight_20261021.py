"""CPU integration tests of new bypass/state additions, no actual policy training."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_bypass_preflight_v1"
sys.path.insert(0, str(BUNDLE))
import training_core as core
from spatial_bypass import SpatialBypassEncoder, attach, MODES, moment_weights
import bypass_state
from token_binding import project_update
from preflight_support import fit_statistics
sys.path.insert(0, str(ROOT / "scripts"))
import check_pusht_project_training_20261020 as fixture


def main():
    torch.set_num_threads(1)
    checks = []
    def check(name, flag):
        if not flag:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, call):
        try:
            call()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    stats = {mode: {"center": np.zeros((4, 6), np.float32), "scale": np.ones((4, 6), np.float32)} for mode in MODES}
    torch.manual_seed(112)
    visual = torch.randn(4, 257, 6)
    obs = {"agent_pos": torch.randn(4, 2)}
    for mode in MODES:
        base = fixture.setup()[0].obs_encoder.eval()
        base.set_tokens(visual)
        original = base(obs).detach().clone()
        base.cached_tokens = None
        old_parameter_ids = {k: id(p) for k, p in base.named_parameters()}
        rng_before = core.tree_digest(core.rng_state())
        wrapped = SpatialBypassEncoder(base, mode).eval()
        check(mode + "_constructor_preserves_rng", core.tree_digest(core.rng_state()) == rng_before)
        check(mode + "_preserves_original_parameter_objects", all(id(dict(wrapped.named_parameters())[k]) == v for k, v in old_parameter_ids.items()))
        wrapped.set_tokens(visual)
        actual = wrapped(obs)
        check(mode + "_zero_head_preserves_all66_outputs", torch.equal(original, actual.detach()))
        check(mode + "_state_columns_unchanged", torch.equal(actual[:, wrapped.state_columns], obs["agent_pos"]))
        actual[:, :64].square().mean().backward()
        check(mode + "_zero_head_receives_first_step_gradient", wrapped.bypass_head.weight.grad is not None and bool(wrapped.bypass_head.weight.grad.abs().max() > 0))
        before_features = wrapped.bypass_features(visual).clone()
        with torch.no_grad():
            wrapped.adapter.linear.weight.add_(4.)
        check(mode + "_spatial_source_precedes_adapter", torch.equal(before_features, wrapped.bypass_features(visual)))
        check(mode + "_fixed_normalization_preserves_amplitude", torch.allclose(wrapped.bypass_features(2 * visual), 2 * before_features, atol=1e-7, rtol=1e-6))
        reject(mode + "_reject_trainable_clip_tokens", lambda: wrapped.bypass_features(visual.clone().requires_grad_(True)))
        reject(mode + "_reject_wrong_patch_count", lambda: wrapped.bypass_features(visual[:, :256]))
        reject(mode + "_reject_repeated_augmentation", lambda: SpatialBypassEncoder(wrapped, mode))
    reject("reject_unknown_mode", lambda: SpatialBypassEncoder(fixture.setup()[0].obs_encoder, "other"))
    reject("reject_invalid_scale", lambda: SpatialBypassEncoder(fixture.setup()[0].obs_encoder, MODES[0], scale=torch.zeros(4, 6)))
    # A layout perturbation with identical patch means and identical CLS.
    tokens = torch.zeros(2, 257, 6)
    tokens[0, 1, 0], tokens[1, -1, 0] = 1., 1.
    for mode in MODES:
        encoder = SpatialBypassEncoder(fixture.setup()[0].obs_encoder, mode).eval()
        f = encoder.bypass_features(tokens)
        if mode == "global_control":
            check("global_control_invariant_to_layout_at_fixed_content", torch.allclose(f[0], f[1], atol=1e-8, rtol=0))
        else:
            check("spatial_candidate_distinguishes_patch_locations", not torch.allclose(f[0], f[1], atol=1e-5, rtol=0))
    # Verify the actual train-statistics helper against an independent NumPy calculation.
    cache = SimpleNamespace(visual=np.random.RandomState(42).randn(18, 257, 1024).astype(np.float32), ids=np.arange(18))
    with tempfile.TemporaryDirectory(prefix="bypass_stats_", dir=ROOT / "reports") as temp:
        fitted, evidence = fit_statistics(cache, Path(temp), device=torch.device("cpu"))
        w = moment_weights(torch.device("cpu"), torch.float32).numpy()
        expected = np.einsum("kc,ncd->nkd", w, cache.visual[:, 1:]).astype(np.float64)
        check("train_statistics_match_independent_moments", np.allclose(fitted[MODES[1]]["center"], expected.mean(0), atol=1e-6, rtol=1e-5)
              and np.allclose(fitted[MODES[1]]["scale"], expected.std(0), atol=1e-6, rtol=1e-5))
        check("global_control_repeats_matched_global_statistics", all(np.array_equal(fitted[MODES[0]][k], np.repeat(fitted[MODES[1]][k][:1], 4, 0)) for k in ("center", "scale")))
        check("statistics_helper_uses_no_labels_or_holdout", evidence["frames"] == 18 and not evidence["labels_used"] and not evidence["nontrain_frames_used"] and not evidence["per_frame_normalization"])
    # Continue an already-warmed toy policy with the pinned original loss and EMA.
    torch.manual_seed(27)
    size = 10  # exercises8+2 weighted microbatches
    batch = {"obs": {"image": torch.randn(size, 2, 6), "agent_pos": torch.randn(size, 2, 2)}, "action": torch.randn(size, 16, 2)}
    tokens = torch.randn(size * 2, 257, 6)
    policy, avg, opt, lr, ema = fixture.setup()
    for step in range(3):
        project_update(core, policy, opt, lr, ema, batch, tokens, step)
    origin = copy.deepcopy(core.capture(policy, avg, opt, lr, ema, 3, "base", "plan"))
    first_losses = []
    for mode in MODES:
        policy, avg, opt, lr, ema = fixture.setup()
        core.restore(origin, policy, avg, opt, lr, ema, "base", "plan")
        old_ids = [id(p) for p in opt.param_groups[0]["params"]]
        state_before = {id(p): core.tree_digest(v) for p, v in opt.state.items()}
        lr_before, ema_before = copy.deepcopy(lr.state_dict()), (ema.optimization_step, ema.decay)
        evidence = attach(policy, avg, opt, mode, stats)
        check(mode + "_preserve_warmed_adam_parameter_order", [id(p) for p in opt.param_groups[0]["params"][:-1]] == old_ids)
        check(mode + "_preserve_warmed_adam_moments", all(core.tree_digest(v) == state_before[id(p)] for p, v in opt.state.items()))
        check(mode + "_preserve_lr_and_ema_counter", lr.state_dict() == lr_before and (ema.optimization_step, ema.decay) == ema_before)
        check(mode + "_new_head_cold_adam_and_frozen_ema", policy.obs_encoder.bypass_head.weight not in opt.state and all(not p.requires_grad for p in avg.parameters()))
        loss, audit = project_update(core, policy, opt, lr, ema, batch, tokens, 3, audit=True)
        first_losses.append(loss)
        check(mode + "_pinned_loss_and_old_gradients_valid", all(audit.values()))
        check(mode + "_new_head_gradients_and_adam_step", bool(policy.obs_encoder.bypass_head.weight.grad.abs().max() > 0) and int(opt.state[policy.obs_encoder.bypass_head.weight]["step"]) == 1)
        check(mode + "_new_raw_and_ema_head_updated", bool(policy.obs_encoder.bypass_head.weight.abs().max() > 0) and bool(avg.obs_encoder.bypass_head.weight.abs().max() > 0))
        capture = lambda completed: bypass_state.capture(policy, avg, opt, lr, ema, completed, "base", "plan", "bypass", mode)
        with tempfile.TemporaryDirectory(prefix="bypass_resume_", dir=ROOT / "reports") as temp:
            path = Path(temp) / "checkpoint.pt"
            core.atomic_checkpoint(path, capture(4))
            loss_a, _ = project_update(core, policy, opt, lr, ema, batch, tokens, 4)
            digest = core.tree_digest(capture(5))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            other = MODES[1] if mode == MODES[0] else MODES[0]
            reject(mode + "_reject_cross_mode_checkpoint", lambda: bypass_state.restore(dict(payload, bypass_mode=other), policy, avg, opt, lr, ema, "base", "plan", "bypass", mode))
            reject(mode + "_reject_changed_protocol", lambda: bypass_state.restore(payload, policy, avg, opt, lr, ema, "base", "plan", "different", mode))
            completed = bypass_state.restore(payload, policy, avg, opt, lr, ema, "base", "plan", "bypass", mode)
            loss_b, _ = project_update(core, policy, opt, lr, ema, batch, tokens, completed)
            check(mode + "_complete_disk_resume_next_update_exact", completed == 4 and loss_a == loss_b and digest == core.tree_digest(capture(5)))
            check(mode + "_resume_keeps_original_ema_and_new_adam_counts", ema.optimization_step == 5 and int(opt.state[policy.obs_encoder.bypass_head.weight]["step"]) == 2)
        check(mode + "_token_binding_cleans_up", policy.obs_encoder.cached_tokens is None and not policy.obs_encoder._forward_pre_hooks)
    check("paired_first_loss_exact_at_zero_head", first_losses[0] == first_losses[1])
    for file in BUNDLE.glob("*.py"):
        ast.parse(file.read_text(encoding="utf-8"), feature_version=(3, 9))
    checks.append("all_bundle_python39_syntax")
    protocol = json.loads((BUNDLE / "protocol.json").read_text())
    for name, digest in protocol["reused_sources_sha256"].items():
        check("reused_byte_exact_" + name, hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest)
    check("no_formal_entry_or_final_access", not protocol["formal_entry_included"] and not protocol["automatic_formal_submission"] and not protocol["sealed_final_access_allowed"] and not (BUNDLE / "formal.sh").exists())
    archive = ROOT / "cluster_packages/pusht_spatial_bypass_preflight_v1.zip"
    with zipfile.ZipFile(archive) as z:
        check("archive_members_exact", all(z.read(BUNDLE.name + "/" + p.name) == p.read_bytes() for p in BUNDLE.iterdir() if p.is_file()))
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    for name in ("preflight.sh", "submit_once.sh"):
        subprocess.run([str(bash), "-n", str(BUNDLE / name)], check=True)
        checks.append("bash_syntax_" + name)
    report = {"all_passed": True, "checks": checks, "count": len(checks), "scope": "New bypass interface/statistics and full state integration on CPU fixtures with original loss/EMA",
              "actual_endpoint_loaded": False, "real_training_data_used": False, "actual_policy_rollouts": 0,
              "actual_clip_model_loaded": False, "cluster_job_submitted": False,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    (ROOT / "reports/pusht_spatial_bypass_preflight_checks_20261021.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
