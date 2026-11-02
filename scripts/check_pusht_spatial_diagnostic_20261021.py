"""Synthetic checks of the NEW diagnostic only; no data, checkpoint or policy run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_spatial_diagnostic_v1"
sys.path.insert(0, str(BUNDLE))
from probe_math import choose_frames, labels_from_state, errors, fit_readout, summarize, paired_interval
from extract_features import spatial_moments, project_features
from project_frontend import ProjectObservationEncoder, adapter_class
from resource_probe import state_hash
from verify_output import validate


def main():
    checks = []
    def check(name, flag):
        if not flag:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, f):
        try:
            f()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    ids, eps, roles = choose_frames(np.array([24, 54, 90]), [0, 2], [1], 12)
    check("whole_episode_split_no_shared_frames", not set(ids[roles == "fit"]) & set(ids[roles == "heldout"]))
    check("uniform_temporal_selection_including_endpoints", ids[0] == 0 and ids[11] == 23 and ids[12] == 54 and ids[23] == 89)
    rejects("reject_shared_episode", lambda: choose_frames([24, 54], [0], [0], 12))
    rejects("reject_duplicate_episode", lambda: choose_frames([24, 54], [0, 0], [1], 12))
    rejects("reject_out_of_range_episode", lambda: choose_frames([24, 54], [0], [2], 12))
    rejects("reject_short_episode", lambda: choose_frames([5, 54], [0], [1], 12))
    rejects("reject_float_boundaries", lambda: choose_frames([24., 54.], [0], [1], 12))
    state = np.array([[0., 0., 128., 256., np.deg2rad(179)], [512., 512., 256., 0., np.deg2rad(-179)]])
    y = labels_from_state(state)
    check("same_frame_xy512_label_roundtrip", np.array_equal(y[:, :4] * 512, state[:, :4]))
    p = y.copy()
    p[0, 4:] = y[1, 4:]
    check("circular_angle_wrap_is_two_degrees", abs(errors(p, y)[0, 2] - 2) < 1e-10)
    check("identity_errors_zero", np.max(errors(y, y)) == 0)
    rejects("reject_bad_state_width", lambda: labels_from_state(np.ones((4, 6))))
    rng = np.random.RandomState(10)
    x = rng.randn(160, 8)
    target = x @ rng.randn(8, 6)
    fit = np.arange(160) < 120
    pred, negative, model = fit_readout(x, target, fit, 1e-7, 12)
    check("linear_signal_readable_on_unseen_rows", np.mean((pred[~fit] - target[~fit]) ** 2) < 1e-8)
    check("shuffled_label_control_breaks_signal", np.mean((negative[~fit] - target[~fit]) ** 2) > .1)
    changed_x, changed_y = x.copy(), target.copy()
    changed_x[~fit] += 1e6
    changed_y[~fit] += 1e8
    _, _, changed = fit_readout(changed_x, changed_y, fit, 1e-7, 12)
    check("heldout_features_and_labels_cannot_change_fit", all(np.array_equal(model[k], changed[k]) for k in model))
    z = (x - model["center"]) / model["scale"] / np.sqrt(8)
    design = np.c_[z[fit], np.ones(fit.sum())]
    direct = np.linalg.lstsq(design, target[fit], rcond=None)[0]
    check("dual_ridge_matches_unregularized_limit", np.max(np.abs(pred - np.c_[z, np.ones(len(z))] @ direct)) < 1e-4)
    zero_features = np.ones((160, 3))
    constant, _, _ = fit_readout(zero_features, target, fit, .001, 12)
    check("constant_features_return_train_mean", np.allclose(constant, target[fit].mean(0)))
    rejects("reject_no_holdout", lambda: fit_readout(x, target, np.ones(160, bool), .001, 12))
    rejects("reject_zero_regularization", lambda: fit_readout(x, target, fit, 0., 12))
    # Spatial moments distinguish patch positions even when global means match.
    tokens = torch.zeros(2, 257, 16)
    tokens[0, 1, 0], tokens[1, -1, 0] = 1., 1.
    check("spatial_coordinates_preserved_by_probe", not torch.equal(spatial_moments(tokens)[0], spatial_moments(tokens)[1]))
    adapter = adapter_class()(num_layers=8, embed_dim=16, text_dim=8, attention_dim=16, num_heads=4,
                              dropout=.1, pooling="cls_patch_mean")
    torch.manual_seed(19)
    encoder = ProjectObservationEncoder(torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.int64), (64, 65), adapter=adapter, width=16)
    encoder.requires_grad_(False).eval()
    tokens = torch.randn(2, 257, 16)
    before = state_hash(encoder)
    with torch.no_grad():
        encoder.set_tokens(tokens)
        expected = encoder({"agent_pos": torch.zeros(2, 2)})[:, :64].numpy()
        encoder.cached_tokens = None
    found = project_features(encoder, tokens)
    check("passive_hooks_preserve_original_forward", np.array_equal(found["projection64"], expected))
    check("all_eight_expected_representation_stages", len(found) == 8)
    check("hooks_cleaned_and_endpoint_unchanged", state_hash(encoder) == before and encoder.cached_tokens is None and not encoder._forward_hooks)
    check("no_frozen_parameter_gradient", all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
    encoder.train()
    rejects("reject_train_mode_diagnostic", lambda: project_features(encoder, tokens))
    synthetic_labels = labels_from_state(np.tile(state, (4, 1)))
    one_ep = np.repeat([1, 2], 4)
    stats = summarize(synthetic_labels, synthetic_labels, one_ep, np.ones(8, bool))
    interval = paired_interval(stats, stats)
    check("paired_episode_interval_zero_for_identity", all(v["mean_error_delta_b_minus_a"] == 0 and v["episode_bootstrap_95"] == [0, 0] for v in interval.values()))
    protocol = json.loads((BUNDLE / "protocol.json").read_text())
    check("fixed_real_split_90_and40_disjoint", len(protocol["split"]["fit_episode_ids"]) == 90 and len(protocol["split"]["heldout_episode_ids"]) == 40 and not set(protocol["split"]["fit_episode_ids"]) & set(protocol["split"]["heldout_episode_ids"]))
    for name, digest in protocol["reused_sources_sha256"].items():
        check("reused_unchanged_" + name, hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest)
    # Exercise the artifact verifier with an explicitly synthetic output, including tampering.
    with tempfile.TemporaryDirectory(prefix="spatial_synthetic_", dir=ROOT / "reports") as tmp:
        tmp = Path(tmp)
        fake_ids, fake_eps, fake_roles = choose_frames(np.arange(1, 207) * 12,
            protocol["split"]["fit_episode_ids"], protocol["split"]["heldout_episode_ids"], 12)
        fake_fit = fake_roles == "fit"
        fake_state = rng.rand(1560, 5).astype(np.float32)
        fake_state[:, :4] *= 512
        fake_state[:, 4] *= 2 * np.pi
        fake_labels = labels_from_state(fake_state)
        mean = np.repeat(fake_labels[fake_fit].mean(0)[None], len(fake_fit), axis=0)
        fake_manifest = {"frame_ids": fake_ids.tolist(), "episode_ids": fake_eps.tolist(), "roles": fake_roles.tolist()}
        (tmp / "frame_manifest.json").write_text(json.dumps(fake_manifest))
        fake_arrays = {"frame_ids": fake_ids, "episode_ids": fake_eps, "fit_mask": fake_fit,
                       "source_state": fake_state, "labels": fake_labels, "train_mean": mean}
        fake_records = {}
        for stage in protocol["readout_stages"] + ["train_mean"]:
            value = mean if stage == "train_mean" else fake_labels.copy()
            fake_arrays[stage] = value
            fake_records[stage] = {role: summarize(value, fake_labels, fake_eps, mask)
                                  for role, mask in (("fit", fake_fit), ("heldout", ~fake_fit))}
            if stage != "train_mean":
                fake_arrays[stage + "_shuffled"] = mean
                fake_records[stage]["shuffled_heldout"] = summarize(mean, fake_labels, fake_eps, ~fake_fit)
        fake_report = {"diagnostic_complete": True, "errors": [], "policy_training_updates": 0,
                       "policy_rollouts": 0, "sealed_final_accessed": False, "policy_development_scores_used": False,
                       "policy_or_clip_weights_updated": False, "diagnostic_heads_fitted": True,
                       "diagnostic_readout_hyperparameter_search": False,
                       "protocol_sha256": hashlib.sha256((BUNDLE / "protocol.json").read_bytes()).hexdigest(),
                       "project_frozen": {"unchanged": True, "clip_worker": {"all_frozen": True, "weights_unchanged": True}},
                       "native_frozen": {"unchanged": True}, "readouts": fake_records,
                       "inputs": {"manifest_sha256": hashlib.sha256((tmp / "frame_manifest.json").read_bytes()).hexdigest()},
                       "comparisons": {a + "__to__" + b: paired_interval(fake_records[a]["heldout"], fake_records[b]["heldout"])
                                       for a, b in protocol["paired_comparisons"]}}
        (tmp / "report.json").write_text(json.dumps(fake_report))
        np.savez(tmp / "predictions.npz", **fake_arrays)
        check("saved_output_roundtrip_verifier", validate(tmp, BUNDLE / "protocol.json")["output_verified"])
        fake_arrays["projection64"][0, 2] += .1
        np.savez(tmp / "predictions.npz", **fake_arrays)
        rejects("reject_altered_saved_prediction", lambda: validate(tmp, BUNDLE / "protocol.json"))
        fake_arrays["projection64"][0, 2] -= .1
        fake_arrays["source_state"][0, 0] += 1
        np.savez(tmp / "predictions.npz", **fake_arrays)
        rejects("reject_altered_saved_source_label", lambda: validate(tmp, BUNDLE / "protocol.json"))
    with zipfile.ZipFile(ROOT / "cluster_packages/pusht_spatial_diagnostic_v1.zip") as archive:
        check("archive_members_byte_identical", all(archive.read(BUNDLE.name + "/" + p.name) == p.read_bytes()
                                                    for p in BUNDLE.iterdir() if p.is_file()))
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    for name in ("diagnostic.sh", "submit_once.sh"):
        subprocess.run([str(bash), "-n", str(BUNDLE / name)], check=True)
        checks.append("bash_syntax_" + name)
    result = {"all_passed": True, "checks": checks, "count": len(checks),
              "scope": "Synthetic readout math/split/leakage and passive-hook checks only",
              "real_cluster_execution": False, "real_checkpoint_loaded": False,
              "policy_training_updates": 0, "policy_rollouts": 0,
              "archive_sha256": hashlib.sha256((ROOT / "cluster_packages/pusht_spatial_diagnostic_v1.zip").read_bytes()).hexdigest()}
    (ROOT / "reports/pusht_spatial_diagnostic_checks_20261021.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
