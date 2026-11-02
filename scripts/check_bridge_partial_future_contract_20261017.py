"""Bounded CPU validation of NEW temporal-mask/data prototype; no training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import ast
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import struct
import hashlib
import time

import numpy as np
import torch
from bridge_partial_future_contract_v1 import (
    HORIZON, make_row, build_numeric_sample, masked_pose_mse, candidate_forward_noise,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/bridge_partial_future_contract_20261017"
BASE = ROOT / "cluster_packages/bridge_growth128_v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def epkey(e):
    return e["shard"], e["record_index"]


def load_frozen_helpers():
    spec = importlib.util.spec_from_file_location("frozen_bridge_data_contract", BASE / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protobuf_fields():
    path = ROOT / "scripts/inventory_bridge_additional_demonstrations_20261017.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("varint", "fields")]
    assert len(nodes) == 2
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["fields"]


def numeric_record(path, record_index, fields):
    # Read only the selected record; other records are skipped without decoding.
    with path.open("rb") as stream:
        for index in range(record_index + 1):
            header = stream.read(12)
            assert len(header) == 12
            length = struct.unpack("<Q", header[:8])[0]
            if index == record_index:
                assert length > 0
                payload = stream.read(length)
                assert len(payload) == length
            else:
                stream.seek(length, 1)
            assert len(stream.read(4)) == 4
    # Protobuf seek traversal skips image/language payloads. No image decoding.
    stream = io.BytesIO(payload)
    wanted = {"steps/observation/state", "steps/action"}
    found = {}
    for outer, a, b in fields(stream, len(payload)):
        if outer != 1:
            continue
        stream.seek(a)
        for entry, c, d in fields(stream, b):
            if entry != 1:
                continue
            stream.seek(c)
            name = span = None
            for field, e, f in fields(stream, d):
                if field == 1:
                    stream.seek(e); name = stream.read(f-e).decode()
                elif field == 2:
                    span = e, f
            if name in wanted:
                stream.seek(span[0])
                for feature_kind, g, h in fields(stream, span[1]):
                    assert feature_kind == 2  # Feature.float_list
                    stream.seek(g)
                    packed = []
                    for value_field, j, k in fields(stream, h):
                        assert value_field == 1 and (k-j) % 4 == 0
                        stream.seek(j); packed.append(stream.read(k-j))
                    found[name] = np.frombuffer(b"".join(packed), dtype="<f4").reshape(-1, 7).copy()
            stream.seek(d)
        stream.seek(b)
    assert set(found) == wanted
    return found["steps/observation/state"], found["steps/action"], hashlib.sha256(payload).hexdigest()


def rejection(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError("Invalid contract input was silently accepted")


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Preserve the original validation output"
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    active = read(ROOT / "configs/active_experiment_reference.json")
    for p, h in active["local_pinned_files"].items():
        assert sha(ROOT / p) == h
    protocol = read(ROOT / "cluster_packages/bridge_coverage_replication_v1/protocol.json")
    assert sha(BASE / "dataset.py") == protocol["inherited_sha256"]["bridge_growth128_v1/dataset.py"]
    helper = load_frozen_helpers()
    encode = helper.relative_pose_action
    euler = helper.euler_xyz_to_quaternion
    pool_path = ROOT / "training_cache/exports/bridge_expansion_training_v1/manifest.json"
    pool = read(pool_path)
    old_rows = read(ROOT / "cluster_packages/bridge_coverage_replication_v1/rows.json")["train"]
    episodes = {epkey(e): e for e in pool["partitions"]["train"]}
    old_key = lambda r: (r["shard"], r["record_index"], r["start_index"])
    retained_keys = set(map(old_key, old_rows))
    retained = [make_row(episodes[epkey(r)], r["start_index"]) for r in old_rows]
    added = []
    for e in pool["partitions"]["train"]:
        for t in range(0, e["steps"]-1, 4):
            row = make_row(e, t)
            if old_key(row) not in retained_keys:
                assert row["valid_future_length"] < 16
                added.append(row)
    candidate = retained + added
    assert len(retained) == 1836 and len(added) == 1558 and len(candidate) == 3394
    assert len(set(map(old_key, candidate))) == 3394
    assert all(r["valid_future_length"] == 16 for r in retained)
    assert [{k:r[k] for k in old_rows[0]} for r in retained] == old_rows
    protected = set(map(epkey, pool["partitions"]["validation"] + pool["reserved_test_identity_only"]))
    assert not set(map(epkey, candidate)) & protected
    # Every row's indices/mask are independently reconstructed arithmetically.
    for r in candidate:
        n = episodes[epkey(r)]["steps"]
        length = min(16, n-1-r["start_index"])
        assert sum(r["validity"]) == length == r["valid_future_length"]
        assert r["future_state_indices"] == [r["start_index"]+1+i for i in range(length)]
        assert r["gripper_command_indices"] == [r["start_index"]+i for i in range(length)]
        assert max(r["future_state_indices"]) < n and max(r["gripper_command_indices"]) < n-1
    # Hand-calculated short episode: fixed yaw90, x motion becomes local -y.
    state = np.zeros((6, 7), dtype=np.float32)
    state[:, 0] = np.arange(6) * .01
    state[:, 5] = np.pi / 2
    state[:, 6] = .25
    raw_command = np.array([1., .5, 0., 1., 0., np.nan], dtype=np.float32)
    commands = helper.bridge_reverse_scan_gripper(raw_command[:-1])
    assert np.array_equal(commands, [1, 0, 0, 1, 0])
    synthetic = build_numeric_sample(state, commands, 3, encode, euler)
    np.testing.assert_allclose(synthetic["target"][:2, :3], [[0, -.1, 0], [0, -.2, 0]], atol=1e-6)
    np.testing.assert_allclose(synthetic["target"][:2, 3:7], [[0,0,0,1]]*2, atol=1e-6)
    assert np.array_equal(synthetic["target"][:2, 7], [1, -1])
    assert np.isnan(synthetic["target"][2:]).all() and synthetic["valid"].sum() == 2
    changed_state = state.copy();changed_state[4:, :3] += .5
    changed = build_numeric_sample(changed_state, commands, 3, encode, euler)
    assert np.array_equal(changed["current_pose"], synthetic["current_pose"])
    assert np.array_equal(changed["current_gripper"], synthetic["current_gripper"])
    assert not np.array_equal(changed["target"][:2], synthetic["target"][:2])
    rejection(lambda: build_numeric_sample(state, commands, 5, encode, euler))
    rejection(lambda: build_numeric_sample(state, raw_command, 3, encode, euler))
    rejection(lambda: build_numeric_sample(state,
        helper.bridge_reverse_scan_gripper(np.full(5, .5)), 3, encode, euler))
    # Two real training records, chosen using metadata only, not model error.
    ordered = sorted(episodes.values(), key=lambda e: (e["steps"], e["shard"], e["record_index"]))
    selected = [ordered[0], ordered[-1]]
    real_checks = []
    parser = protobuf_fields()
    numeric_path = ROOT / "reports/bridge-demo806-fit-208207/reduced_u8192_train.npz"
    with np.load(numeric_path, allow_pickle=False) as saved:
        old_targets = saved["targets"][:1836].copy()
    original_at = {old_key(r):i for i,r in enumerate(old_rows)}
    for e in selected:
        path = pool_path.parent / "data" / e["shard"]
        assert sha(path) == pool["shard_sha256"][e["shard"]]
        states, actions, payload_hash = numeric_record(path, e["record_index"], parser)
        assert states.shape == actions.shape == (e["steps"], 7)
        resolved = helper.bridge_reverse_scan_gripper(actions[:-1, 6])
        assert np.isin(resolved, (0., 1.)).all()
        lengths = []
        max_delta = 0.
        for r in [x for x in candidate if epkey(x) == epkey(e)]:
            t = r["start_index"]
            sample = build_numeric_sample(states, resolved, t, encode, euler)
            length = r["valid_future_length"]
            assert np.array_equal(sample["valid"], r["validity"])
            assert np.isfinite(sample["target"][:length]).all()
            assert np.isnan(sample["target"][length:]).all()
            # Independent world-position reconstruction for NEW partial labels.
            roll, pitch, yaw = states[t, 3:6].astype(float)
            cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
            world = states[t, :3] + (rotation @ (sample["target"][:length, :3] * .1).T).T
            np.testing.assert_allclose(world, states[t+1:t+1+length, :3], atol=1e-6)
            np.testing.assert_array_equal(sample["target"][:length, 7], resolved[t:t+length]*2-1)
            if old_key(r) in original_at:
                # New builder identity only; do not rerun old model/geometry audit.
                delta = float(np.max(np.abs(sample["target"]-old_targets[original_at[old_key(r)]])))
                assert delta < 1e-6
                max_delta = max(max_delta, delta)
            lengths.append(length)
        # Also explicitly test last real transition, even if outside stride lattice.
        last = build_numeric_sample(states, resolved, len(states)-2, encode, euler)
        assert last["valid"].sum() == 1 and last["command_indices"].tolist() == [len(states)-2]
        real_checks.append(dict(shard=e["shard"], record_index=e["record_index"], steps=e["steps"],
            payload_sha256=payload_hash, windows_checked=len(lengths), valid_lengths=sorted(set(lengths)),
            old_full_target_max_difference=max_delta, final_dummy_action_never_used=True))
    # Objective weighting and NaN-safe gradients, with no optimization step.
    prediction = torch.ones((2,16,7), dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros((2,16), dtype=torch.bool);mask[0,:]=True;mask[1,0]=True
    with torch.no_grad():prediction[1] = 3.
    target = target.masked_fill(~mask[...,None], float("nan"))
    loss = masked_pose_mse(prediction, target, mask)
    assert loss.item() == 5.
    loss.backward()
    assert torch.count_nonzero(prediction.grad[~mask]) == 0
    assert torch.all(prediction.grad[mask] != 0)
    full = torch.ones((2,16), dtype=torch.bool)
    a = torch.arange(224, dtype=torch.float64).reshape(2,16,7) / 32.
    b = a.square() / 10
    full_delta = float(abs(masked_pose_mse(a,b,full)-torch.nn.functional.mse_loss(a,b)))
    assert full_delta < 1e-12
    rejection(lambda: masked_pose_mse(a,b,torch.zeros_like(full)))
    hole=full.clone();hole[0,2]=False
    rejection(lambda: masked_pose_mse(a,b,hole))
    # Candidate q(x_t|x0) compatibility only: full masks reproduce old arithmetic.
    noise = torch.linspace(-1,1,224,dtype=torch.float64).reshape(2,16,7)
    alpha = torch.tensor([.75,.25],dtype=torch.float64).reshape(2,1,1)
    noisy = candidate_forward_noise(a,full,noise,alpha)
    original = alpha.sqrt()*a+(1-alpha).sqrt()*noise
    assert torch.equal(noisy,original)
    poisoned = target.clone();poisoned[~mask] = 1e30
    assert torch.equal(candidate_forward_noise(target,mask,noise,alpha),
                       candidate_forward_noise(poisoned,mask,noise,alpha))
    assert torch.isfinite(candidate_forward_noise(target,mask,noise,alpha)).all()
    # Counterexample: zero gradient for invalid OUTPUT is not zero influence of
    # invalid INPUT. Fixed 3-tap convolution, no policy/model/checkpoint loaded.
    x = torch.zeros((1,1,16),dtype=torch.float64,requires_grad=True)
    with torch.no_grad():x[0,0,2]=1.
    kernel = torch.ones((1,1,3),dtype=torch.float64)
    y = torch.nn.functional.conv1d(x,kernel,padding=1)
    valid_loss = y[0,0,:2].square().mean()
    valid_loss.backward()
    invalid_input_gradient = float(x.grad[0,0,2])
    assert invalid_input_gradient > 0 and float(valid_loss) == .5
    r = dict(data_contract_verified=True, training_integration_ready=False,
        reason="Temporal loss mask does not isolate invalid noisy input slots. No accepted reverse-process/mask-conditioning/execution-length contract yet.",
        GPU_or_training_authorized=False, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        project_model_loaded=False, checkpoints_loaded=False, neural_training_updates=0,
        policy_predictions_generated=False, images_decoded=False, raw_training_records_parsed=2,
        evaluation_or_test_numeric_records_read=0, baseline_207558_unchanged=True,
        counts=dict(retained_full_windows=len(retained), added_partial_windows=len(added), candidate_total=len(candidate)),
        partial_length_histogram=dict(sorted(Counter(r["valid_future_length"] for r in added).items())),
        checks=dict(all_candidate_index_boundaries=True, retained1836_index_rows_unchanged=True,
            no_evaluation_episode_overlap=True, synthetic_hand_calculation=True,
            future_numeric_values_do_not_enter_current_state_fields=True,
            final_state_without_next_observation_rejected=True, dummy_last_command_excluded=True,
            unresolved_gripper_not_silently_thresholded=True, masked_loss_equal_window_weighting=True,
            invalid_output_gradient_zero=True, full_mask_mse_max_difference=full_delta,
            full_mask_forward_noise_exact=True, canonical_auxiliary_fill_ignores_invalid_sentinels=True,
            cross_slot_counterexample_invalid_input_gradient=invalid_input_gradient),
        real_training_records=real_checks,
        loss_definition="mean over windows of sum(valid pose squared errors)/(7*valid length). Separate gripper loss not implemented/integrated.",
        storage_definition="NaN in non-observed target slots; valid prefix only. Finite canonical zeroXYZ/identityQuaternion auxiliary values used ONLY in candidate forward-noise numerical probe.",
        nonclaims=["No tail values are claimed to be physical holding observations.",
            "Two-record checks do not validate all1558 new real samples or their image/language loading.",
            "Full-mask arithmetic identity does not establish checkpoint replay of a new training implementation.",
            "Counterexample is a mathematical coupling check, not a measured defect/performance result of207558.",
            "No demonstrated generalization or grasp improvement; no GPU launcher provided."],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            Path(__file__), ROOT/"scripts/bridge_partial_future_contract_v1.py", BASE/"dataset.py",
            BASE/"diffusion_decoder.py", pool_path,
            ROOT/"cluster_packages/bridge_coverage_replication_v1/rows.json",
            ROOT/"scripts/inventory_bridge_additional_demonstrations_20261017.py"]},
        elapsed_seconds=time.monotonic()-started)
    OUT.mkdir()
    (OUT/"candidate_rows.json").write_text(json.dumps(dict(train=candidate),indent=2)+"\n",encoding="utf-8")
    r["candidate_rows_sha256"]=sha(OUT/"candidate_rows.json")
    (OUT/"report.json").write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(r,indent=2))


if __name__ == "__main__":
    main()
