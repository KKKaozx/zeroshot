"""Fixed 32-window batch-composition control; native DDPM and held-out evaluation.

Pose-only diagnostic, not grasp success. Reuses production modules without edits.
All budgets and rows come from a frozen manifest. No packages/assets downloaded.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import UnifiedRobotDataset
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest, evaluate, score, seed
from evaluate_bridge_endpoint_metrics import pose_errors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def schedule(arm):
    batches = []
    for cycle in range(512 if arm == "mixed" else 64):
        order = torch.randperm(32, generator=torch.Generator().manual_seed(440001 + cycle))
        if arm == "mixed":
            batches.extend(order.reshape(4, 8))
        else:
            batches.extend(torch.full((8,), int(i), dtype=torch.long) for i in order)
    result = torch.stack(batches)
    assert result.shape == (2048, 8)
    assert torch.equal(torch.bincount(result.flatten(), minlength=32), torch.full((32,), 512))
    assert all(len(set(b.tolist())) == (8 if arm == "mixed" else 1) for b in result)
    return result


def metrics(pred, target, rows):
    """Keep window-weighted and episode-weighted errors separate."""
    p, r = zip(*(pose_errors(draw, target) for draw in pred))
    p, r = np.stack(p), np.stack(r)
    identity = np.zeros_like(target)
    identity[..., 6] = 1
    sp, sr = pose_errors(identity, target)
    episodes = {}
    for i, row in enumerate(rows):
        episodes.setdefault(row["shard"] + "::" + str(row["record_index"]), []).append(i)
    per_episode = []
    for name, indices in episodes.items():
        per_episode.append(dict(episode=name, windows=len(indices), task=rows[indices[0]]["task"],
            path_position_cm=float(p[:, indices].mean()), path_rotation_deg=float(r[:, indices].mean()),
            endpoint_position_cm=float(p[:, indices, -1].mean()), endpoint_rotation_deg=float(r[:, indices, -1].mean()),
            static_path_position_cm=float(sp[indices].mean()), static_path_rotation_deg=float(sr[indices].mean()),
            static_endpoint_position_cm=float(sp[indices, -1].mean()), static_endpoint_rotation_deg=float(sr[indices, -1].mean())))
    return dict(windows=len(rows), episodes=len(episodes), action_targets=len(rows)*16,
        path_position_cm=float(p.mean()), path_rotation_deg=float(r.mean()),
        endpoint_position_cm=float(p[..., -1].mean()), endpoint_rotation_deg=float(r[..., -1].mean()),
        static_path_position_cm=float(sp.mean()), static_path_rotation_deg=float(sr.mean()),
        static_endpoint_position_cm=float(sp[:, -1].mean()), static_endpoint_rotation_deg=float(sr[:, -1].mean()),
        seed_path_position_cm=p.mean(axis=(1, 2)).tolist(), seed_path_rotation_deg=r.mean(axis=(1, 2)).tolist(),
        episode_mean_path_position_cm=float(np.mean([e["path_position_cm"] for e in per_episode])),
        episode_mean_path_rotation_deg=float(np.mean([e["path_rotation_deg"] for e in per_episode])),
        per_episode=per_episode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", choices=["preflight", "fit"], default="preflight")
    ap.add_argument("--preflight-reference", type=Path)
    ap.add_argument("--data-reference", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    code = Path(__file__).resolve().parent
    sources = {name: sha(code/name) for name in manifest["source_sha256"]}
    report = dict(stage=args.stage, execution_checks_passed=False, manifest_sha256=sha(args.manifest),
        source_sha256=sources, downloads=False, simulation_executed=False, gripper_trained=False,
        checkpoints_saved=False, reserved_test_targets_read=False, runs=[])
    (args.output/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    cfg = copy.deepcopy(manifest["config"])
    cfg["model"]["name"] = str(args.clip)
    (args.output/"plan.json").write_text(json.dumps(dict(manifest=manifest, runtime_config=cfg, stage=args.stage), indent=2)+"\n")
    try:
        assert sources == manifest["source_sha256"], "Source changed after manifest preparation"
        assert torch.cuda.is_available(), "Run in a GPU allocation"
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        train_rows, dev_rows = manifest["train"], manifest["development"]
        rows = train_rows + dev_rows
        identity = lambda row: (row["shard"], row["record_index"])
        assert len(train_rows) == 32 and len(dev_rows) == 40
        assert len(set(map(identity, train_rows))) == 16 and len(set(map(identity, dev_rows))) == 10
        assert not set(map(identity, train_rows)) & set(map(identity, dev_rows))
        assert all(row["partition"] == "train" for row in train_rows)
        assert all(row["partition"] == "validation" for row in dev_rows)
        batches = {arm: schedule(arm) for arm in manifest["arms"]}
        assert {arm: tensor_sha(b) for arm, b in batches.items()} == manifest["schedule_sha256"]
        reader = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        reader.chunk_size = 16
        reader.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
        reader.bridge_current_gripper = "continuous"
        items, checks = [], []
        for i, row in enumerate(rows):
            path = args.data_root/row["shard"]
            item = reader._get_tfrecord_bridge_state_action(dict(file_path=str(path),
                record_index=row["record_index"], start_index=row["start_index"]))
            assert " ".join(item[0].lower().split()) == " ".join(row["task"].lower().split())
            assert torch.isfinite(item[1]).all() and torch.isfinite(item[3]).all()
            check = dict(image_sha256=tensor_sha(item[1]), target_sha256=tensor_sha(item[3]), current_sha256=tensor_sha(item[2]))
            if i < 32:
                assert check["image_sha256"] == manifest["retained_train_checks"][i]["image_sha256"]
                assert check["target_sha256"] == manifest["retained_train_checks"][i]["target_sha256"]
            checks.append(check)
            items.append(item)
        report["data_checks"] = checks
        if args.data_reference:
            assert checks == json.loads(args.data_reference.read_text())["data_checks"], "Local/cluster tensors differ"
        previous = None
        if args.stage == "fit":
            assert args.preflight_reference, "A successful same-package cluster preflight is required"
            previous = json.loads((args.preflight_reference/"report.json").read_text())
            assert previous["stage"] == "preflight" and previous["execution_checks_passed"]
            assert previous["manifest_sha256"] == report["manifest_sha256"] and previous["source_sha256"] == sources
            assert previous["data_checks"] == checks
        targets = torch.stack([v[3] for v in items]).cuda()
        current = torch.stack([v[2] for v in items]).cuda()
        updates = 8 if args.stage == "preflight" else 2048
        init_seeds = [42] if args.stage == "preflight" else manifest["initialization_seeds"]
        report.update(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, clip_runtime_path=str(args.clip))
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        for init in init_seeds:
            seed(init)
            model = RobotAdapterModel(cfg).cuda().eval()
            texts = tokenizer([v[0] for v in items], padding=True, truncation=True, max_length=77, return_tensors="pt")
            with torch.no_grad():
                language = torch.cat([model.text_encoder(**{k:v[i:i+16].cuda() for k,v in texts.items()}).last_hidden_state
                    for i in range(0, len(items), 16)]).detach()
                visual = torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state
                    for i in range(0, len(items), 2)]).detach()
            mask = texts["attention_mask"].cuda()
            model.vision_encoder.cpu()
            model.text_encoder.cpu()
            torch.cuda.empty_cache()
            clip_hash = [digest(model.vision_encoder), digest(model.text_encoder)]
            model.history_projection = torch.nn.Linear(1024, 1024, bias=False).cuda()
            torch.nn.init.zeros_(model.history_projection.weight)
            prefixes = ("adapter.", "diffusion_decoder.", "history_projection.")
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(prefixes))
            initial = {name:v.detach().cpu().clone() for name,v in model.state_dict().items()
                if not name.startswith(("vision_encoder.", "text_encoder."))}
            initial_hash = hashlib.sha256("".join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
            paired_initial = None
            for arm in manifest["arms"]:
                model.load_state_dict(initial, strict=False)
                assert all(torch.equal(model.state_dict()[n].cpu(), v) for n,v in initial.items())
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                def context(indices):
                    now = model.adapter(visual[indices], language[indices], mask[indices])
                    repeated = model.adapter(visual[indices], language[indices], mask[indices])
                    return now + model.history_projection(repeated-now)
                def check(prefix):
                    model.eval()
                    split_results = {}
                    predictions = []
                    with torch.no_grad():
                        for split, indices in [("train", list(range(32))), ("development", list(range(32,72)))]:
                            if args.stage == "preflight":
                                indices = indices[:4]
                            contexts = torch.cat([context(torch.tensor(indices[i:i+2], device="cuda"))
                                for i in range(0, len(indices), 2)])
                            pred, _ = evaluate(model, contexts, current[indices], targets[indices].cpu().numpy(), manifest["sampling_seeds"], batch=2)
                            split_results[split] = metrics(pred, targets[indices].cpu().numpy(), [rows[i] for i in indices])
                            if split == "train":
                                split_results[split]["fit_gate"] = score(pred, targets[indices].cpu().numpy())
                            predictions.append(pred)
                            if prefix:
                                np.savez_compressed(args.output/f"seed{init}_{arm}_{split}.npz", predictions=pred,
                                    targets=targets[indices].cpu().numpy(), row_indices=np.asarray(indices))
                    return split_results, predictions
                initial_metrics, untrained_predictions = check(None) if args.stage == "preflight" else (None, None)
                if args.stage == "preflight":
                    if paired_initial is None:
                        paired_initial = untrained_predictions
                    else:
                        assert all(np.array_equal(a,b) for a,b in zip(paired_initial, untrained_predictions))
                first_grad = None
                losses, times = [], []
                torch.cuda.reset_peak_memory_stats()
                for update in range(1, updates+1):
                    model.train()
                    indices = batches[arm][update-1].cuda()
                    seed(430000+update)
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    ctx = context(indices)
                    seed(420000+update)
                    prediction, truth, _ = model.diffusion_loss(targets[indices], ctx, current[indices])
                    loss = torch.nn.functional.mse_loss(prediction, truth)
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if first_grad is None:
                        first_grad = {p:any(v.grad is not None and bool(torch.any(v.grad != 0))
                            for n,v in model.named_parameters() if n.startswith(p)) for p in prefixes}
                        assert all(first_grad.values())
                        assert all(v.grad is None for v in model.vision_encoder.parameters())
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append(time.monotonic()-tick)
                    losses.append(float(loss.detach()))
                    if update == 1 or update % 256 == 0:
                        print("BATCH_UPDATE", init, arm, update, "/", updates, "loss", losses[-1], flush=True)
                final, _ = check("final")
                changed = {p:any(not torch.equal(model.state_dict()[n].cpu(), v)
                    for n,v in initial.items() if n.startswith(p)) for p in prefixes}
                assert all(changed.values())
                assert all(torch.equal(model.state_dict()[n].cpu(), v) for n,v in initial.items() if not n.startswith(prefixes))
                row = dict(initialization_seed=init, arm=arm, initial_shared_sha256=initial_hash,
                    updates=updates, actual_draw_slots_per_window=torch.bincount(batches[arm][:updates].flatten(), minlength=32).tolist(),
                    first_gradients=first_grad, changed=changed, frozen_other_unchanged=True, initial_metrics=initial_metrics,
                    loss_first=losses[0], loss_last=losses[-1], mean_warm_update_seconds=float(np.mean(times[1:])),
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, final=final)
                if args.stage == "preflight":
                    replay, _ = check(None)
                    assert replay == final
                    row["replay_metrics_exact"] = True
                if previous and init == 42:
                    old = next(r for r in previous["runs"] if r["arm"] == arm)
                    assert old["initial_shared_sha256"] == initial_hash and abs(old["loss_first"]-losses[0]) < 1e-6
                    row["cluster_preflight_initialization_and_first_loss_match"] = True
                report["runs"].append(row)
                (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
                print("BATCH_FINAL", init, arm, "train_cm", final["train"]["path_position_cm"],
                    "dev_cm", final["development"]["path_position_cm"], "fit_gate", final["train"]["fit_gate"]["pose_fit_passed"], flush=True)
                del optimizer, ctx, prediction, truth, loss
                for p in model.parameters():
                    p.grad = None
                torch.cuda.empty_cache()
            assert clip_hash == [digest(model.vision_encoder), digest(model.text_encoder)]
            del model, initial, language, visual, mask
            torch.cuda.empty_cache()
        assert sources == {name:sha(code/name) for name in sources}
        if args.stage == "fit":
            report["paired_differences_homogeneous_minus_mixed"] = []
            for init in init_seeds:
                arms = {r["arm"]:r for r in report["runs"] if r["initialization_seed"] == init}
                report["paired_differences_homogeneous_minus_mixed"].append(dict(initialization_seed=init,
                    **{split:{key:arms["homogeneous"]["final"][split][key]-arms["mixed"]["final"][split][key]
                        for key in ["path_position_cm", "path_rotation_deg", "endpoint_position_cm", "endpoint_rotation_deg"]}
                        for split in ["train", "development"]}))
        report.update(execution_checks_passed=True, clip_unchanged=True, production_source_unchanged=True,
            validation_evaluated=True, development_windows_evaluated=4 if args.stage == "preflight" else 40,
            scientific_improvement_verdict="Requires paired result review; execution passing is not improvement")
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-start
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("BATCH_CONTROL_REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
