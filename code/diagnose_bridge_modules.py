"""Read-only module diagnosis on a checkpoint's train/validation partitions.

No training, label changes, test target access, or simulation. Reuses an existing
fixed-window fit as a positive control before proposing new optimization runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
import pybullet as bullet
from transformers import CLIPTokenizer

from dataset import ACTION_REPRESENTATION, UnifiedRobotDataset
from models import RobotAdapterModel
from train import collate_batch, dataset_split_identity, set_seed, tokenise


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def matrices(quaternions):
    return np.stack([np.array(bullet.getMatrixFromQuaternion((q / np.linalg.norm(q)).tolist())).reshape(3, 3) for q in quaternions])


def matrix_angles(relative):
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def reconstruct(checkpoint):
    cfg = dict(checkpoint["data_config"])
    allowed = set(__import__("inspect").signature(UnifiedRobotDataset).parameters)
    cfg["data_dir"] = cfg.pop("dataset_dir")
    ds = UnifiedRobotDataset(**{k: v for k, v in cfg.items() if k in allowed})
    if checkpoint["config"]["action"]["representation"] != ACTION_REPRESENTATION:
        raise ValueError("Action representation mismatch")
    if dataset_split_identity(ds) != checkpoint["dataset_identity"] or len(ds) != checkpoint["dataset_size"]:
        raise ValueError("Dataset identity/size changed; do not reuse old indices")
    groups = [{ds.group_key(i) for i in checkpoint["split_indices"][name]} for name in ("train", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Episode leakage across partitions")
    return ds


def read_raw(ds, indices):
    """Independent sequential TFRecord reader, not the loader's offset reader."""
    import tensorflow as tf
    wanted = defaultdict(set)
    for i in indices:
        s = ds.samples[i]
        if s["source"] != "tfrecord_bridge_state_action":
            raise ValueError("This diagnostic only supports verified Bridge state/action schema")
        wanted[s["file_path"]].add(s["record_index"])
    episodes = {}
    for path, records in wanted.items():
        for number, payload in enumerate(tf.data.TFRecordDataset([path]).as_numpy_iterator()):
            if number > max(records):
                break
            if number not in records:
                continue
            ex = tf.train.Example.FromString(payload)
            f = ex.features.feature
            n = len(f["steps/language_instruction"].bytes_list.value)
            state = np.asarray(f["steps/observation/state"].float_list.value, dtype=np.float64).reshape(n, 7)
            command = np.asarray(f["steps/action"].float_list.value, dtype=np.float64).reshape(n, 7)
            first = list(f["steps/is_first"].int64_list.value)
            last = list(f["steps/is_last"].int64_list.value)
            if first != [1] + [0] * (n - 1) or last != [0] * (n - 1) + [1]:
                raise ValueError("Invalid RLDS endpoints")
            valid = command[:-1, 6]
            # Independent vectorized next-endpoint implementation of reverse scan.
            if not np.isfinite(state).all() or not np.isfinite(command).all() or np.any((valid < 0) | (valid > 1)):
                raise ValueError("Invalid raw values")
            endpoints = np.flatnonzero((valid < .05) | (valid > .95))
            next_endpoint = np.searchsorted(endpoints, np.arange(len(valid)))
            if not len(endpoints) or np.any(next_endpoint == len(endpoints)):
                raise ValueError("Unresolved nonbinary gripper tail")
            binary = (valid[endpoints[next_endpoint]] > .95).astype(int)
            episodes[path, number] = {"state": state, "command": command, "binary": binary,
                "images": list(f["steps/observation/image_0"].bytes_list.value),
                "languages": list(f["steps/language_instruction"].bytes_list.value)}
        print(f"[原始核验] {Path(path).name}：{len(records)}条指定演示", flush=True)
    if len(episodes) != sum(map(len, wanted.values())):
        raise ValueError("Missing raw episodes")
    return episodes


def audit(ds, indices, raw, output):
    import tensorflow as tf
    mean = np.array([.48145466, .4578275, .40821073])
    std = np.array([.26862954, .26130258, .27577711])
    rows, decoded = [], {}
    for i in indices:
        s = ds.samples[i]
        r = raw[s["file_path"], s["record_index"]]
        start = s["start_index"]
        ts = np.arange(start + 1, start + ds.chunk_size + 1)
        if ts[-1] >= len(r["state"]):
            raise ValueError("Non-full window in fixed Bridge experiment")
        item = ds[i]
        decoded[i] = item
        text, image, current, target, mask = item
        a = target.numpy().astype(np.float64)
        state = r["state"]
        reference_q = np.array(bullet.getQuaternionFromEuler(state[start, 3:6].tolist()))
        future_q = np.array([bullet.getQuaternionFromEuler(v.tolist()) for v in state[ts, 3:6]])
        reference = matrices([reference_q])[0]
        future = matrices(future_q)
        local = (state[ts, :3] - state[start, :3]) @ reference / .1
        inverse_q = reference_q * np.array([-1, -1, -1, 1])
        q = np.array([bullet.multiplyTransforms([0,0,0],inverse_q.tolist(),[0,0,0],v.tolist())[1] for v in future_q])
        q_error = np.minimum(np.max(np.abs(a[:, 3:7] - q), axis=1), np.max(np.abs(a[:, 3:7] + q), axis=1)).max()
        world = (.1 * a[:, :3]) @ reference.T + state[start, :3]
        world_rotation = reference @ matrices(a[:, 3:7])
        raw_rgb = np.asarray(Image.open(io.BytesIO(r["images"][start])).convert("RGB"))
        h, w = raw_rgb.shape[:2]
        scale = min(224 / h, 224 / w)
        rh, rw = round(h * scale), round(w * scale)
        reference_rgb = np.broadcast_to(mean, (224, 224, 3)).copy()
        resized = tf.image.resize(raw_rgb.astype(np.float32) / 255, (rh, rw), method="bilinear").numpy()
        top, left = (224 - rh) // 2, (224 - rw) // 2
        reference_rgb[top:top+rh, left:left+rw] = resized
        restored = image.numpy().transpose(1, 2, 0) * std + mean
        row = {"index": i, "shard": Path(s["file_path"]).name, "record_index": s["record_index"],
            "start_index": start, "pose_observation_indices": ts.tolist(), "gripper_command_indices": (ts-1).tolist(),
            "xyz_max_error": float(np.abs(local-a[:, :3]).max()), "quaternion_max_error_up_to_sign": float(q_error),
            "restored_world_position_max_error_m": float(np.abs(world-state[ts, :3]).max()),
            "restored_world_rotation_max_error_deg": float(matrix_angles(np.swapaxes(future,-1,-2) @ world_rotation).max()),
            "image_max_error_rgb_unit": float(np.abs(restored-reference_rgb).max()),
            "gripper_mismatches": int(np.count_nonzero(a[:, 7] != 2*r["binary"][ts-1]-1)),
            "current_measurement_error": float(abs(current.item()-(2*state[start, 6]-1))),
            "instruction_matches": text == r["languages"][start].decode("utf-8", errors="replace"),
            "full_mask": bool((mask == 1).all()), "image_nonblank": bool(raw_rgb.std() > 0),
            "xyz_components_outside_output_limit": int((np.abs(a[:, :3]) > 3).sum()),
            "input_image_sha256": hashlib.sha256(r["images"][start]).hexdigest()}
        row["numeric_contract_pass"] = bool(row["xyz_max_error"] < 2e-5 and q_error < 2e-5
            and row["restored_world_position_max_error_m"] < 2e-6 and row["restored_world_rotation_max_error_deg"] < .001
            and row["image_max_error_rgb_unit"] < 2e-5 and row["gripper_mismatches"] == 0
            and row["current_measurement_error"] < 1e-6 and row["instruction_matches"] and row["full_mask"])
        rows.append(row)
    report = {"purpose": "independent_pybullet_tf_numeric_contract_audit", "dataset_identity": dataset_split_identity(ds), "windows": len(rows),
        "passed_windows": sum(r["numeric_contract_pass"] for r in rows), "rows": rows,
        "limits": ["Checks implementation against current reached-pose/previous-command contract, not physical control semantics.",
            "Does not establish sensor calibration, control delay, or suitability of reached poses as executable commands.",
            "Shares source files but uses independent record reader, PyBullet rotations and TensorFlow resizing."]}
    save_json(output / "data_contract_audit.json", report)
    if report["passed_windows"] != len(rows):
        raise ValueError("Independent contract check failed; see audit before any model diagnosis")
    return decoded


def metrics(pred, target, teacher=None):
    pred, target = np.asarray(pred), np.asarray(target)
    pn = pred[..., 3:7] / np.linalg.norm(pred[..., 3:7], axis=-1, keepdims=True)
    tn = target[..., 3:7] / np.linalg.norm(target[..., 3:7], axis=-1, keepdims=True)
    pos = np.linalg.norm(pred[..., :3]-target[..., :3], axis=-1)*10
    angle = np.degrees(2*np.arccos(np.clip(np.abs((pn*tn).sum(-1)), 0, 1)))
    truth, guessed = target[..., 7] >= 0, pred[..., 7] >= 0
    opened, closed = truth, ~truth
    op_recall = float(guessed[opened].mean()) if opened.any() else None
    cl_recall = float((~guessed[closed]).mean()) if closed.any() else None
    m = {"windows": len(pred), "action_targets": truth.size, "position_error_cm": float(pos.mean()),
        "zero_motion_position_error_cm": float((np.linalg.norm(target[..., :3], axis=-1)*10).mean()),
        "rotation_error_deg": float(angle.mean()),
        "identity_rotation_error_deg": float(np.degrees(2*np.arccos(np.clip(np.abs(tn[..., 3]), 0, 1))).mean()),
        "gripper_accuracy": float((truth==guessed).mean()), "open_recall": op_recall, "closed_recall": cl_recall,
        "balanced_accuracy": (op_recall+cl_recall)/2 if op_recall is not None and cl_recall is not None else None,
        "true_open_rate": float(truth.mean()), "predicted_open_rate": float(guessed.mean()),
        "position_error_by_waypoint_cm": pos.mean(0).tolist(), "rotation_error_by_waypoint_deg": angle.mean(0).tolist()}
    if teacher is not None:
        m["teacher_pose_gripper_accuracy"] = float(((np.asarray(teacher)>=0)==truth).mean())
    for name, pair in (("open_to_closed",truth[:, :-1]&~truth[:, 1:]),("closed_to_open",~truth[:, :-1]&truth[:, 1:])):
        correct = (guessed[:, :-1]==truth[:, :-1]) & (guessed[:, 1:]==truth[:, 1:])
        m[name+"_pairs"] = int(pair.sum())
        m[name+"_correct"] = int((pair&correct).sum())
    return m


@torch.no_grad()
def predict(model, tokenizer, ds, decoded, indices, device, batch_size):
    result = {}
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start+batch_size]
        texts, images, current, targets, masks = collate_batch([decoded[i] for i in ids])
        text = tokenise(tokenizer, texts, device)
        context = model.get_context_vector(images.to(device), text["input_ids"], text.get("attention_mask"))
        prediction = model.sample(context, current.to(device)).cpu().numpy().astype(np.float64)
        logits = model.predict_gripper_logits(context, torch.tensor(prediction[..., :7],dtype=torch.float32,device=device),current.to(device)).cpu().numpy()
        teacher = model.predict_gripper_logits(context,targets[..., :7].to(device),current.to(device)).cpu().numpy()
        for k, i in enumerate(ids):
            result[i] = {"prediction": prediction[k], "logits": logits[k], "teacher_logits": teacher[k], "target": targets[k].numpy().astype(np.float64)}
        if start % (batch_size*8) == 0:
            print(f"[模型复核] {min(start+batch_size,len(indices))}/{len(indices)}窗口", flush=True)
    return result


def group_report(ds, indices, values):
    groups = defaultdict(list)
    for i in indices:
        groups[ds.group_key(i)].append(i)
    def measure(ids):
        return metrics([values[i]["prediction"] for i in ids], [values[i]["target"] for i in ids], [values[i]["teacher_logits"] for i in ids])
    episodes = []
    for key, ids in groups.items():
        s = ds.samples[ids[0]]
        unique = defaultdict(list)
        for i in ids:
            truth = values[i]["target"][:, 7] >= 0
            guessed = values[i]["prediction"][:, 7] >= 0
            for k in np.flatnonzero(truth[:-1]!=truth[1:]):
                unique[ds.samples[i]["start_index"]+int(k),"open_to_closed" if truth[k] else "closed_to_open"].append(bool((truth[k:k+2]==guessed[k:k+2]).all()))
        unique_report = {}
        for direction in ("open_to_closed", "closed_to_open"):
            views = [v for (t,d),v in unique.items() if d==direction]
            unique_report[direction] = {"unique_raw_command_pairs": len(views),
                "all_window_views_correct": sum(all(v) for v in views), "any_window_view_correct": sum(any(v) for v in views)}
        episodes.append({"group_key":key,"shard":Path(s["file_path"]).name,"record_index":s["record_index"],
            "indices":ids,**measure(ids),"unique_transition_consistency":unique_report})
    return {"overall":measure(indices),"episodes":episodes,
        "episode_equal_weight_mean": {k:float(np.mean([e[k] for e in episodes])) for k in ("position_error_cm","zero_motion_position_error_cm","rotation_error_deg","identity_rotation_error_deg","gripper_accuracy")},
        "note":"Unique events indexed by first raw command step; all/any counts expose overlapping-window disagreement, not new execution success rates."}


def export_plot_input(ds, raw, i, value, path):
    s = ds.samples[i]
    r = raw[s["file_path"],s["record_index"]]
    start = s["start_index"]
    images=[]
    for t in (start,start+1,start+ds.chunk_size):
        filename=f"raw_window_{i}_observation_{t}.png"
        if not (path.parent/filename).exists():
            Image.open(io.BytesIO(r["images"][t])).convert("RGB").save(path.parent/filename)
        images.append({"path":filename,"observation_index":t})
    save_json(path,{"index":i,"shard":Path(s["file_path"]).name,"record_index":s["record_index"],
        "images":images,**{k:v.tolist() for k,v in value.items()}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="results/bridge_single_task_full_validation_v1/latest.pt")
    parser.add_argument("--fit-checkpoint",default="results/bridge_gripper_coverage_fit_v1/latest.pt")
    parser.add_argument("--fit-manifest",default="results/bridge_gripper_coverage_fit_v1/overfit_manifest.json")
    parser.add_argument("--output-dir",default="results/bridge_module_diagnostic_v1")
    parser.add_argument("--cache-dir",default="D:/ntu_related/dissertation/hf_cache")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--plot-python", default=sys.executable, help="Python with NumPy/Pillow; no package installation required")
    args=parser.parse_args()
    if args.batch_size<1: parser.error("batch-size must be positive")
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError("Use a new output directory to preserve earlier diagnosis")
    output.mkdir(parents=True,exist_ok=True)
    set_seed(42)
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if checkpoint["config"]["model"]["decoder_type"]!="regression": raise ValueError("Only deterministic regression checkpoints supported")
    ds=reconstruct(checkpoint)
    train=list(checkpoint["split_indices"]["train"])
    validation=list(checkpoint["split_indices"]["validation"])
    indices=train+validation
    manifest=json.loads(Path(args.fit_manifest).read_text(encoding="utf-8"))
    selected=manifest["selected_indices"]
    if manifest["dataset_identity"]!=checkpoint["dataset_identity"] or not set(selected).issubset(train):
        raise ValueError("Fixed-window control must belong to unchanged training partition")
    save_json(output/"run_config.json",{**vars(args),"seed":42,"dataset_identity":checkpoint["dataset_identity"],
        "train_windows":len(train),"validation_windows":len(validation),"test_targets_inspected":False,
        "trained":False,"fixed_indices":selected,"config":checkpoint["config"]})
    raw=read_raw(ds,indices)
    decoded=audit(ds,indices,raw,output)
    print(f"[核验通过] {len(indices)}个训练/验证窗口；不读取test目标",flush=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=RobotAdapterModel(checkpoint["config"],cache_dir=args.cache_dir).to(device)
    missing=model.load_state_dict(checkpoint["trainable_state_dict"],strict=False)
    trainable_names={name for name,p in model.named_parameters() if p.requires_grad}
    if missing.unexpected_keys or trainable_names.intersection(missing.missing_keys):
        raise ValueError("Missing trainable weights")
    model.eval()
    tokenizer=CLIPTokenizer.from_pretrained(checkpoint["config"]["model"]["name"],cache_dir=args.cache_dir,local_files_only=True)
    values=predict(model,tokenizer,ds,decoded,indices,device,args.batch_size)
    reports={"train":group_report(ds,train,values),"validation":group_report(ds,validation,values)}
    save_json(output/"per_episode_metrics.json",reports)
    columns=["partition","shard","record_index","windows","position_error_cm","zero_motion_position_error_cm",
        "rotation_error_deg","identity_rotation_error_deg","gripper_accuracy","balanced_accuracy","open_recall","closed_recall",
        "open_to_closed_correct","open_to_closed_pairs","closed_to_open_correct","closed_to_open_pairs"]
    with (output/"per_episode_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore"); writer.writeheader()
        for partition,r in reports.items():
            for e in r["episodes"]: writer.writerow({"partition":partition,**e})
    save_json(output/"window_predictions.json",[{"index":i,"partition":"train" if i in set(train) else "validation",
        "start_index":ds.samples[i]["start_index"],"group_key":ds.group_key(i),
        **{k:v.tolist() for k,v in values[i].items()}} for i in indices])
    control=torch.load(args.fit_checkpoint,map_location="cpu",weights_only=False)
    if control["dataset_identity"]!=checkpoint["dataset_identity"] or control["config"]!=checkpoint["config"]:
        raise ValueError("Positive control differs in architecture/data contract")
    if (control.get("experiment_kind") != "training_fit_diagnostic"
            or set(control["split_indices"]["train"]) != set(selected)):
        raise ValueError("Control training partition does not equal the fixed fit manifest")
    model.load_state_dict(control["trainable_state_dict"],strict=False); model.eval()
    fit_values=predict(model,tokenizer,ds,decoded,selected,device,args.batch_size)
    fixed={"scope":"same_14_training_windows_not_generalization", "current_checkpoint":args.checkpoint,
        "positive_control_checkpoint":args.fit_checkpoint,"selected_indices":selected,
        "current":group_report(ds,selected,values),"positive_control":group_report(ds,selected,fit_values),
        "limits":"Existing runs differ in sample distribution/update history; this proves fit capability, not a causal sampling ablation."}
    save_json(output/"fixed_window_control.json",fixed)
    save_json(output/"fixed_control_predictions.json",[{"index":i,**{k:v.tolist() for k,v in fit_values[i].items()}} for i in selected])
    plot_dir=output/"plots"; plot_dir.mkdir()
    plot_ids=set(selected)
    for e in reports["validation"]["episodes"]:
        ids=e["indices"]
        plot_ids.add(max(ids,key=lambda i:metrics([values[i]["prediction"]],[values[i]["target"]])["rotation_error_deg"]))
        plot_ids.add(max(ids,key=lambda i:metrics([values[i]["prediction"]],[values[i]["target"]])["position_error_cm"]))
        plot_ids.update(i for i in ids if np.any(np.diff(values[i]["target"][:,7])!=0))
    for i in sorted(plot_ids): export_plot_input(ds,raw,i,values[i],plot_dir/f"current_window_{i}.json")
    for i in selected: export_plot_input(ds,raw,i,fit_values[i],plot_dir/f"fit_control_window_{i}.json")
    subprocess.run([args.plot_python,str(Path(__file__).with_name("render_bridge_diagnosis.py")),str(output)],check=True)
    print("[完成] 逐演示指标、独立数据核验、同窗口成功对照与失败图已保存：",output.resolve(),flush=True)
    for p,r in reports.items(): print(p,r["overall"],flush=True)
    print("fixed current",fixed["current"]["overall"],flush=True)
    print("fixed positive control",fixed["positive_control"]["overall"],flush=True)


if __name__=="__main__": main()
