"""Read-only audit of a preregistered repeatedly failing training episode.

Rebuild geometry with rotation matrices, inspect existing predictions and frozen
CLIP tokens. No training, checkpoint changes, downloads or held-out evaluation.
"""
import os
os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import CLIPModel
from dataset import UnifiedRobotDataset, prepare_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/bridge_repeated_failure_audit_20261011"
DATA = Path("E:/dataset/bridge_v2_0.0.1/0.0.1")
CLIP = Path("D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
FOCAL = ("bridge_data_v2-train.tfrecord-00008-of-01024", 25)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotation(euler):
    """Independent Rz(yaw) @ Ry(pitch) @ Rx(roll), float64."""
    r, p, y = np.asarray(euler, dtype=np.float64)
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return rz @ ry @ rx

def quat_matrix(quat):
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))

def feature_distance(a,b):
    av,bv=a.double().flatten(),b.double().flatten()
    return dict(exactly_equal=bool(torch.equal(a,b)),
                cosine=float(torch.nn.functional.cosine_similarity(av,bv,dim=0)),
                relative_l2=float((av-bv).norm()/av.norm()),
                mean_absolute_difference=float((av-bv).abs().mean()))

def main():
    started=time.monotonic()
    OUT.mkdir(exist_ok=False)
    ref=ROOT/"reports/bridge_samecount32_fit_20261011"
    plan=json.loads((ref/"plan.json").read_text())
    rows=plan["selection"]
    focus=[i for i,r in enumerate(rows) if (r["shard"],r["record_index"])==FOCAL]
    assert focus==[10,26] and [rows[i]["start_index"] for i in focus]==[12,0]
    sources={n:sha(ROOT/"code"/n) for n in ["dataset.py","models.py","adapter.py","diffusion_decoder.py",Path(__file__).name]}
    audit_plan=dict(stage="raw_episode_frozen_feature_and_retained_prediction_audit",
        focal_episode=list(FOCAL), focal_starts=[0,12], reference=str(ref),
        reference_plan_sha256=sha(ref/"plan.json"), source_sha256=sources,
        trained=False, simulation_executed=False, downloads=False,
        geometry="Independent float64 Rz Ry Rx and quaternion-to-matrix conversions",
        frozen_features="All 32 retained training inputs; no future frame conditioning",
        limitations=["Index agreement does not establish hardware timestamp synchronization",
                     "No trained Adapter weights retained for these controls; learned context cannot be inspected",
                     "Feature similarity alone does not prove ambiguity or a defective encoder"])
    (OUT/"plan.json").write_text(json.dumps(audit_plan,indent=2)+"\n")
    (OUT/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    reader.chunk_size=16
    reader.bridge_gripper_policy="reverse_scan_valid_steps_v2"
    reader.bridge_current_gripper="continuous"
    records={}; pixels=[]; loaded=[]; geometry=[]; focal=None
    retained=np.load(ref/"repeat_current_predictions.npz")
    for i,row in enumerate(rows):
        key=(row["shard"],row["record_index"])
        if key not in records:
            paths=list(DATA.rglob(row["shard"]))
            assert len(paths)==1
            records[key]=UnifiedRobotDataset._load_tfrecord_example(str(paths[0]),key[1])
        example=records[key]; f=example.features.feature
        n=len(f["steps/language_instruction"].bytes_list.value)
        states=np.asarray(f["steps/observation/state"].float_list.value,dtype=np.float32).reshape(n,7)
        actions=np.asarray(f["steps/action"].float_list.value,dtype=np.float32).reshape(n,7)
        assert np.isfinite(states).all() and np.isfinite(actions).all()
        t=row["start_index"]; assert t+16<n
        reader._load_tfrecord_example=lambda *args,e=example:e
        instruction,pixel,grip,target=reader._get_tfrecord_bridge_state_action(dict(file_path=row["shard"],record_index=key[1],start_index=t))
        assert instruction==row["task"] and np.array_equal(target.numpy(),retained["targets"][i])
        image=np.asarray(Image.open(io.BytesIO(f["steps/observation/image_0"].bytes_list.value[t])).convert("RGB"))
        assert torch.equal(pixel,prepare_image(image))
        r0=rotation(states[t,3:6]); truth=target.numpy()
        local=np.stack([r0.T@(states[j,:3].astype(np.float64)-states[t,:3])/.1 for j in range(t+1,t+17)])
        decoded=states[t,:3]+(r0@(truth[:,:3].astype(np.float64)*.1).T).T
        rot_diff=[angle(r0.T@rotation(states[j,3:6]),quat_matrix(truth[k,3:7])) for k,j in enumerate(range(t+1,t+17))]
        geometry.append(dict(index=i,start=t,normalized_xyz_max_abs_difference=float(abs(local-truth[:,:3]).max()),
            world_xyz_max_error_cm=float(np.linalg.norm(decoded-states[t+1:t+17,:3],axis=1).max()*100),
            rotation_matrix_max_error_deg=max(rot_diff)))
        pixels.append(pixel); loaded.append((states,actions))
        if key==FOCAL:
            focal=(example,states,actions)
    print("RAW RECORDS AND INDEPENDENT GEOMETRY CHECKED",flush=True)
    example,states,actions=focal; f=example.features.feature; n=len(states)
    field_inventory={k:dict(kind=v.WhichOneof("kind"),count=len(getattr(v,v.WhichOneof("kind")).value)) for k,v in f.items()}
    langs=[v.decode("utf-8",errors="replace") for v in f["steps/language_instruction"].bytes_list.value]
    imgs=[Image.open(io.BytesIO(v)).convert("RGB") for v in f["steps/observation/image_0"].bytes_list.value]
    assert len(imgs)==n and set(langs)=={"close microwave"}
    selected_frames=sorted(set(list(range(0,n,2))+[0,12,n-1]))
    tiles=Image.new("RGB",(4*320,((len(selected_frames)+3)//4)*265),"white"); draw=ImageDraw.Draw(tiles)
    for k,j in enumerate(selected_frames):
        x,y=(k%4)*320,(k//4)*265
        tile=imgs[j].copy();tile.thumbnail((316,224));tiles.paste(tile,(x,y+35))
        draw.text((x+4,y+3),f"frame {j} | xyz(m) "+" ".join(f"{v:.3f}" for v in states[j,:3]),fill="black")
        draw.text((x+4,y+18),"INPUT t=0" if j==0 else "INPUT t=12" if j==12 else "raw observation",fill="red" if j in [0,12] else "black")
    tiles.save(OUT/"raw_episode_frames.png")
    np.savez_compressed(OUT/"raw_episode_arrays.npz",states=states,actions=actions)
    report=dict(audit_plan,field_inventory=field_inventory,steps=n,languages=sorted(set(langs)),
        image_shapes=sorted(set(str(np.asarray(im).shape) for im in imgs)),
        first_indices=np.flatnonzero(f["steps/is_first"].int64_list.value).tolist(),
        last_indices=np.flatnonzero(f["steps/is_last"].int64_list.value).tolist(),
        timestamp_fields=[k for k in f if "time" in k.lower()],
        independent_geometry=geometry,
        state_step_position_cm=(np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1)*100).tolist(),
        state_step_rotation_deg=[angle(rotation(a[3:6]),rotation(b[3:6])) for a,b in zip(states[:-1],states[1:])],
        raw_image_adjacent_mae=[float(abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float)).mean()) for a,b in zip(imgs[:-1],imgs[1:])],
        duplicate_adjacent_images=[j for j in range(n-1) if np.array_equal(np.asarray(imgs[j]),np.asarray(imgs[j+1]))],
        saved_prediction_metrics={})
    saved_world={"states":states}
    for label,directory in [("samecount32",ref),("timepoint64",ROOT/"reports/bridge_timepoint64_fit_20261011"),("other_episode32",ROOT/"reports/bridge_scale32_fit_20261011")]:
        rp=json.loads((directory/"plan.json").read_text()); z=np.load(directory/"repeat_current_predictions.npz")
        report["saved_prediction_metrics"][label]=[]
        for i,row in enumerate(rp["selection"]):
            if (row["shard"],row["record_index"])!=FOCAL:continue
            t=row["start_index"]; pred=z["predictions"][:,i]; truth=z["targets"][i]
            r0=rotation(states[t,3:6]); world=states[t,:3]+(pred[:,:,:3]*.1)@r0.T
            err=np.linalg.norm(world-states[t+1:t+17,:3],axis=-1)*100
            rot=np.array([[angle(r0@quat_matrix(q),rotation(states[j,3:6])) for q,j in zip(seed[:,3:7],range(t+1,t+17))] for seed in pred])
            report["saved_prediction_metrics"][label].append(dict(index=i,start=t,path_position_mm=float(err.mean()*10),path_rotation_deg=float(rot.mean()),
                endpoint_position_mm=float(err[:,-1].mean()*10),per_horizon_position_mm=(err.mean(0)*10).tolist(),
                per_horizon_rotation_deg=rot.mean(0).tolist(),world_signed_bias_mm=((world-states[t+1:t+17,:3])*1000).mean((0,1)).tolist(),
                seed_path_position_mm=(err.mean(1)*10).tolist()))
            saved_world[f"{label}_t{t}"]=world
    np.savez_compressed(OUT/"world_predictions.npz",**saved_world)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    clip=CLIPModel.from_pretrained(str(CLIP),local_files_only=True).eval()
    vision=clip.vision_model.cuda()
    before=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in vision.parameters())).hexdigest()
    with torch.inference_mode():
        tokens=torch.cat([vision(pixel_values=torch.stack(pixels[i:i+2]).cuda()).last_hidden_state.cpu() for i in range(0,len(pixels),2)])
    after=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in vision.parameters())).hexdigest()
    assert before==after
    feature_pairs=[]
    for i in range(16):
        feature_pairs.append(dict(index=i,task=rows[i]["task"],all_tokens=feature_distance(tokens[i],tokens[i+16]),
            cls_token=feature_distance(tokens[i,0],tokens[i+16,0]),patch_tokens=feature_distance(tokens[i,1:],tokens[i+16,1:]),
            pixel_mae=float((pixels[i]-pixels[i+16]).abs().mean())))
    normalized=torch.nn.functional.normalize(tokens.flatten(1).double(),dim=1)
    distances=torch.cdist(normalized,normalized); distances.fill_diagonal_(float("inf"))
    nearest=[]
    for i in focus:
        j=int(distances[i].argmin());nearest.append(dict(index=i,nearest_index=j,nearest_task=rows[j]["task"],distance=float(distances[i,j]),
            paired_index=i+16 if i<16 else i-16,paired_distance=float(distances[i,i+16 if i<16 else i-16])))
    report.update(frozen_clip_unchanged=True,clip_sha256=before,token_shape=list(tokens.shape),feature_pairs=feature_pairs,
        focal_nearest_neighbor=nearest,elapsed_seconds=time.monotonic()-started,execution_checks_passed=True)
    assert all(v==sha(ROOT/"code"/k) for k,v in sources.items())
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("FOCAL",json.dumps(report["saved_prediction_metrics"]),flush=True)
    print("FEATURE",json.dumps(feature_pairs[10]),flush=True)
    print("AUDIT COMPLETE",OUT,flush=True)

if __name__=="__main__":main()
