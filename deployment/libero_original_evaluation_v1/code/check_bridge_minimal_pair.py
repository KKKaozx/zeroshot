"""Preregistered early-window alone versus early/late joint memorization control.

Reuse production models and prior positive-control evaluation. No downloads,
held-out targets, simulation, or large checkpoints. Read the saved plan for
minibatch replication and the limits of comparisons to earlier 32-window runs.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.update(USE_TF="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import UnifiedRobotDataset
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import digest,evaluate,score,seed

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def t_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--stage",choices=["preflight","fit"],default="preflight")
    ap.add_argument("--preflight-reference",type=Path)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    started=time.monotonic()
    report=dict(stage=args.stage,execution_checks_passed=False,trained=True,
                validation_evaluated=False,simulation_executed=False,downloads=False,
                gripper_trained=False,checkpoints_saved=False,runs=[])
    ref=ROOT/"reports/bridge_samecount32_fit_20261011"
    refplan=json.loads((ref/"plan.json").read_text())
    cfg=copy.deepcopy(refplan["config"])
    rows=[refplan["selection"][i] for i in [26,10]]
    assert [r["start_index"] for r in rows]==[0,12]
    assert all(r["partition"]=="train" and r["record_index"]==25 and r["shard"]=="bridge_data_v2-train.tfrecord-00008-of-01024" for r in rows)
    sources={n:sha(ROOT/"code"/n) for n in ["dataset.py","models.py","adapter.py","diffusion_decoder.py",
        "replay_and_fit_bridge_positive.py","evaluate_bridge_endpoint_metrics.py",Path(__file__).name]}
    for n,h in sources.items():
        if n in json.loads((ref/"report.json").read_text())["source_sha256"]:
            assert h==json.loads((ref/"report.json").read_text())["source_sha256"][n]
    per_window=3 if args.stage=="preflight" else 500
    seeds=[42] if args.stage=="preflight" else [42,43,44]
    plan=dict(config=cfg,selection=rows,source_sha256=sources,reference_plan_sha256=sha(ref/"plan.json"),
        train_initialization_seeds=seeds,sampling_seeds=[1101,1102,1103],arms=["early_alone","early_late_joint"],
        changing_variable="Add the late window, alternating homogeneous batches; early batches and random keys shared",
        batch_size=8,batch_policy="Eight replicas of one window, with independent per-replica diffusion noise/dropout",
        updates_per_window=per_window,draw_slots_per_window=8*per_window,
        optimizer="Adam",lr=1e-4,weight_decay=0,learning_rate_schedule="constant",gradient_clip=1.,
        loss="Pose x0 MSE only",gripper="Frozen; no success or event verdict",
        context="Same repeated-current two-Adapter-call history projection as previous matched controls; W initialized zero",
        sample_steps=100,prediction_type="sample",schedule="squaredcos_cap_v2",
        evaluation_exposure_milestones=[64,500] if args.stage=="fit" else [3],
        main_budget="64 updates/window=512 draw slots, near previous 487..518 slots",
        secondary_budget="500 updates/window=4000 draw slots, fixed before observing results",
        pass_criteria=dict(each_sampling_seed_path_position_cm=.5,each_sampling_seed_path_rotation_deg=5,
                           overall_each_seed_endpoint_position_cm=1.,overall_each_seed_endpoint_rotation_deg=10.,
                           required_windows_fraction=.9,replicated_pass="Every trained window passes at all three initialization seeds"),
        evaluation_batch="Always both inputs in order t0,t12, preserving early noise draws and GPU shapes. Late output from early-alone is explicitly not a fitted target",
        comparison_limits=["Different batch composition and optimizer update counts from prior 32-window runs",
            "Joint arm has twice the total optimizer steps; per-window draw slots and batch size match",
            "Only one seen demonstration; no generalization or unique faulty-layer inference",
            "CUDA attention backward may be nondeterministic; paired seeds are not bitwise training guarantees"],
        storage_policy="Small reports, final contexts, predictions only; no checkpoints or cached CLIP tokens")
    (args.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    (args.output/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        rawpath=Path("E:/dataset/bridge_v2_0.0.1/0.0.1")/rows[0]["shard"]
        example=UnifiedRobotDataset._load_tfrecord_example(str(rawpath),25)
        reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        reader.chunk_size=16;reader.bridge_gripper_policy="reverse_scan_valid_steps_v2";reader.bridge_current_gripper="continuous"
        reader._load_tfrecord_example=lambda *_:example
        items=[reader._get_tfrecord_bridge_state_action(dict(file_path=str(rawpath),record_index=25,start_index=r["start_index"])) for r in rows]
        targets=torch.stack([v[3] for v in items]);currents=torch.stack([v[2] for v in items])
        with np.load(ref/"repeat_current_predictions.npz") as z:
            assert np.array_equal(targets.numpy(),z["targets"][[26,10]])
        prior=json.loads((ref/"report.json").read_text())
        for i,old in enumerate([26,10]):
            assert t_sha(items[i][1])==prior["data_checks"]["lineage"][old]["current_image_sha256"]
        report.update(targets_match_retained=True,images_match_retained=True,gpu=torch.cuda.get_device_name(),source_sha256=sources)
        initial_prefix={}
        for init in seeds:
            seed(init);model=RobotAdapterModel(cfg).cuda().eval()
            tokenizer=CLIPTokenizer.from_pretrained(cfg["model"]["name"],local_files_only=True)
            # Preserve the original 16-row text GEMM shape and pair image ordering.
            text={k:v.cuda() for k,v in tokenizer([v[0] for v in items]*8,padding=True,truncation=True,max_length=77,return_tensors="pt").items()}
            with torch.no_grad():
                language=model.text_encoder(**text).last_hidden_state[:2].detach()
                # Parent used [late, another] and [early, another] in batch=2;
                # pair shape remains two, and frozen feature hashes are recorded.
                visual=model.vision_encoder(pixel_values=torch.stack([v[1] for v in items]).cuda()).last_hidden_state.detach()
            model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
            frozen_clip=[digest(model.vision_encoder),digest(model.text_encoder)]
            model.history_projection=torch.nn.Linear(1024,1024,bias=False).cuda();torch.nn.init.zeros_(model.history_projection.weight)
            prefixes=("adapter.","diffusion_decoder.","history_projection.")
            for n,p in model.named_parameters():p.requires_grad_(n.startswith(prefixes))
            initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(("vision_encoder.","text_encoder."))}
            initial_hash=hashlib.sha256("".join(n+t_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
            target_gpu=targets.cuda();current_gpu=currents.cuda()
            for arm in plan["arms"]:
                model.load_state_dict(initial,strict=False)
                assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items())
                optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
                def context(idx):
                    now=model.adapter(visual[idx],language[idx],text["attention_mask"][:2][idx])
                    old=model.adapter(visual[idx],language[idx],text["attention_mask"][:2][idx])
                    return now+model.history_projection(old-now)
                def check(k):
                    model.eval()
                    with torch.no_grad():
                        ctx=context(torch.arange(2,device="cuda"))
                        pred,_=evaluate(model,ctx,current_gpu,targets.numpy(),plan["sampling_seeds"],batch=2)
                    active=1 if arm=="early_alone" else 2
                    metrics=score(pred[:,:active],targets[:active].numpy())
                    per=[score(pred[:,i:i+1],targets[i:i+1].numpy()) for i in range(2)]
                    return pred,ctx.cpu().numpy(),dict(updates_per_window=k,draw_slots_per_window=k*8,
                        trained_windows=active,training_metrics=metrics,early=per[0],late=per[1],
                        late_is_training_target=arm!="early_alone")
                pred,_,untrained=check(0)
                if arm=="early_alone":initial_prefix[init]=pred.copy()
                else:assert np.array_equal(pred,initial_prefix[init])
                schedule=[(k,j) for k in range(1,per_window+1) for j in ([0] if arm=="early_alone" else [0,1])]
                history=[untrained];losses=[];durations=[];first_grad=None
                torch.cuda.reset_peak_memory_stats()
                for update,(k,j) in enumerate(schedule,1):
                    model.train();idx=torch.full((8,),j,device="cuda",dtype=torch.long)
                    key=k+200000*j
                    seed(430000+key);torch.cuda.synchronize();tick=time.monotonic()
                    ctx=context(idx);seed(420000+key)
                    prediction,truth,_=model.diffusion_loss(target_gpu[idx],ctx,current_gpu[idx])
                    loss=torch.nn.functional.mse_loss(prediction,truth);assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if first_grad is None:
                        first_grad={prefix:any(p.grad is not None and bool(torch.any(p.grad!=0)) for n,p in model.named_parameters() if n.startswith(prefix)) for prefix in prefixes}
                        assert all(first_grad.values())
                        assert all(p.grad is None for p in model.vision_encoder.parameters())
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize();durations.append(time.monotonic()-tick);losses.append(float(loss.detach()))
                    if update==1:
                        if arm=="early_alone":report.setdefault("first_loss",{})[str(init)]=losses[0]
                        else:assert abs(losses[0]-report["first_loss"][str(init)])<1e-7
                    if update==1 or update%100==0:print("MINIMAL_UPDATE",init,arm,update,"/",len(schedule),"loss",losses[-1],flush=True)
                    completed_cycle=(j==0 if arm=="early_alone" else j==1)
                    if completed_cycle and k in plan["evaluation_exposure_milestones"]:
                        pred,ctx_final,metrics=check(k);history.append(metrics)
                        np.savez_compressed(args.output/f"seed{init}_{arm}_u{k}.npz",predictions=pred,targets=targets.numpy(),contexts=ctx_final)
                        print("MINIMAL_METRICS",init,arm,k,json.dumps(metrics),flush=True)
                changed={prefix:any(not torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items() if n.startswith(prefix)) for prefix in prefixes}
                assert all(changed.values())
                assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items() if not n.startswith(prefixes))
                row=dict(initialization_seed=init,arm=arm,initial_shared_sha256=initial_hash,
                    updates=len(schedule),draw_slots=[per_window*8,per_window*8 if arm!="early_alone" else 0],
                    first_gradients=first_grad,changed=changed,frozen_other_unchanged=True,
                    mean_warm_update_seconds=float(np.mean(durations[1:])),peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    history=history,loss_first=losses[0],loss_last=losses[-1],
                    final_hashes={"adapter":digest(model.adapter),"decoder":digest(model.diffusion_decoder),"history_projection":digest(model.history_projection)})
                if args.stage=="preflight":
                    replay,_,_=check(per_window);assert np.array_equal(pred,replay)
                    row["replay_exact"]=True
                if args.preflight_reference and init==42:
                    old=json.loads((args.preflight_reference/"report.json").read_text())
                    checkrow=next(x for x in old["runs"] if x["arm"]==arm)
                    assert initial_hash==checkrow["initial_shared_sha256"] and row["loss_first"]==checkrow["loss_first"]
                    assert row["history"][0]==checkrow["history"][0]
                    row["preflight_initialization_loss_metrics_match"]=True
                report["runs"].append(row)
                report["elapsed_seconds"]=time.monotonic()-started
                (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
                del optimizer,ctx,prediction,truth,loss
                for p in model.parameters():p.grad=None
                torch.cuda.empty_cache()
            assert frozen_clip==[digest(model.vision_encoder),digest(model.text_encoder)]
            del model,initial,visual,language,text,target_gpu,current_gpu
            torch.cuda.empty_cache()
        assert all(h==sha(ROOT/"code"/n) for n,h in sources.items())
        report.update(execution_checks_passed=True,clip_unchanged=True,production_source_unchanged=True,
            fit_verdict_by_arm={arm:all(x["history"][-1]["training_metrics"]["pose_fit_passed"] for x in report["runs"] if x["arm"]==arm) for arm in plan["arms"]} if args.stage=="fit" else None)
    except Exception as error:
        report["error"]=repr(error);raise
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("MINIMAL_PAIR_REPORT",args.output/"report.json",flush=True)

if __name__=="__main__":main()
