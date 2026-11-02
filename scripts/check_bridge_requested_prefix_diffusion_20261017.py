"""CPU-only interface/math checks, using an untrained SMALL U-Net and an oracle.

No CLIP/Adapter, pretrained/checkpoint weights, real targets, GPU or optimization.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["USE_TF"]="0"
import ast
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import torch
from torch import nn
from torch.nn import functional as F

from bridge_requested_prefix_diffusion_v1 import (
    RequestedPrefixDenoiser, request_mask, project_auxiliary, training_objective,
    schedule_from_betas, sample_requested,
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/bridge_requested_prefix_contract_20261017"
BASE=ROOT/"cluster_packages/bridge_growth128_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError("Missing real supervision or invalid request silently accepted")


def archived_functions():
    path=BASE/"models.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    beta=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="diffusion_betas")
    model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RobotAdapterModel")
    sampler=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=="sample")
    namespace={"torch":torch,"F":F,"math":math}
    exec(compile(ast.Module(body=[beta,sampler],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["diffusion_betas"],namespace["sample"]


class RecordedCore(nn.Module):
    def __init__(self,core):
        super().__init__();self.core=core;self.trace=[]

    def forward(self,noisy,times,context):
        result=self.core(noisy,times,context)
        if not torch.is_grad_enabled():
            self.trace.append(dict(input=noisy.clone(),t=times.clone(),context=context.clone(),output=result.clone()))
        return result


class ConstantCleanOracle(nn.Module):
    """Synthetic delta-distribution math oracle; NOT a trained policy."""
    def forward(self,noisy,times,context):
        return context[:,:7,None].transpose(1,2).expand(-1,16,-1)


def main():
    assert not OUT.exists(),"Keep completed checks immutable"
    started=time.monotonic();torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    active=read(ROOT/"configs/active_experiment_reference.json")
    for name,h in active["local_pinned_files"].items():assert sha(ROOT/name)==h
    prior=ROOT/"reports/bridge_partial_future_contract_20261017"
    old=read(prior/"report.json");rows=read(prior/"candidate_rows.json")["train"]
    assert old["data_contract_verified"] and not old["training_integration_ready"]
    assert sha(prior/"candidate_rows.json")==old["candidate_rows_sha256"]
    assert len(rows)==3394
    requests=[dict(row_index=i,requested_horizon=1) for i in range(len(rows))]
    requests += [dict(row_index=i,requested_horizon=16) for i,r in enumerate(rows) if r["valid_future_length"]==16]
    assert len(requests)==5230 and sum(r["requested_horizon"]==16 for r in requests)==1836
    assert all(rows[x["row_index"]]["valid_future_length"]>=x["requested_horizon"] for x in requests)
    # Import only the archived decoder architecture, not the full project model.
    spec=importlib.util.spec_from_file_location("archived_prefix_decoder",BASE/"diffusion_decoder.py")
    decoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(decoder)
    torch.manual_seed(712)
    core=RecordedCore(decoder.MultiscaleConditionalUnet1D(action_dim=7,chunk_size=16,
        context_dim=8,down_dims=(8,16,32),diffusion_step_embed_dim=8)).eval()
    wrapped=RequestedPrefixDenoiser(core,8).eval()
    assert torch.count_nonzero(wrapped.request_projection.weight)==0
    # Even learned/nonzero request-projection parameters cannot alter full16
    # behavior while the shared U-Net parameters are held fixed.
    with torch.no_grad():wrapped.request_projection.weight.normal_(0,.01)
    before=state_sha(wrapped)
    beta_function,source_sample=archived_functions()
    coefficients=schedule_from_betas(beta_function(100,"squaredcos_cap_v2"))
    context=torch.randn(2,8)
    targets=torch.randn(2,16,7)*.1
    targets[...,3:7]=torch.tensor([0.,0.,0.,1.])
    timesteps=torch.tensor([24,74],dtype=torch.long)
    noise=torch.randn_like(targets)
    full=torch.ones(2,16,dtype=torch.bool)
    loss_full,noisy_full,pred_full=training_objective(wrapped,targets,full,context,timesteps,
        noise,coefficients["alpha_bars"],16)
    bars=coefficients["alpha_bars"][timesteps,None,None]
    legacy_noisy=bars.sqrt()*targets+(1-bars).sqrt()*noise
    legacy_pred=core(legacy_noisy,timesteps,context)
    assert torch.equal(noisy_full,legacy_noisy) and torch.equal(pred_full,legacy_pred)
    assert torch.equal(loss_full,F.mse_loss(legacy_pred,targets))
    # A sample with15 real future labels and the same first target must have
    # EXACTLY the sameK1 input/loss as one with only1 label;L is not a condition.
    availability=full.clone();availability[0,1:]=False;availability[1,15:]=False
    short_targets=targets.masked_fill(~availability[...,None],float("nan"))
    loss1,x1,p1=training_objective(wrapped,short_targets,availability,context,timesteps,
        noise,coefficients["alpha_bars"],1)
    poisoned=short_targets.clone();poisoned[:,1:]=float("nan")
    poisoned_noise=noise.clone();poisoned_noise[:,1:]=float("nan")
    available1=torch.zeros_like(availability);available1[:,0]=True
    loss2,x2,p2=training_objective(wrapped,poisoned,available1,context,timesteps,
        poisoned_noise,coefficients["alpha_bars"],1)
    assert torch.equal(loss1,loss2) and torch.equal(x1,x2) and torch.equal(p1,p2)
    assert torch.count_nonzero(x1[:,1:])==torch.count_nonzero(p1[:,1:])==0
    rejected(lambda:training_objective(wrapped,short_targets,availability,context,timesteps,
        noise,coefficients["alpha_bars"],16))
    rejected(lambda:request_mask(2,3,"cpu"))
    # Prove arbitrary raw auxiliary INPUTS cannot cross into the U-Net, including
    # its GroupNorm. Structural zero padding still changes the requested task.
    raw=noise.clone().requires_grad_(True)
    with torch.no_grad():raw[:,1:]=float("nan")
    wrapped.zero_grad(set_to_none=True)
    out=wrapped(raw,timesteps,context,1)
    objective=(out[:,0]-targets[:,0]).square().mean()
    objective.backward()
    invalid_gradient=float(raw.grad[:,1:].abs().max())
    request_gradient=float(wrapped.request_projection.weight.grad.norm())
    assert invalid_gradient==0 and request_gradient>0 and torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in wrapped.parameters() if p.grad is not None)
    # Actual archived .sample AST runs on the same SMALL, untrained U-Net. This
    # verifies the new loop, not the old production checkpoint/policy performance.
    dummy=SimpleNamespace(action_dim=8,separate_gripper_head=True,chunk_size=16,
        decoder_type="diffusion",num_diffusion_steps=100,diffusion_prediction_type="sample",
        clip_denoised=True,max_normalized_position=3.,gripper_target_mode="state",
        diffusion_decoder=core,**coefficients)
    dummy.predict_gripper_logits=lambda ctx,poses,current:torch.zeros(ctx.shape[0],16)
    probe_context=context[:1]
    core.trace=[];torch.manual_seed(7701)
    legacy_result=source_sample(dummy,probe_context)[...,:7]
    legacy_trace=core.trace
    core.trace=[];torch.manual_seed(7701)
    new_result=sample_requested(wrapped,probe_context,coefficients,16)
    new_trace=core.trace
    assert len(legacy_trace)==len(new_trace)==100
    replay_max=0.
    for a,b in zip(legacy_trace,new_trace):
        for key in ("input","t","context","output"):
            assert torch.equal(a[key],b[key]),key
            replay_max=max(replay_max,float((a[key]-b[key]).abs().max()))
    assert torch.equal(legacy_result,new_result)
    # K1 inference API has no record length or label input. All100 U-Net calls
    # receive exactly zero auxiliary slots and only one pose is returned.
    core.trace=[];torch.manual_seed(7702)
    next_action=sample_requested(wrapped,probe_context,coefficients,1)
    assert next_action.shape==(1,1,7) and torch.isfinite(next_action).all()
    assert len(core.trace)==100
    assert all(torch.count_nonzero(x["input"][:,1:])==0 for x in core.trace)
    assert torch.allclose(next_action[...,3:7].norm(dim=-1),torch.ones(1,1),atol=1e-6)
    # An exact synthetic clean-prediction oracle checks closure of theK1 reverse
    # chain; its success is NOT evidence that a neural policy has learned.
    oracle=RequestedPrefixDenoiser(ConstantCleanOracle(),8).eval()
    oracle_context=torch.tensor([[.2,-.1,.3,0.,0.,0.,1.,0.]])
    torch.manual_seed(7703)
    oracle_result=sample_requested(oracle,oracle_context,coefficients,1)
    oracle_difference=float((oracle_result[:,0]-oracle_context[:,:7]).abs().max())
    assert oracle_difference<1e-5
    assert state_sha(wrapped)==before,"Autograd checks must not update weights"
    sampler_parameters=list(inspect.signature(sample_requested).parameters)
    assert sampler_parameters==["denoiser","context","coefficients","requested_horizon"]
    report=dict(
        cpu_request_contract_verified=True, production_training_integration_ready=False,
        new_generalization_result=False, active_reference=207558, baseline_unchanged=True,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],optimization_updates=0,
        project_model_or_checkpoint_loaded=False, raw_dataset_read=False, real_targets_read=False,
        heldout_targets_read=False, learned_policy_rollouts=False, GPU_jobs_submitted=False,
        tests_use="Synthetic tensors, untrained small archived U-Net layout, and a synthetic exact-clean oracle",
        architecture_probe=dict(down_dims=[8,16,32],context_dim=8,
            core_parameters=sum(p.numel() for p in core.parameters()),
            request_projection_parameters=128,production1024_context_additional_parameters=16384),
        requested_horizons=[1,16],diffusion_steps_for_both=100,
        remaining_length_is_model_condition=False,
        training_pair_counts=dict(K1=3394,K16=1836,total=5230,independent_new_demonstrations=0),
        checks=dict(full16_forward_noise_exact=True,full16_denoiser_exact_even_with_nonzero_request_projection=True,
            full16_loss_max_difference=float(abs(loss_full.detach()-F.mse_loss(legacy_pred,targets).detach())),
            full16_source_sample_replay_steps=100,full16_all_latent_context_output_max_difference=replay_max,
            full16_final_pose_difference=float((legacy_result-new_result).abs().max()),
            K1_availability1_vs15_same_input_prediction_loss=True,
            K1_unrequested_target_and_noise_poison_invariant=True,
            requested16_missing_labels_rejected=True,invalid_input_gradient=invalid_gradient,
            new_request_projection_gradient=request_gradient,all_gradients_finite=True,
            K1_all100_inference_inputs_auxiliary_zero=True,K1_output_shape=list(next_action.shape),
            K1_quaternions_unit=True,K1_synthetic_clean_oracle_max_difference=oracle_difference,
            no_optimizer_or_parameter_change=True),
        explicit_changes=["A caller-request conditioning projection is added; not the unchanged model",
            "Auxiliary slots are projected to zero at forward noise, denoiser input/output, each reverse step",
            "K1 supervision on ALL3394 contexts, K16 on1836 full contexts; weights/compute budget not yet specified",
            "K1 deployment would reobserve after one action; no claim that4-step execution is unchanged",
            "Only7D poses; gripper training/execution not implemented here"],
        limits=["Numerical contracts on synthetic tensors are not pretrained checkpoint replay or generalization evidence",
            "Zero padding and request conditioning still impose a different inductive bias",
            "Unknown future16 poses of late contexts remain unidentifiable from these records; do not claim full16 supervision there",
            "K1 cannot replace the original16-horizon evaluation as proof of improving the same metric",
            "No remaining-length or end-of-task prediction/control capability is established",
            "Batch sampler, full model integration, internal/fresh holdout split, joint gripper, and final training protocol remain open"],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/"scripts/bridge_requested_prefix_diffusion_v1.py",ROOT/"scripts/bridge_partial_future_contract_v1.py",
            BASE/"diffusion_decoder.py",BASE/"models.py",prior/"report.json",prior/"candidate_rows.json"]},
        elapsed_seconds=time.monotonic()-started)
    OUT.mkdir()
    (OUT/"requested_training_pairs.json").write_text(json.dumps(requests,indent=2)+"\n")
    report["pairs_sha256"]=sha(OUT/"requested_training_pairs.json")
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
