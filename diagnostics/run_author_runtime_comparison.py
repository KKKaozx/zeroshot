"""Execute original denoisers and diffusers 0.11.1 against project methods.

No optimizer, pretrained-model download, or learned-performance claim.
Run in an isolated environment; see docs/DDPM_RUNTIME_COMPARISON.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1",
                  HF_HOME=str(ROOT / "training_cache/references/runtime-hf"))
import argparse
import ast
import gc
import hashlib
import inspect
import json
import sys
import textwrap
import time
from importlib.metadata import version
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import diffusers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "training_cache/references/diffusion_policy"))
from models import RobotAdapterModel, diffusion_betas
from diffusion_decoder import ConditionalDiffusionDecoder
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def digest_state(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Recorder(nn.Module):
    def __init__(self, clean, abar, kind):
        super().__init__()
        self.clean, self.abar, self.kind, self.rows = clean, abar, kind, []

    def forward(self, x, t, context):
        a = self.abar[t].reshape(-1, 1, 1)
        output = self.clean if self.kind == "sample" else (x - a.sqrt() * self.clean) / (1-a).sqrt()
        self.rows.append((int(t[0]), x.detach().clone(), output.detach().clone()))
        return output


def bare_model(kind, clipping, limit):
    # Bypass CLIP construction only. Call actual production loss/sample methods.
    model = RobotAdapterModel.__new__(RobotAdapterModel)
    nn.Module.__init__(model)
    model.action_dim, model.chunk_size, model.num_diffusion_steps = 8, 16, 100
    model.decoder_type, model.separate_gripper_head = "diffusion", True
    model.gripper_target_mode = "state"
    model.diffusion_prediction_type = kind
    model.clip_denoised, model.max_normalized_position = clipping, limit
    model.beta_schedule = "squaredcos_cap_v2"
    # Execute the production constructor's buffer block rather than copy formulas.
    tree = ast.parse(textwrap.dedent(inspect.getsource(RobotAdapterModel.__init__)))
    body = tree.body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "betas" for t in node.targets))
    module = ast.fix_missing_locations(ast.Module(body=body[start:], type_ignores=[]))
    exec(compile(module, str(ROOT / "code/models.py"), "exec"),
         dict(self=model, torch=torch, F=torch.nn.functional,
              diffusion_betas=diffusion_betas))
    return model


def difference(a, b):
    return float((a-b).abs().max())


def scheduler_case(kind, clipping, limit, clean, initial, noises, context):
    model = bare_model(kind, clipping, limit)
    spy = Recorder(clean, model.alpha_bars, kind)
    model.diffusion_decoder = spy
    model.predict_gripper_logits = lambda c, p, g=None: torch.zeros(c.shape[0], 16)
    noise_calls = iter(range(99, 0, -1))
    with patch("torch.randn", side_effect=lambda *a, **k: initial.clone()), \
         patch("torch.randn_like", side_effect=lambda *a, **k: noises[next(noise_calls)].clone()):
        final = model.sample(context)
    rows = spy.rows
    assert [r[0] for r in rows] == list(range(99, -1, -1))
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                              variance_type="fixed_small", prediction_type=kind,
                              clip_sample=clipping)
    scheduler.set_timesteps(100)
    errors, matched_range_errors = [], []
    for i, (t, x, output) in enumerate(rows):
        actual = rows[i+1][1] if t else final[..., :7]
        with patch("torch.randn", side_effect=lambda *a, **k: noises[t].clone()):
            expected = scheduler.step(output, t, x).prev_sample
        if not t:
            expected[..., :3].clamp_(-limit, limit)
            expected[..., 3:7] = nn.functional.normalize(expected[..., 3:7], dim=-1)
        errors.append(difference(actual, expected))
        if clipping and limit != 1:
            # Keep library code unchanged; explicitly align the application's range.
            a = scheduler.alphas_cumprod[t]
            x0 = output if kind == "sample" else (x-(1-a).sqrt()*output)/a.sqrt()
            x0 = torch.cat([x0[..., :3].clamp(-limit, limit), x0[..., 3:].clamp(-1, 1)], -1)
            aligned = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                    variance_type="fixed_small", prediction_type="sample", clip_sample=False)
            with patch("torch.randn", side_effect=lambda *a, **k: noises[t].clone()):
                expected_aligned = aligned.step(x0, t, x).prev_sample
            if not t:
                expected_aligned[..., :3].clamp_(-limit, limit)
                expected_aligned[..., 3:7] = nn.functional.normalize(expected_aligned[..., 3:7], dim=-1)
            matched_range_errors.append(difference(actual, expected_aligned))
    relevant = matched_range_errors or errors
    return dict(prediction_type=kind, clipping=clipping, project_xyz_limit=limit,
                steps_compared=100, native_max_abs_error=max(errors),
                aligned_range_max_abs_error=max(relevant),
                tolerance_abs=2e-4, passed=max(relevant)<2e-4,
                native_difference_expected=bool(clipping and limit!=1))


def network_case(name, model, x, t, context):
    before = digest_state(model)
    start = time.perf_counter()
    model.eval()
    c = context.clone().requires_grad_()
    sample = x.clone().requires_grad_()
    call = (lambda a,b,d: model(a,b,global_cond=d)) if name == "author" else model
    y = call(sample, t, c)
    with torch.no_grad():
        changed_context = difference(y, call(sample, t, c+0.1))
        changed_time = difference(y, call(sample, (t+17)%100, c))
    y.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    connected = all(g is not None and torch.isfinite(g).all() for g in gradients)
    context_grad = float(c.grad.abs().sum())
    input_grad = float(sample.grad.abs().sum())
    after = digest_state(model)
    passed = (y.shape==x.shape and torch.isfinite(y).all() and connected
              and context_grad>0 and input_grad>0 and changed_context>0 and changed_time>0
              and before==after)
    return dict(name=name,parameters=sum(p.numel() for p in model.parameters()),
                input_shape=list(x.shape),output_shape=list(y.shape),
                condition_change_max_abs=changed_context,timestep_change_max_abs=changed_time,
                context_gradient_l1=context_grad,input_gradient_l1=input_grad,
                all_trainable_parameter_gradients_present_and_finite=bool(connected),
                weights_unchanged=before==after,seconds=time.perf_counter()-start,passed=bool(passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/ddpm_runtime_comparison.json")
    args = parser.parse_args()
    assert diffusers.__version__ == "0.11.1"
    import subprocess
    author_commit=subprocess.check_output([
        "git","-C",str(ROOT/"training_cache/references/diffusion_policy"),"rev-parse","HEAD"
    ]).decode().strip()
    assert author_commit=="5ba07ac6661db573af695b419a7947ecb704690f"
    torch.set_num_threads(4)
    torch.manual_seed(20261006)
    x, noise, context = torch.randn(2,16,7), torch.randn(2,16,7), torch.randn(2,1024)
    noises = torch.randn(100,2,16,7)
    clean = torch.randn_like(x)*0.2
    clean[..., 6] += 1
    clean[..., 3:7] = nn.functional.normalize(clean[..., 3:7], dim=-1)
    steps = torch.tensor([0,99])
    scheduler = DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2")
    forward_clean = clean.repeat(50,1,1)
    forward_noise = torch.randn_like(forward_clean)
    forward_steps = torch.arange(100)
    actions = torch.cat([forward_clean,torch.ones(100,16,1)], -1)
    forward_checks=[]
    for kind in ["epsilon","sample"]:
        model = bare_model(kind, False, 3)
        capture = Recorder(forward_clean,model.alpha_bars,kind)
        model.diffusion_decoder = capture
        model.predict_gripper_logits = lambda c,p,g=None: torch.zeros(c.shape[0],16)
        with patch("torch.randint",return_value=forward_steps),patch("torch.randn_like",return_value=forward_noise):
            _,target,_ = model.diffusion_loss(actions,context.repeat(50,1))
        error=difference(capture.rows[0][1],scheduler.add_noise(forward_clean,forward_noise,forward_steps))
        expected_target=forward_clean if kind=="sample" else forward_noise
        forward_checks.append(dict(prediction_type=kind,timesteps_compared=100,
                                   noise_addition_max_abs_error=error,
                                   training_target_exact=torch.equal(target,expected_target),
                                   passed=error<1e-6 and torch.equal(target,expected_target)))
    forward_error=max(c["noise_addition_max_abs_error"] for c in forward_checks)
    schedule_error = difference(model.betas,scheduler.betas)
    cases=[]
    for kind in ["epsilon","sample"]:
        for clipping,limit in [(False,3),(True,1),(True,3)]:
            boundary=clean.clone()
            if clipping and limit==3:boundary[...,0]=2.0  # expose ±1 versus ±3
            cases.append(scheduler_case(kind,clipping,limit,boundary,x,noises,context))
    print("DDPM cases:",json.dumps(cases),flush=True)
    networks=[]
    for name in ["project","author"]:
        torch.manual_seed(20261006)
        if name=="project":
            net=ConditionalDiffusionDecoder(action_dim=7,context_dim=1024,hidden_dim=128)
        else:
            net=ConditionalUnet1D(input_dim=7,global_cond_dim=1024,
                                  diffusion_step_embed_dim=128,down_dims=[512,1024,2048],
                                  kernel_size=5,n_groups=8,cond_predict_scale=True)
        result=network_case(name,net,x,steps,context)
        networks.append(result)
        print("U-Net:",json.dumps(result),flush=True)
        del net
        gc.collect()
    files=[ROOT/"code/models.py",ROOT/"code/diffusion_decoder.py",
           Path(inspect.getfile(DDPMScheduler)),Path(inspect.getfile(ConditionalUnet1D))]
    report=dict(device="cpu",dtype="float32",torch=torch.__version__,diffusers=diffusers.__version__,
                author_commit=author_commit,
                isolated_dependencies={n:version(n) for n in ["transformers","tokenizers","huggingface-hub","einops"]},
                seed=20261006,date="2026-10-06",batch=2,horizon=16,action_dim=7,context_dim=1024,
                forward=dict(beta_max_abs_error=schedule_error,noise_addition_max_abs_error=forward_error,
                             cases=forward_checks,passed=all(c["passed"] for c in forward_checks) and schedule_error<1e-6),
                reverse_cases=cases,networks=networks,optimizer_steps=0,
                passed=all(r["passed"] for r in cases+networks) and forward_error<1e-6 and schedule_error<1e-6,
                source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                scope="Executed production diffusion_loss/sample and unchanged author U-Net/DDPMScheduler. Synthetic inputs and oracle denoiser test arithmetic; random U-Net weights test connectivity only. No learned performance or GPU check.")
    out=ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    assert report["passed"], "See numerical results; do not claim passing."


if __name__=="__main__":
    main()
