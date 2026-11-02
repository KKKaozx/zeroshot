"""CPU full-size INTERFACE check: local frozen CLIP, fresh Adapter/U-Net weights.

Not a207558 checkpoint replay, training experiment, or a real-data evaluation.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="-1",USE_TF="0",HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1")
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"cluster_packages/bridge_growth128_v1"
OUT=ROOT/"reports/bridge_prefix_full_interface_20261017"
sys.path.insert(0,str(BASE))
from models import RobotAdapterModel
from dataset import prepare_image
from transformers import CLIPTokenizer
from bridge_prefix_model_interface_v1 import PrefixModelInterface


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint(parameters):
    h=hashlib.sha256()
    for n,p in sorted(parameters):
        h.update(n.encode());h.update(memoryview(p.detach().contiguous().numpy()).cast("B"))
    return h.hexdigest()


def main():
    assert not OUT.exists(),"Preserve completed interface check"
    begin=time.monotonic();torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    active=read(ROOT/"configs/active_experiment_reference.json")
    for name,h in active["local_pinned_files"].items():assert sha(ROOT/name)==h
    protocol=read(ROOT/"cluster_packages/bridge_coverage_replication_v1/protocol.json")
    for name,h in protocol["inherited_sha256"].items():assert sha(ROOT/"cluster_packages"/name)==h
    cfg=copy.deepcopy(protocol["config"])
    assets=Path(cfg["model"]["name"])
    assert (assets/"model.safetensors").is_file()
    print("FULL_INTERFACE constructing archived full-size model onCPU; localCLIP only",flush=True)
    torch.manual_seed(42);base=RobotAdapterModel(cfg).cpu().eval()
    base.history_projection=nn.Linear(1024,1024,bias=False)
    nn.init.zeros_(base.history_projection.weight)
    with torch.random.fork_rng(devices=[]):
        base.state_projection=nn.Linear(9,1024,bias=False)
        nn.init.zeros_(base.state_projection.weight)
    prefixes=("adapter.","diffusion_decoder.","history_projection.","state_projection.")
    for n,p in base.named_parameters():p.requires_grad_(n.startswith(prefixes))
    original_count=sum(p.numel() for p in base.parameters() if p.requires_grad)
    assert original_count==132309639
    rng=torch.get_rng_state().clone();interface=PrefixModelInterface(base).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in interface.parameters() if p.requires_grad)==original_count+16384
    assert not any(n.startswith("prefix.core.") for n in interface.state_dict()),"Decoder must not register twice"
    before=fingerprint(base.named_parameters())
    request_before=interface.prefix.request_projection.weight.detach().clone()
    print("FULL_INTERFACE actualCLIP +8layerAdapter +multiscaleU-Net constructed",flush=True)
    # Generated calibration image and instruction, NOT a dataset sample.
    yy,xx=np.indices((224,224))
    rgb=np.stack((xx%256,yy%256,(xx+yy)%256),axis=-1).astype(np.uint8)
    pixels=prepare_image(rgb)[None]
    tokenizer=CLIPTokenizer.from_pretrained(str(assets),local_files_only=True)
    text=tokenizer(["close microwave"],padding=True,return_tensors="pt")
    state_protocol=read(ROOT/"cluster_packages/bridge_state_conditioning_v1/protocol.json")
    path=ROOT/"cluster_packages/bridge_state_conditioning_v1/state_input_contract.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="state_features")
    namespace={"np":np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
    current=np.array([[.2,.1,.3,0.,0.,0.]],dtype=np.float32)
    features=torch.from_numpy(namespace["state_features"](current,state_protocol["state_features"]))
    with torch.no_grad():
        visual=base.vision_encoder(pixel_values=pixels).last_hidden_state
        language=base.text_encoder(**text).last_hidden_state
        now=base.adapter(visual,language,text["attention_mask"])
        repeated=base.adapter(visual,language,text["attention_mask"])
        old_context=now+base.history_projection(repeated-now)+base.state_projection(features)
        new_context=interface.context_from_tokens(visual,language,text["attention_mask"],features)
        from_pixels=interface.context_from_observation(pixels,text["input_ids"],text["attention_mask"],features)
    assert torch.equal(old_context,new_context) and torch.equal(new_context,from_pixels)
    assert visual.shape==(1,257,1024) and new_context.shape==(1,1024)
    actions=torch.zeros((1,16,8));actions[...,6]=1.;actions[...,7]=1.
    actions[0,:,0]=torch.linspace(.01,.16,16)
    availability=torch.ones((1,16),dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(8281)
        original_prediction,original_target,_=base.diffusion_loss(actions,new_context,torch.tensor([[1.]]))
        original_loss=torch.nn.functional.mse_loss(original_prediction,original_target)
        torch.manual_seed(8281)
        new_loss,_,new_prediction=interface.loss(actions[...,:7],availability,new_context,16)
    assert torch.equal(original_prediction,new_prediction) and torch.equal(original_loss,new_loss)
    print("FULL_INTERFACE rawobservation/context/full16loss exact; checking CPU sampler",flush=True)
    # Full-size sampler call once; old-vs-new100-step arithmetic was checked in
    # the prior small-model test, so do not repeat another old100-step chain.
    torch.manual_seed(8282);sample=interface.sample_pose(new_context,1)
    assert sample.shape==(1,1,7) and torch.isfinite(sample).all()
    assert torch.allclose(sample[...,3:7].norm(dim=-1),torch.ones(1,1),atol=1e-6)
    print("FULL_INTERFACE full-size K1 DDPM100 done; checking gradients without optimizer",flush=True)
    interface.train()
    assert not base.vision_encoder.training and not base.text_encoder.training
    assert base.adapter.training
    torch.manual_seed(8283)
    train_context=interface.context_from_tokens(visual.detach(),language.detach(),text["attention_mask"],features)
    partial=actions[...,:7].clone();partial[:,1:]=float("nan")
    one=availability.clone();one[:,1:]=False
    torch.manual_seed(8284)
    loss,_,_=interface.loss(partial,one,train_context,1)
    loss.backward()
    gradients={}
    groups=dict(adapter="base.adapter.",decoder="base.diffusion_decoder.",
        history="base.history_projection.",state="base.state_projection.",request="prefix.request_projection.")
    for label,prefix in groups.items():
        values=[p.grad for n,p in interface.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(v).all() for v in values)
        gradients[label]=float(torch.sqrt(sum(v.double().square().sum() for v in values)))
        assert gradients[label]>0
    assert all(p.grad is None for n,p in base.named_parameters() if not n.startswith(prefixes))
    # New trainable format is checked in memory only. No529MiB checkpoint file.
    payload=interface.trainable_payload()
    assert len(payload["request_projection_state_dict"])==1
    interface.load_trainable_payload(payload)
    bad=dict(payload,base_trainable_state_dict=dict(payload["base_trainable_state_dict"],unknown=torch.zeros(1)))
    try:interface.load_trainable_payload(bad)
    except ValueError:pass
    else:raise AssertionError("Unknown state key was silently accepted")
    assert fingerprint(base.named_parameters())==before
    assert torch.equal(request_before,interface.prefix.request_projection.weight)
    del payload,bad
    result=dict(full_size_CPU_interface_verified=True,baseline207558_weights_loaded=False,
        local_pretrained_CLIP_loaded=True,adapter_and_decoder_weights="Fresh random initialization, NOT trained207558",
        GPU_used=False,optimizer_created=False,optimization_updates=0,checkpoints_saved=False,
        raw_dataset_or_holdout_targets_read=False,synthetic_calibration_image_only=True,
        parameters=dict(base_trainable=original_count,additional_request=16384,total_trainable=original_count+16384),
        checks=dict(interface_creation_preserves_rng=True,decoder_not_double_registered=True,
            raw_observation_and_token_context_exact=True,full16_original_diffusion_loss_exact=True,
            K1_fullsize100step_output_shape=list(sample.shape),K1_quaternions_unit=True,
            gradient_norms=gradients,CLIP_and_gripper_gradients_absent=True,
            in_memory_trainable_format_roundtrip=True,unknown_payload_keys_rejected=True,
            all_parameters_unchanged=True),
        no_generalization_verdict=True,no_grasp_capability_claim=True,
        pending=["Actual207558 checkpoint key/config/hash replay on cluster",
            "All1558 added real observations/labels and frozenCLIP cache input verification",
            "Numeric eligibility of new validation/final metadata candidates",
            "Matched batch schedules, freshAdam continuation, and resource preflight on cluster"],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/"scripts/bridge_prefix_model_interface_v1.py",ROOT/"scripts/bridge_requested_prefix_diffusion_v1.py",
            BASE/"models.py",BASE/"adapter.py",BASE/"diffusion_decoder.py",BASE/"dataset.py",path]},
        elapsed_seconds=time.monotonic()-begin)
    OUT.mkdir();(OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
