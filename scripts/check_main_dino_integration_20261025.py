"""New main DINO path with actual frozen pretrained encoders, not policy training.

One reduced Adapter/decoder fixture checks the newly connected loss, checkpoint
and live policy paths. It does not qualify the formal eight-layer run resources.
"""
import copy
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def rejected(label,call,checks):
    try:call()
    except (ValueError,FileNotFoundError) as exc:
        checks.append({'check':label,'passed':True,'rejection':str(exc)})
        return
    raise AssertionError('Incompatible input accepted: '+label)


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='reports/main_dino_integration_20261025/attempt2')
    options=parser.parse_args()
    out=(ROOT/options.output).resolve()
    out.parent.mkdir(exist_ok=True,parents=True)
    out.mkdir(exist_ok=False)
    usage=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    used,util=map(int,usage.split(','))
    if used>1024 or util>20:raise ValueError('GPU busy; do not disturb an existing workload')
    import numpy as np
    import torch
    from torch import nn
    from transformers import CLIPTokenizer
    from dataset import prepare_image,CLIP_IMAGE_MEAN,CLIP_IMAGE_STD
    from frozen_visual_backbone import (CLIP_MEAN,CLIP_STD,DINO_MEAN,DINO_STD,
        FrozenDinoVision,validate_dino_config)
    from models import RobotAdapterModel
    from train import parse_args,build_config,policy_loss,tokenise,trainable_state_dict
    from libero_project_policy import ProjectWaypointPolicy,load_project_policy

    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    clip=Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    dino=ROOT/'baselines/dinov2_large_frozen_assets_20261023'
    arguments=['--visual-backbone','dinov2','--dino-assets',str(dino),'--model-name',str(clip),
        '--adapter-layers','1','--attention-dim','32','--attention-heads','4',
        '--decoder-hidden-dim','32','--chunk-size','4','--diffusion-steps','4',
        '--beta-schedule','squaredcos_cap_v2','--diffusion-prediction-type','sample',
        '--oxe-command-input','--gripper-change-weight','1','--gripper-target-mode','state']
    args=parse_args(arguments)
    config=build_config(args)
    legacy=build_config(parse_args([]))
    assert not any(k.startswith('visual_') or k=='dino_assets' for k in legacy['model'])
    checks=[{'check':'legacy default config has no added visual fields','passed':True}]
    assert torch.equal(torch.tensor(CLIP_MEAN).view(3,1,1),CLIP_IMAGE_MEAN)
    assert torch.equal(torch.tensor(CLIP_STD).view(3,1,1),CLIP_IMAGE_STD)
    checks.append({'check':'canonical normalization matches actual Dataset constants','passed':True})
    bad=copy.deepcopy(config['model']);bad['visual_input_contract']='wrong'
    rejected('wrong image contract',lambda:validate_dino_config(bad),checks)
    bad=copy.deepcopy(config['model']);bad['visual_asset_identity']['sha256']['model.safetensors']='0'*64
    rejected('wrong checkpoint backbone identity',lambda:validate_dino_config(bad),checks)
    bad=copy.deepcopy(config['model']);bad['dino_assets']=str(out/'absent')
    rejected('no implicit asset download',lambda:validate_dino_config(bad),checks)
    corrupt=out/'corrupt_assets';corrupt.mkdir();(corrupt/'model.safetensors').write_bytes(b'corrupt fixture')
    bad=copy.deepcopy(config['model']);bad['dino_assets']=str(corrupt)
    rejected('corrupt weight bytes',lambda:FrozenDinoVision(bad),checks)

    torch.manual_seed(20261025)
    tick=time.monotonic()
    model=RobotAdapterModel(config).cuda().train()
    tokenizer=CLIPTokenizer.from_pretrained(str(clip),local_files_only=True)
    load_seconds=time.monotonic()-tick
    torch.cuda.reset_peak_memory_stats()
    encoders=(model.vision_encoder,model.text_encoder)
    assert all(not e.training and all(not p.requires_grad for p in e.parameters()) for e in encoders)
    frozen_versions={(i,n):p._version for i,e in enumerate(encoders) for n,p in e.named_parameters()}
    white=prepare_image(np.full((224,224,3),255,dtype=np.uint8))[None].cuda()
    analytic=torch.tensor([(1-m)/s for m,s in zip(DINO_MEAN,DINO_STD)],device='cuda').view(1,3,1,1)
    normal_error=float((model.vision_encoder.dino_pixels(white)-analytic).abs().max())
    assert normal_error<1e-6
    checks.append({'check':'analytic white image uses one DINO normalization','passed':True,'max_abs':normal_error})
    rgb=np.arange(24*40*3,dtype=np.uint8).reshape(24,40,3)
    pixels=prepare_image(rgb)[None].cuda()
    mapped=model.vision_encoder.dino_pixels(pixels)
    # Rectangular source keeps the same content region and CLIP-mean padding.
    pad_expected=torch.tensor([(a-b)/s for a,b,s in zip(CLIP_MEAN,DINO_MEAN,DINO_STD)],device='cuda')
    assert torch.allclose(mapped[0,:,0,0],pad_expected,rtol=0,atol=1e-6)
    text=tokenise(tokenizer,['move the blue block left'],torch.device('cuda'))
    grip=torch.tensor([[1.,1.]],device='cuda')
    actions=torch.zeros(1,4,8,device='cuda');actions[...,6]=1;actions[...,7]=1
    context=model.get_context_vector(pixels,text['input_ids'],text.get('attention_mask'))
    assert context.shape==(1,1024) and context.requires_grad
    output=model(pixels,text['input_ids'],attention_mask=text.get('attention_mask'),
        current_gripper=grip,actions=actions)
    loss=policy_loss(model,output,actions,nn.MSELoss(),current_grippers=grip,
        supervision_masks=torch.ones_like(actions))[0]
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.adapter,model.diffusion_decoder,model.gripper_head):
        grads=[p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool(g.abs().max()>0) for g in grads)
    assert all(p.grad is None and not p.requires_grad and p._version==frozen_versions[(i,n)]
        for i,e in enumerate(encoders) for n,p in e.named_parameters())
    checks.append({'check':'actual pretrained vision/text frozen; new main loss backpropagates to learned heads','passed':True})
    model.zero_grad(set_to_none=True)
    checkpoint_path=out/'random_engineering_fixture.pt'
    torch.save({'config':config,'trainable_state_dict':trainable_state_dict(model),
        'experiment_kind':'random_dino_main_engineering_fixture_not_trained'},checkpoint_path)
    # Structural callback fixture only. The actual executor's SciPy/physics
    # contract was already checked separately and is not being re-run here.
    contract=SimpleNamespace(target_times_seconds=tuple(.1*(i+1) for i in range(4)),
        position_scale_meters=.1,action_representation=config['action']['representation'],
        ticks=lambda:(2,4,6,8))
    rejected('DINO checkpoint without explicit DINO assets',lambda:load_project_policy(
        checkpoint_path,clip,contract,device='cuda',sampling_seed=701),checks)
    runtime=load_project_policy(checkpoint_path,clip,contract,device='cuda',sampling_seed=701,local_dino_path=dino)
    request={'rgb_upright':rgb,'instruction':'move the blue block left','current_open_command':1.}
    before_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    result=runtime(request)
    assert torch.equal(torch.get_rng_state(),before_rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    model.eval()
    with torch.random.fork_rng(devices=[0]),torch.inference_mode():
        torch.manual_seed(701)
        ctx=model.get_context_vector(pixels,text['input_ids'],text.get('attention_mask'))
        direct=model.sample(ctx,current_gripper=grip)[0].cpu().numpy()
    assert np.array_equal(result,direct)
    assert result.shape==(4,8)
    checks.append({'check':'actual main checkpoint reload and live callback exactly match direct main inference; caller RNG unchanged','passed':True})
    wrong=copy.deepcopy(config);wrong['model']['visual_backbone']='clip'
    rejected('runtime CLIP/DINO mismatch',lambda:ProjectWaypointPolicy(model,tokenizer,wrong,
        contract,device='cuda',sampling_seed=701),checks)
    checks.append({'check':'no optimizer constructed or stepped; no target tasks or physics accessed','passed':True})
    r=dict(passed=True,checks=checks,actual_pretrained_dinov2_loaded=True,
        actual_pretrained_clip_text_loaded=True,visual_identity=config['model']['visual_asset_identity'],
        visual_input_contract=config['model']['visual_input_contract'],
        image_geometry='Existing main bilinear letterbox224 and CLIP-mean padding, shared by Dataset and live policy; not Push-T PIL preprocessing',
        fixture_adapter_layers=1,fixture_attention_dim=32,fixture_decoder_hidden_dim=32,
        fixture_sampling_steps=4,fixture_images_and_text='Synthetic RGB and language; no source or target evaluation data',
        actual_eight_layer_formal_resources_qualified=False,load_seconds=load_seconds,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        total_seconds=time.monotonic()-started,backward_only_loss=float(loss.detach()),
        optimization_updates=0,physics_steps=0,new_cluster_jobs=0,
        formal_training_ready=False,policy_generalization_verdict=None,
        code_sha256={name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() for name in (
            'frozen_visual_backbone.py','models.py','train.py','step_training_runner.py','libero_project_policy.py')})
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2),flush=True)


if __name__=='__main__':main()
