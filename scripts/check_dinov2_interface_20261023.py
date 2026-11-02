"""Actual pinned backbone and existing Adapter interface checks; no policy training."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/pusht_dinov2_v1'))
sys.path.insert(0,str(ROOT/'scripts/cluster/pusht_project_training_v1'))
from frozen_visual import FrozenDinoVisual,MEAN,STD,model_sha
from project_frontend import ProjectObservationEncoder


def main():
    start=time.monotonic();assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(43)
    encoder=FrozenDinoVisual(ROOT/'baselines/dinov2_large_frozen_assets_20261023')
    before=model_sha(encoder.model)
    torch.cuda.reset_peak_memory_stats()
    # Previously used TRAIN images only; no evaluation labels or scene scoring.
    source=ROOT/'reports/pusht_action_audit_219129/inputs.npz'
    with np.load(source,allow_pickle=False) as f:
        print('INPUT_KEYS',f.files,flush=True)
        roles=f['roles'];i=int(np.flatnonzero(roles=='train')[0])
        images=f['raw_images'][i].copy()
    if images.ndim==5:images=images.reshape(-1,96,96,3)
    assert images.shape==(2,96,96,3) and images.dtype==np.float32
    white=np.full((1,96,96,3),255,dtype=np.float32)
    w=encoder.preprocess(white)
    expected=torch.tensor([(1-m)/s for m,s in zip(MEAN,STD)])
    # PIL bicubic resizing of float255 yields a ~1.2e-6 normalized FP32
    # rounding difference here; test the analytic color, not byte equality.
    constant_delta=float((w[0,:,100,100]-expected).abs().max())
    assert constant_delta<=3e-6
    encoder.train(True)
    assert not encoder.training and not encoder.model.training
    features=encoder(images)
    # Online image round-trip is the same one used by the old author wrapper.
    observed=images/255
    restored=np.rint(observed*255).astype(np.float32)
    assert np.max(np.abs(restored-images))<1e-4
    replay=encoder(restored)
    assert np.array_equal(features,replay)
    # Padding must give the same real frames as an explicit8-frame call.
    padded=np.concatenate((images,np.repeat(images[-1:],6,axis=0)))
    assert np.array_equal(features,encoder(padded)[:2])
    # Keep actual8-layer architecture; synthetic text tests the interface only.
    text=torch.randn(1,12,768);mask=torch.ones(1,12,dtype=torch.int64)
    torch.manual_seed(43)
    adapter=ProjectObservationEncoder(text,mask,(64,65)).cuda().train()
    frozen=torch.from_numpy(features).cuda()
    adapter.set_tokens(frozen)
    obs={'agent_pos':torch.tensor([[.1,.2],[.3,.4]],device='cuda')}
    context=adapter(obs)
    assert context.shape==(2,66) and torch.equal(context[:,64:],obs['agent_pos'])
    context[:,:64].square().mean().backward()
    for module in (adapter.adapter,adapter.projection):
        grads=[p.grad for p in module.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool(g.abs().max()>0) for g in grads)
    assert not frozen.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    after=model_sha(encoder.model);assert before==after
    r=dict(actual_pretrained_dinov2_loaded=True,backbone_state_before=before,backbone_state_after=after,
        backbone_frozen_and_unchanged=True,token_shape=list(features.shape),token_dtype=str(features.dtype),
        geometry='96 square raw float RGB -> bicubic224 shortest edge and224 center crop; no added field-of-view crop',
        normalization='DINO ImageNet mean/std; rescale once',
        native_dino_last_hidden_final_layernorm=True,
        explicit_deviation_from_published_dino_processor='Resize224 instead of256, to preserve original CLIP camera field of view',
        raw_online_roundtrip_exact=True,fixed_pad8_exact=True,constant_image_normalization_passed=True,
        constant_normalized_image_max_difference=constant_delta,constant_normalization_atol=3e-6,
        actual_eight_layer_adapter_gradients_valid=True,extra_trainable_width_mapping=False,
        original_agent_state_columns_exact=True,text_scope='Synthetic12-token text for interface check; original fixed CLIP text must be bound from qualified server cache before real training',
        source_raw_inputs_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-start,
        versions={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','Pillow')},
        actual_gpu=torch.cuda.get_device_name(0),policy_training_updates=0,rollouts=0,
        closed_loop_generalization_verdict=None,formal_training_ready=False,new_jobs_submitted=False)
    out=ROOT/'reports/dinov2_interface_20261023';out.mkdir(exist_ok=True)
    np.savez(out/'two_train_frame_features.npz',visual=features)
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
