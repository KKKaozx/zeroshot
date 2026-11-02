"""Tiny synthetic eight-block wiring check; NOT production weights or CLIP."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from bridge_spatial_readout_interface_v1 import install_readout,context_from_tokens

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'reports/bridge_spatial_interface_cpu_20261018.json'
    if out.exists():raise FileExistsError('Preserve completed check')
    torch.set_num_threads(4);torch.manual_seed(181026)
    source=ROOT/'cluster_packages/bridge_prefix_pair_v1/frozen/adapter.py'
    spec=importlib.util.spec_from_file_location('legacy_spatial_interface_reference',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reference=torch.nn.Module()
    reference.adapter=module.CrossAttentionAdapter(num_layers=8,embed_dim=32,text_dim=24,
        attention_dim=16,num_heads=4,dropout=.1,pooling='cls_patch_mean')
    reference.history_projection=torch.nn.Linear(32,32,bias=False)
    reference.state_projection=torch.nn.Linear(9,32,bias=False)
    changed=copy.deepcopy(reference)
    original_keys=set(changed.state_dict())
    readout=install_readout(changed.adapter,candidate=True)
    assert set(changed.state_dict())-original_keys=={
        'adapter.spatial_readout.content_score','adapter.spatial_readout.position_score','adapter.spatial_readout.patch_xy'}
    before={k:v.clone() for k,v in changed.state_dict().items()}
    visual=torch.randn(2,257,32);language=torch.randn(2,5,24)
    mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    current=torch.randn(2,9);upstream=torch.randn(2,32)
    deltas={}
    for training in (False,True):
        reference.train(training);changed.train(training)
        reference.zero_grad(set_to_none=True);changed.zero_grad(set_to_none=True)
        torch.manual_seed(99)
        a=context_from_tokens(reference,visual,language,mask,current)
        torch.manual_seed(99)
        b=context_from_tokens(changed,visual,language,mask,current)
        delta=float((a-b).detach().abs().max());assert delta==0,delta
        (a*upstream).sum().backward();(b*upstream).sum().backward()
        reference_parameters=dict(reference.named_parameters());changed_parameters=dict(changed.named_parameters())
        gradient_delta=max(float((v.grad-changed_parameters[k].grad).abs().max())
            for k,v in reference_parameters.items())
        assert gradient_delta<=1e-5,gradient_delta
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in readout.parameters())
        deltas['train_with_dropout' if training else 'eval']=dict(context_max_abs=delta,
            base_gradient_max_abs=gradient_delta)
    assert all(torch.equal(v,changed.state_dict()[k]) for k,v in before.items())
    try:install_readout(changed.adapter,candidate=True)
    except ValueError:pass
    else:raise AssertionError('Double installation must fail')
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,scope='Eight small32D archived cross-attention blocks with synthetic tokens; production1024D trained model NOT executed',
        original_parameter_keys_preserved=True,new_state_keys=sorted(set(changed.state_dict())-original_keys),
        deltas=deltas,parameters_not_updated=True,optimizer_created=False,training_updates=0,
        CLIP_loaded=False,old207558_weights_loaded=False,real_data_read=False,GPU_initialized=False,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (source,Path(__file__),ROOT/'scripts/bridge_spatial_readout_interface_v1.py')})
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
