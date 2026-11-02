"""Exercise the actual update-mask hooks with a tiny CPU model, not project training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_fixed_context_v1'
OUT = ROOT/'reports/bridge_fixed_context_cpu_20261017'


def main():
    assert not OUT.exists(), 'CPU validation already completed.'
    spec = importlib.util.spec_from_file_location('actual_fixed_hooks', PACK/'fixed_hooks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tree = ast.parse((ROOT/'cluster_packages/bridge_state_conditioning_v1/fit_state_arm.py').read_text())
    ns = dict(torch=torch, np=np, hashlib=hashlib, copy=copy,
              PREFIXES=('adapter.', 'diffusion_decoder.', 'history_projection.', 'state_projection.'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('cpu_copy','state_sha','trainable_copy')]
    assert len(functions) == 3
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'frozen_serialization_helpers', 'exec'), ns)
    torch.set_num_threads(2); torch.manual_seed(918)
    m = torch.nn.Module()
    m.adapter = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Dropout(0.25))
    m.history_projection = torch.nn.Linear(3,3,bias=False)
    m.state_projection = torch.nn.Linear(2,3,bias=False)
    m.diffusion_decoder = torch.nn.Linear(3,2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    h = module.FixedHooks(PACK, {}); h.ns = ns
    samples = torch.tensor([[.2,-.4],[.8,.3]])
    truth = torch.tensor([[4.,3.],[-2.,7.]])

    def update(index):
        torch.manual_seed(430000+index); m.train()
        now=m.adapter(samples); repeat=m.adapter(samples)
        ctx=now+m.history_projection(repeat-now)+m.state_projection(samples)
        torch.manual_seed(420000+index)
        noise=torch.randn_like(truth)*.01
        loss=(m.diffusion_decoder(ctx)-truth-noise).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        shadow=float(sum(p.grad.square().sum() for n,p in m.named_parameters() if h.context_name(n)))
        norm=float(torch.nn.utils.clip_grad_norm_(list(m.parameters()),1.,error_if_nonfinite=True))
        h.mask_updates(m); opt.step(); h.after_step(m,index)
        return float(loss.detach()), norm, shadow

    for index in range(1,5): update(index)
    prefix=ns['cpu_copy'](m.state_dict()); adam=ns['cpu_copy'](opt.state_dict())
    before=h.context_hash(m); before_adam=h.context_adam_hash(m,opt)
    def restore():
        m.load_state_dict(prefix); opt.load_state_dict(ns['cpu_copy'](adam)); h.first_decoder=None
    h.first_decoder=None
    control=[update(i) for i in range(5,9)]; first=h.first_decoder
    assert h.context_hash(m)!=before
    restore(); h.block=True
    fixed=[update(i) for i in range(5,9)]
    assert fixed[0]==control[0] and h.first_decoder==first
    assert fixed[0][1]>1. and fixed[0][2]>0.
    assert h.context_hash(m)==before and h.context_adam_hash(m,opt)==before_adam
    expected=ns['state_sha'](m.state_dict()); expected_adam=ns['state_sha'](opt.state_dict())
    assert ns['state_sha']({n:p for n,p in m.state_dict().items() if n.startswith('diffusion_decoder.')}) != ns['state_sha']({n:p for n,p in prefix.items() if n.startswith('diffusion_decoder.')})
    restore()
    assert [update(i) for i in range(5,9)]==fixed
    assert ns['state_sha'](m.state_dict())==expected and ns['state_sha'](opt.state_dict())==expected_adam
    OUT.mkdir()
    temporary=OUT/'tiny_state.pt'
    torch.save(dict(model=ns['cpu_copy'](m.state_dict()),adam=ns['cpu_copy'](opt.state_dict())),temporary)
    loaded=torch.load(temporary,map_location='cpu',weights_only=False)
    assert ns['state_sha'](loaded['model'])==expected and ns['state_sha'](loaded['adam'])==expected_adam
    temporary.unlink()
    # Check callback order and all source syntax without importing production models.
    effective=(PACK/'effective_runner.py.txt').read_text()
    for path in PACK.glob('*.py'): compile(path.read_text(),str(path),'exec')
    compile(effective,'effective_runner','exec')
    order=[effective.index(s) for s in ('norm = torch.nn.utils.clip_grad_norm_', 'FIXED.mask_updates(model)', 'optimizer.step()', 'FIXED.after_step(model, update)')]
    assert order==sorted(order)
    assert "FIXED.run(locals())\n        return" in effective
    assert "common = advance('common'" not in effective
    # The actual production advance closes over batches; validate replacement is writable.
    def outer():
        batches=torch.arange(8192)
        def advance(index): return int(batches[index])
        return advance
    closure=outer(); cells=dict(zip(closure.__code__.co_freevars,closure.__closure__))
    cells['batches'].cell_contents=torch.arange(16170)
    assert closure(16169)==16169
    record=dict(passed=True, tiny_cpu_parameters=sum(p.numel() for p in m.parameters()),
        project_model_executed=False,gpu_used=False,actual_mask_and_hash_hooks_used=True,
        train_dropout_preserved=True,first_decoder_update_exact=True,clipping_active_in_probe=True,
        context_gradients_nonzero_but_parameters_and_adam_unchanged=True,
        masked_adam_continuation_repeated_exact=True,serialization_roundtrip_exact=True,
        tiny_checkpoint_removed=True,source_compilation_passed=True,
        clip_mask_step_callback_order_correct=True,original_prefix_training_removed=True,
        closure_schedule_replacement_checked=True,
        frozen_protocol_sha256=module.sha(PACK/'protocol.json'),
        entry_sha256=module.sha(Path(__file__)),torch_version=str(torch.__version__))
    (OUT/'validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
