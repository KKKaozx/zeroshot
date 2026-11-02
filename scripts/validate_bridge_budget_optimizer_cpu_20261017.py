"""Exercise the actual checkpoint writer and Adam continuation on a tiny CPU model."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT/'cluster_packages/bridge_demo806_budget_v1'
OUT = ROOT/'reports/bridge_budget_optimizer_cpu_20261017'


def main():
    OUT.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('actual_budget_hooks', PACK/'budget_hooks.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Extract the unchanged production serialization/hash helpers, avoiding model imports.
    source = ROOT/'cluster_packages/bridge_state_conditioning_v1/fit_state_arm.py'
    tree = ast.parse(source.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('trainable_copy', 'state_sha', 'cpu_copy')]
    assert len(functions) == 3
    namespace = dict(torch=torch, np=np, hashlib=hashlib, copy=copy, PREFIXES=('adapter.',))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    torch.manual_seed(923)
    model = torch.nn.Module(); model.adapter = torch.nn.Linear(2, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = torch.tensor([[0.2, -0.4], [0.8, 0.3]])
    truth = torch.tensor([[0.4, 0.3, 0.6], [-0.2, 0.1, 0.7]])

    def advance(count):
        for _ in range(count):
            loss = (model.adapter(samples)-truth).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

    advance(4)
    hooks = module.BudgetHooks(PACK, json.loads((PACK/'protocol.json').read_text()))
    hooks.ns = namespace
    x = dict(args=SimpleNamespace(output=OUT, stage='cpu_micro_probe'), cfg={}, p={}, model=model,
             optimizer=optimizer, r={}, step_seconds=[0.0]*4)
    path = hooks.save_state(x, 'tiny_adam_probe.pt', 4)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    advance(2)
    expected_model = namespace['state_sha'](namespace['trainable_copy'](model))
    expected_adam = namespace['state_sha'](optimizer.state_dict())
    model.load_state_dict(loaded['trainable_state_dict'], strict=False)
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    advance(2)
    assert namespace['state_sha'](namespace['trainable_copy'](model)) == expected_model
    assert namespace['state_sha'](optimizer.state_dict()) == expected_adam
    # Demonstrate why weight-only loading is not equivalent to resuming Adam.
    model.load_state_dict(loaded['trainable_state_dict'], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    advance(2)
    assert namespace['state_sha'](namespace['trainable_copy'](model)) != expected_model
    record = dict(passed=True, tiny_cpu_model_parameters=9, project_model_executed=False,
                  gpu_used=False, actual_checkpoint_writer_exercised=True,
                  full_adam_reload_continuation_exact=True, weights_only_restart_differs=True,
                  production_hash_and_copy_helpers_used=True, source_sha256=module.sha(source),
                  checkpoint_metadata=x['r']['optimizer_checkpoints'][0])
    del loaded
    path.unlink()
    record['tiny_checkpoint_removed_after_verification'] = True
    (OUT/'validation.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
