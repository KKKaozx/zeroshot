"""Compare existing real frozen profiles; no encoder reconstruction or forward."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_KEYS = {'betas','alphas','alpha_bars','posterior_variance','posterior_mean_x0','posterior_mean_xt'}


def main():
    import numpy as np
    import torch
    folder = ROOT / 'reports/oxe_noaug_seed42_227223'
    expected = json.loads((folder / 'cluster/preflight.json').read_text(encoding='utf-8'))
    cached = json.loads((ROOT / 'reports/frozen_identity_failure_20261025/process1.json').read_text(encoding='utf-8'))
    verified = json.loads((folder / 'download_verification.json').read_text())
    runtime = json.loads((ROOT / 'reports/oxe_seed42_libero_engineering_20261026/attempt1/report.json').read_text())
    message = runtime['errors'][0]
    actual = json.loads(message[message.index('{'):])
    assert actual['tokenizer_matches'] and not actual['tokenizer_changed_fields']
    assert cached['frozen_sha256'] == actual['actual_frozen_sha256']
    assert expected['legacy_frozen_sha256'] == actual['expected_frozen_sha256'] == verified['frozen_state_sha256']
    assert expected['tokenizer_after_model'] == cached['tokenizer']
    a,b = expected['tensors'],cached['frozen_tensors']
    missing,extra = sorted(set(a)-set(b)),sorted(set(b)-set(a))
    changed = [k for k in sorted(set(a)&set(b)) if a[k]!=b[k]]
    result = {'expected_cluster_frozen_sha256':expected['legacy_frozen_sha256'],
        'actual_local_frozen_sha256':cached['frozen_sha256'],
        'local_profile_matches_actual_loader_failure':True,
        'same_key_order':expected['key_order']==cached['frozen_key_order'],
        'missing_keys':missing,'extra_keys':extra,'changed_keys':changed,
        'changed_encoder_keys':[k for k in changed if k.startswith(('vision_encoder.','text_encoder.'))],
        'changed_schedule_keys':[k for k in changed if k in SCHEDULE_KEYS],
        'unchanged_frozen_tensor_count':len(set(a)&set(b))-len(changed),
        'local_torch_version':torch.__version__, 'guard_weakened':False,
        'model_forward_calls':0, 'optimizer_updates':0, 'neural_models_constructed':0}
    # Recompute only the six small constants, and require their bytes to match
    # the previously captured actual local profile before reporting differences.
    fn = next(n for n in ast.parse((ROOT/'code/models.py').read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='diffusion_betas')
    ns={'torch':torch,'math':math}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_diffusion_betas','exec'),ns)
    cfg=expected['config']['model']
    beta=ns['diffusion_betas'](cfg['num_diffusion_steps'],cfg['beta_schedule'])
    alpha=1-beta; bars=torch.cumprod(alpha,0)
    previous=torch.nn.functional.pad(bars[:-1],(1,0),value=1.)
    values={'betas':beta,'alphas':alpha,'alpha_bars':bars,
        'posterior_variance':(beta*(1-previous)/(1-bars)).clamp(min=1e-20),
        'posterior_mean_x0':beta*previous.sqrt()/(1-bars),
        'posterior_mean_xt':(1-previous)*alpha.sqrt()/(1-bars)}
    result['schedule_differences']={}
    for name,tensor in values.items():
        raw=tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        assert hashlib.sha256(raw).hexdigest()==b[name]['sha256']
        x=np.asarray(expected['schedule_values'][name],dtype=np.float64)
        y=tensor.numpy().astype(np.float64)
        result['schedule_differences'][name]={'max_abs':float(np.max(np.abs(x-y))),
            'unequal_values':int(np.count_nonzero(x!=y))}
    if not missing and not extra and result['same_key_order'] and set(changed)<=SCHEDULE_KEYS and changed:
        result['classification']='Frozen encoder tensors identical; only derived diffusion constants differ across runtimes'
    else:
        result['classification']='Further named differences exist; do not substitute constants or relax the identity guard'
    (folder/'frozen_runtime_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
