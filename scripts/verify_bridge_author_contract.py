"""CPU audit of raw Bridge conversion, author normalizer and real preflight argv."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
from typing import Union
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'cluster_packages/bridge_author_preflight_v1'
AUTH=ROOT/'baselines/diffusion_policy_data_reference'
OUT=ROOT/'reports/bridge_author_contract_20261014'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parser_statements(path):
    main=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    statements=[]
    for n in main.body:
        statements.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets):break
    return ast.Module(body=statements,type_ignores=[])
def main():
    sys.path.insert(0,str(PACK))
    from bridge_author_io import assemble_references,rgb_from_clip
    p=read(PACK/'protocol.json');e=read(OUT/'export_report.json');m=read(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')
    assert e['passed'] and e['audit_only'] and e['protocol_sha256']==sha(PACK/'protocol.json')
    assert p['rows']==e['rows']==m['train']+m['development']
    assert sha(ROOT/'cluster_packages/bridge_growth128_v1/manifest.json')==p['base_manifest_sha256']
    assert all(sha(AUTH/n)==h for n,h in p['author_source_sha256'].items())
    assert sha(PACK/'language_cache.npz')==p['language_cache_sha256']
    original=read(ROOT/'reports/bridge-growth128-fit-201663/report.json')
    assert all(a['raw_identity']==b['raw_identity'] for a,b in zip(e['data_checks'],original['data_checks']))
    assert len(e['data_checks'])==len(e['input_numeric_comparisons'])==168
    assert all(v['max_abs']<=v['atol'] for row in e['input_numeric_comparisons'] for v in row['differences'].values())
    assert e['state_checks']['raw_current_pose_exact'] and e['state_checks']['train_only_statistics']
    assert not e['state_checks']['target_or_future_state_used_as_context'] and not e['future_observations_used']
    refs=assemble_references(ROOT/'cluster_packages');targets=refs['targets'][...,:7]
    rgb,rgb_delta=rgb_from_clip(refs['images'])
    assert rgb.shape==(168,3,224,224) and e['image_inverse_clip_max_difference']<=1e-6
    assert np.allclose(targets[:128].reshape(-1,7).min(0),e['training_action_min'],atol=1e-6,rtol=0)
    assert np.allclose(targets[:128].reshape(-1,7).max(0),e['training_action_max'],atol=1e-6,rtol=0)
    with np.load(PACK/'language_cache.npz',allow_pickle=False) as z:
        assert z['embeddings'].shape==(5,768) and np.isfinite(z['embeddings']).all()
        assert z['texts'].tolist()==sorted({r['task'] for r in m['train']})
    # Execute the pinned author's pure normalization functions, not a copied approximation.
    tree=ast.parse((AUTH/'diffusion_policy/model/common/normalizer.py').read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_fit','_normalize')]
    assert len(functions)==2
    scope=dict(np=np,torch=torch,nn=torch.nn,Union=Union,zarr=SimpleNamespace(Array=type('UnusedZarr',(),{})))
    exec(compile(ast.Module(body=functions,type_ignores=[]),'pinned_author_normalizer','exec'),scope)
    params=scope['_fit'](targets[:128].copy(),mode='limits')
    normalized=scope['_normalize'](targets,params);restored=scope['_normalize'](normalized,params,forward=False)
    roundtrip=float(np.abs(restored.numpy()-targets).max());assert roundtrip<=2e-6
    train=targets[:128].reshape(-1,7).astype(np.float64)
    scale=2/(train.max(0)-train.min(0));offset=-1-scale*train.min(0)
    assert np.max(np.abs(params['scale'].numpy()-scale))<=1e-5
    assert np.max(np.abs(params['offset'].numpy()-offset))<=1e-5
    mutated=targets.copy();mutated[128:]+=10000
    again=scope['_fit'](mutated[:128].copy(),mode='limits')
    assert all(torch.equal(v,again.state_dict()[k]) for k,v in params.state_dict().items())
    cfg=yaml.safe_load((PACK/'preflight_config.yaml').read_text(encoding='utf-8'))
    assert cfg['training']['max_train_steps']==4 and cfg['training']['num_epochs']==1 and not cfg['training']['resume']
    assert cfg['training']['gradient_accumulate_every']==1 and cfg['training']['use_ema']
    assert cfg['checkpoint']['topk']['k']==0 and not cfg['checkpoint']['save_last_ckpt'] and not cfg['checkpoint']['save_last_snapshot']
    assert cfg['logging']['mode']=='disabled'
    policy=cfg['policy'];assert policy['horizon']==16 and policy['n_obs_steps']==1 and policy['n_action_steps']==16
    assert policy['num_inference_steps']==policy['noise_scheduler']['num_train_timesteps']==100
    assert policy['shape_meta']['action']['shape']==[7] and policy['down_dims']==[512,1024,2048]
    shell=(PACK/'preflight.sh').read_text(encoding='utf-8');assert shell.startswith('#!/bin/bash\n') and '\n+' not in shell
    launchers=[]
    for entry in ('export_bridge_author_inputs.py','preflight_bridge_author_workspace.py'):
        begin=shell.index(' -u '+entry)+len(' -u '+entry)
        block=shell[begin:].replace('\\\n',' ').splitlines()[0]
        argv=shlex.split(block)
        before=sys.argv;sys.argv=[entry]+argv
        try:
            env=dict(argparse=argparse,Path=Path,__doc__='parser audit')
            exec(compile(parser_statements(PACK/entry),'real_cluster_argv','exec'),env)
        finally:sys.argv=before
        launchers.append(dict(entry=entry,argv=argv))
    dataset=ast.parse((PACK/'bridge_author_dataset.py').read_text(encoding='utf-8'))
    normalizer=next(n for n in ast.walk(dataset) if isinstance(n,ast.FunctionDef) and n.name=='get_normalizer')
    assert "[:128]" in ast.unparse(normalizer) and 'limits' in ast.unparse(normalizer)
    runner=next(n for n in dataset.body if isinstance(n,ast.ClassDef) and n.name=='NoRolloutRunner')
    assert ast.unparse(runner.body[0].body[0])=='return {}'
    result=dict(passed=True,protocol_sha256=sha(PACK/'protocol.json'),export_report_sha256=sha(OUT/'export_report.json'),
        raw_windows=168,training_statistics_only=True,author_normalizer_functions_executed=True,
        author_normalizer_roundtrip_max_difference=roundtrip,independent_limits_formula_checked=True,
        development_perturbation_does_not_change_fit=True,image_conversion_roundtrip_max_difference=rgb_delta,
        author_workspace_gpu_executed_locally=False,packages_installed=False,new_weight_downloads=False,
        actual_launchers_parsed=launchers,full_training_or_simulation_executed=False)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
