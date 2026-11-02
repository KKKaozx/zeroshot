"""CPU checks of NEW metric/admission/orchestration code; no old GPU experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import numpy as np
import torch
from bridge_prefix_pair_metrics_v1 import summarize,admission

ROOT=Path(__file__).resolve().parents[1]


def main():
    # Two episodes, different window counts, and sign-equivalent quaternions.
    rows=[dict(shard='a',record_index=0,task='x')]+[dict(shard='b',record_index=1,task='y')]*3
    truth=np.zeros((4,16,7));truth[...,6]=1.
    pred=np.repeat(truth[None],2,axis=0);pred[...,3:7]*=-1.
    pred[:,:, :,0]=np.asarray([.1,.3,.3,.3])[None,:,None]
    s=summarize(pred,truth,rows)
    assert abs(s['episode_equal']['path_position_cm']-2.)<1e-12
    assert abs(s['window_weighted']['path_position_cm']-2.5)<1e-12
    assert s['episode_equal']['path_rotation_deg']==0.
    # A 90-degree relative z rotation; quaternion sign must not change this.
    pred[:]=truth[None];pred[...,5]=np.sqrt(.5);pred[...,6]=np.sqrt(.5)
    assert abs(summarize(pred,truth,rows)['episode_equal']['path_rotation_deg']-90.)<1e-10
    # Root of average squared errors or an average prediction must not replace
    # the per-sample Euclidean error; opposite seed errors cannot cancel.
    pred[:]=truth[None];pred[0,...,0]=.1;pred[1,...,0]=-.1
    assert abs(summarize(pred,truth,rows)['episode_equal']['path_position_cm']-1.)<1e-12
    pred[0,0,0,3:7]=0.
    try:summarize(pred,truth,rows)
    except ValueError:pass
    else:raise AssertionError('Invalid quaternion was scored')

    def metric(value):
        return dict(path_position_cm=value,path_rotation_deg=value,
                    endpoint_position_cm=value,endpoint_rotation_deg=value)
    def method(value):
        return dict(episode_equal=metric(value),window_weighted=metric(value),
            per_episode=[dict(episode=str(i),**metric(value)) for i in range(10)])
    results={n:method(2.) for n in ('control','unchanged207558','static')}
    results['candidate']=method(1.)
    assert admission(results)['development_admission_passed']
    # Each comparison wins six episodes, but ONLY TWO wins overlap: reject.
    for i,e in enumerate(results['control']['per_episode']):e['path_position_cm']=2. if i<6 else .5
    for i,e in enumerate(results['unchanged207558']['per_episode']):e['path_position_cm']=2. if i>=4 else .5
    a=admission(results)
    assert len(a['jointly_improved_episode_ids'])==2 and not a['development_admission_passed']
    results={n:method(2.) for n in ('control','unchanged207558','static')};results['candidate']=method(1.)
    results['candidate']['window_weighted']['path_rotation_deg']=3.
    assert not admission(results)['development_admission_passed']

    # Exercise the ACTUAL new update body with a tiny CPU model, replacing only
    # the CUDA transport/timer. No robot data or pretrained policy is loaded.
    path=ROOT/'scripts/run_bridge_prefix_formal_v1.py'
    module=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
    # Use8 instead of4096 for this engineering test; all other statements remain.
    class Limit(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==4096:return ast.copy_location(ast.Constant(8),node)
            if node.value==4097:return ast.copy_location(ast.Constant(9),node)
            return node
    fn=Limit().visit(copy.deepcopy(fn));ast.fix_missing_locations(fn)
    import time,hashlib
    def fingerprint(value):
        h=hashlib.sha256()
        def visit(v):
            if torch.is_tensor(v):h.update(v.detach().cpu().numpy().tobytes())
            elif isinstance(v,dict):
                for k in sorted(v):h.update(str(k).encode());visit(v[k])
            else:h.update(repr(v).encode())
        visit(value);return h.hexdigest()
    ns=dict(torch=torch,np=np,time=time,copy=copy,json=json)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.base=torch.nn.Linear(1,1,bias=False)
            self.prefix=types.SimpleNamespace(request_projection=torch.nn.Linear(16,1,bias=False))
            self.request=self.prefix.request_projection
        def loss(self,targets,validity,context,K):
            pred=self.base(context)+self.request(torch.ones((len(context),16))*(K==1))
            return (pred-targets[:,0,:1]).square().mean(),None,pred
        def trainable_payload(self):
            return dict(base=copy.deepcopy(self.base.state_dict()),
                        request_projection_state_dict=copy.deepcopy(self.request.state_dict()))
        def load_trainable_payload(self,p):
            self.base.load_state_dict(p['base']);self.request.load_state_dict(p['request_projection_state_dict'])
    toy=Tiny();toy.base.weight.data.fill_(.25)
    payload=dict(trainable_state_dict=copy.deepcopy(toy.base.state_dict()),config={})
    data=dict(targets=torch.ones((4,16,7)),validity=torch.ones((4,16),dtype=torch.bool))
    schedule=np.tile([0,1],(8,1));horizons=np.tile([16,1],4)
    helpers=types.SimpleNamespace(fingerprint=fingerprint,sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest())
    report=dict(protocol_sha256='test',initial_checkpoint_sha256='test',formal_source_sums_sha256='test')
    original_sync=torch.cuda.synchronize;torch.cuda.synchronize=lambda:None
    # Avoid a harmless empty timing slice in an8-update smoke.
    original_mean=np.mean
    np.mean=lambda a,*args,**kwargs:original_mean(a if len(a) else [0.],*args,**kwargs)
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            out=Path(folder)
            first=ns['run_arm'](toy,payload,list(toy.parameters()),lambda ids:torch.ones((len(ids),1)),
                data,schedule,horizons,'control',out,helpers,report)
            second=ns['run_arm'](toy,payload,list(toy.parameters()),lambda ids:torch.ones((len(ids),1)),
                data,schedule,horizons,'candidate',out,helpers,report)
            assert first['initial_trainable_sha256']==second['initial_trainable_sha256']
            assert first['first8_losses']==second['first8_losses']
            assert first['final_trainable_sha256']==second['final_trainable_sha256']
    finally:torch.cuda.synchronize=original_sync;np.mean=original_mean
    evidence=dict(passed=True,GPU_used=False,robot_policy_loaded=False,
        metric_checks=['XYZ units','xyzw sign invariance','90degree rotation','episode weighting',
            'seed errors do not cancel','invalid quaternion rejected'],
        admission_checks=['conjunction','same-six intersection','window regression rejection'],
        orchestration='Actual AST update/save/load body,8-update tiny CPU fixture; identical reset/freshAdam arms match',
        formal_training_executed=False,preflight_repeated=False,generalization_verdict=False)
    dest=ROOT/'reports/bridge_prefix_formal_preparation_checks_20261017.json'
    dest.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
