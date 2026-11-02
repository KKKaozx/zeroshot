"""Exercise actual formal loop on a tiny synthetic model, not a research fit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'cluster_packages/bridge_spatial_formal_v1'
PRE=ROOT/'cluster_packages/bridge_spatial_readout_v1'
OUT=ROOT/'reports/bridge_spatial_formal_cpu_20261018'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    if OUT.exists():raise FileExistsError('Preserve completed check')
    OUT.mkdir();torch.set_num_threads(4)
    sys.path[:0]=[str(CODE),str(PRE),str(PRE/'frozen')]
    P=load('spatial_actual_preflight_helpers',PRE/'run_preflight.py')
    F=load('spatial_actual_formal_runner',CODE/'run_formal.py')
    M=load('spatial_actual_formal_metrics',CODE/'bridge_spatial_pair_metrics_v1.py')
    A=load('spatial_tiny_adapter_source',PRE/'frozen/adapter.py')
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter=A.CrossAttentionAdapter(num_layers=8,embed_dim=32,text_dim=24,
                attention_dim=16,num_heads=4,dropout=.1,pooling='cls_patch_mean')
            self.history_projection=torch.nn.Linear(32,32,bias=False)
            self.state_projection=torch.nn.Linear(9,32,bias=False)
            self.diffusion_decoder=torch.nn.Sequential(torch.nn.Linear(32,64),torch.nn.Mish(),torch.nn.Linear(64,112))
            self.frozen_marker=torch.nn.Linear(2,2).requires_grad_(False)
        def diffusion_loss(self,actions,context,current_gripper):
            times=torch.randint(0,100,(len(actions),),device=actions.device)
            noise=torch.randn_like(actions[:,:,:7])
            alpha=(1.-times.float()/101.).reshape(-1,1,1)
            noisy=alpha.sqrt()*actions[:,:,:7]+(1.-alpha).sqrt()*noise
            guess=self.diffusion_decoder(context).reshape(-1,16,7)+.1*noisy
            return guess,actions[:,:,:7],None
    torch.manual_seed(99);model=Tiny()
    keys={n for n,v in model.named_parameters() if v.requires_grad}
    payload=dict(trainable_state_dict={n:v.detach().clone() for n,v in model.named_parameters() if n in keys},
        config={'synthetic_test_only':True},source_checkpoint_sha256='0'*64)
    marker=copy.deepcopy(model.frozen_marker.state_dict())
    readout=P.install_readout(model.adapter,candidate=True)
    visual=torch.randn(24,257,32);language=torch.randn(24,5,24);mask=torch.ones(24,5,dtype=torch.long)
    state=torch.randn(24,9);current=torch.randn(24,1);targets=torch.randn(24,16,8)
    batches=np.random.default_rng(721026).integers(0,24,size=(8,8),dtype=np.int64)
    def context(ids):return P.context_from_tokens(model,visual[ids],language[ids],mask[ids],state[ids])
    runs=[]
    for arm in ('control','candidate'):
        runs.append(F.run_arm(model,readout,payload,keys,context,targets,current,batches,arm,OUT,P,None,updates=8))
    assert runs[0]['initial_base_sha256']==runs[1]['initial_base_sha256']
    assert runs[0]['first8_losses'][0]==runs[1]['first8_losses'][0]
    assert runs[1]['trainable_parameters']-runs[0]['trainable_parameters']==34
    assert all(torch.equal(v,model.frozen_marker.state_dict()[k]) for k,v in marker.items())
    rows=[dict(shard=f'synthetic{i}',record_index=0,task=f'task{i//2}') for i in range(10)]
    y=np.zeros((10,16,8));y[:,:,0]=.1;y[:,:,6]=1.
    original=np.tile(y[None,:,:,:7],(3,1,1,1));original[:,:,:,0]=.2
    control=original.copy();control[:,:,:,0]=.25
    candidate=original.copy();candidate[:,:,:,0]=.11
    identity=np.zeros((1,10,16,7));identity[:,:,:,6]=1.
    def result(c):return {n:M.summarize_pair(a,y,rows) for n,a in
        [('unchanged207558',original),('control',control),('candidate',c),('static',identity)]}
    successful=M.admission(result(candidate));assert successful['development_admission_passed']
    same=M.admission(result(original));assert not same['development_admission_passed']
    first_bad=np.tile(y[None,:,:,:7],(3,1,1,1));first_bad[:,:,0,0]=.4
    first_failure=M.admission(result(first_bad))
    assert not first_failure['development_admission_passed']
    assert first_failure['gates']['full16_path_and_endpoint_episode_no_worse']
    assert not first_failure['gates']['first_episode_position_better_than_all']
    rotation_bad=candidate.copy();rotation_bad[:,:,0,3]=np.sin(np.deg2rad(10)/2)
    rotation_bad[:,:,0,6]=np.cos(np.deg2rad(10)/2)
    assert not M.admission(result(rotation_bad))['gates']['first_episode_rotation_no_worse_than_all']
    invalid=candidate.copy();invalid[:,:,:,3:7]=0.
    try:result(invalid)
    except ValueError:pass
    else:raise AssertionError('Degenerate quaternions must fail')
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,scope='Actual formal run_arm with tiny32D eight-block Adapter and toy denoiser; synthetic inputs only',
        toy_updates_per_arm=8,formal4096_not_executed=True,CLIP_loaded=False,
        actual207558_loaded=False,real_data_read=False,GPU_initialized=False,
        same_initial_base=True,same_first_loss=True,checkpoint_roundtrips_exact=True,
        frozen_marker_unchanged=True,admission_good_case_passes=True,
        admission_unchanged_candidate_rejected=True,
        admission_full16_better_but_first_bad_rejected=True,
        admission_first_rotation_regression_rejected=True,invalid_quaternions_rejected=True,
        runs=runs)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
