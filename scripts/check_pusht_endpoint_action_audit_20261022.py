"""New-entry CPU fixtures; no actual checkpoints, CLIP or environment execution."""
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'scripts/cluster/pusht_endpoint_action_audit_v1'
sys.path.insert(0,str(B))
from audit_math import reference_indices, select_windows, summarize, summarize_epsilon
from endpoint_ops import frozen_batch
from verify_output import validate
from resource_probe import sha
checks=[]


def ok(name,condition=True):
    assert condition,name
    checks.append(name)


def rejects(name,fn):
    try: fn()
    except (ValueError,AssertionError,KeyError,IndexError): checks.append(name);return
    raise AssertionError('Did not reject: '+name)


protocol=json.loads((B/'protocol.json').read_text())
for line in (B/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split(None,1); assert sha(B/name.strip())==digest
ok('All18 published file checksums')
for f in B.glob('*.py'): ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,9))
ok('Python3.9 syntax for all modules')
with zipfile.ZipFile(ROOT/'cluster_packages/pusht_endpoint_action_audit_v1.zip') as z:
    files=sorted(p for p in B.iterdir() if p.is_file())
    ok('Archive exact18 members',len(z.namelist())==len(files)==18)
    ok('Archive bytes match source',all(z.read(B.name+'/'+f.name)==f.read_bytes() for f in files))
source=ast.parse((ROOT/'baselines/diffusion_policy_data_reference/diffusion_policy/common/sampler.py').read_text())
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='create_indices')
fn.decorator_list=[]
namespace={'np':np}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned author sampler>','exec'),namespace)
author=namespace['create_indices']
for length in (1,2,8,9,15,16,17,30,101):
    ends=np.cumsum(np.array([length,31,18,9,27],dtype=np.int64))
    for mask in (np.ones(5,dtype=bool),np.array([True,False,True,False,True])):
        assert np.array_equal(author(ends,16,mask,1,7),reference_indices(ends,16,mask,1,7))
ok('Independent row rule matches author AST across18 variable-length/mask cases')
ends=np.cumsum(np.arange(60,60+206,dtype=np.int64))
info,records=select_windows(ends,protocol['selected_episode_ids'],author)
ok('64 exact windows, no episode substitutions',len(records)==64 and len(set(info['episode_ids']))==32)
ok('32 train and32 unseen windows',sum(info['roles']=='train')==sum(info['roles']=='unseen_demo')==32)
ok('Reference sampler gives exact manifest',select_windows(ends,protocol['selected_episode_ids'],reference_indices)[1]==records)
ok('All observation frames are first2 target slots',all(r['observation_frame_ids']==r['frame_ids'][:2] for r in records))
bad=copy.deepcopy(protocol['selected_episode_ids']);bad['unseen_demo'][0]=bad['train'][0]
rejects('Cross-role episode overlap',lambda:select_windows(ends,bad,author))
rejects('Short episode stops without resampling',lambda:select_windows(np.cumsum(np.full(206,8)),protocol['selected_episode_ids'],author))
rejects('Non-increasing ends',lambda:select_windows(np.zeros(206),protocol['selected_episode_ids'],author))
target=np.zeros((64,16,2),dtype=np.float32)
pred=np.zeros((2,64,16,2),dtype=np.float32);pred[...,0]=3;pred[...,1]=4
valid=info['valid'].copy();ep=info['episode_ids'];roles=info['roles']
m=summarize(pred,target,valid,ep,roles)
ok('Euclidean3/4 displacement is5 pixels',m['train']['full16']['episode_equal_mean_pixels']==5)
padded=valid.copy();padded[:,0]=False
altered=pred.copy();altered[:,:,0]=10000
ok('Padding excluded from metric',summarize(altered,target,padded,ep,roles)['train']['full16']['episode_equal_mean_pixels']==5)
altered=pred.copy();altered[:,:,0]=10000;altered[:,:,9:]=10000
ok('Execution slice is1:9 rather than0:8',summarize(altered,target,valid,ep,roles)['train']['executed8']['episode_equal_mean_pixels']==5)
different=pred.copy();different[1]*=2
ok('Seeds averaged rather than best selected',summarize(different,target,valid,ep,roles)['unseen_demo']['executed8']['episode_equal_mean_pixels']==7.5)
rejects('Wrong prediction horizon',lambda:summarize(pred[:,:,:8],target,valid,ep,roles))
nan=pred.copy();nan[0,0,0,0]=np.nan
rejects('Nonfinite generation',lambda:summarize(nan,target,valid,ep,roles))
rejects('Missing metric mask',lambda:summarize(pred,target,np.zeros_like(valid),ep,roles))
rejects('Epsilon missing seeds',lambda:summarize_epsilon(np.ones((1,3,64)),ep,roles))
ok('Auxiliary epsilon errors retain frozen times',set(summarize_epsilon(np.ones((2,3,64)),ep,roles)['train'])=={'0','49','99'})


class Normalizer(nn.Module):
    def normalize(self,x): return x
    def __getitem__(self,k): return self


class Encoder(nn.Module):
    def __init__(self): super().__init__();self.cached_tokens=None
    def set_tokens(self,t): self.cached_tokens=t
    def forward(self,obs): return obs['agent_pos'] + (0 if self.cached_tokens is None else self.cached_tokens)


class Backend(nn.Module):
    def __init__(self): super().__init__();self.weight=nn.Parameter(torch.tensor(.01))
    def forward(self,x,t,local_cond=None,global_cond=None): return x*self.weight + global_cond.mean(1)[:,None,None]*.01


class Policy(nn.Module):
    def __init__(self):
        super().__init__();self.model=Backend();self.obs_encoder=Encoder();self.normalizer=Normalizer();self.noise_scheduler=self
    @property
    def device(self): return self.model.weight.device
    def add_noise(self,x,noise,t): return x + noise*(t[:,None,None]+1)/100
    def predict_action(self,obs):
        # Deterministic fake scheduler to exercise actual passive hooks/RNG paths.
        n=len(obs['agent_pos']);flat={k:v.reshape(-1,*v.shape[2:]) for k,v in obs.items()}
        cond=self.obs_encoder(flat).reshape(n,-1)
        x=torch.randn(n,16,2)
        for t in reversed(range(100)):
            x=x-self.model(x,torch.tensor(t),global_cond=cond)+torch.randn_like(x)*.001
        return dict(action_pred=x,action=x[:,1:9])


# Avoid registering itself as a child nn.Module in the fixture scheduler.
class Scheduler:
    def add_noise(self,x,noise,t): return x+noise*(t[:,None,None]+1)/100


def fixture():
    policy=Policy.__new__(Policy);nn.Module.__init__(policy)
    policy.model=Backend();policy.obs_encoder=Encoder();policy.normalizer=Normalizer();policy.noise_scheduler=Scheduler()
    return policy.requires_grad_(False).eval()


policy=fixture();obs={'image':torch.zeros(4,2,3,2,2),'agent_pos':torch.arange(16,dtype=torch.float32).reshape(4,2,2)}
targets=torch.zeros(4,16,2)
original={k:v.clone() for k,v in policy.state_dict().items()}
a,e,t=frozen_batch(policy,obs,targets,None,1001,0)
tokens=torch.ones(8,2)
b,f,u=frozen_batch(policy,obs,targets,tokens,1001,0)
ok('Passive sampling captures exact common noise/RNG despite different contexts',t==u)
ok('Fake contexts alter actions while leaving common noise unchanged',not np.array_equal(a,b))
ok('Generated/epsilon array shapes',a.shape==(4,16,2) and e.shape==(3,4))
ok('Context cleared and no parameter update',policy.obs_encoder.cached_tokens is None and all(torch.equal(v,original[k]) for k,v in policy.state_dict().items()))
ok('No gradients created',all(p.grad is None for p in policy.parameters()))
policy.train();rejects('Reject train mode',lambda:frozen_batch(policy,obs,targets,None,1001,0));policy.eval()
policy.model.weight.requires_grad_(True)
rejects('Reject trainable parameter',lambda:frozen_batch(policy,obs,targets,None,1001,0));policy.requires_grad_(False)
original_predict=policy.predict_action
def wrong_slice(obs):
    v=original_predict(obs);v['action']=v['action_pred'][:,:8];return v
policy.predict_action=wrong_slice
rejects('Wrong executed slice caught by actual helper',lambda:frozen_batch(policy,obs,targets,tokens,1001,0))
ok('Exception removes passive hook and clears tokens',not policy.model._forward_pre_hooks and policy.obs_encoder.cached_tokens is None)

# Complete small-output fixture exercises actual validator, including rejection paths.
with tempfile.TemporaryDirectory(dir=ROOT/'reports',prefix='action_audit_check_') as tmp:
    out=Path(tmp)
    def save(name,x): (out/name).write_text(json.dumps(x))
    positions=np.ones((64,2,2),dtype=np.float32)*2
    raw=np.zeros((64,2,96,96,3),dtype=np.float32)
    mean=np.array([3,4],dtype=np.float32)
    np.savez_compressed(out/'inputs.npz',**info,episode_ends=ends,raw_images=raw,targets=target,agent_pos=positions,train_action_mean=mean)
    save('window_manifest.json',dict(windows=records,execution_slice=[1,9],selected_raw_pixels_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    r=dict(complete=True,errors=[],training_updates=0,environment_steps=0,policy_rollouts=0,final50_accessed=False,
        automatic_submission=False,protocol_sha256=sha(B/'protocol.json'),input_sha256=sha(out/'inputs.npz'),
        manifest_sha256=sha(out/'window_manifest.json'),groups={},scope='Synthetic verifier test only')
    traces=[dict(sampling_seed=s,batch_index=i,initial_noise_sha256='fixture',sampling_rng_after_sha256='fixture',teacher_noise_sha256='fixture')
        for s in (1001,1002) for i in range(16)]
    for name in ('native212158','project215999'):
        np.savez(out/(name+'_predictions.npz'),predictions=pred,epsilon_mse=np.ones((2,3,64)))
        r['groups'][name]=dict(state_unchanged=True,all_frozen=True,checkpoint_sha256=protocol['checkpoints'][name]['sha256'],
            predictions_sha256=sha(out/(name+'_predictions.npz')),action_metrics=m,
            epsilon_mse_all16=summarize_epsilon(np.ones((2,3,64)),ep,roles),common_noise_traces=copy.deepcopy(traces))
    r['baselines']={name:summarize(p,target,valid,ep,roles) for name,p in
        (('no_motion',np.broadcast_to(positions[:,-1:,:],(64,16,2))[None]),('train_mean',np.broadcast_to(mean,(64,16,2))[None]))}
    r['clip']=dict(worker=dict(all_frozen=True,weights_unchanged=True),selected_reference_max_abs=0)
    save('report.json',r)
    ok('Full small-output verifier accepts internally consistent fixture',validate(out,B)['output_verified'])
    for name,mutate in [
        ('Incomplete output',lambda d:d.update(complete=False)),
        ('Out-of-scope training',lambda d:d.update(training_updates=1)),
        ('Final scene access',lambda d:d.update(final50_accessed=True)),
        ('Wrong checkpoint',lambda d:d['groups']['native212158'].update(checkpoint_sha256='bad')),
        ('Corrupt predictions hash',lambda d:d['groups']['native212158'].update(predictions_sha256='bad')),
        ('Changed reported score',lambda d:d['groups']['native212158']['action_metrics']['train']['full16'].update(episode_equal_mean_pixels=99)),
        ('Different model sampling stream',lambda d:d['groups']['project215999']['common_noise_traces'][0].update(initial_noise_sha256='different')),
        ('Changed CLIP weights',lambda d:d['clip']['worker'].update(weights_unchanged=False))]:
        bad=copy.deepcopy(r);mutate(bad);save('report.json',bad)
        rejects('Verifier rejects '+name,lambda:validate(out,B))
    save('report.json',r)
    path=out/'native212158_predictions.npz';path.write_bytes(path.read_bytes()+b'changed')
    rejects('Verifier rejects changed prediction bytes',lambda:validate(out,B))

for name in ('audit.sh','submit_once.sh'):
    subprocess.run(['C:/Program Files/Git/bin/bash.exe','-n',str(B/name)],check=True,capture_output=True)
ok('Both Bash entry points pass syntax check')
assert '--time=00:30:00' in (B/'audit.sh').read_text() and '--no-requeue' in (B/'audit.sh').read_text()
ok('30minute no-requeue cap')
tree=ast.parse((B/'run_audit.py').read_text())
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
ok('Runtime contains no backward/optimizer/step training call',not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','fit') for n in calls))
result=dict(checks_passed=len(checks),checks=checks,scope='Synthetic new-entry checks only; actual endpoints/CLIP/CUDA/environment never executed',
    actual_checkpoint_loaded=False,real_server_runtime_measured=False,training_updates=0,cluster_job_submitted=False,
    archive_sha256=sha(ROOT/'cluster_packages/pusht_endpoint_action_audit_v1.zip'))
(ROOT/'reports/pusht_endpoint_action_audit_checks_20261022.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
