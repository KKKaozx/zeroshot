"""Bind native target-only roles, batch/noise streams and original first batch."""
import copy
import hashlib
import json
import numpy as np
import torch
from protocols import HERE,read,sha


def tensor_bytes(value):
    value=value.detach().cpu().contiguous()
    return json.dumps([str(value.dtype),list(value.shape)]).encode()+value.reshape(-1).view(torch.uint8).numpy().tobytes()


class Hook:
    def __init__(self,target,loop_sha,method_sha):
        self.target=target;self.loop_sha=loop_sha;self.method_sha=method_sha
        self.reference=read(HERE/'reference_contract.json');self.first=read(HERE/'reference_first_batch.json')
        with np.load(HERE/'reference_prefix.npz') as z:self.schedule={k:z[k] for k in z.files}
        self.done=0;self.captures=0;self.pending=None;self.noise=None;self.reference_values=None
        self.hashers={k:hashlib.sha256() for k in ['noise','timesteps','noisy_actions']}
        self.initial_sha=None;self.first_replay=None

    def bind(self,model,api,contract,plan):
        from completed_policy_export import model_sha
        canonical=copy.deepcopy(contract)
        assert canonical['arguments']['diffusion_prediction_type']==self.target
        assert canonical['config']['model']['diffusion_prediction_type']==model.diffusion_prediction_type==self.target
        canonical['arguments']['diffusion_prediction_type']='epsilon'
        canonical['config']['model']['diffusion_prediction_type']='epsilon'
        canonical['offline_training_protocol']=contract['offline_training_protocol']['source_reference_admission']
        assert canonical==self.reference,'Other main-entry/config/data/runtime changes detected'
        assert plan.total_updates==65536 and plan.batch_size==8
        self.initial_sha=model_sha(api.trainable_state_dict(model))
        assert self.initial_sha==read(HERE/'pair_protocol.json')['first_initial_learned_sha256']
        contract['diagnostic_target_pair']={'target':self.target,'pair_protocol_sha256':sha(HERE/'pair_protocol.json'),
            'loop_source_sha256':self.loop_sha,'diffusion_loss_source_sha256':self.method_sha,
            'hook_source_sha256':sha(__file__),'protocol_loader_sha256':sha(HERE/'protocols.py'),
            'initial_learned_sha256':self.initial_sha,'short_updates':8192,'lr_horizon':65536,
            'not_main_thesis_condition':True,'no_source_supervision_filter':True}

    def begin(self,indices,completed):
        assert self.pending is None and self.done==completed
        n=int(self.schedule['lengths'][completed])
        assert indices==self.schedule['batches'][completed,:n].tolist(),('Original batch prefix differs',completed)
        self.pending=completed+1

    def capture_noise(self,model,noise,timesteps,noisy_actions):
        assert self.pending==self.done+1==self.captures+1
        assert model.diffusion_prediction_type==self.target
        for key,value in [('noise',noise),('timesteps',timesteps),('noisy_actions',noisy_actions)]:
            self.hashers[key].update(tensor_bytes(value))
        self.noise=noise;self.captures+=1

    def observe(self,model,api,output,actions,current,masks,criterion):
        assert self.noise is not None
        desired=actions[...,:7] if self.target=='sample' else self.noise
        assert torch.equal(output[1],desired),'Native training target differs from declared role'
        assert masks.sum(dim=(0,1)).cpu().tolist()==self.schedule['valid_dimensions'][self.done].tolist()
        if self.done==0:
            legacy=(output[0],self.noise,output[2])
            self.reference_values=api.policy_loss(model,legacy,actions,criterion,current_grippers=current,supervision_masks=masks)
        self.noise=None

    def audit_first(self,model,optimizer,named):
        if self.done:return
        assert self.reference_values is not None and not optimizer.state
        parameters=[p for _,p in named];versions=[p._version for p in parameters]
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if parameters[0].is_cuda else []
        loss,pose,grip=self.reference_values;loss.backward(retain_graph=True)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        actual={'loss':float(loss.detach()),'pose_loss':float(pose.detach()),'gripper_loss':float(grip.detach()),'gradient_norm':float(norm)}
        expected={key:self.first[key] for key in actual}
        assert actual==expected,('Original epsilon first-batch reference differs',actual,expected)
        assert versions==[p._version for p in parameters] and torch.equal(cpu,torch.get_rng_state())
        if cuda:assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        optimizer.zero_grad(set_to_none=True);self.reference_values=None
        self.first_replay={'passed':True,'actual':actual,'expected':expected,'rng_and_parameter_versions_unchanged':True,'optimizer_steps':0}
        print('ORIGINAL_MAIN_FIRST_BATCH_REPLAY_EXACT',self.target,json.dumps(actual),flush=True)

    def commit(self,completed):
        assert completed==self.pending==self.done+1==self.captures
        self.done=completed;self.pending=None

    def state(self):
        assert self.pending is None and self.noise is None and self.reference_values is None
        return {'target':self.target,'completed_updates':self.done,'captured_batches':self.captures,
            'stream_sha256':{key:value.hexdigest() for key,value in self.hashers.items()},
            'initial_learned_sha256':self.initial_sha,'first_reference_replay':self.first_replay}
