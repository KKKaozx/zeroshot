"""Remove non-Bridge supervised numerators; retain original denominators/steps."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bridge_loss(model,output,actions,masks,keep):
    """Only for the exact declared epsilon/state/unbalanced-BCE main recipe."""
    assert model.diffusion_prediction_type=='epsilon' and model.gripper_target_mode=='state'
    assert not model.balanced_gripper_loss and model.gripper_change_weight==1
    assert len(output)==3 and float(model.gripper_loss_weight)==.25
    predicted,noise,logits=output
    keep=torch.as_tensor(keep,device=predicted.device,dtype=predicted.dtype)
    assert keep.shape==(len(actions),) and bool(((keep==0)|(keep==1)).all())
    pose_mask=masks[...,:7].to(predicted.dtype)
    # The denominator is deliberately NOT multiplied by keep.
    pose=((predicted-noise).square()*pose_mask*keep[:,None,None]).sum()/pose_mask.sum().clamp_min(1)
    valid=masks[...,7]>.5
    gripper=torch.zeros((),device=predicted.device)
    if bool(valid.any()):
        bce=nn.functional.binary_cross_entropy_with_logits(logits,(actions[...,7]>0).float(),reduction='none')
        selected=keep[:,None].expand_as(logits)
        # Preserve a zero-valued gradient graph when only non-Bridge valid
        # grippers occur, rather than turning original AdamW slots into None.
        gripper=(bce[valid]*selected[valid]).sum()/valid.sum().clamp_min(1)
    return pose+model.gripper_loss_weight*gripper,pose,gripper


class Supervisor:
    def __init__(self,bundle,mode,effective_sha,preflight=None):
        self.bundle=Path(bundle);self.mode=mode;self.effective_sha=effective_sha
        self.protocol=read(self.bundle/'protocol.json')
        self.reference=read(self.bundle/'reference_contract.json')
        self.first=read(self.bundle/'reference_first_batch.json')
        # Materialize once: indexing an NpzFile would decompress the complete
        # archived schedule on every update, adding avoidable CPU/I/O work.
        with np.load(self.bundle/'reference_schedule.npz') as archive:
            self.schedule={key:archive[key] for key in archive.files}
        self.keep=np.load(self.bundle/'bridge_windows.npy')
        self.preflight=Path(preflight) if preflight else None
        self.done=self.bridge_draws=self.all_draws=self.no_bridge=0
        self.pending=None;self.last_committed_keep=0;self.reference_values=None
        self.first_replay=None;self.initial_sha=None
        self.engineering_end=2

    def limit_end(self,end):return self.engineering_end if self.mode=='preflight' else end

    def bind(self,model,api,contract,plan,dataset):
        from completed_policy_export import model_sha
        old=self.reference
        assert set(contract)==set(old),'Original training-contract fields changed'
        for key in old:
            assert contract[key]==old[key],('Reference main contract differs',key)
        assert plan.total_updates==65536 and plan.batch_size==8
        assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()),'No hidden batch-stat coupling permitted'
        assert dataset.train_count==len(self.keep)
        actual=np.asarray([row['source']=='tfrecord_bridge_state_action'
            for row in dataset.samples[:dataset.train_count]],dtype=np.uint8)
        assert np.array_equal(actual,self.keep),'Bridge source identity differs from actual dataset'
        initial=model_sha(api.trainable_state_dict(model));self.initial_sha=initial
        pre_sha=None
        if self.mode!='preflight':
            pre=read(self.preflight/'diagnostic_receipt.json')
            assert pre['passed'] and pre['mode']=='preflight' and pre['completed_updates']==2
            assert pre['first_reference_replay']['passed']
            assert pre['initial_learned_sha256']==initial,'Fresh initial weights differ from engineering preflight'
            assert pre['protocol_sha256']==sha(self.bundle/'protocol.json')
            pre_sha=sha(self.preflight/'diagnostic_receipt.json')
        binding=dict(experiment='bridge_supervision_counterfactual',diagnostic_only=True,
            protocol_sha256=sha(self.bundle/'protocol.json'),loss_module_sha256=sha(__file__),
            source_loop_sha256=sha(self.bundle/'code/step_training_runner.py'),
            effective_loop_sha256=self.effective_sha,
            schedule_sha256=sha(self.bundle/'reference_schedule.npz'),
            initial_learned_sha256=initial,engineering_preflight_receipt_sha256=pre_sha,
            original_initial_digest_was_not_archived=True,
            old_initialization_admission='Exact reference code/runtime/seed/config plus original first-batch full objective and gradient norm replay before any update')
        contract['supervision_counterfactual']=binding
        # The model was already constructed. This metadata is not an input or
        # architecture change; it travels with the diagnostic exported policy.
        contract['config']['diagnostic_supervision']=binding

    def begin(self,indices,completed):
        assert self.pending is None and self.done==completed
        length=int(self.schedule['lengths'][completed])
        assert indices==self.schedule['batches'][completed,:length].tolist(),('Batch schedule differs',completed)
        flags=self.keep[np.asarray(indices,dtype=np.int64)]
        self.pending=dict(update=completed+1,indices=list(indices),flags=flags,
            keep_count=int(flags.sum()),draws=len(indices))

    def loss(self,model,output,actions,criterion,current_grippers=None,supervision_masks=None):
        assert self.pending is not None and supervision_masks is not None
        if self.done==0:
            from train import policy_loss
            self.reference_values=policy_loss(model,output,actions,criterion,
                current_grippers=current_grippers,supervision_masks=supervision_masks)
        return bridge_loss(model,output,actions,supervision_masks,self.pending['flags'])

    def audit_initial(self,model,optimizer):
        if self.done!=0:return
        assert self.pending['indices']==self.first['indices']
        assert self.reference_values is not None
        loss,pose,grip=self.reference_values
        parameters=[p for p in model.parameters() if p.requires_grad]
        versions=[p._version for p in parameters]
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        loss.backward(retain_graph=True)
        norm=nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        actual=dict(loss=float(loss.detach()),pose_loss=float(pose.detach()),
            gripper_loss=float(grip.detach()),gradient_norm=float(norm))
        expected={k:self.first[k] for k in actual}
        # Stop before ANY optimizer step if the archived initial trajectory
        # cannot be reproduced. Never widen this on an unexplained failure.
        assert actual==expected,('Original first-batch numerical replay differs',actual,expected)
        assert versions==[p._version for p in parameters]
        assert torch.equal(cpu,torch.get_rng_state())
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        optimizer.zero_grad(set_to_none=True)
        self.reference_values=None
        self.first_replay=dict(passed=True,actual=actual,expected=expected,
            optimizer_steps_before_replay=0,parameter_versions_unchanged=True,rng_unchanged=True)
        print('ORIGINAL_FIRST_BATCH_REPLAY_EXACT',json.dumps(actual),flush=True)

    def commit(self,completed):
        assert self.pending is not None and completed==self.done+1==self.pending['update']
        self.done=completed;self.bridge_draws+=self.pending['keep_count'];self.all_draws+=self.pending['draws']
        self.no_bridge+=int(self.pending['keep_count']==0)
        self.last_committed_keep=self.pending['keep_count'];self.pending=None

    def state_dict(self):
        assert self.pending is None and self.reference_values is None
        return dict(done=self.done,bridge_draws=self.bridge_draws,all_draws=self.all_draws,
            no_bridge=self.no_bridge,last_committed_keep=self.last_committed_keep,
            first_reference_replay=self.first_replay,initial_learned_sha256=self.initial_sha)

    def restore_state(self,state,completed):
        assert state['done']==completed and state['first_reference_replay']['passed']
        assert state['initial_learned_sha256']==self.initial_sha
        # Recompute counters from the immutable prefix, not a trusting counter.
        lengths=self.schedule['lengths'][:completed]
        batches=self.schedule['batches'][:completed]
        ids=batches[batches>=0]
        assert len(ids)==int(lengths.sum())==state['all_draws']
        assert int(self.keep[ids].sum())==state['bridge_draws']
        counts=np.asarray([self.keep[row[:int(n)]].sum() for row,n in zip(batches,lengths)])
        assert int((counts==0).sum())==state['no_bridge']
        for key in ('done','bridge_draws','all_draws','no_bridge','last_committed_keep'):
            setattr(self,key,state[key])
        self.first_replay=state['first_reference_replay']
