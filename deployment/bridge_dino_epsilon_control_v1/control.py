"""Constant identity, saved positive replay and first-update reference guards."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from checks import read,sha

HERE=Path(__file__).resolve().parent


def initial_digest(initial,tensor_sha):
    return hashlib.sha256(''.join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()


class Control:
    def __init__(self,root):
        self.root=Path(root);self.protocol=read(HERE/'protocol.json')
        self.baseline=self.root/'runs/bridge-backbone-pair-a40-229116/dino'
        self.first=read(HERE/'reference_first_update.json');self.report=None
        self.reference_passed=False;self.baseline_replayed=False
        self.effective_source_sha256=None

    def choose_constants(self,model,initial,tensor_sha,report):
        self.report=report;p=self.protocol;expected=p['baseline_initial_shared_sha256']
        report['constant_identity_admission']={'selected':'actual_constructor','original_initializer_rng_kept':True,
            'full_original_initial_digest_matched':True}
        if initial_digest(initial,tensor_sha)==expected:return initial
        names=p['mutable_constant_names'];before=torch.get_rng_state().clone()
        canonical=read(HERE/'canonical_constants.json')['constants']
        variant=read(self.root/p['known_variant_profile_relative'])
        candidates=[('canonical_archived_profile',{k:v['values'] for k,v in canonical.items()},
                     {k:v['metadata'] for k,v in canonical.items()}),
                    ('known_saved_runtime_variant',variant['schedule_values'],variant['tensors'])]
        for label,values,metadata in candidates:
            # Four other schedule buffers must match already. The repair may
            # never hide encoder, learned parameter or another buffer changes.
            for key in ['betas','alphas','alpha_bars','posterior_variance']:
                assert tensor_sha(initial[key])==metadata[key]['sha256'],key
            trial=dict(initial)
            for key in names:
                value=torch.tensor(values[key],dtype=torch.float32)
                assert list(value.shape)==metadata[key]['shape'] and tensor_sha(value)==metadata[key]['sha256']
                assert key in model._buffers and key not in dict(model.named_parameters())
                trial[key]=value
            if initial_digest(trial,tensor_sha)!=expected:continue
            with torch.no_grad():
                for key in names:model.get_buffer(key).copy_(trial[key].to(model.get_buffer(key).device))
            assert torch.equal(before,torch.get_rng_state())
            report['constant_identity_admission'].update(selected=label,buffer_names=names,
                full_original_initial_digest_matched=True,encoder_or_learned_parameters_written=False)
            return trial
        raise ValueError('Initial state does not match positive229116 under declared exact buffer variants; no training')

    def replay_and_activate(self,model,context,current,targets,observer,evaluate,trainable_copy,state_sha,cfg,report):
        assert model.diffusion_prediction_type=='sample'
        saved=trainable_copy(model);initial=state_sha(saved)
        assert initial==self.protocol['baseline_augmented_initial_sha256']
        payload=torch.load(self.baseline/'final_trainable.pt',map_location='cpu',weights_only=False,mmap=True)
        assert payload['update']==8192 and payload['config']['model']['diffusion_prediction_type']=='sample'
        assert set(payload['trainable_state_dict'])==set(saved)
        source=np.load(self.baseline/'reduced_u8192_train.npz')
        ids=self.protocol['baseline_replay_windows'];assert ids==[0,1,2,3]
        assert np.array_equal(targets[ids].cpu().numpy(),source['targets'][ids])
        try:
            missing=model.load_state_dict(payload['trainable_state_dict'],strict=False)
            assert not missing.unexpected_keys and not set(saved).intersection(missing.missing_keys)
            with observer(model):
                ctx=torch.cat([context(torch.tensor(ids[i:i+2],device=current.device)) for i in range(0,4,2)])
                assert np.array_equal(ctx.cpu().numpy(),source['contexts'][ids]),'Saved positive context differs'
                pred,_=evaluate(model,ctx,current[ids],source['targets'][ids],self.protocol['sampling_seeds'],batch=2)
                assert np.array_equal(pred,source['predictions'][:,ids]),'Saved positive predictions differ'
        finally:
            model.load_state_dict(saved,strict=False)
        assert state_sha(trainable_copy(model))==initial
        self.baseline_replayed=True
        report['positive_saved_replay']={'passed':True,'windows':ids,'sampling_seeds':self.protocol['sampling_seeds'],
            'context_max_abs':0.,'prediction_max_abs':0.,'fresh_initial_weights_restored':True,'optimizer_updates':0}
        model.diffusion_prediction_type='epsilon';cfg['model']['diffusion_prediction_type']='epsilon'
        report['target_intervention']=self.binding()
        print('POSITIVE_X0_REPLAY_EXACT; fresh initialization restored; epsilon active',flush=True)

    def audit_first(self,update,model,optimizer,guess,clean,gradient_norms,trainables):
        if update!=1:return
        assert self.baseline_replayed and not self.reference_passed and not optimizer.state
        cpu=torch.get_rng_state().clone()
        cuda=torch.cuda.get_rng_state_all() if guess.is_cuda else []
        original_loss=torch.nn.functional.mse_loss(guess,clean[...,:7])
        original_loss.backward(retain_graph=True)
        modules=gradient_norms(model)
        norm=torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
        actual={'loss':float(original_loss.detach()),'global_unclipped_gradient_norm':float(norm),
            'unclipped_module_gradient_norms':modules}
        expected={k:self.first[k] for k in actual}
        assert actual==expected,('Original first x0 batch replay mismatch; no epsilon step',actual,expected)
        assert torch.equal(cpu,torch.get_rng_state())
        if cuda:assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
        optimizer.zero_grad(set_to_none=True)
        self.reference_passed=True
        self.report['original_first_x0_batch_replay']={'passed':True,'actual':actual,'expected':expected,
            'optimizer_steps_before_replay':0,'rng_unchanged':True}
        print('ORIGINAL_X0_FIRST_BATCH_REPLAY_EXACT; before first epsilon update',flush=True)

    def binding(self):
        return {'experiment':'positive_dino_prediction_target_control','candidate_prediction_type':'epsilon',
            'baseline_job':229116,'baseline_target':'sample','contract_sha256':sha(HERE/'protocol.json'),
            'control_source_sha256':sha(__file__),'entry_source_sha256':sha(HERE/'run.py'),
            'effective_source_sha256':self.effective_source_sha256,'new_x0_training':False,'updates':8192,
            'positive_replay_passed':self.baseline_replayed,'first_x0_reference_passed':self.reference_passed,
            'main_oxe_model_replaced':False}
