"""Actual first control batch replays without an optimizer call, then probe new pool."""
import torch

def run(bundle,plans,control):
    from step_training_state import restore_rng
    from pair_steps import batch,backward,rng_digest,initial_identity,check_paired_stochastic
    before=initial_identity(bundle);records={}
    for role in ['replacement','moderate_pool']:
        restore_rng(bundle['source']['rng']);inputs=batch(bundle,plans[role][0].tolist())
        noise=bundle['noise'];noise.events=[];noise.current=10241
        row={'local_update':1,'rng_before':rng_digest(bundle)};row.update(backward(bundle,inputs))
        parameters=[p for p in bundle['model'].parameters() if p.requires_grad]+list(bundle['port'].parameters())
        row['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        row['rng_after']=rng_digest(bundle);event=noise.events[-1]
        reference=control['history'][0];reference_noise=control['noise'][0]
        check_paired_stochastic(row,event,reference,reference_noise)
        if role=='replacement':
            assert all(row[k]==reference[k] for k in ['loss','pose_loss','gripper_loss','gradient_norm']),'Original control first loss/gradient changed'
            assert event==reference_noise,'Original control noisy target differs'
        records[role]={'measured':row,'noise':event,'token_shape':list(inputs['tokens']['input_ids'].shape)}
    assert initial_identity(bundle)==before,'Probe changed learned state or Adam'
    bundle['optimizer'].zero_grad(set_to_none=True);bundle['noise'].events=[];restore_rng(bundle['source']['rng'])
    return {'passed':True,'optimizer_updates':0,'initial_identity':before,'first_batch':records,
        'old_control_loss_gradient_noise_rng_replay_exact':True,'new_pool_noise_t_rng_matched':True,
        'different_targets_noisy_actions_not_required_equal':True}
