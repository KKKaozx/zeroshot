"""Paired source-policy intervention; dense fixed77training padding controls model RNG shapes."""
import json
import os
import time
import numpy as np
import torch
from pair_qualification import HERE,read,sha


def rng_digest(bundle):
    from pose_runtime import digest
    from step_training_state import capture_rng
    return digest(capture_rng(include_cuda=bundle['source']['rng']['include_cuda']))


def batch(bundle,indices):
    from pose_features import batch_features
    api=bundle['api'];texts,images,current,actions,masks=api.collate_batch([bundle['data'][int(i)] for i in indices])
    tokens=bundle['tokenizer'](texts,padding='max_length',truncation=True,max_length=77,return_tensors='pt')
    assert tokens['input_ids'].shape==tokens['attention_mask'].shape==(len(indices),77)
    assert images.shape[0]==len(indices) and actions.shape==masks.shape==(len(indices),16,8)
    state=batch_features(bundle['data'].training.store,indices,bundle['stats'],'real_pose')
    return {'images':images.cuda(),'tokens':{k:v.cuda() for k,v in tokens.items()},
        'current':current.cuda(),'actions':actions.cuda(),'masks':masks.cuda(),'pose':torch.from_numpy(state).cuda()}


def backward(bundle,inputs):
    model=bundle['model'];port=bundle['port'];api=bundle['api'];model.train();port.train()
    bundle['optimizer'].zero_grad(set_to_none=True);tokens=inputs['tokens']
    context=model.get_context_vector(inputs['images'],tokens['input_ids'],tokens['attention_mask'])+port(inputs['pose'])
    output=model.diffusion_loss(inputs['actions'],context,inputs['current'])
    assert torch.equal(output[1],inputs['actions'][...,:7])
    loss,pose,grip=api.policy_loss(model,output,inputs['actions'],torch.nn.MSELoss(),
        current_grippers=inputs['current'],supervision_masks=inputs['masks'])
    assert torch.isfinite(loss);loss.backward()
    parameters=[p for p in model.parameters() if p.requires_grad]+list(port.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
    return {'loss':float(loss.detach()),'pose_loss':float(pose.detach()),'gripper_loss':float(grip.detach())}


def initial_identity(bundle):
    from pose_runtime import digest
    from completed_policy_export import model_sha
    return {'base_learned_sha256':model_sha(bundle['api'].trainable_state_dict(bundle['model'])),
        'port_sha256':digest(bundle['port'].state_dict()),'adam_sha256':digest(bundle['optimizer'].state_dict()),
        'source_rng_sha256':digest(bundle['source']['rng'])}


def preflight_pair_rng(bundle,plans):
    from step_training_state import restore_rng
    before=initial_identity(bundle);records={}
    for role in ['replacement','cycle']:
        restore_rng(bundle['source']['rng']);inputs=batch(bundle,plans[role][0].tolist())
        bundle['noise'].current=10241;start=rng_digest(bundle);row=backward(bundle,inputs)
        records[role]={'loss':row,'token_shape':list(inputs['tokens']['input_ids'].shape),
            'rng_before':start,'rng_after':rng_digest(bundle),'noise':bundle['noise'].events[-1]}
    a,b=records['replacement'],records['cycle']
    assert a['rng_before']==b['rng_before'] and a['rng_after']==b['rng_after'],'Different first-batch stochastic shapes'
    assert all(a['noise'][k]==b['noise'][k] for k in ['noise','timesteps'])
    assert initial_identity(bundle)==before,'Zero-update probe changed parameters or Adam'
    bundle['optimizer'].zero_grad(set_to_none=True);bundle['noise'].events=[];restore_rng(bundle['source']['rng'])
    return {'passed':True,'optimizer_updates':0,'initial_identity':before,'first_batch':records,
        'both_training_token_shapes':[8,77],'primitive_noise_and_timesteps_matched':True,
        'noisy_actions_or_losses_expected_equal':False}


def check_paired_stochastic(row,event,reference_row,reference_event):
    assert row['rng_before']==reference_row['rng_before'] and row['rng_after']==reference_row['rng_after'],('Paired model RNG differs',row['local_update'])
    assert all(event[k]==reference_event[k] for k in ['step','noise','timesteps']),('Paired actual noise/t differs',row['local_update'])
    # Noisy inputs deliberately contain different expert targets. Do not assert their hashes equal.


def train(bundle,role,plans,output,contract,control=None,resume=False):
    from step_training_state import restore_rng
    from step_training_runner import frozen_digest
    from pose_runtime import digest
    from completed_policy_export import model_sha,contract_sha
    import pair_state
    identity=initial_identity(bundle);history=[];noise=bundle['noise'];noise.events=[];done=0
    bundle['training_job_ids']=[os.environ.get('SLURM_JOB_ID','cpu-fixture')]
    restore_rng(bundle['source']['rng']);checkpoint=output/'latest_state.pt'
    if resume and checkpoint.exists():done,history,noise.events=pair_state.restore(checkpoint,bundle,role,contract,plans,control)
    start=done;tick=time.monotonic();interval=contract['checkpoint_every']
    for i in range(done,contract['updates_each']):
        assert time.monotonic()-tick<contract['arm_training_cap_seconds'],'Bounded arm cap; retained committed checkpoint, no automatic extension'
        indices=plans[role][i].tolist();inputs=batch(bundle,indices)
        assert inputs['masks'].sum((0,1)).cpu().tolist()==plans[role+'_valid_dimensions'][i].tolist()
        rates=[g['lr'] for g in bundle['optimizer'].param_groups]
        assert rates==plans['lrs'][i].tolist()==bundle['schedule'].rates(10240+i)
        noise.current=10241+i;before=rng_digest(bundle)
        if control is not None:assert before==control['history'][i]['rng_before'],('Paired RNG before forward differs',i+1)
        measured=time.monotonic();row=backward(bundle,inputs)
        parameters=[p for p in bundle['model'].parameters() if p.requires_grad]+list(bundle['port'].parameters())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        assert all(not p.requires_grad and p.grad is None for m in [bundle['model'].vision_encoder,bundle['model'].text_encoder] for p in m.parameters())
        bundle['optimizer'].step();bundle['schedule'].apply(bundle['optimizer'],10241+i);after=rng_digest(bundle)
        row.update(local_update=i+1,source_schedule_update=10241+i,indices=indices,lr_used=rates,
            valid_dimensions=plans[role+'_valid_dimensions'][i].tolist(),gradient_norm=float(norm),
            token_shape=list(inputs['tokens']['input_ids'].shape),rng_before=before,rng_after=after,
            elapsed_step_seconds=time.monotonic()-measured)
        if control is not None:check_paired_stochastic(row,noise.events[-1],control['history'][i],control['noise'][i])
        history.append(row)
        if (i+1)%interval==0 or i+1==contract['updates_each']:
            assert frozen_digest(bundle['model'],bundle['keys'])==contract['frozen_sha256']
            pair_state.save(checkpoint,bundle,role,i+1,history,noise.events,contract)
        if i<2 or (i+1)%256==0 or i+1==contract['updates_each']:
            print('SAMPLING_PAIR_UPDATE',role,json.dumps(row),flush=True)
    assert len(history)==len(noise.events)==contract['updates_each']
    receipt={'training_passed':True,'role':role,'updates_each':contract['updates_each'],'completed_source_schedule':10240+contract['updates_each'],
        'initial_identity':identity,'base_learned_sha256':model_sha(bundle['api'].trainable_state_dict(bundle['model'])),
        'port_sha256':digest(bundle['port'].state_dict()),'contract_sha256':contract_sha(contract),
        'frozen_sha256':frozen_digest(bundle['model'],bundle['keys']),'new_updates_this_segment':contract['updates_each']-start,
        'training_elapsed_seconds':time.monotonic()-tick,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'training_job_ids':bundle['training_job_ids']}
    assert receipt['frozen_sha256']==contract['frozen_sha256']
    (output/'training_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt,{'history':history,'noise':noise.events}


def sample(bundle,cache,seeds):
    model=bundle['model'];port=bundle['port'];model.eval();port.eval();contexts=[];draws=[]
    with torch.inference_mode():
        for lo in range(0,120,8):
            contexts.append(model.adapter(cache['visual'][lo:lo+8],cache['language'][lo:lo+8],cache['attention'][lo:lo+8])+
                port(torch.from_numpy(cache['features']['real_pose'][lo:lo+8]).cuda()))
        contexts=torch.cat(contexts)
        for seed in seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            draws.append(np.concatenate([model.sample(contexts[lo:lo+8],cache['current'][lo:lo+8]).cpu().numpy() for lo in range(0,120,8)]))
            print('SAMPLING_PAIR_SAMPLE',seed,'120/120',flush=True)
    return np.stack(draws)
