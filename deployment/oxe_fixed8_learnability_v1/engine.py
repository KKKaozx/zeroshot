"""Repeated fixed-batch learning and complete DDPM generation, no model search."""
import json
import time
import numpy as np
import torch
from qualification import HERE,read,sha


def context(bundle,cache,native=False):
    model=bundle['model'];port=bundle['port'];t=cache['train_tokens']
    if native:core=model.get_context_vector(cache['images'],t['input_ids'],t['attention_mask'])
    else:core=model.adapter(cache['train_visual'],cache['train_language'],t['attention_mask'])
    return core+port(cache['pose'])


def backward(bundle,cache,native=False):
    api=bundle['api'];model=bundle['model'];port=bundle['port'];optimizer=bundle['optimizer']
    model.train();port.train();optimizer.zero_grad(set_to_none=True)
    output=model.diffusion_loss(cache['actions'],context(bundle,cache,native),cache['current'])
    assert torch.equal(output[1],cache['actions'][...,:7])
    loss,pose,grip=api.policy_loss(model,output,cache['actions'],torch.nn.MSELoss(),
        current_grippers=cache['current'],supervision_masks=cache['masks'])
    assert torch.isfinite(loss);loss.backward()
    return {'loss':float(loss.detach()),'pose_loss':float(pose.detach()),'gripper_loss':float(grip.detach())}


def cache_guard(bundle,cache):
    from pose_runtime import digest
    from step_training_state import restore_rng,capture_rng
    model=bundle['model'];port=bundle['port'];optimizer=bundle['optimizer'];noise=bundle['noise']
    api=bundle['api'];rng=bundle['source']['rng'];records=[]
    before={'base':digest(api.trainable_state_dict(model)),'port':digest(port.state_dict()),'adam':digest(optimizer.state_dict())}
    for native in [True,False]:
        restore_rng(rng);noise.current=10241
        row=backward(bundle,cache,native)
        gradient=digest({'base':{n:p.grad for n,p in model.named_parameters() if p.requires_grad},'port':port.weight.grad})
        assert port.weight.grad is not None and torch.isfinite(port.weight.grad).all()
        row.update(gradient_sha256=gradient,noise=noise.events[-1],rng_after=digest(capture_rng(include_cuda=rng['include_cuda'])))
        records.append(row)
    assert records[0]==records[1],'Cached frozen features changed actual native training gradients/noise'
    assert before=={'base':digest(api.trainable_state_dict(model)),'port':digest(port.state_dict()),'adam':digest(optimizer.state_dict())}
    optimizer.zero_grad(set_to_none=True);noise.events=[];restore_rng(rng)
    return {'passed':True,'native_and_cached_loss_all_gradients_noise_rng_exact':True,
        'optimizer_updates':0,'identities_unchanged':before,'first_cached_row':records[1]}


def sample(bundle,cache,seeds):
    model=bundle['model'];port=bundle['port'];model.eval();port.eval();draws=[]
    with torch.inference_mode():
        fused=model.adapter(cache['eval_visual'],cache['eval_language'],cache['eval_attention'])+port(cache['pose'])
        for seed in seeds:
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            draws.append(model.sample(fused,cache['current']).cpu().numpy())
    return np.stack(draws)


def metrics(outputs,targets,masks):
    from metrics import metrics as original_metrics
    return {name:original_metrics(p,targets,masks,list(range(8))) for name,p in outputs.items()}


def decision(calculated,protocol):
    keys=['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']
    end=calculated['after'];static=calculated['static'];limits={}
    for key in keys:
        floor=protocol['mean_position_floor'] if 'position' in key else protocol['mean_rotation_floor_deg']
        limits[key]=max(floor,protocol['mean_static_fraction']*static[key])
    good=[]
    for e,s in zip(end['per_window'],static['per_window']):
        good.append(all(e[k]<=max(protocol['window_position_floor'] if 'position' in k else protocol['window_rotation_floor_deg'],
            protocol['window_static_fraction']*s[k]) for k in keys))
    passed=all(end[k]<=limits[k] for k in keys) and sum(good)>=protocol['required_windows']
    return {'mean_limits':limits,'means_pass':all(end[k]<=limits[k] for k in keys),
        'windows_passing_all_four':sum(good),'window_pass_mask':good,'fixed8_pose_fit_gate_passed':passed,
        'interpretation':'Basic generation fit demonstrated on these8repeated training windows only' if passed else
            'This fixed512update budget did not establish repeated8window generation fit; not proof of mathematical unlearnability',
        'new_budget_or_variants_authorized':False,'libero_or_generalization_verdict':None}


def run(bundle,cache,output,protocol,bound):
    from pose_runtime import digest
    from completed_policy_export import model_sha
    from step_training_runner import frozen_digest
    from step_training_state import restore_rng
    root=bundle['root'];source=bundle['source'];api=bundle['api'];model=bundle['model'];port=bundle['port'];optimizer=bundle['optimizer']
    targets=cache['actions'].cpu().numpy();masks=cache['masks'].cpu().numpy();noise=bundle['noise'];seeds=protocol['sampling_seeds']
    outputs={};report={'execution_complete':False,'protocol_sha256':sha(HERE/'protocol.json'),'source_binding':bound,
        'additional_optimizer_calls':0,'sampling_seeds':seeds,'source_update':10240,'endpoint_update':10752,
        'scope':'Repeated fixed8training-window pose generation diagnostic; not main training or new grasp/generalization evidence',
        'automatic_following_action':False};started=time.monotonic()
    try:
        outputs['before']=sample(bundle,cache,seeds)
        with np.load(root/'runs/oxe-pose-pair-232587/evaluation/predictions_and_targets.npz') as z:
            assert np.array_equal(targets,z['targets'][:8]) and np.array_equal(masks,z['masks'][:8])
            assert np.array_equal(outputs['before'],z['real_pose'][:,:8]),'Actual232587 first8predictions must replay exactly'
            outputs['static']=z['static'][:,:8].copy()
        report['source_prediction_replay_exact']=True
        report['cached_training_guard']=cache_guard(bundle,cache)
        restore_rng(source['rng']);history=[]
        frozen=frozen_digest(model,bundle['keys']);assert frozen==protocol['source_identity']['frozen_state_sha256']
        for step in range(1,protocol['updates']+1):
            assert time.monotonic()-started<protocol['runtime_cap_seconds'],'Bounded diagnostic runtime cap; no automatic retry'
            rates=bundle['schedule'].rates(10240+step-1)
            assert [g['lr'] for g in optimizer.param_groups]==rates
            noise.current=10240+step;row=backward(bundle,cache)
            if step==1:
                expected=report['cached_training_guard']['first_cached_row']
                assert all(row[k]==expected[k] for k in ['loss','pose_loss','gripper_loss'])
                assert noise.events[-1]==expected['noise']
            parameters=[p for p in model.parameters() if p.requires_grad]+list(port.parameters())
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            assert all(p.grad is None and not p.requires_grad for e in [model.vision_encoder,model.text_encoder] for p in e.parameters())
            optimizer.step();bundle['schedule'].apply(optimizer,10240+step)
            row.update(diagnostic_update=step,source_schedule_update=10240+step,indices=cache['indices'],
                lr_used=rates,gradient_norm=float(norm));history.append(row);report['additional_optimizer_calls']=step
            if step<=2 or step%64==0:print('OXE_FIXED8_UPDATE',json.dumps(row),flush=True)
        assert len(noise.events)==len(history)==protocol['updates']
        outputs['after']=sample(bundle,cache,seeds)
        assert frozen_digest(model,bundle['keys'])==frozen
        report['metrics']=metrics(outputs,targets,masks);report['fit_decision']=decision(report['metrics'],protocol['fit_gate'])
        report['history']=history;report['noise_stream']=noise.events
        # One diagnostic policy only; retain the original full source state without copying its Adam file.
        policy={'format':'current-oxe-fixed8-diagnostic-policy-v1','protocol_sha256':sha(HERE/'protocol.json'),
            'source_binding':bound,'source_contract_sha256':source['contract_sha256'],'source_update':10240,
            'diagnostic_updates':protocol['updates'],'base_config_unchanged':bundle['config'],
            'base':api.trainable_state_dict(model),'pose_port':{k:v.detach().cpu() for k,v in port.state_dict().items()},
            'warning':'Separate repeated-window diagnostic, not a completed main65536training policy or resumable source plan'}
        report['final_base_sha256']=model_sha(policy['base']);report['final_port_sha256']=digest(policy['pose_port'])
        temp=output/'diagnostic_policy.pt.tmp';torch.save(policy,temp);temp.replace(output/'diagnostic_policy.pt')
        report['diagnostic_policy_bytes']=(output/'diagnostic_policy.pt').stat().st_size
        path=bundle['source_path'];assert [path.stat().st_size,path.stat().st_mtime_ns]==bundle['source_file_size_mtime']
        report.update(execution_complete=True,source_file_size_mtime_unchanged=True,
            frozen_sha256=frozen,elapsed_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            training_token_shape=cache['training_token_shape'],evaluation_token_shape=cache['evaluation_token_shape'])
    finally:
        np.savez_compressed(output/'predictions_and_targets.npz',targets=targets,masks=masks,**outputs)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('OXE_FIXED8_REPORT',output/'report.json',flush=True)
    print('OXE_FIXED8_GENERATION',json.dumps({k:{n:v for n,v in m.items() if n!='per_window'} for k,m in report['metrics'].items()}),flush=True)
    print('OXE_FIXED8_FIT_DECISION',json.dumps(report['fit_decision']),flush=True)
    return report
