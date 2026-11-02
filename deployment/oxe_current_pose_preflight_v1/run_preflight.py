"""Disposable paired current-pose engineering updates, no formal model selection."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
import numpy as np
from pose_features import HERE,read,sha,batch_features,validate_stats
from common import metadata


def initialize_runtime():
    import torch
    import tensorflow as tf
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
    assert str(torch.__version__)=='2.5.1+cu118' and torch.version.cuda=='11.8'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.set_num_interop_threads(48);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def replay_and_features(root,data,model,port,tokenizer,api,stats,selection):
    import torch
    from step_training_runner import tokenizer_identity
    from pose_runtime import digest
    assert tokenizer_identity(tokenizer)==read(root/'oxe_target_pair_short_v1/reference_contract.json')['tokenizer']
    rows=selection['train']+selection['validation'];items=[];state={role:[] for role in ['mask_control','real_pose']}
    for role in ['train','validation']:
        part=data.training if role=='train' else data.validation
        indices=[row['window_index'] for row in selection[role]]
        for row in selection[role]:
            _,start,_,e,_=part.store._row(row['window_index'])
            assert all(e[k]==row[k] for k in ['source','group_id','episode_key','payload_sha256']) and int(start)==row['input_row']
            items.append(part[row['window_index']])
        for arm in state:state[arm].append(batch_features(part.store,indices,stats,arm))
    state={arm:np.concatenate(v) for arm,v in state.items()}
    with np.load(HERE/'selected_state_reference.npz') as z:
        for arm in state:assert np.allclose(state[arm],z[arm],rtol=0,atol=1e-6),arm
    text=tokenizer([x[0] for x in items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    visual=[];language=[];model.eval()
    with torch.inference_mode():
        for lo in range(0,120,4):
            visual.append(model.vision_encoder(pixel_values=torch.stack([x[1] for x in items[lo:lo+4]]).cuda()).last_hidden_state.cpu())
            language.append(model.text_encoder(**{k:v[lo:lo+4].cuda() for k,v in text.items()}).last_hidden_state.cpu())
        visual=torch.cat(visual).cuda();language=torch.cat(language).cuda();mask=text['attention_mask'].cuda()
        contexts=[]
        for lo in range(0,120,8):
            original=model.adapter(visual[lo:lo+8],language[lo:lo+8],mask[lo:lo+8])
            for arm in state:
                delta=port(torch.from_numpy(state[arm][lo:lo+8]).cuda())
                assert torch.count_nonzero(delta)==0 and torch.equal(original,original+delta)
            contexts.append(original+port(torch.from_numpy(state['real_pose'][lo:lo+8]).cuda()))
        contexts=torch.cat(contexts);current=torch.stack([x[2] for x in items]).cuda()
        archive=read(HERE/'protocol.json');path=root/'runs/oxe-target-short-eval-fixed-232091/predictions_and_targets.npz'
        with np.load(path) as z:
            assert np.array_equal(z['targets'],np.stack([x[3].numpy() for x in items]))
            assert np.array_equal(z['masks'],np.stack([x[4].numpy() for x in items]))
            for j,seed in enumerate(selection['noise_seeds']):
                torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                pred=np.concatenate([model.sample(contexts[lo:lo+8],current[lo:lo+8]).cpu().numpy() for lo in range(0,120,8)])
                assert np.array_equal(pred,z['sample'][j]),('Original x0 endpoint replay differs',seed,float(np.max(np.abs(pred-z['sample'][j]))))
                print('CURRENT_POSE_ZERO_ENDPOINT_REPLAY',seed,'120/120 exact',flush=True)
    return {'passed':True,'both_ports_context_exact':True,'archived_x0_predictions_exact':True,
        'windows':120,'sampling_seeds':selection['noise_seeds'],'additional_sampling_draws_per_window':3,
        'null_prediction_identity_basis':'Exact same zero-augmented context and unchanged sampler, not a separate duplicate sampling run',
        'feature_sha256':{arm:hashlib.sha256(state[arm].tobytes()).hexdigest() for arm in state}}


def run_arm(root,role,payload,data,output,reference,stats):
    import torch
    import train as api
    import models
    from completed_policy_export import model_sha
    from step_training_runner import frozen_digest,tokenizer_identity
    from step_training_state import ResumableBatchPlan,UpdateSchedule,restore_checkpoint,restore_rng
    from resume_constants import training_constant_constructor
    from pose_runtime import zero_port,check_moments,digest,snapshot,restore,current_state,NoiseLog
    base=root/'oxe_target_pair_short_v1';contract=payload['contract'];config=contract['config'];events=[]
    assert config['model']['diffusion_prediction_type']=='sample'
    tokenizer=api.CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    assert tokenizer_identity(tokenizer)==contract['tokenizer']
    with training_constant_constructor(models.RobotAdapterModel,contract=contract,profile_path=base/'profile_sample.json',
            learned_state=api.trainable_state_dict,on_restored=events.append):
        model=models.RobotAdapterModel(config).cuda()
    saved_keys=api.trainable_state_dict(model);named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups=[[p for n,p in named if not n.startswith(api.GRIPPER_PARAMETER_PREFIXES)],
            [p for n,p in named if n.startswith(api.GRIPPER_PARAMETER_PREFIXES)]]
    optimizer=torch.optim.AdamW([{'params':g,'lr':rate} for g,rate in zip(groups,payload['schedule']['base_lrs'])],
        weight_decay=contract['arguments']['weight_decay'])
    indices=contract['split_indices']['train'];counts=Counter(data.samples[i]['source'] for i in indices)
    plan=ResumableBatchPlan(indices,batch_size=8,total_updates=65536,seed=42,dataset_identity=contract['dataset_identity'],
        mode='weighted_replacement',weights=[1/counts[data.samples[i]['source']] for i in indices])
    original_schedule=UpdateSchedule(**payload['schedule'])
    def load_source(state):
        assert set(state)==set(saved_keys)
        result=model.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and not set(saved_keys).intersection(result.missing_keys)
    restore_checkpoint(root/'runs/oxe-target-short-231297/sample/latest_step.pt',model_state_loader=load_source,
        optimizer=optimizer,plan=plan,schedule=original_schedule,contract=contract)
    check_moments(optimizer)
    assert plan.completed_updates==8192 and model_sha(api.trainable_state_dict(model))==reference['source_learned_sha256']
    assert frozen_digest(model,saved_keys)==reference['frozen_sha256']
    source_optimizer=digest(optimizer.state_dict());source_rng=digest(payload['rng'])
    port=zero_port('cuda',width=1024)
    optimizer.add_param_group({'params':list(port.parameters()),'lr':original_schedule.rates(8192)[0]})
    schedule=UpdateSchedule(total_updates=65536,base_lrs=payload['schedule']['base_lrs']+[payload['schedule']['base_lrs'][0]],
        kind=payload['schedule']['kind'],warmup_updates=payload['schedule']['warmup_updates'])
    schedule.apply(optimizer,8192)
    replay=None
    if role=='mask_control':replay=replay_and_features(root,data,model,port,tokenizer,api,stats,read(base/'selection.json'))
    # All construction, encoding and archival sampling precede restoration of the exact source RNG.
    restore_rng(payload['rng'])
    with np.load(HERE/'reference_suffix.npz') as z:suffix={k:z[k] for k in z.files}
    from instrument import method_source
    path,method=method_source();assert hashlib.sha256(method.encode()).hexdigest()==contract['diagnostic_target_pair']['diffusion_loss_source_sha256']
    noise=NoiseLog();models.__dict__['DIAG_HOOK']=noise
    exec(compile(method,str(path),'exec'),models.__dict__);models.RobotAdapterModel.diffusion_loss=models.__dict__['diffusion_loss']
    new_contract={'format':'current-pose-preflight-augmentation-v1','role':role,'protocol_sha256':sha(HERE/'protocol.json'),
        'source_contract_sha256':payload['contract_sha256'],'source_learned_sha256':reference['source_learned_sha256'],
        'features_and_stats_sha256':sha(HERE/'state_stats.json'),'base_config_unchanged':config,
        'new_port':{'shape':[1024,18],'bias':False,'initial_weight':'allzero','added_optimizer_group':True},
        'logical_steps':2,'physical_optimizer_calls_including_restore_replay':3}
    torch.cuda.reset_peak_memory_stats();port_initial=digest(port.state_dict());rows=[]
    def step():
        model.train();port.train();before=plan.completed_updates;ix=plan.begin_batch();j=before-8192
        assert ix==suffix['batches'][j,:int(suffix['lengths'][j])].tolist()
        assert [g['lr'] for g in optimizer.param_groups]==suffix['lrs'][j].tolist()+[float(suffix['lrs'][j,0])]
        torch.cuda.synchronize();tick=time.monotonic()
        texts,images,current,actions,masks=api.collate_batch([data[i] for i in ix])
        text=api.tokenise(tokenizer,texts,torch.device('cuda'));actions=actions.cuda();current=current.cuda();masks=masks.cuda()
        assert masks.sum((0,1)).cpu().tolist()==suffix['valid_dimensions'][j].tolist()
        features=torch.from_numpy(batch_features(data.training.store,ix,stats,role)).cuda()
        optimizer.zero_grad(set_to_none=True);noise.current=before+1
        context=model.get_context_vector(images.cuda(),text['input_ids'],text.get('attention_mask'))+port(features)
        output=model.diffusion_loss(actions,context,current)
        assert torch.equal(output[1],actions[...,:7])
        loss,pose,grip=api.policy_loss(model,output,actions,torch.nn.MSELoss(),current_grippers=current,supervision_masks=masks)
        assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
        audit_start=time.monotonic()
        base_grad=digest({n:p.grad for n,p in named}) if before==8192 else None
        assert port.weight.grad is not None and torch.isfinite(port.weight.grad).all()
        vg=float(port.weight.grad[:,:9].norm());mg=float(port.weight.grad[:,9:].norm())
        if role=='mask_control':assert vg==0
        else:assert vg>0
        assert mg>0
        audit_seconds=time.monotonic()-audit_start
        norm=torch.nn.utils.clip_grad_norm_([p for _,p in named]+list(port.parameters()),1.,error_if_nonfinite=True)
        optimizer.step();plan.commit_batch();schedule.apply(optimizer,plan.completed_updates);torch.cuda.synchronize()
        row={'update':plan.completed_updates,'indices':ix,'loss':float(loss.detach()),'pose_loss':float(pose.detach()),
            'gripper_loss':float(grip.detach()),'gradient_norm':float(norm),'pose_value_grad_norm':vg,'mask_grad_norm':mg,
            'base_gradient_sha256_before_clipping':base_grad,'measured_update_seconds':time.monotonic()-tick-audit_seconds}
        print('CURRENT_POSE_ENGINEERING_UPDATE',role,json.dumps(row),flush=True);return row
    rows.append(step())
    # Serialize one committed boundary in RAM; do not write new1.6GB checkpoints to disk.
    memory=snapshot(model,port,optimizer,plan,schedule,new_contract,api.trainable_state_dict,True)
    serialized_bytes=memory.getbuffer().nbytes
    midpoint=current_state(model,port,optimizer,plan,schedule,api.trainable_state_dict,True)
    rows.append(step());uninterrupted=current_state(model,port,optimizer,plan,schedule,api.trainable_state_dict,True)
    uninterrupted_noise=copy.deepcopy(noise.events)
    restore(memory,model,port,optimizer,plan,schedule,new_contract,api.trainable_state_dict)
    assert current_state(model,port,optimizer,plan,schedule,api.trainable_state_dict,True)==midpoint
    repeated=step();after=current_state(model,port,optimizer,plan,schedule,api.trainable_state_dict,True)
    assert after==uninterrupted and noise.events[-1]==uninterrupted_noise[-1]
    assert {k:v for k,v in repeated.items() if k!='measured_update_seconds'}=={k:v for k,v in rows[-1].items() if k!='measured_update_seconds'}
    assert digest(port.state_dict())!=port_initial
    if role=='mask_control':assert torch.count_nonzero(port.weight[:,:9])==0
    else:assert torch.count_nonzero(port.weight[:,:9])>0
    assert frozen_digest(model,saved_keys)==reference['frozen_sha256']
    result={'passed':True,'role':role,'source_optimizer_sha256_before_augmentation':source_optimizer,'source_rng_sha256':source_rng,
        'source_learned_sha256':reference['source_learned_sha256'],'initial_port_sha256':port_initial,
        'new_port_parameter_count':port.weight.numel(),'rows':rows,'noise_stream':uninterrupted_noise,
        'serialized_midpoint_bytes_in_ram':serialized_bytes,'disk_full_checkpoints_written':0,
        'optimizer_calls':3,'logical_new_updates':2,'completed_update':plan.completed_updates,
        'restore_second_step_exact':True,'restore_base_port_adam_plan_schedule_rng_exact':True,
        'final_state_digests':after,'frozen_digest_unchanged':True,'frozen_restore_events':events,
        'zero_endpoint_replay':replay,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'warm_update_seconds':rows[1]['measured_update_seconds'],'trained_weights_retained_for_formal_use':False}
    memory.close();return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();root=a.root.resolve()
    admission=metadata(root)
    if a.check_only:print(json.dumps(admission,indent=2));return
    assert os.environ.get('SLURM_JOB_ID') and a.output is not None and not a.output.exists()
    a.output.mkdir();base=root/'oxe_target_pair_short_v1';sys.path[:0]=[str(base),str(base/'code')]
    initialize_runtime()
    import torch
    import train as api
    from verify import arm
    from completed_policy_export import contract_sha,model_sha
    from oxe_nominal_dataset import NominalTrainValidationDataset
    arm(root/'runs/oxe-target-short-231297/sample','sample')
    path=root/'runs/oxe-target-short-231297/sample/latest_step.pt';stat_before=(path.stat().st_size,path.stat().st_mtime_ns)
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True);reference=read(HERE/'protocol.json')
    assert payload['contract_sha256']==contract_sha(payload['contract'])==reference['source_contract_sha256']
    assert model_sha(payload['model'])==reference['source_learned_sha256'] and payload['plan']['completed_updates']==8192
    data=NominalTrainValidationDataset(root/'data/oxe_nominal_pool_v1',root/'oxe_dino_main_preflight_v1/inputs/train',root/'oxe_dino_main_preflight_v1/inputs/validation')
    assert data.nominal_target_binding==payload['contract']['config']['action']['nominal_target_binding']
    assert data.frozen_nominal_splits==payload['contract']['split_indices']
    assert api.dataset_split_identity(data)==payload['contract']['dataset_identity']
    assert data.record_integrity_contract==payload['contract']['source_record_integrity']
    stats=read(HERE/'state_stats.json');validate_stats(data.training.store,stats)
    from step_training_state import UpdateSchedule
    schedule=UpdateSchedule(**payload['schedule'])
    with np.load(HERE/'reference_suffix.npz') as z:
        assert all(schedule.rates(8192+i)==z['lrs'][i].tolist() for i in range(2048)),'LR prefix mismatch before model creation'
    report={'engineering_passed':False,'source_job':231297,'source_update':8192,'formal_training_submitted':False,
        'generalization_verdict':None,'groups':{},'protocol_sha256':sha(HERE/'protocol.json')}
    try:
        for role in ['mask_control','real_pose']:
            report['groups'][role]=run_arm(root,role,payload,data,a.output,reference,stats)
            gc.collect();torch.cuda.empty_cache()
        c,r=[report['groups'][role] for role in ['mask_control','real_pose']]
        assert c['source_optimizer_sha256_before_augmentation']==r['source_optimizer_sha256_before_augmentation']
        assert c['source_rng_sha256']==r['source_rng_sha256'] and c['initial_port_sha256']==r['initial_port_sha256']
        assert c['noise_stream']==r['noise_stream']
        for key in ['loss','pose_loss','gripper_loss','base_gradient_sha256_before_clipping']:assert c['rows'][0][key]==r['rows'][0][key],key
        assert (path.stat().st_size,path.stat().st_mtime_ns)==stat_before
        report.update(engineering_passed=True,physical_optimizer_calls=6,logical_steps_per_group=2,
            source_checkpoint_size_mtime_unchanged=True,paired_initial_loss_base_gradients_and_noise_exact=True,
            proposed_additional_formal_updates_per_group=2048,
            estimated_pair_update_seconds_excluding_replay_eval_io=2048*(c['warm_update_seconds']+r['warm_update_seconds']),
            formal_code_released=False,automatic_following_action=False,
            scope='Disposable engineering updates with retained sourceAdam/RNG; no new performance or main-policy verdict')
    finally:
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('CURRENT_POSE_PREFLIGHT_REPORT',a.output/'report.json',flush=True)
    print('CURRENT POSE PREFLIGHT PASSED; no formal submission or generalization verdict',flush=True)


if __name__=='__main__':main()
