"""Read-only audit of the three contracts raised by Lipo's fifth email.

Uses all frozen expansion windows for labels/input checks and one retained
train-only model for live forward/scheduler checks. No optimizer or downloads.
The latter cannot certify absent cluster checkpoints byte-for-byte.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from dataset import UnifiedRobotDataset, prepare_image, POSITION_SCALE_METERS, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
from models import RobotAdapterModel, diffusion_betas
from train import collate_batch, tokenise
from evaluate_multitask_pilot import metrics
from evaluate_bridge_endpoint_metrics import pose_errors
from bridge_xyz_normalization import NormalizedRobotAdapterModel
from fit_pybullet_fixed_grasp import trainable_hash
from replay_and_fit_bridge_positive import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/teacher_contract_audit_20261011'
DATA=Path('E:/dataset/bridge_v2_0.0.1/0.0.1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def rotation(angles):
    # Independent world-frame matrix: Rz(yaw) Ry(pitch) Rx(roll).
    r,p,y=map(float,angles);cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                     [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def distribution(x):
    x=np.asarray(x,dtype=np.float64)
    flat=x.reshape(-1,3)
    return dict(windows=len(x),targets=len(flat),min=flat.min(0).tolist(),max=flat.max(0).tolist(),
                mean=flat.mean(0).tolist(),std_population=flat.std(0).tolist(),
                percentiles={str(p):np.percentile(flat,p,axis=0).tolist() for p in [1,50,99]},
                outside_abs3_window_fraction=float((np.abs(x)>3).any((1,2)).mean()))


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    began=time.monotonic()
    files=['dataset.py','models.py','adapter.py','diffusion_decoder.py','train.py',
           'evaluate_multitask_pilot.py','evaluate_bridge_endpoint_metrics.py','bridge_xyz_normalization.py']
    source={n:sha(ROOT/'code'/n) for n in files}
    reference_path=ROOT/'reports/bridge-xyz-eval-197349.json'
    ref=json.loads(reference_path.read_text())
    rows=ref['window_selection']
    assert len(rows)==1876
    stats=json.loads((ROOT/'reports/bridge-xyz-stats-196693.json').read_text())
    data=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
    data.chunk_size=16;data.bridge_gripper_policy='reverse_scan_valid_steps_v2';data.bridge_current_gripper='continuous'
    original_loader=UnifiedRobotDataset._load_tfrecord_example
    # Read each episode once. Production getter still parses all original fields,
    # constructs all action labels, decodes RGB and runs prepare_image unchanged.
    current_key=None;current_example=None
    xyz={'train':[],'validation':[]}; representatives={'train':[],'validation':[]}
    instructions={'train':set(),'validation':set()};episode_sets={'train':set(),'validation':set()}
    per_episode={};pixel_mean={'train':[],'validation':[]}
    geometry_error=0.;independent_metric_error=0.;same_image_error=0.;per_shard={};instruction_spelling=[]
    for number,row in enumerate(rows,1):
        partition=row['partition'];key=(row['shard'],row['record_index'])
        path=DATA/row['shard']
        if key!=current_key:
            current_example=original_loader(str(path),row['record_index']);current_key=key
            per_shard.setdefault(row['shard'],dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
        data._load_tfrecord_example=lambda *_:current_example
        sample=dict(file_path=str(path),record_index=row['record_index'],start_index=row['start_index'])
        item=data._get_tfrecord_bridge_state_action(sample)
        language,image,current,target=item
        assert torch.isfinite(image).all() and target.shape==(16,8)
        assert ' '.join(language.lower().split())==' '.join(row['task'].lower().split()), (language,row['task'])
        if language != row['task'] and len(instruction_spelling)<10:
            instruction_spelling.append(dict(raw_language=language,manifest_task=row['task']))
        xyz[partition].append(target[:,:3].numpy())
        instructions[partition].add(language);episode_sets[partition].add(key)
        feature=current_example.features.feature
        raw=np.asarray(Image.open(io.BytesIO(feature['steps/observation/image_0'].bytes_list.value[row['start_index']])).convert('RGB'))
        same_image_error=max(same_image_error,float((prepare_image(raw)-image).abs().max()))
        pixel_mean[partition].append(image.mean((1,2)).numpy())
        states=np.asarray(feature['steps/observation/state'].float_list.value,dtype=np.float32).reshape(-1,7)
        s=row['start_index'];R=rotation(states[s,3:6])
        observed=states[s+1:s+17,:3].astype(np.float64)
        reconstructed=states[s,:3]+(target[:,:3].numpy()*POSITION_SCALE_METERS)@R.T
        geometry_error=max(geometry_error,float(np.abs(reconstructed-observed).max()))
        # Known physical offset checks the actual production metric's unit.
        perturbed=target.numpy().copy();perturbed[:,:3]+=np.array([.1,-.2,.05],dtype=np.float32)
        predicted_world=states[s,:3]+(perturbed[:,:3]*POSITION_SCALE_METERS)@R.T
        physical_error=float(np.linalg.norm(predicted_world-reconstructed,axis=-1).mean()*100)
        reported=metrics(perturbed[None],target.numpy()[None])['position_cm']
        independent_metric_error=max(independent_metric_error,abs(reported-physical_error))
        ek=row['shard']+'::'+str(row['record_index'])
        per_episode.setdefault(ek,dict(partition=partition,task=language,xyz=[]))['xyz'].append(target[:,:3].numpy())
        if len(representatives[partition])<3 and (not representatives[partition] or representatives[partition][-1]['episode']!=ek):
            representatives[partition].append(dict(item=(*item,torch.ones_like(target)),raw=raw.copy(),episode=ek))
        if number%200==0 or number==len(rows): print('TEACHER_LABEL_AUDIT',number,'/',len(rows),flush=True)
    xyz={k:np.stack(v) for k,v in xyz.items()}
    assert len(xyz['train'])==1836 and len(xyz['validation'])==40
    assert not (episode_sets['train'] & episode_sets['validation'])
    train_stats=distribution(xyz['train']);validation_stats=distribution(xyz['validation'])
    statistics_error=max(float(np.max(np.abs(np.asarray(train_stats[k])-stats['overall'][k])))
                         for k in ['min','max','mean','std_population'])
    assert statistics_error<1e-7 and geometry_error<1e-6 and independent_metric_error<1e-4
    candidate=stats['candidate_axis_limits_normalizer']
    scale=np.asarray(candidate['scale_from_existing_xyz']);offset=np.asarray(candidate['offset'])
    assert stats['normalizer_fit_partitions']==['train'] and not stats['validation_windows_materialized']
    assert np.allclose(scale,2/(xyz['train'].reshape(-1,3).max(0)-xyz['train'].reshape(-1,3).min(0)),atol=1e-7)
    assert np.allclose(ref['normalizers']['axis_limits']['scale'],scale)
    assert np.allclose(ref['normalizers']['axis_limits']['offset'],offset)
    converted={k:x*scale+offset for k,x in xyz.items()}
    inverse_error=max(float(np.abs((x-offset)/scale-xyz[k]).max()) for k,x in converted.items())
    normalization=dict(position_scale_meters=POSITION_SCALE_METERS,fixed_scale_fitted_on_neither_split=True,
        training=train_stats,validation=validation_stats,training_statistics_recomputed_max_difference=statistics_error,
        label_to_independent_world_position_max_difference_m=geometry_error,
        production_metric_vs_independent_world_error_max_difference_cm=independent_metric_error,
        cached_statistics_fit_train_only=True,axis_limits_inverse_max_difference_existing_units=inverse_error,
        validation_outside_training_axis_min_max_component_fraction=((converted['validation']<-1)|(converted['validation']>1)).mean((0,1)).tolist(),
        validation_axis_limits_range=dict(min=converted['validation'].min((0,1)).tolist(),max=converted['validation'].max((0,1)).tolist()))
    per_episode={k:dict(partition=v['partition'],task=v['task'],xyz=distribution(v['xyz'])) for k,v in per_episode.items() if v['partition']=='validation'}
    ck=ROOT/'reports/bridge_joint_both_events_1000_local_20261011/final.pt'
    payload=torch.load(ck,map_location='cpu',weights_only=False)
    assert payload['config']['action']['position_scale_meters']==POSITION_SCALE_METERS
    model=RobotAdapterModel(payload['config']).cuda().eval()
    missing=model.load_state_dict(payload['trainable_state_dict'],strict=False)
    assert not missing.unexpected_keys and all(n.startswith(('vision_encoder.','text_encoder.')) for n in missing.missing_keys)
    before=[trainable_hash(model),digest(model.vision_encoder),digest(model.text_encoder)]
    tokenizer=CLIPTokenizer.from_pretrained(payload['config']['model']['name'],local_files_only=True)
    selected=representatives['train']+representatives['validation']
    language,images,current,targets,_=collate_batch([x['item'] for x in selected])
    train_text=tokenise(tokenizer,language,torch.device('cuda'))
    eval_text={k:v.cuda() for k,v in tokenizer(language,padding=True,truncation=True,return_tensors='pt').items()}
    assert all(torch.equal(train_text[k],eval_text[k]) for k in train_text)
    pixels=images.cuda()
    model.train()
    train_visual=model.vision_encoder(pixel_values=pixels).last_hidden_state
    train_language=model.text_encoder(**train_text).last_hidden_state
    frozen_train_modes=[model.vision_encoder.training,model.text_encoder.training]
    model.eval()
    eval_visual=model.vision_encoder(pixel_values=pixels).last_hidden_state
    eval_language=model.text_encoder(**eval_text).last_hidden_state
    visual_error=float((train_visual-eval_visual).abs().max());text_error=float((train_language-eval_language).abs().max())
    context=model.get_context_vector(pixels,eval_text['input_ids'],eval_text['attention_mask'])
    cached_context=model.adapter(train_visual,train_language,train_text['attention_mask'])
    context_error=float((context-cached_context).abs().max())
    assert visual_error==0 and text_error==0 and context_error<1e-5
    # Explicit official OpenAI transform; the local weight/tokenizer snapshot
    # contains no preprocessor_config.json. Do not download or modify that cache.
    official_transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),
                                CenterCrop(224),ToTensor(),
                                Normalize(CLIP_IMAGE_MEAN.flatten().tolist(),CLIP_IMAGE_STD.flatten().tolist())])
    official=torch.stack([official_transform(Image.fromarray(x['raw']).convert('RGB')) for x in selected])
    official_diff=(official-images).abs()
    preprocessing=dict(all_1876_raw_rgb_to_loader_max_difference=same_image_error,
        representative_samples=len(selected),tokenizer_train_eval_identical=True,
        train_mode_frozen_encoder_training_flags=frozen_train_modes,
        visual_train_eval_max_difference=visual_error,text_train_eval_max_difference=text_error,
        eval_context_vs_recomputed_cached_tokens_max_difference=context_error,
        clip_mean=CLIP_IMAGE_MEAN.flatten().tolist(),clip_std=CLIP_IMAGE_STD.flatten().tolist(),
        official_reference_source='https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py',
        official_reference_transform='Resize224 BICUBIC; CenterCrop224; RGB; ToTensor; CLIP mean/std',
        local_preprocessor_config_present=False,
        project_vs_official_processor_mean_abs_difference=float(official_diff.mean()),
        project_vs_official_processor_max_abs_difference=float(official_diff.max()),
        image_sizes=[list(x['raw'].shape) for x in selected],
        mean_normalized_image_channel_value={k:np.stack(v).mean(0).tolist() for k,v in pixel_mean.items()},
        training_instructions=sorted(instructions['train']),validation_instructions=sorted(instructions['validation']),
        instruction_sets_equal=instructions['train']==instructions['validation'],raw_vs_manifest_spelling_examples=instruction_spelling)
    assert model.num_diffusion_steps==100 and model.beta_schedule=='squaredcos_cap_v2'
    beta_error=float((model.betas-diffusion_betas(100,model.beta_schedule).cuda()).abs().max())
    stored={k:float((model.state_dict()[k].cpu()-payload['trainable_state_dict'][k]).abs().max())
            for k in ['betas','alphas','alpha_bars','posterior_variance','posterior_mean_x0','posterior_mean_xt']}
    seen=[]
    hook=model.diffusion_decoder.register_forward_pre_hook(lambda _,args:seen.append(args[1].detach().cpu().tolist()))
    torch.manual_seed(9511)
    _=model.diffusion_loss(targets[:1].cuda(),context[:1],current[:1].cuda())
    training_timestep=seen.pop()
    _=model.sample(context[:1],current[:1].cuda())
    hook.remove()
    sequence=[x[0] for x in seen]
    assert sequence==list(range(99,-1,-1)) and beta_error==0 and max(stored.values())==0
    scheduler=dict(checkpoint_sha256=sha(ck),live_checkpoint_scope='retained 8 training-window joint-fit positive control; not absent full cluster checkpoint',
        training_num_timesteps=model.num_diffusion_steps,beta_schedule=model.beta_schedule,
        prediction_type=model.diffusion_prediction_type,registered_beta_recomputed_max_difference=beta_error,
        checkpoint_scheduler_buffer_max_differences=stored,observed_training_timestep=training_timestep,
        observed_native_inference_timesteps=sequence,alpha_bar_terminal=float(model.alpha_bars[-1]),
        historical_official_comparison=json.loads((ROOT/'reports/bridge-scheduler-196610.json').read_text()))
    # Exercise the candidate's actual inverse function on real validation labels.
    normalizer=NormalizedRobotAdapterModel.__new__(NormalizedRobotAdapterModel)
    torch.nn.Module.__init__(normalizer);normalizer.xyz_normalizer=dict(scale=scale.tolist(),offset=offset.tolist())
    val=torch.zeros((40,16,8));val[...,:3]=torch.as_tensor(xyz['validation'])
    actual_inverse=float((normalizer.convert(normalizer.convert(val),inverse=True)-val).abs().max())
    normalization['actual_axis_convert_roundtrip_max_difference_existing_units']=actual_inverse
    assert actual_inverse<1e-6
    assert [trainable_hash(model),digest(model.vision_encoder),digest(model.text_encoder)]==before
    assert all(sha(ROOT/'code'/name)==value for name,value in source.items())
    assert all((DATA/name).stat().st_size==v['bytes'] and (DATA/name).stat().st_mtime_ns==v['mtime_ns'] for name,v in per_shard.items())
    report=dict(stage='teacher_three_contracts_readonly_audit',trained=False,downloads=False,
        production_code_changed=False,parameters_unchanged=True,source_sha256=source,
        probe_source_sha256=sha(Path(__file__)),window_reference_sha256=sha(reference_path),
        raw_shards_stat=per_shard,counts=dict(train_windows=1836,validation_windows=40,train_episodes=len(episode_sets['train']),validation_episodes=len(episode_sets['validation'])),
        normalization=normalization,preprocessing=preprocessing,scheduler=scheduler,
        validation_episode_ranges=per_episode,elapsed_seconds=time.monotonic()-began,
        limits=['No optimization or new generalization evaluation.',
                'All frozen expansion windows re-read, but live model check used available small-sample checkpoint.',
                'Preprocessing interpolation differs from official CLIP; no causal attribution without controlled experiment.',
                'Matching contracts do not prove distribution coverage, adequate information, or fitting ability.'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('NORMALIZATION',json.dumps(normalization),flush=True)
    print('PREPROCESSING',json.dumps(preprocessing),flush=True)
    print('SCHEDULE',model.num_diffusion_steps,model.beta_schedule,'buffers+steps verified',flush=True)
    print('TEACHER CONTRACT AUDIT COMPLETE',OUT/'report.json',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    main()
