"""Two actually trained native targets, fixed expert windows, zero optimization."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from protocols import HERE,read,sha
from metrics import summarize


def verify_evaluation(output):
    r=read(output/'evaluation/report.json');selection=read(HERE/'selection.json')
    assert r['execution_complete'] and r['optimizer_updates']==0 and r['selection_sha256']==sha(HERE/'selection.json')
    assert r['final_frozen_sha256']==read(HERE/'pair_protocol.json')['frozen_sha256']
    with np.load(output/'evaluation/predictions_and_targets.npz') as z:
        assert z['targets'].shape==z['masks'].shape==(120,16,8)
        outputs={name:z[name] for name in ['epsilon','sample','static']}
        assert all(v.shape==(3,120,16,8) and np.isfinite(v).all() for v in outputs.values())
        static=np.zeros_like(z['targets']);static[...,6]=1
        assert np.array_equal(outputs['static'],np.repeat(static[None],3,axis=0))
        calculated=summarize(outputs,z['targets'],z['masks'],selection)
        assert calculated==r['metrics'],'Stored generated metrics differ from predictions'
    for target in ['epsilon','sample']:
        assert r['checkpoint_metadata'][target]['learned_sha256']==read(output/target/'arm_receipt.json')['final_learned_sha256']
    return {'verified':True,'metrics':r['metrics'],'optimizer_updates':0,'automatic_following_action':False}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();out=a.output/'evaluation';assert not out.exists();out.mkdir()
    sys.path.insert(0,str(HERE/'code'))
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.set_num_interop_threads(48);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    assert torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA A40'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    from models import RobotAdapterModel
    from transformers import CLIPTokenizer
    from oxe_nominal_dataset import NominalOXEDataset
    from train import trainable_state_dict
    from completed_policy_export import model_sha,contract_sha
    from step_training_runner import frozen_digest,tokenizer_identity
    from resume_constants import training_constant_constructor
    from verify import verify_pair
    verify_pair(a.output)
    selection=read(HERE/'selection.json');reference=read(HERE/'reference_contract.json');rows=selection['train']+selection['validation']
    bound=reference['offline_training_protocol']['cluster_preflight']
    assert str(torch.__version__)==bound['torch_version'] and torch.version.cuda==bound['cuda_runtime']
    assert len(rows)==120 and selection['noise_seeds']==[1101,1102,1103] and selection['sampling_steps']==100
    items=[]
    with np.load(HERE/'reference_prefix.npz') as z:
        presented=np.bincount(z['batches'].reshape(-1),minlength=145812)
    for role in ['train','validation']:
        data=NominalOXEDataset(root/'data/oxe_nominal_pool_v1',root/f'oxe_dino_main_preflight_v1/inputs/{role}',partition=role)
        assert data.nominal_target_binding==reference['config']['action']['nominal_target_binding']['partition_bindings'][role]
        for row in selection[role]:
            wi=row['window_index'];_,start,_,e,_=data.store._row(wi)
            for key in ['source','group_id','episode_key','payload_sha256']:assert e[key]==row[key],key
            assert int(start)==row['input_row'] and row['role']==role
            if role=='train':assert int(presented[wi])==row['prefix_presentations']>0
            items.append(data[wi])
        del data
    assert not ({(r['source'],r['group_id']) for r in selection['train']}&{(r['source'],r['group_id']) for r in selection['validation']})
    tokenizer=CLIPTokenizer.from_pretrained(str(root/'baseline_setup/clip-vit-large-patch14-v1'),local_files_only=True)
    assert tokenizer_identity(tokenizer)==reference['tokenizer']
    events=[]
    with training_constant_constructor(RobotAdapterModel,contract=reference,profile_path=HERE/'profile_epsilon.json',
            learned_state=trainable_state_dict,on_restored=events.append):
        model=RobotAdapterModel(reference['config']).cuda().eval()
    keys=trainable_state_dict(model);initial_frozen=frozen_digest(model,keys)
    assert initial_frozen==reference['frozen_state_sha256'] and len(events)==1
    texts=tokenizer([x[0] for x in items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    visual=[];language=[]
    with torch.inference_mode():
        for lo in range(0,len(items),4):
            visual.append(model.vision_encoder(pixel_values=torch.stack([x[1] for x in items[lo:lo+4]]).cuda()).last_hidden_state.cpu())
            language.append(model.text_encoder(**{k:v[lo:lo+4].cuda() for k,v in texts.items()}).last_hidden_state.cpu())
    visual=torch.cat(visual).cuda();language=torch.cat(language).cuda();attention=texts['attention_mask'].cuda()
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    current=torch.stack([x[2] for x in items]).cuda();targets=np.stack([x[3].numpy() for x in items]);masks=np.stack([x[4].numpy() for x in items])
    outputs={};result={'execution_complete':False,'optimizer_updates':0,'checkpoint_metadata':{},
        'selection_sha256':sha(HERE/'selection.json'),'frozen_restore_events':events,
        'scope':'Native target-specific trained weights; no flag reinterpretation of old epsilon weights, no LIBERO',
        'automatic_following_action':False}
    tick=time.monotonic()
    try:
        for target in ['epsilon','sample']:
            payload=torch.load(a.output/target/'latest_step.pt',map_location='cpu',weights_only=False,mmap=True)
            c=payload['contract'];assert contract_sha(c)==payload['contract_sha256']
            cfg=copy.deepcopy(c['config']);assert cfg['model']['diffusion_prediction_type']==target
            cfg['model']['diffusion_prediction_type']='epsilon';assert cfg==reference['config']
            state=payload['model'];digest=model_sha(state)
            assert digest==read(a.output/target/'arm_receipt.json')['final_learned_sha256'] and set(state)==set(keys)
            loaded=model.load_state_dict(state,strict=False)
            assert not loaded.unexpected_keys and not set(keys).intersection(loaded.missing_keys)
            # This flag follows the corresponding NATIVE TRAINED checkpoint.
            model.diffusion_prediction_type=target
            assert model_sha(trainable_state_dict(model))==digest
            model.eval();contexts=[];draws=[]
            with torch.inference_mode():
                for lo in range(0,120,8):contexts.append(model.adapter(visual[lo:lo+8],language[lo:lo+8],attention[lo:lo+8]))
                contexts=torch.cat(contexts)
                for seed in selection['noise_seeds']:
                    torch.manual_seed(seed);torch.cuda.manual_seed(seed);prediction=[]
                    for lo in range(0,120,8):prediction.append(model.sample(contexts[lo:lo+8],current[lo:lo+8]).cpu().numpy())
                    draws.append(np.concatenate(prediction))
                    print('MAIN_TARGET_GENERATION',target,seed,'120/120',flush=True)
            outputs[target]=np.stack(draws)
            assert model_sha(trainable_state_dict(model))==digest
            result['checkpoint_metadata'][target]={'learned_sha256':digest,'trained_native_target':target,
                'checkpoint_contract_sha256':payload['contract_sha256'],'prefix_updates':8192}
            del payload,state,contexts,draws
        static=np.zeros_like(targets);static[...,6]=1
        outputs['static']=np.repeat(static[None],3,axis=0)
        result['metrics']=summarize(outputs,targets,masks,selection)
        result.update(execution_complete=True,final_frozen_sha256=frozen_digest(model,keys),elapsed_seconds=time.monotonic()-tick)
        assert result['final_frozen_sha256']==initial_frozen
    finally:
        np.savez_compressed(out/'predictions_and_targets.npz',targets=targets,masks=masks,**outputs)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('MAIN_TARGET_GENERATED_REPORT',out/'report.json',flush=True)
    print('MAIN_TARGET_BRIDGE64',json.dumps(result['metrics']['primary_bridge64']),flush=True)
    verify_evaluation(a.output)


if __name__=='__main__':main()
