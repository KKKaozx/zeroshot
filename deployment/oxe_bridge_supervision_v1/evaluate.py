"""Paired old-mixed/counterfactual inference, no parameter optimization."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import argparse

BUNDLE=Path(__file__).resolve().parent
sys.path.insert(0,str(BUNDLE/'code'))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(pred,target,mask,indices):
    import numpy as np
    output=[]
    for j in indices:
        valid=mask[j,:,0]>.5
        assert valid.any() and np.array_equal(mask[j,:,:7]>.5,np.repeat(valid[:,None],7,axis=1))
        p=pred[:,j,valid,:7].astype(float);t=target[j,valid,:7].astype(float)
        assert np.isfinite(p).all() and np.isfinite(t).all()
        position=np.linalg.norm(p[...,:3]-t[...,:3],axis=-1)
        pq=p[...,3:7];tq=t[...,3:7]
        assert (np.linalg.norm(pq,axis=-1)>0).all() and (np.linalg.norm(tq,axis=-1)>0).all()
        pq/=np.linalg.norm(pq,axis=-1,keepdims=True);tq/=np.linalg.norm(tq,axis=-1,keepdims=True)
        rotation=np.degrees(2*np.arccos(np.clip(np.abs((pq*tq).sum(axis=-1)),0,1)))
        output.append(dict(window_index=int(j),path_position_normalized=float(position.mean()),
            path_rotation_deg=float(rotation.mean()),endpoint_position_normalized=float(position[:,-1].mean()),
            endpoint_rotation_deg=float(rotation[:,-1].mean())))
    return dict(windows=len(output),**{key:float(np.mean([x[key] for x in output])) for key in output[0] if key!='window_index'},per_window=output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/projects/Zeroshot'))
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    assert not args.output.exists();args.output.mkdir(parents=True)
    import numpy as np
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    reference=read(BUNDLE/'reference_contract.json');protocol=read(BUNDLE/'evaluation_windows.json')
    assert torch.cuda.get_device_name()=='NVIDIA A40'
    assert str(torch.__version__)==reference['offline_training_protocol']['cluster_preflight']['torch_version']
    from models import RobotAdapterModel
    from transformers import CLIPTokenizer
    from oxe_nominal_dataset import NominalOXEDataset
    from train import trainable_state_dict
    from completed_policy_export import model_sha
    from step_training_runner import frozen_digest,tokenizer_identity
    from verify import verify
    trial=root/'runs/oxe-bridge-supervision-s42';verify(trial,'fit',2)
    data=NominalOXEDataset(root/'data/oxe_nominal_pool_v1',root/'oxe_dino_main_preflight_v1/inputs/validation',partition='validation')
    assert data.nominal_target_binding==reference['config']['action']['nominal_target_binding']['partition_bindings']['validation']
    items=[]
    for row in protocol['rows']:
        wi=row['window_index'];ei,start,_,episode,_=data.store._row(wi)
        assert episode['group_id']==row['group_id'] and episode['payload_sha256']==row['payload_sha256']
        assert int(start)==row['input_row'];items.append(data[wi])
    tokenizer=CLIPTokenizer.from_pretrained(str(root/'baseline_setup/clip-vit-large-patch14-v1'),local_files_only=True)
    assert tokenizer_identity(tokenizer)==reference['tokenizer']
    model=RobotAdapterModel(reference['config']).cuda().eval()
    keys=trainable_state_dict(model)
    initial_frozen=frozen_digest(model,keys);assert initial_frozen==reference['frozen_state_sha256']
    texts=tokenizer([x[0] for x in items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    visual=[];language=[]
    with torch.inference_mode():
        for lo in range(0,64,4):
            visual.append(model.vision_encoder(pixel_values=torch.stack([x[1] for x in items[lo:lo+4]]).cuda()).last_hidden_state.cpu())
            language.append(model.text_encoder(**{k:v[lo:lo+4].cuda() for k,v in texts.items()}).last_hidden_state.cpu())
    visual=torch.cat(visual).cuda();language=torch.cat(language).cuda();attention=texts['attention_mask'].cuda()
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
    current=torch.stack([x[2] for x in items]).cuda()
    targets=np.stack([x[3].numpy() for x in items]);masks=np.stack([x[4].numpy() for x in items])
    outputs={};result=dict(execution_complete=False,scope=protocol['scope'],optimizer_updates=0,
        primary_final_tasks_accessed=False,checkpoint_metadata={},selection_sha256=sha(BUNDLE/'evaluation_windows.json'))
    tick=time.monotonic()
    try:
        for label,path in [('mixed',root/'runs/oxe-dino-noaug-s42/final_policy.pt'),('bridge_supervision',trial/'final_policy.pt')]:
            payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            state=payload['trainable_state_dict'];digest=model_sha(state)
            assert digest==payload['learned_state_sha256'] and set(state)==set(keys)
            if label=='mixed':
                assert payload['training_contract']==reference
            else:assert payload['training_contract']['supervision_counterfactual']['protocol_sha256']==sha(BUNDLE/'protocol.json')
            missing=model.load_state_dict(state,strict=False)
            assert not missing.unexpected_keys and not set(keys).intersection(missing.missing_keys)
            assert model_sha(trainable_state_dict(model))==digest
            model.eval();contexts=[];draws=[]
            with torch.inference_mode():
                for lo in range(0,64,8):contexts.append(model.adapter(visual[lo:lo+8],language[lo:lo+8],attention[lo:lo+8]))
                contexts=torch.cat(contexts)
                for seed in protocol['noise_seeds']:
                    torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                    prediction=[]
                    for lo in range(0,64,8):
                        prediction.append(model.sample(contexts[lo:lo+8],current[lo:lo+8]).cpu().numpy())
                    draws.append(np.concatenate(prediction))
                    print('BRIDGE_SUPERVISION_EVAL_DRAW',label,seed,'64/64',flush=True)
            outputs[label]=np.stack(draws)
            assert model_sha(trainable_state_dict(model))==digest
            result['checkpoint_metadata'][label]=dict(path=str(path),learned_sha256=digest)
            del payload,state,contexts,draws
        identity=np.zeros_like(targets);identity[...,6]=1
        outputs['static']=np.repeat(identity[None],3,axis=0)
        result['primary64']={name:metrics(pred,targets,masks,range(64)) for name,pred in outputs.items()}
        result['anchor8_secondary']={name:metrics(pred,targets,masks,range(8)) for name,pred in outputs.items()}
        baseline=result['primary64']['mixed'];candidate=result['primary64']['bridge_supervision']
        result['candidate_minus_mixed']={key:candidate[key]-baseline[key] for key in ['path_position_normalized','path_rotation_deg','endpoint_position_normalized','endpoint_rotation_deg']}
        rng=np.random.default_rng(20261027)
        bootstrap_indices=rng.integers(0,64,size=(10000,64))
        result['paired_validation_group_bootstrap95']={}
        for key in result['candidate_minus_mixed']:
            differences=np.asarray([new[key]-old[key] for new,old in zip(candidate['per_window'],baseline['per_window'])])
            result['paired_validation_group_bootstrap95'][key]=np.quantile(differences[bootstrap_indices].mean(axis=1),[.025,.975]).tolist()
        result['uncertainty_scope']='Paired validation-group variability after averaging three sampling draws; excludes training-seed uncertainty'
        result.update(execution_complete=True,final_frozen_sha256=frozen_digest(model,keys),
            elapsed_seconds=time.monotonic()-tick,units='Position normalized recorded units (divisor0.1); no new physical calibration',
            interpretation='Removal of other-source supervised numerators at fixed Bridge exposure; not ordinary Bridge-only training or final task success',
            automatic_following_action=False,main_model_replaced=False)
        assert result['final_frozen_sha256']==initial_frozen
    finally:
        np.savez_compressed(args.output/'predictions_and_targets.npz',targets=targets,masks=masks,
            **{name:values for name,values in outputs.items()})
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BRIDGE_SUPERVISION_EVALUATION_REPORT',str(args.output/'report.json'),flush=True)


if __name__=='__main__':main()
