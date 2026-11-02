"""Prepare one labelled in-domain control; no training/encoding/simulation."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deployment/libero_native_supervised_v1'
sys.path.insert(0,str(BUNDLE))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    import h5py
    import numpy as np
    from transformers import CLIPTokenizer
    from safetensors import safe_open
    from native_data import TASK,INSTRUCTION,causal_chunks,fit_normalizer,proprio_features,normalize_actions
    started=time.monotonic();out=ROOT/'reports/libero_native_supervised_preparation_20261031'
    if (out/'receipt.json').exists():raise FileExistsError('Preparation already bound; do not overwrite')
    roster_path=ROOT/'deployment/libero_original_evaluation_v1/task_roster.json'
    roster=json.loads(roster_path.read_text(encoding='utf-8'))
    if any(r['task']==TASK or r['official_language']==INSTRUCTION for r in roster):
        raise ValueError('Task must be excluded from the formal40 roster')
    path=ROOT.parent/'dataset/LIBERO/libero_90'/(TASK+'_demo.hdf5')
    clip=ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
    rows=[];parts=[];train_proprio=[];train_actions=[];bindings=[]
    with h5py.File(path,'r') as f:
        group=f['data'];meta=json.loads(group.attrs['env_args'])
        if group.attrs['macros_image_convention']!='opengl' or meta['env_kwargs']['control_freq']!=20:
            raise ValueError('Camera/timing metadata differs')
        if json.loads(group.attrs['problem_info'])['language_instruction']!=INSTRUCTION:raise ValueError('Task language differs')
        for demo_id in range(25):
            demo=group[f'demo_{demo_id}'];actions=demo['actions'][:].astype(np.float32)
            targets,mask=causal_chunks(actions);count=len(targets)
            rgb=demo['obs/agentview_rgb'][:count]
            proprio=np.concatenate((demo['obs/joint_states'][:count],demo['obs/gripper_states'][:count]),axis=1).astype(np.float32)
            if rgb.shape!=(count,128,128,3) or not np.isfinite(proprio).all():raise ValueError('Observation shape/value differs')
            role=0 if demo_id<20 else 1
            if role==0:train_proprio.append(proprio);train_actions.append(actions[1:])
            previous=actions[:count]
            parts.append({'images':rgb,'raw_proprio':proprio,'raw_targets':targets,'mask':mask,'previous_action':previous,
                'split':np.full(count,role,np.uint8),'demo_id':np.full(count,demo_id,np.int16),
                'obs_index':np.arange(count,dtype=np.int16)})
            bindings.append({'demo':demo_id,'role':'train' if role==0 else 'development','observations':count,
                'raw_actions_sha256':hashlib.sha256(demo['actions'][:].tobytes()).hexdigest(),
                'raw_images_used_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                'raw_proprio_used_sha256':hashlib.sha256(proprio.tobytes()).hexdigest()})
    normalizer=fit_normalizer(np.concatenate(train_proprio),np.concatenate(train_actions))
    arrays={key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    arrays['proprio']=proprio_features(arrays.pop('raw_proprio'),normalizer)
    arrays['targets']=normalize_actions(arrays['raw_targets'],normalizer)
    arrays['targets']*=arrays['mask']
    tokens=CLIPTokenizer.from_pretrained(clip,local_files_only=True)([INSTRUCTION],padding=True,
        truncation=True,max_length=77,return_tensors='np')
    arrays['input_ids']=tokens['input_ids'];arrays['attention_mask']=tokens['attention_mask']
    seed=np.random.default_rng(42);train_ids=np.flatnonzero(arrays['split']==0)
    plan=seed.choice(train_ids,size=(4096,16),replace=True)
    np.savez_compressed(BUNDLE/'inputs.npz',**arrays);np.savez_compressed(BUNDLE/'plan.npz',batches=plan)
    (BUNDLE/'normalizer.json').write_text(json.dumps(normalizer,indent=2)+'\n',encoding='utf-8')
    # Bind text parameters independent of .bin versus .safetensors container.
    text_hash=hashlib.sha256();keys=[]
    with safe_open(clip/'model.safetensors',framework='pt',device='cpu') as weights:
        for name in sorted(k for k in weights.keys() if k.startswith('text_model.')):
            if name.endswith('position_ids'):continue
            value=weights.get_tensor(name);keys.append(name)
            text_hash.update(name.encode());text_hash.update(value.contiguous().numpy().tobytes())
    config=json.loads((clip/'config.json').read_text(encoding='utf-8'))
    (BUNDLE/'text_config.json').write_text(json.dumps(config['text_config'],indent=2)+'\n',encoding='utf-8')
    receipt={'format':'libero90-native-supervised-inputs-v1','task':TASK,'instruction':INSTRUCTION,
        'target_supervised':True,'zero_shot':False,'formal_matrix_admitted':False,'source_hdf5':str(path),
        'source_hdf5_sha256':sha(path),'formal_roster_sha256':sha(roster_path),'excluded_from_formal40':True,
        'train_demos':list(range(20)),'development_demos':list(range(20,25)),'rollout_demos':list(range(25,30)),
        'rollout_demonstration_payloads_decoded_during_preparation':False,
        'observation_action_alignment':'Storedobs[i] post action[i]; target starts recorded action[i+1]',
        'generation_source_file_sha256':sha(ROOT/'training_cache/libero_author_8f1084e/scripts/create_dataset.py'),
        'train_frames':int((arrays['split']==0).sum()),'development_frames':int((arrays['split']==1).sum()),
        'inputs_sha256':sha(BUNDLE/'inputs.npz'),'inputs_bytes':(BUNDLE/'inputs.npz').stat().st_size,
        'plan_sha256':sha(BUNDLE/'plan.npz'),'normalizer_sha256':sha(BUNDLE/'normalizer.json'),
        'text_parameter_names':keys,'text_parameters_sha256':text_hash.hexdigest(),'row_bindings':bindings,
        'model_loaded':False,'new_encoder_forwards':0,'training_updates':0,'simulation_steps':0,
        'elapsed_seconds':time.monotonic()-started}
    (BUNDLE/'inputs_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('row_bindings','text_parameter_names')},indent=2))


if __name__=='__main__':main()
