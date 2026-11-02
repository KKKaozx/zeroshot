"""Actual endpoint evaluator; cached original reference, no optimizer or new window selection."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
from binding import HERE,read,sha,admit,attach


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();binding=admit(root,space=False);pg=attach(root);pg.initialize_runtime()
    import torch
    import train as api
    import models
    from completed_policy_export import contract_sha,model_sha
    from step_training_runner import frozen_digest,tokenizer_identity
    from resume_constants import training_constant_constructor
    from pose_runtime import zero_port,digest
    from runtime_inputs import load
    from instrument_formal import prepare_source
    from verify_formal import verify_training,verify_evaluation
    from pair_metrics import summarize
    verify_training(a.output)
    payload,data,reference,stats=load(root,pg);config=payload['contract']['config'];base=root/'oxe_target_pair_short_v1'
    selection=read(base/'selection.json')
    assert selection['generation_steps']==config['model']['num_diffusion_steps']==100
    assert selection['generation_batch_size']==8 and selection['noise_seeds']==[1101,1102,1103]
    assert config['model']['diffusion_prediction_type']=='sample'
    tokenizer=api.CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    assert tokenizer_identity(tokenizer)==payload['contract']['tokenizer']
    events=[]
    with training_constant_constructor(models.RobotAdapterModel,contract=payload['contract'],profile_path=base/'profile_sample.json',
            learned_state=api.trainable_state_dict,on_restored=events.append):
        model=models.RobotAdapterModel(config).cuda().eval()
    keys=api.trainable_state_dict(model);initial_frozen=frozen_digest(model,keys)
    assert initial_frozen==reference['frozen_sha256']
    port=zero_port('cuda',width=1024).eval();code=prepare_source(pg)
    exec(compile(code,'<qualified-fixed-evaluation-inputs>','exec'),pg.__dict__)
    features=pg.prepare_inputs(root,data,model,port,tokenizer,api,stats,selection)
    (a.output/'effective_evaluation_inputs.py').write_bytes(code.encode())
    # Frozen encoder features are shared by both endpoints; only adapter/denoiser/port weights change.
    model.vision_encoder.cpu();model.text_encoder.cpu();del data,payload;gc.collect();torch.cuda.empty_cache()
    out=a.output/'evaluation';out.mkdir(exist_ok=False)
    outputs={};rng_streams={};result={'execution_complete':False,'optimizer_updates':0,'checkpoint_metadata':{},
        'formal_contract_sha256':sha(HERE/'formal_contract.json'),'selection_sha256':sha(base/'selection.json'),
        'preflight_report_sha256':binding['preflight_report_sha256'],'generation_steps':100,'sampling_seeds':selection['noise_seeds'],
        'frozen_restore_events':events,'automatic_following_action':False};tick=time.monotonic()
    try:
        for role in ['mask_control','real_pose']:
            p=torch.load(a.output/role/'latest_state.pt',map_location='cpu',weights_only=False,mmap=True)
            receipt=read(a.output/role/'arm_receipt.json');c=p['contract']
            assert c['base_config_unchanged']==config and c['role']==role
            assert p['contract_sha256']==contract_sha(c)==receipt['contract_sha256']
            assert c['preflight_report_sha256']==binding['preflight_report_sha256']
            state=p['base'];learned=model_sha(state);port_hash=digest(p['pose_port'])
            assert set(state)==set(keys) and learned==receipt['base_learned_sha256'] and port_hash==receipt['port_sha256']
            loaded=model.load_state_dict(state,strict=False)
            assert not loaded.unexpected_keys and not set(keys).intersection(loaded.missing_keys)
            port.load_state_dict(p['pose_port'],strict=True);model.eval();port.eval()
            draws=[];contexts=[];rng_streams[role]=[]
            with torch.inference_mode():
                for lo in range(0,120,8):
                    original=model.adapter(features['visual'][lo:lo+8],features['language'][lo:lo+8],features['attention'][lo:lo+8])
                    contexts.append(original+port(torch.from_numpy(features['features'][role][lo:lo+8]).cuda()))
                contexts=torch.cat(contexts)
                for seed in selection['noise_seeds']:
                    torch.manual_seed(seed);torch.cuda.manual_seed(seed);prediction=[]
                    for lo in range(0,120,8):prediction.append(model.sample(contexts[lo:lo+8],features['current'][lo:lo+8]).cpu().numpy())
                    draws.append(np.concatenate(prediction));rng_streams[role].append(digest(torch.cuda.get_rng_state_all()))
                    print('CURRENT_POSE_ENDPOINT_SAMPLE',role,seed,'120/120',flush=True)
            outputs[role]=np.stack(draws)
            assert model_sha(api.trainable_state_dict(model))==learned and digest(port.state_dict())==port_hash
            assert frozen_digest(model,keys)==initial_frozen
            result['checkpoint_metadata'][role]={'base_learned_sha256':learned,'port_sha256':port_hash,
                'checkpoint_contract_sha256':p['contract_sha256'],'completed_update':10240,'trained_native_target':'sample'}
            del p,state,contexts,draws
        assert rng_streams['mask_control']==rng_streams['real_pose']
        with np.load(root/'runs/oxe-target-short-eval-fixed-232091/predictions_and_targets.npz') as z:
            assert np.array_equal(features['targets'],z['targets']) and np.array_equal(features['masks'],z['masks'])
            outputs['unchanged231297']=z['sample'].copy();outputs['static']=z['static'].copy()
        result['metrics']=summarize(outputs,features['targets'],features['masks'],selection)
        result.update(execution_complete=True,final_frozen_sha256=initial_frozen,paired_sampling_rng_exact=True,
            sampling_rng_sha256=rng_streams,elapsed_seconds=time.monotonic()-tick)
    finally:
        np.savez_compressed(out/'predictions_and_targets.npz',targets=features['targets'],masks=features['masks'],**outputs)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('CURRENT_POSE_ENDPOINT_REPORT',out/'report.json',flush=True)
    verified=verify_evaluation(a.output)
    print('CURRENT_POSE_ENDPOINT_METRICS',json.dumps(verified,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
