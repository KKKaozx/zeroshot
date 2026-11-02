"""Fresh-process reload of the diagnostic model and its frozen condition stats."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import run_panda_spatial_fixed_trial_20261015 as trial
from population_condition_transform import PopulationConditionDecoder

OUT=trial.ROOT/'reports/panda_joint_condition_transform_20261015'


@torch.inference_mode()
def main():
    assert not (OUT/'reload_report.json').exists()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    p,r=trial.read(OUT/'plan.json'),trial.read(OUT/'report.json');assert r['complete']
    assert trial.sha(OUT/'final.pt')==r['checkpoint_sha256']
    assert all(trial.sha(trial.ROOT/n)==h for n,h in {**p['source_sha256'],**p['production_source_sha256']}.items())
    payload=torch.load(OUT/'final.pt',map_location='cpu',weights_only=False)
    assert payload['condition_transform']=='train_population_eval_v1_unet_only' and payload['updates']==1024
    assert payload['plan_sha256']==trial.sha(OUT/'plan.json')
    trial.seed(43);model=trial.RobotAdapterModel(p['config'])
    model.diffusion_decoder=PopulationConditionDecoder(model.diffusion_decoder)
    missing=model.load_state_dict(payload['state_dict_nonclip'],strict=False)
    assert not missing.unexpected_keys and all(n.startswith(trial.GROUPS['clip']) for n in missing.missing_keys)
    model.cuda().eval()
    assert {k:trial.inputs.module_sha(model,prefix) for k,prefix in trial.GROUPS.items()}==r['final_group_sha256']
    with np.load(OUT/'frozen_tokens.npz') as z:
        visual=torch.from_numpy(z['visual']).cuda();language=torch.from_numpy(z['language']).cuda();mask=torch.from_numpy(z['attention_mask']).cuda()
    with np.load(OUT/'predictions.npz') as z:
        old=z['predictions'];raw=z['contexts'];target=z['targets'];norm=z['normalized_contexts'];mean=z['context_mean'];scale=z['context_scale']
    assert np.array_equal(model.diffusion_decoder.context_mean.cpu().numpy(),mean)
    assert np.array_equal(model.diffusion_decoder.context_scale.cpu().numpy(),scale)
    actual=torch.cat([model.adapter(visual[i:i+2],language[i:i+2],mask[i:i+2]) for i in range(0,16,2)])
    context_delta=float(np.abs(actual.cpu().numpy()-raw).max());assert context_delta<1e-4
    normalized_delta=float(np.abs(model.diffusion_decoder.transform(actual).cpu().numpy()-norm).max())
    current=[]
    for row in p['selection']:
        with np.load(trial.DATA/'episodes'/(row['episode']+'.npz')) as z:current.append([float(z['currents'][0])])
    current=torch.tensor(current,dtype=torch.float32,device='cuda')
    replay,_=trial.evaluate(model,torch.from_numpy(raw).cuda(),current,target,p['sampling_seeds'],batch=2)
    assert np.array_equal(replay,old)
    trial.seed(1101);actual_first=model.sample(actual[:2],current[:2]).cpu().numpy()
    adapter_path_delta=float(np.abs(actual_first-old[0,:2]).max())
    assert np.array_equal(model.diffusion_decoder.context_mean.cpu().numpy(),mean)
    assert np.array_equal(model.diffusion_decoder.context_scale.cpu().numpy(),scale)
    # Verify actual U-Net parameter changes independently of newly added buffers.
    h=hashlib.sha256()
    for n,v in sorted(model.diffusion_decoder.core.state_dict().items()):
        h.update(('diffusion_decoder.'+n).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    assert h.hexdigest()!=r['initial_group_sha256']['decoder']
    result=dict(verified=True,checkpoint_sha256=r['checkpoint_sha256'],all_parameter_groups_match=True,
        actual_unet_core_changed=True,full16_windows_three_draws_exact=True,
        reloaded_adapter_raw_context_max_difference=context_delta,normalized_context_max_difference=normalized_delta,
        actual_adapter_first_batch_prediction_max_difference=adapter_path_delta,
        saved_statistics_unchanged_by_inference=True,statistics_recalibrated_at_reload=False,
        token_cache_sha256=trial.sha(OUT/'frozen_tokens.npz'),training_updates=0,development_used=False,test_used=False,physical_rollout=False)
    trial.write(OUT/'reload_report.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
