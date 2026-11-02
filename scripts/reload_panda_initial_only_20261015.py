"""Verify the final diagnostic weights reproduce their saved training predictions."""
import numpy as np
import torch
import run_panda_spatial_fixed_trial_20261015 as trial

OUT=trial.ROOT/'reports/panda_initial_only_fit_20261015'


@torch.inference_mode()
def main():
    assert not (OUT/'reload_report.json').exists()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    report=trial.read(OUT/'report.json');assert report['complete'] and trial.sha(OUT/'final.pt')==report['checkpoint_sha256']
    model,tokenizer,original,_=trial.load_final()
    payload=torch.load(OUT/'final.pt',map_location='cpu',weights_only=False)
    assert payload['plan_sha256']==trial.sha(OUT/'plan.json') and payload['update']==1024
    missing=model.load_state_dict(payload['state_dict_nonclip'],strict=False)
    assert not missing.unexpected_keys and all(n.startswith(trial.GROUPS['clip']) for n in missing.missing_keys)
    assert {k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()}==report['final_group_sha256']
    images,target,current,rows=trial.inputs.examples('train');images=images[::8];target=target[::8];current=current[::8].cuda()
    text=tokenizer([trial.read(trial.DATA/'protocol.json')['instruction']]*16,padding=True,truncation=True,max_length=77,return_tensors='pt')
    context=torch.cat([model.get_context_vector(images[i:i+2].cuda(),text['input_ids'][i:i+2].cuda(),text['attention_mask'][i:i+2].cuda()) for i in range(0,16,2)])
    with np.load(OUT/'final_train_predictions.npz') as z:saved=z['predictions'];cached=z['contexts'];truth=z['targets']
    assert np.array_equal(target.numpy(),truth)
    delta=float(np.abs(context.cpu().numpy()-cached).max());assert delta<1e-4
    replay,_=trial.evaluate(model,torch.from_numpy(cached).cuda(),current,truth,original['sample_seeds'],batch=2)
    difference=float(np.abs(replay-saved).max());assert difference<1e-4
    trial.guard(original)
    result=dict(verified=True,checkpoint_sha256=report['checkpoint_sha256'],parameters_match=True,
        online_context_max_difference=delta,prediction_max_difference=difference,predictions_exact=bool(np.array_equal(replay,saved)),
        training_updates=0,training_windows=16,development_used=False,test_used=False,physical_rollout=False)
    trial.write(OUT/'reload_report.json',result);print(result,flush=True)


if __name__=='__main__':main()
