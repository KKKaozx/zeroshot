"""One bounded train-only initial-approach capacity check; not a grasp policy."""
import json
from pathlib import Path
import time
import numpy as np
import run_panda_spatial_fixed_trial_20261015 as trial
import torch

OUT=trial.ROOT/'reports/panda_initial_only_fit_20261015'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    original=trial.read(trial.OUT/'plan.json');old=trial.read(trial.OUT/'training_report.json');trial.guard(original)
    images,targets,current,rows=trial.inputs.examples('train');selected=list(range(0,128,8))
    assert all(rows[i]['start']==0 for i in selected) and len({rows[i]['episode'] for i in selected})==16
    original_batches=torch.cat([torch.randperm(128,generator=torch.Generator().manual_seed(440001+c)).reshape(16,8) for c in range(512)])
    flattened=original_batches.flatten();batches=(flattened[flattened%8==0]//8).reshape(1024,8)
    assert torch.equal(torch.bincount(batches.flatten()),torch.full((16,),512))
    assert torch.equal(torch.bincount(batches[:512].flatten(),minlength=16),torch.full((16,),256))
    plan=dict(source_sha256=original['source_sha256'],entry_sha256=trial.sha(Path(__file__)),initialization=43,
        selection=[rows[i] for i in selected],updates=1024,draws_per_window=512,batch_size=8,
        learning_rate='First512 at1e-4, last512 at1e-5; same exposure per initial window and rate phase as original',
        scope='Initial-only capacity diagnostic. Removing other stages also changes gradient mixtures and Adam update count; not a single-cause proof.',
        fresh=True,starting_checkpoint=False,training_only=True,development_used=False,test_used=False,
        inference='Unchanged nativeDDPM100, original3 seeds, final fixed update; no model selection',
        limits='No extra updates, initialization search, new data or physical rollout. Initial windows have no closing actions.',
        checkpoint='One final nonCLIP only, <1GiB; no optimizer or intermediate weights',config=original['config'])
    trial.write(OUT/'plan.json',plan);(OUT/'entry_snapshot.py').write_bytes(Path(__file__).read_bytes())
    trial.seed(43);model=trial.RobotAdapterModel(original['config']).cuda().eval()
    before={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()}
    assert before==old['initial_group_sha256'],'Fresh model must match original initialization'
    tokenizer=trial.CLIPTokenizer.from_pretrained(original['config']['model']['name'],local_files_only=True)
    text=tokenizer([trial.read(trial.DATA/'protocol.json')['instruction']]*16,padding=True,truncation=True,max_length=77,return_tensors='pt')
    text={k:v.cuda() for k,v in text.items()}
    images=images[selected];targets=targets[selected].cuda();current=current[selected].cuda()
    assert bool((targets[...,7]==1).all())
    with torch.no_grad():
        visual=torch.cat([model.vision_encoder(pixel_values=images[i:i+2].cuda()).last_hidden_state for i in range(0,16,2)]).detach()
        language=model.text_encoder(**text).last_hidden_state.detach()
    model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    trainable=[p for n,p in model.named_parameters() if not n.startswith(trial.GROUPS['clip'])]
    optimizer=torch.optim.Adam(trainable,lr=1e-4);started=time.monotonic()
    with (OUT/'updates.jsonl').open('w') as log:
        for step,ix in enumerate(batches,1):
            lr=1e-4 if step<=512 else 1e-5
            for group in optimizer.param_groups:group['lr']=lr
            model.train();ix=ix.cuda();trial.seed(430000+step)
            ctx=model.adapter(visual[ix],language[ix],text['attention_mask'][ix])
            trial.seed(420000+step);output=model.diffusion_loss(targets[ix],ctx,current[ix])
            loss,pose,grip=trial.policy_loss(model,output,targets[ix],torch.nn.MSELoss(),current_grippers=current[ix])
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
            log.write(json.dumps(dict(update=step,loss=float(loss.detach()),pose=float(pose.detach()),gripper=float(grip.detach()),lr=lr))+'\n')
            if step==1 or step%128==0:
                log.flush();print('INITIAL_ONLY_UPDATE',step,'/1024',float(loss.detach()),'seconds',round(time.monotonic()-started,1),flush=True)
    model.eval()
    with torch.no_grad():ctx=torch.cat([model.adapter(visual[i:i+2],language[i:i+2],text['attention_mask'][i:i+2]) for i in range(0,16,2)])
    prediction,metrics=trial.evaluate(model,ctx,current,targets.cpu().numpy(),original['sample_seeds'],batch=2)
    np.savez_compressed(OUT/'final_train_predictions.npz',predictions=prediction,targets=targets.cpu().numpy(),contexts=ctx.cpu().numpy())
    after={k:trial.inputs.module_sha(model,p) for k,p in trial.GROUPS.items()};assert before['clip']==after['clip']
    assert all(before[k]!=after[k] for k in ['adapter','decoder','gripper'])
    checkpoint=OUT/'final.pt'
    torch.save(dict(config=original['config'],state_dict_nonclip={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(trial.GROUPS['clip'])},
        final_group_sha256=after,initialization=43,update=1024,plan_sha256=trial.sha(OUT/'plan.json')),checkpoint)
    assert checkpoint.stat().st_size<1024**3;trial.guard(original)
    report=dict(complete=True,updates=1024,initialization=43,initial_group_sha256=before,final_group_sha256=after,
        original_initialization_exact=True,clip_unchanged=True,source_unchanged=True,checkpoint_sha256=trial.sha(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,draws_per_window=torch.bincount(batches.flatten()).tolist(),
        metrics=metrics,gripper=trial.gripper_metrics(prediction,targets.cpu().numpy()),
        sampling_seeds=original['sample_seeds'],development_used=False,test_used=False,physical_rollout=False,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.monotonic()-started)
    trial.write(OUT/'report.json',report);print('INITIAL ONLY FINAL',json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
