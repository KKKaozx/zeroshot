"""Read-only diagnosis of downloaded fixed-window learned diffusion weights."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'results/bridge_diffusion_fit_prepare_v1'
RESULT = Path(r'D:\ntu_related\dissertation\GPU cluster\bridge-diffusion-fit-185979')
sys.path.insert(0, str(PACKAGE))
from cached_diffusion import RobotAdapterModel, metrics


def save(name, value):
    (RESULT/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    source_hash = hashlib.sha256((RESULT/'final_head.pt').read_bytes()).hexdigest()
    bundle = torch.load(PACKAGE/'fixed_context.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(RESULT/'final_head.pt', map_location='cpu', weights_only=False)
    experiment = json.loads((RESULT/'experiment.json').read_text(encoding='utf-8'))
    final = json.loads((RESULT/'final_metrics.json').read_text(encoding='utf-8'))
    assert checkpoint['steps'] == 2000 and checkpoint['indices'] == bundle['indices'] == experiment['indices']
    assert checkpoint['config'] == bundle['config']
    for k, v in bundle['gripper_state_dict'].items():
        torch.testing.assert_close(checkpoint['state_dict'][k], v, atol=0, rtol=0)
    with np.load(RESULT/'final_predictions.npz', allow_pickle=False) as a:
        predictions, target, indices, seeds = [a[k].copy() for k in ['predictions','targets','indices','seeds']]
    np.testing.assert_array_equal(target, bundle['targets'].numpy().astype(np.float64))
    np.testing.assert_array_equal(indices, bundle['indices'])
    np.testing.assert_array_equal(seeds, experiment['sample_seeds'])
    assert predictions.shape == (3,14,16,8)
    rows, scores = [], []
    for p, seed_value, expected in zip(predictions, seeds, final['samples']):
        m = metrics(p, target)
        for k,v in m.items():
            if v is not None:
                np.testing.assert_allclose(v,expected['metrics'][k],atol=1e-7,rtol=1e-6)
        scores.append(dict(seed=int(seed_value),metrics=m))
        for j,index in enumerate(indices):
            for t in range(16):
                one = metrics(p[j:j+1,t:t+1],target[j:j+1,t:t+1])
                rows.append(dict(seed=int(seed_value),index=int(index),waypoint=t+1,
                    position_cm=one['position_error_cm'],rotation_deg=one['rotation_error_deg'],
                    dx_error_cm=float((p[j,t,0]-target[j,t,0])*10),
                    dy_error_cm=float((p[j,t,1]-target[j,t,1])*10),
                    dz_error_cm=float((p[j,t,2]-target[j,t,2])*10)))
    with (RESULT/'per_window_waypoint_errors.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    position=np.linalg.norm(predictions[:,:,:,:3]-target[None,:,:,:3],axis=-1)*10
    mean_prediction=predictions[:,:,:,:3].mean(0)
    bias=np.linalg.norm(mean_prediction-target[:,:,:3],axis=-1)*10
    seed_spread=np.sqrt(np.mean(np.sum((predictions[:,:,:,:3]-mean_prediction)**2,axis=-1)))*10
    per_window=[dict(index=int(index),position_cm=float(position[:,j].mean()),
        max_position_cm=float(position[:,j].max()),window=bundle['windows'][j]) for j,index in enumerate(indices)]
    summary=dict(verified=True,scores=scores,gripper_frozen_weights_exact_match=True,
        position_by_waypoint_cm=position.mean((0,1)).tolist(),
        xyz_mae_cm=(np.abs(predictions[:,:,:,:3]-target[None,:,:,:3]).mean((0,1,2))*10).tolist(),
        max_position_cm=float(position.max()),p95_position_cm=float(np.quantile(position,.95)),
        per_window=sorted(per_window,key=lambda r:r['position_cm'],reverse=True),
        three_sample_mean_position_cm=float(bias.mean()),three_sample_spread_rms_cm=float(seed_spread),
        note='Three-sample mean/spread are diagnostic only; the mean is not an accepted policy or a fourth evaluated sample.')
    save('prediction_analysis.json',summary)

    device='cuda'
    model=RobotAdapterModel(bundle['config']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    c,g,actions=[bundle[k].to(device) for k in ['context','current','targets']]
    x0=actions[:,:,:7]
    def measure_pose(pose):
        p=torch.cat([pose,actions[:,:,7:8]],dim=-1).cpu().numpy().astype(np.float64)
        m=metrics(p,target)
        return dict(position_cm=m['position_error_cm'],rotation_deg=m['rotation_error_deg'])
    one_step=[]
    for t in [0,1,5,10,25,50,75,90,99]:
        trials=[]
        for value in [2201,2202,2203,2204]:
            torch.manual_seed(value)
            noise=torch.randn_like(x0);a=model.alpha_bars[t]
            xt=a.sqrt()*x0+(1-a).sqrt()*noise
            ts=torch.full((14,),t,device=device,dtype=torch.long)
            predicted=model.diffusion_decoder(xt,ts,c)
            clean=(xt-(1-a).sqrt()*predicted)/a.sqrt()
            trials.append(dict(noise_xyz_mse=float((predicted[:,:,:3]-noise[:,:,:3]).square().mean()),
                noise_quaternion_mse=float((predicted[:,:,3:]-noise[:,:,3:]).square().mean()),
                **measure_pose(clean)))
        one_step.append(dict(timestep=t,alpha_bar=float(model.alpha_bars[t]),
            averages={k:float(np.mean([r[k] for r in trials])) for k in trials[0]}))
    comparisons=[]
    for value,recorded in zip(seeds,predictions):
        torch.manual_seed(int(value));actual=model.sample(c,g).cpu().numpy().astype(np.float64)
        comparisons.append(dict(seed=int(value),local_score=metrics(actual,target),
            max_difference_to_cluster=float(np.abs(actual-recorded).max())))

    def reverse(value,mode):
        torch.manual_seed(int(value))
        x=torch.randn_like(x0);path=[]
        for step in reversed(range(100)):
            ts=torch.full((14,),step,device=device,dtype=torch.long)
            conditional=c if mode != 'shuffled_context' else torch.roll(c,1,0)
            eps=model.diffusion_decoder(x,ts,conditional)
            a=model.alpha_bars[step]
            clean=(x-(1-a).sqrt()*eps)/a.sqrt()
            clipped=torch.cat([clean[:,:,:3].clamp(-3,3),clean[:,:,3:].clamp(-1,1)],dim=-1)
            mean=model.posterior_mean_x0[step]*clipped+model.posterior_mean_xt[step]*x
            if step in [99,90,75,50,25,10,5,1,0]:
                path.append(dict(step=step,xyz_clip_fraction=float((clean[:,:,:3].abs()>3).float().mean()),
                    raw_clean=measure_pose(clean),clipped_clean=measure_pose(clipped)))
            x=mean+model.posterior_variance[step].sqrt()*torch.randn_like(x) if step and mode!='zero_posterior_noise' else mean
        pose=x.clone();pose[:,:,:3].clamp_(-3,3)
        pose[:,:,3:7]=torch.nn.functional.normalize(pose[:,:,3:7],dim=-1)
        return dict(seed=int(value),mode=mode,final=measure_pose(pose),path=path)
    paths=[reverse(value,mode) for mode in ['original','zero_posterior_noise','shuffled_context'] for value in seeds]
    original=np.mean([r['final']['position_cm'] for r in paths if r['mode']=='original'])
    deterministic=np.mean([r['final']['position_cm'] for r in paths if r['mode']=='zero_posterior_noise'])
    shuffled=np.mean([r['final']['position_cm'] for r in paths if r['mode']=='shuffled_context'])
    report=dict(read_only=True,optimizer_updates=0,one_step_with_known_noisy_targets=one_step,
        local_sample_reproduction=comparisons,reverse_paths=paths,
        mean_position_by_diagnostic_mode_cm=dict(original=float(original),zero_posterior_noise=float(deterministic),
                                                shuffled_context=float(shuffled)),
        original_checkpoint_sha256=source_hash,
        limits=['Forward noisy targets use truth solely for denoiser diagnosis, not deployment.',
                'Changing posterior noise or shuffling context is a diagnostic intervention, not a proposed trained policy.',
                'Local Torch/GPU differ from cluster; downloaded predictions remain the authoritative run results.',
                'No fresh training or held-out targets accessed; this cannot establish the unique cause of failed fitting.'])
    assert hashlib.sha256((RESULT/'final_head.pt').read_bytes()).hexdigest()==source_hash
    save('denoiser_analysis.json',report)
    print(json.dumps(dict(prediction_summary={k:v for k,v in summary.items() if k not in ['scores','per_window']},
        worst_windows=summary['per_window'][:3],one_step=one_step,
        reverse_mode_average=report['mean_position_by_diagnostic_mode_cm']),indent=2))


if __name__=='__main__':
    main()
