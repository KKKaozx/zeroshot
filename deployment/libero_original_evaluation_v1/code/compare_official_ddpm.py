"""Consume real production tensors using the installed diffusers scheduler."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import diffusers
from diffusers import DDPMScheduler


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def injected_step(scheduler,x0,step,latent,noise):
    # Only replace the random draw; scheduler.step code remains unmodified.
    def draw(shape, **kwargs):
        if tuple(shape)!=tuple(noise.shape):
            raise ValueError('Unexpected official noise shape')
        return noise.clone()
    with patch.object(torch,'randn',side_effect=draw) as mocked:
        output=scheduler.step(x0,step,latent).prev_sample
    if mocked.call_count != int(step>0):
        raise ValueError('Official scheduler noise contract changed')
    return output


def finish_pose(value,limit):
    value=value.clone();value[...,:3].clamp_(-limit,limit)
    q=value[...,3:7];identity=torch.zeros_like(q);identity[...,3]=1
    value[...,3:7]=torch.where(q.norm(dim=-1,keepdim=True)>1e-6,
                              torch.nn.functional.normalize(q,dim=-1),identity)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if diffusers.__version__!='0.11.1': raise ValueError('Expected existing official environment diffusers 0.11.1')
    report=json.loads(args.export.read_text())
    if not report['execution_checks_passed'] or not report['capture_replay_exact']:
        raise ValueError('Production capture is not eligible')
    path=Path(report['tensors_path'])
    if sha256(path)!=report['tensors_sha256']: raise ValueError('Export tensors changed')
    with np.load(path) as archive: data={k:torch.from_numpy(archive[k].copy()) for k in archive.files}
    # Preserve Bridge x0 target and per-axis bounds; copying Push-T epsilon/[-1,1] would change the protocol.
    scheduler=DDPMScheduler(num_train_timesteps=100,trained_betas=data['betas'].numpy(),
                            variance_type='fixed_small',prediction_type='sample',clip_sample=False)
    scheduler.set_timesteps(100)
    cosine=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2')
    results=[];limit=report['position_limit']
    with torch.no_grad():
        for step in reversed(range(100)):
            latent=data['inputs'][step];raw=data['estimates'][step]
            clean=torch.cat((raw[...,:3].clamp(-limit,limit),raw[...,3:].clamp(-1,1)),dim=-1)
            noise=data['noises'][step]
            reference=injected_step(scheduler,clean,step,latent,noise)
            mean=injected_step(scheduler,clean,step,latent,torch.zeros_like(noise))
            # Reconstruct the project's mean using the exact exported buffers and its x0->epsilon->x0 roundtrip.
            alpha=data['alpha_bars'][step]
            epsilon=(latent-alpha.sqrt()*raw)/(1-alpha).sqrt()
            native_clean=(latent-(1-alpha).sqrt()*epsilon)/alpha.sqrt()
            native_clean=torch.cat((native_clean[...,:3].clamp(-limit,limit),native_clean[...,3:].clamp(-1,1)),dim=-1)
            project_mean=data['coeff_x0'][step]*native_clean+data['coeff_xt'][step]*latent
            if step:
                observed=data['inputs'][step-1]
                compared=reference
            else:
                observed=data['final_poses'];compared=finish_pose(reference,limit)
            difference=(compared-observed).abs()
            results.append({'step':step,'update_max_absolute_difference':float(difference.max()),
                            'update_mean_absolute_difference':float(difference.mean()),
                            'update_xyz_max_difference_cm':float(difference[...,:3].max())*10,
                            'update_allclose_atol_rtol_1e_5':bool(torch.allclose(compared,observed,atol=1e-5,rtol=1e-5)),
                            'mean_max_absolute_difference':float((mean-project_mean).abs().max()),
                            'variance_absolute_difference':float((scheduler._get_variance(step)-data['variance'][step]).abs()),
                            'x0_roundtrip_max_difference':float((native_clean-clean).abs().max())})
    first=next((r['step'] for r in results if not r['update_allclose_atol_rtol_1e_5']),None)
    result={'stage':'real_tensor_official_ddpm_step_comparison','execution_checks_passed':True,
            'trained':False,'weights_updated':False,'export_sha256':sha256(args.export),
            'diffusers_version':diffusers.__version__,'torch_version':torch.__version__,'reference_device':'cpu',
            'scheduler_source_sha256':sha256(inspect.getfile(DDPMScheduler)),
            'reference_source':'https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/schedulers/scheduling_ddpm.py',
            'shared_betas_alpha_bar_max_difference':float((scheduler.alphas_cumprod-data['alpha_bars']).abs().max()),
            'independent_cosine_beta_max_difference':float((cosine.betas-data['betas']).abs().max()),
            'all_updates_within_tolerance':first is None,'first_step_outside_tolerance':first,
            'max_update_absolute_difference':max(r['update_max_absolute_difference'] for r in results),
            'max_update_xyz_difference_cm':max(r['update_xyz_max_difference_cm'] for r in results),
            'steps':results,'window_selection':report['window_selection'],
            'limits':['One seed, forty training windows; conditional update audit, not a new policy evaluation.',
                      'Official step uses recorded noise and unchanged Bridge x0 parameterization/bounds.',
                      'Per-step inputs reset to production tensors; this is not a full free-running official rollout.',
                      'CPU reference versus CUDA production can differ in floating-point rounding.',
                      'Tolerance is numeric comparison criteria, not a task success threshold.',
                      'Step zero compares final pose after the existing quaternion normalization.']}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('OFFICIAL_DDPM_COMPARISON',json.dumps({k:result[k] for k in ['all_updates_within_tolerance','first_step_outside_tolerance','max_update_absolute_difference','max_update_xyz_difference_cm','shared_betas_alpha_bar_max_difference','independent_cosine_beta_max_difference']}),flush=True)
    print('REPORT',args.output,flush=True)


if __name__=='__main__': main()
