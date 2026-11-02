"""One within-source fused-context permutation on the fixed32 validation windows.

Reuse correct-context trajectories; replay only window0 as an identity guard.
Probe the same context intervention on fixed noisy expert actions at five t.
No optimizer, new labels, simulator, checkpoint/seed selection, or extra donors.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'diagnostics')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',TF_CPP_MIN_LOG_LEVEL='3',
                  OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
OUT=ROOT/'reports/oxe_conditioning_sampling_audit_20261027/attempt1'
BASE=ROOT/'reports/oxe_source_validation_decode_20261027/attempt1'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mem,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','))
    if mem>1024 or util>20:raise RuntimeError('Existing GPU workload; stop before model loading')
    OUT.mkdir(parents=True,exist_ok=False)
    selection=read(BASE/'selection_protocol.json');previous=read(BASE/'report.json')
    assert previous['execution_complete'] and previous['model_windows_decoded']==32
    assert read(BASE/'independent_metric_verification.json')['passed']
    selected=selection['selected']
    donors=[(i//8)*8+(i+1)%8 for i in range(32)]
    assert all(d!=i and selected[d]['source']==selected[i]['source'] for i,d in enumerate(donors))
    protocol=dict(scope='Single fused-context permutation mechanism diagnosis; not new training or a policy improvement experiment',
        reference_report_sha256=sha(BASE/'report.json'),reference_arrays_sha256=sha(BASE/'predictions_and_targets.npz'),
        sample_selection_unchanged=True,donors=donors,donor_rule='Cycle next item within each fixed8window source block, chosen before predictions; no replacement search',
        intervention='Change only fused image-language context; retain target window current-command features and all random draws',
        correct_context_outputs='Reuse saved32 predictions; exactly one window0 replay guard, atol1e-5, rtol0',
        sampling='Original100step DDPM and each original window sampling seed',
        noise_probe_steps=[0,25,50,75,99],noise_probe_seed='4101+100*window_index+t',
        noise_probe='Same noisy expert pose and mask with original versus permuted context; epsilon MSE and unbounded one-step x0 MSE, not complete trajectory performance',
        noise_static_prior='Context-free epsilon implied by static clean pose [zero position,identity quaternion], evaluated on the same noisy expert input',
        statistical_scope='Eight fixed groups per source, one full sample per condition; no training-seed uncertainty or task-success verdict',
        gpu_inference_wall_cap_seconds=600,training_updates=0,simulator_steps=0,script_sha256=sha(__file__))
    write(OUT/'protocol.json',protocol)
    (OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    report=dict(execution_complete=False,errors=[],training_updates=0,simulator_steps=0,
                original_correct_samples_reused=32,identity_replay_samples=0,permuted_samples=0,
                primary_final_tasks_accessed=0,scope=protocol['scope'])
    rows=[];swapped=[];started=time.monotonic()
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
        import numpy as np
        import torch
        from oxe_nominal_dataset import NominalOXEDataset
        from libero_project_policy import load_project_policy
        from libero_waypoint_execution import WaypointExecutionContract
        from train import tokenise,trainable_state_dict
        from completed_policy_export import model_sha
        from step_training_runner import frozen_digest
        from run_oxe_source_validation_decode_20261027 import window_metrics,aggregate
        saved=np.load(BASE/'predictions_and_targets.npz')
        data=NominalOXEDataset(ROOT,ROOT/'reports/oxe_nominal_validation_20261025',BASE/'source_locations.json',partition='validation')
        assert data.nominal_target_binding==selection['target_binding']
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        torch.manual_seed(1101)
        k=read(ROOT/'configs/libero_nominal_execution_candidate_20261025.json')['constructor_kwargs']
        k['target_times_seconds']=tuple(k['target_times_seconds']);k['output_scales']=tuple(k['output_scales'])
        policy=load_project_policy(ROOT/selection['checkpoint'],
            ROOT.parent/'hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41',
            WaypointExecutionContract(**k),device='cuda',sampling_seed=1101,
            local_dino_path=ROOT/'baselines/dinov2_large_frozen_assets_20261023',
            frozen_profile_path=ROOT/'reports/oxe_noaug_seed42_227223/cluster/preflight.json')
        model=policy.model
        assert model.diffusion_prediction_type=='epsilon' and model.num_diffusion_steps==100
        assert model.beta_schedule=='squaredcos_cap_v2' and model.diffusion_architecture=='multiscale'
        assert not model.training
        initial_learned=model_sha(trainable_state_dict(model));initial_frozen=frozen_digest(model,trainable_state_dict(model))
        assert initial_learned==previous['initial_learned_sha256'] and initial_frozen==previous['initial_frozen_sha256']
        contexts=[];instructions=[]
        inference_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for j,row in enumerate(selected):
            text,pixels,features,target,mask=data[row['window_index']]
            assert np.array_equal(features.numpy(),saved['current_features'][j])
            assert np.array_equal(target.numpy(),saved['targets'][j]) and np.array_equal(mask.numpy(),saved['masks'][j])
            tokens=tokenise(policy.tokenizer,[text],torch.device('cuda'))
            with torch.inference_mode():
                context=model.get_context_vector(pixels[None].cuda(),tokens['input_ids'],attention_mask=tokens.get('attention_mask'))
            contexts.append(context);instructions.append(text)
        np.save(OUT/'contexts.npy',torch.cat(contexts).cpu().numpy(),allow_pickle=False)
        def sample(context,j):
            with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                torch.random.default_generator.manual_seed(selected[j]['sampling_seed'])
                torch.cuda.manual_seed(selected[j]['sampling_seed'])
                return model.sample(context,current_gripper=torch.from_numpy(saved['current_features'][j:j+1]).cuda())[0].cpu().numpy()
        replay=sample(contexts[0],0)
        delta=float(np.max(np.abs(replay-saved['predictions'][0])))
        report['identity_replay_samples']=1
        report['replay_max_absolute_difference']=delta
        if delta>1e-5:raise ValueError('Original context/noise replay guard failed; do not score the intervention')
        for j,row in enumerate(selected):
            if time.monotonic()-inference_started>600:raise RuntimeError('Fixed diagnostic budget reached; retain partial outputs')
            donor=donors[j];pred=sample(contexts[donor],j);swapped.append(pred)
            metric=window_metrics(pred,saved['targets'][j],saved['masks'][j],saved['current_features'][j])
            m=torch.from_numpy(saved['masks'][j:j+1,:,:7]).cuda()
            clean=torch.from_numpy(saved['targets'][j:j+1,:,:7]).cuda()
            static=torch.zeros_like(clean);static[...,6]=1.
            probes=[]
            for t in protocol['noise_probe_steps']:
                with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                    seed=4101+100*j+t
                    torch.random.default_generator.manual_seed(seed);torch.cuda.manual_seed(seed)
                    eps=torch.randn_like(clean);a=model.alpha_bars[t]
                    noisy=a.sqrt()*clean+(1-a).sqrt()*eps
                    step=torch.tensor([t],device='cuda',dtype=torch.long)
                    ec=model.diffusion_decoder(noisy,step,contexts[j]);ew=model.diffusion_decoder(noisy,step,contexts[donor])
                    es=(noisy-a.sqrt()*static)/(1-a).sqrt()
                    mse=lambda x,y:float(((x-y).square()*m).sum().item()/m.sum().item())
                    probes.append(dict(t=t,alpha_bar=float(a),epsilon_correct_mse=mse(ec,eps),
                                       epsilon_permuted_mse=mse(ew,eps),epsilon_static_clean_prior_mse=mse(es,eps),
                                       x0_correct_unclipped_mse=mse((noisy-(1-a).sqrt()*ec)/a.sqrt(),clean),
                                       x0_permuted_unclipped_mse=mse((noisy-(1-a).sqrt()*ew)/a.sqrt(),clean),
                                       epsilon_output_context_change_rms=mse(ec,ew)**.5))
            valid=saved['masks'][j,:,0]>.5;width=2 if row['source']=='Language Table' else 3
            shift=float(np.linalg.norm(pred[valid,:width]-saved['predictions'][j,valid,:width],axis=1).mean())
            c0=contexts[j].flatten();c1=contexts[donor].flatten()
            cosine=float(torch.nn.functional.cosine_similarity(c0[None],c1[None]).item())
            result=dict(row,donor_index=donor,donor_window_index=selected[donor]['window_index'],
                        donor_instruction_identical=instructions[j]==instructions[donor],
                        fused_context_cosine=cosine,position_output_change_normalized=shift,
                        metrics=metric,noise_probes=probes)
            rows.append(result);report['permuted_samples']=len(rows)
            write(OUT/'window_results.json',rows)
            print('CONDITIONING_AUDIT',j+1,'/32',row['source'],flush=True)
        torch.cuda.synchronize()
        summary={}
        baseline_rows=read(BASE/'window_metrics.json')
        for source in ('Bridge','Language Table','BC-Z','Fractal'):
            ids=[i for i,r in enumerate(selected) if r['source']==source]
            old=[baseline_rows[i]['metrics']['valid_path']['model_position_normalized'] for i in ids]
            new=[rows[i]['metrics']['valid_path']['model_position_normalized'] for i in ids]
            s=dict(windows=len(ids),correct_position_error=float(np.mean(old)),permuted_position_error=float(np.mean(new)),
                   correct_context_better_windows=sum(x<y for x,y in zip(old,new)),
                   mean_output_position_change_normalized=float(np.mean([rows[i]['position_output_change_normalized'] for i in ids])),
                   mean_context_cosine=float(np.mean([rows[i]['fused_context_cosine'] for i in ids])),
                   identical_donor_instruction_windows=sum(rows[i]['donor_instruction_identical'] for i in ids),noise_probes=[])
            for ti,t in enumerate(protocol['noise_probe_steps']):
                keys=[k for k in rows[ids[0]]['noise_probes'][ti] if k not in ('t','alpha_bar')]
                s['noise_probes'].append(dict(t=t,alpha_bar=rows[ids[0]]['noise_probes'][ti]['alpha_bar'],
                    **{key:float(np.mean([rows[i]['noise_probes'][ti][key] for i in ids])) for key in keys}))
            summary[source]=s
        final_learned=model_sha(trainable_state_dict(model));final_frozen=frozen_digest(model,trainable_state_dict(model))
        assert final_learned==initial_learned and final_frozen==initial_frozen
        report.update(execution_complete=True,source_summary=summary,
                      wrong_context_metrics=aggregate(rows),initial_learned_sha256=initial_learned,final_learned_sha256=final_learned,
                      initial_frozen_sha256=initial_frozen,final_frozen_sha256=final_frozen,
                      terminal_alpha_bar=float(model.alpha_bars[-1]),
                      inference_seconds=time.monotonic()-inference_started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                      static_code_findings=dict(multiresolution_FiLM_conditioning=True,frozen_encoders_eval_during_train=True,
                                               common_noise_schedule_and_steps=True,
                                               gripper_training_uses_clean_expert_pose_inference_uses_generated_pose=True),
                      interpretation_limit='Whole fused context was permuted, so this does not isolate visual versus language pathways. Sensitivity alone does not establish useful conditioning or manipulation success.')
    except Exception as e:
        report['errors'].append(type(e).__name__+': '+str(e));raise
    finally:
        if swapped:
            import numpy as np
            np.savez_compressed(OUT/'permuted_predictions.npz',predictions=np.stack(swapped))
        report['elapsed_seconds']=time.monotonic()-started
        write(OUT/'report.json',report)
        print('CONDITIONING_SAMPLING_AUDIT_REPORT',OUT/'report.json',flush=True)
        print(json.dumps({s:{k:v for k,v in x.items() if k!='noise_probes'} for s,x in report.get('source_summary',{}).items()},indent=2),flush=True)


if __name__=='__main__':main()
