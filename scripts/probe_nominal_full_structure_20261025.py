"""Two actual training batches for resource measurement, never model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/nominal_full_resource_20261025/attempt1')
    options=parser.parse_args();out=ROOT/options.output;out.parent.mkdir(exist_ok=True,parents=True);out.mkdir(exist_ok=False)
    usage=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    used,util=map(int,usage.split(','))
    if used>1024 or util>20:raise ValueError('GPU busy; leave existing workload alone')
    import torch
    import tensorflow as tf
    from torch import nn
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    import train as entry
    from oxe_nominal_dataset import NominalOXEDataset
    from step_training_state import ResumableBatchPlan
    clip=Path('D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41')
    locations=ROOT/'reports/nominal_main_entry_20261025/attempt2/source_locations.json'
    data=NominalOXEDataset(ROOT/'training_cache',ROOT/'reports/oxe_nominal_targets_20261025',locations)
    argv=['--visual-backbone','dinov2','--dino-assets',str(ROOT/'baselines/dinov2_large_frozen_assets_20261023'),
        '--model-name',str(clip),'--adapter-layers','8','--attention-dim','512','--attention-heads','8',
        '--adapter-pooling','cls','--fusion-type','cross_attention','--dropout','0.1',
        '--chunk-size','16','--diffusion-steps','100','--diffusion-prediction-type','epsilon',
        '--beta-schedule','squaredcos_cap_v2','--oxe-command-input','--action-frame','source_cartesian',
        '--max-normalized-position','8','--gripper-target-mode','state','--gripper-change-weight','1']
    args=entry.parse_args(argv);config=entry.build_config(args)
    config['action'].update(nominal_target_binding=data.nominal_target_binding,source_position_to_meters_verified=False)
    config['model'].update(diffusion_architecture='multiscale',diffusion_down_dims=[256,512,1024],
        diffusion_step_embed_dim=128,diffusion_kernel_size=5)
    (out/'candidate_model_config.json').write_text(json.dumps(config,indent=2)+'\n')
    entry.set_seed(42);device=torch.device('cuda')
    tick=time.monotonic();model=entry.RobotAdapterModel(config).to(device).train()
    tokenizer=entry.CLIPTokenizer.from_pretrained(str(clip),local_files_only=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.0001)
    torch.cuda.synchronize();load_seconds=time.monotonic()-tick
    frozen=[(p,p._version) for encoder in (model.vision_encoder,model.text_encoder) for p in encoder.parameters()]
    trainable=[p for p in model.parameters() if p.requires_grad]
    from collections import Counter
    counts=Counter(s['source'] for s in data.samples)
    weights=[1/counts[s['source']] for s in data.samples]
    plan=ResumableBatchPlan(list(range(len(data))),batch_size=8,total_updates=2,seed=42,
        dataset_identity=data.nominal_target_binding['dataset_identity'],mode='weighted_replacement',weights=weights)
    torch.cuda.reset_peak_memory_stats();updates=[]
    for _ in range(2):
        indices=plan.begin_batch();tick=time.monotonic()
        instructions,images,current,actions,masks=entry.collate_batch([data[i] for i in indices])
        read_seconds=time.monotonic()-tick
        tick=time.monotonic();text=entry.tokenise(tokenizer,instructions,device)
        actions,current,masks=actions.to(device),current.to(device),masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        output=model(images.to(device),text['input_ids'],attention_mask=text.get('attention_mask'),current_gripper=current,actions=actions)
        loss=entry.policy_loss(model,output,actions,nn.MSELoss(),current_grippers=current,supervision_masks=masks)[0]
        if not torch.isfinite(loss):raise ValueError('Nonfinite full-structure loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();update_seconds=time.monotonic()-tick;plan.commit_batch()
        updates.append({'update':plan.completed_updates,'indices':indices,
            'sources':[data.samples[i]['source'] for i in indices],
            'input_read_seconds':read_seconds,'optimizer_update_seconds':update_seconds,'loss':float(loss.detach())})
        print('NOMINAL_FULL_RESOURCE',plan.completed_updates,'/2','read',round(read_seconds,3),'update',round(update_seconds,3),flush=True)
    assert all(not p.requires_grad and p.grad is None and p._version==version for p,version in frozen)
    parameter_bytes=sum(p.numel()*p.element_size() for p in trainable)
    report={'passed':True,'scope':'Actual full main architecture local resource probe only',
        'gpu':torch.cuda.get_device_name(),'torch_version':torch.__version__,'cuda_runtime':torch.version.cuda,
        'batch_size':8,'adapter_layers':8,'attention_dim':512,'attention_heads':8,
        'full_multiscale_decoder':True,'diffusion_steps':100,'optimizer_updates':2,
        'target_binding':data.nominal_target_binding,'source_balanced_replacement':True,
        'updates':updates,'frozen_encoders_unchanged':True,
        'trainable_parameters':sum(p.numel() for p in trainable),'trainable_parameter_bytes':parameter_bytes,
        'weights_plus_two_adam_moment_bytes':3*parameter_bytes,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'model_load_seconds':load_seconds,
        'checkpoint_resume': 'covered by main-entry continuous2 versus1+1 route, not repeated here',
        'local_gpu_rate_is_cluster_estimate':False,'formal_budget_frozen':False,
        'validation_test_or_libero_examples_read':0,'new_cluster_jobs':0,'policy_performance_verdict':None,
        'trained_resource_weights_saved':False,'formal_training_ready':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
