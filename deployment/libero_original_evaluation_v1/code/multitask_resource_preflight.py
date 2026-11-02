"""Eight joint updates on ten real training windows, without model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip-dir', type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Report already exists')
    root = Path(__file__).resolve().parent
    report = dict(passed=False, purpose='multitask_joint_model_resource_preflight',
        formal_training=False, validation_evaluated=False, reserved_test_targets_read=False,
        updates=8, batch_size=2, seed=42, losses=[])
    try:
        manifest = json.loads((root/'manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        if args.clip_dir:
            clip = args.clip_dir
        else:
            candidates = sorted(Path('/projects/Zeroshot').glob('**/models--openai--clip-vit-large-patch14/snapshots/*/config.json'))
            valid = [p.parent for p in candidates if any((p.parent/n).exists() for n in ('model.safetensors','pytorch_model.bin'))]
            if not valid: raise RuntimeError('No existing full CLIP ViT-L/14 cache found; no weights downloaded. Provide --clip-dir or prepare weights separately.')
            clip = valid[0]
        report['clip_path'] = str(clip)
        import numpy as np
        import torch
        from transformers import CLIPTokenizer
        from models import RobotAdapterModel
        from train import policy_loss, set_seed
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        set_seed(42)
        torch.set_num_threads(4)
        config = {'action':{'max_normalized_position':3.0}, 'model':{
            'name':str(clip), 'num_adapter_layers':8, 'attention_dim':512,
            'num_attention_heads':8, 'adapter_pooling':'cls_patch_mean','dropout':0.1,
            'decoder_type':'diffusion','decoder_hidden_dim':256,'chunk_size':16,'action_dim':8,
            'num_diffusion_steps':100,'beta_schedule':'squaredcos_cap_v2', 'clip_denoised':True,
            'diffusion_prediction_type':'sample','separate_gripper_head':True,
            'trajectory_conditioned_gripper':True,'condition_on_current_gripper':True,
            'current_gripper_encoding':'bridge_measured_affine_unbounded_v1',
            'gripper_target_mode':'state','balanced_gripper_loss':False,'gripper_loss_weight':0.25}}
        report['config'] = config
        model = RobotAdapterModel(config).cuda().train()
        assert model.vision_encoder.config.hidden_size == 1024 and model.text_encoder.config.hidden_size == 768
        tokenizer = CLIPTokenizer.from_pretrained(str(clip),local_files_only=True)
        data = np.load(root/'training_probe.npz',allow_pickle=False)
        assert len(data['images']) == 10 and data['masks'].shape == (10,16,8)
        def digest(params):
            h = hashlib.sha256()
            for name,p in params: h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        frozen = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
        assert frozen and all(n.startswith(('vision_encoder.','text_encoder.')) for n,p in frozen)
        groups = {key:[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefix)]
            for key,prefix in [('adapter',('adapter.',)),('pose',('diffusion_decoder.',)),
                ('gripper',('gripper_','current_gripper_projection.'))]}
        before = {k:digest(v) for k,v in groups.items()}
        frozen_before = digest(frozen)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
        report['probe_learning_rate'] = 1e-4
        report['gpu'] = torch.cuda.get_device_name()
        report['torch'] = torch.__version__
        torch.cuda.reset_peak_memory_stats()
        durations=[]
        for step in range(8):
            ix = [(step*2+j)%10 for j in range(2)]
            text = tokenizer(data['instructions'][ix].tolist(),padding=True,truncation=True,return_tensors='pt')
            text = {k:v.cuda() for k,v in text.items()}
            images,current,actions,masks = [torch.from_numpy(data[k][ix]).cuda() for k in ('images','current','actions','masks')]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); started=time.monotonic()
            output=model(images,text['input_ids'],attention_mask=text['attention_mask'],current_gripper=current,actions=actions)
            loss,pose,gripper=policy_loss(model,output,actions,torch.nn.MSELoss(),current_grippers=current,supervision_masks=masks)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            for key,params in groups.items():
                grads=[p.grad for n,p in params if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.any(g!=0) for g in grads):
                    raise RuntimeError('Missing/nonfinite/zero gradients: '+key)
            assert all(p.grad is None for n,p in frozen)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step(); torch.cuda.synchronize(); durations.append(time.monotonic()-started)
            report['losses'].append({'step':step+1,'loss':float(loss.detach()),'pose':float(pose.detach()),'gripper':float(gripper.detach())})
            print('UPDATE',report['losses'][-1],flush=True)
        report['module_changed']={k:digest(v)!=before[k] for k,v in groups.items()}
        report['clip_unchanged']=digest(frozen)==frozen_before
        assert all(report['module_changed'].values()) and report['clip_unchanged']
        report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
        report['peak_reserved_gib']=torch.cuda.max_memory_reserved()/1024**3
        report['mean_warm_update_seconds']=sum(durations[1:])/len(durations[1:])
        report['passed']=True
    except Exception as error:
        report['error']=repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT',args.output,flush=True)


if __name__ == '__main__': main()
