"""Paired joint-training control using archived model, loss and metric code."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HEADS = ('regression_head.', 'diffusion_decoder.')


def orders(indices, epochs=20):
    rng = np.random.default_rng(42)
    return [rng.permutation(indices).tolist() for _ in range(epochs)]


def digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def common_state(state):
    return {k:v.clone() for k,v in state.items() if not k.startswith(HEADS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare','preflight','train'], default='preflight')
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(args.pack.resolve()))
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import torch
    from transformers import CLIPTokenizer
    from dataset import UnifiedRobotDataset
    from models import RobotAdapterModel
    from train import (bridge_plan_selection, bridge_plan_splits, collate_batch,
                       set_seed, trainable_state_dict, policy_loss, build_learning_rate_scheduler)
    from evaluate_multitask_pilot import metrics
    from probe_bridge_conditioning import select_windows, sha256
    torch.set_num_threads(4)
    ref = json.loads(args.reference.read_text())
    module_hashes = {name:sha256(args.pack/name) for name in ref['module_sha256']}
    assert module_hashes == ref['module_sha256']
    manifest = args.pack/'manifest.json'
    selected = bridge_plan_selection(manifest)
    dataset = UnifiedRobotDataset(data_dir=str(args.pack/'data'), chunk_size=16, stride=4,
        sources=['tfrecord'], min_trajectory_steps=17, exclude_path_parts=[], exclude_schemas=[],
        tfrecord_splits=['train'], bridge_gripper_policy='reverse_scan_valid_steps_v2',
        bridge_current_gripper='continuous', bridge_episode_selection=selected)
    splits = bridge_plan_splits(dataset)
    assert {k:len(v) for k,v in splits.items()} == json.loads(manifest.read_text())['expected_windows']
    rows = select_windows(dataset,splits,selected,8,all_train=True)
    assert rows == ref['window_selection']
    items = [dataset[r['dataset_index']] for r in rows]
    assert all(torch.isfinite(t).all() for item in items for t in item[1:])
    assert all(torch.all(item[4] == 1) for item in items)
    targets = np.stack([x[3].numpy() for x in items])
    train_indices = [i for i,r in enumerate(rows) if r['partition']=='train']
    schedule = orders(train_indices)
    assert len(train_indices)==1836 and all(len(x)==len(set(x))==1836 and set(x)==set(train_indices) for x in schedule)
    protocol = dict(module_sha256=module_hashes, manifest_sha256=sha256(manifest),
        driver_sha256=sha256(Path(__file__)), reference_sha256=sha256(args.reference),
        order_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        window_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        epochs=20, batch_size=2, updates_per_group=18360, init_seed=42,
        optimizer='AdamW', learning_rate=1e-4, weight_decay=1e-4, gradient_clip_norm=1.,
        lr_schedule='cosine_per_epoch', selection='fixed_final_epoch_20',
        shared_initialization='All trainable state except the two alternative pose heads',
        step_rng='reset seed 420000 + update before each forward; sample order uses independent NumPy RNG',
        objective='Archived regression xyz MSE + sign-invariant quaternion loss versus diffusion x0 MSE; shared gripper loss',
        limitations='Different head capacity/objective; one training seed; not a sampler-only ablation or final generalization benchmark')
    report = dict(phase=args.phase, protocol=protocol, reserved_test_targets_read=False,
        model_loaded=False, trained=False, counts={p:sum(r['partition']==p for r in rows) for p in ['train','validation']},groups={})
    args.output.mkdir(parents=True)
    if args.phase=='prepare':
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('HEAD CONTROL DATA AND ORDER: PASSED',flush=True)
        return
    assert torch.cuda.is_available()
    assert sha256(args.source_run/'latest.pt')==ref['checkpoint_sha256']
    source = torch.load(args.source_run/'latest.pt',map_location='cpu',weights_only=False)
    assert source['split_indices']==splits
    config = copy.deepcopy(source['config'])
    del source
    gc.collect()
    protocol['source_config'] = config
    if args.phase=='train':
        if args.preflight is None:
            raise ValueError('Training requires the passed preflight report')
        preflight = json.loads(args.preflight.read_text())
        assert preflight['passed'] and preflight['phase']=='preflight' and preflight['protocol']==protocol
        report['preflight_sha256'] = sha256(args.preflight)
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['name'],local_files_only=True)
    tokens = tokenizer([x[0] for x in items],padding=True,truncation=True,return_tensors='pt')
    groups = {}
    for part in ['train','validation']:
        indices = [i for i,r in enumerate(rows) if r['partition']==part]
        groups[part+'/overall'] = indices
        for task in sorted({rows[i]['task'] for i in indices}):
            groups[part+'/task/'+task] = [i for i in indices if rows[i]['task']==task]
        for episode in sorted({(rows[i]['shard'],rows[i]['record_index']) for i in indices}):
            groups[part+'/episode/'+episode[0]+'::'+str(episode[1])] = [i for i in indices if (rows[i]['shard'],rows[i]['record_index'])==episode]

    def batch(indices):
        _,images,current,action,mask = collate_batch([items[i] for i in indices])
        return (images.cuda(),tokens['input_ids'][indices].cuda(),tokens['attention_mask'][indices].cuda(),
                current.cuda(),action.cuda(),mask.cuda())

    def evaluate(model, indices, seeds):
        model.eval()
        predictions = np.empty((len(seeds),len(indices),16,8),np.float32)
        with torch.inference_mode():
            for start in range(0,len(indices),2):
                ids=indices[start:start+2]
                images,text,mask,current,_,_=batch(ids)
                context=model.get_context_vector(images,text,mask)
                for si,seed in enumerate(seeds):
                    set_seed(seed*10000+start)
                    value=model.sample(context,current)
                    assert torch.isfinite(value).all()
                    predictions[si,start:start+len(ids)]=value.cpu().numpy()
        return predictions

    shared = None
    clip_hash = None
    for kind in ['regression','diffusion']:
        set_seed(42)
        cfg=copy.deepcopy(config); cfg['model']['decoder_type']=kind
        model=RobotAdapterModel(cfg)
        state=trainable_state_dict(model)
        if shared is None:
            shared=common_state(state)
        else:
            assert set(common_state(state))==set(shared)
            model.load_state_dict(shared,strict=False)
        initial_shared=digest(common_state(trainable_state_dict(model)))
        assert initial_shared==digest(shared)
        encoders=lambda: {k:v for k,v in model.state_dict().items() if k.startswith(('vision_encoder.','text_encoder.'))}
        before_clip=digest(encoders())
        if clip_hash is None: clip_hash=before_clip
        assert before_clip==clip_hash
        del state
        model=model.cuda()
        assert model.fusion_type=='cross_attention' and model.adapter_pooling=='cls_patch_mean'
        assert model.chunk_size==16 and model.separate_gripper_head and model.num_diffusion_steps==100
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        scheduler=build_learning_rate_scheduler(optimizer,'cosine',20)
        torch.cuda.reset_peak_memory_stats()
        times=[]; history=[]; updates=0
        before_trainable=digest(trainable_state_dict(model))
        for epoch,order in enumerate(schedule,1):
            model.train(); losses=[]
            for start in range(0,len(order),2):
                if args.phase=='preflight' and updates==6:break
                images,text,attention,current,action,mask=batch(order[start:start+2])
                set_seed(420000+updates)
                torch.cuda.synchronize(); began=time.monotonic()
                output=model(images,text,attention_mask=attention,current_gripper=current,actions=action)
                loss,pose,grip=policy_loss(model,output,action,torch.nn.MSELoss(),current_grippers=current,supervision_masks=mask)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                times.append(time.monotonic()-began); losses.append(float(loss)); updates+=1
                if updates==1:
                    assert all(not p.requires_grad and p.grad is None for name,p in model.named_parameters() if name.startswith(('vision_encoder.','text_encoder.')))
                    for prefix in ['adapter.', 'regression_head.' if kind=='regression' else 'diffusion_decoder.', 'gripper_head.']:
                        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for name,p in model.named_parameters() if name.startswith(prefix))
                if updates==1 or updates%200==0 or args.phase=='preflight':
                    print('UPDATE',kind,updates,'loss',float(loss),'pose',float(pose),'gripper',float(grip),flush=True)
            scheduler.step()
            history.append(dict(epoch=epoch,updates=updates,mean_loss=float(np.mean(losses))))
            if args.phase=='preflight':break
            (args.output/(kind+'-history.json')).write_text(json.dumps(history,indent=2)+'\n')
        assert digest(trainable_state_dict(model))!=before_trainable
        assert digest(encoders())==before_clip
        evaluation_ids=train_indices[:8] if args.phase=='preflight' else list(range(len(rows)))
        seeds=[0] if args.phase=='preflight' or kind=='regression' else [0,1,2]
        began=time.monotonic(); predictions=evaluate(model,evaluation_ids,seeds); torch.cuda.synchronize()
        evaluation_seconds=time.monotonic()-began
        group_report=dict(updates=updates,trainable_parameters=sum(p.numel() for p in params),
            initial_shared_sha256=initial_shared,clip_sha256=before_clip,clip_unchanged=True,
            warm_update_seconds=float(np.mean(times[1:])),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            evaluation_seconds=evaluation_seconds,evaluation_windows=len(evaluation_ids),sampling_seeds=seeds)
        if args.phase=='train':
            assert updates==18360
            group_report['metrics']={g:[metrics(prediction[ids],targets[ids]) for prediction in predictions] for g,ids in groups.items()}
            torch.save(dict(config=cfg,epoch=20,split_indices=splits,trainable_state_dict=trainable_state_dict(model)),args.output/(kind+'-final.pt'))
            np.savez_compressed(args.output/(kind+'-predictions.npz'),predictions=predictions,targets=targets)
        report['groups'][kind]=group_report
        print('GROUP',kind,json.dumps({k:v for k,v in group_report.items() if k!='metrics'}),flush=True)
        del optimizer,scheduler,params,model,output,loss,pose,grip,images,text,attention,current,action,mask
        gc.collect();torch.cuda.empty_cache()
    report.update(model_loaded=True,trained=True,preflight_weights_discarded=args.phase=='preflight')
    if args.phase=='preflight':
        seconds=sum(g['warm_update_seconds']*18360+g['evaluation_seconds']/8*len(rows)*(3 if k=='diffusion' else 1) for k,g in report['groups'].items())
        report['estimated_training_and_evaluation_hours_with_50_percent_margin']=seconds*1.5/3600
        report['passed']=seconds*1.5 < 3*3600*.8 and all(g['peak_reserved_gib'] < torch.cuda.get_device_properties(0).total_memory/1024**3*.85 for g in report['groups'].values())
    else: report['passed']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HEAD CONTROL',args.phase,'PASSED',report['passed'],args.output/'report.json',flush=True)
    if not report['passed']:raise RuntimeError('Resource gate failed; do not submit training')


if __name__=='__main__':
    main()
