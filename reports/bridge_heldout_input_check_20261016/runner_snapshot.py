"""Guarded, one-time evaluation of frozen207558/207746 on13 reserved Bridge episodes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
CODE=Path(__file__).resolve().parent
BASE=CODE.parent/'bridge_growth128_v1'
for path in [BASE,CODE.parent/'bridge_state_conditioning_v1']:
    sys.path.insert(0,str(path))
import numpy as np
import torch
from dataset import UnifiedRobotDataset
from bridge_input_contract import raw_identity
from state_input_contract import state_features,add_zero_port
from models import RobotAdapterModel
from transformers import CLIPTokenizer
from fit_state_arm import PREFIXES,trainable_copy,state_sha
from replay_and_fit_bridge_positive import seed,digest,evaluate

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def load_inputs(protocol,report):
    for line in (CODE/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1);path=CODE/name
        assert path.resolve().is_relative_to(CODE.resolve()) and sha(path)==expected,name
    for name,expected in protocol['inherited_sha256'].items():assert sha(CODE.parent/name)==expected,name
    selection=read(CODE/'selection.json');assets=read(CODE/'assets.json')
    assert sha(CODE/'selection.json')==protocol['selection_sha256']
    assert assets['protocol_frozen_before_decoding_sha256']==sha(CODE/'protocol.json')
    assert assets['source_contract_passed'] and assets['no_episodes_or_windows_excluded']
    rows=selection['replay_rows']+selection['rows'];assert len(rows)==64
    for name,expected in assets['files'].items():assert sha(CODE/name)==expected,name
    # TF only parses serialized records; keep it from allocating the policy GPU.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset);reader.chunk_size=16
    reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
    payloads={};items=[];observed=[];checks=[]
    with np.load(CODE/'input_reference.npz',allow_pickle=False) as z:
        for i,row in enumerate(rows):
            key=(row['shard'],row['record_index'])
            if key not in payloads:payloads[key]=UnifiedRobotDataset._load_tfrecord_example(str(CODE/'data'/row['shard']),row['record_index'])
            example=payloads[key];reference=assets['row_checks'][i];t=row['start_index']
            assert reference['row']==row and reference['index']==i
            assert hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest()==reference['payload_sha256']
            assert raw_identity(example,t)==reference['raw_identity']
            reader._load_tfrecord_example=lambda *_:example
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=row['shard'],record_index=row['record_index'],start_index=t))
            assert item[0]==reference['text'] and ' '.join(item[0].lower().split())==row['task']
            assert item[1].shape==(3,224,224) and item[2].shape==(1,) and item[3].shape==(16,8)
            differences={}
            for name,k,limit in [('images',1,1e-6),('currents',2,0),('targets',3,1e-6)]:
                actual=item[k].numpy();expected=z[name][i]
                assert actual.dtype==expected.dtype and actual.shape==expected.shape and np.isfinite(actual).all()
                delta=float(np.abs(actual.astype(float)-expected.astype(float)).max())
                assert delta<=limit,(i,name,delta)
                differences[name]=delta
            assert np.array_equal(item[3].numpy()[:,7],z['targets'][i,:,7])
            states=np.asarray(example.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
            assert t+16<len(states) and np.array_equal(states[t,:6],z['observed'][i])
            observed.append(states[t,:6]);items.append(item);checks.append(dict(index=i,raw_exact=True,max_absolute_differences=differences))
        state_protocol=read(CODE.parent/'bridge_state_conditioning_v1/protocol.json')
        features=state_features(np.stack(observed),state_protocol['state_features'])
        assert np.abs(features-z['state_features']).max()<=1e-6
    report.update(input_contract_passed=True,input_checks=checks,state_statistics_changed=False,raw_payloads_checked=len(payloads),
        input_binding_sha256=hashlib.sha256(json.dumps(dict(protocol=sha(CODE/'protocol.json'),assets=sha(CODE/'assets.json')),sort_keys=True).encode()).hexdigest())
    return rows,items,features

def errors(pred,truth):
    pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    assert pred.shape[-2:]==truth.shape[-2:]==(16,8) and np.isfinite(pred).all() and np.isfinite(truth).all()
    pos=np.linalg.norm(pred[...,:3]-truth[...,:3],axis=-1)*10.
    q=pred[...,3:7];t=truth[...,3:7];qn=np.linalg.norm(q,axis=-1,keepdims=True);tn=np.linalg.norm(t,axis=-1,keepdims=True)
    assert qn.min()>1e-8 and tn.min()>1e-8
    angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum((q/qn)*(t/tn),axis=-1)),0,1)))
    return pos,angle

def summary(pred,truth,rows):
    pos,rot=errors(pred,truth)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['shard']+'::'+str(row['record_index'])].append(i)
    def mean(indices):
        return dict(path_position_cm=float(pos[:,indices].mean()),path_rotation_deg=float(rot[:,indices].mean()),
            endpoint_position_cm=float(pos[:,indices,-1].mean()),endpoint_rotation_deg=float(rot[:,indices,-1].mean()))
    episodes=[dict(episode=key,task=rows[ix[0]]['task'],windows=len(ix),**mean(ix)) for key,ix in groups.items()]
    tasks=[]
    for task in sorted({r['task'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['task']==task]
        tasks.append(dict(task=task,windows=len(ix),**mean(ix)))
    return dict(windows=len(rows),episodes=len(groups),window_weighted=mean(list(range(len(rows)))),
        episode_equal={k:float(np.mean([e[k] for e in episodes])) for k in episodes[0] if k.endswith(('_cm','_deg'))},
        per_episode=episodes,per_task=tasks,per_target=[dict(target=k+1,position_cm=float(pos[...,k].mean()),rotation_deg=float(rot[...,k].mean())) for k in range(16)])

@torch.inference_mode()
def encode(model,tokenizer,items):
    text=tokenizer([v[0] for v in items],padding=True,truncation=True,max_length=77,return_tensors='pt')
    visual=[];language=[]
    for lo in range(0,len(items),2):
        visual.append(model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[lo:lo+2]]).cuda()).last_hidden_state)
        language.append(model.text_encoder(**{k:v[lo:lo+2].cuda() for k,v in text.items()}).last_hidden_state)
    return torch.cat(visual),torch.cat(language),text['attention_mask'].cuda()

@torch.inference_mode()
def contexts(model,visual,language,mask,state):
    result=[]
    for lo in range(0,len(state),2):
        now=model.adapter(visual[lo:lo+2],language[lo:lo+2],mask[lo:lo+2])
        repeated=model.adapter(visual[lo:lo+2],language[lo:lo+2],mask[lo:lo+2])
        result.append(now+model.history_projection(repeated-now)+model.state_projection(state[lo:lo+2]))
    return torch.cat(result)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True);ap.add_argument('--clip',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--mode',choices=['input-check','evaluate'],default='evaluate')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();p=read(CODE/'protocol.json')
    report=dict(stage=args.mode,execution_checks_passed=False,training=False,optimizer_created=False,gripper_trained=False,
        downloads=False,simulation=False,protocol_sha256=sha(CODE/'protocol.json'),assets_sha256=sha(CODE/'assets.json'),
        entry_sha256=sha(Path(__file__)),test_predictions_generated=False,replay_checks={},models={},input_contract_passed=False)
    def persist():
        report['elapsed_seconds']=time.monotonic()-started;write(args.output/'report.json',report)
    for source,dest in [('protocol.json','protocol_snapshot.json'),('assets.json','assets_snapshot.json'),('run_evaluation.py','runner_snapshot.py')]:
        (args.output/dest).write_bytes((CODE/source).read_bytes())
    try:
        if args.mode=='evaluate':assert not (CODE/'evaluation_claim.json').exists(),'This test evaluation was already claimed; inspect saved evidence, do not resubmit'
        rows,items,features=load_inputs(p,report);persist()
        if args.mode=='input-check':
            report.update(execution_checks_passed=True,gpu_used=False,model_loaded=False)
            return
        assert torch.cuda.is_available() and args.clip.is_dir()
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
        # Restore BOTH frozen models before any new test prediction is generated.
        source_reports={}
        for fixed in p['models']:
            directory=args.project/'runs'/fixed['report_directory'];old=read(directory/'report.json')
            assert sha(directory/'report.json')==fixed['report_sha256']
            checkpoint=directory/fixed['checkpoint']['path']
            assert checkpoint.stat().st_size==fixed['checkpoint']['bytes'] and sha(checkpoint)==fixed['checkpoint']['sha256']
            source_reports[fixed['label']]=old
        cfg=copy.deepcopy(read(CODE.parent/'bridge_coverage_replication_v1/protocol.json')['config']);cfg['model']['name']=str(args.clip)
        seed(42);model=RobotAdapterModel(cfg).cuda().eval()
        tokenizer=CLIPTokenizer.from_pretrained(str(args.clip),local_files_only=True)
        visual,language,mask=encode(model,tokenizer,items)
        clip_before=[digest(model.vision_encoder),digest(model.text_encoder)]
        model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
        model.history_projection=torch.nn.Linear(1024,1024,bias=False).cuda();torch.nn.init.zeros_(model.history_projection.weight)
        initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(('vision_encoder.','text_encoder.'))}
        initial_shared=hashlib.sha256(''.join(n+tsha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
        add_zero_port(model);initial_augmented=state_sha(trainable_copy(model))
        frozen_before={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(PREFIXES) and not n.startswith(('vision_encoder.','text_encoder.'))}
        for fixed in p['models']:
            assert str(torch.__version__)==fixed['torch_version'] and initial_shared==fixed['initial_shared_sha256'] and initial_augmented==fixed['augmented_initial_sha256']
        for param in model.parameters():param.requires_grad_(False)
        state=torch.from_numpy(features).cuda();current=torch.stack([v[2] for v in items]).cuda();truth=np.stack([v[3].numpy() for v in items])
        expected_keys=set(trainable_copy(model));contexts_by_model={}
        def restore(fixed):
            payload=torch.load(args.project/'runs'/fixed['report_directory']/fixed['checkpoint']['path'],map_location='cpu',weights_only=False)
            assert payload['config']==cfg and payload['update']==8192 and payload['initial_shared_sha256']==initial_shared
            old=source_reports[fixed['label']]
            assert payload['protocol']==read(CODE.parent/('bridge_coverage_replication_v1' if fixed['label']=='coverage' else 'bridge_rotation_weight_v1')/'protocol.json')
            assert set(payload['trainable_state_dict'])==expected_keys
            wanted=state_sha(payload['trainable_state_dict']);loaded=model.load_state_dict(payload['trainable_state_dict'],strict=False)
            assert not loaded.unexpected_keys and not any(n.startswith(PREFIXES) for n in loaded.missing_keys)
            assert state_sha(trainable_copy(model))==wanted
            assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in frozen_before.items())
            model.eval();return wanted
        for fixed in p['models']:
            label=fixed['label'];weight_hash=restore(fixed)
            with torch.inference_mode():ctx=contexts(model,visual,language,mask,state)
            with np.load(CODE/(label+'_replay.npz'),allow_pickle=False) as z:
                context_delta=float(np.abs(ctx[:4].cpu().numpy()-z['contexts']).max())
                assert context_delta<=p['replay_guard']['context_atol'],(label,'context',context_delta)
                prediction,_=evaluate(model,ctx[:4],current[:4],truth[:4],p['sampling']['seeds'],batch=2)
                delta=float(np.abs(prediction-z['predictions']).max())
                assert delta<=p['replay_guard']['prediction_atol'],(label,'replay',delta)
            assert state_sha(trainable_copy(model))==weight_hash
            np.savez_compressed(args.output/(label+'_replay_check.npz'),predictions=prediction,contexts=ctx[:4].cpu().numpy(),targets=truth[:4])
            contexts_by_model[label]=ctx
            report['replay_checks'][label]=dict(passed=True,context_max_abs_difference=context_delta,prediction_max_abs_difference=delta,checkpoint_trainable_sha256=weight_hash)
            persist();print('FROZEN_CHECKPOINT_REPLAY',label,'PASSED',context_delta,delta,flush=True)
        assert len(report['replay_checks'])==2 and all(x['passed'] for x in report['replay_checks'].values())
        assert torch.cuda.max_memory_reserved()/2**30<p['resource_limits']['gpu_reserved_gib']
        # Atomic package-wide claim prevents two submitted jobs from scoring this set twice.
        with (CODE/'evaluation_claim.json').open('x') as f:
            json.dump(dict(protocol_sha256=report['protocol_sha256'],output=str(args.output),slurm_job_id=os.environ.get('SLURM_JOB_ID')),f)
        static=np.zeros_like(truth[4:]);static[...,6]=1
        report['static']=summary(static[None],truth[4:],rows[4:]);persist()
        for fixed in p['models']:
            label=fixed['label'];weight_hash=restore(fixed);ctx=contexts_by_model[label][4:]
            prediction,_=evaluate(model,ctx,current[4:],truth[4:],p['sampling']['seeds'],batch=2)
            assert prediction.shape==(3,60,16,8) and state_sha(trainable_copy(model))==weight_hash
            assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in frozen_before.items())
            np.savez_compressed(args.output/(label+'_test_predictions.npz'),predictions=prediction,targets=truth[4:],contexts=ctx.cpu().numpy(),row_indices=np.arange(60))
            m=summary(prediction,truth[4:],rows[4:])
            m['descriptive_mean_advantage_over_static']=all(m[group][key]<report['static'][group][key] for group in ['episode_equal','window_weighted'] for key in ['path_position_cm','path_rotation_deg'])
            report['models'][label]=m;report['test_predictions_generated']=True;persist()
            print('HELDOUT_METRICS',label,json.dumps(dict(window_weighted=m['window_weighted'],episode_equal=m['episode_equal'])),flush=True)
        assert clip_before==[digest(model.vision_encoder),digest(model.text_encoder)]
        report.update(execution_checks_passed=True,clip_unchanged=True,other_frozen_unchanged=True,parameters_updated=False,
            input_state_statistics_from_original_train_only=True,test_set_consumed=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
            interpretation='Offline same-task heldout13episodes only. No hyperparameter/model selection from these scores; no physical grasp, unseen-task or unique-overfitting-root-cause claim.')
        print('FROZEN HELDOUT EVALUATION COMPLETE',args.output/'report.json',flush=True)
    except Exception as error:
        report['error']=repr(error);raise
    finally:
        persist();print('HELDOUT_REPORT',args.output/'report.json',flush=True)

if __name__=='__main__':main()
