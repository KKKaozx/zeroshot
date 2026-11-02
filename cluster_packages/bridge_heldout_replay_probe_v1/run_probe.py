"""Old4-window diagnostic only. Never score new13 holdout or relax its replay guard."""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
CODE=Path(__file__).resolve().parent
ORIGINAL=CODE.parent/'bridge_heldout_evaluation_v1'
BASE=CODE.parent/'bridge_growth128_v1'
for path in [BASE,CODE.parent/'bridge_state_conditioning_v1']:sys.path.insert(0,str(path))
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import UnifiedRobotDataset
from bridge_input_contract import raw_identity
from models import RobotAdapterModel
from replay_and_fit_bridge_positive import seed,digest,evaluate
from state_input_contract import state_features,add_zero_port
from fit_state_arm import PREFIXES,trainable_copy,state_sha

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def tsha(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def delta(actual,reference):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    d=np.abs(a-b)
    q=a[...,3:7];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    r=b[...,3:7];r=r/np.linalg.norm(r,axis=-1,keepdims=True)
    angle=np.degrees(2*np.arccos(np.clip(np.abs((q*r).sum(-1)),0,1)))
    return dict(max_abs=float(d.max()),xyz_max_abs_normalized=float(d[...,:3].max()),
        xyz_max_component_difference_cm=float(d[...,:3].max()*10),
        position_vector_max_difference_cm=float(np.linalg.norm(a[...,:3]-b[...,:3],axis=-1).max()*10),
        quaternion_max_abs=float(d[...,3:7].max()),rotation_max_difference_deg=float(angle.max()),gripper_max_abs=float(d[...,7].max()))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--clip',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    contract=read(CODE/'probe_contract.json');p=read(ORIGINAL/'protocol.json')
    r=dict(execution_checks_passed=False,probe_contract_sha256=sha(CODE/'probe_contract.json'),entry_sha256=sha(Path(__file__)),
        new_holdout_scored=False,new_holdout_predictions_generated=False,training=False,optimizer_created=False,
        old_windows=4,new_windows=0,original_numeric_tolerances_changed=False,models={})
    def persist():r['elapsed_seconds']=time.monotonic()-start;write(args.output/'report.json',r)
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'probe_contract_snapshot.json').write_bytes((CODE/'probe_contract.json').read_bytes())
    try:
        for line in (CODE/'SHA256SUMS').read_text().splitlines():
            h,name=line.split('  ',1);assert sha(CODE/name)==h
        assert sha(ORIGINAL/'protocol.json')==contract['original_protocol_sha256']
        assert sha(ORIGINAL/'run_evaluation.py')==contract['failed_entry_sha256']
        for name,h in p['inherited_sha256'].items():assert sha(CODE.parent/name)==h,name
        assert not (ORIGINAL/'evaluation_claim.json').exists(), 'Do not diagnose by repeating an already-scored new test'
        failed_path=args.project/'runs/bridge-heldout-eval-208005/report.json';failed=read(failed_path)
        assert failed['entry_sha256']==contract['failed_entry_sha256'] and failed['protocol_sha256']==contract['original_protocol_sha256']
        assert not failed['test_predictions_generated'] and not failed['models'] and not failed['execution_checks_passed']
        assert 'coverage' in failed['error'] and 'replay' in failed['error'] and failed['error'].startswith('AssertionError')
        (args.output/'failed_208005_report.json').write_bytes(failed_path.read_bytes());r['failed_report_sha256']=sha(failed_path)
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        selection=read(ORIGINAL/'selection.json');assets=read(ORIGINAL/'assets.json');rows=selection['replay_rows']
        assert sha(ORIGINAL/'selection.json')==p['selection_sha256'] and sha(ORIGINAL/'assets.json')==failed['assets_sha256']
        assert sha(ORIGINAL/'input_reference.npz')==assets['files']['input_reference.npz']
        assert len(rows)==4
        assert not {(v['shard'],v['record_index']) for v in rows}&{(v['shard'],v['record_index']) for v in selection['rows']}
        reader=UnifiedRobotDataset.__new__(UnifiedRobotDataset);reader.chunk_size=16
        reader.bridge_gripper_policy='reverse_scan_valid_steps_v2';reader.bridge_current_gripper='continuous'
        items=[];examples={};observed=[]
        for i,row in enumerate(rows):
            key=(row['shard'],row['record_index'])
            if key not in examples:examples[key]=UnifiedRobotDataset._load_tfrecord_example(str(ORIGINAL/'data'/row['shard']),row['record_index'])
            ex=examples[key];ref=assets['row_checks'][i];t=row['start_index']
            assert raw_identity(ex,t)==ref['raw_identity']
            assert hashlib.sha256(ex.SerializeToString(deterministic=True)).hexdigest()==ref['payload_sha256']
            reader._load_tfrecord_example=lambda *_:ex
            item=reader._get_tfrecord_bridge_state_action(dict(file_path=row['shard'],record_index=row['record_index'],start_index=t))
            assert item[0]==ref['text'];items.append(item)
            states=np.asarray(ex.features.feature['steps/observation/state'].float_list.value,dtype='<f4').reshape(-1,7)
            observed.append(states[t,:6])
        targets=np.stack([v[3].numpy() for v in items]);assert targets.shape==(4,16,8)
        sp=read(CODE.parent/'bridge_state_conditioning_v1/protocol.json')
        with np.load(CODE.parent/'bridge_state_conditioning_v1/current_pose.npz',allow_pickle=False) as z:old_observed=z['current_pose'].copy()
        assert np.array_equal(old_observed[128:132],np.stack(observed))
        original_features=state_features(old_observed,sp['state_features'])
        # Read only already-decoded current observations to reproduce the failed64-row arithmetic.
        # No new test targets/images are loaded or predicted here.
        with np.load(ORIGINAL/'input_reference.npz',allow_pickle=False) as z:failed_observed=z['observed'].copy()
        failed_features=state_features(failed_observed,sp['state_features'])[:4]
        r['state_feature_max_difference']=float(np.abs(original_features[128:132]-failed_features).max())
        assert np.array_equal(failed_observed[:4],np.stack(observed))
        assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.reset_peak_memory_stats()
        cfg=copy.deepcopy(read(CODE.parent/'bridge_coverage_replication_v1/protocol.json')['config']);cfg['model']['name']=str(args.clip)
        seed(42);model=RobotAdapterModel(cfg).cuda().eval();tokenizer=CLIPTokenizer.from_pretrained(str(args.clip),local_files_only=True)
        clip_hash=[digest(model.vision_encoder),digest(model.text_encoder)]
        small=read(BASE/'manifest.json');old_report=read(args.project/'runs/bridge-coverage-rep-fit-207558/report.json')
        assert sha(args.project/'runs/bridge-coverage-rep-fit-207558/report.json')==p['models'][0]['report_sha256']
        key_identity=hashlib.sha256(json.dumps(dict(manifest=sha(BASE/'manifest.json'),checks=old_report['retained_data_checks'],sources=small['source_sha256']),sort_keys=True).encode()).hexdigest()
        cached=torch.load(args.project/'bridge_state_prior_v1/full_tokens.pt',map_location='cpu',weights_only=True)
        assert cached['key']['identity']==key_identity and cached['key']['clip']==clip_hash and cached['key']['torch_version']==str(torch.__version__)
        assert cached['hashes']=={k:tsha(cached[k]) for k in ['visual','language']}
        assert cached['visual'].shape[0]==cached['language'].shape[0]==168
        assert hashlib.sha256(original_features.tobytes()).hexdigest()==old_report['retained_state_features_sha256']
        old_text=tokenizer([v['task'] for v in small['train']+small['development']],padding=True,truncation=True,max_length=77,return_tensors='pt')
        assert cached['key']['tokenizer_inputs']=={k:tsha(v) for k,v in old_text.items()}
        # Reproduce failed encoding shapes: first16 text inputs, first4 images. Other text rows are metadata only.
        text=tokenizer([v['text'] for v in assets['row_checks'][:16]],padding='max_length',truncation=True,max_length=cached['language'].shape[1],return_tensors='pt')
        assert all(torch.equal(text[k][:4],v[128:132]) for k,v in old_text.items())
        mask=text['attention_mask'][:4].cuda()
        cached_pair=(cached['visual'][128:132].cuda(),cached['language'][128:132].cuda());del cached
        encoded={};r['encoding_differences']={}
        for style in ['original','failed']:
            scope=torch.no_grad() if style=='original' else torch.inference_mode()
            with scope:
                visual=torch.cat([model.vision_encoder(pixel_values=torch.stack([v[1] for v in items[i:i+2]]).cuda()).last_hidden_state for i in range(0,4,2)])
                language=model.text_encoder(**{k:v.cuda() for k,v in text.items()}).last_hidden_state[:4]
            encoded[style]=(visual,language)
            r['encoding_differences'][style]=dict(visual_max_abs=float((visual-cached_pair[0]).abs().max()),language_max_abs=float((language-cached_pair[1]).abs().max()),
                visual_exact=torch.equal(visual,cached_pair[0]),language_exact=torch.equal(language,cached_pair[1]))
        model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
        model.history_projection=torch.nn.Linear(1024,1024,bias=False).cuda();torch.nn.init.zeros_(model.history_projection.weight)
        initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(('vision_encoder.','text_encoder.'))}
        initial_sha=hashlib.sha256(''.join(n+tsha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
        add_zero_port(model);augmented=state_sha(trainable_copy(model))
        current=torch.stack([v[2] for v in items]).cuda();states={'original':torch.from_numpy(original_features[128:132]).cuda(),'failed':torch.from_numpy(failed_features).cuda()}
        cases=contract['cases'];persist()
        for fixed in p['models']:
            label=fixed['label'];directory=args.project/'runs'/fixed['report_directory'];checkpoint=directory/fixed['checkpoint']['path']
            assert sha(directory/'report.json')==fixed['report_sha256'] and sha(checkpoint)==fixed['checkpoint']['sha256']
            assert initial_sha==fixed['initial_shared_sha256'] and augmented==fixed['augmented_initial_sha256'] and str(torch.__version__)==fixed['torch_version']
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert payload['config']==cfg and payload['update']==8192 and set(payload['trainable_state_dict'])==set(trainable_copy(model))
            model.load_state_dict(payload['trainable_state_dict'],strict=False);assert state_sha(trainable_copy(model))==state_sha(payload['trainable_state_dict']);del payload
            parameter_hash=digest(model)
            assert sha(ORIGINAL/(label+'_replay.npz'))==assets['files'][label+'_replay.npz']
            with np.load(ORIGINAL/(label+'_replay.npz'),allow_pickle=False) as z:
                reference_prediction=z['predictions'].copy();reference_context=z['contexts'].copy()
                assert reference_prediction.shape==(3,4,16,8) and reference_context.shape==(4,1024)
                assert np.max(np.abs(targets-z['targets']))<=1e-6
            r['models'][label]={}
            sampled={}
            for case in cases:
                name=case['name'];original=case['context_style']=='original'
                for n,param in model.named_parameters():param.requires_grad_(original and n.startswith(PREFIXES))
                model.eval()
                if case['tokens']=='saved_context':ctx=torch.from_numpy(reference_context).cuda()
                else:
                    visual,language=cached_pair if case['tokens']=='cached' else encoded[case['tokens']]
                    scope=torch.no_grad() if original else torch.inference_mode()
                    with scope:
                        result=[]
                        for lo in range(0,4,2):
                            now=model.adapter(visual[lo:lo+2],language[lo:lo+2],mask[lo:lo+2])
                            repeated=model.adapter(visual[lo:lo+2],language[lo:lo+2],mask[lo:lo+2])
                            result.append(now+model.history_projection(repeated-now)+model.state_projection(states[case['state_style']][lo:lo+2]))
                        ctx=torch.cat(result)
                # Avoid repeating sampling when BOTH exact context and parameter flags match.
                sample_key=(tsha(ctx),case['context_style'])
                reused_from=None
                if sample_key in sampled:
                    prediction,reused_from=sampled[sample_key]
                else:
                    prediction,_=evaluate(model,ctx,current,targets,[1101,1102,1103],batch=2)
                    sampled[sample_key]=(prediction.copy(),name)
                # Save diagnostic evidence even when the old tolerance is exceeded; never proceed to new test.
                np.savez_compressed(args.output/(label+'_'+name+'.npz'),predictions=prediction,contexts=ctx.cpu().numpy(),targets=targets)
                context_delta=float(np.abs(ctx.cpu().numpy()-reference_context).max());d=delta(prediction,reference_prediction)
                result=dict(context_max_abs=context_delta,prediction_difference=d,context_within_original_tolerance=context_delta<=p['replay_guard']['context_atol'],
                    prediction_within_original_tolerance=d['max_abs']<=p['replay_guard']['prediction_atol'],reused_identical_sampling_from=reused_from)
                r['models'][label][name]=result
                assert digest(model)==parameter_hash and all(v.grad is None for v in model.parameters())
                persist();print('OLD_REPLAY_ONLY',label,name,'context',context_delta,'prediction',d['max_abs'],flush=True)
        assert clip_hash==[digest(model.vision_encoder),digest(model.text_encoder)]
        r.update(execution_checks_passed=True,parameters_unchanged=True,clip_unchanged=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
            interpretation='Diagnostic completion only. Original tolerance is unchanged. No authorization or automatic continuation to new13 test inference.')
    except Exception as error:r['error']=repr(error);raise
    finally:
        persist();print('OLD REPLAY DIAGNOSTIC REPORT',args.output/'report.json',flush=True)

if __name__=='__main__':main()
