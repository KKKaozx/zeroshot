"""One larger training pool, matched task slots and frozen DDPM recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
CODE=Path(__file__).resolve().parent
BASE=CODE.parent/'bridge_growth128_v1'
for p in [CODE.parent/'bridge_state_conditioning_v1',CODE.parent/'bridge_inner_selection_v1',BASE]: sys.path.insert(0,str(p))
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='2')
import numpy as np
import torch
from transformers import CLIPTokenizer
from models import RobotAdapterModel
from check_bridge_batch_control import metrics,tensor_sha,sha
from replay_and_fit_bridge_positive import digest,evaluate,seed
from trace_bridge_batch_training import observation_guard
from fit_state_arm import PREFIXES,trainable_copy,gradient_norms,cpu_copy,state_sha
from state_input_contract import add_zero_port
from coverage_inputs import load_inputs,tokens


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data-root','data-reference','clip','old-cache','cache','output','control-report']:
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--stage',choices=['preflight','fit'],default='preflight')
    ap.add_argument('--preflight-reference',type=Path)
    ap.add_argument('--local-bounded',action='store_true')
    args=ap.parse_args();assert not (args.local_bounded and args.stage=='fit')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    p=json.loads((CODE/'protocol.json').read_text())
    r=dict(stage=args.stage,local_bounded=args.local_bounded,execution_checks_passed=False,
        protocol_sha256=sha(CODE/'protocol.json'),entry_sha256=sha(CODE/'run_training_coverage.py'),
        input_helper_sha256=sha(CODE/'coverage_inputs.py'),downloads=False,simulation_executed=False,
        gripper_trained=False,reserved_test_targets_read=False,old_control_retrained=False,runs=[],checkpoints=[])
    def persist():
        r['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    (args.output/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:
        for path,h in p['inherited_sha256'].items(): assert sha(CODE.parent/path)==h,path
        for name,expected in [('rows.json',p['rows_sha256']),('schedule.npz',p['schedule_file_sha256']),
                              ('episode_reference.json',p['episode_reference_sha256'])]:assert sha(CODE/name)==expected
        assert sha(args.control_report)==p['control_report_sha256']
        control=json.loads(args.control_report.read_text())
        rows_json=json.loads((CODE/'rows.json').read_text());rows=rows_json['train']+rows_json['development']
        with np.load(CODE/'schedule.npz') as z:batches=torch.from_numpy(z['batches'].copy())
        assert tensor_sha(batches)==p['schedule_tensor_sha256'] and batches.shape==(8192,8)
        assert torch.cuda.is_available()
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.cuda.reset_peak_memory_stats()
        items,features,old_items,checks,retained_indices=load_inputs(args,CODE,p,rows,batches.numpy(),r)
        r['state_statistics_changed']=False
        if not args.local_bounded:
            assert r['full_input_contract_passed'] and r['retained_data_checks']==control['data_checks']
            assert r['retained_state_features_sha256']==control['state_features_sha256']
        if args.stage=='fit':
            assert args.preflight_reference
            pre=json.loads((args.preflight_reference/'report.json').read_text())
            assert pre['stage']=='preflight' and not pre['local_bounded'] and pre['execution_checks_passed']
            assert pre['branch_restore_passed'] and pre['full_input_contract_passed']
            for k in ['protocol_sha256','entry_sha256','input_helper_sha256','input_identity_sha256']:
                assert pre[k]==r[k],k
            assert pre['peak_reserved_gib']<7.5 and pre['estimated_formal_seconds']<9000
        cfg=copy.deepcopy(p['config']);cfg['model']['name']=str(args.clip)
        seed(43);model=RobotAdapterModel(cfg).cuda().eval()
        tokenizer=CLIPTokenizer.from_pretrained(str(args.clip),local_files_only=True)
        visual,language,mask=tokens(args,CODE,p,model,tokenizer,items,old_items,checks,retained_indices,r,digest)
        model.vision_encoder.cpu();model.text_encoder.cpu();torch.cuda.empty_cache()
        clip_hash=[digest(model.vision_encoder),digest(model.text_encoder)]
        model.history_projection=torch.nn.Linear(1024,1024,bias=False).cuda()
        torch.nn.init.zeros_(model.history_projection.weight)
        for n,v in model.named_parameters():v.requires_grad_(n.startswith(PREFIXES))
        initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith(('vision_encoder.','text_encoder.'))}
        initial_hash=hashlib.sha256(''.join(n+tensor_sha(v) for n,v in sorted(initial.items())).encode()).hexdigest()
        add_zero_port(model);initial['state_projection.weight']=model.state_projection.weight.detach().cpu().clone()
        model.state_projection.weight.requires_grad_(True)
        initial_augmented=state_sha(trainable_copy(model))
        if not args.local_bounded:
            assert initial_hash==control['initial_shared_sha256'] and initial_augmented==control['augmented_initial_sha256']
            assert str(torch.__version__)==control['torch_version']
        r.update(initialization_seed=43,initial_shared_sha256=initial_hash,augmented_initial_sha256=initial_augmented,
            torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(),sampling_steps=100,sampling_seeds=[1101,1102,1103])
        trainables=[v for v in model.parameters() if v.requires_grad]
        assert all(n.startswith(PREFIXES)==v.requires_grad for n,v in model.named_parameters())
        optimizer=torch.optim.Adam(trainables,lr=1e-4)
        targets=torch.zeros((1876,16,8),device='cuda');current=torch.zeros((1876,1),device='cuda')
        for i in r['verified_row_indices']:targets[i]=items[i][3].cuda();current[i]=items[i][2].cuda()
        state=torch.from_numpy(features).cuda();valid=set(r['verified_row_indices'])
        def context(indices):
            assert set(indices.detach().cpu().tolist())<=valid
            now=model.adapter(visual[indices],language[indices],mask[indices])
            repeated=model.adapter(visual[indices],language[indices],mask[indices])
            return now+model.history_projection(repeated-now)+model.state_projection(state[indices])
        check_seconds=[];step_seconds=[]
        def check(arm,update,split):
            ids=([0,32,64,96] if args.stage=='preflight' else list(range(1836))) if split=='train' else list(range(1836,1876))
            begin=time.monotonic()
            with observation_guard(model):
                ctx=torch.cat([context(torch.tensor(ids[i:i+2],device='cuda')) for i in range(0,len(ids),2)])
                truth=targets[ids].cpu().numpy()
                pred,fit=evaluate(model,ctx,current[ids],truth,[1101,1102,1103],batch=2)
                m=metrics(pred,truth,[rows[i] for i in ids]);m['fit_gate']=fit
            check_seconds.append((time.monotonic()-begin)/len(ids))
            np.savez_compressed(args.output/f'{arm}_u{update:04d}_{split}.npz',predictions=pred,targets=truth,
                                contexts=ctx.cpu().numpy(),row_indices=np.array(ids))
            print('COVERAGE_METRIC',arm,update,split,'position_cm',m['path_position_cm'],'rotation_deg',m['path_rotation_deg'],flush=True)
            return m,pred
        def advance(arm,first,last):
            losses=[]
            with (args.output/f'{arm}_updates.jsonl').open('w') as log:
                for update in range(first,last+1):
                    begin=time.monotonic();model.train();ids=batches[update-1].cuda()
                    seed(430000+update);ctx=context(ids);seed(420000+update)
                    guess,truth,_=model.diffusion_loss(targets[ids],ctx,current[ids])
                    loss=torch.nn.functional.mse_loss(guess,truth);assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True);loss.backward();grads=gradient_norms(model)
                    assert all(v.grad is None for v in model.vision_encoder.parameters())
                    assert all(v.grad is None for v in model.text_encoder.parameters())
                    norm=torch.nn.utils.clip_grad_norm_(trainables,1.,error_if_nonfinite=True)
                    optimizer.step();torch.cuda.synchronize();step_seconds.append(time.monotonic()-begin)
                    value=float(loss);losses.append(value)
                    log.write(json.dumps(dict(update=update,learning_rate=optimizer.param_groups[0]['lr'],loss=value,
                        global_unclipped_gradient_norm=float(norm),unclipped_module_gradient_norms=grads))+'\n')
                    if update==first or update%256==0:
                        log.flush();print('COVERAGE_UPDATE',arm,update,'/',last,'loss',value,flush=True)
            return losses
        end=8 if args.stage=='preflight' else 8192;switch=4 if args.stage=='preflight' else 4096
        r.update(final_update=end,branch_update=switch)
        if args.stage=='preflight':check('common',0,'train')
        common=advance('common',1,switch)
        r['common_first_loss']=common[0];r['common_last_loss']=common[-1]
        if args.stage=='preflight':
            m,_=check('common',switch,'train');r['common_train']=m
            weights=trainable_copy(model);adam=cpu_copy(optimizer.state_dict());weight_hash=state_sha(weights);adam_hash=state_sha(adam)
            baseline=None
            for arm,lr in [('constant',1e-4),('restore_same_lr',1e-4),('reduced',1e-5)]:
                if arm!='constant':model.load_state_dict(weights,strict=False);optimizer.load_state_dict(cpu_copy(adam))
                assert state_sha(trainable_copy(model))==weight_hash and state_sha(optimizer.state_dict())==adam_hash
                for group in optimizer.param_groups:group['lr']=lr
                losses=advance(arm,switch+1,end);m,pred=check(arm,end,'train')
                r['runs'].append(dict(arm=arm,final_train=m,loss_first=losses[0],loss_last=losses[-1]))
                if arm=='constant':baseline=(state_sha(trainable_copy(model)),losses,pred.copy())
                elif arm=='restore_same_lr':
                    assert state_sha(trainable_copy(model))==baseline[0] and losses==baseline[1] and np.array_equal(pred,baseline[2])
                    r['branch_restore_passed']=True
                else:assert losses[0]==baseline[1][0] and state_sha(trainable_copy(model))!=baseline[0]
        else:
            for group in optimizer.param_groups:group['lr']=1e-5
            losses=advance('reduced',switch+1,end)
            train,_=check('reduced',end,'train');dev,_=check('reduced',end,'development')
            r['runs']=[dict(arm='reduced',final_train=train,final_development=dev,loss_first=losses[0],loss_last=losses[-1])]
            payload=dict(config=cfg,trainable_state_dict=trainable_copy(model),protocol=p,
                initial_shared_sha256=initial_hash,update=8192,original_state_protocol=json.loads((CODE.parent/'bridge_state_conditioning_v1/protocol.json').read_text()))
            checkpoint=args.output/'final_trainable.pt';torch.save(payload,checkpoint)
            restored=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state_sha(restored['trainable_state_dict'])==state_sha(payload['trainable_state_dict'])
            assert checkpoint.stat().st_size<1024**3
            r['checkpoints']=[dict(path=checkpoint.name,sha256=sha(checkpoint),bytes=checkpoint.stat().st_size,roundtrip_exact=True)]
        assert clip_hash==[digest(model.vision_encoder),digest(model.text_encoder)]
        assert all(torch.equal(model.state_dict()[n].cpu(),v) for n,v in initial.items() if not n.startswith(PREFIXES))
        r.update(execution_checks_passed=True,clip_unchanged=True,other_frozen_unchanged=True,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,warm_update_seconds=float(np.mean(step_seconds[1:])),
            estimated_formal_seconds=300+1.2*(float(np.mean(step_seconds[1:]))*8192+float(np.median(check_seconds))*1876),
            trainable_parameters=sum(v.numel() for v in trainables),optimization_operations=len(step_seconds),
            development_scored=(args.stage=='fit'),scientific_verdict='Execution success is not a generalization improvement verdict.')
    except Exception as error:r['error']=repr(error);raise
    finally:persist();print('TRAINING_COVERAGE_REPORT',args.output/'report.json',flush=True)


if __name__=='__main__':main()
