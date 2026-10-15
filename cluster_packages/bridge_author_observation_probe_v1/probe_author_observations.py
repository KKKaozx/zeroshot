"""Frozen finalEMA input sensitivity; replay original predictions before interpretation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from probe_helpers import observation_batch,replace_observation,rng_state,restore_rng,paired_difference

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('project','source','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();code=Path(__file__).resolve().parent;p=read(code/'protocol.json')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    r=dict(stage='frozen_author_observation_intervention',execution_passed=False,trained_updates=0,
        origin_job=203699,primary='fixed finalEMA',simulation_executed=False,reserved_test_used=False,
        protocol_sha256=sha(code/'protocol.json'),mixed_observation_accuracy_evaluated=False)
    def persist():
        r['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    try:
        pre=args.project/'bridge_author_preflight_v1';ref=args.project/'bridge_author_reference_v1'
        original=args.project/'runs/bridge-author-reference-203699';cache=args.project/'runs/bridge-author-preflight-203498/inputs'
        for name,h in p['required_file_sha256'].items():assert sha(args.project/name)==h,(name,'source/input identity')
        author=read(pre/'protocol.json')
        assert all(sha(args.source/n)==h for n,h in author['author_source_sha256'].items())
        checkpoint=original/'ema_final.pt'
        assert sha(checkpoint)==p['checkpoint_sha256']
        sys.path.insert(0,str(args.source));sys.path.insert(1,str(pre));sys.path.insert(2,str(ref))
        import torch
        import hydra
        from omegaconf import OmegaConf
        from bridge_author_dataset import BridgeAuthorDataset
        from reference_metrics import summarize
        assert torch.cuda.is_available() and torch.__version__==p['torch_version']
        device=torch.device('cuda:0');torch.set_num_threads(4);torch.set_grad_enabled(False)
        data=BridgeAuthorDataset(cache,pre/'protocol.json');arrays=data.arrays
        payload=torch.load(checkpoint,map_location='cpu')
        cfg=OmegaConf.load(original/'runtime_config.yaml')
        assert payload['updates']==8192 and payload['initialization_seed']==43 and payload['primary']
        assert payload['policy_config']==OmegaConf.to_container(cfg.policy,resolve=True)
        assert payload['training_protocol']==read(ref/'training_protocol.json')
        policy=hydra.utils.instantiate(OmegaConf.create(payload['policy_config']))
        normalizer=data.get_normalizer()
        policy.set_normalizer(normalizer);policy.load_state_dict(payload['policy_state'],strict=True)
        assert all(torch.equal(v,policy.normalizer.state_dict()[k]) for k,v in normalizer.state_dict().items())
        assert all(torch.equal(v,payload['policy_state'][k]) for k,v in policy.state_dict().items())
        policy.to(device).eval()
        assert policy.num_inference_steps==100 and policy.noise_scheduler.config.prediction_type=='epsilon'
        assert policy.noise_scheduler.config.num_train_timesteps==100
        assert policy.noise_scheduler.config.clip_sample and policy.n_obs_steps==1 and policy.horizon==16
        r.update(torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
            checkpoint_sha256=p['checkpoint_sha256'],sampling_seeds=p['sampling_seeds'],
            normalizer_matches_training_statistics=True,source_verified=True)
        expected=np.load(original/'ema_development.npz',allow_pickle=False)
        indices=np.arange(128,168);assert np.array_equal(expected['row_indices'],indices)
        assert np.array_equal(expected['targets'],arrays['action'][indices])
        donor_map={x['window']:x['donor_window'] for x in p['donors']}
        windows=np.array(sorted(donor_map));donors=np.array([donor_map[int(i)] for i in windows])
        assert len(windows)==9 and all(author['rows'][int(i)]['task']==author['rows'][int(j)]['task']=='take broccoli out of pan' for i,j in zip(windows,donors))
        assert (donors<128).all() and (windows>=128).all()
        replay=[];before_states={};all_calls=0
        def sample(obs):
            nonlocal all_calls
            calls=[]
            def record(module,inputs,output):calls.append(int(inputs[1]))
            hook=policy.model.register_forward_hook(record)
            try:answer=policy.predict_action(obs)['action_pred'].detach().cpu().numpy()
            finally:hook.remove()
            assert calls==list(range(99,-1,-1)),'Native100-step schedule changed'
            all_calls+=1;assert answer.shape==(8,16,7) and np.isfinite(answer).all()
            return answer
        torch.cuda.reset_peak_memory_stats()
        # First fully replay all40 controls with the unchanged native batch and seed order.
        for seed in p['sampling_seeds']:
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);blocks=[]
            for start in range(0,40,8):
                ix=indices[start:start+8];obs=observation_batch(torch,arrays,ix,device)
                before_states[(seed,start)]=rng_state(torch,device)
                blocks.append(sample(obs))
            replay.append(np.concatenate(blocks))
            print('AUTHOR_OBS_REPLAY',seed,'40/40',flush=True)
        replay=np.stack(replay)
        r['original_replay_max_absolute_difference']=float(np.max(np.abs(replay-expected['predictions'])))
        r['replay_tolerance']=p['replay_tolerance']
        assert r['original_replay_max_absolute_difference']<=p['replay_tolerance'],'Original replay mismatch; do not interpret intervention'
        r['original_full40_metrics']=summarize(replay,expected['targets']);persist()
        outputs={case:[] for case in ('original','image_only','state_only','both')}
        untouched_max=0.
        for draw,seed in enumerate(p['sampling_seeds']):
            draw_outputs={case:[] for case in outputs}
            for start in range(0,40,8):
                ix=indices[start:start+8];selected=np.array([j for j,i in enumerate(ix) if int(i) in donor_map])
                if not len(selected):continue
                obs=observation_batch(torch,arrays,ix,device)
                untouched=np.array([j for j in range(8) if j not in selected])
                native=replay[draw,start:start+8]
                for case in outputs:
                    changed=replace_observation(torch,obs,arrays,ix,donor_map,case)
                    restore_rng(torch,device,before_states[(seed,start)])
                    answer=sample(changed)
                    if len(untouched):untouched_max=max(untouched_max,float(np.max(np.abs(answer[untouched]-native[untouched]))))
                    if case=='original':assert np.max(np.abs(answer-native))<=p['replay_tolerance']
                    draw_outputs[case].append(answer[selected])
            for case in outputs:outputs[case].append(np.concatenate(draw_outputs[case]))
            print('AUTHOR_OBS_INTERVENTION',seed,'9/9 all4 contexts',flush=True)
        outputs={k:np.stack(v) for k,v in outputs.items()}
        assert all(v.shape==(3,9,16,7) for v in outputs.values())
        assert np.max(np.abs(outputs['original']-replay[:,windows-128]))<=p['replay_tolerance']
        assert untouched_max<=p['replay_tolerance'],'Nonselected rows changed; paired interpretation invalid'
        r['unchanged_batch_rows_max_absolute_difference']=untouched_max
        r['output_changes']={k:paired_difference(v,outputs['original']) for k,v in outputs.items() if k!='original'}
        r['valid_context_quality']=dict(original=summarize(outputs['original'],arrays['action'][windows]),
            both_donor=summarize(outputs['both'],arrays['action'][donors]))
        np.savez_compressed(args.output/'predictions.npz',**outputs,
            full40_replay=replay,original_targets=arrays['action'][windows],donor_targets=arrays['action'][donors],
            window_indices=windows,donor_indices=donors)
        assert all(torch.equal(v.cpu(),payload['policy_state'][k]) for k,v in policy.state_dict().items())
        r.update(execution_passed=True,parameters_unchanged=True,sampler_calls=all_calls,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            scientific_verdict='Sensitivity under fixed weights; crossed contexts are not valid task-success or improvement tests.')
        assert all_calls==39 and r['peak_reserved_gib']<20
        assert sum(x.stat().st_size for x in args.output.iterdir() if x.is_file())<5*2**20
        for case,value in r['output_changes'].items():
            print('AUTHOR_OBS_CHANGE',case,'cm',round(value['position_cm'],4),'deg',round(value['rotation_deg'],4),flush=True)
    except Exception as e:r['error']=repr(e);raise
    finally:persist();print('AUTHOR OBSERVATION PROBE REPORT',args.output/'report.json',flush=True)

if __name__=='__main__':main()
