"""Fixed full-training-window x0 fit; frozen contexts/gripper, development-only evaluation."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from cached_diffusion import RobotAdapterModel, policy_loss, metrics


def save(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


class Sampler:
    def __init__(self):
        self.rng=np.random.default_rng(42)
        self.order=self.rng.permutation(316); self.cursor=0
        self.frequency=np.zeros(316,dtype=np.int64)

    def take(self):
        chunks=[]; needed=64
        while needed:
            n=min(needed,316-self.cursor)
            chunks.append(self.order[self.cursor:self.cursor+n]); self.cursor+=n; needed-=n
            if self.cursor==316:
                self.cursor=0; self.order=self.rng.permutation(316)
        result=np.concatenate(chunks)
        np.add.at(self.frequency,result,1)
        return result


def gripper_checks(m):
    return dict(open_recall=m['open_recall'] is not None and m['open_recall']>=.95,
        closed_recall=m['closed_recall'] is not None and m['closed_recall']>=.95,
        open_to_closed=m['open_to_closed_correct']==m['open_to_closed_pairs'],
        closed_to_open=m['closed_to_open_correct']==m['closed_to_open_pairs'])


def training_checks(m):
    return dict(position=m['position_error_cm']<=.5,rotation=m['rotation_error_deg']<=5.,**gripper_checks(m))


def pose_beats_baseline(m,b):
    return m['position_error_cm']<b['position_error_cm'] and m['rotation_error_deg']<b['rotation_error_deg']


def summary(prediction,target,keys,starts,teacher=None):
    total=metrics(prediction,target,teacher)
    total['by_episode']={}
    for key in np.unique(keys):
        mask=keys==key
        total['by_episode'][str(key)]=metrics(prediction[mask],target[mask])
    events={}
    for j in range(len(target)):
        truth=target[j,:,7]>=0; guess=prediction[j,:,7]>=0
        for t in np.flatnonzero(truth[:-1]!=truth[1:]):
            direction='open_to_closed' if truth[t] else 'closed_to_open'
            event=(str(keys[j]),int(starts[j]+t),direction)
            events.setdefault(event,[]).append(bool(np.all(guess[t:t+2]==truth[t:t+2])))
    total['unique_command_transition_views']={direction:dict(
        events=sum(k[2]==direction for k in events),
        all_views_correct=sum(all(v) for k,v in events.items() if k[2]==direction),
        any_view_correct=sum(any(v) for k,v in events.items() if k[2]==direction))
        for direction in ['open_to_closed','closed_to_open']}
    return total


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    for name,digest in json.loads((root/'integrity.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    seed(42)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    bundle=torch.load(root/'full_context.pt',map_location='cpu',weights_only=False)
    protocol=json.loads((root/'provenance.json').read_text())
    initial=torch.load(root/'shared_initial_weights.pt',map_location='cpu',weights_only=False)
    model=RobotAdapterModel(bundle['config']).cuda()
    model.load_state_dict(initial,strict=True)
    digest=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    assert digest.hexdigest()==protocol['initial_state_sha256']
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('diffusion_decoder.'))
    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    data={part:{k:v.cuda() if isinstance(v,torch.Tensor) else np.array(v) for k,v in d.items()}
          for part,d in bundle['data'].items()}
    train=data['train']; validation=data['validation']
    assert train['targets'].shape==(316,16,8) and validation['targets'].shape==(55,16,8)
    assert not set(train['indices']) & set(validation['indices'])
    baselines={}
    for part,d in data.items():
        t=d['targets'].cpu().numpy().astype(np.float64)
        p=np.zeros_like(t); p[:,:,6]=1; p[:,:,7]=-1
        baselines[part]=summary(p,t,d['episode_keys'],d['starts'])
    protocol['smoke_only']=args.smoke; protocol['actual_steps']=5 if args.smoke else 9875
    protocol['baselines']=baselines
    for part,d in data.items():
        xyz=d['targets'][:,:,:3]
        protocol[part+'_clipping']=dict(outside_components=int((xyz.abs()>3).sum()),
            position_error_floor_cm=float((xyz-xyz.clamp(-3,3)).norm(dim=-1).mean()*10))
    save(args.output_dir/'experiment.json',protocol)
    optimizer=torch.optim.Adam(model.diffusion_decoder.parameters(),lr=3e-4,weight_decay=0)
    history=[]

    @torch.no_grad()
    def evaluate(step):
        model.eval(); reports=[]; payload={}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for value in [1101,1102,1103]:
                scores={}
                for part,d in data.items():
                    seed(value); predicted=[]; teacher=[]
                    for start in range(0,len(d['indices']),64):
                        s=slice(start,start+64)
                        predicted.append(model.sample(d['context'][s],d['current'][s]).cpu().numpy().astype(np.float64))
                        teacher.append(model.predict_gripper_logits(d['context'][s],d['targets'][s,:,:7],
                            d['current'][s]).cpu().numpy())
                    p=np.concatenate(predicted); t=d['targets'].cpu().numpy().astype(np.float64)
                    scores[part]=summary(p,t,d['episode_keys'],d['starts'],np.concatenate(teacher))
                    payload.setdefault(part,[]).append(p)
                    m=scores[part]
                    print(f"step={step} seed={value} {part} n={len(p)} position={m['position_error_cm']:.4f}cm "
                        f"rotation={m['rotation_error_deg']:.3f}deg balance={m['balanced_accuracy']:.3f} "
                        f"switches={m['open_to_closed_correct']}/{m['open_to_closed_pairs']},"
                        f"{m['closed_to_open_correct']}/{m['closed_to_open_pairs']}",flush=True)
                    if part=='validation':
                        for key,episode in m['by_episode'].items():
                            print(f"  validation episode={key} n={episode['windows']} "
                                f"position={episode['position_error_cm']:.3f}cm rotation={episode['rotation_error_deg']:.3f}deg "
                                f"open_recall={episode['open_recall']}",flush=True)
                tr=training_checks(scores['train']); va=scores['validation']; b=baselines['validation']
                vc=dict(overall_pose=pose_beats_baseline(va,b),
                    all_three_episode_poses=all(pose_beats_baseline(m,b['by_episode'][key]) for key,m in va['by_episode'].items()),
                    **gripper_checks(va))
                reports.append(dict(seed=value,partitions=scores,train_checks=tr,development_checks=vc))
        row=dict(step=step,samples=reports);history.append(row)
        save(args.output_dir/'history.json',dict(baselines=baselines,evaluations=history))
        if step==protocol['actual_steps']:
            save(args.output_dir/'final_metrics.json',row)
            decision=dict(smoke_only=args.smoke,checkpoint_step=step,
                train_pose_fit_passed=all(r['train_checks']['position'] and r['train_checks']['rotation'] for r in reports) if not args.smoke else None,
                train_chain_passed=all(all(r['train_checks'].values()) for r in reports) if not args.smoke else None,
                development_passed=all(all(r['development_checks'].values()) for r in reports) if not args.smoke else None,
                samples=[dict(seed=r['seed'],train_checks=r['train_checks'],development_checks=r['development_checks']) for r in reports],
                frozen_gripper=True,test_targets_used=False,scope=protocol['scope'])
            save(args.output_dir/'diagnostic_decision.json',decision)
            npz=dict(seeds=np.array([1101,1102,1103]))
            for part,d in data.items():
                npz[part+'_predictions']=np.stack(payload[part])
                for key in ['targets','indices','episode_keys','starts']:
                    npz[part+'_'+key]=d[key].cpu().numpy() if isinstance(d[key],torch.Tensor) else d[key]
            np.savez_compressed(args.output_dir/'final_predictions.npz',**npz)
            torch.save(dict(config=bundle['config'],state_dict=model.state_dict(),steps=step,
                dataset_identity=bundle['dataset_identity'],not_a_deployable_checkpoint=True),args.output_dir/'final_head.pt')
        model.train()

    evaluate(0);sampler=Sampler(); started=time.monotonic()
    for step in range(1,protocol['actual_steps']+1):
        selected=sampler.take(); batch={k:train[k][selected] for k in ['targets','context','current','masks']}
        optimizer.zero_grad(set_to_none=True)
        output=model.diffusion_loss(batch['targets'],batch['context'],batch['current'])
        total,pose,grip=policy_loss(model,output,batch['targets'],torch.nn.MSELoss(),
            current_grippers=batch['current'],supervision_masks=batch['masks'])
        assert torch.isfinite(total)
        total.backward(); norm=torch.nn.utils.clip_grad_norm_(model.diffusion_decoder.parameters(),1.)
        assert torch.isfinite(norm);optimizer.step()
        if step==1 or step%250==0:
            print(f'update={step}/{protocol["actual_steps"]} x0_pose_loss={float(pose.detach()):.6f} '
                  f'elapsed={time.monotonic()-started:.1f}s',flush=True)
        if step%1000==0 or step==protocol['actual_steps']:evaluate(step)
    assert all(torch.equal(p,frozen[name]) for name,p in model.named_parameters() if name in frozen)
    if not args.smoke:assert np.all(sampler.frequency==2000)
    save(args.output_dir/'training_coverage.json',dict(train_windows=316,train_action_targets=5056,
        validation_windows_evaluated=55,validation_action_targets_evaluated=880,updates=protocol['actual_steps'],
        draws=int(sampler.frequency.sum()),min_draws=int(sampler.frequency.min()),max_draws=int(sampler.frequency.max()),
        indices=train['indices'].tolist(),draws_by_window=sampler.frequency.tolist(),
        frozen_gripper_unchanged=True,validation_used_for_optimizer=False,test_targets_used=False))
    print('FULL X0 SMOKE COMPLETED' if args.smoke else 'FULL TRAIN X0 FIT COMPLETED',flush=True)
    print('Results:',args.output_dir,flush=True)


if __name__=='__main__':main()
