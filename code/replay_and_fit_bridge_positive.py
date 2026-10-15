"""Local train-only positive control: retained compact replay and fresh multiscale fit.
No CLIP/data download, no validation targets, no simulator, no joint adapter training.
"""
import argparse,copy,hashlib,importlib.util,io,json,os,sys,time,zipfile
from pathlib import Path
os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
from models import RobotAdapterModel as Production, diffusion_betas
from diffusion_decoder import MultiscaleConditionalUnet1D
from evaluate_bridge_endpoint_metrics import pose_errors

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def seed(n):np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def head(config,state):
 spec=importlib.util.spec_from_file_location('archived_cached_head',ROOT/'diagnostics/full_x0/cached_diffusion.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 old=module.RobotAdapterModel(config);old.load_state_dict(state,strict=True)
 # Bypass only full CLIP/Adapter construction: supply already frozen real contexts.
 # All executed sample, diffusion_loss and gripper methods resolve on production class.
 m=Production.__new__(Production);torch.nn.Module.__init__(m)
 for k,v in vars(old).items():
  if not k.startswith('_'):setattr(m,k,copy.deepcopy(v))
 for k,v in old.named_children():m.add_module(k,copy.deepcopy(v))
 for k,v in old.named_buffers(recurse=False):m.register_buffer(k,v.clone())
 m.diffusion_architecture='compact';m.load_state_dict(state,strict=True)
 assert torch.equal(m.betas,diffusion_betas(100,'squaredcos_cap_v2'))
 return m

def score(pred,target):
 positions=[];rotations=[]
 for sample in pred:
  p,r=pose_errors(sample,target);positions.append(p);rotations.append(r)
 p=np.stack(positions);r=np.stack(rotations)
 window_good=(p.mean(axis=2)<=.5)&(r.mean(axis=2)<=5)
 static=np.zeros_like(target);static[...,6]=1
 sp,sr=pose_errors(static,target)
 endpoint_p=p[:,:,-1].mean(axis=1);endpoint_r=r[:,:,-1].mean(axis=1)
 passed=bool(np.all(p.mean(axis=(1,2))<=.5) and np.all(r.mean(axis=(1,2))<=5)
             and np.mean(window_good.all(axis=0))>=.9
             and np.all(endpoint_p<=1.) and np.all(endpoint_r<=10.))
 return {'position_cm':float(p.mean()),'rotation_deg':float(r.mean()),
         'endpoint_position_cm':float(p[:,:,-1].mean()),'endpoint_rotation_deg':float(r[:,:,-1].mean()),
         'seed_position_cm':p.mean(axis=(1,2)).tolist(),'seed_rotation_deg':r.mean(axis=(1,2)).tolist(),
         'windows_all_three_seeds_pass':int(window_good.all(axis=0).sum()),'windows':len(target),
         'static_position_cm':float(sp.mean()),'static_rotation_deg':float(sr.mean()),'pose_fit_passed':passed}

@torch.inference_mode()
def evaluate(m,ctx,current,targets,seeds,batch=64):
 m.eval();values=[]
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  for n in seeds:
   seed(n);rows=[]
   for start in range(0,len(ctx),batch):
    value=m.sample(ctx[start:start+batch],current[start:start+batch])
    if not torch.isfinite(value).all():raise ValueError('Non-finite production sample')
    rows.append(value.cpu().numpy())
   values.append(np.concatenate(rows))
 values=np.stack(values)
 return values,score(values,targets)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--archive',type=Path,default=ROOT.parent/'GPU cluster/bridge_diffusion_full_fit_v1.zip')
 ap.add_argument('--checkpoint',type=Path,default=ROOT.parent/'GPU cluster/diffusion-full-x0-186047/final_head.pt')
 a=ap.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for this bounded local test')
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True,warn_only=True)
 t0=time.monotonic();source_hashes={name:sha(ROOT/'code'/name) for name in ['models.py','diffusion_decoder.py','replay_and_fit_bridge_positive.py','evaluate_bridge_endpoint_metrics.py']}
 with zipfile.ZipFile(a.archive) as z:
  integrity=json.loads(z.read('bridge_diffusion_full_fit_v1/integrity.json'))
  for name,expected in integrity.items():
   if hashlib.sha256(z.read('bridge_diffusion_full_fit_v1/'+name)).hexdigest()!=expected:raise ValueError('Archive changed: '+name)
  b=torch.load(io.BytesIO(z.read('bridge_diffusion_full_fit_v1/full_context.pt')),map_location='cpu',weights_only=False)
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if payload['dataset_identity']!=b['dataset_identity'] or payload['config']!=b['config']:raise ValueError('Cache/checkpoint mismatch')
 train=b['data']['train'];assert train['targets'].shape==(316,16,8)
 assert bool(torch.all(train['masks'][...,:7]==1))
 del b['data']['validation'] # Never evaluate or optimize a development target here.
 ctx=train['context'].cuda();current=train['current'].cuda();targets=train['targets'].numpy()
 m=head(b['config'],payload['state_dict']).cuda();before=digest(m);seeds=[1101,1102,1103]
 prediction,replay=evaluate(m,ctx,current,targets,seeds)
 assert digest(m)==before
 np.savez_compressed(a.output/'replay_train_predictions.npz',predictions=prediction,targets=targets,seeds=seeds,
                    episode_keys=np.asarray(train['episode_keys']),starts=np.asarray(train['starts']))
 print('RETAINED_CURRENT_SAMPLER_REPLAY',json.dumps(replay),flush=True)
 # First window in each of first eight distinct training demonstrations, no metric-based selection.
 selected=[];episodes=set()
 for i,key in enumerate(train['episode_keys']):
  if key not in episodes:selected.append(i);episodes.add(key)
  if len(selected)==8:break
 assert len(selected)==8
 selection=[{'cache_index':i,'original_dataset_index':int(train['indices'][i]),'episode':str(train['episode_keys'][i]),'start':int(train['starts'][i])} for i in selected]
 smallctx=ctx[selected];smallcurrent=current[selected];smalltarget=train['targets'][selected].cuda();smallnp=smalltarget.cpu().numpy()
 seed(42)
 m.diffusion_decoder=MultiscaleConditionalUnet1D(action_dim=7,context_dim=1024,chunk_size=16,down_dims=(256,512,1024),diffusion_step_embed_dim=128,kernel_size=5,num_steps=100).cuda()
 m.diffusion_architecture='multiscale'
 for k,p in m.named_parameters():p.requires_grad_(k.startswith('diffusion_decoder.'))
 frozen_before=digest_frozen(m);initial_decoder=digest(m.diffusion_decoder)
 optimizer=torch.optim.Adam(m.diffusion_decoder.parameters(),lr=3e-4,weight_decay=0)
 torch.cuda.reset_peak_memory_stats();history=[];updates=0
 def check(step):
  pred,metrics=evaluate(m,smallctx,smallcurrent,smallnp,seeds,batch=8)
  row={'update':step,'elapsed_seconds':time.monotonic()-t0,**metrics};history.append(row)
  (a.output/'fit_history.json').write_text(json.dumps(history,indent=2)+'\n')
  print('CURRENT_MULTISCALE_FIT',json.dumps(row),flush=True)
  return pred,metrics
 finalpred,final=check(0);durations=[];first_grad=None
 for update in range(1,2001):
  m.train();seed(420000+update)
  torch.cuda.synchronize();start=time.monotonic()
  prediction,truth,_=m.diffusion_loss(smalltarget,smallctx,smallcurrent)
  loss=torch.nn.functional.mse_loss(prediction,truth)
  if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
  optimizer.zero_grad(set_to_none=True);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(m.diffusion_decoder.parameters(),1.,error_if_nonfinite=True)
  if first_grad is None:first_grad=float(norm)
  optimizer.step();updates=update
  torch.cuda.synchronize();durations.append(time.monotonic()-start)
  if update==1 or update%100==0:print('POSITIVE_UPDATE',update,'/2000','loss',float(loss),'elapsed',round(time.monotonic()-t0,1),flush=True)
  if update in [500,1000,2000]:
   finalpred,final=check(update)
   if final['pose_fit_passed']:break
 assert frozen_before==digest_frozen(m)
 final_decoder=digest(m.diffusion_decoder)
 assert initial_decoder!=final_decoder
 np.savez_compressed(a.output/'fresh_multiscale_predictions.npz',predictions=finalpred,targets=smallnp,seeds=seeds)
 if any(sha(ROOT/'code'/n)!=h for n,h in source_hashes.items()):raise ValueError('Source changed during execution')
 report={'stage':'local_train_only_positive_control','execution_checks_passed':True,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
 'train_only':True,'new_dataset_downloads':False,'clip_adapter_executed':False,'gripper_trained':False,'development_targets_used':False,'test_targets_used':False,
 'retained_compact_replay':replay,'fresh_current_multiscale_fit':final,'selection':selection,'updates':updates,'draws_per_window':updates,
 'batch_size':8,'lr':3e-4,'optimizer':'Adam','weight_decay':0,'learning_rate_schedule':'constant','prediction_type':'sample',
 'decoder_parameters':sum(p.numel() for p in m.diffusion_decoder.parameters()),'frozen_nondecoder_unchanged':True,
 'first_gradient_norm':first_grad,'decoder_initial_sha256':initial_decoder,'decoder_final_sha256':final_decoder,
 'source_sha256':source_hashes,'archive_sha256':sha(a.archive),'retained_checkpoint_sha256':sha(a.checkpoint),
 'sampling_steps':100,'sampling_seeds':seeds,'pose_pass_criteria':{'all_seed_mean_position_cm':.5,'all_seed_mean_rotation_deg':5.,'windows_passing_every_seed_fraction':.9,'all_seed_endpoint_position_cm':1.,'all_seed_endpoint_rotation_deg':10.},
 'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'mean_update_seconds':float(np.mean(durations[1:])),'elapsed_seconds':time.monotonic()-t0,
 'limits':['Same training windows evaluated; no generalization claim.','Real cached adapter contexts bypass the raw dataloader and CLIP/Adapter forward path.','Fresh decoder uses a deliberate memorization recipe, not the previous joint training optimizer.','Frozen gripper output is not evidence of new gripper learning.','No new checkpoint is retained to limit local disk usage; script/data/seed hashes bind reproduction.']}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('POSITIVE CONTROL COMPLETE',a.output/'report.json',flush=True)

def digest_frozen(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  if not k.startswith('diffusion_decoder.'):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
if __name__=='__main__':main()
