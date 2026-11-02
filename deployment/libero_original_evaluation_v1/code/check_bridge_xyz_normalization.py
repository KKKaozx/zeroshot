"""CPU numeric contract checks; no pretrained model or data downloads."""
import ast,json,os
from pathlib import Path
os.environ["USE_TF"]="0"
import torch
from bridge_xyz_normalization import NormalizedRobotAdapterModel as N
from models import RobotAdapterModel as B,diffusion_betas

def dummy(norm):
 m=N.__new__(N);torch.nn.Module.__init__(m)
 m.xyz_normalizer=norm;m.identity_xyz=norm['scale']==[1.,1.,1.] and norm['offset']==[0.,0.,0.]
 m.chunk_size=16;m.action_dim=8;m.decoder_type='diffusion';m.diffusion_prediction_type='sample'
 m.separate_gripper_head=True;m.trajectory_conditioned_gripper=True;m.gripper_target_mode='state'
 m.condition_on_current_gripper=False;m.num_diffusion_steps=100;m.clip_denoised=True;m.max_normalized_position=3.
 m.gripper_context_projection=torch.nn.Linear(4,4);m.gripper_pose_projection=torch.nn.Linear(7,4)
 m.gripper_time_embedding=torch.nn.Embedding(16,4);m.gripper_head=torch.nn.Linear(4,1)
 class Decoder(torch.nn.Module):
  def forward(self,x,t,c):return x*.03+c[:,:1,None]*.001
 m.diffusion_decoder=Decoder()
 b=diffusion_betas(100,'squaredcos_cap_v2');m.alphas=1-b;m.alpha_bars=torch.cumprod(m.alphas,0)
 prev=torch.cat([torch.ones(1),m.alpha_bars[:-1]])
 m.posterior_mean_x0=b*prev.sqrt()/(1-m.alpha_bars)
 m.posterior_mean_xt=(1-prev)*m.alphas.sqrt()/(1-m.alpha_bars)
 m.posterior_variance=b*(1-prev)/(1-m.alpha_bars)
 return m

def main():
 torch.set_num_threads(2);torch.manual_seed(11)
 stats=json.loads((Path(__file__).parent/'training_xyz_stats.json').read_text())['candidate_axis_limits_normalizer']
 norm={'scale':stats['scale_from_existing_xyz'],'offset':stats['offset']}
 m=dummy(norm);x=torch.randn(2,16,8);c=torch.randn(2,4)
 assert torch.allclose(m.convert(m.convert(x),inverse=True),x,atol=1e-6)
 z=(m.convert(x)+.1).detach().requires_grad_()
 loss=(m.convert(z,inverse=True)-x).square().mean()
 weighted=((z[...,:3]-m.convert(x)[...,:3]).square()/z.new_tensor(norm['scale']).square()).sum()+(z[...,3:]-x[...,3:]).square().sum()
 assert torch.allclose(loss,weighted/x.numel(),atol=1e-6)
 loss.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
 physical=m.convert(m.clip_encoded_clean(m.convert(x*8)[...,:7]),inverse=True)
 assert physical[...,:3].abs().max()<=3.000001 and physical[...,3:].abs().max()<=1.000001
 identity=dummy({'scale':[1.,1.,1.],'offset':[0.,0.,0.]})
 for force in [False,True]:
  identity.identity_xyz=not force
  torch.manual_seed(7);a=B.sample(identity,c);torch.manual_seed(7);b=identity.sample(c);assert torch.equal(a,b)
  torch.manual_seed(9);a=B.diffusion_loss(identity,x,c);rng_a=torch.random.get_rng_state()
  torch.manual_seed(9);b=identity.diffusion_loss(x,c);rng_b=torch.random.get_rng_state()
  differences=[float((v-w).detach().abs().max()) for v,w in zip(a,b)]
  print('IDENTITY_TRAIN_REPLAY',json.dumps({'force_custom_path':force,'max_absolute_difference_prediction_target_gripper':differences,'rng_state_equal':torch.equal(rng_a,rng_b),'torch':torch.__version__}),flush=True)
  assert torch.equal(rng_a,rng_b), 'Training noise consumption differs'
  if not force:
   assert all(torch.equal(v,w) for v,w in zip(a,b)), 'Original baseline must remain bitwise identical'
  else:
   # Artificially forcing the converted path adds inverse transforms and changes
   # gripper input strides; CPU GEMM kernels can differ by float32 roundoff.
   # Actual baseline never uses this path. Still reject substantive differences.
   for component,(v,w) in enumerate(zip(a,b)):
    torch.testing.assert_close(v,w,atol=1e-6,rtol=1e-6,msg=f'Custom identity component {component} outside float32 tolerance')
 seen=[];h=m.gripper_pose_projection.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
 out=m.diffusion_loss(x,c);assert torch.equal(out[1],x[...,:7]) and torch.equal(seen[-1],x[...,:7])
 seen.clear();torch.manual_seed(7);sample=m.sample(c)
 assert torch.isfinite(sample).all() and sample[...,:3].abs().max()<=3
 assert torch.equal(seen[-1],sample[...,:7])
 assert torch.allclose(sample[...,3:7].norm(dim=-1),torch.ones(2,16),atol=1e-6)
 h.remove()
 print('XYZ NUMERIC CONTRACT: PASSED; original identity bitwise replay, custom identity training float32 tolerance with identical RNG, inverse, physical-unit loss gradient, clipping, original gripper conditions')

if __name__=='__main__':main()
