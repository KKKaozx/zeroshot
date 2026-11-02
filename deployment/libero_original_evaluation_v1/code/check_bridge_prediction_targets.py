"""CPU checks of core sample/epsilon training and full DDPM contracts; no CLIP load."""
import ast,copy,json,os
from pathlib import Path
os.environ['USE_TF']='0'
import torch
import torch.nn as nn
from models import RobotAdapterModel as B,diffusion_betas
from diffusion_decoder import MultiscaleConditionalUnet1D

def dummy(target):
 m=B.__new__(B);nn.Module.__init__(m)
 m.chunk_size=16;m.action_dim=8;m.decoder_type='diffusion';m.diffusion_prediction_type=target
 m.separate_gripper_head=True;m.trajectory_conditioned_gripper=False;m.gripper_target_mode='state'
 m.gripper_head=nn.Linear(4,16);m.num_diffusion_steps=100;m.clip_denoised=True;m.max_normalized_position=3.
 m.diffusion_decoder=MultiscaleConditionalUnet1D(action_dim=7,context_dim=4,down_dims=[8,16,32],diffusion_step_embed_dim=8,kernel_size=5)
 b=diffusion_betas(100,'squaredcos_cap_v2');m.alphas=1-b;m.alpha_bars=torch.cumprod(m.alphas,0)
 prev=torch.cat([torch.ones(1),m.alpha_bars[:-1]])
 m.posterior_mean_x0=b*prev.sqrt()/(1-m.alpha_bars)
 m.posterior_mean_xt=(1-prev)*m.alphas.sqrt()/(1-m.alpha_bars)
 m.posterior_variance=b*(1-prev)/(1-m.alpha_bars)
 return m

def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 m=dummy('sample');e=copy.deepcopy(m);e.diffusion_prediction_type='epsilon'
 x=torch.randn(2,16,8);c=torch.randn(2,4);captures={}
 def hook(name):
  return lambda module,args:captures.update({name:(args[0].detach().clone(),args[1].detach().clone())})
 h=m.diffusion_decoder.register_forward_pre_hook(hook('sample'));he=e.diffusion_decoder.register_forward_pre_hook(hook('epsilon'))
 torch.manual_seed(123);sx=m.diffusion_loss(x,c);rng_s=torch.random.get_rng_state()
 torch.manual_seed(123);ex=e.diffusion_loss(x,c);rng_e=torch.random.get_rng_state()
 assert all(torch.equal(a,b) for a,b in zip(captures['sample'],captures['epsilon']))
 assert torch.equal(rng_s,rng_e) and torch.equal(sx[1],x[...,:7]) and torch.equal(sx[2],ex[2])
 torch.manual_seed(123);ts=torch.randint(0,100,(2,));noise=torch.randn_like(x[...,:7]);abar=m.alpha_bars[ts].reshape(-1,1,1)
 assert torch.equal(ex[1],noise)
 assert torch.equal(captures['sample'][0],abar.sqrt()*x[...,:7]+(1-abar).sqrt()*noise)
 for net,out in [(m,sx),(e,ex)]:
  loss=(out[0]-out[1]).square().mean()+.25*nn.functional.binary_cross_entropy_with_logits(out[2],(x[...,7]>0).float())
  loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in net.diffusion_decoder.parameters())
  assert sum(p.grad.abs().sum() for p in net.diffusion_decoder.parameters())>0
 h.remove();he.remove()
 # Equivalent known clean predictions test interpretation, not learned performance.
 clean=torch.randn(2,16,7)*.2
 class Fixed(nn.Module):
  def __init__(self,owner,prediction):super().__init__();object.__setattr__(self,'owner',owner);self.prediction=prediction
  def forward(self,xt,ts,context):
   if self.prediction=='sample':return clean
   a=self.owner.alpha_bars[ts].reshape(-1,1,1)
   return (xt-a.sqrt()*clean)/(1-a).sqrt()
 m.diffusion_decoder=Fixed(m,'sample');e.diffusion_decoder=Fixed(e,'epsilon')
 torch.manual_seed(9);sp=m.sample(c);rng_s=torch.random.get_rng_state()
 torch.manual_seed(9);ep=e.sample(c);rng_e=torch.random.get_rng_state()
 torch.testing.assert_close(sp,ep,atol=1e-6,rtol=1e-6)
 assert torch.equal(rng_s,rng_e) and torch.isfinite(sp).all() and torch.isfinite(ep).all()
 assert sp[...,:3].abs().max()<=3 and ep[...,:3].abs().max()<=3
 assert torch.allclose(ep[...,3:7].norm(dim=-1),torch.ones(2,16),atol=1e-6)
 print('PREDICTION TARGET CPU CHECK PASSED',json.dumps({'torch':torch.__version__,'training_noisy_inputs_and_timesteps_identical':True,'training_rng_identical':True,'correct_x0_and_epsilon_targets':True,'both_decoder_gradients_finite_nonzero':True,'full100step_known_clean_prediction_max_difference':float((sp-ep).abs().max()),'sampling_rng_identical':True,'clip_and_quaternion_constraints_passed':True}),flush=True)

if __name__=='__main__':main()
