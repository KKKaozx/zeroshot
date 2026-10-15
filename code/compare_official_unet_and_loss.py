"""Direct official/project U-Net and loss comparison; CPU only, no weights/data."""
import ast,hashlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from typing import Tuple
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baselines/diffusion_policy_data_reference'))
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_decoder import MultiscaleConditionalUnet1D

def official_key(name):
 name=name.replace('time_mlp.','diffusion_step_encoder.').replace('.condition.','.cond_encoder.').replace('.residual.','.residual_conv.')
 if name.startswith('final.'):name='final_conv.'+name[len('final.'):]
 parts=name.split('.')
 if parts[0] in ('down_modules','up_modules') and parts[2]=='2':
  parts.insert(3,'conv');name='.'.join(parts)
 return name

def extract_function(path,name,globals_):
 tree=ast.parse(path.read_text(encoding="utf-8"));node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 node.decorator_list=[]
 for arg in node.args.args:arg.annotation=None
 node.returns=None
 namespace=dict(globals_);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]

def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 official=lambda dims:ConditionalUnet1D(input_dim=7,global_cond_dim=1024,down_dims=dims,diffusion_step_embed_dim=128,kernel_size=5,n_groups=8,cond_predict_scale=True)
 project=lambda dims:MultiscaleConditionalUnet1D(action_dim=7,context_dim=1024,down_dims=dims,diffusion_step_embed_dim=128,kernel_size=5)
 with torch.device('meta'):
  large_o=official([256,512,1024]);large_p=project([256,512,1024])
 mapping={k:official_key(k) for k in large_p.state_dict()}
 assert set(mapping.values())==set(large_o.state_dict())
 assert all(large_p.state_dict()[k].shape==large_o.state_dict()[v].shape for k,v in mapping.items())
 actual_parameters=sum(p.numel() for p in large_p.parameters());del large_o,large_p
 # Smaller equal widths exercise the exact same classes/operators while limiting local RAM.
 o=official([32,64,128]);p=project([32,64,128]);mapped={official_key(k):v for k,v in p.state_dict().items()};o.load_state_dict(mapped,strict=True)
 x=torch.randn(2,16,7);t=torch.tensor([0,99]);c=torch.randn(2,1024)
 xp=x.clone().requires_grad_();xo=x.clone().requires_grad_();cp=c.clone().requires_grad_();co=c.clone().requires_grad_()
 yp=p(xp,t,cp);yo=o(xo,t,global_cond=co)
 torch.testing.assert_close(yp,yo,atol=1e-6,rtol=1e-6)
 yp.square().mean().backward();yo.square().mean().backward()
 torch.testing.assert_close(xp.grad,xo.grad,atol=1e-7,rtol=1e-5)
 torch.testing.assert_close(cp.grad,co.grad,atol=1e-7,rtol=1e-5)
 op=dict(o.named_parameters());parameter_grad_max=0.
 for name,value in p.named_parameters():
  other=op[official_key(name)]
  torch.testing.assert_close(value.grad,other.grad,atol=1e-7,rtol=1e-5)
  parameter_grad_max=max(parameter_grad_max,float((value.grad-other.grad).abs().max()))
 unet={'matched_actual_architecture_state_keys':len(mapping),'matched_actual_parameters':actual_parameters,
       'actual_width_shape_mapping_passed':True,'numeric_test_widths':[32,64,128],
       'forward_max_abs_difference':float((yp-yo).detach().abs().max()),
       'action_input_gradient_max_abs_difference':float((xp.grad-xo.grad).abs().max()),
       'context_gradient_max_abs_difference':float((cp.grad-co.grad).abs().max()),
       'parameter_gradient_max_abs_difference':parameter_grad_max}
 # Execute actual official compute_loss body with synthetic observation/scheduler objects.
 official_loss=extract_function(ROOT/'baselines/diffusion_policy_data_reference/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py','compute_loss',{'torch':torch,'F':F,'reduce':reduce,'dict_apply':lambda d,fn:{k:fn(v) for k,v in d.items()}})
 project_loss=extract_function(ROOT/'code/train.py','policy_loss',{'torch':torch,'nn':nn,'Tuple':Tuple})
 class Identity:
  def normalize(self,v):return v
  def __getitem__(self,k):return self
 from models import diffusion_betas
 abar=torch.cumprod(1-diffusion_betas(100,'squaredcos_cap_v2'),0)
 loss_checks=[]
 for prediction_type in ['sample','epsilon']:
  scheduler=types.SimpleNamespace(config=types.SimpleNamespace(num_train_timesteps=100,prediction_type=prediction_type))
  saved={}
  def add_noise(traj,noise,ts):
   saved.update(noise=noise,ts=ts)
   a=abar[ts].reshape(-1,1,1);return a.sqrt()*traj+(1-a).sqrt()*noise
  scheduler.add_noise=add_noise
  pred=torch.randn(2,16,7,requires_grad=True);truth=torch.randn(2,16,7)
  obj=types.SimpleNamespace(normalizer=Identity(),obs_as_global_cond=True,n_obs_steps=2,obs_encoder=lambda obs:obs['signal'],mask_generator=lambda shape:torch.zeros(shape,dtype=torch.bool),noise_scheduler=scheduler,model=lambda *args,**kw:pred)
  lo=official_loss(obj,{'obs':{'signal':torch.randn(2,2,4)},'action':truth})
  stub=types.SimpleNamespace(decoder_type='diffusion',gripper_loss_weight=.25,diffusion_prediction_type=prediction_type)
  actions=torch.cat([truth,torch.ones(2,16,1)],dim=-1)
  target=truth if prediction_type=='sample' else saved['noise']
  lp,_,_=project_loss(stub,(pred,target),actions,nn.MSELoss(),supervision_masks=torch.ones_like(actions))
  go=torch.autograd.grad(lo,pred,retain_graph=True)[0];gp=torch.autograd.grad(lp,pred)[0]
  torch.testing.assert_close(lo,lp,atol=1e-6,rtol=1e-6);torch.testing.assert_close(go,gp,atol=1e-7,rtol=1e-6)
  loss_checks.append({'prediction_type':prediction_type,'official_loss':float(lo.detach()),'project_loss':float(lp.detach()),'prediction_gradient_max_difference':float((go-gp).abs().max()),'valid_pose_components':'all','observation_features':'synthetic_global_condition; no encoder instantiated','gripper_loss_included':False})
 result={'stage':'official_project_unet_loss_direct_comparison','execution_checks_passed':True,'trained':False,'weights_downloaded':False,'data_downloaded':False,'torch_version':torch.__version__,'official_commit':'5ba07ac6661db573af695b419a7947ecb704690f','unet':unet,'loss':loss_checks,'epsilon_x0_effective_weight_alpha_over_one_minus_alpha':{str(t):float(abar[t]/(1-abar[t])) for t in [0,24,49,74,99]},'limits':['Full-width mapping is parameter shape/schema comparison; forward/backward numerical run uses smaller equal widths.','Same weights and architecture tests implementation equivalence, not learned behavior or training quality.','Loss test uses synthetic global observations, fully valid pose entries and no gripper BCE; mask/observation encoders not fully reproduced.','Different prediction types are valid supported objectives; equality within a type does not equate epsilon and x0 optimization.']}
 (ROOT/'reports/official_project_unet_loss_comparison_20261011.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))

if __name__=='__main__':main()
