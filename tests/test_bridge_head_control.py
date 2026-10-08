import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
os.environ['USE_TF']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import torch
from torch import nn
from models import RobotAdapterModel
from train import trainable_state_dict, policy_loss
from run_bridge_head_control import orders, common_state, digest


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=16)
        self.weight=nn.Parameter(torch.ones(16))
    def forward(self,pixel_values=None,input_ids=None,attention_mask=None):
        x=pixel_values if pixel_values is not None else input_ids
        features=x.float().unsqueeze(-1)*self.weight
        return SimpleNamespace(last_hidden_state=features,pooler_output=features[:,0])


class HeadControlTests(unittest.TestCase):
    def test_orders_do_not_depend_on_model_rng_consumption(self):
        expected=orders(list(range(10)))
        np.random.seed(975); np.random.randn(1000); torch.randn(127)
        self.assertEqual(expected,orders(list(range(10))))
        for epoch in expected:self.assertEqual(sorted(epoch),list(range(10)))

    def test_actual_heads_share_initial_modules_and_both_backpropagate(self):
        torch.set_num_threads(2)
        cfg={'model':dict(name='mock',chunk_size=16,separate_gripper_head=True,
            trajectory_conditioned_gripper=True,condition_on_current_gripper=True,
            num_adapter_layers=1,attention_dim=16,num_attention_heads=4,
            adapter_pooling='cls_patch_mean',decoder_hidden_dim=16,
            diffusion_prediction_type='sample',num_diffusion_steps=100,
            beta_schedule='squaredcos_cap_v2',dropout=.1)}
        shared=None
        with patch('models.CLIPModel.from_pretrained',side_effect=lambda *a,**k:SimpleNamespace(vision_model=Encoder(),text_model=Encoder())):
            for kind in ['regression','diffusion']:
                torch.manual_seed(42)
                config=copy.deepcopy(cfg);config['model']['decoder_type']=kind
                model=RobotAdapterModel(config)
                if shared is None:shared=common_state(trainable_state_dict(model))
                else:model.load_state_dict(shared,strict=False)
                self.assertEqual(digest(shared),digest(common_state(trainable_state_dict(model))))
                model.train()
                images=torch.randn(2,5);text=torch.ones(2,3,dtype=torch.long)
                current=torch.zeros(2,1);action=torch.randn(2,16,8)
                action[...,3:7]=nn.functional.normalize(action[...,3:7],dim=-1)
                action[...,7]=1
                output=model(images,text,attention_mask=torch.ones_like(text),current_gripper=current,actions=action)
                loss,_,_=policy_loss(model,output,action,nn.MSELoss(),current_grippers=current,supervision_masks=torch.ones_like(action))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for prefix in ['adapter.','gripper_head.','regression_head.' if kind=='regression' else 'diffusion_decoder.']:
                    self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for n,p in model.named_parameters() if n.startswith(prefix)))
                self.assertTrue(all(p.grad is None for p in model.vision_encoder.parameters()))
                self.assertFalse(model.vision_encoder.training)


if __name__=='__main__':unittest.main()
