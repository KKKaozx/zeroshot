"""Expose an existing bound; use a tiny clean-prediction sampler fixture only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import torch
from torch import nn
import models
from train import parse_args,build_config


class EmptyEncoder(nn.Module):
    def __init__(self):super().__init__();self.config=SimpleNamespace(hidden_size=8)


class FixedCleanPose(nn.Module):
    def forward(self,actions,timesteps,context):
        result=torch.zeros_like(actions);result[...,:3]=9.595791399478912;result[...,6]=1
        return result


def main():
    torch.set_num_threads(1)
    base=build_config(parse_args(['--chunk-size','4','--adapter-layers','1','--attention-dim','8',
        '--attention-heads','2','--decoder-hidden-dim','8','--diffusion-steps','4',
        '--beta-schedule','squaredcos_cap_v2','--diffusion-prediction-type','sample']))
    assert base['action']['max_normalized_position']==3
    large=copy.deepcopy(base);large['action']['max_normalized_position']=10.
    args=parse_args(['--max-normalized-position','10'])
    assert build_config(args)['action']['max_normalized_position']==10
    for value in ('0','-1','nan','inf'):
        try:build_config(parse_args(['--max-normalized-position',value]))
        except ValueError:pass
        else:raise AssertionError('Invalid normalized bound accepted')
    outputs=[]
    with patch.object(models.CLIPModel,'from_pretrained',side_effect=lambda *a,**kw:
            SimpleNamespace(vision_model=EmptyEncoder(),text_model=EmptyEncoder())):
        for config in (base,large):
            torch.manual_seed(7)
            model=models.RobotAdapterModel(config).eval()
            model.diffusion_decoder=FixedCleanPose()
            result=model.sample(torch.zeros(1,8),current_gripper=torch.ones(1,1))
            outputs.append(result)
    assert torch.allclose(outputs[0][...,:3],torch.full((1,4,3),3.),rtol=0,atol=1e-5)
    assert torch.allclose(outputs[1][...,:3],torch.full((1,4,3),9.595791399478912),rtol=0,atol=1e-5)
    assert torch.equal(outputs[0][...,3:],outputs[1][...,3:])
    report={'passed':True,'legacy_default':3.,'explicit_candidate_bound':10.,
        'fixture_clean_prediction':9.595791399478912,'legacy_clips_as_expected':True,
        'explicit_bound_preserves_fixture_position':True,'rotation_and_gripper_unchanged':True,
        'source':'frozen training role statistics only; no validation/test labels used',
        'formal_bound_frozen_before_physical_scale_resolution':False,
        'pretrained_models_loaded':False,'training_updates':0,'physics_steps':0,
        'source_sha256':{n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in ('models.py','train.py')}}
    out=ROOT/'reports/oxe_bounded_numeric_20261025/position_bound_check.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
