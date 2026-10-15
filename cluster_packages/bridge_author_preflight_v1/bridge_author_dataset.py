"""Thin dataset/runner adapters for the unmodified pinned author workspace."""
from pathlib import Path
import numpy as np
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.normalizer import LinearNormalizer,SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from bridge_author_io import load_cache

class BridgeAuthorDataset(BaseImageDataset):
    def __init__(self,cache,protocol,split='train'):
        self.cache=Path(cache);self.protocol=Path(protocol);self.split=split
        self.arrays,self.report=load_cache(self.cache,self.protocol)
        # Resource preflight validates on4 training rows, never scores development.
        self.indices=np.arange(128) if split=='train' else np.arange(4)
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        i=int(self.indices[index])
        return dict(obs={k:torch.from_numpy(self.arrays[k][i:i+1].copy()) for k in ('image','ee_state','language')},
                    action=torch.from_numpy(self.arrays['action'][i].copy()))
    def get_validation_dataset(self):return BridgeAuthorDataset(self.cache,self.protocol,'preflight_train_checks')
    def get_all_actions(self):return torch.from_numpy(self.arrays['action'][:128].copy())
    def get_normalizer(self,**kwargs):
        n=LinearNormalizer()
        n.fit(dict(action=self.arrays['action'][:128].copy(),ee_state=self.arrays['ee_state'][:128].copy()),last_n_dims=1,mode='limits')
        n['image']=get_image_range_normalizer()
        n['language']=SingleFieldLinearNormalizer.create_identity()
        return n

class NoRolloutRunner(BaseImageRunner):
    def run(self,policy):return {} # Offline protocol; no robot, task success or simulator.
