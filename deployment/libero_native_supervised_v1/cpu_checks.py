"""Meaningful native-contract checks; tiny synthetic updates only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from native_data import causal_chunks,fit_normalizer,normalize_actions,decode_actions,proprio_features,prepare_pixels
from native_policy import NativeActionPolicy


class NativeControlTests(unittest.TestCase):
    def test_causal_shift_mask_no_cross_episode(self):
        actions=np.zeros((5,7),np.float32);actions[:,0]=np.arange(5)/5;actions[:,6]=-1
        target,mask=causal_chunks(actions)
        self.assertTrue(np.array_equal(target[0,:4],actions[1:]))
        self.assertEqual(mask[:,0,0].sum(),4);self.assertEqual(mask[-1].sum(),1)
        self.assertTrue(np.all(target[0,4:]==0))
        other=np.ones((2,7),np.float32);t,m=causal_chunks(other)
        self.assertTrue(np.array_equal(t[0,0],other[1]));self.assertEqual(m.sum(),1)

    def test_train_only_normalization_and_native_constant_decode(self):
        state=np.arange(27,dtype=np.float32).reshape(3,9)
        action=np.array([[0,.2,0,.1,0,.1,-1],[.8,-.5,.2,-.2,.1,0,-1]],np.float32)
        n=fit_normalizer(state,action);before=copy.deepcopy(n)
        self.assertTrue(np.allclose(decode_actions(normalize_actions(action,n),n),action))
        outside=np.full((2,16,7),1e9,np.float32);decoded=decode_actions(outside,n)
        self.assertTrue(np.all(decoded[...,6]==-1));self.assertTrue(np.all(np.abs(decoded)<=1))
        normalize_actions(np.full((20,7),1000),n);proprio_features(np.full((3,9),1e6),n)
        self.assertEqual(before,n)

    def test_opengl_flip_once(self):
        image=np.zeros((128,128,3),np.uint8);image[0,:,0]=255;image[-1,:,2]=255
        pixels=prepare_pixels(image)[0];mean=torch.tensor([.48145466,.4578275,.40821073]);std=torch.tensor([.26862954,.26130258,.27577711])
        self.assertTrue(torch.allclose(pixels[:,0,0],(torch.tensor([0.,0.,1.])-mean)/std))
        self.assertTrue(torch.allclose(pixels[:,-1,0],(torch.tensor([1.,0.,0.])-mean)/std))

    def test_native_sampler_masked_loss_and_full_adam_roundtrip(self):
        torch.set_num_threads(2)
        cfg={'context_dim':32,'text_dim':24,'adapter_layers':1,'attention_dim':16,'heads':4,
             'dropout':0.,'down_dims':[8,16],'step_embed_dim':16}
        model=NativeActionPolicy(cfg);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        visual=torch.randn(2,5,32);language=torch.randn(2,4,24);attention=torch.ones(2,4,dtype=torch.long);state=torch.randn(2,9)
        target=torch.rand(2,16,7)*2-1;mask=torch.ones(2,16,1);mask[:,-3:]=0
        for _ in range(2):
            model.train();ctx=model.context(visual,language,attention,state);loss=model.loss(target,mask,ctx)
            self.assertTrue(torch.isfinite(loss));optimizer.zero_grad();loss.backward();optimizer.step()
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            assert Path(folder).resolve().is_relative_to(HERE.resolve())
            path=Path(folder)/'roundtrip.pt';torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict()},path)
            payload=torch.load(path,weights_only=False);copy_model=NativeActionPolicy(cfg);copy_model.load_state_dict(payload['state'],strict=True)
            self.assertTrue(all(torch.equal(v,copy_model.state_dict()[k]) for k,v in model.state_dict().items()))
            self.assertTrue(all(int(v['step'])==2 for v in payload['optimizer']['state'].values()))
            self.assertEqual(len(payload['optimizer']['state']),len(list(model.parameters())))
        model.eval()
        with torch.no_grad():sample=model.sample(model.context(visual,language,attention,state))
        self.assertEqual(tuple(sample.shape),(2,16,7));self.assertTrue(torch.isfinite(sample).all())
        self.assertTrue(torch.all(sample.abs()<=1))
        # Raw native rotation/gripper columns are not a normalized quaternion.
        class Constant(torch.nn.Module):
            def forward(self,x,t,ctx):return torch.full_like(x,.2)
        model.diffusion_decoder=Constant()
        with torch.no_grad():constant=model.sample(torch.zeros(2,32))
        self.assertTrue(torch.allclose(constant,torch.full_like(constant,.2),atol=1e-5))
        masked_target=torch.full((2,16,7),.2);masked_target[:,-3:]=1000
        self.assertEqual(float(model.loss(masked_target,mask,torch.zeros(2,32))),0.)

    def test_bound_partition_and_alignment_metadata(self):
        binding=json.loads((HERE/'inputs_receipt.json').read_text(encoding='utf-8'))
        self.assertTrue(binding['target_supervised']);self.assertFalse(binding['zero_shot']);self.assertTrue(binding['excluded_from_formal40'])
        self.assertFalse(binding['rollout_demonstration_payloads_decoded_during_preparation'])
        with np.load(HERE/'inputs.npz',allow_pickle=False) as z, np.load(HERE/'plan.npz',allow_pickle=False) as plan:
            self.assertTrue(np.all(z['split'][plan['batches']]==0))
            self.assertEqual(set(np.unique(z['demo_id'])),set(range(25)))
            self.assertEqual(int((z['split']==0).sum()),1509)
            self.assertEqual(int((z['split']==1).sum()),379)
            self.assertTrue(np.all(z['mask'].sum(axis=(1,2))>0))

    def test_complete_text_namespace_restore_for_both_api_shapes(self):
        from frozen_encoders import restore_text
        flat=torch.nn.Linear(3,2)
        weights={'text_model.'+k:v.detach().clone() for k,v in flat.state_dict().items()}
        other=torch.nn.Linear(3,2);restore_text(other,weights)
        self.assertTrue(all(torch.equal(v,flat.state_dict()[k]) for k,v in other.state_dict().items()))
        wrapped=torch.nn.Module();wrapped.text_model=torch.nn.Linear(3,2);restore_text(wrapped,weights)
        self.assertTrue(torch.equal(wrapped.text_model.weight,flat.weight))
        bad=weights.copy();bad.pop('text_model.bias')
        with self.assertRaisesRegex(ValueError,'complete parameter'):restore_text(other,bad)


if __name__=='__main__':unittest.main()
