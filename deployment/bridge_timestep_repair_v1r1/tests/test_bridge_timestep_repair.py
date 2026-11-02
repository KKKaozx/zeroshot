"""Focused objective/gradient, frozen native integration and pairing checks."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'vendor' if (ROOT / 'vendor').is_dir() else ROOT / 'deployment/bridge_full_capability_v1'
sys.path[:0] = [str(ROOT / 'code'), str(FROZEN / 'code'), str(FROZEN)]
from bridge_timestep_repair import timestep_weights, masked_x0_mse, check_matched_random_stream, backward


class TimestepRepairTest(unittest.TestCase):
    def test_formula_mask_and_gradient(self):
        a = torch.linspace(.999, .001, 100, dtype=torch.float64)
        t = torch.arange(100)
        w = timestep_weights(a, t, 'min_snr5_grid_mean_one')
        expected = np.minimum(a.numpy() / (1 - a.numpy()), 5)
        expected /= expected.mean()
        np.testing.assert_allclose(w.numpy(), expected, rtol=1e-13)
        self.assertAlmostEqual(float(w.mean()), 1, places=14)
        p = torch.tensor([[[1.]*7, [9.]*7], [[2.]*7, [3.]*7]], requires_grad=True)
        mask = torch.ones_like(p); mask[0,1] = 0
        loss = masked_x0_mse(p, torch.zeros_like(p), mask, torch.tensor([2., .5]))
        self.assertAlmostEqual(float(loss.detach()), (2 + .5*4 + .5*9)/3, places=6)
        loss.backward()
        self.assertEqual(float(p.grad[0,1].abs().sum()), 0)
        self.assertAlmostEqual(float(p.grad[0,0,0]), 4/21, places=6)

    def test_native_loss_sampler_hooks_and_paired_updates(self):
        import train
        import bcap_core
        from step_training_state import UpdateSchedule
        from types import SimpleNamespace
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.6))
            def forward(self, noisy, t, c):
                return self.weight * noisy + c[:,None,:].expand(-1,2,7)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_decoder = Decoder()
                self.vision_encoder = torch.nn.Linear(1,1).requires_grad_(False)
                self.text_encoder = torch.nn.Linear(1,1).requires_grad_(False)
                self.diffusion_prediction_type = 'sample'; self.chunk_size = 2; self.action_dim = 8
                self.decoder_type = 'diffusion'; self.num_diffusion_steps = 100; self.separate_gripper_head = True
                self.gripper_loss_weight = .25; self.gripper_target_mode = 'state'
                self.gripper_change_weight = 1.; self.balanced_gripper_loss = False
                self.register_buffer('alpha_bars', torch.linspace(.999,.001,100))
            def get_context_vector(self, image, ids, attention): return image
            def predict_gripper_logits(self, context, actions, current): return context[:,:1].expand(-1,2)
            from models import RobotAdapterModel
            diffusion_loss = RobotAdapterModel.diffusion_loss
        actions = torch.zeros(2,2,8); actions[...,6] = 1; actions[...,7] = 1
        masks = torch.ones_like(actions); masks[0,1] = 0
        batch = {'images':torch.zeros(2,7), 'tokens':{'input_ids':None,'attention_mask':None},
            'pose':torch.ones(2,18), 'actions':actions, 'masks':masks, 'current':torch.zeros(2,1)}
        histories = []
        states = []
        for mode in ('uniform','min_snr5_grid_mean_one'):
            torch.manual_seed(42); model = Model(); port = torch.nn.Linear(18,7,bias=False)
            opt = torch.optim.AdamW(list(model.parameters()) + list(port.parameters()), lr=1e-5)
            b = dict(model=model,port=port,optimizer=opt,batch=lambda ids:batch,
                api=SimpleNamespace(policy_loss=train.policy_loss,trainable_state_dict=train.trainable_state_dict),
                repair_mode=mode,cuda=False,plan=bcap_core.FixedPlan(np.array([[0,1]]*3,dtype=np.int64),'toy'),
                schedule=UpdateSchedule(total_updates=3,base_lrs=[1e-5],kind='constant'),
                contract={'test_only':True,'mode':mode},checkpoint_every=3,training_cap_seconds=60,check_frozen=lambda:None)
            torch.manual_seed(84)
            original_backward = bcap_core.backward; bcap_core.backward = backward
            try:
                guard = bcap_core.zero_guard(b)
                self.assertTrue(guard['repeat_loss_gradient_rng_exact'])
                self.assertEqual(b['plan'].completed_updates,0)
                self.assertFalse(opt.state)
                (ROOT / 'reports').mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as temporary:
                    extra = bcap_core.train(b,Path(temporary),{'history':[],'zero_update_guard':guard})
                    history = extra['history']
                    saved = torch.load(Path(temporary)/'latest_state.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(saved['plan']['completed_updates'],3)
                    self.assertEqual(set(saved['model']),{'base','pose_port'})
                    self.assertTrue(saved['optimizer']['state'])
                    self.assertTrue(all(int(s['step'])==3 for s in saved['optimizer']['state'].values()))
            finally:
                bcap_core.backward = original_backward
            histories.append(history); states.append(model.diffusion_decoder.weight.detach().clone())
        self.assertEqual(check_matched_random_stream(*histories)['matched_updates'],3)
        self.assertNotEqual(histories[0][0]['pose_loss'], histories[1][0]['pose_loss'])
        changed = [x.copy() for x in histories[1]]; changed[1]['timesteps'] = [0,0]
        with self.assertRaises(ValueError): check_matched_random_stream(histories[0], changed)

    def test_invalid_schedule_and_mode_rejected(self):
        a = torch.linspace(.999,.001,100)
        for schedule,t,mode in [(a,torch.tensor([100]),'uniform'),(a[:2],torch.tensor([0]),'uniform'),
                (a,torch.tensor([0]),'epsilon'),(a,torch.tensor([0.]),'uniform')]:
            with self.assertRaises(ValueError): timestep_weights(schedule,t,mode)


if __name__ == '__main__': unittest.main()
