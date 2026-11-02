from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from gripper_event_objective import gripper_event_loss,fit_event_frequencies


class EventObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.commands=torch.full((2,16),-1.)
        self.commands[0,5:]=1.; self.commands[1,:8]=1.
        self.frequencies=fit_event_frequencies(self.commands)

    def test_formula_matches_independent_probability_calculation(self):
        logits=torch.linspace(-2,2,32,dtype=torch.float64).reshape(2,16).requires_grad_(True)
        loss,parts=gripper_event_loss(logits,self.commands,self.frequencies)
        p=1/(1+np.exp(-logits.detach().numpy())); y=self.commands.numpy()>0
        expected=[]
        for name,direction in [('opening',1),('closing',-1)]:
            event=(1-p[:,:-1])*p[:,1:] if direction==1 else p[:,:-1]*(1-p[:,1:])
            label=np.diff(y.astype(np.int8),axis=1)==direction
            rate=self.frequencies['rates'][name]
            value=np.where(label,-np.log(event)*.5/rate,-np.log1p(-event)*.5/(1-rate)).mean()
            self.assertAlmostEqual(float(parts[name].detach()),value,places=12); expected.append(value)
        state=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()
        self.assertAlmostEqual(float(loss.detach()),state+np.mean(expected),places=12)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_true_boundary_and_saturated_logits_have_finite_gradients(self):
        logits=(self.commands*12).clone(); logits[0,4]=2.; logits[0,5]=-2.; logits.requires_grad_(True)
        loss,_=gripper_event_loss(logits,self.commands,self.frequencies); loss.backward()
        self.assertGreater(float(logits.grad[0,4]),0) #Descent closes the prior step.
        self.assertLess(float(logits.grad[0,5]),0) #Descent opens at the true step.
        extreme=(self.commands*-1000).requires_grad_(True)
        loss,_=gripper_event_loss(extreme,self.commands,self.frequencies); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_frequency_binding_and_rejections(self):
        self.assertEqual(self.frequencies['windows'],2)
        self.assertAlmostEqual(self.frequencies['rates']['opening'],1/30)
        self.assertAlmostEqual(self.frequencies['rates']['closing'],1/30)
        invalid=dict(self.frequencies,fit_partition='development')
        with self.assertRaises(ValueError): gripper_event_loss(torch.zeros(2,16),self.commands,invalid)
        with self.assertRaises(ValueError): fit_event_frequencies(torch.ones(2,16))


if __name__=='__main__': unittest.main()
