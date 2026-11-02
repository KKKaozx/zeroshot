"""Native t0forward equivalence, actual sampler RNG reference and saved-array QA on CPU."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/bridge_low_noise_pair_v1'
BASE=ROOT/'deployment/bridge_full_capability_v1';PARENT=ROOT/'deployment/bridge_rotation_trace_v1'
sys.path[:0]=[str(BUNDLE),str(BASE/'code'),str(BASE),str(PARENT)]
spec=importlib.util.spec_from_file_location('low_pair_native_fixture',ROOT/'tests/test_bridge_full_capability.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
from lowpair_core import initial_noise,require_original_noise,teacher_forward,summarize
from rtrace_engine import instrument
from bcap_core import identity
from pose_runtime import digest
from oxe_nominal_target_store import NominalTargetStore
import run_low_pair
import verify_low_pair


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.stores={'train':NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025'),
            'validation':NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')}

    def test_actual_native_diffusion_loss_t0_input_and_raw_output_equivalence(self):
        b=fixture.make_bundle(self.stores);b['model'].eval();b['port'].eval();model=b['model']
        x=b['batch'](b['plan'].batches[0].tolist());noise=torch.linspace(-2,2,8*16*7).reshape(8,16,7)
        with torch.no_grad():c=model.get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+b['port'](x['pose'])
        before=identity(b);adam=digest(b['optimizer'].state_dict());seen=[];old=model.diffusion_decoder
        def observe(a,t,c):seen.append((a.detach().clone(),t.detach().clone()));return old(a,t,c)
        model.diffusion_decoder=observe
        with patch.object(b['optimizer'],'step',side_effect=AssertionError('No updates allowed')):
            xt,raw=teacher_forward(model,c,x['actions'][...,:7],noise)
            with torch.no_grad(),patch.object(torch,'randint',return_value=torch.zeros(8,dtype=torch.long)),patch.object(torch,'randn_like',return_value=noise):
                native=model.diffusion_loss(x['actions'],c,x['current'])
        self.assertTrue(torch.equal(xt,seen[0][0]));self.assertTrue(torch.equal(xt,seen[1][0]));self.assertTrue(torch.equal(raw,native[0]))
        self.assertTrue((seen[0][1]==0).all());self.assertFalse(raw.requires_grad)
        self.assertEqual(before,identity(b));self.assertEqual(adam,digest(b['optimizer'].state_dict()))
        self.assertTrue(all(p.grad is None for p in list(model.parameters())+list(b['port'].parameters())))

    def test_initial_gaussian_reconstruction_matches_actual_native100step_rng(self):
        b=fixture.make_bundle(self.stores);model=b['model'];model.eval()
        class Hook:
            def __init__(self):self.initial={};self.key=None
            def raw(self,t,x,clean):
                if t==99:self.initial[self.key]=x[...,3:7].cpu().numpy().copy()
            def reconstructed(self,*args):pass
            def clipped(self,*args):pass
            def final_raw(self,*args):pass
        hook=Hook();ns={'torch':torch,'F':F,'RTRACE_HOOK':hook}
        exec(compile(instrument(BASE/'code/models.py'),'<native-rng-reference>','exec'),ns)
        original=type(model).sample;type(model).sample=ns['sample']
        try:
            with torch.inference_mode(),patch.object(torch.cuda,'manual_seed_all'):
                for seed in [1101,1102,1103]:
                    torch.manual_seed(seed)
                    for batch in range(17):
                        hook.key=(seed,batch);model.sample(torch.zeros(8,12),torch.zeros(8,2))
        finally:type(model).sample=original
        with patch.object(torch.cuda,'manual_seed_all'):
            for seed in [1101,1102,1103]:
                for role,batch in [('train',0),('validation',16)]:
                    eps=initial_noise(seed,role,'cpu');require_original_noise(eps,hook.initial[(seed,batch)])
                    wrong=hook.initial[(seed,batch)].copy();wrong[0,0,0]+=.01
                    with self.assertRaisesRegex(AssertionError,'exactly'):require_original_noise(eps,wrong)

    def test_saved_verifier_metrics_and_corruption_rejection_with_real_source_arrays(self):
        source=ROOT/'reports/bridge_rotation_trace_235002/cluster'
        with np.load(source/'trace.npz',allow_pickle=False) as z:
            target=z['targets'].copy();mask=z['masks'].copy();native=z['raw'][:,:,-1].copy();epsq=z['latent'][:,:,0].copy()
        eps=np.zeros((3,16,16,7),np.float32);eps[...,3:7]=epsq
        a=np.float32(run_low_pair.read(BUNDLE/'protocol.json')['alpha_bar_t0'])
        xt=np.sqrt(a)*target[None,...,:7]+np.sqrt(np.float32(1)-a)*eps
        perfect=np.repeat(target[None,...,:7],3,axis=0)
        arrays={'teacher_input':xt,'teacher_raw':perfect,'epsilon':eps,'native_raw_t0':native,'targets':target,'masks':mask}
        report=summarize(arrays)
        self.assertLess(report['roles']['train']['rotation']['teacher_denoised_output']['angle_deg'],1e-5)
        self.assertLess(report['roles']['validation']['comparisons']['teacher_output_minus_existing_native_raw_t0']['mean_deg'],0)
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temporary=tempfile.TemporaryDirectory(dir=parent,prefix='low_pair_verify_')
        assert Path(temporary.name).resolve().parent==parent.resolve()
        with temporary as folder:
            out=Path(folder);np.savez_compressed(out/'paired_inputs_outputs.npz',**arrays)
            p=run_low_pair.read(BUNDLE/'protocol.json')
            metadata={'execution_complete':True,'optimizer_updates':0,'new_full_sampling_runs':0,'direct_denoiser_forwards':6,
                'source_policy_and_gradients_adam_unchanged':True,'original_initial_gaussians_replayed_exactly':True,
                'source_identity_unchanged':p['source_final_identity'],'protocol_sha256':run_low_pair.sha(BUNDLE/'protocol.json'),
                'arrays_sha256':run_low_pair.sha(out/'paired_inputs_outputs.npz'),'analysis':report}
            run_low_pair.write(out/'report.json',metadata)
            self.assertTrue(verify_low_pair.verify(out,source)['verified'])
            metadata['analysis']['roles']['train']['rotation']['teacher_denoised_output']['angle_deg']+=1
            run_low_pair.write(out/'report.json',metadata)
            with self.assertRaises(AssertionError):verify_low_pair.verify(out,source)


if __name__=='__main__':unittest.main(verbosity=2)
