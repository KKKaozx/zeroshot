"""Production Bridge loop: real targets, native x0 loss/100DDPM; toy CPU networks."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/bridge_full_capability_v1'
sys.path[:0]=[str(BUNDLE/'code'),str(BUNDLE)]
import bcap_core as core
import bcap_runtime as runtime
from pose_runtime import zero_port,digest
from pose_features import batch_features
from step_training_state import UpdateSchedule,capture_rng
from oxe_nominal_target_store import NominalTargetStore


def native(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    ns={'torch':torch,'nn':nn,'F':F,'extract':lambda v,t,x:v.gather(0,t).reshape(len(t),1,1)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+body,type_ignores=[])),'<vendored-native-functions>','exec'),ns)
    return ns


def make_bundle(stores,total=4):
    methods=native(BUNDLE/'code/models.py',{'diffusion_loss','sample'})
    loss=native(BUNDLE/'code/train.py',{'policy_loss'})['policy_loss']
    class Model(nn.Module):
        diffusion_loss=methods['diffusion_loss'];sample=methods['sample']
        def __init__(self):
            super().__init__();self.vision_encoder=nn.Linear(3,3);self.text_encoder=nn.Embedding(8,2)
            self.vision_encoder.requires_grad_(False);self.text_encoder.requires_grad_(False)
            self.adapter=nn.Sequential(nn.Linear(3,12),nn.Dropout(.1));self.decoder=nn.Linear(19,7);self.gripper_head=nn.Linear(12,16)
            self.action_dim=8;self.chunk_size=16;self.num_diffusion_steps=100;self.decoder_type='diffusion';self.diffusion_prediction_type='sample'
            self.separate_gripper_head=True;self.clip_denoised=True;self.max_normalized_position=8.
            self.gripper_target_mode='state';self.balanced_gripper_loss=False;self.gripper_change_weight=1.;self.gripper_loss_weight=.25
            betas=torch.linspace(.0001,.02,100);a=1-betas;ab=a.cumprod(0);prev=torch.cat([torch.ones(1),ab[:-1]])
            for k,v in {'alpha_bars':ab,'alphas':a,'posterior_variance':betas*(1-prev)/(1-ab),
                    'posterior_mean_x0':betas*prev.sqrt()/(1-ab),'posterior_mean_xt':(1-prev)*a.sqrt()/(1-ab)}.items():self.register_buffer(k,v)
        def get_context_vector(self,images,input_ids,attention_mask):
            with torch.no_grad():v=self.vision_encoder(images)
            return self.adapter(v)
        def diffusion_decoder(self,x,t,c):return self.decoder(torch.cat([x,c[:,None].expand(-1,16,-1)],-1))
        def predict_gripper_logits(self,c,actions,current):return self.gripper_head(c)
    torch.manual_seed(42);model=Model();port=zero_port('cpu',width=12)
    groups=[[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('gripper_head.')],
        list(model.gripper_head.parameters()),list(port.parameters())]
    opt=torch.optim.AdamW([{'params':g,'lr':2e-4} for g in groups],weight_decay=.0001)
    schedule=UpdateSchedule(total_updates=total,base_lrs=[2e-4]*3,kind='cosine',warmup_updates=1);schedule.apply(opt,0)
    ids=np.load(BUNDLE/'batches.npy',mmap_mode='r')[:total].copy()
    plan=core.FixedPlan(ids,'cpu-real-Bridge-fixture-'+str(total));stats=runtime.read(BUNDLE/'state_stats.json')
    def pack(role,indices):
        s=stores[role]
        assert all(s.episodes[int(s.windows[i,0])]['source']=='Bridge' for i in indices)
        items=[s.get(i) for i in indices]
        return {'images':torch.tensor([[i/145812,.1,1.] for i in indices]),
            'tokens':{'input_ids':torch.ones(len(indices),77,dtype=torch.long),'attention_mask':torch.ones(len(indices),77,dtype=torch.long)},
            'current':torch.from_numpy(np.stack([x['current_features'] for x in items])),
            'actions':torch.from_numpy(np.stack([x['actions'] for x in items])),
            'masks':torch.from_numpy(np.stack([x['supervision_mask'] for x in items])),
            'pose':torch.from_numpy(batch_features(s,indices,stats,'real_pose'))}
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    def check_frozen():
        assert all(torch.equal(v,dict(model.named_parameters())[k]) for k,v in frozen.items())
    return {'model':model,'port':port,'optimizer':opt,'schedule':schedule,'plan':plan,'cuda':False,
        'api':SimpleNamespace(policy_loss=loss,trainable_state_dict=lambda m:{n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad}),
        'contract':{'fixture':'real-target-native-x0-Bridge','total':total},'checkpoint_every':2,'training_cap_seconds':120,
        'check_frozen':check_frozen,'batch':lambda ids:pack('train',ids),'pack':pack}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.stores={'train':NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025'),
            'validation':NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')}

    def test_actual_full_bridge_plan_stats_and_cpu_admission(self):
        train=self.stores['train'];ids=np.load(BUNDLE/'train_indices.npy');batches=np.load(BUNDLE/'batches.npy')
        self.assertEqual(batches.shape,(65536,8));self.assertTrue(np.isin(batches,ids).all())
        self.assertEqual(len(ids),10352);self.assertEqual(runtime.fit_stats(train,ids.tolist()),runtime.read(BUNDLE/'state_stats.json'))
        with self.assertRaisesRegex(AssertionError,'training role'):runtime.fit_stats(self.stores['validation'],[0])
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='bridge_admit_') as folder:
            root=Path(folder)
            for s in self.stores.values():
                for e in s.episodes:
                    if e['source']=='Bridge':
                        f=root/'data/oxe_nominal_pool_v1'/e['stage_relative_path'];f.parent.mkdir(parents=True,exist_ok=True);f.touch()
            def store(path,partition='train'):return self.stores[partition]
            with patch('oxe_nominal_target_store.NominalTargetStore',side_effect=store),patch.object(torch,'__version__','2.5.1+cu118'),patch.object(torch.version,'cuda','11.8'):
                checked=runtime.admit(root,space=False)
                self.assertTrue(checked['passed']);self.assertFalse(checked['cuda_initialized'])
                one=next(root.rglob('*.tfrecord-*'));one.unlink()
                with self.assertRaisesRegex(AssertionError,'raw image'):runtime.admit(root,space=False)

    def test_native_loss_zero_guard_fresh_adam_and_exact_resume(self):
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='bridge_resume_') as folder:
            folder=Path(folder);full=folder/'full';half=folder/'half';full.mkdir();half.mkdir()
            a=make_bundle(self.stores);guard=core.zero_guard(a)
            self.assertFalse(a['optimizer'].state);self.assertFalse(a['port'].weight.count_nonzero())
            extra={'history':[],'training_seconds':0.,'zero_update_guard':guard}
            with redirect_stdout(io.StringIO()):core.train(a,full,extra)
            final_rng=digest(capture_rng(include_cuda=False));full_identity=core.identity(a);full_adam=digest(a['optimizer'].state_dict())
            b=make_bundle(self.stores);g=core.zero_guard(b);e={'history':[],'training_seconds':0.,'zero_update_guard':g}
            with redirect_stdout(io.StringIO()):core.train(b,half,e,stop=2)
            restored=make_bundle(self.stores);e=core.resume(restored,half/'latest_state.pt')
            with redirect_stdout(io.StringIO()):core.train(restored,half,e)
            self.assertEqual(full_identity,core.identity(restored));self.assertEqual(full_adam,digest(restored['optimizer'].state_dict()))
            self.assertEqual(final_rng,digest(capture_rng(include_cuda=False)))
            self.assertEqual(len(restored['optimizer'].param_groups),3);self.assertTrue(restored['port'].weight.count_nonzero())
            wrong=make_bundle(self.stores);wrong['contract']['total']=99;before=core.identity(wrong)
            with self.assertRaises(ValueError):core.resume(wrong,half/'latest_state.pt')
            self.assertEqual(before,core.identity(wrong))

    def test_real_192_targets_native100step_eval_preserves_training_rng(self):
        b=make_bundle(self.stores);b['check_frozen']();before=core.identity(b);rng=digest(capture_rng(include_cuda=False))
        import step_training_state as ss
        old_capture=ss.capture_rng
        def cpu_capture(include_cuda):return old_capture(include_cuda=False)
        with patch.object(ss,'capture_rng',side_effect=cpu_capture),patch.object(torch.cuda,'manual_seed_all'),redirect_stdout(io.StringIO()):
            pred,target,mask=runtime.evaluate(b)
        self.assertEqual(pred.shape,(3,192,16,8));self.assertEqual(core.identity(b),before)
        self.assertEqual(digest(capture_rng(include_cuda=False)),rng)
        np.testing.assert_allclose(np.linalg.norm(pred[...,3:7],axis=-1),1,atol=1e-6)
        selection=runtime.read(BUNDLE/'selection.json');static=np.zeros_like(pred);static[...,6]=1
        perfect=np.repeat(target[None],3,axis=0)
        good=core.summarize({'before':pred,'after':perfect,'static':static},target,mask,selection)
        bad=core.summarize({'before':pred,'after':static,'static':static},target,mask,selection)
        self.assertTrue(good['train']['basic_capability_passed']);self.assertTrue(good['validation']['basic_capability_passed'])
        self.assertFalse(bad['train']['basic_capability_passed']);self.assertFalse(bad['validation']['basic_capability_passed'])

    def test_small_output_verifier_recomputes_and_rejects_tampered_targets(self):
        import bcap_verify
        b=make_bundle(self.stores)
        initial=core.identity(b)
        with np.load(BUNDLE/'evaluation_reference.npz') as z:target=z['targets'].copy();mask=z['masks'].copy()
        static=np.zeros((3,192,16,8),np.float32);static[...,6]=1
        perfect=np.repeat(target[None],3,axis=0);selection=runtime.read(BUNDLE/'selection.json')
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='bridge_small_') as folder:
            out=Path(folder)
            np.savez_compressed(out/'initial_predictions.npz',before=static,targets=target,masks=mask)
            np.savez_compressed(out/'predictions_and_targets.npz',before=static,after=perfect,static=static,targets=target,masks=mask)
            runtime.write(out/'initialization.json',{'empty_adam':True,'pose_port_zero':True,'source_learned_checkpoints_read':False,'identity':initial})
            runtime.write(out/'zero_update_guard.json',{'passed':True,'optimizer_updates':0})
            runtime.write(out/'report.json',{'execution_complete':True,'completed_updates':65536,'training_seed':42,
                'protocol_sha256':runtime.sha(BUNDLE/'protocol.json'),'automatic_following_action':False,
                'initial_identity':initial,'metrics':core.summarize({'before':static,'after':perfect,'static':static},target,mask,selection)})
            checked=bcap_verify.verify(out,full=False)
            self.assertTrue(checked['training_passed']);self.assertFalse(checked['full_policy_adam_plan_checked'])
            target[0,0,0]+=1
            np.savez_compressed(out/'predictions_and_targets.npz',before=static,after=perfect,static=static,targets=target,masks=mask)
            with self.assertRaises(AssertionError):bcap_verify.verify(out,full=False)

    def test_cap_and_wrong_source_rejected_without_extra_update(self):
        b=make_bundle(self.stores);guard=core.zero_guard(b);before=core.identity(b)
        wrong=next(i for i,e in enumerate(self.stores['train'].windows[:,0]) if self.stores['train'].episodes[int(e)]['source']!='Bridge')
        with self.assertRaises(AssertionError):core.backward(b,[wrong]*8)
        self.assertEqual(before,core.identity(b));self.assertFalse(b['optimizer'].state)
        b['request_stop']=True
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent,prefix='bridge_cap_') as folder:
            with self.assertRaises(TimeoutError):core.train(b,folder,{'history':[],'training_seconds':0.,'zero_update_guard':guard})
            saved=torch.load(Path(folder)/'latest_state.pt',weights_only=False)
            self.assertEqual(saved['plan']['completed_updates'],0)
        self.assertEqual(before,core.identity(b))


if __name__=='__main__':unittest.main(verbosity=2)
