"""Current-pose feature semantics and actual continuation driver with CPU toy models."""
import ast
from contextlib import contextmanager,ExitStack,redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/oxe_current_pose_preflight_v1';BASE=ROOT/'deployment/oxe_target_pair_short_v1'
sys.path[:0]=[str(BUNDLE),str(BASE),str(BASE/'code')]
from pose_features import raw_features,encode,current_rows,fit_training_stats,batch_features,read,sha,validate_stats
import pose_runtime
from step_training_state import ResumableBatchPlan,UpdateSchedule,save_checkpoint,capture_rng
from completed_policy_export import model_sha,contract_sha
from step_training_runner import frozen_digest
from instrument import method_source


class Store:
    partition='train'
    def __init__(self):
        self.windows=np.asarray([[0,i,1] for i in range(16)],dtype=np.int64);self.offsets=np.array([0,17])
        self.pose=np.zeros((17,7));self.pose[:,:3]=np.arange(17)[:,None]*np.array([.1,.2,.3]);self.pose[:,6]=1
        self.episodes=[{'source':'Bridge'}]
    def __len__(self):return 16


def native_functions(names):
    tree=ast.parse((BASE/'code/train.py').read_text(encoding='utf-8'))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
    ns={'torch':torch,'nn':nn};exec(compile(ast.fix_missing_locations(module),'<native-functions>','exec'),ns);return ns


class Toy(nn.Module):
    def __init__(self,config):
        super().__init__();self.chunk_size=16;self.action_dim=8;self.num_diffusion_steps=100
        self.decoder_type='diffusion';self.diffusion_prediction_type='sample';self.separate_gripper_head=True
        self.gripper_target_mode='state';self.gripper_change_weight=1.;self.balanced_gripper_loss=False;self.gripper_loss_weight=.25
        self.adapter=nn.Sequential(nn.Linear(3,1024),nn.Dropout(.1));self.decoder=nn.Linear(1031,7);self.gripper_head=nn.Linear(1024,16)
        self.register_buffer('alpha_bars',torch.linspace(.999,.001,100))
    def get_context_vector(self,images,input_ids,mask):return self.adapter(images)
    def diffusion_decoder(self,x,t,c):return self.decoder(torch.cat([x,c[:,None].expand(-1,16,-1)],-1))
    def predict_gripper_logits(self,c,actions,current):return self.gripper_head(c)


class Tests(unittest.TestCase):
    def test_quaternion_sign_and_rotation6d_matches_old_geometry(self):
        q=np.sqrt(.5);p=np.array([[1,2,3,0,0,q,q],[1,2,3,0,0,-q,-q]])
        x,m=raw_features(p,['Bridge','Bridge'])
        np.testing.assert_allclose(x[0],x[1],atol=1e-15)
        np.testing.assert_allclose(x[0,3:],[0,1,0,-1,0,0],atol=1e-15)
        self.assertTrue((m==1).all())
    def test_lt_missing_dims_never_used_and_mask_matched(self):
        p=np.array([[1,2,np.nan,np.nan,np.nan,np.nan,np.nan]])
        stats={'xyz_mean':[.1,.2,.3],'xyz_std':[1,2,3]}
        a=encode(p,['Language Table'],stats,'real_pose');b=encode(p,['Language Table'],stats,'mask_control')
        self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a[:,9:],b[:,9:])
        np.testing.assert_array_equal(a[:,2:9],0);np.testing.assert_array_equal(b[:,:9],0)
        np.testing.assert_array_equal(a[0,9:],[1,1,0,0,0,0,0,0,0])
    def test_current_row_causal_and_train_only_stats(self):
        s=Store();stats=fit_training_stats(s);a=batch_features(s,[0],stats,'real_pose')
        s.pose[1:,:3]+=999
        np.testing.assert_array_equal(a,batch_features(s,[0],stats,'real_pose'))
        s.partition='validation'
        with self.assertRaises(AssertionError):fit_training_stats(s)
    def test_real_frozen_store_stats_reference_and_suffix(self):
        from oxe_nominal_target_store import NominalTargetStore
        train=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025');stats=read(BUNDLE/'state_stats.json')
        validate_stats(train,stats)
        selection=read(BASE/'selection.json');parts={'train':train,'validation':NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')}
        with np.load(BUNDLE/'selected_state_reference.npz') as z:
            for role in ['mask_control','real_pose']:
                actual=np.concatenate([batch_features(parts[part],[r['window_index'] for r in selection[part]],stats,role) for part in ['train','validation']])
                np.testing.assert_array_equal(actual,z[role])
        with np.load(BUNDLE/'reference_suffix.npz') as z:
            self.assertEqual(z['batches'].shape,(2048,8))
            np.testing.assert_array_equal(train.masks[z['batches'][:2]].sum((1,2)),z['valid_dimensions'][:2])
    def test_zero_port_preserves_rng_and_zero_endpoint(self):
        torch.manual_seed(42);before=torch.get_rng_state().clone();port=pose_runtime.zero_port('cpu',width=7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.randn(4,18);context=torch.randn(4,7)
        self.assertTrue(torch.equal(context,context+port(x)))

    def test_actual_check_only_entry_with_bound_metadata(self):
        import shutil
        spec=importlib.util.spec_from_file_location('pose_metadata_driver',BUNDLE/'run_preflight.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        protocol=read(BUNDLE/'protocol.json')
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temp=tempfile.TemporaryDirectory(prefix='pose_metadata_entry_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as folder:
            folder=Path(folder);base=folder/'oxe_target_pair_short_v1';base.mkdir()
            for name in protocol['original_bundle_files']:
                target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
            evaluation=folder/'runs/oxe-target-short-eval-fixed-232091';evaluation.mkdir(parents=True)
            for name in protocol['evaluation_file_sha256']:shutil.copyfile(ROOT/'reports/oxe_target_eval_repair_232091/cluster'/name,evaluation/name)
            source=folder/'runs/oxe-target-short-231297/sample';source.mkdir(parents=True)
            (source/'latest_step.pt').write_bytes(b'CPU metadata fixture only, never deserialize')
            (source/'arm_receipt.json').write_text(json.dumps({'passed':True,'target':'sample','prefix_updates':8192,
                'final_learned_sha256':protocol['source_learned_sha256'],'checkpoint_contract_sha256':protocol['source_contract_sha256']}))
            with patch.object(sys,'argv',['run_preflight.py','--root',str(folder),'--check-only']),patch.object(driver,'initialize_runtime',side_effect=AssertionError('check-only must not initialize CUDA')):
                with redirect_stdout(io.StringIO()) as text:driver.main()
                self.assertIn('Metadata only',text.getvalue())

    def test_real_replay_entry_120rows_with_cpu_encoder_sampler_standins(self):
        from oxe_nominal_target_store import NominalTargetStore
        import step_training_runner
        spec=importlib.util.spec_from_file_location('pose_replay_driver',BUNDLE/'run_preflight.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        train=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
        val=NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')
        selection=read(BASE/'selection.json');reference=read(BASE/'reference_contract.json');stats=read(BUNDLE/'state_stats.json')
        class Part:
            def __init__(self,store):self.store=store
            def __getitem__(self,i):
                v=self.store.get(i)
                return 'fixture',torch.zeros(3,2,2),torch.from_numpy(v['current_features']),torch.from_numpy(v['actions']),torch.from_numpy(v['supervision_mask'])
        data=SimpleNamespace(training=Part(train),validation=Part(val));calls=[]
        class Vision(nn.Module):
            def forward(self,pixel_values):return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values),3,1024))
        class Text(nn.Module):
            def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids),77,4))
        class Model(nn.Module):
            def __init__(self):super().__init__();self.vision_encoder=Vision();self.text_encoder=Text()
            def adapter(self,vision,text,mask):return vision[:,0,:]
            def sample(self,context,current):
                calls.append(len(context));p=torch.zeros(len(context),16,8);p[...,6]=1;return p
        def tokenizer(text,**kw):return {'input_ids':torch.ones(len(text),77,dtype=torch.long),'attention_mask':torch.ones(len(text),77,dtype=torch.long)}
        items=[part[r['window_index']] for role,part in [('train',data.training),('validation',data.validation)] for r in selection[role]]
        targets=np.stack([v[3].numpy() for v in items]);masks=np.stack([v[4].numpy() for v in items])
        pred=np.zeros((3,120,16,8),np.float32);pred[...,6]=1
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temp=tempfile.TemporaryDirectory(prefix='pose_replay_entry_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as folder:
            folder=Path(folder);base=folder/'oxe_target_pair_short_v1';base.mkdir()
            (base/'reference_contract.json').write_bytes((BASE/'reference_contract.json').read_bytes())
            evaluation=folder/'runs/oxe-target-short-eval-fixed-232091';evaluation.mkdir(parents=True)
            np.savez_compressed(evaluation/'predictions_and_targets.npz',targets=targets,masks=masks,sample=pred)
            model=Model();port=pose_runtime.zero_port('cpu')
            with ExitStack() as stack:
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value=reference['tokenizer']))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
                stack.enter_context(patch.object(torch.cuda,'manual_seed'));stack.enter_context(patch.object(torch.cuda,'manual_seed_all'))
                with redirect_stdout(io.StringIO()):r=driver.replay_and_features(folder,data,model,port,tokenizer,None,stats,selection)
                self.assertTrue(r['passed']);self.assertEqual(calls,[8]*45)
                wrong=copy.deepcopy(selection);wrong['train'][0]['input_row']+=1
                with self.assertRaises(AssertionError):driver.replay_and_features(folder,data,model,port,tokenizer,None,stats,wrong)

    def test_actual_driver_two_roles_six_updates_moments_and_resume_on_cpu(self):
        spec=importlib.util.spec_from_file_location('current_pose_probe_driver',BUNDLE/'run_preflight.py')
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        import step_training_runner
        torch.set_num_threads(1);torch.manual_seed(42)
        functions=native_functions({'policy_loss','collate_batch'})
        model_module=ModuleType('models');model_module.RobotAdapterModel=Toy
        model_module.__dict__.update(torch=torch,Tuple=tuple,extract=lambda v,t,x:v.gather(0,t).reshape(len(t),1,1))
        api=ModuleType('train');api.__dict__.update({k:functions[k] for k in ['policy_loss','collate_batch']})
        api.GRIPPER_PARAMETER_PREFIXES=('gripper_head.',)
        api.trainable_state_dict=lambda m:{n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad}
        api.CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:SimpleNamespace())
        api.tokenise=lambda t,texts,device:{'input_ids':torch.ones(len(texts),1,dtype=torch.long),'attention_mask':torch.ones(len(texts),1,dtype=torch.long)}
        @contextmanager
        def constants(cls,*,on_restored,**kw):on_restored({'cpu_fixture':True});yield
        constructor=ModuleType('resume_constants');constructor.training_constant_constructor=constants
        store=Store();stats=fit_training_stats(store)
        class Data:
            samples=[{'source':'fixture_bridge'} for _ in range(16)];training=SimpleNamespace(store=store)
            def __getitem__(self,i):
                actions=torch.zeros(16,8);actions[:,0]=i*.01;actions[:,6]=1;actions[:,7]=1
                return 'fixture',torch.tensor([i/16,0.,1.]),torch.tensor([1.,1.]),actions,torch.ones_like(actions)
        data=Data();config={'model':{'diffusion_prediction_type':'sample','name':'cpu-fixture'}};model=Toy(config)
        named=list(model.named_parameters());groups=[[p for n,p in named if not n.startswith('gripper_head.')],[p for n,p in named if n.startswith('gripper_head.')]]
        optimizer=torch.optim.AdamW([{'params':g,'lr':.0002} for g in groups],weight_decay=.0001)
        # Nonempty original Adam moments are required in this fixture.
        sum(p.square().mean() for _,p in named).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        plan=ResumableBatchPlan(list(range(16)),batch_size=8,total_updates=65536,seed=42,dataset_identity='cpu-fixture',mode='weighted_replacement',weights=[1/16]*16)
        for _ in range(8192):plan.begin_batch();plan.commit_batch()
        schedule=UpdateSchedule(total_updates=65536,base_lrs=[.0002,.0002],kind='cosine',warmup_updates=1024);schedule.apply(optimizer,8192)
        method=method_source()[1]
        c={'config':config,'arguments':{'weight_decay':.0001},'split_indices':{'train':list(range(16))},'dataset_identity':'cpu-fixture',
            'tokenizer':{'cpu_fixture':True},'diagnostic_target_pair':{'diffusion_loss_source_sha256':hashlib.sha256(method.encode()).hexdigest()}}
        saved=api.trainable_state_dict(model);ref={'source_learned_sha256':model_sha(saved),'frozen_sha256':frozen_digest(model,saved)}
        pstate=plan.state_dict();batches=[]
        for _ in range(2):batches.append(plan.begin_batch());plan.commit_batch()
        plan.load_state_dict(pstate)
        suffix={'batches':np.asarray(batches),'lengths':np.array([8,8]),'lrs':np.array([schedule.rates(8192+i) for i in range(2)]),'valid_dimensions':np.ones((2,8))*128}
        class Zip(dict):
            @property
            def files(self):return list(self)
            def __enter__(self):return self
            def __exit__(self,*a):return False
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temp=tempfile.TemporaryDirectory(prefix='pose_driver_cpu_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        original_snapshot=pose_runtime.snapshot;original_current=pose_runtime.current_state;original_port=pose_runtime.zero_port
        def cpu_snapshot(b,p,o,plan,s,c,l,include):return original_snapshot(b,p,o,plan,s,c,l,False)
        def cpu_current(b,p,o,plan,s,l,include):return original_current(b,p,o,plan,s,l,False)
        with temp as folder:
            folder=Path(folder);source=folder/'runs/oxe-target-short-231297/sample';source.mkdir(parents=True)
            base=folder/'oxe_target_pair_short_v1';base.mkdir();(base/'selection.json').write_bytes((BASE/'selection.json').read_bytes())
            save_checkpoint(source/'latest_step.pt',model_state=saved,optimizer=optimizer,plan=plan,schedule=schedule,contract=c,include_cuda=False)
            payload=torch.load(source/'latest_step.pt',weights_only=False,map_location='cpu')
            before=sha(source/'latest_step.pt')
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,{'models':model_module,'train':api,'resume_constants':constructor}))
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value=c['tokenizer']))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
                stack.enter_context(patch.object(nn.Module,'cuda',lambda self,*a,**kw:self))
                for name in ['synchronize','reset_peak_memory_stats']:stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.object(torch.cuda,'max_memory_reserved',return_value=0))
                stack.enter_context(patch.object(pose_runtime,'zero_port',side_effect=lambda device,width:original_port('cpu',width)))
                stack.enter_context(patch.object(pose_runtime,'snapshot',side_effect=cpu_snapshot))
                stack.enter_context(patch.object(pose_runtime,'current_state',side_effect=cpu_current))
                stack.enter_context(patch.object(driver,'np',SimpleNamespace(load=lambda *a:Zip(suffix))))
                stack.enter_context(patch.object(driver,'replay_and_features',return_value={'cpu_fixture_only':True}))
                with redirect_stdout(io.StringIO()):
                    results=[driver.run_arm(folder,role,payload,data,folder,ref,stats) for role in ['mask_control','real_pose']]
            a,b=results
            self.assertTrue(all(r['restore_base_port_adam_plan_schedule_rng_exact'] for r in results))
            self.assertEqual(a['noise_stream'],b['noise_stream']);self.assertEqual(a['initial_port_sha256'],b['initial_port_sha256'])
            self.assertEqual(a['source_optimizer_sha256_before_augmentation'],b['source_optimizer_sha256_before_augmentation'])
            for key in ['loss','pose_loss','gripper_loss','base_gradient_sha256_before_clipping']:self.assertEqual(a['rows'][0][key],b['rows'][0][key])
            self.assertEqual(a['rows'][0]['pose_value_grad_norm'],0);self.assertGreater(b['rows'][0]['pose_value_grad_norm'],0)
            self.assertEqual(sha(source/'latest_step.pt'),before)
            self.assertEqual(sum(r['optimizer_calls'] for r in results),6)


if __name__=='__main__':unittest.main(verbosity=2)
