"""Exercise the real evaluator entry with CPU-only stand-ins, not real policies."""
import copy
from contextlib import contextmanager,ExitStack,redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'deployment/oxe_target_pair_short_v1'
REPAIR=ROOT/'deployment/oxe_target_pair_eval_repair_v1'
sys.path[:0]=[str(REPAIR),str(BASE),str(BASE/'code')]
from repair_core import read,sha,fixed_source,validate_selection,load_evaluator
from completed_policy_export import model_sha,contract_sha


class RepairTests(unittest.TestCase):
    def test_original_field_is_absent_fix_uses_frozen_actual_schema(self):
        selection=read(BASE/'selection.json');ref=read(BASE/'reference_contract.json')
        with self.assertRaises(KeyError):selection['sampling_steps']
        self.assertEqual(validate_selection(selection,ref)['generation_steps'],100)
        for key,value in [('generation_steps',99),('generation_batch_size',4),('noise_seeds',[1101])]:
            bad=copy.deepcopy(selection);bad[key]=value
            with self.assertRaises(AssertionError):validate_selection(bad,ref)
        bad=copy.deepcopy(selection);bad['sampling_steps']=bad.pop('generation_steps')
        with self.assertRaises(KeyError):validate_selection(bad,ref)

    def test_original_archive_and_instrumentation_identity(self):
        c=read(REPAIR/'repair_contract.json')
        for name,digest in c['original_bundle_files'].items():self.assertEqual(sha(BASE/name),digest)
        source=(BASE/'evaluate.py').read_text(encoding='utf-8');new=fixed_source(source)
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(),c['fixed_evaluator_sha256'])
        self.assertIn("selection['sampling_steps']",source);self.assertNotIn("selection['sampling_steps']",new)
        self.assertEqual(new.count('verify_pair(a.output)'),1)
        self.assertEqual(new.count('model.sample(contexts[lo:lo+8],current[lo:lo+8])'),1)
        self.assertEqual(new.count('summarize(outputs,targets,masks,selection)'),1)
        with self.assertRaises(AssertionError):fixed_source(source.replace("selection['sampling_steps']","selection['different_steps']"))

    def test_real_main_old_failure_then_new_directory_complete_and_recomputed(self):
        from oxe_nominal_target_store import NominalTargetStore
        import step_training_runner
        reference=read(BASE/'reference_contract.json');selection=read(BASE/'selection.json')
        stores={role:NominalTargetStore(ROOT/'reports'/folder,partition=role) for role,folder in
            [('train','oxe_nominal_targets_20261025'),('validation','oxe_nominal_validation_20261025')]}
        counters=dict(constructors=0,vision=0,text=0,samples=0)
        class Data:
            def __init__(self,unused_data,unused_store,partition):
                self.store=stores[partition];self.nominal_target_binding=self.store.binding
            def __getitem__(self,i):
                item=self.store.get(i)
                return ('CPU fixture',torch.zeros(3,2,2),torch.from_numpy(item['current_features']),
                    torch.from_numpy(item['actions']),torch.from_numpy(item['supervision_mask']))
        class Vision(nn.Module):
            def forward(self,pixel_values):
                counters['vision']+=1
                return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values),3,4))
        class Text(nn.Module):
            def forward(self,input_ids,attention_mask):
                counters['text']+=1
                return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids),77,4))
        class Adapter(nn.Module):
            def __init__(self):super().__init__();self.weight=nn.Parameter(torch.zeros(1))
            def forward(self,vision,text,mask):return vision.mean(1)*self.weight
        class Model(nn.Module):
            def __init__(self,config):
                super().__init__();counters['constructors']+=1;self.vision_encoder=Vision();self.text_encoder=Text();self.adapter=Adapter()
            def sample(self,context,current):
                counters['samples']+=1;p=torch.zeros(len(context),16,8);p[...,6]=1;p[...,0]=float(self.adapter.weight)*.01
                return p
        class Tokenizer:
            @classmethod
            def from_pretrained(cls,*args,**kw):return cls()
            def __call__(self,texts,**kw):return dict(input_ids=torch.ones(len(texts),77,dtype=torch.long),attention_mask=torch.ones(len(texts),77,dtype=torch.long))
        @contextmanager
        def constants(model_class,*,on_restored,**kw):on_restored({'fixture_only':True});yield
        modules={}
        for name,attrs in [('models',dict(RobotAdapterModel=Model)),('transformers',dict(CLIPTokenizer=Tokenizer)),
            ('oxe_nominal_dataset',dict(NominalOXEDataset=Data)),('train',dict(trainable_state_dict=lambda m:m.state_dict())),
            ('resume_constants',dict(training_constant_constructor=constants)),('verify',dict(verify_pair=lambda p:{'fixture_only':True})),
            ('tensorflow',dict(config=SimpleNamespace(set_visible_devices=lambda *a:None,threading=SimpleNamespace(
                set_intra_op_parallelism_threads=lambda *a:None,set_inter_op_parallelism_threads=lambda *a:None))))]:
            modules[name]=ModuleType(name);modules[name].__dict__.update(attrs)
            modules[name].__spec__=__import__('importlib.machinery',fromlist=['ModuleSpec']).ModuleSpec(name,loader=None)
        states={t:{'adapter.weight':torch.tensor([float(i)])} for t,i in [('epsilon',1),('sample',2)]}
        payloads={}
        for target in states:
            c=copy.deepcopy(reference);c['config']['model']['diffusion_prediction_type']=target
            payloads[target]={'contract':c,'contract_sha256':contract_sha(c),'model':states[target]}
        tmp_parent=ROOT/'.tmp';tmp_parent.mkdir(exist_ok=True)
        # Verify the only auto-cleanup target's absolute parent is inside this workspace.
        temp=tempfile.TemporaryDirectory(prefix='target_eval_regression_',dir=tmp_parent)
        assert Path(temp.name).resolve().parent==tmp_parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as directory:
            work=Path(directory);training=work/'oxe-target-short-231297';training.mkdir();output=work/'corrected_evaluation'
            sentinels={}
            for target,state in states.items():
                arm=training/target;arm.mkdir();path=arm/'latest_step.pt';path.write_bytes(b'CPU fixture only')
                sentinels[path]=sha(path)
                (arm/'arm_receipt.json').write_text(json.dumps({'final_learned_sha256':model_sha(state)}))
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,modules))
                stack.enter_context(patch.object(step_training_runner,'frozen_digest',return_value=reference['frozen_state_sha256']))
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value=reference['tokenizer']))
                stack.enter_context(patch.object(torch,'__version__','2.5.1+cu118'));stack.enter_context(patch.object(torch.version,'cuda','11.8'))
                stack.enter_context(patch.object(torch,'set_num_interop_threads'))
                stack.enter_context(patch.object(torch.cuda,'is_available',return_value=True))
                stack.enter_context(patch.object(torch.cuda,'get_device_name',return_value='NVIDIA A40'))
                for name in ['manual_seed','manual_seed_all','empty_cache']:stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
                stack.enter_context(patch.object(nn.Module,'cuda',lambda self,*a,**kw:self))
                loader=stack.enter_context(patch.object(torch,'load',side_effect=lambda p,**kw:copy.deepcopy(payloads[Path(p).parent.name])))
                stack.enter_context(patch.object(torch.optim,'AdamW',side_effect=AssertionError('Evaluation must not construct an optimizer')))
                stack.enter_context(patch.dict(__import__('os').environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
                stack.enter_context(patch.object(sys,'argv',['evaluate.py','--root',str(work),'--output',str(training)]))
                original={'__name__':'old_eval_fixture','__file__':str(BASE/'evaluate.py')}
                exec(compile((BASE/'evaluate.py').read_text(encoding='utf-8'),str(BASE/'evaluate.py'),'exec'),original)
                with self.assertRaisesRegex(KeyError,'sampling_steps'):original['main']()
                old_directory=training/'evaluation';self.assertTrue(old_directory.is_dir());self.assertFalse(list(old_directory.iterdir()))
                self.assertEqual(counters['constructors'],0);self.assertEqual(loader.call_count,0)
                ns,source=load_evaluator(BASE,output)
                with redirect_stdout(io.StringIO()):ns['main']()
                self.assertTrue(ns['verify_evaluation'](training,output)['verified'])
                r=read(output/'report.json');self.assertTrue(r['execution_complete']);self.assertEqual(r['optimizer_updates'],0)
                self.assertEqual(counters,dict(constructors=1,vision=30,text=30,samples=90));self.assertEqual(loader.call_count,2)
                self.assertFalse(list(old_directory.iterdir()))
                for path,digest in sentinels.items():self.assertEqual(sha(path),digest)
                self.assertIsNone(r['metrics']['by_source']['validation']['Language Table']['sample']['path_rotation_deg'])
                self.assertEqual(r['checkpoint_metadata']['sample']['trained_native_target'],'sample')


if __name__=='__main__':unittest.main(verbosity=2)
