"""Actual fixed8 runner/control flow with real numeric targets and CPU toy encoders."""
import ast
from contextlib import ExitStack,redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/oxe_fixed8_learnability_v1'
FORMAL=ROOT/'deployment/oxe_current_pose_formal_v1';PREF=ROOT/'deployment/oxe_current_pose_preflight_v1';BASE=ROOT/'deployment/oxe_target_pair_short_v1'
sys.path[:0]=[str(BUNDLE),str(FORMAL),str(PREF),str(BASE),str(BASE/'code')]
import qualification as q
import engine
import fixed_inputs
from pose_runtime import zero_port,digest,NoiseLog
from step_training_state import UpdateSchedule,capture_rng
from oxe_nominal_target_store import NominalTargetStore
from instrument import method_source


def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def temporary(prefix):
    parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);t=tempfile.TemporaryDirectory(prefix=prefix,dir=parent)
    assert Path(t.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(t.name).resolve().parents
    return t


def toy_bundle(root):
    fixture=module('fixed8_native_functions',ROOT/'tests/test_oxe_current_pose_preflight.py')
    functions=fixture.native_functions({'policy_loss','collate_batch'});noise=NoiseLog()
    class Vision(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Linear(3,3)
        def forward(self,pixel_values):return SimpleNamespace(last_hidden_state=self.proj(pixel_values)[:,None])
    class Text(nn.Module):
        def __init__(self):super().__init__();self.embed=nn.Embedding(256,2)
        def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=self.embed(input_ids))
    class Adapter(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(5,1024),nn.Dropout(.1))
        def forward(self,v,t,m):return self.net(torch.cat([v[:,0],(t*m[...,None]).sum(1)/m.sum(1,keepdim=True)],-1))
    class Model(fixture.Toy):
        def __init__(self):
            super().__init__({});self.adapter=Adapter();self.vision_encoder=Vision();self.text_encoder=Text()
            for e in [self.vision_encoder,self.text_encoder]:
                e.requires_grad_(False);e.eval()
            self.max_normalized_position=8.;self.clip_denoised=True
            betas=torch.linspace(.0001,.02,100);a=1-betas;ab=a.cumprod(0);previous=torch.cat([torch.ones(1),ab[:-1]])
            self.alpha_bars.copy_(ab)
            for k,v in {'alphas':a,'posterior_variance':betas*(1-previous)/(1-ab),
                'posterior_mean_x0':betas*previous.sqrt()/(1-ab),'posterior_mean_xt':(1-previous)*a.sqrt()/(1-ab)}.items():self.register_buffer(k,v)
        def train(self,mode=True):
            super().train(mode);self.vision_encoder.eval();self.text_encoder.eval();return self
        def get_context_vector(self,images,input_ids,mask):
            with torch.no_grad():v=self.vision_encoder(images).last_hidden_state;t=self.text_encoder(input_ids,mask).last_hidden_state
            return self.adapter(v,t,mask)
    ns={'torch':torch,'Tuple':tuple,'DIAG_HOOK':noise,'extract':lambda v,t,x:v.gather(0,t).reshape(len(t),1,1)}
    exec(compile(method_source()[1],'<native-instrumented-diffusion-loss>','exec'),ns);Model.diffusion_loss=ns['diffusion_loss']
    tree=ast.parse((BASE/'code/models.py').read_text(encoding='utf-8'))
    cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RobotAdapterModel')
    sample=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='sample');sample=copy.deepcopy(sample)
    # Actual production sampler, with its original100steps, running on toy decoder/encoders.
    mod=ast.Module(body=[sample],type_ignores=[]);n={'torch':torch,'F':F}
    exec(compile(ast.fix_missing_locations(mod),'<native100step-sampler>','exec'),n);Model.sample=n['sample']
    def tokenizer(texts,**kwargs):
        lengths=[min(len(s.split())+2,77) for s in texts];width=max(lengths)
        ids=torch.zeros(len(texts),width,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,(s,length) in enumerate(zip(texts,lengths)):
            ids[i,:length]=torch.tensor([(ord(c)%255)+1 for c in (s+'x'*length)[:length]])
            mask[i,:length]=1
        return {'input_ids':ids,'attention_mask':mask}
    api=SimpleNamespace(policy_loss=functions['policy_loss'],collate_batch=functions['collate_batch'],
        tokenise=lambda t,texts,device:t(texts),trainable_state_dict=lambda m:{n:p.detach().cpu() for n,p in m.named_parameters() if p.requires_grad})
    oldselection=q.read(BASE/'selection.json');textmaps={}
    for role in ['train','validation']:
        store=NominalTargetStore(ROOT/('reports/oxe_nominal_targets_20261025' if role=='train' else 'reports/oxe_nominal_validation_20261025'),partition=role)
        selected={x['window_index'] for x in oldselection[role]};textmap={}
        name='training_original_instructions.jsonl' if role=='train' else 'validation_original_instructions.jsonl'
        for line in (store.root/name).read_text().splitlines():
            row=json.loads(line)
            if row['window_index'] in selected:textmap[row['window_index']]=row['instruction']
        class Part:
            def __init__(self,store,text):self.store=store;self.original_text=text
            def __getitem__(self,i):
                v=self.store.get(i)
                return self.original_text[i],torch.tensor([i/145812,.1,1.]),torch.from_numpy(v['current_features']),torch.from_numpy(v['actions']),torch.from_numpy(v['supervision_mask'])
        textmaps[role]=Part(store,textmap)
    data=SimpleNamespace(training=textmaps['train'],validation=textmaps['validation'])
    torch.manual_seed(42);model=Model();port=zero_port('cpu')
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    groups=[[p for n,p in named if not n.startswith('gripper_head.')],[p for n,p in named if n.startswith('gripper_head.')],list(port.parameters())]
    optimizer=torch.optim.AdamW([{'params':g,'lr':.0002} for g in groups],weight_decay=.0001)
    sum(p.square().mean() for group in groups for p in group).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    schedule=UpdateSchedule(total_updates=65536,base_lrs=[.0002]*3,kind='cosine',warmup_updates=1024);schedule.apply(optimizer,10240)
    sourcepath=root/'source_state.pt';sourcepath.write_bytes(b'Immutable CPU source fixture')
    return {'root':root,'model':model,'port':port,'optimizer':optimizer,'schedule':schedule,'source':{
        'rng':capture_rng(include_cuda=False),'contract_sha256':q.read(BUNDLE/'protocol.json')['source_identity']['contract_sha256']},
        'data':data,'api':api,'tokenizer':tokenizer,'stats':q.read(PREF/'state_stats.json'),
        'tokenizer_identity':{'cpu_fixture':True},'keys':api.trainable_state_dict(model),'noise':noise,'source_path':sourcepath,
        'source_file_size_mtime':[sourcepath.stat().st_size,sourcepath.stat().st_mtime_ns],
        'config':{'model':{'diffusion_prediction_type':'sample','scope':'cpu-toy-only'}}}


class Tests(unittest.TestCase):
    def test_actual_runner_512_updates_original_sampler_and_saved_verifier_cpu(self):
        import step_training_runner
        driver=module('actual_fixed8_driver',BUNDLE/'run_check.py')
        import verify_output
        torch.set_num_threads(1)
        with temporary('fixed8_actual_cpu_') as folder:
            root=Path(folder);base=root/'oxe_target_pair_short_v1';base.mkdir();shutil.copyfile(BASE/'selection.json',base/'selection.json')
            bundle=toy_bundle(root);bound={'scope':'cpu-only-source-binding-fixture'}
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
                for name in ['manual_seed','manual_seed_all','reset_peak_memory_stats']:stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.object(torch.cuda,'max_memory_reserved',return_value=0))
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value={'cpu_fixture':True}))
                stack.enter_context(patch.object(step_training_runner,'frozen_digest',return_value=q.read(BUNDLE/'protocol.json')['source_identity']['frozen_state_sha256']))
                cache=fixed_inputs.prepare(root,bundle)
                before=engine.sample(bundle,cache,[1101,1102,1103]);targets=cache['actions'].numpy();masks=cache['masks'].numpy()
                static=np.zeros_like(targets);static[...,6]=1
                archive=root/'runs/oxe-pose-pair-232587/evaluation';archive.mkdir(parents=True)
                np.savez_compressed(archive/'predictions_and_targets.npz',targets=targets,masks=masks,
                    real_pose=before,static=np.repeat(static[None],3,0))
                source_binding=root/'source_binding.json';source_binding.write_text(json.dumps(bound))
                original_read=verify_output.read
                stack.enter_context(patch.object(verify_output,'read',side_effect=lambda p:bound if Path(p).name=='source_binding.json' else original_read(p)))
                stack.enter_context(patch.object(driver,'admit',return_value=bound))
                stack.enter_context(patch.object(driver,'attach',return_value=(None,SimpleNamespace(initialize_runtime=lambda:None))))
                stack.enter_context(patch.object(driver,'load_source',return_value=bundle))
                stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'cpu_fixture'}))
                output=root/'out';stack.enter_context(patch.object(sys,'argv',['run_check.py','--root',str(root),'--output',str(output)]))
                with redirect_stdout(io.StringIO()):driver.main()
                checked=verify_output.verify(root,output)
                self.assertTrue(checked['output_verified']);self.assertEqual(checked['updates'],512)
                report=q.read(output/'report.json');self.assertEqual(report['additional_optimizer_calls'],512)
                self.assertTrue(report['cached_training_guard']['passed']);self.assertTrue(report['source_prediction_replay_exact'])
                self.assertEqual(len(report['history']),512);self.assertEqual(len(report['noise_stream']),512)
                self.assertTrue(all(r['indices']==cache['indices'] for r in report['history']))
                self.assertTrue(all(not p.requires_grad and p.grad is None for e in [bundle['model'].vision_encoder,bundle['model'].text_encoder] for p in e.parameters()))
                report['additional_optimizer_calls']=513;(output/'report.json').write_text(json.dumps(report))
                with self.assertRaises(AssertionError):verify_output.verify(root,output)

    def test_cache_guard_rejects_altered_features_before_optimizer(self):
        import step_training_runner
        with temporary('fixed8_cache_guard_cpu_') as folder:
            root=Path(folder);base=root/'oxe_target_pair_short_v1';base.mkdir();shutil.copyfile(BASE/'selection.json',base/'selection.json')
            bundle=toy_bundle(root)
            with patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self),patch.object(step_training_runner,'tokenizer_identity',return_value={'cpu_fixture':True}):
                cache=fixed_inputs.prepare(root,bundle)
                before=digest(bundle['optimizer'].state_dict());cache['train_visual']=cache['train_visual']+.01
                with self.assertRaisesRegex(AssertionError,'Cached frozen features changed'):engine.cache_guard(bundle,cache)
                self.assertEqual(digest(bundle['optimizer'].state_dict()),before)

    def test_fit_gate_real_masks_and_rejects_static_as_fit(self):
        p=q.read(BUNDLE/'protocol.json')['fit_gate']
        with np.load(BUNDLE/'numeric_reference.npz') as z:
            targets=z['targets'];masks=z['masks'];perfect=np.repeat(targets[None],3,0)
        static=np.zeros_like(perfect);static[...,6]=1
        good=engine.metrics({'before':static,'after':perfect,'static':static},targets,masks)
        self.assertTrue(engine.decision(good,p)['fixed8_pose_fit_gate_passed'])
        bad=engine.metrics({'before':static,'after':static,'static':static},targets,masks)
        self.assertFalse(engine.decision(bad,p)['fixed8_pose_fit_gate_passed'])

    def test_actual_check_only_binding_rejects_changed_source_identity(self):
        driver=module('fixed8_check_only_driver',BUNDLE/'run_check.py');p=q.read(BUNDLE/'protocol.json')
        with temporary('fixed8_metadata_cpu_') as folder:
            root=Path(folder);bundle=root/BUNDLE.name;bundle.mkdir();formal=root/FORMAL.name;formal.mkdir()
            for n in [line.split('  ',1)[1] for line in (BUNDLE/'SHA256SUMS').read_text().splitlines()]+['SHA256SUMS']:shutil.copyfile(BUNDLE/n,bundle/n)
            for n in p['formal_bundle_files']:shutil.copyfile(FORMAL/n,formal/n)
            source=root/'runs/oxe-pose-pair-232587';(source/'real_pose').mkdir(parents=True);(source/'evaluation').mkdir()
            receipt=dict(p['source_identity'],passed=True,training_jobs=['232587'])
            (source/'real_pose/arm_receipt.json').write_text(json.dumps(receipt))
            (source/'pair_report.json').write_text(json.dumps({'execution_complete':True}))
            (source/'evaluation/report.json').write_text(json.dumps({'execution_complete':True,'optimizer_updates':0,'generation_steps':100}))
            (source/'evaluation/predictions_and_targets.npz').write_bytes(b'Metadata fixture only, never deserialize')
            with ExitStack() as stack:
                stack.enter_context(patch.object(q,'HERE',bundle))
                stack.enter_context(patch.object(q,'attach',return_value=(SimpleNamespace(admit=lambda *a,**kw:None),None)))
                stack.enter_context(patch.object(q.subprocess,'check_output',return_value='1000000 fixture'))
                stack.enter_context(patch.object(q.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
                stack.enter_context(patch.object(torch.cuda,'init',side_effect=AssertionError('metadata must not initialize CUDA')))
                stack.enter_context(patch.object(sys,'argv',['run_check.py','--root',str(root),'--check-only']))
                with redirect_stdout(io.StringIO()) as text:driver.main()
                self.assertIn('metadata_bound',text.getvalue());self.assertTrue((bundle/'source_binding.json').exists())
                receipt['base_learned_sha256']='wrong';(source/'real_pose/arm_receipt.json').write_text(json.dumps(receipt))
                with self.assertRaisesRegex(AssertionError,'base_learned_sha256'):q.admit(root,space=False)


if __name__=='__main__':unittest.main(verbosity=2)
