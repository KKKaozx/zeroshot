"""CPU integration of new target-only instrumentation; no real encoder or CUDA."""
from __future__ import annotations
import argparse
import ast
import copy
from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deployment/oxe_target_pair_short_v1'
sys.path[:0]=[str(BUNDLE),str(BUNDLE/'code')]
from protocols import read,sha,validate_diagnostic
from instrument import method_source,loop_source
from hook import Hook
from metrics import metrics,summarize
from completed_policy_export import contract_sha


def native_functions(path,names,extra=None):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    assert len(selected)==len(names)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+selected,type_ignores=[])
    ns={'torch':torch,'nn':nn,'argparse':argparse,'Path':Path,'__file__':str(path),'np':np}
    ns.update(extra or {});exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns);return ns


class Tiny(nn.Module):
    def __init__(self,target):
        super().__init__();self.chunk_size=16;self.action_dim=8;self.num_diffusion_steps=100
        self.diffusion_prediction_type=target;self.decoder_type='diffusion';self.separate_gripper_head=True
        self.register_buffer('alpha_bars',torch.linspace(.999,.001,100))
        self.net=nn.Sequential(nn.Linear(10,12),nn.Dropout(.1),nn.Linear(12,7))
        self.grip=nn.Linear(3,16);self.gripper_target_mode='state';self.gripper_change_weight=1.
        self.balanced_gripper_loss=False;self.gripper_loss_weight=.25
    def diffusion_decoder(self,x,t,c):return self.net(torch.cat([x,c[:,None].expand(-1,16,-1)],-1))
    def predict_gripper_logits(self,c,actions,current):return self.grip(c)


class PairTests(unittest.TestCase):
    def test_immutable_source_profiles_and_explicit_native_arguments(self):
        ref=read(BUNDLE/'reference_contract.json');p=read(BUNDLE/'pair_protocol.json')
        self.assertEqual(contract_sha(ref),p['source_reference_contract_sha256'])
        for name,digest in ref['source_code_sha256'].items():self.assertEqual(sha(BUNDLE/'code'/name),digest)
        a,b=[read(BUNDLE/f'profile_{t}.json') for t in ['epsilon','sample']]
        self.assertEqual(a['tensors'],b['tensors']);self.assertEqual(a['key_order'],b['key_order'])
        self.assertEqual(a['schedule_values'],b['schedule_values']);c=copy.deepcopy(b['config'])
        c['model']['diffusion_prediction_type']='epsilon';self.assertEqual(c,a['config'])
        self.assertIn('metadata_derivation',b)
        parser=native_functions(BUNDLE/'code/train.py',{'parse_args'})['parse_args']
        from cartesian_action_contract import representation_for_frame
        build=native_functions(BUNDLE/'code/train.py',{'build_config'},dict(math=__import__('math'),
            POSITION_SCALE_METERS=.1,ACTION_DIM=8,representation_for_frame=representation_for_frame))['build_config']
        from run_arm import options
        from oxe_formal_protocol import load_and_validate
        for target in ['epsilon','sample']:
            args=options(SimpleNamespace(parse_args=parser),Path('/projects/_ssd/Zeroshot'),Path('unused-output'),target)
            # Windows Path rewrites a POSIX-looking path; cluster uses actual POSIX paths.
            args.model_name=ref['arguments']['model_name'];args.dino_assets=ref['arguments']['dino_assets']
            # Real source protocol/binding and local immutable input files, no model.
            args.split_manifest=str(ROOT/'deployment/oxe_dino_main_preflight_v1/inputs/split_manifest.json')
            args.original_instruction_catalog=str(ROOT/'deployment/oxe_dino_main_preflight_v1/inputs/instruction_catalog.json')
            result=validate_diagnostic(args,None,load_and_validate)
            self.assertEqual(result['source_reference_admission'],ref['offline_training_protocol'])
            self.assertEqual(args.diffusion_prediction_type,target)
            config=build(args);config['action']['nominal_target_binding']=ref['config']['action']['nominal_target_binding']
            config['action']['source_position_to_meters_verified']=False
            config['model'].update(diffusion_architecture='multiscale',diffusion_down_dims=[256,512,1024],
                diffusion_step_embed_dim=128,diffusion_kernel_size=5)
            native=copy.deepcopy(ref['config']);native['model']['diffusion_prediction_type']=target
            self.assertEqual(config,native)
            args.adapter_pooling='mean'
            with self.assertRaises(AssertionError):validate_diagnostic(args,None,load_and_validate)

    def test_real_schedule_prefix_and_selection_exposure(self):
        from oxe_nominal_target_store import NominalTargetStore,SOURCE_SCHEMAS
        from step_training_state import ResumableBatchPlan,UpdateSchedule
        ref=read(BUNDLE/'reference_contract.json');train=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
        val=NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')
        source=[train.episodes[int(row[0])]['source'] for row in train.windows];counts=Counter(source)
        indices=ref['split_indices']['train']
        plan=ResumableBatchPlan(indices,batch_size=8,total_updates=65536,seed=42,dataset_identity=ref['dataset_identity'],
            mode='weighted_replacement',weights=[1/counts[source[i]] for i in indices])
        schedule=UpdateSchedule(total_updates=65536,base_lrs=[.0002,.0002],kind='cosine',warmup_updates=1024)
        with np.load(BUNDLE/'reference_prefix.npz') as z:
            for i in range(8192):
                batch=plan.begin_batch();self.assertEqual(batch,z['batches'][i,:z['lengths'][i]].tolist())
                # Windows math.cos differs from Linux libm by <=one float64 ULP.
                np.testing.assert_allclose(schedule.rates(i),z['lrs'][i],rtol=0,atol=6e-20)
                self.assertEqual(train.masks[batch].sum((0,1)).tolist(),z['valid_dimensions'][i].tolist())
                plan.commit_batch()
            exposure=Counter(z['batches'].reshape(-1).tolist())
        selection=read(BUNDLE/'selection.json');self.assertEqual(len(selection['train'])+len(selection['validation']),120)
        seen={}
        for role,store in [('train',train),('validation',val)]:
            seen[role]=set()
            for row in selection[role]:
                _,start,_,e,_=store._row(row['window_index'])
                for k in ['source','group_id','episode_key','payload_sha256']:self.assertEqual(e[k],row[k])
                self.assertEqual(int(start),row['input_row']);seen[role].add((row['source'],row['group_id']))
                if role=='train':self.assertGreater(exposure[row['window_index']],0);self.assertEqual(exposure[row['window_index']],row['prefix_presentations'])
        self.assertFalse(seen['train']&seen['validation'])
        self.assertEqual(Counter(r['source'] for r in selection['train']),dict.fromkeys(['Bridge','Language Table','BC-Z','Fractal'],8))
        self.assertEqual(Counter(r['source'] for r in selection['validation']),{'Bridge':64,'Language Table':8,'BC-Z':8,'Fractal':8})

    def test_instrumentation_native_output_rng_first_backward_and_streams(self):
        tree=ast.parse((BUNDLE/'code/models.py').read_text(encoding='utf-8'))
        fn=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='diffusion_loss'))
        original_source=ast.unparse(ast.Module(body=[fn],type_ignores=[]))
        _,instrumented=method_source();changed=ast.parse(instrumented).body[0]
        changed.body=[n for n in changed.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='DIAG_HOOK')]
        self.assertEqual(ast.dump(fn,include_attributes=False),ast.dump(changed,include_attributes=False))
        compile(loop_source()[1],'<effective-runner>','exec')
        loss=native_functions(BUNDLE/'code/train.py',{'policy_loss'})['policy_loss']
        extract=lambda values,t,target:values.gather(0,t).reshape(t.shape[0],1,1)
        torch.set_num_threads(1);streams=[]
        torch.manual_seed(5);actions=torch.randn(2,16,8);context=torch.randn(2,3);current=torch.zeros(2,2)
        masks=torch.ones_like(actions);masks[0,12:]=0;masks[1,:,2:7]=0
        for target in ['epsilon','sample']:
            torch.manual_seed(42);m=Tiny(target);plain=copy.deepcopy(m)
            h=Hook(target,'loop-fixture','method-fixture');h.pending=1
            h.schedule['valid_dimensions'][0]=masks.sum((0,1)).numpy()
            ns={'torch':torch,'Tuple':tuple,'extract':extract,'DIAG_HOOK':h}
            exec(instrumented,ns);instrumented_fn=ns['diffusion_loss']
            plain_ns={'torch':torch,'Tuple':tuple,'extract':extract};exec(original_source,plain_ns)
            torch.manual_seed(123);before=torch.get_rng_state().clone()
            x=plain_ns['diffusion_loss'](plain,actions,context,current);after=torch.get_rng_state().clone()
            torch.set_rng_state(before);y=instrumented_fn(m,actions,context,current)
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for u,v in zip(x,y):self.assertTrue(torch.equal(u,v))
            api=SimpleNamespace(policy_loss=loss);h.observe(m,api,y,actions,current,masks,nn.MSELoss())
            native=loss(m,y,actions,nn.MSELoss(),current_grippers=current,supervision_masks=masks)[0]
            legacy=h.reference_values
            legacy[0].backward(retain_graph=True);norm=nn.utils.clip_grad_norm_(list(m.parameters()),1.)
            h.first={**{k:float(v.detach()) for k,v in zip(['loss','pose_loss','gripper_loss'],legacy)},'gradient_norm':float(norm)}
            opt=torch.optim.AdamW(m.parameters(),lr=.0001);opt.zero_grad(set_to_none=True)
            versions=[p._version for p in m.parameters()];rng=torch.get_rng_state().clone()
            h.audit_first(m,opt,list(m.named_parameters()))
            self.assertFalse(opt.state);self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertEqual(versions,[p._version for p in m.parameters()]);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            native.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            opt.step();h.commit(1);streams.append(h.state()['stream_sha256'])
        self.assertEqual(streams[0],streams[1])

    def test_contract_only_target_and_explicit_diagnostic_metadata(self):
        ref=read(BUNDLE/'reference_contract.json');p=read(BUNDLE/'pair_protocol.json')
        plan=SimpleNamespace(total_updates=65536,batch_size=8)
        for target in ['epsilon','sample']:
            base=copy.deepcopy(ref);base['arguments']['diffusion_prediction_type']=target;base['config']['model']['diffusion_prediction_type']=target
            base['offline_training_protocol']={'source_reference_admission':ref['offline_training_protocol']}
            model=SimpleNamespace(diffusion_prediction_type=target);api=SimpleNamespace(trainable_state_dict=lambda m:{})
            with patch('completed_policy_export.model_sha',return_value=p['first_initial_learned_sha256']):
                h=Hook(target,'loop','method');h.bind(model,api,copy.deepcopy(base),plan)
                for mutate in [lambda c:c['arguments'].__setitem__('learning_rate',1e-5),
                    lambda c:c['config']['model'].__setitem__('adapter_pooling','mean'),
                    lambda c:c['config']['model'].__setitem__('current_pose_input',True),
                    lambda c:c['arguments'].__setitem__('diffusion_prediction_type','wrong')]:
                    bad=copy.deepcopy(base);mutate(bad)
                    with self.assertRaises(AssertionError):Hook(target,'loop','method').bind(model,api,bad,plan)

    def test_masked_metrics_no_lt_rotation_no_padding_endpoint(self):
        t=np.zeros((2,16,8),dtype=np.float32);t[...,6]=1;m=np.zeros_like(t)
        m[0,:3,:2]=1;m[1,:2,:7]=1
        pred=np.repeat(t[None],3,axis=0);pred[:,0,:3,0]=2;pred[:,0,:,2]=999
        pred[:,0,:,3:7]=0;pred[:,1,0,0]=3;pred[:,1,1,0]=4;pred[:,1,2:,0]=999
        lt=metrics(pred,t,m,[0]);self.assertEqual(lt['position_axes'],['x','y']);self.assertIsNone(lt['path_rotation_deg'])
        self.assertEqual(lt['path_position_normalized'],2.)
        bridge=metrics(pred,t,m,[1]);self.assertEqual(bridge['path_position_normalized'],3.5)
        self.assertEqual(bridge['endpoint_position_normalized'],4.);self.assertEqual(bridge['path_rotation_deg'],0.)
        m[1,0,3]=0
        with self.assertRaises(AssertionError):metrics(pred,t,m,[1])

    def test_real_selected_targets_metrics_and_bootstrap_group_count(self):
        from oxe_nominal_target_store import NominalTargetStore
        selection=read(BUNDLE/'selection.json');target=[];mask=[]
        for role,name in [('train','oxe_nominal_targets_20261025'),('validation','oxe_nominal_validation_20261025')]:
            store=NominalTargetStore(ROOT/'reports'/name,partition=role)
            for row in selection[role]:
                data=store.get(row['window_index']);target.append(data['actions']);mask.append(data['supervision_mask'])
        target=np.stack(target);mask=np.stack(mask)
        static=np.zeros_like(target);static[...,6]=1
        outputs={k:np.repeat(v[None],3,axis=0) for k,v in [('epsilon',static),('sample',target),('static',static)]}
        r=summarize(outputs,target,mask,selection)
        self.assertEqual(r['primary_bridge64']['sample']['path_position_normalized'],0)
        for role in ['train','validation']:
            self.assertIsNone(r['by_source'][role]['Language Table']['sample']['path_rotation_deg'])
        self.assertLessEqual(r['paired_bridge_group_bootstrap95']['path_position_normalized'][1],0)


if __name__=='__main__':unittest.main(verbosity=2)
