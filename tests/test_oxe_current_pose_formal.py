"""Actual bounded driver/evaluator flow with CPU models; no CUDA qualification."""
import copy
import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from contextlib import ExitStack,redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];FORMAL=ROOT/'deployment/oxe_current_pose_formal_v1'
PREF=ROOT/'deployment/oxe_current_pose_preflight_v1';BASE=ROOT/'deployment/oxe_target_pair_short_v1'
sys.path[:0]=[str(FORMAL),str(PREF),str(BASE),str(BASE/'code')]
import binding
import instrument_formal
import formal_state
import pose_runtime
from completed_policy_export import model_sha
from pair_metrics import summarize


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Tests(unittest.TestCase):
    def test_actual_check_only_binds_pinned_files_and_exact_resource_receipt_without_cuda(self):
        import shutil
        import pose_features
        driver=module('formal_check_only_driver',FORMAL/'run_formal.py')
        contract=binding.read(FORMAL/'formal_contract.json');resource=binding.read(FORMAL/'expected_preflight_receipt.json')
        protocol=binding.read(PREF/'protocol.json')
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temp=tempfile.TemporaryDirectory(prefix='formal_admission_cpu_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as folder:
            folder=Path(folder);pre=folder/PREF.name;pre.mkdir();base=folder/BASE.name;base.mkdir()
            formal=folder/FORMAL.name;formal.mkdir()
            for name in contract['qualified_preflight_files']:shutil.copyfile(PREF/name,pre/name)
            for line in (FORMAL/'SHA256SUMS').read_text().splitlines():
                name=line.split('  ',1)[1];shutil.copyfile(FORMAL/name,formal/name)
            shutil.copyfile(FORMAL/'SHA256SUMS',formal/'SHA256SUMS')
            for name in protocol['original_bundle_files']:
                target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
            archive=folder/'runs/oxe-target-short-eval-fixed-232091';archive.mkdir(parents=True)
            for name in protocol['evaluation_file_sha256']:shutil.copyfile(ROOT/'reports/oxe_target_eval_repair_232091/cluster'/name,archive/name)
            source=folder/'runs/oxe-target-short-231297/sample';source.mkdir(parents=True)
            (source/'latest_step.pt').write_bytes(b'CPU metadata fixture only, never deserialize')
            (source/'arm_receipt.json').write_text(json.dumps({'passed':True,'target':'sample','prefix_updates':8192,
                'final_learned_sha256':protocol['source_learned_sha256'],'checkpoint_contract_sha256':protocol['source_contract_sha256']}))
            groups={}
            for role in ['mask_control','real_pose']:
                groups[role]={'passed':True,'restore_second_step_exact':True,'restore_base_port_adam_plan_schedule_rng_exact':True,
                    'optimizer_calls':3,'logical_new_updates':2,'completed_update':8194,'new_port_parameter_count':18432,
                    'disk_full_checkpoints_written':0,'frozen_digest_unchanged':True,'trained_weights_retained_for_formal_use':False,
                    'peak_reserved_gib':resource['peak_reserved_gib'][role]}
            groups['mask_control']['zero_endpoint_replay']={'passed':True,'archived_x0_predictions_exact':True}
            report={'engineering_passed':True,'protocol_sha256':binding.sha(PREF/'protocol.json'),'physical_optimizer_calls':6,
                'logical_steps_per_group':2,'paired_initial_loss_base_gradients_and_noise_exact':True,'source_checkpoint_size_mtime_unchanged':True,
                'formal_training_submitted':False,'formal_code_released':False,'groups':groups,
                'estimated_pair_update_seconds_excluding_replay_eval_io':resource['estimated_pair_update_minutes_excluding_replay_eval_io']*60}
            p=folder/'runs/oxe-pose-preflight-232371/report.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(report))
            with ExitStack() as stack:
                stack.enter_context(patch.object(binding,'HERE',formal));stack.enter_context(patch.object(pose_features,'HERE',pre))
                stack.enter_context(patch.object(binding.subprocess,'check_output',return_value='1000000 fixture'))
                stack.enter_context(patch.object(binding.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
                stack.enter_context(patch.object(torch.cuda,'init',side_effect=AssertionError('metadata must not initialize CUDA')))
                stack.enter_context(patch.object(sys,'argv',['run_formal.py','--root',str(folder),'--check-only']))
                with redirect_stdout(io.StringIO()) as text:driver.main()
                self.assertIn('qualified_preflight_bound',text.getvalue());self.assertTrue((formal/'preflight_binding.json').is_file())
                report['groups']['real_pose']['peak_reserved_gib']+=1;p.write_text(json.dumps(report))
                with self.assertRaisesRegex(AssertionError,'receipt differs'):binding.admit(folder,space=False)

    def test_actual_promoted_driver_matches_engineering_and_explicit_resume_cpu(self):
        # Reuse the existing nonempty-Adam CPU source fixture, without changing it.
        fixture=module('passed_pose_cpu_fixture',ROOT/'tests/test_oxe_current_pose_preflight.py')
        source=textwrap.dedent(inspect.getsource(fixture.Tests.test_actual_driver_two_roles_six_updates_moments_and_resume_on_cpu))
        old="results=[driver.run_arm(folder,role,payload,data,folder,ref,stats) for role in ['mask_control','real_pose']]"
        assert source.count(old)==1
        source=source.replace(old,old+"\n                formal_results=exercise_formal(self,driver,folder,payload,data,ref,stats,results,api)")
        ns=dict(fixture.__dict__);ns['exercise_formal']=self.exercise_formal
        exec(compile(source,'<reuse-qualified-cpu-source-fixture>','exec'),ns)
        with redirect_stdout(io.StringIO()):ns['test_actual_driver_two_roles_six_updates_moments_and_resume_on_cpu'](self)

    @staticmethod
    def exercise_formal(test,driver,folder,payload,data,ref,stats,qualified,api):
        code=instrument_formal.source(driver)
        # Only shorten the loop horizon in this disposable CPU fixture.
        code=code.replace('10240','8194').replace('2048','2').replace('%1024==0','%1==0')
        driver.__dict__.update(FORMAL_CONTRACT_SHA=binding.sha(FORMAL/'formal_contract.json'),
            QUALIFIED_PREFLIGHT_REPORT={'groups':dict(zip(['mask_control','real_pose'],qualified))},QUALIFIED_PREFLIGHT_REPORT_SHA='cpu-fixture')
        exec(compile(code,'<bounded-formal-two-step-cpu-fixture>','exec'),driver.__dict__)
        native_save=formal_state.save
        def cpu_save(*a,**kw):return native_save(*a,**dict(kw,include_cuda=False))
        results=[]
        with patch.dict('os.environ',{'SLURM_JOB_ID':'cpu-fixture'}),patch.object(formal_state,'save',side_effect=cpu_save):
            for role,old in zip(['mask_control','real_pose'],qualified):
                out=folder/role;out.mkdir()
                r=driver.run_arm(folder,role,payload,data,out,ref,stats)
                p=torch.load(out/'latest_state.pt',map_location='cpu',weights_only=False)
                test.assertEqual(p['format'],formal_state.FORMAT)
                test.assertEqual(pose_runtime.digest(p['base']),old['final_state_digests']['base'])
                test.assertEqual(pose_runtime.digest(p['pose_port']),old['final_state_digests']['port'])
                test.assertEqual(pose_runtime.digest(p['optimizer']),old['final_state_digests']['optimizer'])
                test.assertEqual(p['extra']['noise_stream'],old['noise_stream'])
                for i,row in enumerate(p['extra']['rows']):
                    test.assertEqual({k:row[k] for k in old['rows'][i] if k!='measured_update_seconds'},
                        {k:v for k,v in old['rows'][i].items() if k!='measured_update_seconds'})
                # Stop after one committed formal save, then resume the same state.
                interrupted=folder/(role+'_interrupted');interrupted.mkdir()
                def save_then_stop(*a,**kw):
                    cpu_save(*a,**kw)
                    raise RuntimeError('CPU fixture intentional interruption after committed save')
                with patch.object(formal_state,'save',side_effect=save_then_stop):
                    with test.assertRaisesRegex(RuntimeError,'intentional interruption'):
                        driver.run_arm(folder,role,payload,data,interrupted,ref,stats)
                midpoint=torch.load(interrupted/'latest_state.pt',map_location='cpu',weights_only=False)
                test.assertEqual(midpoint['plan']['completed_updates'],8193)
                rr=driver.run_arm(folder,role,payload,data,interrupted,ref,stats,resume_path=interrupted/'latest_state.pt')
                resumed=torch.load(interrupted/'latest_state.pt',map_location='cpu',weights_only=False)
                for key in ['base','pose_port','optimizer','plan','schedule','rng']:
                    test.assertEqual(pose_runtime.digest(p[key]),pose_runtime.digest(resumed[key]),key)
                test.assertEqual(rr['base_learned_sha256'],r['base_learned_sha256'])
                test.assertEqual(rr['port_sha256'],r['port_sha256'])
                # Never reinterpret an engineering checkpoint as a formal checkpoint.
                wrong=copy.deepcopy(midpoint);wrong['format']='current-pose-disposable-preflight-state-v1'
                torch.save(wrong,interrupted/'wrong_format.pt')
                with test.assertRaises(AssertionError):
                    driver.run_arm(folder,role,payload,data,interrupted,ref,stats,resume_path=interrupted/'wrong_format.pt')
                results.append(r)
        return results

    def test_real_labels_metrics_masking_and_gate_directions(self):
        selection=binding.read(BASE/'selection.json')
        with np.load(ROOT/'reports/oxe_target_eval_repair_232091/cluster/predictions_and_targets.npz') as z:
            targets=z['targets'];masks=z['masks'];static=z['static'];unchanged=z['sample']
        perfect=np.repeat(targets[None],3,0)
        out={'mask_control':unchanged,'real_pose':perfect,'unchanged231297':unchanged,'static':static}
        good=summarize(out,targets,masks,selection)
        self.assertTrue(good['candidate_for_next_qualification'])
        self.assertIsNone(good['by_source']['validation']['Language Table']['real_pose']['path_rotation_deg'])
        self.assertLess(good['paired_comparisons']['real_minus_mask_control']['path_position_normalized']['paired_group_bootstrap95'][1],0)
        out['real_pose']=unchanged.copy()
        self.assertFalse(summarize(out,targets,masks,selection)['candidate_for_next_qualification'])
        out['real_pose']=perfect.copy();lt=[i for i,r in enumerate(selection['train']+selection['validation']) if r['source']=='Language Table']
        out['real_pose'][:,lt,:,2:6]=12345  # unavailable dimensions cannot affect LT position/rotation
        masked=summarize(out,targets,masks,selection)
        self.assertEqual(masked['by_source']['validation']['Language Table']['real_pose']['path_position_normalized'],0)

    def test_actual_endpoint_entry_120_real_windows_with_cpu_encoder_sampler_standins(self):
        from oxe_nominal_target_store import NominalTargetStore
        import step_training_runner
        import verify_formal
        import runtime_inputs
        from types import ModuleType
        from contextlib import contextmanager
        from torch import nn
        pg=module('formal_eval_qualified_driver',PREF/'run_preflight.py')
        driver=module('formal_endpoint_evaluator',FORMAL/'evaluate_formal.py')
        selection=binding.read(BASE/'selection.json');stats=binding.read(PREF/'state_stats.json')
        train=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
        val=NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')
        class Part:
            def __init__(self,store):self.store=store
            def __getitem__(self,i):
                v=self.store.get(i)
                return 'fixture',torch.zeros(3,2,2),torch.from_numpy(v['current_features']),torch.from_numpy(v['actions']),torch.from_numpy(v['supervision_mask'])
        data=SimpleNamespace(training=Part(train),validation=Part(val))
        class Vision(nn.Module):
            def forward(self,pixel_values):return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values),3,1024))
        class Text(nn.Module):
            def forward(self,input_ids,attention_mask):return SimpleNamespace(last_hidden_state=torch.ones(len(input_ids),77,4))
        calls=[]
        class Model(nn.Module):
            def __init__(self,config):
                super().__init__();self.core=nn.Parameter(torch.zeros(1));self.vision_encoder=Vision();self.text_encoder=Text()
            def adapter(self,v,t,m):return v[:,0,:]+self.core
            def sample(self,c,current):
                calls.append(len(c));p=torch.zeros(len(c),16,8);p[...,6]=1;return p
        tokenizer=lambda texts,**kw:{'input_ids':torch.ones(len(texts),77,dtype=torch.long),'attention_mask':torch.ones(len(texts),77,dtype=torch.long)}
        api=ModuleType('train');api.CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:tokenizer)
        api.trainable_state_dict=lambda m:{'core':m.core.detach().cpu()}
        models=ModuleType('models');models.RobotAdapterModel=Model
        @contextmanager
        def constants(cls,*,on_restored,**kw):on_restored({'cpu_fixture':True});yield
        constructor=ModuleType('resume_constants');constructor.training_constant_constructor=constants
        from completed_policy_export import contract_sha
        reference=binding.read(PREF/'protocol.json');config=binding.read(BASE/'reference_contract.json')['config'];config['model']['diffusion_prediction_type']='sample'
        source={'contract':{'config':config,'tokenizer':{'fixture':True}}}
        metadata={}
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True);temp=tempfile.TemporaryDirectory(prefix='formal_endpoint_cpu_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as folder:
            folder=Path(folder);base=folder/'oxe_target_pair_short_v1';base.mkdir()
            (base/'selection.json').write_bytes((BASE/'selection.json').read_bytes());(base/'reference_contract.json').write_text(json.dumps({'tokenizer':{'fixture':True}}))
            evaluation=folder/'runs/oxe-target-short-eval-fixed-232091';evaluation.mkdir(parents=True)
            with np.load(ROOT/'reports/oxe_target_eval_repair_232091/cluster/predictions_and_targets.npz') as z:
                np.savez_compressed(evaluation/'predictions_and_targets.npz',targets=z['targets'],masks=z['masks'],sample=z['sample'],static=z['static'])
            output=folder/'pair';output.mkdir()
            for role in ['mask_control','real_pose']:
                path=output/role;path.mkdir();port=pose_runtime.zero_port('cpu')
                if role=='real_pose':port.weight.data[:,0]=.001
                c={'base_config_unchanged':config,'role':role,'preflight_report_sha256':'cpu-fixture'}
                learned={'core':torch.zeros(1)};porthash=pose_runtime.digest(port.state_dict());basehash=model_sha(learned)
                p={'base':learned,'pose_port':port.state_dict(),'contract':c,'contract_sha256':contract_sha(c)}
                torch.save(p,path/'latest_state.pt');metadata[role]={'base_learned_sha256':basehash,'port_sha256':porthash,'contract_sha256':contract_sha(c)}
                (path/'arm_receipt.json').write_text(json.dumps(metadata[role]))
            actual_verify_eval=verify_formal.verify_evaluation
            original_port=pose_runtime.zero_port
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules,{'train':api,'models':models,'resume_constants':constructor}))
                stack.enter_context(patch.object(driver,'admit',return_value={'preflight_report_sha256':'cpu-fixture'}))
                stack.enter_context(patch.object(driver,'attach',return_value=pg));stack.enter_context(patch.object(pg,'initialize_runtime'))
                stack.enter_context(patch.object(runtime_inputs,'load',return_value=(source,data,reference,stats)))
                stack.enter_context(patch.object(verify_formal,'verify_training',return_value={'passed':True}))
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value={'fixture':True}))
                stack.enter_context(patch.object(step_training_runner,'frozen_digest',return_value=reference['frozen_sha256']))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self));stack.enter_context(patch.object(nn.Module,'cuda',lambda self,*a,**kw:self))
                for name in ['manual_seed','manual_seed_all','empty_cache']:stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.object(torch.cuda,'get_rng_state_all',return_value=[]))
                stack.enter_context(patch.object(pose_runtime,'zero_port',side_effect=lambda device,width:original_port('cpu',width)))
                stack.enter_context(patch.object(sys,'argv',['evaluate_formal.py','--root',str(folder),'--output',str(output)]))
                with redirect_stdout(io.StringIO()):driver.main()
                checked=actual_verify_eval(output)
                self.assertTrue(checked['verified']);self.assertEqual(calls,[8]*90)
                report=binding.read(output/'evaluation/report.json')
                self.assertEqual(report['optimizer_updates'],0);self.assertTrue(report['paired_sampling_rng_exact'])
                self.assertEqual(report['generation_steps'],100)
                with np.load(output/'evaluation/predictions_and_targets.npz') as z:
                    self.assertEqual(set(z.files),{'targets','masks','real_pose','mask_control','unchanged231297','static'})
                    self.assertEqual(z['real_pose'].shape,(3,120,16,8))


if __name__=='__main__':unittest.main(verbosity=2)
