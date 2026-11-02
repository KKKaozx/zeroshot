"""Reproduce actual loader failure and run repaired loader/main with CPU source fixture."""
import copy
from contextlib import ExitStack,contextmanager,redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/oxe_fixed8_lr_repair_v1'
OLD=ROOT/'deployment/oxe_fixed8_learnability_v1';FORMAL=ROOT/'deployment/oxe_current_pose_formal_v1'
PREF=ROOT/'deployment/oxe_current_pose_preflight_v1';BASE=ROOT/'deployment/oxe_target_pair_short_v1'
sys.path[:0]=[str(BUNDLE),str(OLD),str(FORMAL),str(PREF),str(BASE),str(BASE/'code')]
import repair_core as rc
import qualification as q
import engine
import verify_output
from pose_runtime import digest
from step_training_state import UpdateSchedule
from completed_policy_export import model_sha
from instrument import method_source


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


fixture=module('fixed8_lr_cpu_fixture',ROOT/'tests/test_oxe_fixed8_learnability.py')


def linux_rates_fixture():
    # Actual archived production floats act as the Linux schedule producer in a Windows CPU toy test.
    rows=json.loads((ROOT/'reports/oxe_noaug_seed42_227223/cluster/reports/step_report.json').read_text())['updates']
    rates={r['update']-1:r['lr']+[r['lr'][0]] for r in rows[10240:10753]}
    original=UpdateSchedule.rates
    def rates_fn(self,completed):
        if completed in rates and self.contract==rc.read(BUNDLE/'repair_contract.json')['expected_schedule']:return rates[completed]
        return original(self,completed)
    return rates_fn


class Tests(unittest.TestCase):
    def test_exact_five_step_failure_reproduced_from_archived_actual_history(self):
        c=rc.read(BUNDLE/'repair_contract.json')
        with np.load(OLD/'reference_lrs.npz') as a,np.load(BUNDLE/'reference_lrs_from_cluster.npz') as b:
            original=a['lrs'];fixed=b['lrs']
            different=np.flatnonzero((original!=fixed).any(1))
            self.assertEqual((10241+different).tolist(),[10258,10301,10405,10492,10552])
            self.assertEqual(float(np.abs(original-fixed).max()),2.710505431213761e-20)
            schedule=UpdateSchedule(**c['expected_schedule'])
            with patch.object(UpdateSchedule,'rates',linux_rates_fixture()):
                self.assertFalse(all(schedule.rates(10240+i)==original[i].tolist() for i in range(512)))
                self.assertTrue(all(schedule.rates(10240+i)==fixed[i].tolist() for i in range(512)))
        self.assertEqual(rc.sha(ROOT/'cluster_packages/oxe_fixed8_learnability_v1.zip'),c['original_package_sha256'])

    def test_actual_loader_failure_then_repaired_actual_entry_full512_cpu(self):
        import runtime_inputs
        import step_training_runner
        import completed_policy_export
        import pose_runtime
        driver=module('actual_fixed8_lr_repair_driver',BUNDLE/'run_repair.py')
        torch.set_num_threads(1)
        with fixture.temporary('fixed8_lr_actual_loader_') as folder:
            root=Path(folder);base=root/'oxe_target_pair_short_v1';base.mkdir();shutil.copyfile(BASE/'selection.json',base/'selection.json')
            seed_bundle=fixture.toy_bundle(root);seed_model=seed_bundle['model'];seed_port=seed_bundle['port']
            config={'model':{'diffusion_prediction_type':'sample','name':'cpu-source-fixture'}}
            protocol=copy.deepcopy(q.read(OLD/'protocol.json'))
            protocol['source_identity']['base_learned_sha256']=model_sha(seed_bundle['api'].trainable_state_dict(seed_model))
            protocol['source_identity']['port_sha256']=digest(seed_port.state_dict())
            source_contract={'role':'real_pose','base_config_unchanged':config}
            contract_digest=protocol['source_identity']['contract_sha256']
            payload={'format':'current-pose-bounded-formal-state-v1','contract':source_contract,'contract_sha256':contract_digest,
                'plan':{'completed_updates':10240},'schedule':seed_bundle['schedule'].contract,
                'base':seed_bundle['api'].trainable_state_dict(seed_model),'pose_port':seed_port.state_dict(),
                'optimizer':seed_bundle['optimizer'].state_dict(),'rng':seed_bundle['source']['rng']}
            source=root/'runs/oxe-pose-pair-232587/real_pose';source.mkdir(parents=True)
            torch.save(payload,source/'latest_state.pt')
            seed_bundle['data'].nominal_target_binding={'cpu_fixture':True}
            original={'contract':{'config':config,'tokenizer':{'cpu_fixture':True},'arguments':{'weight_decay':.0001},
                'diagnostic_target_pair':{'diffusion_loss_source_sha256':hashlib.sha256(method_source()[1].encode()).hexdigest()}}}
            api=ModuleType('train');api.__dict__.update(seed_bundle['api'].__dict__)
            api.GRIPPER_PARAMETER_PREFIXES=('gripper_head.',)
            api.CLIPTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:seed_bundle['tokenizer'])
            class SourceModel(type(seed_model)):
                def __init__(self,config):
                    nn.Module.__init__(self);self.__dict__.update(copy.deepcopy(seed_model.__dict__))
            models=ModuleType('models');models.__dict__.update(torch=torch,Tuple=tuple,
                extract=lambda v,t,x:v.gather(0,t).reshape(len(t),1,1),RobotAdapterModel=SourceModel)
            @contextmanager
            def constant_constructor(cls,*,on_restored,**kw):on_restored({'cpu_fixture':True});yield
            constants=ModuleType('resume_constants');constants.training_constant_constructor=constant_constructor
            original_read=q.read;original_port=pose_runtime.zero_port
            bound={'scope':'CPU repaired loader fixture; no actual cluster qualification'}
            # Roll back monkey-compiled production functions after this test.
            with ExitStack() as stack:
                for obj,name in [(q,'load_source'),(engine,'run'),(verify_output,'verify')]:stack.enter_context(patch.object(obj,name,getattr(obj,name)))
                stack.enter_context(patch.dict(sys.modules,{'train':api,'models':models,'resume_constants':constants}))
                stack.enter_context(patch.object(q,'read',side_effect=lambda p:protocol if Path(p).resolve()==(OLD/'protocol.json').resolve() else original_read(p)))
                stack.enter_context(patch.object(runtime_inputs,'load',return_value=(original,seed_bundle['data'],{},seed_bundle['stats'])))
                stack.enter_context(patch.object(completed_policy_export,'contract_sha',return_value=contract_digest))
                stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value={'cpu_fixture':True}))
                stack.enter_context(patch.object(step_training_runner,'frozen_digest',return_value=protocol['source_identity']['frozen_state_sha256']))
                stack.enter_context(patch.object(pose_runtime,'zero_port',side_effect=lambda device,width:original_port('cpu',width)))
                stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self));stack.enter_context(patch.object(nn.Module,'cuda',lambda self,*a,**kw:self))
                for name in ['manual_seed','manual_seed_all','reset_peak_memory_stats']:stack.enter_context(patch.object(torch.cuda,name))
                stack.enter_context(patch.object(torch.cuda,'max_memory_reserved',return_value=0))
                stack.enter_context(patch.object(UpdateSchedule,'rates',linux_rates_fixture()))
                stack.enter_context(patch.object(q,'attach',return_value=(None,SimpleNamespace(initialize_runtime=lambda:None))))
                # This is the REAL original loading function, not mocked out: it reproduces233339's assertion.
                with self.assertRaises(AssertionError):q.load_source(root)
                sources=rc.fixed_sources(q,engine,verify_output)
                ns=dict(q.__dict__,REPAIRED_LRS=BUNDLE/'reference_lrs_from_cluster.npz')
                exec(compile(sources['load_source'],'<test-real-repaired-loader>','exec'),ns)
                restored=ns['load_source'](root);restored['root']=root
                self.assertEqual(digest(restored['optimizer'].state_dict()),digest(payload['optimizer']))
                self.assertEqual(model_sha(api.trainable_state_dict(restored['model'])),protocol['source_identity']['base_learned_sha256'])
                self.assertEqual(digest(restored['port'].state_dict()),protocol['source_identity']['port_sha256'])
                cache=fixture.fixed_inputs.prepare(root,restored);before=engine.sample(restored,cache,[1101,1102,1103])
                targets=cache['actions'].numpy();masks=cache['masks'].numpy();static=np.zeros_like(targets);static[...,6]=1
                archive=root/'runs/oxe-pose-pair-232587/evaluation';archive.mkdir()
                np.savez_compressed(archive/'predictions_and_targets.npz',targets=targets,masks=masks,real_pose=before,static=np.repeat(static[None],3,0))
                read_verify=verify_output.read
                stack.enter_context(patch.object(verify_output,'read',side_effect=lambda p:bound if Path(p).name=='source_binding.json' else read_verify(p)))
                stack.enter_context(patch.object(driver,'admit',return_value=bound))
                stack.enter_context(patch.object(q,'admit',return_value=bound))
                stack.enter_context(patch.object(rc,'modules',return_value=q))
                stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'cpu_lr_repair_fixture'}))
                output=root/'fixed_result';stack.enter_context(patch.object(sys,'argv',['run_repair.py','--root',str(root),'--output',str(output)]))
                # No patch of load_source here: repair main imports and runs the ACTUAL repaired loader.
                previous=sys.modules.pop('run_check',None)
                try:
                    with redirect_stdout(io.StringIO()):driver.main()
                finally:
                    sys.modules.pop('run_check',None)
                    if previous is not None:sys.modules['run_check']=previous
                report=rc.read(output/'report.json')
                self.assertTrue(report['execution_complete']);self.assertEqual(report['additional_optimizer_calls'],512)
                self.assertEqual(report['lr_reference_repair'],bound)
                self.assertTrue(report['cached_training_guard']['passed'])
                self.assertTrue(rc.verify_extra(output,bound)['repair_binding_verified'])
                self.assertTrue(verify_output.verify(root,output)['output_verified'])

    def test_preallocation_cpu_metadata_entry_has_no_model_constructor(self):
        driver=module('fixed8_lr_check_only',BUNDLE/'run_repair.py')
        # Actual repair admission's metadata branch with a real CPU mmap checkpoint;
        # the already qualified original asset admission is a metadata-only stand-in.
        import step_training_state
        from completed_policy_export import contract_sha
        c=rc.read(BUNDLE/'repair_contract.json')
        with fixture.temporary('fixed8_lr_cpu_metadata_') as folder:
            root=Path(folder);repair=root/BUNDLE.name;repair.mkdir();old=root/OLD.name;old.mkdir()
            for name in [l.split('  ',1)[1] for l in (BUNDLE/'SHA256SUMS').read_text().splitlines()]+['SHA256SUMS']:shutil.copyfile(BUNDLE/name,repair/name)
            for name in c['original_bundle_files']:shutil.copyfile(OLD/name,old/name)
            logs=root/'logs';logs.mkdir();(logs/'oxe-fixed8-233339.out').write_text('load_source\nassert all(schedule.rates(10240+i)==z[\'lrs\'][i].tolist() for i in range(p[\'updates\']))\nAssertionError\n')
            contract={'role':'real_pose','base_config_unchanged':{'scope':'CPU metadata fixture'}}
            digest_value=contract_sha(contract)
            source=root/'runs/oxe-pose-pair-232587/real_pose';source.mkdir(parents=True)
            rates=linux_rates_fixture();s=UpdateSchedule(**c['expected_schedule'])
            payload={'format':'current-pose-bounded-formal-state-v1','plan':{'completed_updates':10240},
                'contract':contract,'contract_sha256':digest_value,'schedule':s.contract,
                'optimizer':{'param_groups':[{'lr':v} for v in rates(s,10240)]}}
            torch.save(payload,source/'latest_state.pt')
            meta=SimpleNamespace(admit=lambda *a,**kw:{'fixture':True},attach=lambda *a:None,
                read=lambda *a:{'source_identity':{'contract_sha256':digest_value}})
            with ExitStack() as stack:
                stack.enter_context(patch.object(rc,'HERE',repair));stack.enter_context(patch.object(rc,'modules',return_value=meta))
                stack.enter_context(patch.object(UpdateSchedule,'rates',rates))
                stack.enter_context(patch.object(torch.cuda,'init',side_effect=AssertionError('Must not initialize CUDA')))
                stack.enter_context(patch.object(sys,'argv',['run_repair.py','--root',str(root),'--check-only']))
                with redirect_stdout(io.StringIO()) as text:driver.main()
                self.assertIn('all512_steps_exact',text.getvalue());self.assertTrue((repair/'source_repair_binding.json').exists())
                (logs/'oxe-fixed8-233339.out').write_text('load_source schedule.rates(10240+i) AssertionError\nOXE_FIXED8_UPDATE already_updated')
                with self.assertRaisesRegex(AssertionError,'already trained'):rc.admit(root)


if __name__=='__main__':unittest.main(verbosity=2)
