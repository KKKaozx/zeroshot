"""Bounded paired entry, real numeric labels/native sampler, CPU toy encoders only."""
import copy
from contextlib import ExitStack,redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/oxe_sampling_pair_v1'
spec=importlib.util.spec_from_file_location('sampling_native_fixture',ROOT/'tests/test_oxe_fixed8_learnability.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
sys.path.insert(0,str(BUNDLE))
import pair_qualification as pq
import pair_state
import pair_steps as steps
import sampling_verify as verifier
import sampling_metrics
from pose_runtime import digest
from step_training_state import UpdateSchedule,capture_rng,_snapshot
from completed_policy_export import model_sha,contract_sha
from oxe_nominal_target_store import NominalTargetStore


def bundle(root):
    b=fixture.toy_bundle(root)
    class Data:
        def __init__(self,data):self.training=data.training;self.validation=data.validation
        def __getitem__(self,i):
            part=self.training;v=part.store.get(i)
            text=part.original_text.get(i,'move object '+str(i%100))
            return text,torch.tensor([i/145812,.1,1.]),torch.from_numpy(v['current_features']),torch.from_numpy(v['actions']),torch.from_numpy(v['supervision_mask'])
    b['data']=Data(b['data']);original=b['tokenizer']
    def padded(texts,**kw):
        tokens=original(texts,**kw)
        if kw.get('padding')=='max_length':
            tokens={k:F.pad(v,(0,77-v.shape[1])) for k,v in tokens.items()}
        return tokens
    b['tokenizer']=padded
    adapter=b['model'].adapter;oldforward=adapter.forward
    # Attention-length-dependent stochastic consumption, in addition to original adapter dropout.
    adapter.forward=lambda v,t,m:oldforward(v,F.dropout(t,p=.1,training=adapter.training),m)
    return b


def plans_for(b,n=4):
    with np.load(BUNDLE/'plans.npz') as z:plans={k:z[k][:n].copy() if k!='source_codes' else z[k].copy() for k in z.files}
    plans['lrs']=np.asarray([b['schedule'].rates(10240+i) for i in range(n)])
    return plans


def contract(n=4):
    return {'updates_each':n,'checkpoint_every':2,'arm_training_cap_seconds':120,
        'frozen_sha256':pq.read(BUNDLE/'protocol.json')['source_identity']['frozen_state_sha256']}


def cpu(stack):
    import step_training_runner
    stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self))
    for name in ['manual_seed','manual_seed_all','reset_peak_memory_stats','empty_cache']:
        stack.enter_context(patch.object(torch.cuda,name))
    stack.enter_context(patch.object(torch.cuda,'max_memory_reserved',return_value=0))
    stack.enter_context(patch.object(step_training_runner,'frozen_digest',return_value=contract()['frozen_sha256']))
    stack.enter_context(patch.object(step_training_runner,'tokenizer_identity',return_value={'cpu_fixture':True}))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_native_training_pair_rng_checkpoint_exact_resume_and_negative_trace(self):
        with fixture.temporary('sampling_engine_') as folder,ExitStack() as stack:
            root=Path(folder);cpu(stack);b=bundle(root);plans=plans_for(b);c=contract()
            identity=steps.initial_identity(b);guard=steps.preflight_pair_rng(b,plans)
            self.assertEqual(steps.initial_identity(b),identity);self.assertTrue(guard['passed'])
            self.assertNotEqual(guard['first_batch']['replacement']['noise']['noisy_actions'],guard['first_batch']['cycle']['noise']['noisy_actions'])
            control=root/'control';control.mkdir()
            with redirect_stdout(io.StringIO()):receipt,trace=steps.train(b,'replacement',plans,control,c)
            initial=bundle(root);candidate=root/'candidate';candidate.mkdir()
            # Interrupt after a committed halfway checkpoint, then resume from all three Adam groups and RNG.
            original=steps.backward;calls=0
            def interrupt(b,inputs):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('CPU intentional interruption')
                return original(b,inputs)
            with patch.object(steps,'backward',side_effect=interrupt),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'intentional'):steps.train(initial,'cycle',plans,candidate,c,control=trace)
            halfway=torch.load(candidate/'latest_state.pt',weights_only=False)
            self.assertEqual(halfway['counter'],2)
            resumed=bundle(root)
            with redirect_stdout(io.StringIO()):r,tr=steps.train(resumed,'cycle',plans,candidate,c,control=trace,resume=True)
            uninterrupted=bundle(root);other=root/'full';other.mkdir()
            with redirect_stdout(io.StringIO()):r2,tr2=steps.train(uninterrupted,'cycle',plans,other,c,control=trace)
            for key in ['base_learned_sha256','port_sha256','initial_identity']:self.assertEqual(r[key],r2[key])
            self.assertEqual(digest(resumed['optimizer'].state_dict()),digest(uninterrupted['optimizer'].state_dict()))
            self.assertEqual(steps.rng_digest(resumed),steps.rng_digest(uninterrupted))
            for a,brow in zip(tr['history'],tr2['history']):
                self.assertEqual({k:v for k,v in a.items() if k!='elapsed_step_seconds'},{k:v for k,v in brow.items() if k!='elapsed_step_seconds'})
            self.assertEqual(tr['noise'],tr2['noise']);self.assertEqual(r['new_updates_this_segment'],2)
            state=copy.deepcopy(halfway);state['history'][0]['indices'][0]+=1
            bad=root/'bad.pt';torch.save(state,bad)
            untouched=steps.initial_identity(resumed)
            with self.assertRaises(AssertionError):pair_state.restore(bad,resumed,'cycle',c,plans,trace)
            self.assertEqual(steps.initial_identity(resumed),untouched)
            with self.assertRaises(AssertionError):pair_state.restore(candidate/'latest_state.pt',resumed,'replacement',c,plans)
            altered=dict(tr['noise'][0],noise='wrong')
            with self.assertRaisesRegex(AssertionError,'noise/t'):steps.check_paired_stochastic(tr['history'][0],altered,trace['history'][0],trace['noise'][0])
            # A different stochastic tensor width must be rejected before any optimizer step.
            badbundle=bundle(root);old=badbundle['tokenizer'];counter=0
            def variable(texts,**kw):
                nonlocal counter
                counter+=1;r=old(texts,**kw)
                if counter==2:torch.rand(1)
                return r
            badbundle['tokenizer']=variable
            with self.assertRaisesRegex(AssertionError,'stochastic shapes'):steps.preflight_pair_rng(badbundle,plans)

    def test_actual_pair_main_original120_input_preparation_and_saved_verifier(self):
        import instrument_formal
        driver=fixture.module('sampling_actual_main',BUNDLE/'run_pair.py')
        pg=fixture.module('sampling_original_preflight',fixture.PREF/'run_preflight.py')
        with fixture.temporary('sampling_main_') as folder,ExitStack() as stack:
            root=Path(folder);cpu(stack);base=root/'oxe_target_pair_short_v1';base.mkdir()
            shutil.copyfile(fixture.BASE/'selection.json',base/'selection.json')
            (base/'reference_contract.json').write_text(json.dumps({'tokenizer':{'cpu_fixture':True}}))
            common=root/'source_immutable.pt';common.write_bytes(b'CPU fixture only, unchanged across roles')
            stat=[common.stat().st_size,common.stat().st_mtime_ns]
            def loader(root):
                b=bundle(root);b['source_path']=common;b['source_file_size_mtime']=stat;return b
            first=loader(root);plans=plans_for(first);local=root/BUNDLE.name;local.mkdir()
            p=pq.read(BUNDLE/'protocol.json');p.update(updates_each=4,checkpoint_every=2,arm_training_cap_seconds=120)
            (local/'protocol.json').write_text(json.dumps(p));np.savez_compressed(local/'plans.npz',**plans)
            selection=pq.read(base/'selection.json');targets=[];masks=[]
            for role in ['train','validation']:
                part=first['data'].training if role=='train' else first['data'].validation
                for row in selection[role]:
                    item=part[row['window_index']];targets.append(item[3].numpy());masks.append(item[4].numpy())
            target=np.stack(targets);mask=np.stack(masks)
            original=root/'runs/oxe-target-short-eval-fixed-232091';original.mkdir(parents=True)
            np.savez_compressed(original/'predictions_and_targets.npz',targets=target,masks=mask)
            # Execute the actual mechanically qualified original120-window feature preparation.
            source=instrument_formal.prepare_source(pg);exec(compile(source,'<cpu-original120-inputs>','exec'),pg.__dict__)
            cache=pg.prepare_inputs(root,first['data'],first['model'],first['port'],first['tokenizer'],first['api'],first['stats'],selection)
            with redirect_stdout(io.StringIO()):prediction=steps.sample(first,cache,[1101,1102,1103])
            static=np.zeros_like(prediction);static[...,6]=1
            old=root/'runs/oxe-pose-pair-232587/evaluation';old.mkdir(parents=True)
            np.savez_compressed(old/'predictions_and_targets.npz',targets=target,masks=mask,real_pose=prediction,static=static)
            bound={'source_file_size_mtime':stat,'scope':'CPU entry fixture only'}
            for obj in [pq,driver,verifier]:stack.enter_context(patch.object(obj,'HERE',local))
            stack.enter_context(patch.object(driver,'admit',return_value=bound))
            stack.enter_context(patch.object(driver,'attach',return_value=(None,pg)))
            stack.enter_context(patch.object(pg,'initialize_runtime',lambda:None))
            stack.enter_context(patch.object(driver,'install_loader',return_value=(loader,'# CPU model constructor fixture\n')))
            stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'cpu_main_fixture'}))
            output=root/'output';stack.enter_context(patch.object(sys,'argv',['run_pair.py','--root',str(root),'--output',str(output)]))
            with redirect_stdout(io.StringIO()):driver.main()
            checked=verifier.verify_outputs(root,output)
            self.assertTrue(checked['verified']);self.assertEqual(checked['updates_each'],4)
            report=pq.read(output/'report.json');self.assertTrue(report['same_training_allocation'])
            self.assertTrue(report['source_replay_exact']);self.assertTrue(report['all_actual_noise_t_and_rng_matched'])
            self.assertEqual(report['actual_paired_steps_matched'],4)
            # Completed endpoint/ledger verification must reject altered paired trace metadata.
            receipt=output/'cycle/receipt.json';r=pq.read(receipt);r['prediction_file_sha256']='bad'
            receipt.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):verifier.verify_outputs(root,output)

    def test_actual_source_cpu_checkpoint_order_lrs_and_weights_guard(self):
        with fixture.temporary('sampling_metadata_') as folder:
            root=Path(folder);b=bundle(root);p=copy.deepcopy(pq.read(BUNDLE/'protocol.json'))
            with np.load(BUNDLE/'plans.npz') as z:plans={k:z[k] for k in z.files}
            # CPU fixture uses this runtime's schedule. Production plans still retain archived Linux floats.
            plans['lrs']=np.asarray([b['schedule'].rates(10240+i) for i in range(4096)])
            source_contract={'scope':'CPU toy checkpoint only'}
            p['expected_runtime']={'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda}
            p['source_identity'].update(contract_sha256=contract_sha(source_contract),base_learned_sha256=model_sha(b['api'].trainable_state_dict(b['model'])),port_sha256=digest(b['port'].state_dict()))
            rng={'include_cuda':True,'torch_cuda':[torch.zeros(1,dtype=torch.uint8)],
                'environment':{'torch':str(torch.__version__),'cuda_version':torch.version.cuda,'cuda_devices':1}}
            order=torch.zeros(145812,dtype=torch.int64);order[81920:81920+32768]=torch.from_numpy(plans['replacement'].reshape(-1))
            payload={'format':'current-pose-bounded-formal-state-v1','contract':source_contract,'contract_sha256':contract_sha(source_contract),
                'base':b['api'].trainable_state_dict(b['model']),'pose_port':b['port'].state_dict(),'rng':rng,
                'plan':{'completed_updates':10240,'cursor':81920,'pass_index':0,'order':order,'contract':p['original_full_plan_contract']},
                'schedule':b['schedule'].contract,'optimizer':b['optimizer'].state_dict()}
            path=root/'runs/oxe-pose-pair-232587/real_pose/latest_state.pt';path.parent.mkdir(parents=True);torch.save(payload,path)
            actual=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            with patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA metadata use')):
                self.assertTrue(pq.check_checkpoint(actual,p,plans)['actual_source_plan_control_suffix_exact'])
                wrong=copy.deepcopy(plans);wrong['replacement'][0,0]+=1
                with self.assertRaisesRegex(AssertionError,'ACTUAL source plan'):pq.check_checkpoint(actual,p,wrong)
                wrong=copy.deepcopy(plans);wrong['lrs'][3000,0]=np.nextafter(wrong['lrs'][3000,0],np.inf)
                with self.assertRaisesRegex(AssertionError,'LR differs'):pq.check_checkpoint(actual,p,wrong)
                changed=dict(p,source_identity=dict(p['source_identity'],base_learned_sha256='changed'))
                with self.assertRaises(AssertionError):pq.check_checkpoint(actual,changed,plans)
            # Actual check-only admission: own byte manifest, full real numeric store/masks,
            # exact source-slot/uniqueness checks and actual CPU mmap checkpoint, before CUDA.
            local=root/BUNDLE.name;local.mkdir()
            for name in ['pair_qualification.py','run_pair.py','pair_state.py','pair_steps.py','sampling_verify.py','sampling_metrics.py']:
                shutil.copyfile(BUNDLE/name,local/name)
            p['inherited_files']={}  # Previously qualified parent admission is the metadata stand-in here.
            (local/'protocol.json').write_text(json.dumps(p));np.savez_compressed(local/'plans.npz',**plans)
            names=[x.name for x in local.iterdir()]
            (local/'SHA256SUMS').write_text(''.join(f'{pq.sha(local/n)}  {n}\n' for n in names))
            driver=fixture.module('sampling_actual_cpu_admission',BUNDLE/'run_pair.py')
            import oxe_nominal_target_store
            store=NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
            with ExitStack() as stack:
                stack.enter_context(patch.object(pq,'HERE',local))
                stack.enter_context(patch.object(pq,'attach',return_value=(SimpleNamespace(admit=lambda *a,**kw:{'scope':'qualified-parent-metadata-fixture'}),None)))
                stack.enter_context(patch.object(oxe_nominal_target_store,'NominalTargetStore',return_value=store))
                stack.enter_context(patch.object(pq.subprocess,'check_output',return_value='1000000 fixture'))
                stack.enter_context(patch.object(pq.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
                stack.enter_context(patch.object(torch.cuda,'init',side_effect=AssertionError('No CUDA metadata use')))
                stack.enter_context(patch.object(sys,'argv',['run_pair.py','--root',str(root),'--check-only']))
                with redirect_stdout(io.StringIO()) as output:driver.main()
                self.assertIn('ACTUAL CPU ADMISSION PASSED',output.getvalue())
                self.assertTrue((local/'source_pair_binding.json').is_file())
                (local/'pair_state.py').write_text('changed source')
                with self.assertRaises(AssertionError):driver.main()

    def test_reuses_exact_qualified_corrected_loader_function(self):
        # This function was exercised with the actual loader in the prior LR repair regression.
        # Do not silently substitute another model/data loader for the new pair.
        import inspect,hashlib
        q=fixture.q
        original=q.load_source
        try:
            loader,text=pq.install_loader(q)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),pq.read(BUNDLE/'protocol.json')['qualified_corrected_loader_sha256'])
            self.assertEqual(q.REPAIRED_LRS,BUNDLE/'plans.npz')
            self.assertIs(loader,q.load_source)
        finally:q.load_source=original

    def test_metrics_real_masks_perfect_vs_static_and_shared_budget_plan(self):
        selection=pq.read(fixture.BASE/'selection.json');targets=[];masks=[]
        for role in ['train','validation']:
            store=NominalTargetStore(ROOT/('reports/oxe_nominal_targets_20261025' if role=='train' else 'reports/oxe_nominal_validation_20261025'),partition=role)
            for row in selection[role]:
                v=store.get(row['window_index']);targets.append(v['actions']);masks.append(v['supervision_mask'])
        target=np.stack(targets);mask=np.stack(masks);perfect=np.repeat(target[None],3,0)
        static=np.zeros_like(perfect);static[...,6]=1
        good=sampling_metrics.summarize({'replacement':static,'cycle':perfect,'source232587':static,'static':static},target,mask,selection)
        self.assertTrue(good['candidate_for_review_only'])
        self.assertIsNone(good['by_source']['validation']['Language Table']['cycle']['path_rotation_deg'])
        bad=sampling_metrics.summarize({n:static for n in ['replacement','cycle','source232587','static']},target,mask,selection)
        self.assertFalse(bad['basic_competence_gate_passed']);self.assertFalse(bad['candidate_for_review_only'])
        with np.load(BUNDLE/'plans.npz') as z:
            self.assertEqual(z['replacement'].shape,(4096,8));self.assertTrue(np.array_equal(z['source_codes'][z['replacement']],z['source_codes'][z['cycle']]))
            rows=json.loads((ROOT/'reports/oxe_noaug_seed42_227223/cluster/reports/step_report.json').read_text())['updates'][10240:14336]
            self.assertTrue(np.array_equal(z['lrs'],np.array([r['lr']+[r['lr'][0]] for r in rows])))
            for c in range(4):
                picked=z['cycle'].reshape(-1);picked=picked[z['source_codes'][picked]==c]
                self.assertEqual(len(picked),len(np.unique(picked)))


if __name__=='__main__':unittest.main(verbosity=2)
