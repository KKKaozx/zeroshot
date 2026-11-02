"""Single-arm integration with real OXE labels/native losses and sampler; CPU toy encoders."""
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

ROOT=Path(__file__).resolve().parents[1];BUNDLE=ROOT/'deployment/oxe_competence_pool_v1'
spec=importlib.util.spec_from_file_location('competence_sampling_fixture',ROOT/'tests/test_oxe_sampling_pair.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
sys.path.insert(0,str(BUNDLE))
import competence_bind as binding
import competence_metrics as metrics
import competence_probe as probe
import competence_verify as verifier
import run_competence as driver


def plans_for(b,n=4):
    with np.load(BUNDLE/'plans.npz') as z:plans={k:z[k][:n].copy() if k!='source_codes' else z[k].copy() for k in z.files}
    plans['lrs']=np.asarray([b['schedule'].rates(10240+i) for i in range(n)])
    return plans


def save_receipt(folder,receipt,prediction,target,mask,stat):
    np.savez_compressed(folder/'predictions.npz',predictions=prediction,targets=target,masks=mask)
    receipt.update(prediction_file_sha256=binding.sha(folder/'predictions.npz'),source_file_size_mtime_unchanged=True)
    (folder/'receipt.json').write_text(json.dumps(receipt))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual512_pool_plan_validation_and_reject_corrupted_membership(self):
        store=f.NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
        plans=binding.load_plans();rows=binding.read(BUNDLE/'pool_rows.json')['rows'];p=binding.read(BUNDLE/'protocol.json')
        self.assertTrue(all(binding.validate_pool(store,plans,rows,p).values()))
        bad=copy.deepcopy(rows);bad[0]['group_id']='validation_or_wrong_group'
        with self.assertRaises(AssertionError):binding.validate_pool(store,plans,bad,p)
        bad={k:v.copy() for k,v in plans.items()};bad['moderate_pool'][0,0]=bad['moderate_pool'][0,1]
        with self.assertRaises(AssertionError):binding.validate_pool(store,bad,rows,p)
        counts={}
        for i,name in enumerate(p['sources']):
            ids=[r['window_index'] for r in rows if r['source']==name]
            counts[name]=np.asarray([(plans['moderate_pool']==j).sum() for j in ids])
            self.assertEqual(int(counts[name].min()),p['pool_counts'][name]['min_presentations'])
            self.assertEqual(int(counts[name].max()),p['pool_counts'][name]['max_presentations'])
            self.assertTrue((counts[name]>0).all())

    def test_actual_probe_resume_and_control_loss_rejection_before_any_optimizer_step(self):
        with f.fixture.temporary('competence_probe_') as folder,ExitStack() as stack:
            root=Path(folder);f.cpu(stack);b=f.bundle(root);plans=plans_for(b);c=f.contract()
            old=root/'old';old.mkdir()
            with redirect_stdout(io.StringIO()):receipt,trace=f.steps.train(b,'replacement',plans,old,c)
            candidate=f.bundle(root);identity=f.steps.initial_identity(candidate)
            with patch.object(candidate['optimizer'],'step',side_effect=AssertionError('Probe must never step')):
                guard=probe.run(candidate,plans,trace)
            self.assertEqual(identity,f.steps.initial_identity(candidate));self.assertEqual(guard['optimizer_updates'],0)
            bad=copy.deepcopy(trace);bad['history'][0]['loss']+=1
            with self.assertRaisesRegex(AssertionError,'control first loss'):probe.run(candidate,plans,bad)
            new=root/'new';new.mkdir();original=f.steps.backward;calls=0
            def interrupt(bundle,inputs):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('intentional halfway interruption')
                return original(bundle,inputs)
            with patch.object(f.steps,'backward',side_effect=interrupt),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'halfway'):f.steps.train(f.bundle(root),'moderate_pool',plans,new,c,trace)
            saved=torch.load(new/'latest_state.pt',weights_only=False);self.assertEqual(saved['counter'],2)
            resumed=f.bundle(root)
            with redirect_stdout(io.StringIO()):r,tr=f.steps.train(resumed,'moderate_pool',plans,new,c,trace,resume=True)
            full=root/'full';full.mkdir();fresh=f.bundle(root)
            with redirect_stdout(io.StringIO()):r2,tr2=f.steps.train(fresh,'moderate_pool',plans,full,c,trace)
            self.assertEqual(r['base_learned_sha256'],r2['base_learned_sha256']);self.assertEqual(r['port_sha256'],r2['port_sha256'])
            self.assertEqual(f.digest(resumed['optimizer'].state_dict()),f.digest(fresh['optimizer'].state_dict()))
            self.assertEqual(tr['noise'],tr2['noise']);self.assertEqual(f.steps.rng_digest(resumed),f.steps.rng_digest(fresh))
            wrong=dict(c,pool_rows_sha256='different-pool')
            before=f.steps.initial_identity(resumed)
            with self.assertRaises(AssertionError):f.pair_state.restore(new/'latest_state.pt',resumed,'moderate_pool',wrong,plans,trace)
            self.assertEqual(before,f.steps.initial_identity(resumed))

    def test_actual_main_source120_native_sampler_reused_control_and_saved_output_verifier(self):
        import instrument_formal
        pg=f.fixture.module('competence_original_preflight',f.fixture.PREF/'run_preflight.py')
        with f.fixture.temporary('competence_main_') as folder,ExitStack() as stack:
            root=Path(folder);f.cpu(stack);local=root/BUNDLE.name;local.mkdir()
            for n in ['selection.json','pool_rows.json']:shutil.copyfile(BUNDLE/n,local/n)
            base=root/'oxe_target_pair_short_v1';base.mkdir();shutil.copyfile(BUNDLE/'selection.json',base/'selection.json')
            (base/'reference_contract.json').write_text(json.dumps({'tokenizer':{'cpu_fixture':True}}))
            common=root/'source.pt';common.write_bytes(b'CPU fixture immutable source');stat=[common.stat().st_size,common.stat().st_mtime_ns]
            def loader(root):
                b=f.bundle(root);b['source_path']=common;b['source_file_size_mtime']=stat;return b
            first=loader(root);plans=plans_for(first)
            p=binding.read(BUNDLE/'protocol.json');p.update(updates_each=4,checkpoint_every=2,arm_training_cap_seconds=120)
            (local/'protocol.json').write_text(json.dumps(p));np.savez_compressed(local/'plans.npz',**plans)
            selection=binding.read(base/'selection.json');targets=[];masks=[]
            for role in ['train','validation']:
                part=first['data'].training if role=='train' else first['data'].validation
                for row in selection[role]:
                    item=part[row['window_index']];targets.append(item[3].numpy());masks.append(item[4].numpy())
            target=np.stack(targets);mask=np.stack(masks)
            original=root/'runs/oxe-target-short-eval-fixed-232091';original.mkdir(parents=True)
            np.savez_compressed(original/'predictions_and_targets.npz',targets=target,masks=mask)
            source=instrument_formal.prepare_source(pg);exec(compile(source,'<actual-original120>','exec'),pg.__dict__)
            cache=pg.prepare_inputs(root,first['data'],first['model'],first['port'],first['tokenizer'],first['api'],first['stats'],selection)
            with redirect_stdout(io.StringIO()):pred=f.steps.sample(first,cache,[1101,1102,1103])
            static=np.zeros_like(pred);static[...,6]=1
            oldsource=root/'runs/oxe-pose-pair-232587/evaluation';oldsource.mkdir(parents=True)
            np.savez_compressed(oldsource/'predictions_and_targets.npz',targets=target,masks=mask,real_pose=pred,static=static)
            old=root/'runs/oxe-sampling-pair-233531';control=old/'replacement';control.mkdir(parents=True)
            cb=loader(root);cc=f.contract()
            with redirect_stdout(io.StringIO()):cr,ctrace=f.steps.train(cb,'replacement',plans,control,cc);cpred=f.steps.sample(cb,cache,[1101,1102,1103])
            save_receipt(control,cr,cpred,target,mask,stat)
            actual_control=f.verifier.verify_training(control,'replacement',cc,plans)
            np.savez_compressed(old/'predictions_and_targets.npz',targets=target,masks=mask,source232587=pred,static=static,replacement=cpred)
            old_sha=binding.sha(control/'latest_state.pt');old_stamp=[(control/'latest_state.pt').stat().st_size,(control/'latest_state.pt').stat().st_mtime_ns]
            bound={'source_binding':{'source_file_size_mtime':stat},'control_checkpoint_size_mtime':old_stamp}
            (local/'source_control_binding.json').write_text(json.dumps(bound))
            stack.enter_context(patch.object(binding,'HERE',local));stack.enter_context(patch.object(binding,'admit',return_value=bound))
            stack.enter_context(patch.object(binding,'parent',return_value=f.pq));stack.enter_context(patch.object(f.pq,'attach',return_value=(None,pg)))
            stack.enter_context(patch.object(f.pq,'install_loader',return_value=(loader,'# CPU toy constructor fixture\n')))
            stack.enter_context(patch.object(binding,'control',return_value=(actual_control,{'training_job_ids':cr['training_job_ids']})))
            stack.enter_context(patch.object(pg,'initialize_runtime',lambda:None));stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'cpu_candidate'}))
            output=root/'output';stack.enter_context(patch.object(sys,'argv',['run_competence.py','--root',str(root),'--output',str(output)]))
            with redirect_stdout(io.StringIO()):driver.main()
            checked=verifier.verify_outputs(root,output)
            self.assertTrue(checked['output_verified']);self.assertFalse(checked['control_retrained'])
            self.assertEqual(binding.sha(control/'latest_state.pt'),old_sha)
            r=binding.read(output/'report.json');self.assertEqual(r['actual_matched_steps'],4)
            self.assertEqual(r['training_job_ids'],['cpu_candidate'])
            before=binding.sha(output/'moderate_pool/latest_state.pt')
            with patch.object(sys,'argv',['run_competence.py','--root',str(root),'--output',str(output),'--resume']),redirect_stdout(io.StringIO()):driver.main()
            self.assertEqual(before,binding.sha(output/'moderate_pool/latest_state.pt'))
            receipt=output/'moderate_pool/receipt.json';bad=binding.read(receipt);bad['prediction_file_sha256']='corrupt';receipt.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):verifier.verify_outputs(root,output)

    def test_real_masked_metrics_perfect_anchors_pass_static_equal_fails(self):
        selection=binding.read(BUNDLE/'selection.json');target=[];mask=[]
        for role in ['train','validation']:
            store=f.NominalTargetStore(ROOT/('reports/oxe_nominal_targets_20261025' if role=='train' else 'reports/oxe_nominal_validation_20261025'),partition=role)
            for row in selection[role]:
                v=store.get(row['window_index']);target.append(v['actions']);mask.append(v['supervision_mask'])
        target=np.stack(target);mask=np.stack(mask);perfect=np.repeat(target[None],3,0);static=np.zeros_like(perfect);static[...,6]=1
        outputs={k:static for k in ['moderate_pool','replacement','source232587','static']};outputs['moderate_pool']=perfect
        good=metrics.summarize(outputs,target,mask,selection)
        self.assertTrue(good['limited_training_anchor_support'])
        self.assertAlmostEqual(good['motion']['train']['Bridge']['moderate_pool']['prediction_to_expert_amplitude_ratio'],1.)
        self.assertAlmostEqual(good['motion']['train']['Language Table']['moderate_pool']['mean_direction_cosine_nonzero_pairs'],1.)
        self.assertIsNone(good['motion']['train']['Bridge']['static']['mean_direction_cosine_nonzero_pairs'])
        self.assertIsNone(good['by_source']['train']['Language Table']['moderate_pool']['path_rotation_deg'])
        outputs['moderate_pool']=static;self.assertFalse(metrics.summarize(outputs,target,mask,selection)['limited_training_anchor_support'])

    def test_actual_control_mmap_metadata_and_cpu_admission_no_cuda(self):
        # Full4096-row metadata fixture, not4096actual model updates. Native training is exercised above.
        with f.fixture.temporary('competence_admission_') as folder,ExitStack() as stack:
            root=Path(folder);f.cpu(stack);b=f.bundle(root);small=plans_for(b);cc=f.contract()
            old=root/'runs/oxe-sampling-pair-233531';arm=old/'replacement';arm.mkdir(parents=True)
            with redirect_stdout(io.StringIO()):r,tr=f.steps.train(b,'replacement',small,arm,cc)
            payload=torch.load(arm/'latest_state.pt',weights_only=False)
            with np.load(f.BUNDLE/'plans.npz') as z:oldplans={k:z[k] for k in z.files}
            payload['counter']=4096;payload['contract']=f.contract(4096);payload['contract_sha256']=f.contract_sha(payload['contract'])
            payload['history']=[];payload['noise']=[];payload['training_job_ids']=['233531']
            for i in range(4096):
                row=copy.deepcopy(tr['history'][i%4]);event=copy.deepcopy(tr['noise'][i%4])
                row.update(local_update=i+1,source_schedule_update=10241+i,indices=oldplans['replacement'][i].tolist(),
                    lr_used=oldplans['lrs'][i].tolist(),valid_dimensions=oldplans['replacement_valid_dimensions'][i].tolist())
                event['step']=10241+i;payload['history'].append(row);payload['noise'].append(event)
            for g,lr in zip(payload['optimizer']['param_groups'],b['schedule'].rates(14336)):g['lr']=lr
            torch.save(payload,arm/'latest_state.pt')
            np.savez_compressed(arm/'predictions.npz',predictions=np.zeros(1))
            r.update(updates_each=4096,completed_source_schedule=14336,contract_sha256=payload['contract_sha256'],
                training_job_ids=['233531'],prediction_file_sha256=binding.sha(arm/'predictions.npz'),source_file_size_mtime_unchanged=True)
            (arm/'receipt.json').write_text(json.dumps(r))
            oldreport={'execution_complete':True,'source_replay_exact':True,'all_actual_noise_t_and_rng_matched':True,
                'actual_paired_steps_matched':4096,'training_job_ids':['233531'],'training':{'replacement':r},'contract':payload['contract']}
            (old/'report.json').write_text(json.dumps(oldreport))
            local=root/BUNDLE.name;local.mkdir()
            for n in ['pool_rows.json','plans.npz','selection.json']:shutil.copyfile(BUNDLE/n,local/n)
            p=binding.read(BUNDLE/'protocol.json');p['inherited_pair_files']={}
            p['control_files']={n:binding.sha(old/n) for n in ['report.json','replacement/receipt.json']};p['control_receipt']=r
            p['source_identity'].update(base_learned_sha256=r['initial_identity']['base_learned_sha256'],port_sha256=r['initial_identity']['port_sha256'])
            (local/'protocol.json').write_text(json.dumps(p))
            (local/'SHA256SUMS').write_text(''.join(f'{binding.sha(local/n)}  {n}\n' for n in ['pool_rows.json','plans.npz','selection.json','protocol.json']))
            store=f.NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025')
            import oxe_nominal_target_store
            stack.enter_context(patch.object(binding,'HERE',local));stack.enter_context(patch.object(binding,'parent',return_value=f.pq))
            stack.enter_context(patch.object(f.pq,'admit',return_value={'source_file_size_mtime':[1,2],'scope':'previously-tested-parent-admission-fixture'}))
            stack.enter_context(patch.object(oxe_nominal_target_store,'NominalTargetStore',return_value=store))
            stack.enter_context(patch.object(binding.subprocess,'check_output',return_value='1000000 fixture'))
            stack.enter_context(patch.object(binding.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)))
            stack.enter_context(patch.object(torch.cuda,'init',side_effect=AssertionError('CPU admission must not initialize CUDA')))
            saved,report=binding.control(root);self.assertEqual(len(saved['trace']['history']),4096)
            stack.enter_context(patch.object(sys,'argv',['run_competence.py','--root',str(root),'--check-only']))
            with redirect_stdout(io.StringIO()) as output:driver.main()
            self.assertIn('CPU ADMISSION PASSED',output.getvalue());bound=binding.read(local/'source_control_binding.json')
            self.assertTrue(bound['completed_control_verified_on_cpu'])
            # Changed control bytes fail before a new source/control binding can be accepted.
            (arm/'receipt.json').write_text('{}')
            with self.assertRaises(AssertionError):binding.admit(root)


if __name__=='__main__':unittest.main(verbosity=2)
