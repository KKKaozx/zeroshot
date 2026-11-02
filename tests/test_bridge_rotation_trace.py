"""Read-only native sampler instrumentation/replay, real targets and CPU toy networks."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];TRACE=ROOT/'deployment/bridge_rotation_trace_v1';BASE=ROOT/'deployment/bridge_full_capability_v1'
sys.path[:0]=[str(TRACE),str(BASE/'code'),str(BASE)]
spec=importlib.util.spec_from_file_location('rotation_trace_native_fixture',ROOT/'tests/test_bridge_full_capability.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
from rtrace_engine import Recorder,instrument,advance_sample_rng,analyze
import run_trace
from bcap_core import identity
from pose_runtime import digest
from oxe_nominal_target_store import NominalTargetStore


def reference_evaluation(b,rows):
    b['model'].eval();b['port'].eval();draws=[]
    with torch.inference_mode():
        for seed in [1101,1102,1103]:
            torch.manual_seed(seed);per=[]
            for lo in range(0,192,8):
                selected=rows[lo:lo+8];x=b['pack'](selected[0]['role'],[r['window_index'] for r in selected])
                c=b['model'].get_context_vector(x['images'],x['tokens']['input_ids'],x['tokens']['attention_mask'])+b['port'](x['pose'])
                per.append(b['model'].sample(c,x['current']).numpy())
            draws.append(np.concatenate(per))
    return np.stack(draws)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.stores={'train':NominalTargetStore(ROOT/'reports/oxe_nominal_targets_20261025'),
            'validation':NominalTargetStore(ROOT/'reports/oxe_nominal_validation_20261025',partition='validation')}

    def test_native100step_instrumentation_exact_original_replay_and_rng_skip(self):
        b=fixture.make_bundle(self.stores);rows=json.loads((BASE/'selection.json').read_text())['rows']
        with patch.object(torch.cuda,'manual_seed_all'),redirect_stdout(io.StringIO()):reference=reference_evaluation(b,rows)
        before=identity(b);adam=digest(b['optimizer'].state_dict());record=Recorder(cap=120)
        source=instrument(BASE/'code/models.py');ns={'torch':torch,'F':F,'RTRACE_HOOK':record}
        exec(compile(source,'<actual-native-traced-sample>','exec'),ns)
        original=type(b['model']).sample;type(b['model']).sample=ns['sample']
        try:
            with patch.object(torch.cuda,'manual_seed_all'),patch.object(b['optimizer'],'step',side_effect=AssertionError('No optimizer calls allowed')),redirect_stdout(io.StringIO()):
                final,target,mask,checks=run_trace.replay(b,reference,rows,record)
        finally:type(b['model']).sample=original
        self.assertEqual(identity(b),before);self.assertEqual(digest(b['optimizer'].state_dict()),adam)
        self.assertTrue(all(p.grad is None for p in list(b['model'].parameters())+list(b['port'].parameters())))
        self.assertEqual(checks['skipped_rng_primitive_matches_actual_sample_for_seeds'],[1101,1102,1103])
        arrays=record.arrays();self.assertEqual(arrays['raw'].shape,(3,16,100,16,4))
        self.assertTrue(all(d['steps']==list(reversed(range(100))) for d in record.data.values()))
        expected=reference[:,list(range(8))+list(range(128,136))]
        report=analyze(arrays,target,mask,final,expected)
        self.assertTrue(report['existing16window_predictions_replayed_exactly']);self.assertEqual(report['optimizer_updates'],0)
        # Actual saved-array CPU verifier, with explicitly synthetic toy endpoint metadata.
        import verify_trace
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=parent,prefix='rotation_trace_verifier_')
        assert Path(temporary.name).resolve().parent==parent.resolve()
        with temporary as folder:
            folder=Path(folder);out=folder/'out';out.mkdir();sourcefolder=folder/'source';sourcefolder.mkdir()
            source_targets=[];source_masks=[]
            for row in rows:
                v=self.stores[row['role']].get(row['window_index']);source_targets.append(v['actions']);source_masks.append(v['supervision_mask'])
            np.savez_compressed(sourcefolder/'predictions_and_targets.npz',after=reference,targets=np.stack(source_targets),masks=np.stack(source_masks))
            np.savez_compressed(out/'trace.npz',targets=target,masks=mask,final=final,reference=expected,**arrays)
            (out/'effective_sample.py').write_text(source,encoding='utf-8')
            p={'final_identity':before,'effective_sample_sha256':run_trace.sha(out/'effective_sample.py'),
                'evaluation_indices':list(range(8))+list(range(128,136)),
                'source_small_files_sha256':{'predictions_and_targets.npz':run_trace.sha(sourcefolder/'predictions_and_targets.npz')}}
            run_trace.write(folder/'protocol.json',p)
            saved={'execution_complete':True,'optimizer_updates':0,'automatic_following_action':False,'protocol_sha256':run_trace.sha(folder/'protocol.json'),
                'model_adam_gradients_unchanged':True,'source_identity_unchanged':before,'trace_sha256':run_trace.sha(out/'trace.npz'),'analysis':report}
            run_trace.write(out/'report.json',saved)
            with patch.object(verify_trace,'HERE',folder):
                self.assertTrue(verify_trace.verify(out,sourcefolder)['verified'])
                saved['analysis']['roles']['train']['all100_steps'][0]['raw_x0']['quaternion_component_mse_window_equal']+=1
                run_trace.write(out/'report.json',saved)
                with self.assertRaises(AssertionError):verify_trace.verify(out,sourcefolder)
        wrong=expected.copy();wrong[0,8,0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'replay'):analyze(arrays,target,mask,final,wrong)
        # Actual replay function also rejects changed existing endpoint outputs, not just the CPU summarizer.
        corrupt=reference.copy();corrupt[0,128,0,0]+=.1
        badrecord=Recorder(cap=120);ns['RTRACE_HOOK']=badrecord;type(b['model']).sample=ns['sample']
        try:
            with patch.object(torch.cuda,'manual_seed_all'),redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError,'exact replay'):run_trace.replay(b,corrupt,rows,badrecord)
        finally:type(b['model']).sample=original

    def test_instrumentation_rejects_wrong_native_source_and_zero_norm_diagnostic(self):
        source=instrument(BASE/'code/models.py')
        self.assertEqual(source.count('RTRACE_HOOK.raw('),1);self.assertEqual(source.count('RTRACE_HOOK.final_raw('),1)
        shape=(3,16,100,16,4);arrays={k:np.zeros(shape,np.float32) for k in ['raw','latent','reconstructed','clipped']}
        arrays['final_raw']=np.zeros((3,16,16,7),np.float32)
        targets=np.zeros((16,16,8),np.float32);targets[...,6]=1;masks=np.ones_like(targets)
        final=np.repeat(targets[None],3,axis=0)
        r=analyze(arrays,targets,masks,final,final)
        self.assertEqual(r['roles']['train']['all100_steps'][0]['raw_x0']['invalid_norm_count'],3*8*16)
        self.assertIsNone(r['roles']['train']['all100_steps'][0]['raw_x0']['angle_deg_window_equal_nonzero_only'])
        self.assertEqual(r['roles']['train']['final_identity_fallback_count_all_frames'],3*8*16)

    def test_source_policy_metadata_and_parameter_identity_rejection(self):
        b=fixture.make_bundle(self.stores);ids=identity(b)
        payload={'format':'fresh-bridge-capability-policy-v1','completed_updates':65536,
            'contract':{'protocol_sha256':'protocol','plan_sha256':'plan','training_seed':42,'single_source_supervision':'Bridge',
                'config':{'model':{'diffusion_prediction_type':'sample'}}},'final_identity':ids,
            'state':{'base':b['api'].trainable_state_dict(b['model']),'pose_port':b['port'].state_dict()}}
        protocol={'base_protocol_sha256':'protocol','base_plan_sha256':'plan','final_identity':ids}
        self.assertTrue(run_trace.check_policy(payload,protocol)['source_policy_parameter_identity_checked_on_cpu'])
        bad=copy.deepcopy(payload);bad['completed_updates']=10240
        with self.assertRaises(AssertionError):run_trace.check_policy(bad,protocol)
        bad=copy.deepcopy(payload);bad['state']['pose_port']['weight'][0,0]+=1
        with self.assertRaises(AssertionError):run_trace.check_policy(bad,protocol)


if __name__=='__main__':unittest.main(verbosity=2)
