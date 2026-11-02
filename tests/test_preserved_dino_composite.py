from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from dino_gripper_sidecar import GripperSidecar,SOURCE_SHA
from preserved_dino_composite import load_versioned_head,SOURCE_HEAD_SHA,sha


def receipt(selected=None,passed=False):
    flags={role:{'balanced_accuracy_protected':True,'opening_event_gate':passed,'closing_event_gate':passed}
           for role in ('train','development')}
    return {'execution_complete':True,'total_new_head_updates':4096,'source_pose_updates':0,
        'original_pose_and_head_files_unchanged':True,'selected_head_for_further_engineering':selected,
        'absolute_guards':{'event_aware':{'passed':passed,'criteria':flags}}}


class CompositeHeadTest(unittest.TestCase):
    def artifact(self,path,head):
        torch.save({'format':'gripper-event-repair-head-v1','source_pose_sha256':SOURCE_SHA,
            'source_gripper_sha256':SOURCE_HEAD_SHA,'mode':'event_aware','additional_updates':2048,
            'total_head_updates':4096,'state':head.state_dict()},path)

    def test_real_format_roundtrip_no_false_admission(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            path=Path(folder)/'head.pt'; original=GripperSidecar(); self.artifact(path,original)
            report=receipt()
            with self.assertRaisesRegex(ValueError,'not admitted'): load_versioned_head(path,pair_report=report)
            head,meta=load_versioned_head(path,pair_report=report,allow_unadmitted_engineering=True,
                expected_head_sha256=sha(path))
            self.assertFalse(meta['offline_head_admitted']); self.assertFalse(meta['formal_scoring_admitted'])
            self.assertTrue(meta['explicit_unadmitted_engineering_mode'])
            self.assertIsNone(report['selected_head_for_further_engineering'])
            self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in original.state_dict().items()))
            self.assertTrue(all(not p.requires_grad for p in head.parameters()))

    def test_receipt_selection_and_corrupt_state_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            path=Path(folder)/'head.pt'; self.artifact(path,GripperSidecar())
            head,meta=load_versioned_head(path,pair_report=receipt('event_aware',True))
            self.assertTrue(meta['offline_head_admitted']); self.assertFalse(meta['formal_scoring_admitted'])
            with self.assertRaises(ValueError): load_versioned_head(path,pair_report=receipt('state_control',True))
            with self.assertRaises(ValueError): load_versioned_head(path,pair_report=receipt('event_aware',True),expected_head_sha256='bad')
            value=torch.load(path,weights_only=False); value['state'].pop('pose_projection.bias'); torch.save(value,path)
            with self.assertRaises(RuntimeError): load_versioned_head(path,pair_report=receipt(),allow_unadmitted_engineering=True)

    def test_source_schema_and_wrong_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
            path=Path(folder)/'head.pt'
            value={'format':'dino-gripper-sidecar-v1','source_checkpoint_sha256':SOURCE_SHA,
                'current_input_semantics':'bridge_measured_affine_unbounded_v1','updates':2048,
                'offline_gripper_admitted':False,'state':GripperSidecar().state_dict()}
            torch.save(value,path)
            _,meta=load_versioned_head(path,allow_unadmitted_engineering=True)
            self.assertEqual(meta['total_head_updates'],2048)
            value['source_checkpoint_sha256']='changed'; torch.save(value,path)
            with self.assertRaises(ValueError): load_versioned_head(path,allow_unadmitted_engineering=True)


if __name__=='__main__': unittest.main()
