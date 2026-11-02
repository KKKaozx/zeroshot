import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np

path=Path(__file__).resolve().parents[1]/'tools/audit_existing_oxe_position.py'
spec=importlib.util.spec_from_file_location('existing_oxe_audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class PositionAuditTests(unittest.TestCase):
    def inputs(self):
        target=np.zeros((1,16,8));target[...,0]=1;mask=np.zeros_like(target);mask[...,:2]=1
        pred=np.repeat(target[None],3,axis=0);return pred,target,mask

    def test_draw_variance_separated_from_mean_prediction_error(self):
        p,t,m=self.inputs();p[:,0,:,0]=np.array([0,1,2])[:,None]
        r=audit.position_stats(p,t,m,[0])
        self.assertAlmostEqual(r['mse'],2/3);self.assertEqual(r['mean_prediction_error_squared'],0)
        self.assertAlmostEqual(r['between_draw_variance'],2/3);self.assertAlmostEqual(r['mse_excess_over_static'],-1/3)
    def test_wrong_direction_is_systematic_not_draw_variance(self):
        p,t,m=self.inputs();p[...,0]=-1
        r=audit.position_stats(p,t,m,[0]);self.assertEqual(r['energy_weighted_direction_alignment'],-1)
        self.assertEqual(r['between_draw_variance'],0);self.assertEqual(r['mse'],4);self.assertEqual(r['mse_excess_over_static'],3)
    def test_unobserved_lt_z_and_padded_future_excluded(self):
        p,t,m=self.inputs();p[...,2]=999;p[:,:,2:,0]=999;m[:,2:]=0
        r=audit.position_stats(p,t,m,[0]);self.assertEqual(r['mse'],0)
        self.assertEqual(r['by_future_step']['4']['valid_windows'],0)
        self.assertIsNone(r['by_future_step']['4']['position_error'])
        self.assertEqual(r['by_future_step']['1']['position_error'],0)
    def test_metric_comparison_handles_null_and_rejects_changed_metrics(self):
        audit.close_tree({'a':[None,1.]},{'a':[None,1.+1e-12]})
        with self.assertRaises(AssertionError):audit.close_tree({'a':[None,1.]},{'a':[None,1.01]})

    def test_full_reader_recomputes_existing_npz_without_models(self):
        import sys
        sys.path.insert(0,str(audit.BASE));from metrics import summarize
        selection=audit.read(audit.BASE/'selection.json');rows=selection['train']+selection['validation']
        t=np.zeros((120,16,8),np.float32);t[...,0]=1;t[...,6]=1;m=np.ones_like(t)
        for i,row in enumerate(rows):
            if row['source']=='Language Table':m[i,:,2:]=0
        outputs={name:np.repeat(t[None],3,axis=0) for name in ['epsilon','sample','static']}
        outputs['epsilon'][...,0]=2;outputs['sample'][...,0]=1.5;outputs['static'][...,0]=0
        metrics=summarize(outputs,t,m,selection)
        report={'execution_complete':True,'optimizer_updates':0,'selection_sha256':audit.sha(audit.BASE/'selection.json'),'metrics':metrics}
        receipt={'execution_complete':True,'source_training_job':231297,'evaluation_job':'232091','training_updates':0,
            'original_selection_sha256':report['selection_sha256'],'evaluation':{'metrics':metrics}}
        parent=audit.ROOT/'.tmp';parent.mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(prefix='existing_position_read_',dir=parent)
        assert Path(temp.name).resolve().parent==parent.resolve() and audit.ROOT.resolve() in Path(temp.name).resolve().parents
        with temp as folder:
            folder=Path(folder)
            for name,data in [('report.json',report),('repair_receipt.json',receipt)]:
                (folder/name).write_text(json.dumps(data),encoding='utf-8')
            np.savez_compressed(folder/'predictions_and_targets.npz',targets=t,masks=m,**outputs)
            r=audit.audit(folder)
            self.assertTrue(r['generated_metrics_recomputed_from_original_arrays']);self.assertEqual(r['model_calls'],0)
            self.assertEqual(r['statistics']['validation']['Bridge']['sample']['mse'],.25)
            self.assertEqual(r['statistics']['train']['Language Table']['static']['predicted_motion_energy'],0)
            report['metrics']['by_source']['train']['Bridge']['epsilon']['path_position_normalized']+=.1
            (folder/'report.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises(AssertionError):audit.audit(folder)


if __name__=='__main__':unittest.main(verbosity=2)
