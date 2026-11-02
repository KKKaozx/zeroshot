"""Small accounting fixtures, plus actual archived schedule consistency."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exposure_schedule_audit',ROOT/'tools/audit_oxe_training_exposure_schedule_20261029.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class Tests(unittest.TestCase):
    def test_counts_keep_unseen_windows_in_denominator(self):
        r=audit.count_summary([0,0,1,3])
        self.assertEqual(r['presentations'],4);self.assertEqual(r['mean_presentations'],1)
        self.assertEqual(r['never_sampled'],2);self.assertEqual(r['sampled_fraction'],.5)

    def test_low_warmup_lr_is_not_late_fine_training(self):
        rows=[{'indices':[0,1],'lr':1e-6,'clock_before':0},
            {'indices':[0,2],'lr':1e-5,'clock_before':1024}]
        s=audit.stage_summary(rows,np.array([0,1,1]),[0,1,2],1024)
        self.assertEqual(s['sources']['Bridge']['presentations'],2)
        self.assertEqual(s['sources']['Language Table']['presentations'],2)
        self.assertEqual(s['sources']['Bridge']['late_presentations_at_lr_at_most1e_5'],1)
        self.assertEqual(s['lr']['late_steps_at_most1e_5'],1)

    def test_candidate_preserves_source_slots_and_cycles_without_replacement(self):
        codes=np.array([0,0,1,1,1],dtype=np.int8)
        rows=[{'indices':[0,2,0,2],'lr':.1,'clock_before':0},
            {'indices':[0,2,0,2],'lr':.05,'clock_before':1},
            {'indices':[2],'lr':.01,'clock_before':2}]
        a,_=audit.source_cycle_candidate(rows,codes);b,_=audit.source_cycle_candidate(rows,codes)
        self.assertEqual(a,b)
        self.assertEqual([len(r['indices']) for r in a],[4,4,1])
        for old,new in zip(rows,a):
            np.testing.assert_array_equal(codes[old['indices']],codes[new['indices']])
            self.assertEqual(old['lr'],new['lr'])
        flat=np.concatenate([r['indices'] for r in a]);bridge=flat[codes[flat]==0];lt=flat[codes[flat]==1]
        self.assertEqual(set(bridge[:2]),{0,1});self.assertEqual(set(lt[:3]),{2,3,4})

    def test_actual_old_short_and_full_schedules(self):
        r=audit.audit();s=r['stages']
        self.assertEqual(s['old_positive_dino229116']['updates'],8192)
        self.assertEqual(s['old_positive_dino229116']['sources']['Bridge']['presentations'],65536)
        self.assertEqual(s['current_sample231297_prefix8192']['lr']['late_steps_at_most1e_5'],0)
        self.assertEqual(s['fixed8_diagnostic233426_extra512']['sources']['Bridge']['mean_presentations'],512)
        self.assertEqual(s['original_epsilon_full65536']['sources']['BC-Z']['never_sampled'],24389)
        self.assertEqual(r['new_training_updates'],0)
        self.assertTrue(r['cpu_only_source_cycle_candidate']['all_sources_all_eligible_windows_covered'])


if __name__=='__main__':unittest.main(verbosity=2)
