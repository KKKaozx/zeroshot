"""Deletion guards exercised only on tiny files inside an owned workspace .tmp."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cleanup_policy',ROOT/'scripts/cluster_reclaim_space_20261029.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def write(root,name,data=b'tiny fixture'):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p

class Tests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'.tmp';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='space_reclaim_test_',dir=parent)
        self.root=Path(self.temp.name).resolve()
        assert self.root.parent==parent.resolve() and ROOT.resolve() in self.root.parents
        self.stack=[]
    def tearDown(self):
        for p in reversed(self.stack):p.stop()
        self.temp.cleanup()
    def replace(self,name,value):
        p=patch.object(m,name,value);p.start();self.stack.append(p)
    def fixtures(self):
        keep={}
        for rel in m.ENDPOINTS:
            p=write(self.root,rel,('final '+rel).encode());keep[rel]=m.sha(p)
        self.replace('ENDPOINTS',keep)
        targets=[]
        for rel,reason,group,expected in m.FIXED:
            p=write(self.root,rel,('obsolete '+rel).encode())
            targets.append((rel,reason,group,m.sha(p) if expected else None))
        self.replace('FIXED',tuple(targets));self.replace('BRIDGE_PROFILES',{})
        write(self.root,'runs/oxe-pose-pair-232587/real_pose/latest_state.pt',b'current immutable source')
        write(self.root,'runs/oxe-target-short-231297/sample/latest_step.pt',b'current parent source')
        write(self.root,'.tmp/pip/http-v2/download.body');write(self.root,'.tmp/pip/wheels/wheel.whl')
        write(self.root,'.tmp/conda/package.tar.bz2');write(self.root,'.tmp/conda/cache/repodata.json')
        write(self.root,'.tmp/conda/extracted_package/lib/runtime.so',b'keep extracted package')
        write(self.root,'envs/current/lib/runtime.so',b'keep installed env')
        write(self.root,'runs/pusht-dino-part2-222758/report.json',b'keep scientific result')
        write(self.root,'data/oxe_nominal_pool_v1/input.tfrecord',b'keep raw data')
        return m.build_plan(self.root)

    def test_all_declared_categories_and_exact_deletion_idempotency(self):
        entries,guards,kept=self.fixtures();self.assertGreater(len(entries),15)
        keep=[self.root/p for p in m.ENDPOINTS]+[
            self.root/'data/oxe_nominal_pool_v1/input.tfrecord',self.root/'envs/current/lib/runtime.so',
            self.root/'.tmp/conda/extracted_package/lib/runtime.so',self.root/'runs/pusht-dino-part2-222758/report.json']
        before={str(p):p.read_bytes() for p in keep}
        result=m.execute(self.root,entries,guards,kept,guard=lambda:None)
        self.assertEqual(result['deleted_files'],len(entries));self.assertTrue(all(not (self.root/r['path']).exists() for r in entries))
        self.assertTrue(all(p.read_bytes()==before[str(p)] for p in keep))
        records=[json.loads(x) for x in Path(result['receipt']).read_text().splitlines()]
        self.assertEqual(records[-1]['type'],'complete');self.assertEqual(sum(r['type']=='deleted' for r in records),len(entries))
        self.assertFalse(m.build_plan(self.root)[0])

    def test_busy_guard_prevents_every_deletion(self):
        entries,guards,kept=self.fixtures()
        def busy():raise ValueError('Other queued/running job')
        with self.assertRaisesRegex(ValueError,'Other queued'):m.execute(self.root,entries,guards,kept,guard=busy)
        self.assertTrue(all((self.root/r['path']).exists() for r in entries))

    def test_only_known_disjoint_233531_allowed(self):
        with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'233531|RUNNING|oxe-sample-pair\n','')):
            self.assertIn('233531',m.jobs_guard())
        for value in ['233999|PENDING|something\n','233531|RUNNING|different\n']:
            with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,value,'')):
                with self.assertRaises(ValueError):m.jobs_guard()

    def test_changed_retained_model_stops_all(self):
        entries,guards,kept=self.fixtures();key=next(iter(guards));(self.root/key).write_bytes(b'corrupt final')
        with self.assertRaisesRegex(ValueError,'Retained model'):m.execute(self.root,entries,guards,kept,guard=lambda:None)
        self.assertTrue(all((self.root/r['path']).exists() for r in entries))

    def test_changed_candidate_or_parent_digest_stops_all(self):
        entries,guards,kept=self.fixtures();(self.root/entries[0]['path']).write_bytes(b'changed candidate')
        with self.assertRaisesRegex(ValueError,'Candidate changed'):m.execute(self.root,entries,guards,kept,guard=lambda:None)
        self.assertTrue(all((self.root/r['path']).exists() for r in entries))
        entries,guards,kept=m.build_plan(self.root)
        next(r for r in entries if r['expected_sha256'])['expected_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Obsolete checkpoint differs'):m.execute(self.root,entries,guards,kept,guard=lambda:None)
        self.assertTrue(all((self.root/r['path']).exists() for r in entries))

    def test_protected_source_inode_alias_prevents_deletion(self):
        self.fixtures();candidate=self.root/'runs/pusht-dino-cache-221055/visual.npy'
        source=self.root/'runs/oxe-pose-pair-232587/real_pose/latest_state.pt'
        original=m.stamp;inode=original(source)[:2]
        # Windows sandbox disallows creating hardlinks. Exercise the real rejection
        # path with a stat fixture; no assertion of an actual filesystem hardlink.
        def collision(p):
            value=original(p)
            if p==candidate:value[:2]=inode
            return value
        with patch.object(m,'stamp',side_effect=collision):
            entries,guards,kept=m.build_plan(self.root)
            with self.assertRaisesRegex(ValueError,'shares retained'):m.execute(self.root,entries,guards,kept,guard=lambda:None)
        self.assertTrue(candidate.is_file())

    def test_path_escape_and_protected_prefixes(self):
        for rel in ['../oops','/tmp/oops','a\\b']:
            with self.assertRaises(ValueError):m.checked(self.root,rel)
        for prefix in m.PROTECTED_PREFIXES:self.assertTrue(m.protected(prefix+'/a.pt'))

    def test_missing_successor_blocks_only_its_retirement(self):
        self.fixtures();(self.root/m.GROUP_ENDPOINT['dino']).unlink()
        entries,guards,kept=m.build_plan(self.root)
        self.assertNotIn('runs/pusht-dino-cache-221055/visual.npy',[r['path'] for r in entries])
        self.assertIn('runs/pusht-dino-cache-221055/visual.npy',[r['path'] for r in kept])
        self.assertTrue(any(r['path'].startswith('.tmp/pip/') for r in entries))

    def test_duplicate_zip_requires_all_members_byte_exact(self):
        self.fixtures();write(self.root,'old_bundle/file.py',b'original source');write(self.root,'old_bundle/raw.dat',b'raw fixture')
        good=self.root/'old_bundle.zip'
        with zipfile.ZipFile(good,'w') as z:
            z.write(self.root/'old_bundle/file.py','old_bundle/file.py');z.write(self.root/'old_bundle/raw.dat','old_bundle/raw.dat')
        self.assertTrue(m.unpacked_zip_exact(self.root,good))
        entries,_,_=m.build_plan(self.root);self.assertIn('old_bundle.zip',[r['path'] for r in entries])
        (self.root/'old_bundle/raw.dat').write_bytes(b'changed')
        self.assertFalse(m.unpacked_zip_exact(self.root,good))
        for name in ['../escape','loose.txt','old_bundle/missing']:
            bad=self.root/'bad.zip'
            with zipfile.ZipFile(bad,'w') as z:z.writestr(name,b'data')
            self.assertFalse(m.unpacked_zip_exact(self.root,bad))

    def test_bridge_old_adam_requires_standalone_final_and_report(self):
        self.fixtures();folder='runs/old_bridge';report=write(self.root,folder+'/report.json',b'original report')
        final=write(self.root,folder+'/final.pt',b'complete final');old=write(self.root,folder+'/adam.pt',b'old optimizer')
        self.replace('BRIDGE_PROFILES',{folder:{'report_sha256':m.sha(report),'retain':{'final.pt':{'sha256':m.sha(final)}},
            'retire':{'adam.pt':{'sha256':m.sha(old)}}}})
        entries,guards,_=m.build_plan(self.root)
        self.assertIn(folder+'/adam.pt',[r['path'] for r in entries]);self.assertIn(folder+'/final.pt',guards)
        final.unlink();entries,guards,kept=m.build_plan(self.root)
        self.assertNotIn(folder+'/adam.pt',[r['path'] for r in entries]);self.assertTrue(any(r['path']==folder for r in kept))


if __name__=='__main__':unittest.main(verbosity=2)
