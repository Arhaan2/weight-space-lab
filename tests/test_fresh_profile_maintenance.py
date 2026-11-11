import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('maintenance',Path(__file__).resolve().parents[1] / 'scripts/maintain_fresh_profiles.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.key='quality-01-pristine'
        self.session=self.root/'sessions'/self.key;self.target=self.session/'execution/baseline-1';self.profile=self.target/'chrome-profile'
        self.leaf=self.profile/m.ALLOWLIST[0];self.leaf.mkdir(parents=True)
        for name in ('a','b'):(self.leaf/name).write_bytes(b'cache bytes')
        for name in ('Last Version','Local State'):(self.profile/name).write_text('preserved metadata')
        leveldb=self.profile/'Default/IndexedDB/http_127.0.0.1_4198.indexeddb.leveldb';leveldb.mkdir();(leveldb/'CURRENT').write_text('immutable cache metadata')
        self.supervisor={'status':'completed','study_manifest_sha256':'study','cleanup':dict.fromkeys(m.REQUIRED_CLEANUP,True)}
        self.supervisor['cleanup'].update(retained_owned_pids_alive=[],forced_termination=False)
        self.row={'key':self.key,'artifact_key':'pristine','status':'completed','observed':{'status':'completed','cleanup':self.supervisor['cleanup']}}
        self.state={'active':None,'study_sha256':'study'};self.study={'files':{},'benchmark_source':str(self.root/'benchmark')}
        (self.root/'benchmark').mkdir();(self.session/'benchmark').symlink_to(self.root/'benchmark',target_is_directory=True)
        for p,v in [(self.target/'supervisor-result.json',self.supervisor),(self.target/'browser-session.json',{'profilePath':str(self.profile)}),(self.session/'manifest.json',{'files':{}})]:p.write_text(json.dumps(v))
        self.verify=lambda *args:({'files':{}},{})
    def plan(self,**kw):return m.plan_cleanup(self.root,self.study,self.state,self.row,self.verify,processes=kw.get('processes',lambda p:[]))
    def test_only_allowlisted_payload_removed(self):
        plan=self.plan();removed=m.remove_planned(plan,self.session,processes=lambda p:[])
        self.assertEqual(len(removed),2);self.assertFalse(self.leaf.exists())
        self.assertEqual(m.evidence_snapshot(self.session,self.profile,self.study['benchmark_source']),plan['preserved_evidence_sha256'])
        self.assertTrue((self.profile/'Local State').exists())
    def test_active_or_incomplete_rejected(self):
        for field,value in [('active',{'key':'running'})]:
            self.state[field]=value
            with self.assertRaises(ValueError):self.plan()
        self.state['active']=None;self.row['status']='running'
        with self.assertRaises(ValueError):self.plan()
    def test_live_owned_process_rejected(self):
        with self.assertRaises(ValueError):self.plan(processes=lambda p:[123])
    def test_cache_symlink_rejected(self):
        (self.leaf/'unsafe').symlink_to(self.profile/'Local State')
        with self.assertRaises(ValueError):self.plan()
    def test_shared_hardlink_rejected(self):
        import os
        os.link(self.profile/'Local State',self.leaf/'unsafe')
        with self.assertRaises(ValueError):self.plan()
    def test_frozen_reference_rejected(self):
        self.study['files'][str(self.leaf/'a')]='digest'
        with self.assertRaises(ValueError):self.plan()
    def test_changed_evidence_rejected(self):
        plan=self.plan();(self.profile/'Local State').write_text('changed')
        with self.assertRaises(ValueError):m.remove_planned(plan,self.session,processes=lambda p:[])
        self.assertTrue((self.leaf/'a').exists())
    def test_changed_cache_inventory_rejected(self):
        plan=self.plan();(self.leaf/'c').write_text('new')
        with self.assertRaises(ValueError):m.remove_planned(plan,self.session,processes=lambda p:[])
        self.assertTrue((self.leaf/'a').exists())
    def test_retains_unrelated_chrome_singleton_links(self):
        for name,target in [('SingletonSocket','/tmp/stale-socket'),('RunningChromeVersion','157.0:1')]:
            (self.profile/name).symlink_to(target)
        plan=self.plan();m.remove_planned(plan,self.session,processes=lambda p:[])
        self.assertTrue((self.profile/'SingletonSocket').is_symlink())
        self.assertTrue((self.session/'benchmark').is_symlink())
    def test_benchmark_link_retarget_rejected(self):
        (self.session/'benchmark').unlink();(self.root/'wrong').mkdir()
        (self.session/'benchmark').symlink_to(self.root/'wrong',target_is_directory=True)
        with self.assertRaises(ValueError):self.plan()
    def test_partial_removal_progress_retained(self):
        plan=self.plan();removed=[];original=Path.unlink
        def fail_second(path,*args,**kw):
            if path.name=='b':raise OSError('fixture failure')
            return original(path,*args,**kw)
        with patch.object(Path,'unlink',fail_second):
            with self.assertRaises(OSError):m.remove_planned(plan,self.session,processes=lambda p:[],removed=removed)
        self.assertEqual(len(removed),1);self.assertFalse((self.leaf/'a').exists());self.assertTrue((self.leaf/'b').exists())

if __name__=='__main__':unittest.main()
