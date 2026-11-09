"""Prove suffix eligibility from synthetic stopped-run evidence, no model calls."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.prepare_ifeval_continuation import (stopped_prefix, validate_browser_offline,
    _BROWSER_OFFLINE_CHECKS, _BROWSER_UNCHANGED_FILES, _BROWSER_CHANGED_FILES, _BROWSER_BUILD_FILES)
from weight_space_lab.inventory import file_sha256


class ContinuationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.execution=self.root/'candidate/execution/candidate-1'
        self.execution.mkdir(parents=True)
        items=[{'item_id':str(i),'target':{'instruction_id_list':['fixture']*(2 if 94<=i<105 else 1)}} for i in range(128)]
        self.write('evaluation-items.json',items)
        self.write('candidate/baseline-config.json',{'items':[{'item_id':i['item_id']} for i in items]})
        self.write('candidate/execution/candidate-1/supervisor-result.json',{
            'status':'stopped','scored_dispatches':94,'generation_dispatches':95,
            'outputs':[{'item_id':i['item_id']} for i in items[:94]]})
        self.write('candidate/candidate/graded.json',{'records':[
            {'item_id':i['item_id'],'status':'ok' if n<94 else 'missing',
             'attempts':[{}] if n<94 else [],'raw_output':'fixture' if n<94 else None}
            for n,i in enumerate(items)],'grades':[
            {'item_id':i['item_id'],'status':'graded' if n<94 else 'not_applicable'} for n,i in enumerate(items)]})
        path=self.root/'candidate/generation-budget.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript('CREATE TABLE metadata(singleton INTEGER, cancelled INTEGER); INSERT INTO metadata VALUES(1,1); CREATE TABLE attempts(request_id TEXT,status TEXT,retry_of TEXT);')
            db.executemany('INSERT INTO attempts VALUES(?,?,?)',[(f'generation-{i}','completed',None) for i in range(1,96)])
        self.write('candidate/execution/candidate-1/generation-96-intent.json',{'request_id':'generation-96'})
        (self.execution/'events.jsonl').write_text(''.join(json.dumps({'type':'generation-call-intent',
            'data':{'request_id':f'generation-{i}'}})+'\n' for i in range(1,96)))

    def write(self,name,value):
        path=self.root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_denied_gate_leaves_exact_suffix_and_does_not_change_history(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        items,_,_,_=stopped_prefix(self.root)
        self.assertEqual(len(items[94:]),34)
        self.assertEqual(sum(len(i['target']['instruction_id_list']) for i in items[94:]),45)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_acknowledged_next_gate_cannot_be_called_unexecuted(self):
        self.write('candidate/execution/candidate-1/generation-96-ack.json',{'approved':True})
        with self.assertRaises(ValueError):
            stopped_prefix(self.root)

    def test_reserved_next_request_cannot_gain_fresh_capacity(self):
        with closing(sqlite3.connect(self.root/'candidate/generation-budget.sqlite3')) as db, db:
            db.execute('INSERT INTO attempts VALUES(?,?,?)',('generation-96','reserved',None))
        with self.assertRaises(ValueError):
            stopped_prefix(self.root)

    def test_attempted_tail_and_modified_prefix_are_rejected(self):
        path=self.root/'candidate/candidate/graded.json'
        value=json.loads(path.read_text())
        value['records'][94]['attempts']=[{}]
        self.write('candidate/candidate/graded.json',value)
        with self.assertRaises(ValueError):
            stopped_prefix(self.root)
        value['records'][94]['attempts']=[]
        value['records'][0]['item_id']='off-schedule'
        self.write('candidate/candidate/graded.json',value)
        with self.assertRaises(ValueError):
            stopped_prefix(self.root)

    def test_model_call_without_ack_is_not_unexecuted(self):
        with (self.execution/'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'type':'generation-call-intent','data':{'request_id':'generation-96'}})+'\n')
        with self.assertRaises(ValueError):
            stopped_prefix(self.root)


class BrowserOfflineBindingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base=Path(temp.name)
        self.old,self.root,self.repo=(self.base/name for name in ('original','continuation','repo'))
        for name in _BROWSER_UNCHANGED_FILES:
            for root in (self.old,self.root):
                self.write(root/'benchmark'/name,'unchanged synthetic source '+name)
        for name in _BROWSER_CHANGED_FILES:
            self.write(self.old/'benchmark'/name,'original synthetic source '+name)
            self.write(self.root/'benchmark'/name,'changed synthetic source '+name)
        for name in _BROWSER_BUILD_FILES:
            if not (self.root/'benchmark'/name).exists():
                self.write(self.root/'benchmark'/name,'synthetic built artifact '+name)
        self.write(self.old/'browser-schedule-offline.json','{"fixture":true}')
        self.write(self.repo/'integrations/ifeval-heldout-continuation.patch','synthetic source-only patch')
        self.record={'schema_version':1,'kind':'heldout_candidate_continuation_browser_offline_verification',
            'prepared_only':True,'execution_authorized':False,'engine_loads':0,'generation_dispatches':0,
            'target_forward_passes':0,'model_conversion_calls':0,
            'original_frozen_browser_sources_and_build_verified':True,
            'all_cloned_model_files_byte_identical':True,
            'required_configuration':{'study':'ifeval-browser-v1','evaluationStage':'heldout-continuation',
                'arm':'candidate','originalScheduleOffset':94,'originalScheduleLength':128,'scored_items':34,
                'warmups':1,'generation_dispatches':35,'max_generated_tokens':34832},
            'checks':[{'command':_BROWSER_OFFLINE_CHECKS[0],'passed':17,'failed':0}]
                +[{'command':command,'exit_code':0} for command in _BROWSER_OFFLINE_CHECKS[1:]],
            'source_patch_sha256':file_sha256(self.repo/'integrations/ifeval-heldout-continuation.patch'),
            'original_browser_offline_record_sha256':file_sha256(self.old/'browser-schedule-offline.json'),
            'unchanged_generation_runtime_and_harness':{name:file_sha256(self.root/'benchmark'/name)
                for name in _BROWSER_UNCHANGED_FILES},
            'changed_source':{name:{'original_sha256':file_sha256(self.old/'benchmark'/name),
                'continuation_sha256':file_sha256(self.root/'benchmark'/name)} for name in _BROWSER_CHANGED_FILES},
            'served_build_files':{name:file_sha256(self.root/'benchmark'/name) for name in _BROWSER_BUILD_FILES}}
        self.save_record()

    def write(self,path,text):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)

    def save_record(self):
        self.write(self.root/'browser-continuation-offline.json',json.dumps(self.record))

    def validate(self):
        return validate_browser_offline(self.old,self.root,self.repo)

    def test_successful_checks_validate_all_pins_without_mutation(self):
        before={str(p):p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        result=self.validate()
        self.assertTrue(result['passed'])
        self.assertEqual(result['verified_recorded_file_hashes'],38)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.base.rglob('*') if p.is_file()})

    def test_changed_generation_source_is_not_blessed_by_new_pin(self):
        self.write(self.root/'benchmark/src/benchmark/worker/benchmark.worker.ts','untested changed worker')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            self.validate()

    def test_stale_build_is_rejected_before_new_config_pin(self):
        self.write(self.root/'benchmark/public/task-evaluation.js','different untested bundle')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            self.validate()

    def test_failed_or_missing_offline_checks_cannot_freeze(self):
        self.record['checks'][0]['failed']=1
        self.save_record()
        with self.assertRaisesRegex(ValueError,'did not all pass'):
            self.validate()
        self.record['checks'][0]['failed']=0
        self.record['checks'].pop()
        self.save_record()
        with self.assertRaisesRegex(ValueError,'missing or duplicated'):
            self.validate()

    def test_changed_original_and_patch_hash_are_rejected(self):
        path=self.old/'benchmark'/(_BROWSER_CHANGED_FILES[0])
        original=path.read_text()
        self.write(path,'changed original schema')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            self.validate()
        self.write(path,original)
        self.write(self.repo/'integrations/ifeval-heldout-continuation.patch','different patch')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            self.validate()

    def test_nonzero_inference_or_different_schedule_is_rejected(self):
        self.record['engine_loads']=1
        self.save_record()
        with self.assertRaisesRegex(ValueError,'zero-inference'):
            self.validate()
        self.record['engine_loads']=0
        self.record['required_configuration']['scored_items']=35
        self.save_record()
        with self.assertRaisesRegex(ValueError,'candidate-only'):
            self.validate()


if __name__=='__main__':
    unittest.main()
