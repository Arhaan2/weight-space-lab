"""A pre-load continuation must never cover an existing or ambiguous model call."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_mechanistic_preload_continuation.py'
spec = importlib.util.spec_from_file_location('preload_continuation', SCRIPT)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


class PreloadContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.target = self.root / 'execution/candidate-1'
        self.target.mkdir(parents=True)
        self.supervisor = {'status': 'stopped', **dict.fromkeys(continuation.ZERO_COUNTS, 0),
            'outputs': [], 'warmup_outputs': [], 'supervisor_seconds': 6.1,
            'cleanup': {**dict.fromkeys(continuation.CLEANUP, True), 'retained_owned_pids_alive': []},
            'generation_budget': {'charged': dict.fromkeys(('model_probes','requested_sequences','generated_tokens','pending_requests','retries'), 0)}}
        driver = {k:False for k in ('load_intent_observed','load_acknowledged','generation_intent_observed',
                                    'generation_acknowledged','generation_call_intent_observed')}
        driver.update(generation_acknowledgements=0, generation_intents=0)
        self.terminal = {'status':'failed','driver':driver,'execution_error':'fixture pre-load closure'}
        for path,value in ((self.target/'supervisor-result.json',self.supervisor),(self.target/'result.json',self.terminal),
                           (self.root/'manifest.json',{}),(self.root/'joint-report.json',{})):
            path.write_text(json.dumps(value))
        (self.target/'events.jsonl').write_text(json.dumps({'type':'driver-error'})+'\n')
        with sqlite3.connect(self.root/'generation-budget.sqlite3') as c:
            c.execute('CREATE TABLE attempts (request_id TEXT)')

    def test_zero_call_failure_accepts_only_durable_empty_capacity(self):
        proof = continuation.verify_zero_call_failure(self.root)
        self.assertEqual(proof['prior_target_generations'],0)
        self.assertEqual(proof['prior_engine_loads'],0)
        self.assertEqual(proof['preload_elapsed_seconds'],6.1)
        self.assertEqual(len(proof['original_ledger_sha256']),64)

    def test_any_target_event_intent_grant_or_attempt_blocks_continuation(self):
        for event in continuation.FORBIDDEN_EVENTS:
            with self.subTest(event=event):
                (self.target/'events.jsonl').write_text(json.dumps({'type':event})+'\n')
                with self.assertRaises(ValueError): continuation.verify_zero_call_failure(self.root)
        (self.target/'events.jsonl').write_text(json.dumps({'type':'driver-error'})+'\n')
        for path in (self.root/'load-grant.json',self.target/'load-intent.json',self.target/'generation-1-ack.json'):
            path.write_text('{}')
            with self.assertRaises(ValueError): continuation.verify_zero_call_failure(self.root)
            path.unlink()
        with sqlite3.connect(self.root/'generation-budget.sqlite3') as c:
            c.execute("INSERT INTO attempts VALUES ('generation-1')")
        with self.assertRaisesRegex(ValueError,'ledger'): continuation.verify_zero_call_failure(self.root)

    def test_unknown_counts_or_unconfirmed_cleanup_cannot_be_reinterpreted_as_zero(self):
        self.supervisor['engine_load_attempts']=None
        (self.target/'supervisor-result.json').write_text(json.dumps(self.supervisor))
        with self.assertRaises(ValueError): continuation.verify_zero_call_failure(self.root)
        self.supervisor['engine_load_attempts']=0
        self.supervisor['cleanup']['chrome_os_exit_verified']=False
        (self.target/'supervisor-result.json').write_text(json.dumps(self.supervisor))
        with self.assertRaisesRegex(ValueError,'cleanup'): continuation.verify_zero_call_failure(self.root)

    def test_consumed_continuation_cannot_replay(self):
        frozen=(self.root, {}, {'started':True}, {}, {}, {})
        with patch.object(continuation,'load',return_value=frozen), patch.object(continuation.joint_browser,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'already spent'): continuation.run(self.root)
            execute.assert_not_called()


if __name__=='__main__': unittest.main()
