import copy
import json
from pathlib import Path
import tempfile
import unittest
from weight_space_lab.browser_resource import (RESOURCE_SETTINGS,RESOURCE_PROMPT,RESOURCE_ORDER,
    validate_derivation,validate_session,summarize_report,aggregate,freeze_comparison,analyze,served_build)
from weight_space_lab.protocol import canonical_hash

class BrowserResourceTests(unittest.TestCase):
    def standard(self):
        return json.loads(Path('configs/browser_resource_comparison.json').read_text())['resolved_standard']

    def test_actual_pinned_profile_is_frozen_without_catalog_queue(self):
        value=self.standard();validate_derivation(value)
        self.assertEqual(RESOURCE_ORDER,('pristine_baseline','selected_astra','selected_astra','pristine_baseline'))
        for field in ('measuredRequests','requestLimitMs','baselineMs','idleMs','recoveryMs','totalLimitMs'):
            changed=copy.deepcopy(value);changed['plan'][field]+=1
            with self.assertRaises(ValueError):validate_derivation(changed)
        changed=copy.deepcopy(value);changed['prompts'][0]='new benchmark'
        with self.assertRaises(ValueError):validate_derivation(changed)
        changed=copy.deepcopy(value);changed['plan']['countSchedule']['betweenMs']=89000
        with self.assertRaises(ValueError):validate_derivation(changed)

    def test_session_rejects_task_substitution_extra_requests_and_hash_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            items=[{'item_id':f'resource-{i+1}','prompt':RESOURCE_PROMPT} for i in range(60)]
            path=Path(tmp)/'items.json';path.write_text(json.dumps(items))
            manifest={'workload_identity':'browser-llm-bench-standard','settings':RESOURCE_SETTINGS,
                'scheduled_prompts':60,'items_path':str(path),'items_sha256':canonical_hash(items)}
            config={'resourceComparison':True,'settings':RESOURCE_SETTINGS,'totalLimitMs':1800000,'items':items}
            config['configurationSha256']=canonical_hash(config)
            validate_session(tmp,manifest,config)
            for mutate in (lambda c:c.update(study='ifeval-browser-v1'),lambda c:c['items'].pop(),
                lambda c:c.update(totalLimitMs=5400000),lambda c:c.update(configurationSha256='0'*64)):
                bad=copy.deepcopy(config);mutate(bad)
                with self.assertRaises(ValueError):validate_session(tmp,manifest,bad)

    def report(self):
        return {'model':'pristine','status':'complete','loadMs':12,'samples':[
            {'warmup':True,'outputTokens':5,'responseMs':10,'finishReason':'stop'},
            {'warmup':False,'outputTokens':20,'responseMs':100,'finishReason':'stop'},
            {'warmup':False,'outputTokens':60,'responseMs':300,'finishReason':'length'}],
            'profile':{'totalMs':500,'points':[{'gpuBufferBytes':123,'wasmBytes':45}]}}

    def test_actual_lengths_warmup_and_distinct_memory_scopes(self):
        result=summarize_report(self.report())
        self.assertEqual(result['scored_output_tokens'],80)
        self.assertEqual(result['warmup_output_tokens'],5)
        self.assertEqual(result['response_ms_sum'],400)
        self.assertEqual(result['output_tokens_each_request'],[20,60])
        self.assertEqual(result['truncations'],1)
        self.assertEqual(result['gpu_buffer_descriptor_bytes_sampled_max'],123)
        self.assertEqual(result['wasm_address_space_bytes_sampled_max'],45)
        self.assertIsNone(result['energy']);self.assertIsNone(result['native_coverage'])

    def test_missing_is_not_zero_and_aggregates_use_sessions(self):
        report=self.report();report['samples'][1]['outputTokens']=None;report['profile']['points']=[]
        unknown=summarize_report(report);known=summarize_report(self.report())
        self.assertIsNone(unknown['scored_output_tokens'])
        self.assertIsNone(unknown['gpu_buffer_descriptor_bytes_sampled_max'])
        value=aggregate([unknown,known])['pristine']
        self.assertEqual(value['sessions'],2)
        self.assertEqual(value['metrics']['response_ms_sum']['mean'],400)
        self.assertIsNone(value['metrics']['scored_output_tokens']['mean'])

    def test_served_build_uses_actual_current_chunk_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);public=root/'public';(public/'benchmark-worker').mkdir(parents=True)
            for name in ('task-evaluation.html','task-evaluation.js','benchmark-worker/benchmark.worker.js','benchmark-worker/lib-newhash.js'):
                (public/name).write_text('synthetic build fixture')
            rows=served_build(root,'http://127.0.0.1:4198')
            self.assertEqual(len(rows),4)
            self.assertIn('http://127.0.0.1:4198/benchmark-worker/lib-newhash.js',[r['url'] for r in rows])
            self.assertTrue(all(len(r['sha256'])==64 for r in rows))
            (public/'benchmark-worker/benchmark.worker.js').unlink()
            with self.assertRaises(ValueError):served_build(root,'http://127.0.0.1:4198')

    def test_empty_observation_is_missing_not_zero(self):
        report=self.report(); report['samples']=[]
        result=summarize_report(report)
        self.assertEqual(result['measured_requests'],0)
        self.assertIsNone(result['scored_output_tokens'])
        self.assertIsNone(result['response_ms_sum'])

    def test_frozen_preset_queues_only_distinct_selected_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'baseline';base.mkdir()
            (base/'baseline-config.json').write_text(json.dumps({'model':{'id':'pristine-sha'},'convertedArtifactSha256':'a'*64}))
            artifact=root/'artifact.json';artifact.write_text(json.dumps({'model_id':'selected-sha','artifact_sha256':'b'*64}))
            study={'baseline':{'baseline_root':str(base)},'resource_preset':{'pinned':'source'}}
            selection={'resource_order':list(RESOURCE_ORDER),'resource_artifacts':{'selected_astra':{'artifact_path':str(artifact),'artifact_sha256':'b'*64}}}
            digest=freeze_comparison(root,study,selection)
            saved=json.loads((root/'resource-comparison-preset.json').read_text())
            self.assertEqual(saved['preset']['models'],['pristine-sha','selected-sha'])
            self.assertFalse(saved['preset']['requirePower'])
            self.assertEqual(saved['session_order'],list(RESOURCE_ORDER))
            self.assertEqual(saved['target_tokens_max'],15424)
            self.assertEqual(digest,freeze_comparison(root,study,selection))
            artifact.write_text(json.dumps({'model_id':'selected-sha','artifact_sha256':'a'*64}))
            with self.assertRaises(ValueError):freeze_comparison(root,study,selection)

    def test_baseline_fallback_has_no_resource_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selection={'resource_order':[]}
            (root/'state.json').write_text(json.dumps({'selection':selection}))
            (root/'final-selection.json').write_text(json.dumps(selection))
            result=analyze(root)
            self.assertEqual(result['scheduled_sessions'],0)
            self.assertEqual(result['sessions'],[])
            self.assertEqual(result['status'],'pristine_retained_no_comparison')
            self.assertFalse(result['matched_environment'])

if __name__=='__main__':unittest.main()
