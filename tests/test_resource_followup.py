"""No-load checks for the separate resource controller's safety/accounting boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab import resource_followup as study


def initial():
    return {'halted':False,'active':None,'charged_browser_seconds':0,'charged_local_seconds':0,
        'deadline_epoch_seconds':100000,'sessions':[{'position':i+1,'artifact_key':key,
        'key':f'{i+1:02d}-{key}','status':'pending'} for i,key in enumerate(study.ORDER)]}


def execution():
    return {'status':'completed','engine_load_attempts':1,'warmup_dispatches':1,
        'scored_dispatches':60,'generation_dispatches':61,
        'cleanup':{**{k:True for k in (*study.CLEANUP,'native_capture_exit_confirmed')},'retained_owned_pids_alive':[]}}


class ResourceController(unittest.TestCase):
    def reserve(self,state,pos=1):
        return study.reserve(state,pos,study.LIMITS,100,100*1024**3,0)

    def test_fixed_order_once_only_and_exact_total_ceiling(self):
        state=initial()
        with self.assertRaises(ValueError):self.reserve(state,2)
        for pos in range(1,11):
            self.reserve(state,pos)
            with self.assertRaises(RuntimeError):self.reserve(state,pos)
            self.assertTrue(study.finish(state,pos,300,execution(),{'status':'complete','measured_requests':60,'warmups':1}))
        self.assertEqual(sum(r['capacity_reserved']['dispatches'] for r in state['sessions']),610)
        self.assertEqual(sum(r['capacity_reserved']['target_tokens'] for r in state['sessions']),38560)
        with self.assertRaises(ValueError):self.reserve(state,1)

    def test_missing_cleanup_stops_preserves_cost_and_never_retries(self):
        state=initial();self.reserve(state);bad=execution();bad['cleanup']['native_capture_exit_confirmed']=False
        self.assertFalse(study.finish(state,1,17,bad,{'status':'complete','measured_requests':60,'warmups':1}))
        self.assertEqual(state['charged_browser_seconds'],17)
        self.assertEqual(state['sessions'][0]['capacity_reserved']['target_tokens'],3856)
        self.assertEqual(state['sessions'][0]['status'],'stopped')
        with self.assertRaises(RuntimeError):self.reserve(state,1)
        with self.assertRaises(RuntimeError):self.reserve(state,2)

    def test_unknown_partial_result_and_exhausted_budget_stop(self):
        state=initial();self.reserve(state)
        self.assertFalse(study.finish(state,1,9,None,None,'Interrupted'))
        self.assertEqual(state['charged_local_seconds'],9)
        other=initial();other['charged_browser_seconds']=5101
        with self.assertRaises(TimeoutError):self.reserve(other)
        self.assertIsNone(other['active'])

    def test_missing_replicate_or_signal_never_becomes_mean(self):
        rows=[dict(r,summary={'scored_output_tokens':10}) for r in initial()['sessions']]
        rows[0]['status']='completed'
        groups,comparisons=study.aggregate_sessions(rows)
        self.assertEqual(groups['pristine']['metrics']['scored_output_tokens']['values'],[10,None])
        self.assertIsNone(groups['pristine']['metrics']['scored_output_tokens']['mean'])
        for row in rows:row['status']='completed'
        groups,comparisons=study.aggregate_sessions(rows,matched_environment=False)
        self.assertEqual(groups['v1']['metrics']['scored_output_tokens']['mean'],10)
        self.assertIsNone(comparisons['v1']['metrics']['scored_output_tokens']['absolute_delta'])
        rows[-1]['summary']['scored_output_tokens']=float('nan')
        self.assertIsNone(study.aggregate_sessions(rows)[0]['pristine']['metrics']['scored_output_tokens']['mean'])

    def test_manifest_binds_artifact_study_position_and_build(self):
        artifact={'artifact_sha256':'artifact','source_checkpoint_sha256':'source'}
        frozen={'artifacts':{'pristine':artifact},'environment':{'browser':'one'},'config':{'study_id':'test'},'served_build':[{'sha256':'build'}]}
        row=initial()['sessions'][0]
        manifest={'slot':row['key'],'study_manifest_sha256':study.canonical_hash(frozen),'artifact_sha256':'artifact',
            'checkpoint':{'sha256':'source'},'arm':'baseline','environment':frozen['environment']}
        config={'convertedArtifactSha256':'artifact','sourceCheckpointSha256':'source','runId':'test-'+row['key'],'servedBuild':frozen['served_build']}
        with tempfile.TemporaryDirectory() as folder, patch.object(study.runtime,'load') as runtime_load:
            path=Path(folder)
            (path/'baseline-config.json').write_text(json.dumps(config))
            (path/'manifest.json').write_text(json.dumps(manifest))
            study.verify_session_binding(path,frozen,row);runtime_load.assert_called_once_with(path)
            for key,value in [('artifact_sha256','other'),('study_manifest_sha256','other'),('slot','02-v1')]:
                changed=copy.deepcopy(manifest);changed[key]=value
                (path/'manifest.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):study.verify_session_binding(path,frozen,row)


if __name__=='__main__':unittest.main()
