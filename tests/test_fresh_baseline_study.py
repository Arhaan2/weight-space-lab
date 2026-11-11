import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.fresh_baseline_study import (
    KEYS, LIMITS, ORDER, aggregate_resources, begin_action, environment_matches,
    expected_rows, finish_action, reserve, verify_converter_interpreter,
)


class FreshStudyContracts(unittest.TestCase):
    def test_converter_selector_retarget_is_rejected(self):
        from weight_space_lab.inventory import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);binary=root/'binary';binary.write_bytes(b'fixed interpreter')
            config=root/'pyvenv.cfg';config.write_text('fixed config')
            selector=root/'python';selector.symlink_to(binary.name)
            build={'converter_python':str(selector),'converter_interpreter_binding':{
                'selector':str(selector),'symlink_target':binary.name,'resolved_executable':str(binary.resolve()),
                'resolved_executable_sha256':file_sha256(binary),'pyvenv_config':str(config),'pyvenv_config_sha256':file_sha256(config)}}
            verify_converter_interpreter(build)
            replacement=root/'replacement';replacement.write_bytes(binary.read_bytes())
            selector.unlink();selector.symlink_to(replacement.name)
            with self.assertRaises(ValueError):verify_converter_interpreter(build)

    def state(self):
        return {'halted':False,'active':None,'sessions':expected_rows(),
            'charged_local_seconds':0,'charged_browser_seconds':0,'action_attempts':{},'stages':[],
            'editor_proposals_consumed':0,'build_consumed':0}

    def test_capacities_cover_exact_frozen_work(self):
        self.assertEqual(len(expected_rows()),18)
        self.assertEqual(6*32+12*60+18,LIMITS['dispatches'])
        self.assertEqual(6*32*1024+12*60*64+18*16,LIMITS['reserved_target_tokens'])
        self.assertEqual(ORDER,KEYS+list(reversed(KEYS)))

    def test_reservation_is_single_use_and_predecessor_bound(self):
        state=self.state();pristine=copy.deepcopy(state)
        with self.assertRaises(ValueError):reserve(state,'quality-02-v1',300,1)
        self.assertEqual(state,pristine)
        row=reserve(state,'quality-01-pristine',300,1)
        self.assertEqual(row['capacity_reserved']['dispatches'],33)
        self.assertEqual(row['capacity_reserved']['target_tokens'],32784)
        with self.assertRaises(RuntimeError):reserve(state,row['key'],300,2)
        state['active']=None
        with self.assertRaises(ValueError):reserve(state,row['key'],300,2)

    def test_halted_state_cannot_dispatch(self):
        state=self.state();state['halted']=True
        with self.assertRaises(RuntimeError):reserve(state,'quality-01-pristine',300,1)

    def test_unresolved_report_reservation_blocks_next_dispatch(self):
        state=self.state();state['reporting_active']={'reserved_seconds':900}
        with self.assertRaises(RuntimeError):reserve(state,'quality-01-pristine',300,1)

    def test_failed_editor_action_consumes_capacity(self):
        state=self.state();study={'deadline_epoch_seconds':time.time()+3600}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch('weight_space_lab.fresh_baseline_study.available'),patch('weight_space_lab.fresh_baseline_study.shutil.disk_usage') as disk:
                disk.return_value.free=64*1024**3
                begin_action(root,study,state,'editor',300)
                self.assertEqual(state['editor_proposals_consumed'],1)
                self.assertEqual(state['action_attempts']['editor']['status'],'running')
                with self.assertRaises(RuntimeError):finish_action(root,state,'editor',12,'Invalid response')
                self.assertTrue(state['halted'])
                self.assertEqual(state['charged_local_seconds'],12)
                with self.assertRaises(ValueError):begin_action(root,study,state,'editor',300)

    def test_missing_pairs_are_not_averaged_or_zero_filled(self):
        rows=[{'artifact_key':key,'status':'completed','summary':{'mean_response_ms':10,'host_power_watts':None}} for key in ORDER]
        rows[0]['status']='stopped'
        aggregates,comparisons=aggregate_resources(rows,True)
        self.assertIsNone(aggregates['pristine']['metrics']['mean_response_ms']['mean'])
        self.assertIsNone(comparisons['astra-r4']['metrics']['mean_response_ms']['absolute_delta'])
        self.assertIsNone(aggregates['astra-r4']['metrics']['host_power_watts']['mean'])

    def test_unmatched_environment_suppresses_deltas(self):
        rows=[{'artifact_key':key,'status':'completed','summary':{'mean_response_ms':10}} for key in ORDER]
        aggregates,comparisons=aggregate_resources(rows,False)
        self.assertEqual(aggregates['astra-r4']['metrics']['mean_response_ms']['mean'],10)
        self.assertIsNone(comparisons['astra-r4']['metrics']['mean_response_ms']['absolute_delta'])

    def test_observed_environment_requires_actual_fields(self):
        expected={'browser_version':'157.0.8103.39','hardware':{'hw.logicalcpu':'14'}}
        observed={'userAgent':'Macintosh Chrome/157.0.0.0','platform':'MacIntel','hardwareConcurrency':14,'chrome_version':'157.0.8103.39'}
        self.assertTrue(environment_matches(expected,observed))
        self.assertFalse(environment_matches(expected,{**observed,'platform':None}))
        self.assertFalse(environment_matches(expected,{**observed,'chrome_version':'156.0.0.0'}))


if __name__=='__main__':unittest.main()
