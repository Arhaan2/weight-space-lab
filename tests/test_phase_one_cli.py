import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_space_lab.__main__ import main, read_json


ROOT = Path(__file__).parents[1]


class PhaseOneCliTests(unittest.TestCase):
    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(args)
        return status, json.loads(output.getvalue())

    def test_diagnostics_work_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, '-S', '-m', 'weight_space_lab', 'diagnose'],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsInstance(json.loads(result.stdout), dict)
        self.assertNotIn(str(ROOT), result.stdout)

    def test_diagnostic_output_is_append_only_and_probe_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'diagnostics.json'
            with patch('weight_space_lab.diagnostics.collect_diagnostics', return_value={'schema_version': 1}) as collect:
                status, report = self.invoke(['diagnose', '--out', str(out)])
                self.assertEqual(status, 0)
                collect.assert_called_once_with(probe_backends=False)
                self.assertEqual(json.loads(out.read_text()), report)
                status, report = self.invoke(['diagnose', '--out', str(out)])
                self.assertEqual(status, 2)
                self.assertNotIn(str(out), json.dumps(report))
                self.assertEqual(json.loads(out.read_text()), {'schema_version': 1})
                self.invoke(['diagnose', '--probe-backends'])
                collect.assert_called_with(probe_backends=True)

    def test_invalid_evidence_is_a_nonzero_cli_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for command, payload in [('check-records', [{}]), ('check-status', {}),
                                     ('check-schedule', [])]:
                with self.subTest(command=command):
                    path.write_text(json.dumps(payload))
                    status, report = self.invoke([command, str(path)])
                    self.assertEqual(status, 2)
                    self.assertIn('error', report)

    def test_schedule_cli_and_unready_design_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'schedule.json'
            path.write_text(json.dumps([
                {'benchmark': 'fixture', 'item_id': 'a', 'sample_index': 0, 'seed': 7},
                {'benchmark': 'fixture', 'item_id': 'b', 'sample_index': 0, 'seed': 7},
            ]))
            self.assertEqual(self.invoke(['check-schedule', str(path)])[0], 0)
        template = str(ROOT / 'configs/qwen_math_design.json')
        self.assertEqual(self.invoke(['check', template])[0], 0)
        self.assertEqual(self.invoke(['check', template, '--require-ready'])[0], 2)

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.json'
            for text in ['{"a": 1, "a": 2}', '{"a": NaN}', '[Infinity]', '[-Infinity]']:
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    read_json(path)


if __name__ == '__main__':
    unittest.main()
