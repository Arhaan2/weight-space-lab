import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from weight_space_lab.__main__ import main


class CliTests(unittest.TestCase):
    def test_synthetic_demo_and_paired_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "demo"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["demo", "--out", str(out)])
                comparison = main(["compare", "--schedule", str(out / "schedule.json"),
                                   "--baseline", str(out / "baseline.json"),
                                   "--candidate", str(out / "candidate.json"),
                                   "--out", str(out / "reanalysis.json")])
            self.assertEqual((status, comparison), (0, 0))
            report = json.loads((out / "mechanics_report.json").read_text())
            self.assertFalse(report["language_model_evaluated"])
            self.assertFalse(report["scientific_result"])
            self.assertEqual(report["patch_audit"]["changed_scalars"], 2)
            self.assertEqual(report["synthetic_comparison"]["delta"], 0)

    def test_demo_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--out", temp]), 2)


if __name__ == "__main__":
    unittest.main()
