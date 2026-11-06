"""Exercise the public artifact flow with independently authored synthetic answers."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from weight_space_lab.__main__ import main
from weight_space_lab.protocol import canonical_hash
from test_analysis_pipeline import item, make_manifest, raw


class PhaseFourCliTests(unittest.TestCase):
    def invoke(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = main(arguments)
        return status, json.loads(output.getvalue())

    def test_frozen_schedule_grade_projection_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = make_manifest()
            items = [item(), item("train:000001")]
            split = {"schema_version": 1, "kind": "benchmark_split", "split": "development",
                     "source_manifest_sha256": canonical_hash("synthetic source manifest"),
                     "items": [{"benchmark": x["benchmark"], "item_id": x["item_id"],
                                "dataset_revision": x["dataset_revision"], "item_sha256": canonical_hash(x)}
                               for x in items]}
            payloads = {"split": split, "items": items, "seeds": [7, 11],
                        "policy": expected["policy"], "arms": expected["arms"],
                        "baseline": [raw(expected, output="#### 2")], "candidate": [], "empty-grades": []}
            for name, value in payloads.items():
                (root / (name + ".json")).write_text(json.dumps(value))
            def path(name):
                return str(root / (name + ".json"))
            arguments = ["freeze-analysis", "--benchmark", "gsm8k", "--stage", "smoke",
                         "--evidence-kind", "fixture", "--split-manifest", path("split"),
                         "--items", path("items"), "--seeds", path("seeds"), "--policy", path("policy"),
                         "--arms", path("arms"), "--out", path("manifest")]
            status, manifest = self.invoke(arguments)
            self.assertEqual(status, 0, manifest)
            self.assertEqual(manifest, expected)
            status, grades = self.invoke(["grade-records", "--records", path("baseline"),
                                         "--items", path("items"), "--policy", path("policy"),
                                         "--out", path("grades")])
            self.assertEqual(status, 0, grades)
            self.assertTrue(grades[0]["correct"])
            status, projection = self.invoke(["project-analysis", "--manifest", path("manifest"),
                                               "--records", path("baseline"), "--grades", path("grades"),
                                               "--arm", "baseline", "--out", path("projection")])
            self.assertEqual(status, 0, projection)
            self.assertEqual(len(projection["records"]), 4)
            status, report = self.invoke(["compare-graded", "--manifest", path("manifest"),
                                         "--baseline-records", path("baseline"), "--candidate-records", path("candidate"),
                                         "--baseline-grades", path("grades"), "--candidate-grades", path("empty-grades"),
                                         "--bootstrap-samples", "20", "--out", path("comparison")])
            self.assertEqual(status, 0, report)
            self.assertEqual(report["comparison"]["scheduled_cases"], 4)
            self.assertEqual(report["comparison"]["baseline"]["mean"], 0.25)
            self.assertFalse(report["scientific_result"])
            self.assertEqual(self.invoke(arguments)[0], 2)

    def test_missing_grade_does_not_write_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_manifest()
            for name, value in {"manifest": manifest, "records": [raw(manifest)], "grades": []}.items():
                (root / (name + ".json")).write_text(json.dumps(value))
            status, result = self.invoke(["project-analysis", "--manifest", str(root / "manifest.json"),
                                         "--records", str(root / "records.json"), "--grades", str(root / "grades.json"),
                                         "--arm", "baseline", "--out", str(root / "out.json")])
            self.assertEqual(status, 2)
            self.assertIn("explicit", result["error"])
            self.assertFalse((root / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
