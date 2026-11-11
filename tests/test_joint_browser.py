"""Same-generation identity, resource attribution, failures and one-pass contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weight_space_lab.joint_browser import (install_adapter, interval_metric, resource_windows,
    summarize, pair, verify_report, WORKLOAD)
from weight_space_lab.inventory import file_sha256
from weight_space_lab.protocol import canonical_hash


class JointBrowserTests(unittest.TestCase):
    def fixture(self):
        items = [{"item_id": "a", "benchmark": "ifeval"}, {"item_id": "b", "benchmark": "ifeval"}]
        manifest = {"configuration_sha256": "c" * 64, "artifact_sha256": "a" * 64,
            "study_manifest_sha256": "d" * 64, "checkpoint": {"sha256": "s" * 64},
            "items_sha256": canonical_hash(items), "settings": {"temperature": 0},
            "environment": {"browser": "fixture"}, "model_lib_sha256": "f" * 64,
            "harness_sha256": "h" * 64}
        report = {"configuration_sha256": manifest["configuration_sha256"], "artifact_sha256": manifest["artifact_sha256"],
            "modelLib": "local-library", "runtime": "fixture", "quantization": "fixture",
            "environment": {"userAgent": "fixture"}, "loadMs": 50,
            "samples": [
                {"item_id": "__warmup__", "request_id": "generation-1", "warmup": True, "outputTokens": 12},
                {"item_id": "a", "request_id": "generation-2", "warmup": False, "startedMs": 1000, "endedMs": 3000,
                 "responseMs": 2000, "outputTokens": 10, "output": "correct"},
                {"item_id": "b", "request_id": "generation-3", "warmup": False, "startedMs": 4000, "endedMs": 6000,
                 "responseMs": 2000, "outputTokens": 30, "output": "incorrect"}],
            "profile": {"points": [{"tMs": 2000, "gpuBufferBytes": 200, "wasmBytes": 40},
                {"tMs": 3500, "gpuBufferBytes": 999, "wasmBytes": 99}]},
            "nativeTrace": {"logicalCpus": 2, "points": [
                {"tMs": t, "cpuOneCorePercent": 40, "incompleteCpuIntervals": 0, "footprintBytes": t * 1000}
                for t in (500, 1500, 2500, 3500, 4500, 5500, 6500)],
                "gpu": [{"startMs": 1000, "tMs": 2000, "intervalMs": 1000, "gpuTimeMs": 500, "issues": []}],
                "power": []}}
        graded = {"manifest_sha256": canonical_hash(manifest), "records": [
            {"item_id": "a", "status": "ok", "duration_seconds": 2, "token_counts": {"output": 10}, "raw_output": "correct"},
            {"item_id": "b", "status": "truncated", "duration_seconds": 2, "token_counts": {"output": 30}, "raw_output": "incorrect"}],
            "grades": [{"item_id": i, "status": "graded", "metrics": {
                "prompt_level_strict_accuracy": i == "a", "prompt_level_loose_accuracy": i == "a"}} for i in ("a", "b")]}
        return items, graded, report, {"status": "completed"}, manifest

    def result(self):
        items, grades, report, execution, manifest = self.fixture()
        return summarize(items, grades, report, execution=execution, manifest=manifest)

    def test_quality_and_resource_cost_share_requests_and_include_wrong_outputs(self):
        result = self.result()
        self.assertEqual(result["workload"], WORKLOAD)
        self.assertEqual(result["quality"]["strict_accuracy"], .5)
        self.assertEqual(result["quality"]["loose_accuracy"], .5)
        self.assertEqual(result["totals"]["output_tokens"], 40)
        self.assertEqual(result["cost_per_compliant_answer"]["tokens_per_strict_pass"], 40)
        self.assertEqual(result["cost_per_compliant_answer"]["response_ms_per_strict_pass"], 4000)
        self.assertEqual(result["warmup"]["output_tokens"], 12)
        self.assertEqual(result["totals"]["full_response_tokens_per_second"], 10)
        self.assertNotIn("raw_output", json.dumps(result))
        self.assertNotIn('"incorrect"', json.dumps(result))

    def test_request_windows_exclude_gaps_and_boundary_straddling_intervals(self):
        result = self.result()
        a = result["items"][0]["resources"]
        self.assertEqual(a["cpu_capacity_percent"], 20)
        self.assertEqual(a["cpu_coverage"], .5)  # 1500..2500 only, not 500..1500 or 2500..3500.
        self.assertEqual(a["observed_cpu_time_ms"], 400)
        self.assertIsNone(a["cpu_time_ms"])
        self.assertEqual(a["gpu_time_ms_per_second"], 500)
        self.assertEqual(a["gpu_coverage"], .5)
        self.assertEqual(a["observed_gpu_time_ms"], 500)
        self.assertIsNone(a["gpu_time_ms"])
        self.assertEqual(result["resources"]["gpu_buffer_descriptor_bytes_sampled_max"], 200)
        self.assertEqual(result["items"][1]["resources"]["gpu_buffer_descriptor_bytes_sampled_max"], None)
        self.assertIsNone(a["host_power_watts"])
        self.assertIsNone(a["host_energy_joules"])
        self.assertIsNone(result["cost_per_compliant_answer"]["host_joules_per_strict_pass"])

    def test_short_request_cannot_inherit_session_telemetry(self):
        _, _, report, _, _ = self.fixture()
        short = resource_windows(report, [(1600, 1700)])
        self.assertIsNone(short["cpu_capacity_percent"])
        self.assertIsNone(short["native_peak_footprint_gb"])
        self.assertEqual(short["cpu_coverage"], 0)

    def test_overlapping_intervals_are_not_double_counted(self):
        result = interval_metric([(1, 4, 10), (3, 5, 20)], [(0, 6)])
        self.assertEqual(result["covered_ms"], 4)
        self.assertEqual(result["mean"], 12.5)

    def test_partial_power_does_not_become_full_energy_cost(self):
        _, _, report, _, _ = self.fixture()
        report["nativeTrace"]["power"] = [{"tMs": 2000, "intervalMs": 1000, "combinedWatts": 5}]
        result = resource_windows(report, [(1000, 3000)])
        self.assertEqual(result["observed_host_energy_joules"], 5)
        self.assertEqual(result["power_coverage"], .5)
        self.assertIsNone(result["host_energy_joules"])
        report["nativeTrace"]["power"].append({"tMs": 3000, "intervalMs": 1000, "combinedWatts": 7})
        self.assertEqual(resource_windows(report, [(1000, 3000)])["host_energy_joules"], 12)

    def test_full_cpu_gpu_integrals_retain_physical_units(self):
        report = {"nativeTrace": {"logicalCpus": 8, "points": [
            {"tMs": 1000, "cpuOneCorePercent": 0, "incompleteCpuIntervals": 0},
            {"tMs": 3000, "cpuOneCorePercent": 200, "incompleteCpuIntervals": 0}],
            "gpu": [{"startMs": 1000, "tMs": 3000, "intervalMs": 2000, "gpuTimeMs": 1500, "issues": []}]}}
        result = resource_windows(report, [(1000, 3000)])
        self.assertEqual(result["cpu_capacity_percent"], 25)
        self.assertEqual(result["cpu_time_ms"], 4000)  # Two CPU cores busy for two seconds.
        self.assertEqual(result["gpu_time_ms"], 1500)

    def test_missing_request_stays_in_quality_denominator_and_cost_is_unknown(self):
        items, grades, report, execution, manifest = self.fixture()
        report["samples"].pop()
        grades["records"][1].update(status="missing", duration_seconds=None, token_counts={"output": None}, raw_output=None)
        grades["grades"][1].update(status="not_graded", metrics={})
        execution["status"] = "stopped"
        result = summarize(items, grades, report, execution=execution, manifest=manifest)
        self.assertEqual(result["quality"]["scheduled"], 2)
        self.assertEqual(result["quality"]["strict_accuracy"], .5)
        self.assertEqual(result["totals"]["known_output_tokens"], 10)
        self.assertIsNone(result["totals"]["output_tokens"])
        self.assertIsNone(result["cost_per_compliant_answer"]["tokens_per_strict_pass"])
        self.assertEqual(result["status"], "incomplete")

    def test_unexecuted_work_cannot_be_labeled_complete_energy(self):
        items, grades, report, execution, manifest = self.fixture()
        report["samples"].pop()
        report["nativeTrace"]["power"] = [{"tMs": 3000, "intervalMs": 2000, "combinedWatts": 5}]
        grades["records"][1].update(status="missing", duration_seconds=None, token_counts={"output": None}, raw_output=None)
        grades["grades"][1].update(status="not_graded", metrics={})
        execution["status"] = "stopped"
        result = summarize(items, grades, report, execution=execution, manifest=manifest)
        self.assertEqual(result["resources"]["observed_host_energy_joules"], 10)
        self.assertIsNone(result["resources"]["host_energy_joules"])

    def test_zero_success_or_unresolved_grader_cannot_produce_cost_per_success(self):
        for unresolved in (False, True):
            items, grades, report, execution, manifest = self.fixture()
            for g in grades["grades"]:
                g["metrics"]["prompt_level_strict_accuracy"] = False
            if unresolved:
                grades["grades"][0]["status"] = "grader_timeout"
            result = summarize(items, grades, report, execution=execution, manifest=manifest)
            self.assertIsNone(result["cost_per_compliant_answer"]["response_ms_per_strict_pass"])

    def test_substitution_duplication_and_warmup_identity_rejected(self):
        changes = [lambda r: r["samples"].append(copy.deepcopy(r["samples"][1])),
            lambda r: r["samples"][1].update(output="different"),
            lambda r: r["samples"][1].update(outputTokens=11),
            lambda r: r["samples"][1].update(request_id="generation-3"),
            lambda r: r["samples"][0].update(item_id="a"),
            lambda r: r.update(artifact_sha256="wrong"),
            lambda r: r["samples"][1].update(endedMs=6000, responseMs=5000),
            lambda r: r["samples"][1].update(outputTokens=float("nan"))]
        for mutate in changes:
            items, grades, report, execution, manifest = self.fixture()
            mutate(report)
            with self.assertRaises(ValueError):
                summarize(items, grades, report, execution=execution, manifest=manifest)

    def test_pair_only_reports_deltas_for_complete_matched_sessions(self):
        a = self.result(); b = copy.deepcopy(a); b["artifact_sha256"] = "b" * 64
        b["totals"]["output_tokens"] = 30
        result = pair(a, b)
        self.assertEqual(result["output_tokens_delta"], -10)
        self.assertEqual(result["paired_outcomes"], {"improved": 0, "regressed": 0, "tied": 2})
        b["environment"] = {"browser": "another"}
        self.assertIsNone(pair(a, b)["output_tokens_delta"])
        b = copy.deepcopy(a); b["artifact_sha256"] = "b" * 64; b["harness_sha256"] = "changed"
        self.assertIsNone(pair(a, b)["output_tokens_delta"])
        b = copy.deepcopy(a); b["artifact_sha256"] = "b" * 64; b["status"] = "incomplete"
        self.assertIsNone(pair(a, b)["paired_outcomes"])
        with self.assertRaises(ValueError):
            pair(a, a)

    def test_editor_report_rejects_changed_saved_outputs_without_regrading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            items, graded, report, execution, manifest = self.fixture()
            manifest.update(arm="baseline", workload_identity=WORKLOAD, items_path=str(root / "items.json"))
            graded["manifest_sha256"] = canonical_hash(manifest)
            result = summarize(items, graded, report, execution=execution, manifest=manifest)
            values = {"items.json": items, "baseline/graded.json": graded,
                "execution/baseline-1/joint-browser-profile.json": report,
                "execution/baseline-1/supervisor-result.json": execution, "joint-report.json": result}
            for name, value in values.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            evidence = {name: file_sha256(root / name) for name in values if name not in ("items.json", "joint-report.json")}
            result["evidence_sha256"] = canonical_hash(evidence)
            (root / "joint-report.json").write_text(json.dumps(result))
            (root / "joint-report-binding.json").write_text(json.dumps({"manifest_sha256": canonical_hash(manifest),
                "result_sha256": canonical_hash(result), "files": evidence}))
            with patch("weight_space_lab.ifeval_v2_runtime.load", return_value=(root, manifest)):
                self.assertEqual(verify_report(root), result)
                graded["records"][0]["raw_output"] = "substituted"
                (root / "baseline/graded.json").write_text(json.dumps(graded))
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    verify_report(root)

    def install_fixture(self, root):
        sources = {
            "src/benchmark/task-evaluation/protocol.ts": '''    resourceComparison: z.literal(true).optional(),
    const continuation = c.evaluationStage === "heldout-continuation";
c.items.length !== (continuation ? 34 : c.evaluationStage === "heldout" ? (c.narrowedFinalPromptCount ?? 128) : 32)
''',
            "src/benchmark/task-evaluation/client.ts": '''import { resourceReport } from "../resource-comparison/report";
  const standard = task.resourceComparison ? resourceReport(task) : undefined;
  const startedEpochMs = standard?.report.profile!.startedEpochMs ?? clockMs();
    standard?.observe(type, data);
        standard_report: standard?.finish(done, forced),
''',
            "scripts/run-task-evaluation.mjs": '''  if (config.resourceComparison) {
    const capture = nativeCapture
"--max-seconds", "1800"], "native-capture");
: !config.resourceComparison,
(!config.resourceComparison || result.cleanup.native_capture_exit_confirmed)
'''}
        for name, source in sources.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
        return sources

    def test_installer_idempotence_and_frozen_source_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.install_fixture(root)
            first = install_adapter(root)
            self.assertEqual(first, install_adapter(root))
            self.assertEqual(first["target_calls"], 0)
            (root / "scripts/run-task-evaluation.mjs").write_text("changed")
            with self.assertRaises(ValueError):
                install_adapter(root)

    def test_installer_checks_all_anchors_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sources = self.install_fixture(root)
            (root / "scripts/run-task-evaluation.mjs").write_text("unexpected")
            with self.assertRaises(ValueError):
                install_adapter(root)
            self.assertEqual((root / "src/benchmark/task-evaluation/protocol.ts").read_text(), sources["src/benchmark/task-evaluation/protocol.ts"])


if __name__ == "__main__":
    unittest.main()
