"""Synthetic source-bound continuation joins; no inference or grader calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from weight_space_lab.analysis_pipeline import build_analysis_manifest
from weight_space_lab.grading import grader_identity, ifeval_metrics
from weight_space_lab.ifeval_continuation_merge import prepare_merge_plan, merge_evidence
from weight_space_lab.ifeval_heldout_analysis import ANALYSIS_SPEC
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.records import run_record_sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class ContinuationMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.tail = self.root / "original", self.root / "continuation"
        self.items = []
        for index in range(128):
            count = 2 if index < 34 or 94 <= index < 105 else 1
            self.items.append({"benchmark": "ifeval", "item_id": f"fixture-{index:03}",
                "dataset_revision": "a" * 40, "split": "test", "task_text": "Synthetic task " + str(index),
                "target": {"instruction_id_list": ["keywords:existence"] * count,
                           "kwargs": [{"keywords": ["hello"]}] * count},
                "source": {"dataset_id": "synthetic-known-answers", "revision": "a" * 40,
                    "split": "official", "row_id": str(index)}, "metadata": {}})
        self.policy = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
            "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
            "grader_timeout_seconds": 20, "grader_retry": {"mode": "none", "max_attempts": 1}}
        self.manifests, self.configs = {}, {}
        for arm in ("baseline", "candidate"):
            config = {"runId": "synthetic-" + arm, "configurationSha256": canonical_hash(arm),
                "items": [{"item_id": i["item_id"], "prompt": i["task_text"]} for i in self.items],
                "settings": {"thinking": False, "temperature": 0}, "warmupPrompt": "Synthetic warmup",
                "sourceCheckpointSha256": canonical_hash(arm + " checkpoint"),
                "convertedArtifactSha256": canonical_hash(arm + " artifact")}
            manifest = {"checkpoint": {"model_id": "synthetic-array", "revision": "b" * 40,
                "sha256": config["sourceCheckpointSha256"]},
                "configuration_sha256": config["configurationSha256"], "settings": config["settings"],
                "artifact_sha256": config["convertedArtifactSha256"], "policy": self.policy,
                "grader": grader_identity("ifeval")}
            self.manifests[arm], self.configs[arm] = manifest, config
            save(self.old / arm / "manifest.json", manifest)
            save(self.old / arm / "baseline-config.json", config)
        split = {"schema_version": 1, "kind": "benchmark_split", "split": "test",
            "source_manifest_sha256": canonical_hash("synthetic source"),
            "items": [{"benchmark": "ifeval", "item_id": i["item_id"],
                "dataset_revision": i["dataset_revision"], "item_sha256": canonical_hash(i)} for i in self.items]}
        arms = {arm: {"run_id": self.configs[arm]["runId"], "profile": "browser_evaluation",
            "configuration_sha256": self.configs[arm]["configurationSha256"],
            "checkpoint": self.manifests[arm]["checkpoint"]} for arm in ("baseline", "candidate")}
        for mode in ("strict", "loose"):
            paired = build_analysis_manifest(split, self.items, benchmark="ifeval", seeds=[42],
                policy=self.policy, arms=arms, stage="full", evidence_kind="fixture",
                metric="prompt_level_" + mode + "_accuracy")
            save(self.old / ("paired-analysis-" + mode + ".json"), paired)
        save(self.old / "evaluation-items.json", self.items)
        save(self.old / "evaluation-freeze.json", {"analysis_specification": ANALYSIS_SPEC})
        save(self.old / "evaluation-ready.json", {"fixture": True})
        save(self.old / "analysis.json", {"status": "incomplete"})
        save(self.old / "run-result.json", {"status": "stopped"})
        for arm in ("baseline", "candidate"):
            evidence = self.evidence(arm, self.items, completed=128 if arm == "baseline" else 94)
            self.save_evidence(self.old / arm, arm, evidence)
            save(self.old / arm / "execution" / (arm + "-1") / "supervisor-result.json", {"fixture": True})
        manifest = copy.deepcopy(self.manifests["candidate"])
        config = copy.deepcopy(self.configs["candidate"])
        config.update(runId="ifeval-browser-v1-heldout-continuation-candidate",
                      configurationSha256=canonical_hash("new continuation config"), items=config["items"][94:])
        manifest.update(configuration_sha256=config["configurationSha256"], original_root=str(self.old),
            original_freeze_sha256=canonical_hash(read_json(self.old / "evaluation-freeze.json")),
            original_candidate_manifest_sha256=canonical_hash(self.manifests["candidate"]),
            original_candidate_config_sha256=file_sha256(self.old / "candidate/baseline-config.json"),
            items_path=str(self.tail / "remaining-items.json"))
        save(self.tail / "manifest.json", manifest)
        save(self.tail / "baseline-config.json", config)
        save(self.tail / "remaining-items.json", self.items[94:])

    def evidence(self, arm, items, *, completed, continuation=False):
        if continuation:
            manifest, config = read_json(self.tail / "manifest.json"), read_json(self.tail / "baseline-config.json")
        else:
            manifest, config = self.manifests[arm], self.configs[arm]
        records, grades = [], []
        for index, item in enumerate(items):
            good = index < completed
            outcome = {"status": "ok" if good else "missing", "raw_output": "hello" if good else None,
                "finish_reason": "stop" if good else "not_started",
                "token_counts": {"input": 1, "output": 1, "total": 2} if good else dict.fromkeys(("input", "output", "total")),
                "duration_seconds": 0.1 if good else None, "error": None}
            record = {"schema_version": 1, "record_type": "run", "run_id": config["runId"],
                "profile": "browser_evaluation", "configuration_sha256": config["configurationSha256"],
                "checkpoint": manifest["checkpoint"], "evidence_kind": "fixture", "stage": "full",
                "benchmark": "ifeval", "split": "test", "dataset_revision": item["dataset_revision"],
                "item_id": item["item_id"], "sample_index": 0, "seed": 42,
                "retry_policy": self.policy["generation_retry"],
                "attempts": [{"attempt_index": 0, **copy.deepcopy(outcome)}] if good else [], **outcome}
            flags = [True] * len(item["target"]["instruction_id_list"])
            grade = {"schema_version": 1, "record_type": "grade", "benchmark": "ifeval",
                "item_id": item["item_id"], "seed": 42, "sample_index": 0, "evidence_kind": "fixture",
                "run_record_sha256": run_record_sha256(record), "item_sha256": canonical_hash(item),
                "policy_sha256": canonical_hash(self.policy), "grader": grader_identity("ifeval"),
                "status": "graded" if good else "not_applicable", "correct": True if good else None,
                "metrics": ifeval_metrics(flags, flags) if good else {},
                "diagnostics": {"code": "synthetic_grade", "extracted_answer": None,
                    "output_sha256": canonical_hash("hello") if good else None, "reasoning_sha256": None}}
            records.append(record)
            grades.append(grade)
        return {"manifest_sha256": canonical_hash(manifest), "records": records, "grades": grades,
            "summary": {"technically_valid": completed == len(items)}}

    def save_evidence(self, root, arm, evidence):
        save(root / arm / "graded.json", evidence)
        for key in ("records", "grades"):
            save(root / arm / (key + ".json"), evidence[key])

    def prepare(self):
        return prepare_merge_plan(self.old, self.tail / "manifest.json", self.tail / "merge-plan.json")

    def test_preparation_creates_only_plan_and_no_execution_capacity(self):
        before = {str(p) for p in self.root.rglob("*") if p.is_file()}
        plan = self.prepare()
        after = {str(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after - before, {str(self.tail / "merge-plan.json")})
        self.assertFalse(plan["inference_authorized"])
        self.assertEqual(len(plan["original_candidate_record_grade_references"]), 94)
        self.assertEqual(plan["tail_instructions"], 45)

    def test_join_preserves_original_records_and_true_tail_identity(self):
        plan = self.prepare()
        tail = self.evidence("candidate", self.items[94:], completed=34, continuation=True)
        self.save_evidence(self.tail, "candidate", tail)
        before = {p: file_sha256(p) for p in plan["preserved_file_sha256"]}
        result = merge_evidence(self.old, self.tail, plan)
        self.assertEqual(result["status"], "complete_manually_continued")
        self.assertEqual(result["original_records_rewritten"], 0)
        self.assertEqual(result["original_outputs_regraded"], 0)
        self.assertEqual(len(result["candidate_source_record_grade_references"]), 128)
        self.assertNotEqual(result["candidate_source_runs"][0]["run_id"], result["candidate_source_runs"][1]["run_id"])
        self.assertEqual(result["instruction_comparisons"]["strict"]["candidate"]["numerator"], 173)
        self.assertEqual(result["prompt_comparisons"]["strict"]["comparison"]["delta"], 0)
        self.assertEqual(before, {p: file_sha256(p) for p in before})

    def test_changed_historical_grade_file_is_rejected(self):
        plan = self.prepare()
        tail = self.evidence("candidate", self.items[94:], completed=34, continuation=True)
        self.save_evidence(self.tail, "candidate", tail)
        path = self.old / "candidate/candidate/grades.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "preserved source"):
            merge_evidence(self.old, self.tail, plan)

    def test_tail_cannot_replay_an_already_completed_item(self):
        plan = self.prepare()
        wrong = [self.items[93]] + self.items[95:]
        self.save_evidence(self.tail, "candidate", self.evidence("candidate", wrong, completed=34, continuation=True))
        with self.assertRaisesRegex(ValueError, "exactly the 34"):
            merge_evidence(self.old, self.tail, plan)

    def test_falsified_tail_run_identity_is_rejected(self):
        plan = self.prepare()
        tail = self.evidence("candidate", self.items[94:], completed=34, continuation=True)
        tail["records"][0]["run_id"] = self.configs["candidate"]["runId"]
        self.save_evidence(self.tail, "candidate", tail)
        with self.assertRaisesRegex(ValueError, "run_id"):
            merge_evidence(self.old, self.tail, plan)

    def test_partial_tail_retains_full_denominators_and_incomplete_status(self):
        plan = self.prepare()
        tail = self.evidence("candidate", self.items[94:], completed=2, continuation=True)
        self.save_evidence(self.tail, "candidate", tail)
        result = merge_evidence(self.old, self.tail, plan)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["tail_unexecuted_slots"], 32)
        self.assertEqual(result["paired_outcomes"]["strict"], {"improved": 0, "regressed": 0, "tied": 96, "unresolved": 32})
        self.assertEqual(result["instruction_comparisons"]["strict"]["candidate"]["denominator"], 173)
        self.assertEqual(result["prompt_comparisons"]["strict"]["comparison"]["scheduled_cases"], 128)

    def test_changed_tail_order_or_decoding_is_rejected_before_plan(self):
        config = read_json(self.tail / "baseline-config.json")
        config["items"].reverse()
        save(self.tail / "baseline-config.json", config)
        with self.assertRaisesRegex(ValueError, "unchanged 34-item tail"):
            self.prepare()

    def test_changed_original_item_is_rejected_before_plan(self):
        items = read_json(self.old / "evaluation-items.json")
        items[0]["task_text"] = "Different synthetic text"
        save(self.old / "evaluation-items.json", items)
        with self.assertRaisesRegex(ValueError, "original 128-prompt"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
