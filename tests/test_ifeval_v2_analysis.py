"""Synthetic analysis checks; never read private final contents or target outputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_ifeval_heldout_analysis import evidence, fixtures
from weight_space_lab.ifeval_heldout_analysis import ANALYSIS_SPEC
from weight_space_lab.ifeval_v2_analysis import _evidence, analyze, compare_selected, freeze_analysis_plan
from weight_space_lab.inventory import file_sha256
from weight_space_lab.protocol import canonical_hash


def plan_for(items, manifest):
    return {"policy": manifest["policy"], "analysis_spec": ANALYSIS_SPEC,
        "split_manifest": {"schema_version": 1, "kind": "benchmark_split", "split": "test",
            "source_manifest_sha256": canonical_hash("synthetic final dataset"),
            "items": [{"benchmark": item["benchmark"], "item_id": item["item_id"],
                "dataset_revision": item["dataset_revision"], "item_sha256": canonical_hash(item)} for item in items]}}


class FinalAnalysisTests(unittest.TestCase):
    def test_identical_final_alias_has_zero_contrasts_without_mutating_raw_records(self):
        items, (strict, _) = fixtures()
        base = evidence(items, strict, "baseline", [[True], [False, True, False]])
        before = deepcopy(base)
        identity = strict["arms"]["baseline"]
        result = compare_selected(items, plan_for(items, strict), (identity, base), (identity, base),
                                  alias="selected_astra_minus_pristine", evidence_kind="fixture")
        self.assertEqual(base, before)
        self.assertIs(result["analytical_alias_only"], True)
        self.assertEqual(result["added_generation_dispatches"], 0)
        for mode in ("strict", "loose"):
            self.assertEqual(result["prompt_comparisons"][mode]["comparison"]["delta"], 0)
            self.assertEqual(result["instruction_comparisons"][mode]["delta"], 0)
            self.assertEqual(result["instruction_comparisons"][mode]["baseline"]["denominator"], 4)
            self.assertEqual(result["instruction_comparisons"][mode]["confidence_interval"]["low"], 0)
            self.assertEqual(result["instruction_comparisons"][mode]["confidence_interval"]["high"], 0)

    def test_distinct_final_arms_reuse_existing_prompt_and_instruction_estimators(self):
        items, (strict, _) = fixtures()
        baseline = evidence(items, strict, "baseline", [[True], [False] * 3])
        candidate = evidence(items, strict, "candidate", [[False], [True] * 3])
        result = compare_selected(items, plan_for(items, strict), (strict["arms"]["baseline"], baseline),
            (strict["arms"]["candidate"], candidate), alias="synthetic_contrast", evidence_kind="fixture")
        self.assertIs(result["analytical_alias_only"], False)
        self.assertEqual(result["prompt_comparisons"]["strict"]["comparison"]["delta"], 0)
        self.assertEqual(result["instruction_comparisons"]["strict"]["delta"], .5)
        self.assertEqual(result["paired_outcomes"]["strict"], {"improved": 1, "regressed": 1, "tied": 0, "unresolved": 0})
        self.assertEqual(result["instruction_comparisons"]["strict"]["confidence_interval"]["resamples"], 10000)
        self.assertEqual(result["instruction_comparisons"]["strict"]["confidence_interval"]["seed"], 0)

    def test_incomplete_final_keeps_denominators_and_unknown_dispatch_accounting(self):
        items, (strict, _) = fixtures()
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}, "_v2_dispatched_item_ids": []}
        unknown = {**deepcopy(absent), "_v2_dispatched_item_ids": [items[0]["item_id"]]}
        result = compare_selected(items, plan_for(items, strict), (strict["arms"]["baseline"], unknown),
            (strict["arms"]["candidate"], absent), alias="incomplete", evidence_kind="fixture")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["scheduled_prompts"], 2)
        self.assertEqual(result["scheduled_instructions"], 4)
        self.assertEqual(result["arm_accounting"]["baseline"]["dispatched_without_observed_output"], 1)
        self.assertEqual(result["arm_accounting"]["candidate"]["unexecuted_slots"], 2)

    def test_absent_entire_final_arms_keep_all_128_prompts_and_160_instructions(self):
        items, (strict, _) = fixtures()
        expanded = []
        for index in range(128):
            item = deepcopy(items[0])
            item["item_id"] = str(index)
            if index >= 96:
                item["target"]["instruction_id_list"] *= 2
                item["target"]["kwargs"] *= 2
            expanded.append(item)
        absent = {"records": [], "grades": [], "summary": {"technically_valid": False}}
        result = compare_selected(expanded, plan_for(expanded, strict), (strict["arms"]["baseline"], absent),
            (strict["arms"]["candidate"], absent), alias="fully_absent", evidence_kind="fixture")
        self.assertEqual(result["scheduled_prompts"], 128)
        self.assertEqual(result["scheduled_instructions"], 160)
        for mode in ("strict", "loose"):
            self.assertEqual(result["instruction_comparisons"][mode]["candidate"]["denominator"], 160)
        self.assertEqual(result["paired_outcomes"]["strict"]["unresolved"], 128)

    def test_observed_ungraded_outputs_and_modified_session_files_fail_closed(self):
        items, (strict, _) = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session = root / "sessions/final-pristine_baseline"
            session.mkdir(parents=True)
            config = {"runId": "synthetic-final"}
            config["configurationSha256"] = canonical_hash(config)
            (session / "baseline-config.json").write_text(json.dumps(config))
            study = {"config": {"grading_policy": strict["policy"]}}
            row = {"dispatch_arm": "pristine_baseline", "artifact_sha256": "a" * 64}
            manifest = {"artifact_sha256": row["artifact_sha256"], "items_sha256": canonical_hash(items),
                "study_manifest_sha256": canonical_hash(study), "policy": strict["policy"],
                "configuration_sha256": config["configurationSha256"], "checkpoint": strict["arms"]["baseline"]["checkpoint"],
                "arm": "baseline", "files": {str(session / "baseline-config.json"): file_sha256(session / "baseline-config.json")}}
            (session / "manifest.json").write_text(json.dumps(manifest))
            (session / "ready.json").write_text(json.dumps({"manifest_sha256": canonical_hash(manifest)}))
            (session / "execution/baseline-1").mkdir(parents=True)
            (session / "execution/baseline-1/supervisor-result.json").write_text(json.dumps({"outputs": [{"item_id": "0"}], "scored_dispatches": 1}))
            with self.assertRaisesRegex(ValueError, "independent grades"):
                _evidence(root, study, row, items)
            (session / "baseline-config.json").write_text(json.dumps(config) + "\n")
            with self.assertRaisesRegex(ValueError, "session file changed"):
                _evidence(root, study, row, items)

    def test_final_plan_is_frozen_before_dispatch_and_aliases_map_to_one_evidence_set(self):
        items, (strict, _) = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = root / "evaluator"
            evaluator.mkdir()
            final_items = []
            for index in range(128):
                item = deepcopy(items[0])
                item["item_id"] = str(index)
                if index >= 96:
                    item["target"]["instruction_id_list"] *= 2
                    item["target"]["kwargs"] *= 2
                final_items.append(item)
            (evaluator / "final-items.json").write_text(json.dumps(final_items))
            dataset = {"sha256": "d" * 64, "counts": {"prompts": 128, "instructions": 160}}
            study = {"evaluator_root": str(evaluator), "config": {"grading_policy": strict["policy"],
                "final_set": {"manifest_sha256": dataset["sha256"]}}}
            named = {name: "a" * 64 for name in ("pristine_baseline", "selected_astra")}
            selection = {"named_arms": named, "distinct_evaluations": []}
            state = {"selection": selection, "final": {}, "slots": {
                f"r{r}-{m}": {"status": "failed"} for r in range(1, 5) for m in ("astra", "control")}}
            (root / "state.json").write_text(json.dumps(state))
            (root / "final-selection.json").write_text(json.dumps(selection))
            with patch("weight_space_lab.ifeval_v2_analysis.verify_frozen", return_value=dataset):
                plan = freeze_analysis_plan(root, study, selection)
                self.assertEqual(freeze_analysis_plan(root, study, selection), plan)
                self.assertEqual(len(plan["split_manifest"]["items"]), 128)
                with patch("weight_space_lab.ifeval_v2_study.load", return_value=(root, study, state)), \
                        patch("weight_space_lab.ifeval_v2_analysis._evidence", return_value=({"run_id": "one"}, {})) as read_evidence, \
                        patch("weight_space_lab.ifeval_v2_analysis.compare_selected", return_value={"synthetic_test_only": True}) as compare:
                    result = analyze(root)
                self.assertEqual(read_evidence.call_count, 0)
                self.assertEqual(compare.call_count, 0)
                self.assertEqual(result["distinct_final_artifacts"], 0)
                self.assertEqual(result["scheduled_prompts_per_named_arm"], 0)
                self.assertEqual(result["frozen_final_counts"], {"prompts": 128, "instructions": 160})
                self.assertEqual(result["status"], "skipped_pristine_retained")
                self.assertFalse(result["final_outcomes_returned_to_editor"])
                state["final"]["pristine_baseline"] = {"status": "started"}
                (root / "state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, "precede"):
                    freeze_analysis_plan(root, study, selection)

    def test_zero_final_items_freezes_zero_and_skips_quality_analysis(self):
        items, (strict, _) = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluator = root / 'evaluator'
            evaluator.mkdir()
            (evaluator / 'final-items.json').write_text('[]')
            dataset = {'sha256': 'd' * 64, 'counts': {'prompts': 0, 'instructions': 0}}
            study = {'evaluator_root': str(evaluator), 'config': {'grading_policy': strict['policy'],
                'final_set': {'manifest_sha256': dataset['sha256']}}}
            selection = {'named_arms': {'pristine_baseline': 'a' * 64, 'selected_astra': 'b' * 64},
                'distinct_evaluations': []}
            state = {'selection': selection, 'final': {}, 'slots': {
                f'r{r}-astra': {'status': 'failed'} for r in range(1, 5)}}
            (root / 'state.json').write_text(json.dumps(state))
            (root / 'final-selection.json').write_text(json.dumps(selection))
            with patch('weight_space_lab.ifeval_v2_analysis.verify_frozen', return_value=dataset), \
                    patch('weight_space_lab.ifeval_v2_study.load', return_value=(root, study, state)):
                plan = freeze_analysis_plan(root, study, selection)
                result = analyze(root)
            self.assertEqual(plan['counts'], {'prompts': 0, 'instructions': 0})
            self.assertEqual(result['status'], 'unavailable_no_eligible_official_final_items')
            self.assertEqual(result['contrasts'], {})


if __name__ == "__main__":
    unittest.main()
