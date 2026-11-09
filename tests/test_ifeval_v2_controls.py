"""Offline synthetic V2 support, selection and provenance regression checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from weight_space_lab import ifeval_v2_candidate as adapter
from weight_space_lab.browser_conversion import decode_q4
from weight_space_lab.ifeval_v2_controls import (
    BLOCK_WIDTH, CONTROL_SEED, GROUP_SIZE, MAPPING, ROWS, TENSORS, WIDTH,
    final_arms, generate_control, select_candidate, support_profile,
    validate_v2_proposal, verify_control,
)
from weight_space_lab.inventory import file_sha256
from weight_space_lab.protocol import canonical_hash


BASE = "b" * 64


def proposal(starts=(0, 3524), factors=(0.5, 1.5)):
    return {"schema_version": 1, "base_sha256": BASE, "provenance": "editor_proposal",
        "composition": "sequential", "operations": [
            {"op": "scale_columns", "tensor": tensor, "columns": list(range(start, start + 60)), "factor": factor}
            for tensor, start, factor in zip(TENSORS, starts, factors)]}


def completed(round_number, identity, passes, **kwargs):
    return {"round": round_number, "artifact_sha256": identity * 64,
        "strict_passes": passes, "completed": True, "technically_valid": True,
        "independently_graded": True, "scheduled_prompts": 32, "graded_prompts": 32, **kwargs}


class ControlTests(unittest.TestCase):
    def test_every_alignment_preserves_actual_occupied_groups_and_counts(self):
        # Exhaust the possible source support, not a convenient alignment sample.
        for start in range(WIDTH - BLOCK_WIDTH + 1):
            other = start + GROUP_SIZE if start + GROUP_SIZE <= WIDTH - BLOCK_WIDTH else start - GROUP_SIZE
            left, right = support_profile(start), support_profile(other)
            for key in ("groups_per_row", "occupied_row_groups", "selected_coordinates",
                        "selected_coordinates_per_group_per_row"):
                self.assertEqual(left[key], right[key], (start, other, key))
            self.assertEqual(left["selected_coordinates"], 61440)
            self.assertEqual(sum(left["selected_coordinates_per_group_per_row"]), 60)
            self.assertEqual(left["occupied_row_groups"], ROWS * left["groups_per_row"])
        self.assertEqual(support_profile(0)["selected_coordinates_per_group_per_row"], [32, 28])
        self.assertEqual(support_profile(5)["selected_coordinates_per_group_per_row"], [27, 32, 1])

    def test_profile_agrees_with_production_q4_decoder(self):
        words = np.full((2, 448), 0x88888888, dtype=np.uint32)
        scales = np.arange(1, 113, dtype=np.float16)[None, :].repeat(2, axis=0)
        decoded = decode_q4(words, scales)
        self.assertEqual(list(decoded.shape), [2, WIDTH])
        for start in (0, 1, 4, 5, 31, 32, 3524):
            observed = decoded[0, start:start + 60]
            _, counts = np.unique(observed, return_counts=True)
            self.assertEqual(counts.tolist(), support_profile(start)["selected_coordinates_per_group_per_row"])

    def test_reproducible_ranking_and_full_private_audit_without_scores(self):
        original = proposal((19, 3475))
        before = deepcopy(original)
        for round_number in range(1, 5):
            control, audit = generate_control(original, round_number)
            self.assertEqual((control, audit), generate_control(original, round_number))
            self.assertEqual(control["provenance"], "engineering_edit")
            self.assertEqual(audit["provenance"], "matched_random_control")
            self.assertIs(audit["uses_candidate_scores"], False)
            self.assertEqual(verify_control(control, audit), audit)
            for op in control["operations"]:
                record = audit["matching"][op["tensor"]]
                expected = min(record["eligible_starts"], key=lambda start: (
                    canonical_hash({"seed": CONTROL_SEED, "round": round_number,
                        "tensor": op["tensor"], "start": start}), start))
                self.assertEqual(op["columns"][0], expected)
                self.assertNotEqual(expected, record["astra"]["start"])
                self.assertEqual(expected % 32, record["astra"]["start"] % 32)
        self.assertEqual(original, before)

    def test_partial_overlap_and_prior_round_overlap_are_not_filtered(self):
        _, audit = generate_control(proposal((64, 96)), 1)
        for tensor, start in zip(TENSORS, (64, 96)):
            choices = audit["matching"][tensor]["eligible_starts"]
            self.assertIn(start - 32, choices)
            self.assertIn(start + 32, choices)
            self.assertNotIn(start, choices)
        # No prior-round candidate/score argument is accepted by the generator.
        with self.assertRaises(TypeError):
            generate_control(proposal(), 1, prior_scores=[32])

    def test_provenance_tampering_fails_closed(self):
        control, audit = generate_control(proposal(), 2)
        changed = deepcopy(control)
        changed["operations"][0]["factor"] = 0.75
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_control(changed, audit)
        changed_audit = deepcopy(audit)
        changed_audit["seed"] += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_control(control, changed_audit)

    def test_narrow_search_rejects_every_unmatched_operation_and_malformed_support(self):
        self.assertEqual(validate_v2_proposal(proposal(), BASE), proposal())
        mutations = [
            lambda p: p.update(composition="independent"),
            lambda p: p.update(provenance="engineering_edit"),
            lambda p: p["operations"].pop(),
            lambda p: p["operations"][0].update(op="remove_direction_from_columns"),
            lambda p: p["operations"][0].update(factor=1.0),
            lambda p: p["operations"][0].update(factor=True),
            lambda p: p["operations"][0].update(factor=float("nan")),
            lambda p: p["operations"][0].update(columns=list(range(59))),
            lambda p: p["operations"][0].update(columns=list(reversed(range(60)))),
            lambda p: p["operations"][0].update(columns=list(range(3525, 3585))),
            lambda p: p["operations"][1].update(tensor=TENSORS[0]),
            lambda p: p["operations"][0].update(extra=True),
        ]
        for mutate in mutations:
            value = proposal()
            mutate(value)
            with self.assertRaises(ValueError):
                validate_v2_proposal(value, BASE)
        for round_number in (0, 5, True):
            with self.assertRaises(ValueError):
                generate_control(proposal(), round_number)


class SelectionTests(unittest.TestCase):
    def test_baseline_ties_and_all_failures_keep_pristine(self):
        for candidates in ([], [completed(1, "a", 16)], [completed(1, "a", 32, completed=False)],
                           [completed(1, "a", 32, graded_prompts=31)],
                           [completed(1, "a", 32, independently_graded=False)]):
            selected = select_candidate(candidates, BASE)
            self.assertEqual(selected["artifact_sha256"], BASE)
            self.assertIsNone(selected["round"])

    def test_earliest_completed_tie_ignores_other_metrics_and_input_order(self):
        candidates = [completed(4, "d", 23), completed(2, "c", 23, sampled_tokens=99),
                      completed(1, "a", 23, technically_valid=False), completed(3, "e", 17)]
        self.assertEqual(select_candidate(candidates, BASE)["round"], 2)
        self.assertEqual(select_candidate(list(reversed(candidates)), BASE)["round"], 2)

    def test_duplicate_identity_reuses_earliest_and_inconsistent_grade_rejected(self):
        candidates = [completed(1, "a", 20), completed(3, "a", 20)]
        self.assertEqual(select_candidate(candidates, BASE)["round"], 1)
        candidates[1]["strict_passes"] = 21
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            select_candidate(candidates, BASE)
        with self.assertRaisesRegex(ValueError, "pristine"):
            select_candidate([completed(1, "b", 20)], BASE)

    def test_final_names_deduplicate_by_artifact_identity_in_fixed_order(self):
        selected = lambda identity: {"artifact_sha256": identity * 64}
        full = final_arms(BASE, selected("a"), selected("c"))
        self.assertEqual([row["dispatch_arm"] for row in full["distinct_evaluations"]],
                         ["pristine_baseline", "selected_astra", "selected_random_control"])
        shared = final_arms(BASE, selected("a"), selected("a"))
        self.assertEqual(shared["distinct_evaluations"][1]["named_arms"],
                         ["selected_astra", "selected_random_control"])
        self.assertEqual(shared["unused_final_slots_expire"], 1)
        fallback = final_arms(BASE, selected("b"), selected("b"))
        self.assertEqual(len(fallback["distinct_evaluations"]), 1)
        self.assertEqual(fallback["unused_final_slots_expire"], 2)


class CandidateAdapterTests(unittest.TestCase):
    def test_package_reuses_quantized_audit_and_keeps_random_control_provenance(self):
        control, control_audit = generate_control(proposal(), 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def cache(path, changed=False):
                path.mkdir()
                raw, records = b"", []
                for operation in control["operations"]:
                    prefix = operation["tensor"].replace("model.language_model.", "model.").removesuffix(".weight")
                    words = np.full((2, 448), 0x88888888, dtype="<u4")
                    scales = np.ones((2, 112), dtype="<f2")
                    if changed:
                        scales[:, operation["columns"][0] // 32] = 2
                    for suffix, array in (("q_weight", words), ("q_scale", scales)):
                        payload = array.tobytes()
                        records.append({"name": f"{prefix}.{suffix}", "shape": list(array.shape),
                            "dtype": array.dtype.name, "format": "raw", "byteOffset": len(raw), "nbytes": len(payload)})
                        raw += payload
                (path / "params.bin").write_bytes(raw)
                (path / "tensor-cache.json").write_text(json.dumps({"metadata": {"ParamSize": 4},
                    "records": [{"dataPath": "params.bin", "nbytes": len(raw),
                                 "md5sum": hashlib.md5(raw).hexdigest(), "records": records}]}))
            baseline, converted = root / "baseline", root / "converted"
            cache(baseline)
            cache(converted, changed=True)
            metadata = {"mlc-chat-config.json": json.dumps({"tokenizer_files": ["tokenizer.json"]}),
                        "tokenizer.json": "{}", "model.wasm": "synthetic", "LICENSE": "synthetic"}
            for name, content in metadata.items():
                (baseline / name).write_text(content)
            files = adapter._files(baseline)
            manifest = {"quantization": "q4f16_1", "webllm_version": "0.2.86", "arms": {"baseline": {
                "path": str(baseline), "files": files, "artifact_sha256": canonical_hash(files),
                "source_checkpoint_sha256": BASE}}}
            manifest_path = root / "baseline.json"
            manifest_path.write_text(json.dumps(manifest))
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps({"allowed_tensors": list(TENSORS)}))
            identity = {"source_checkpoint_sha256": "c" * 64, "v2_provenance": "matched_random_control",
                        "control_audit_sha256": canonical_hash(control_audit)}
            binding = {**identity, "status": "converted_source_bound", "baseline_manifest_sha256": file_sha256(manifest_path),
                "v2_adapter_sha256": file_sha256(adapter.__file__), "converted_files": adapter._files(converted),
                "converted_files_sha256": canonical_hash(adapter._files(converted))}
            (root / "converted-source-binding.json").write_text(json.dumps(binding))
            with patch.object(adapter, "source_identity", return_value=(identity, {}, control)):
                result = adapter.package_candidate("unused", "unused", manifest_path, converted, root / "package",
                    policy_path=policy_path, expected_provenance="engineering_edit", control_audit=control_audit)
            artifact = json.loads(Path(result["artifact_manifest"]).read_text())
            self.assertEqual(artifact["provenance"], "matched_random_control")
            self.assertFalse(result["conversion_erases_edit"])
            self.assertEqual(adapter._files(baseline), files)
            for name in metadata:
                self.assertEqual((Path(result["artifact_path"]) / name).read_bytes(), (baseline / name).read_bytes())
            self.assertEqual(json.loads((root / "package/matched-control-audit.json").read_text()), control_audit)

    def test_control_identity_requires_reproducible_truthful_provenance(self):
        control, control_audit = generate_control(proposal(), 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "proposal.json").write_text(json.dumps(control))
            (root / "policy.json").write_text("{}")
            manifest = {"arms": {"baseline": {"source_checkpoint_sha256": BASE}}}
            with patch.object(adapter, "verify_candidate", return_value={"base_sha256": BASE, "result_sha256": "c" * 64}):
                identity, _, _ = adapter.source_identity("unused", root, manifest,
                    policy_path=root / "policy.json", expected_provenance="engineering_edit", control_audit=control_audit)
                self.assertEqual(identity["v2_provenance"], "matched_random_control")
                self.assertEqual(identity["control_audit_sha256"], canonical_hash(control_audit))
                with self.assertRaises(ValueError):
                    adapter.source_identity("unused", root, manifest, policy_path=root / "policy.json")
                with self.assertRaises(ValueError):
                    adapter.source_identity("unused", root, manifest,
                        policy_path=root / "policy.json", expected_provenance="engineering_edit")

    def test_failed_precheck_is_durable_and_cannot_retry_or_start_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "baseline.json"
            manifest_path.write_text("{}")
            script = Path(adapter.__file__).absolute().parent.parent / "scripts/browser_convert.py"
            manifest = {"converter_adapter_sha256": file_sha256(script), "converter_packages": {"test-package": "1"}}
            callback = unittest.mock.Mock()
            with patch.object(adapter, "verify_baseline", return_value=manifest), \
                    patch.object(adapter, "source_identity", return_value=({}, {}, {})), \
                    patch.object(adapter, "run_converter_python", return_value=unittest.mock.Mock(stdout='{"packages":{"test-package":"2"}}')) as runner:
                with self.assertRaisesRegex(ValueError, "packages"):
                    adapter.convert("unused", "unused", manifest_path, root / "converted", "unused", on_conversion=callback)
                callback.assert_not_called()
                self.assertEqual(runner.call_count, 1)
                failure = json.loads((root / "converted-conversion-failure.json").read_text())
                self.assertFalse(failure["conversion_subprocess_started"])
                with self.assertRaisesRegex(FileExistsError, "zero retries"):
                    adapter.convert("unused", "unused", manifest_path, root / "converted", "unused")
                self.assertEqual(runner.call_count, 1)

    def test_conversion_callback_closes_precheck_before_single_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "baseline.json"
            manifest_path.write_text("{}")
            script = Path(adapter.__file__).absolute().parent.parent / "scripts/browser_convert.py"
            manifest = {"converter_adapter_sha256": file_sha256(script), "converter_packages": {"test-package": "1"}}
            identity = {"source_checkpoint_sha256": "c" * 64}
            calls = []
            def run(*args, **kwargs):
                calls.append("precheck" if len(calls) == 0 else "conversion")
                return unittest.mock.Mock(stdout=json.dumps({"packages": {"test-package": "1"},
                    "mapping_source_sha256": MAPPING["mapping_source_sha256"]}), returncode=0)
            def stage():
                calls.append("stage_transition")
                return 12
            with patch.object(adapter, "verify_baseline", return_value=manifest), \
                    patch.object(adapter, "source_identity", return_value=(identity, {}, {})), \
                    patch.object(adapter, "run_converter_python", side_effect=run) as runner, \
                    patch.object(adapter, "converter_python_command", return_value=["python", "synthetic-conversion"]), \
                    patch.object(adapter, "_files", return_value={}), patch.object(adapter, "tensor_index", return_value={}):
                result = adapter.convert("unused", "unused", manifest_path, root / "converted", "unused", on_conversion=stage)
                self.assertEqual(calls, ["precheck", "stage_transition", "conversion"])
                self.assertEqual(runner.call_args.kwargs["timeout"], 12)
                self.assertEqual(result["source_checkpoint_sha256"], identity["source_checkpoint_sha256"])

    def test_unchanged_package_version_with_modified_mapping_is_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "baseline.json"
            manifest_path.write_text("{}")
            script = Path(adapter.__file__).absolute().parent.parent / "scripts/browser_convert.py"
            manifest = {"converter_adapter_sha256": file_sha256(script), "converter_packages": {"test-package": "1"}}
            changed_mapping = {**MAPPING["mapping_source_sha256"], "qwen35_loader.py": "0" * 64}
            callback = unittest.mock.Mock()
            with patch.object(adapter, "verify_baseline", return_value=manifest), \
                    patch.object(adapter, "source_identity", return_value=({}, {}, {})), \
                    patch.object(adapter, "run_converter_python", return_value=unittest.mock.Mock(stdout=json.dumps({
                        "packages": {"test-package": "1"}, "mapping_source_sha256": changed_mapping}))) as runner:
                with self.assertRaisesRegex(ValueError, "quantization mapping"):
                    adapter.convert("unused", "unused", manifest_path, root / "converted", "unused", on_conversion=callback)
                callback.assert_not_called()
                self.assertEqual(runner.call_count, 1)
                code = runner.call_args.args[1][1]
                self.assertIn("find_spec('mlc_llm')", code)
                self.assertIn("mapping_source_sha256", code)
                self.assertIn("group_quantization.py", code)


if __name__ == "__main__":
    unittest.main()
