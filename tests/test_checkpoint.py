"""Saved-byte acceptance tests for the bounded candidate checkpoint adapter."""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

import numpy as np

from test_inventory import write_shard
from weight_space_lab.checkpoint import CheckpointEditor, verify_candidate
from weight_space_lab.inventory import inventory_model


EDITABLE = ["model.language_model.layers.0.mlp.down_proj.weight", "half.weight",
            "single.weight", "double.weight"]


def fixture_policy():
    return {
        "schema_version": 1,
        "allowed_tensors": EDITABLE.copy(),
        "allowed_operations": ["scale_columns", "remove_direction_from_columns"],
        "allowed_columns": {name: "all" for name in EDITABLE},
        "max_changed_fraction": 1.0,
        "max_changed_coordinates": 16,
        "max_targeted_coordinates": 16,
        "max_operations": 8,
        "max_working_elements": 100,
        "scale_factor_range": [-2.0, 2.0],
        "direction_strength_range": [0.0, 1.0],
        "max_abs_change": 10.0,
        "inspection": {
            "allowed_tensors": EDITABLE + ["integer_buffer", "boolean_buffer", "vector", "empty"],
            "max_requests": 30,
            "max_values": 50,
            "max_scanned_coordinates": 64,
            "max_total_values": 512,
            "max_total_scanned_coordinates": 512,
        },
    }


def checkpoint_fixture(root):
    model = root / "original"
    model.mkdir()
    for name, value in (
        ("config.json", {"architectures": ["FixtureArchitecture"], "model_type": "fixture"}),
        ("tokenizer_config.json", {"chat_template": "{{ messages }}"}),
        ("tokenizer.json", {"fixture": "tokenizer"}),
        ("generation_config.json", {"fixture": "unchanged"}),
    ):
        (model / name).write_text(json.dumps(value))
    (model / "LICENSE").write_text("Synthetic fixture only.\n")
    (model / "nested").mkdir()
    (model / "nested" / "unrelated.txt").write_text("Keep exact bytes.\n")
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    values = [[1.0, 2.0], [3.0, 4.0]]
    left = [
        (EDITABLE[0], "BF16", [2, 2], np.array([0x3f80, 0x4000, 0x4040, 0x4080], dtype="<u2").tobytes()),
        (EDITABLE[1], "F16", [2, 2], np.array(values, dtype="<f2").tobytes()),
        ("integer_buffer", "I64", [2], np.array([2**60 + 1, -3], dtype="<i8").tobytes()),
    ]
    right = [
        (EDITABLE[2], "F32", [2, 2], np.array(values, dtype="<f4").tobytes()),
        (EDITABLE[3], "F64", [2, 2], np.array(values, dtype="<f8").tobytes()),
        ("boolean_buffer", "BOOL", [2], bytes([0, 1])),
        ("vector", "F32", [2], np.array([10.0, 20.0], dtype="<f4").tobytes()),
        ("empty", "F16", [0], b""),
    ]
    write_shard(model / first, left)
    write_shard(model / second, right)
    index = {"metadata": {"total_size": sum(len(t[3]) for t in left + right)},
             "weight_map": {**{t[0]: first for t in left}, **{t[0]: second for t in right}}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    return model


def files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def tensor_bytes(model, name):
    info = inventory_model(model)["tensors"][name]
    start, end = info["data_offsets"]
    with (model / info["shard"]).open("rb") as stream:
        stream.seek(info["data_start"] + start)
        return stream.read(end - start)


def replace_tensor_bytes(model, name, data):
    info = inventory_model(model)["tensors"][name]
    start, end = info["data_offsets"]
    assert end - start == len(data)
    path = model / info["shard"]
    path.chmod(0o644)
    with path.open("r+b") as stream:
        stream.seek(info["data_start"] + start)
        stream.write(data)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = checkpoint_fixture(self.root)
        self.policy = fixture_policy()
        self.log = self.root / "exposure.jsonl"

    def editor(self, policy=None, log=None):
        return CheckpointEditor(self.model, self.policy if policy is None else policy,
                                exposure_log=self.log if log is None else log)

    def proposal(self, editor, operations=None, **kwargs):
        return {"schema_version": 1, "base_sha256": editor.original_sha256,
                "operations": [] if operations is None else operations,
                "provenance": "engineering_edit", "composition": "independent", **kwargs}

    def scale(self, *, tensor=EDITABLE[0], factor=0.5, columns=None):
        return {"op": "scale_columns", "tensor": tensor,
                "columns": [0] if columns is None else columns, "factor": factor}

    def assert_rejected(self, editor, proposal, name="rejected"):
        source = files(self.model)
        output = self.root / name
        with self.assertRaises((ValueError, TypeError)):
            editor.propose(proposal, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(f".{name}.incomplete-*")), [])
        self.assertEqual(files(self.model), source)

    def test_identity_preserves_every_stored_bit_and_asset(self):
        editor = self.editor()
        original = files(self.model)
        output = self.root / "identity"
        audit = editor.propose(self.proposal(editor), output)
        self.assertEqual(audit["changed_stored_entries"], 0)
        self.assertEqual(audit["changed_numerical_entries"], 0)
        self.assertEqual(audit["unchanged_tensor_count"], len(editor.inventory["tensors"]))
        self.assertEqual(files(output / "model"), original)
        self.assertEqual(files(self.model), original)
        self.assertNotEqual((output / "model/LICENSE").stat().st_ino,
                            (self.model / "LICENSE").stat().st_ino)
        self.assertEqual(inventory_model(output / "model"), editor.inventory)
        self.assertIsInstance(verify_candidate(self.model, output, self.policy), dict)
        for artifact in ("audit.json", "proposal.json", "policy.json", "exposure.json"):
            self.assertTrue((output / artifact).is_file())

    def test_valid_mixed_dtype_edit_preserves_layout_buffers_and_assets(self):
        editor = self.editor()
        before = editor.inventory
        output = self.root / "edited"
        audit = editor.propose(self.proposal(editor, [self.scale(tensor=name) for name in EDITABLE],
                                            composition="sequential"), output)
        self.assertEqual(audit["changed_stored_entries"], 8)
        self.assertEqual(audit["changed_numerical_entries"], 8)
        self.assertEqual(audit["targeted_coordinates"], 8)
        self.assertEqual(audit["unchanged_tensor_count"], 4)
        self.assertEqual(audit["total_stored_coordinates"], 22)
        self.assertEqual(audit["changed_fraction"], 8 / 22)
        after = inventory_model(output / "model")
        self.assertEqual(after["shards"], before["shards"])
        self.assertEqual(after["index"], before["index"])
        for name, info in before["tensors"].items():
            self.assertEqual({k: v for k, v in info.items() if k != "sha256"},
                             {k: v for k, v in after["tensors"][name].items() if k != "sha256"})
            if name not in EDITABLE:
                self.assertEqual(tensor_bytes(output / "model", name), tensor_bytes(self.model, name))
        self.assertEqual(tensor_bytes(output / "model", EDITABLE[0]),
                         np.array([0x3f00, 0x4000, 0x3fc0, 0x4080], dtype="<u2").tobytes())
        for name, dtype in zip(EDITABLE[1:], ("<f2", "<f4", "<f8")):
            np.testing.assert_array_equal(np.frombuffer(tensor_bytes(output / "model", name), dtype=dtype),
                                          [0.5, 2.0, 1.5, 4.0])
        for name, data in files(self.model).items():
            if not name.endswith(".safetensors"):
                self.assertEqual((output / "model" / name).read_bytes(), data)
        verify_candidate(self.model, output, self.policy)

    def test_each_candidate_starts_from_original_and_composition_is_explicit(self):
        editor = self.editor()
        first, second, composed = (self.root / name for name in ("first", "second", "composed"))
        proposal = self.proposal(editor, [self.scale()])
        editor.propose(proposal, first)
        editor.propose(proposal, second)
        self.assertEqual(files(first / "model"), files(second / "model"))
        editor.propose(self.proposal(editor, [self.scale(), self.scale()], composition="sequential"), composed)
        self.assertEqual(tensor_bytes(composed / "model", EDITABLE[0]),
                         np.array([0x3e80, 0x4000, 0x3f40, 0x4080], dtype="<u2").tobytes())
        self.assert_rejected(editor, self.proposal(editor, [self.scale(), self.scale()]))

    def test_projection_uses_declared_direction_and_columns(self):
        editor = self.editor()
        op = {"op": "remove_direction_from_columns", "tensor": EDITABLE[0],
              "columns": [1], "direction": [1.0, 0.0], "strength": 1.0}
        output = self.root / "projection"
        editor.propose(self.proposal(editor, [op]), output)
        self.assertEqual(tensor_bytes(output / "model", EDITABLE[0]),
                         np.array([0x3f80, 0, 0x4040, 0x4080], dtype="<u2").tobytes())

    def test_bad_proposals_leave_no_candidate_or_original_change(self):
        editor = self.editor()
        cases = [
            self.proposal(editor, [self.scale()], base_sha256="0" * 64),
            self.proposal(editor, [self.scale()], schema_version=True),
            self.proposal(editor, [self.scale()], provenance="astra_discovered"),
            self.proposal(editor, [self.scale()], composition="inherit_last_candidate"),
            self.proposal(editor, [self.scale()], extra="ignored"),
            self.proposal(editor, [self.scale(tensor="missing")]),
            self.proposal(editor, [self.scale(tensor="vector")]),
            self.proposal(editor, [self.scale(tensor="integer_buffer")]),
            self.proposal(editor, [self.scale(columns=[0, 0])]),
            self.proposal(editor, [self.scale(columns=[-1])]),
            self.proposal(editor, [self.scale(columns=[2])]),
            self.proposal(editor, [self.scale(columns=[True])]),
            self.proposal(editor, [self.scale(columns=[])]),
            self.proposal(editor, [self.scale(factor=float("inf"))]),
            self.proposal(editor, [self.scale(factor=float("nan"))]),
            self.proposal(editor, [self.scale(factor=True)]),
            self.proposal(editor, [{**self.scale(), "shape": [4, 1]}]),
            self.proposal(editor, [{"op": "python", "source": "pass"}]),
            self.proposal(editor, [{"op": "remove_direction_from_columns", "tensor": EDITABLE[0],
                                    "columns": [0], "direction": [0, 0], "strength": 1}]),
            self.proposal(editor, [{"op": "remove_direction_from_columns", "tensor": EDITABLE[0],
                                    "columns": [0], "direction": [1], "strength": 1}]),
        ]
        for i, proposal in enumerate(cases):
            with self.subTest(i=i):
                self.assert_rejected(editor, proposal, f"rejected-{i}")

    def test_intervention_policy_restricts_operations_columns_magnitudes_and_budgets(self):
        policies = [
            {"allowed_operations": ["remove_direction_from_columns"]},
            {"allowed_columns": {**self.policy["allowed_columns"], EDITABLE[0]: [1]}},
            {"scale_factor_range": [0.9, 1.1]},
            {"max_abs_change": 0.1},
            {"max_changed_coordinates": 1},
            {"max_changed_fraction": 0.01},
            {"max_targeted_coordinates": 1},
            {"max_working_elements": 1},
            {"max_operations": 0},
        ]
        for i, overrides in enumerate(policies):
            with self.subTest(overrides=overrides):
                policy = {**copy.deepcopy(self.policy), **overrides}
                editor = self.editor(policy, self.root / f"log-{i}.jsonl")
                self.assert_rejected(editor, self.proposal(editor, [self.scale()]), f"policy-rejected-{i}")

    def test_signed_zero_bit_changes_consume_budget_even_without_numeric_difference(self):
        replace_tensor_bytes(self.model, EDITABLE[0], np.zeros(4, dtype="<u2").tobytes())
        policy = {**self.policy, "max_changed_coordinates": 0}
        editor = self.editor(policy)
        self.assert_rejected(editor, self.proposal(editor, [self.scale(factor=-1.0)]))
        editor = self.editor(log=self.root / "accepted-log.jsonl")
        output = self.root / "signed-zero"
        audit = editor.propose(self.proposal(editor, [self.scale(factor=-1.0)]), output)
        self.assertEqual(audit["changed_stored_entries"], 2)
        self.assertEqual(audit["changed_numerical_entries"], 0)
        self.assertEqual(audit["bit_only_changed_entries"], 2)
        self.assertEqual(audit["l2_norm"], 0.0)
        self.assertEqual(tensor_bytes(output / "model", EDITABLE[0]),
                         np.array([0x8000, 0, 0x8000, 0], dtype="<u2").tobytes())
        verify_candidate(self.model, output, self.policy)

    def test_rounding_to_identity_fits_zero_changed_budget(self):
        policy = {**self.policy, "max_changed_coordinates": 0, "max_changed_fraction": 0.0}
        editor = self.editor(policy)
        output = self.root / "rounded-identity"
        audit = editor.propose(self.proposal(editor, [self.scale(factor=1.000001)]), output)
        self.assertEqual(audit["targeted_coordinates"], 2)
        self.assertEqual(audit["changed_stored_entries"], 0)
        self.assertEqual(files(output / "model"), files(self.model))

    def test_composition_budgets_final_changes_and_union_of_declared_coordinates(self):
        policy = {**self.policy, "max_changed_coordinates": 0,
                  "max_changed_fraction": 0.0, "max_targeted_coordinates": 2}
        editor = self.editor(policy)
        output = self.root / "restored-composition"
        editor.propose(self.proposal(editor, [self.scale(), self.scale(factor=2.0)],
                                    composition="sequential"), output)
        self.assertEqual(files(output / "model"), files(self.model))

    def test_stored_dtype_overflow_is_rejected(self):
        replace_tensor_bytes(self.model, EDITABLE[1], np.full(4, 65504, dtype="<f2").tobytes())
        editor = self.editor()
        self.assert_rejected(editor, self.proposal(editor, [self.scale(tensor=EDITABLE[1], factor=2.0)]))

    def test_original_drift_after_construction_is_rejected(self):
        editor = self.editor()
        (self.model / "LICENSE").write_text("changed after identity freeze")
        self.assert_rejected(editor, self.proposal(editor, [self.scale()]))

    def test_nonfinite_untouched_tensor_rejected(self):
        replace_tensor_bytes(self.model, "vector", np.array([float("nan"), 20], dtype="<f4").tobytes())
        with self.assertRaises(ValueError):
            editor = self.editor()
            editor.propose(self.proposal(editor), self.root / "nonfinite")
        self.assertFalse((self.root / "nonfinite").exists())

    def test_reopened_verification_rejects_tampering_with_tensor_or_asset(self):
        editor = self.editor()
        for name in ("integer_buffer", EDITABLE[0], "LICENSE", "model.safetensors.index.json"):
            output = self.root / ("tamper-" + name.replace("/", "-").replace(".", "-"))
            editor.propose(self.proposal(editor, [self.scale()]), output)
            if name in ("integer_buffer", EDITABLE[0]):
                raw = bytearray(tensor_bytes(output / "model", name))
                raw[0] ^= 1
                replace_tensor_bytes(output / "model", name, bytes(raw))
            else:
                target = output / "model" / name
                target.chmod(0o644)
                target.write_bytes(target.read_bytes() + b" ")
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_candidate(self.model, output, self.policy)

    def test_existing_output_and_nested_destination_are_refused(self):
        editor = self.editor()
        output = self.root / "existing"
        output.mkdir()
        (output / "sentinel").write_text("preserve")
        with self.assertRaises((ValueError, FileExistsError)):
            editor.propose(self.proposal(editor), output)
        self.assertEqual(files(output), {"sentinel": b"preserve"})
        with self.assertRaises(ValueError):
            editor.propose(self.proposal(editor), self.model / "nested-candidate")

    def test_exposure_log_cannot_modify_original_assets(self):
        before = files(self.model)
        with self.assertRaises((ValueError, OSError)):
            editor = self.editor(log=self.model / "LICENSE")
            editor.inspect({"op": "metadata", "tensors": [EDITABLE[0]]})
        self.assertEqual(files(self.model), before)

    def test_exposure_log_cannot_modify_original_through_a_hardlink(self):
        empty_asset = self.model / "empty-asset.txt"
        empty_asset.write_bytes(b"")
        linked_log = self.root / "linked-exposure.jsonl"
        os.link(empty_asset, linked_log)
        before = files(self.model)
        with self.assertRaises(ValueError):
            self.editor(log=linked_log)
        self.assertEqual(files(self.model), before)

    def test_verification_rejects_a_different_policy(self):
        editor = self.editor()
        output = self.root / "frozen-policy"
        editor.propose(self.proposal(editor, [self.scale()]), output)
        altered_policy = {**self.policy, "max_changed_coordinates": 0}
        with self.assertRaises(ValueError):
            verify_candidate(self.model, output, altered_policy)

    def test_failed_partial_copy_publishes_nothing_and_next_proposal_is_clean(self):
        editor = self.editor()
        output = self.root / "copy-failed"
        before = files(self.model)
        real_copy = shutil.copyfile
        calls = 0

        def fail_after_one(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated failed checkpoint write")
            return real_copy(*args, **kwargs)

        with mock_patch("weight_space_lab.checkpoint.shutil.copyfile", side_effect=fail_after_one):
            with self.assertRaises(OSError):
                editor.propose(self.proposal(editor, [self.scale()]), output)
        self.assertGreaterEqual(calls, 2)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".copy-failed.incomplete-*")), [])
        self.assertEqual(files(self.model), before)
        editor.propose(self.proposal(editor), output)
        self.assertEqual(files(output / "model"), before)

    def test_failed_audit_write_discards_edited_checkpoint(self):
        editor = self.editor()
        output = self.root / "audit-failed"
        before = files(self.model)
        from weight_space_lab.checkpoint import _write_json

        def fail_audit(path, value):
            if Path(path).name == "audit.json":
                raise OSError("simulated audit write failure")
            return _write_json(path, value)

        with mock_patch("weight_space_lab.checkpoint._write_json", side_effect=fail_audit):
            with self.assertRaises(OSError):
                editor.propose(self.proposal(editor, [self.scale()]), output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".audit-failed.incomplete-*")), [])
        self.assertEqual(files(self.model), before)
        editor.propose(self.proposal(editor), output)
        self.assertEqual(files(output / "model"), before)

    def test_failed_operation_after_mutation_discards_entire_candidate(self):
        editor = self.editor()
        output = self.root / "operation-failed"
        before = files(self.model)
        from weight_space_lab.checkpoint import _write_operation

        def edit_then_fail(*args, **kwargs):
            _write_operation(*args, **kwargs)
            raise OSError("simulated failure after writing selected weights")

        with mock_patch("weight_space_lab.checkpoint._write_operation", side_effect=edit_then_fail):
            with self.assertRaises(OSError):
                editor.propose(self.proposal(editor, [self.scale()]), output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".operation-failed.incomplete-*")), [])
        self.assertEqual(files(self.model), before)

    def test_reloaded_bytes_must_match_declared_operation_not_just_policy(self):
        editor = self.editor()
        output = self.root / "wrong-written-value"
        from weight_space_lab.checkpoint import _write_operation

        def corrupt_selected_value(model, tensor, op):
            result = _write_operation(model, tensor, op)
            data = bytearray(tensor_bytes(Path(model), op["tensor"]))
            # The first BF16 value should become 0.5; 0.25 is finite, inside
            # the declared column and within all coordinate/magnitude limits.
            data[:2] = np.array([0x3e80], dtype="<u2").tobytes()
            replace_tensor_bytes(Path(model), op["tensor"], bytes(data))
            return result

        with mock_patch("weight_space_lab.checkpoint._write_operation", side_effect=corrupt_selected_value):
            with self.assertRaises(ValueError):
                editor.propose(self.proposal(editor, [self.scale()]), output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".wrong-written-value.incomplete-*")), [])

    def test_inspection_exposes_only_declared_slice_and_records_response(self):
        editor = self.editor()
        response = editor.inspect({"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2], [1, 2]]})
        self.assertEqual(response["values"], [[2.0], [4.0]])
        metadata = editor.inspect({"op": "metadata", "tensors": ["integer_buffer"]})
        self.assertIn("I64", json.dumps(metadata))
        self.assertNotIn("double.weight", json.dumps(metadata))
        integer = editor.inspect({"op": "slice", "tensor": "integer_buffer", "slices": [[0, 1]]})
        self.assertEqual(integer["values"], [2**60 + 1])
        stats = editor.inspect({"op": "statistics", "tensor": EDITABLE[0], "slices": [[0, 2], [1, 2]]})
        self.assertEqual(stats["statistics"]["min"], 2.0)
        self.assertEqual(stats["statistics"]["max"], 4.0)
        self.assertEqual(stats["statistics"]["mean"], 3.0)
        entries = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertTrue(any(entry.get("response") == response for entry in entries))
        self.assertTrue(any(entry.get("response") == integer for entry in entries))

    def test_inspection_refuses_unknown_names_undeclared_axes_and_out_of_bounds(self):
        editor = self.editor()
        requests = [
            {"op": "metadata", "tensors": ["missing"]},
            {"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2]]},
            {"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 3], [0, 1]]},
            {"op": "slice", "tensor": EDITABLE[0], "slices": [[-1, 1], [0, 1]]},
            {"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2, 1], [0, 1]]},
            {"op": "statistics", "tensor": EDITABLE[0]},
            {"op": "python", "source": "print(1)"},
        ]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                editor.inspect(request)

    def test_inspection_limits_persist_when_editor_is_reopened(self):
        policy = copy.deepcopy(self.policy)
        policy["inspection"].update(max_requests=1)
        editor = self.editor(policy)
        request = {"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 1], [0, 1]]}
        editor.inspect(request)
        reopened = self.editor(policy)
        with self.assertRaises(ValueError):
            reopened.inspect(request)

    def test_exposure_counter_tampering_cannot_reset_persisted_budgets(self):
        editor = self.editor()
        editor.inspect({"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2], [0, 2]]})
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        records[1]["values_exposed"] = 0
        records[1]["coordinates_scanned"] = 0
        self.log.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaises(ValueError):
            self.editor()

    def test_exposure_sequence_requires_an_integer_not_boolean(self):
        editor = self.editor()
        editor.inspect({"op": "metadata", "tensors": [EDITABLE[0]]})
        records = [json.loads(line) for line in self.log.read_text().splitlines()]
        records[1]["sequence"] = True
        self.log.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaises(ValueError):
            self.editor()

    def test_inspection_value_and_scan_budgets_are_enforced(self):
        for index, limit in enumerate(("max_values", "max_scanned_coordinates",
                                       "max_total_values", "max_total_scanned_coordinates")):
            policy = copy.deepcopy(self.policy)
            policy["inspection"][limit] = 1
            editor = self.editor(policy, self.root / f"inspection-{index}.jsonl")
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                editor.inspect({"op": "slice", "tensor": EDITABLE[0], "slices": [[0, 2], [0, 1]]})


if __name__ == "__main__":
    unittest.main()
