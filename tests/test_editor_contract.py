"""Development-only handoff validation; these fixtures invoke no model."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from weight_space_lab.editor_contract import (
    ALLOWED_TOOLS, read_json_bounded, validate_development, validate_request, validate_response,
)
from weight_space_lab.protocol import canonical_hash


def request_fixture():
    return {
        "schema_version": 1,
        "kind": "editor_request",
        "request_id": "a" * 32,
        "session_sha256": "b" * 64,
        "mode": "offline_integration",
        "instruction": "Inspect the approved synthetic development model.",
        "approved_context": {
            "model_info": {"provenance": "synthetic_fixture"},
            "development": [{"item_id": "dev-one", "split": "development",
                             "messages": [{"role": "user", "content": "Development fixture."}]}],
            "policy": {},
            "feedback": None,
        },
        "limits": {"max_output_tokens": 64, "timeout_seconds": 2.5},
        "allowed_tools": list(ALLOWED_TOOLS),
    }


def response_fixture(request=None, tool="stop", arguments=None):
    request = request or request_fixture()
    return {
        "schema_version": 1,
        "kind": "editor_response",
        "request_id": request["request_id"],
        "request_sha256": canonical_hash(request),
        "evidence_kind": "simulated",
        "model_identity": {"provider": None, "model": None, "version": None, "interface": None},
        "action": {"tool": tool, "arguments": arguments or {}},
        "usage": {"generated_tokens": None, "billed_units": None},
        "scientific_result": False,
    }


class EditorContractTests(unittest.TestCase):
    def test_round_trip_is_detached_and_unverified(self):
        request = request_fixture()
        result = validate_request(request)
        result["approved_context"]["model_info"]["provenance"] = "changed"
        self.assertEqual(request["approved_context"]["model_info"]["provenance"], "synthetic_fixture")
        response = response_fixture(request)
        response["evidence_kind"] = "unverified_file_handoff"
        response["model_identity"]["model"] = "untrusted-claim"
        result = validate_response(response, request)
        self.assertEqual(result, response)
        self.assertIsNot(result, response)
        self.assertIsNone(result["model_identity"]["version"])

    def test_test_assets_extra_fields_and_other_roles_are_rejected(self):
        base = request_fixture()["approved_context"]["development"]
        invalid = []
        for split in ("test", "validation", "final-test"):
            value = copy.deepcopy(base)
            value[0]["split"] = split
            invalid.append(value)
        value = copy.deepcopy(base)
        value[0]["target"] = "evaluator-only"
        invalid.append(value)
        value = copy.deepcopy(base)
        value[0]["messages"][0]["role"] = "system"
        invalid.append(value)
        invalid.extend((base * 2, base * 33, [], [{"item_id": "dev"}]))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_development(value)

    def test_denies_unbounded_or_changed_request_surface(self):
        changes = [
            ("schema_version", True), ("mode", "live"), ("request_id", "../outside"),
            ("session_sha256", "unknown"), ("instruction", "x" * 16001),
            ("allowed_tools", list(ALLOWED_TOOLS) + ["shell"]),
            ("limits", {"max_output_tokens": True, "timeout_seconds": 1}),
            ("limits", {"max_output_tokens": 1, "timeout_seconds": float("inf")}),
            ("limits", {"max_output_tokens": 1, "timeout_seconds": 0}),
        ]
        for name, value in changes:
            request = request_fixture()
            request[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_request(request)
        request = request_fixture()
        request["final_test"] = "unapproved"
        with self.assertRaises(ValueError):
            validate_request(request)

    def test_response_is_bound_to_exact_frozen_request(self):
        request = request_fixture()
        response = response_fixture(request)
        request["instruction"] = "Changed after dispatch."
        with self.assertRaisesRegex(ValueError, "exact request"):
            validate_response(response, request)
        request = request_fixture()
        response["request_id"] = "c" * 32
        with self.assertRaises(ValueError):
            validate_response(response, request)

    def test_never_accepts_live_or_scientific_response_claim(self):
        for key, value in (("evidence_kind", "live"), ("evidence_kind", "astra"),
                           ("scientific_result", True), ("scientific_result", 0)):
            response = response_fixture()
            response[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_response(response, request_fixture())

    def test_no_code_files_network_training_or_evaluation_tools(self):
        for name in ("shell", "exec", "read_file", "network", "train", "gradient", "optimizer",
                     "set_architecture", "evaluate_test", "change_evaluation"):
            response = response_fixture(tool=name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "approved tool"):
                validate_response(response, request_fixture())

    def test_probe_uses_only_frozen_task_and_candidate_references(self):
        valid = response_fixture(tool="probe", arguments={"task_ids": ["dev-one"], "candidate_id": "base"})
        self.assertEqual(validate_response(valid, request_fixture()), valid)
        for arguments in (
            {"task_ids": ["test-one"], "candidate_id": "base"},
            {"task_ids": ["dev-one", "dev-one"], "candidate_id": "base"},
            {"task_ids": ["dev-one"], "candidate_id": "/outside/checkpoint"},
            {"task_ids": ["dev-one"], "candidate_id": "base", "seed": 44},
            {"task_ids": ["dev-one"], "candidate_id": "base", "max_tokens": 100000},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_response(response_fixture(tool="probe", arguments=arguments), request_fixture())

    def test_invalid_usage_and_arguments_fail(self):
        for usage in ({"generated_tokens": True, "billed_units": None},
                      {"generated_tokens": -1, "billed_units": 0},
                      {"generated_tokens": 1, "billed_units": float("nan")},
                      {"generated_tokens": 1, "billed_units": -1},
                      {"generated_tokens": None, "billed_units": None, "refund": 100}):
            response = response_fixture()
            response["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                validate_response(response, request_fixture())
        for tool, arguments in (("stop", {"path": "outside"}), ("propose", {}),
                                ("propose", {"proposal": "code"})):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                validate_response(response_fixture(tool=tool, arguments=arguments), request_fixture())

    def test_json_rejects_duplicate_keys_nonfinite_growth_and_invalid_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "response.json"
            for content in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'\xff',
                            b'[' * 40 + b'0' + b']' * 40):
                path.write_bytes(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_json_bounded(path)
            path.write_text(json.dumps(response_fixture()))
            self.assertEqual(read_json_bounded(path), response_fixture())
            with self.assertRaises(ValueError):
                read_json_bounded(path, max_bytes=16)

    def test_json_denies_symlink_leaf_ancestor_traversal_and_fifo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory = root / "real"
            directory.mkdir()
            path = directory / "response.json"
            path.write_text("{}")
            (root / "link.json").symlink_to(path)
            (root / "alias").symlink_to(directory, target_is_directory=True)
            os.mkfifo(root / "pipe")
            for unsafe in (root / "link.json", root / "alias" / "response.json",
                           directory / ".." / "real" / "response.json", root / "pipe", directory):
                with self.subTest(path=unsafe), self.assertRaises(ValueError):
                    read_json_bounded(unsafe)


if __name__ == "__main__":
    unittest.main()
