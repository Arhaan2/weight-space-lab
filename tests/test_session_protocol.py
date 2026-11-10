"""Synthetic contract tests; no authenticated session or target model is used."""

import copy
import json
import unittest

from weight_space_lab.editor_contract import validate_request, validate_response
from weight_space_lab.protocol import canonical_hash
from weight_space_lab.session_protocol import (
    ASTRA_SELECTOR, SESSION_VARIANT, default_session_protocol, make_session_request,
    public_session_summary, validate_session_protocol, validate_session_provenance,
    validate_session_request, validate_session_response,
)


def protocol_fixture():
    return default_session_protocol(max_new_tokens=8, target_timeout_seconds=3,
                                    local_elapsed_seconds=20)


def policy_fixture():
    return {
        "schema_version": 1, "allowed_tensors": ["synthetic.weight"],
        "allowed_operations": ["scale_columns", "remove_direction_from_columns"],
        "allowed_columns": {"synthetic.weight": "all"},
        "max_changed_fraction": 1, "max_changed_coordinates": 4,
        "max_targeted_coordinates": 4, "max_operations": 1,
        "max_working_elements": 4, "scale_factor_range": [0, 2],
        "direction_strength_range": [0, 1], "max_abs_change": 1,
        "inspection": {"allowed_tensors": ["synthetic.weight"], "max_requests": 1,
                       "max_values": 4, "max_scanned_coordinates": 4,
                       "max_total_values": 4, "max_total_scanned_coordinates": 4},
    }


def request_fixture(protocol=None):
    return make_session_request(
        request_id="a" * 32, session_sha256="b" * 64,
        instruction="Return a synthetic development proposal.",
        approved_context={
            "model_info": {"checkpoint_sha256": "c" * 64,
                           "architectures": ["SyntheticModel"], "stored_parameter_count": 4,
                           "allowed_tensors": {"synthetic.weight": {
                               "shape": [2, 2], "dtype": "F32", "parameter_count": 4}}},
            "development": [{"item_id": "dev-one", "split": "development",
                             "messages": [{"role": "user", "content": "Synthetic fixture."}]}],
            "policy": policy_fixture(), "feedback": None,
        }, protocol=protocol or protocol_fixture(),
    )


def response_fixture(request=None):
    request = request or request_fixture()
    return {
        "schema_version": 1, "kind": "editor_response",
        "request_id": request["request_id"], "request_sha256": canonical_hash(request),
        "evidence_kind": "session_handoff",
        "model_identity": {"provider": None, "model": None, "version": None, "interface": None},
        "action": {"tool": "propose", "arguments": {"proposal": {
            "schema_version": 1, "base_sha256": "c" * 64,
            "provenance": "editor_proposal", "composition": "independent",
            "operations": [{"op": "scale_columns", "tensor": "synthetic.weight",
                            "columns": [0], "factor": 0.99}],
        }}},
        "usage": {"generated_tokens": None, "billed_units": None},
        "scientific_result": False,
    }


def provenance_fixture():
    return {
        "schema_version": 1, "source": "unavailable",
        "requested_selector": ASTRA_SELECTOR, "requested_effort": "high",
        "accepted_selector": None, "observed_model": None, "observed_effort": None,
        "immutable_model_version": None,
        "interface": {"name": "codex-app", "version": None}, "evidence_sha256": None,
    }


class SessionProtocolTests(unittest.TestCase):
    def test_variant_is_explicit_without_fabricating_frontier_cap(self):
        protocol = protocol_fixture()
        request = request_fixture(protocol)
        self.assertEqual(request["mode"], SESSION_VARIANT)
        self.assertEqual(request["allowed_tools"], ["propose"])
        self.assertIsNone(request["limits"]["frontier_provider_token_cap"])
        self.assertNotIn("max_output_tokens", request["limits"])
        self.assertEqual(validate_session_protocol(protocol), protocol)
        self.assertEqual(validate_session_request(request, protocol), request)
        response = response_fixture(request)
        self.assertEqual(validate_session_response(response, request, protocol), response)
        # A new variant does not relax or overwrite the original strict mode.
        with self.assertRaises(ValueError):
            validate_request(request)
        with self.assertRaises(ValueError):
            validate_response(response, request)

    def test_scope_cannot_expand_or_change_restrictions(self):
        for key in protocol_fixture()["scope"]:
            protocol = protocol_fixture()
            protocol["scope"][key] += 1
            with self.subTest(scope=key), self.assertRaises(ValueError):
                validate_session_protocol(protocol)
        for key in protocol_fixture()["restrictions"]:
            protocol = protocol_fixture()
            protocol["restrictions"][key] = not protocol["restrictions"][key]
            with self.subTest(restriction=key), self.assertRaises(ValueError):
                validate_session_protocol(protocol)
        protocol = protocol_fixture()
        protocol["scope"]["editor_requests"] = True
        with self.assertRaises(ValueError):
            validate_session_protocol(protocol)

    def test_protocol_rejects_wrong_model_cap_and_invalid_frozen_limits(self):
        changes = [
            ("editor", "requested_selector", "other-model"),
            ("editor", "requested_effort", "unknown"),
            ("limits", "frontier_provider_token_cap", 128),
            ("limits", "frontier_provider_token_cap_status", "local_timeout"),
            ("limits", "target_max_new_tokens", 0),
            ("limits", "target_max_new_tokens", True),
            ("limits", "target_timeout_seconds", float("inf")),
            ("limits", "local_elapsed_seconds", -1),
        ]
        for group, key, item in changes:
            protocol = protocol_fixture()
            protocol[group][key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_protocol(protocol)
        protocol = protocol_fixture()
        protocol["provider_api_key"] = "not-public"
        with self.assertRaises(ValueError):
            validate_session_protocol(protocol)

    def test_request_rejects_changed_protocol_and_extra_channels(self):
        protocol = protocol_fixture()
        request = request_fixture(protocol)
        protocol["limits"]["target_max_new_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            validate_session_request(request, protocol)
        for key, item in (
            ("allowed_tools", ["propose", "shell"]),
            ("mode", "offline_integration"),
            ("limits", {"max_response_bytes": 262144, "local_elapsed_seconds": 20,
                        "frontier_provider_token_cap": 128}),
        ):
            request = request_fixture()
            request[key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        request = request_fixture()
        request["approved_context"]["final_answers"] = "unapproved"
        with self.assertRaises(ValueError):
            validate_session_request(request, protocol_fixture())
        request = request_fixture()
        request["approved_context"]["development"][0]["split"] = "test"
        with self.assertRaises(ValueError):
            validate_session_request(request, protocol_fixture())
        request = request_fixture()
        another = copy.deepcopy(request["approved_context"]["development"][0])
        another["item_id"] = "dev-two"
        request["approved_context"]["development"].append(another)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_session_request(request, protocol_fixture())

    def test_response_binding_and_proposal_only_surface(self):
        request = request_fixture()
        response = response_fixture(request)
        request["instruction"] = "Changed request."
        with self.assertRaisesRegex(ValueError, "exact request"):
            validate_session_response(response, request, protocol_fixture())
        for tool in ("probe", "inspect", "shell", "network", "train", "gradient", "optimizer",
                     "evaluate_test", "write_file", "stop"):
            response = response_fixture()
            response["action"] = {"tool": tool, "arguments": {}}
            with self.subTest(tool=tool), self.assertRaisesRegex(ValueError, "only a proposal"):
                validate_session_response(response, request_fixture(), protocol_fixture())
        response = response_fixture()
        response["action"]["arguments"]["path"] = "unapproved"
        with self.assertRaises(ValueError):
            validate_session_response(response, request_fixture(), protocol_fixture())

    def test_metadata_policy_and_feedback_have_no_arbitrary_information_channels(self):
        for field in ("final_tasks", "final_answers", "final_results", "storage", "path"):
            request = request_fixture()
            request["approved_context"]["model_info"][field] = "undeclared"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        for group in ("policy", "model_info"):
            request = request_fixture()
            nested = request["approved_context"][group]
            if group == "policy":
                nested["inspection"]["final_answers"] = "undeclared"
            else:
                nested["allowed_tensors"]["synthetic.weight"]["storage_path"] = "/private/model"
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        for feedback in ({}, {"final_results": "undeclared"}, {"development": "prior run"}):
            request = request_fixture()
            request["approved_context"]["feedback"] = feedback
            with self.subTest(feedback=feedback), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())

    def test_approved_metadata_counts_identifiers_and_policy_are_validated(self):
        changes = [
            ("checkpoint_sha256", "not-a-digest"),
            ("architectures", ["/private/model"]),
            ("architectures", ["SyntheticModel", "SyntheticModel"]),
            ("stored_parameter_count", True),
            ("stored_parameter_count", -1),
            ("stored_parameter_count", 3),
            ("allowed_tensors", {}),
        ]
        for key, item in changes:
            request = request_fixture()
            request["approved_context"]["model_info"][key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        for key, item in (("shape", [2]), ("shape", [2, True]),
                          ("shape", [-1, -4]), ("parameter_count", 3),
                          ("parameter_count", True), ("dtype", "unknown")):
            request = request_fixture()
            request["approved_context"]["model_info"]["allowed_tensors"]["synthetic.weight"][key] = item
            with self.subTest(tensor_field=key), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        for key, item in (("allowed_tensors", ["other.weight"]),
                          ("allowed_operations", ["optimizer_update"]),
                          ("max_changed_fraction", 2),
                          ("allowed_columns", {"synthetic.weight": [2]})):
            request = request_fixture()
            request["approved_context"]["policy"][key] = item
            with self.subTest(policy_field=key), self.assertRaises(ValueError):
                validate_session_request(request, protocol_fixture())
        request = request_fixture()
        request["approved_context"]["model_info"]["architectures"] = []
        tensor = request["approved_context"]["model_info"]["allowed_tensors"]["synthetic.weight"]
        tensor.update(shape=[0, 2], parameter_count=0)
        request["approved_context"]["model_info"]["stored_parameter_count"] = 0
        self.assertEqual(validate_session_request(request, protocol_fixture()), request)

    def test_response_cannot_establish_identity_or_scientific_claim(self):
        for key, item in (("evidence_kind", "authenticated"), ("scientific_result", True),
                          ("scientific_result", 0)):
            response = response_fixture()
            response[key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_response(response, request_fixture(), protocol_fixture())
        for key in ("provider", "model", "version", "interface"):
            response = response_fixture()
            response["model_identity"][key] = ASTRA_SELECTOR
            with self.subTest(identity=key), self.assertRaisesRegex(ValueError, "model-authored identity"):
                validate_session_response(response, request_fixture(), protocol_fixture())
        response = response_fixture()
        response["usage"]["generated_tokens"] = True
        with self.assertRaises(ValueError):
            validate_session_response(response, request_fixture(), protocol_fixture())

    def test_invalid_proposal_envelopes_fail_before_checkpoint_execution(self):
        for key, item in (("operations", []), ("operations", "executable-code"),
                          ("provenance", "engineering_edit"), ("base_sha256", "unknown"),
                          ("composition", "search"), ("schema_version", True)):
            response = response_fixture()
            response["action"]["arguments"]["proposal"][key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_response(response, request_fixture(), protocol_fixture())
        response = response_fixture()
        response["action"]["arguments"]["proposal"]["harness_patch"] = "unapproved"
        with self.assertRaises(ValueError):
            validate_session_response(response, request_fixture(), protocol_fixture())

    def test_provenance_distinguishes_configured_accepted_observed_and_version(self):
        protocol = protocol_fixture()
        provenance = provenance_fixture()
        self.assertEqual(validate_session_provenance(provenance, protocol), provenance)
        provenance.update(source="operator_attestation", accepted_selector=ASTRA_SELECTOR)
        result = validate_session_provenance(provenance, protocol)
        self.assertEqual(result["source"], "operator_attestation")
        self.assertIsNone(result["observed_model"])
        self.assertIsNone(result["immutable_model_version"])
        provenance.update(source="transport_metadata", observed_model=ASTRA_SELECTOR,
                          observed_effort="high", evidence_sha256="d" * 64)
        provenance["interface"]["version"] = "0.163.0"
        result = validate_session_provenance(provenance, protocol)
        self.assertEqual(result["interface"]["version"], "0.163.0")
        self.assertIsNone(result["immutable_model_version"])
        self.assertEqual(public_session_summary(protocol, result)["identity_verification"],
                         "external_receipt_required")

    def test_unavailable_and_self_description_provenance_cannot_authenticate(self):
        for key, item in (("source", "model_self_description"),
                          ("accepted_selector", "other-model"),
                          ("accepted_selector", ASTRA_SELECTOR),
                          ("requested_effort", "low"),
                          ("observed_effort", "unknown"),
                          ("immutable_model_version", "/private/version")):
            provenance = provenance_fixture()
            provenance[key] = item
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session_provenance(provenance, protocol_fixture())
        provenance = provenance_fixture()
        provenance["source"] = "transport_metadata"
        with self.assertRaisesRegex(ValueError, "retained evidence"):
            validate_session_provenance(provenance, protocol_fixture())

    def test_public_projection_is_bounded_metadata_only(self):
        protocol = protocol_fixture()
        summary = public_session_summary(protocol, provenance_fixture())
        self.assertEqual(summary["editor"]["requested_selector"], ASTRA_SELECTOR)
        self.assertIsNone(summary["observed_frontier_usage"]["generated_tokens"])
        self.assertIsNone(summary["observed_frontier_usage"]["billed_units"])
        self.assertNotIn("Synthetic fixture.", json.dumps(summary))
        for key in ("instruction", "approved_context", "conversation", "credentials", "path"):
            self.assertNotIn(key, summary)
        for item in (request_fixture(), {"generated_tokens": None, "billed_units": None,
                                         "conversation": "private"}):
            with self.assertRaises(ValueError):
                public_session_summary(protocol, usage=item)
        provenance = provenance_fixture()
        provenance["interface"]["name"] = "/machine-specific/path"
        with self.assertRaises(ValueError):
            public_session_summary(protocol, provenance)
        summary["scope"]["editor_requests"] = 100
        self.assertEqual(protocol["scope"]["editor_requests"], 1)


if __name__ == "__main__":
    unittest.main()
