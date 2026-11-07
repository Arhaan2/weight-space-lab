"""Packet checks use only authored synthetic metadata and statistics."""

import copy
import unittest

from weight_space_lab.protocol import canonical_hash
from weight_space_lab.session_packet import build_manual_packet
from weight_space_lab.session_protocol import default_session_protocol, make_session_request


class SessionPacketTests(unittest.TestCase):
    def setUp(self):
        self.tensor = "model.layers.0.down_proj.weight"
        self.protocol = default_session_protocol(max_new_tokens=128, target_timeout_seconds=300,
                                                local_elapsed_seconds=900, effort="ultra")
        policy = {
            "schema_version": 1, "allowed_tensors": [self.tensor],
            "allowed_operations": ["scale_columns", "remove_direction_from_columns"],
            "allowed_columns": {self.tensor: "all"}, "max_changed_fraction": 1.0,
            "max_changed_coordinates": 6, "max_targeted_coordinates": 6,
            "max_operations": 4, "max_working_elements": 6,
            "scale_factor_range": [0.0, 2.0], "direction_strength_range": [0.0, 1.0],
            "max_abs_change": 1.0,
            "inspection": {"allowed_tensors": [self.tensor], "max_requests": 2,
                "max_values": 6, "max_scanned_coordinates": 6,
                "max_total_values": 12, "max_total_scanned_coordinates": 12},
        }
        self.request = make_session_request(request_id="1" * 32, session_sha256="2" * 64,
            instruction="Synthetic packet rendering check.", protocol=self.protocol,
            approved_context={"model_info": {"checkpoint_sha256": "3" * 64,
                "architectures": ["SyntheticArchitecture"], "stored_parameter_count": 6,
                "allowed_tensors": {self.tensor: {"shape": [2, 3], "dtype": "BF16", "parameter_count": 6}}},
                "policy": policy, "development": [{"item_id": "synthetic-development",
                    "split": "development", "messages": [{"role": "user", "content": "Synthetic objective only."}]}],
                "feedback": None})
        self.inspections = [{"request": {"op": "statistics", "tensor": self.tensor, "slices": [[0, 2], [0, 3]]},
            "result": {"tensor": self.tensor, "dtype": "BF16", "shape": [2, 3],
                "statistics": {"count": 6, "min": -0.25, "max": 0.5,
                    "mean": 0.1, "l2_norm": 0.75, "finite": True},
                "statistics_precision": "exact extrema; binary64 approximate mean and L2"}}]

    def render(self, **kwargs):
        values = {"request": self.request, "protocol": self.protocol, "inspections": self.inspections,
                  "operator_model": "Astra", "operator_mode": "Ultra",
                  "submission_expires_at_utc": "2030-01-02T03:04:05Z"}
        values.update(kwargs)
        return build_manual_packet(**values)

    def test_actual_bindings_statistics_and_exact_operation_grammar_present(self):
        packet = self.render()
        self.assertIn(canonical_hash(self.request), packet)
        self.assertIn("3" * 64, packet)
        self.assertIn('slices=[[0,2],[0,3]]', packet)
        self.assertIn('"l2_norm":0.75', packet)
        self.assertIn('"factor":FACTOR', packet)
        self.assertIn('"direction":DIRECTION,"strength":STRENGTH', packet)
        self.assertIn('"proposal":PROPOSAL', packet)
        self.assertIn("Synthetic objective only.", packet)
        self.assertNotIn('"factor":0.99', packet)

    def test_exploratory_scope_and_operator_selection_are_not_runtime_identity(self):
        packet = self.render()
        self.assertIn("Astra, Ultra mode", packet)
        self.assertIn("effort ultra", packet)
        self.assertNotIn("effort high", packet)
        self.assertIn("Editor isolation is UNVERIFIED", packet)
        self.assertIn("Runtime model identity, immutable version, effort metadata and frontier usage are unavailable", packet)
        self.assertIn('"model_identity":{"provider":null,"model":null,"version":null,"interface":null}', packet)
        self.assertIn("runner neither launches nor cancels", packet)
        self.assertIn("128 generated target tokens each (256 total)", packet)
        self.assertIn("2030-01-02T03:04:05Z", packet)

    def test_no_silent_mode_mapping(self):
        with self.assertRaisesRegex(ValueError, "Ultra"):
            self.render(operator_mode="high")
        protocol = copy.deepcopy(self.protocol)
        protocol["editor"]["requested_effort"] = "high"
        request = copy.deepcopy(self.request)
        request["protocol_sha256"] = canonical_hash(protocol)
        with self.assertRaisesRegex(ValueError, "Ultra"):
            self.render(request=request, protocol=protocol)

    def test_unapproved_and_extra_inspection_channels_rejected(self):
        for mutate in (lambda value: value[0]["result"].update(final_answers="forbidden"),
                       lambda value: value[0]["request"].update(tensor="unapproved.tensor"),
                       lambda value: value[0]["result"].update(statistics_precision="/Users/private/data")):
            inspections = copy.deepcopy(self.inspections)
            mutate(inspections)
            with self.assertRaises(ValueError):
                self.render(inspections=inspections)

    def test_mismatched_statistics_or_ranges_rejected(self):
        for mutate in (lambda value: value[0]["result"]["statistics"].update(count=100),
                       lambda value: value[0]["request"].update(slices=[[0, 4], [0, 3]]),
                       lambda value: value[0]["result"]["statistics"].update(mean=float("nan")),
                       lambda value: value[0]["result"].update(shape=[3, 2])):
            inspections = copy.deepcopy(self.inspections)
            mutate(inspections)
            with self.assertRaises(ValueError):
                self.render(inspections=inspections)

    def test_expiry_requires_explicit_utc(self):
        for expiry in ("2030-01-02T03:04:05", "2030-01-02T03:04:05-07:00", "not-a-time", None):
            with self.assertRaisesRegex(ValueError, "UTC"):
                self.render(submission_expires_at_utc=expiry)

    def test_machine_paths_in_approved_text_are_not_exported(self):
        request = copy.deepcopy(self.request)
        request["approved_context"]["development"][0]["messages"][0]["content"] = "Read /Users/private/file"
        with self.assertRaisesRegex(ValueError, "machine-specific"):
            self.render(request=request)


if __name__ == "__main__":
    unittest.main()
