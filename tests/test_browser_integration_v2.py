"""Synthetic durable gate tests only: no browser, worker, model, or inference."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from weight_space_lab import browser_integration_v2 as v2
from weight_space_lab.inventory import file_sha256


CLEAN = {key: True for key in ("worker_terminated", "chrome_exit_confirmed",
    "local_harness_exit_confirmed", "controller_exit_verified", "unload_confirmed")}
PREGEN = {"confirmed_pre_generation": True, "generation_intent_absent": True,
    "generation_ack_absent": True, "unknown_execution_state": False,
    "oom_or_device_loss": False, "resource_stop": False,
    "diagnosis": "Synthetic browser fetch received HTML for the model configuration URL"}


class BrowserIntegrationV2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.root = self.path / "v2"
        self.authorization = self.path / "authorization.txt"
        self.authorization.write_text("Synthetic fixture only; not real authorization")
        self.pair = self.path / "artifacts.json"
        self.write(self.pair, {"arms": {arm: {"source_checkpoint_sha256": str(i)*64,
            "artifact_sha256": str(i+2)*64} for i, arm in enumerate(v2.ARMS, 1)}})
        self.configs = {}
        for i, arm in enumerate(v2.ARMS, 1):
            path = self.path / (arm + ".json")
            self.write(path, {"arm": arm, "settings": dict(v2.SETTINGS),
                "items": [{"item_id": "synthetic", "prompt": "Synthetic smoke"}],
                "sourceCheckpointSha256": str(i)*64, "convertedArtifactSha256": str(i+2)*64,
                "configurationSha256": "a"*64, "runId": "synthetic-"+arm,
                "model": {"modelUrl": "http://localhost/models/"+arm+"/",
                          "modelLib": "http://localhost/model.wasm", "id": arm}})
            self.configs[arm] = path
        self.prepare()
        self.controller = v2.Controller(self.root)
        self.addCleanup(lambda: self.controller.close())
        self.proof = self.path / "preflight.json"
        self.write(self.proof, {"status": "passed", "fetch_only": True,
                              "engine_loads": 0, "generation_dispatches": 0})
        self.build = self.path / "build.js"
        self.version_index = 0
        self.code = self.version()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def prepare(self):
        return v2.prepare(self.root, authorization_source=self.authorization,
                          arm_configs=self.configs, artifact_manifest=self.pair)

    def version(self, previous=None, failure_class="fetch", contents=None):
        self.version_index += 1
        self.build.write_text(contents or "synthetic code "+str(self.version_index))
        correction = None if previous is None else {"previous_attempt_id": previous,
            "failure_class": failure_class, "diagnosis": "Confirmed synthetic asset route failure",
            "correction": "Serve the exact model file instead of application HTML"}
        return self.controller.register_code_version({str(self.build): file_sha256(self.build)},
                    preflight_path=self.proof, correction=correction)

    def intent(self, kind="load", arm="baseline", attempt="base-1", code=None):
        config = self.controller.configs[arm]
        return {"schema": "browser-v2-dispatch-intent", "kind": kind, "arm": arm,
                "request_id": kind+"-1", "attempt_id": attempt,
                "code_version_sha256": code or self.code,
                "configuration_sha256": config["configurationSha256"],
                "artifact_sha256": config["convertedArtifactSha256"],
                "run_id": config["runId"], "created_at_epoch_ms": time.time()*1000}

    def gate(self, kind="load", arm="baseline", attempt="base-1", code=None):
        intent = self.intent(kind, arm, attempt, code)
        return self.controller.handle_intent(intent, arm=arm, attempt_id=attempt,
                                             code_sha256=code or self.code)

    def succeed(self, arm="baseline", attempt="base-1", code=None, tokens=10):
        self.gate("load", arm, attempt, code)
        self.gate("generation", arm, attempt, code)
        return self.controller.finish_attempt(attempt, outcome="completed", cleanup=CLEAN,
                                              generated_tokens=tokens, generation_seconds=1)

    def load_failure(self, attempt="base-1", failure_class="fetch", evidence=None, cleanup=None):
        return self.controller.finish_attempt(attempt, outcome="load_failed",
            failure_class=failure_class, evidence=PREGEN if evidence is None else evidence,
            cleanup=CLEAN if cleanup is None else cleanup)

    def test_fresh_scope_and_deadline_starts_only_with_load_ack(self):
        before = self.controller.summary()
        self.assertEqual(before["engine_load_attempts"], 0)
        self.assertEqual(before["generation_dispatches_consumed"], 0)
        self.assertIsNone(before["clock"]["deadline_epoch_ms"])
        ack = self.gate()
        self.assertAlmostEqual(ack["deadline_epoch_ms"]-ack["started_epoch_ms"], 1200000)
        self.assertEqual(self.controller.summary()["engine_load_attempts"], 1)

    def test_ack_exact_worker_identity_contract(self):
        for kind in ("load", "generation"):
            ack = self.gate(kind)
            intent = self.intent(kind)
            self.assertTrue(ack["approved"])
            for key in ("kind", "arm", "request_id", "attempt_id", "code_version_sha256"):
                self.assertEqual(ack[key], intent[key])
            self.assertGreater(ack["deadline_epoch_ms"], time.time()*1000)

    def test_two_sequential_arms_and_actual_token_accounting(self):
        self.succeed(tokens=256)
        self.succeed("candidate", "candidate-1", tokens=17)
        summary = self.controller.summary()
        self.assertEqual(summary["engine_load_attempts"], 2)
        self.assertEqual(summary["generation_dispatches_consumed"], 2)
        self.assertEqual(summary["generation_budget"]["observed"]["generated_tokens"], 273)
        self.assertEqual(summary["generation_slots_remaining"], {"baseline": 0, "candidate": 0})
        with self.assertRaisesRegex(ValueError, "slot was dispatched"):
            self.controller.begin_load("candidate", "candidate-2", self.code)

    def test_duplicate_ack_does_not_spend_a_second_slot(self):
        self.gate()
        self.gate("generation")
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.gate("generation")
        self.assertEqual(self.controller.summary()["generation_dispatches_consumed"], 1)

    def test_zero_tokens_never_refunds_generation_slot(self):
        self.succeed(tokens=0)
        with self.assertRaisesRegex(ValueError, "slot was dispatched"):
            self.gate(attempt="base-2")
        self.assertEqual(self.controller.summary()["generation_budget"]["charged"]["requested_sequences"], 1)

    def test_unknown_generation_retains_full_reservation_and_stops(self):
        self.gate()
        self.gate("generation")
        self.controller.finish_attempt("base-1", outcome="unknown", cleanup=CLEAN)
        result = self.controller.summary()
        self.assertEqual(result["generation_budget"]["charged"]["generated_tokens"], 256)
        self.assertTrue(result["stopped"])
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.gate(arm="candidate", attempt="candidate-1")

    def test_cross_database_interruption_consumes_arm_without_ack(self):
        self.gate()
        with patch.object(self.controller.generations, "reserve", side_effect=RuntimeError("interruption")):
            with self.assertRaises(RuntimeError):
                self.gate("generation")
        result = self.controller.summary()
        self.assertEqual(result["generation_dispatches_consumed"], 1)
        self.assertEqual(result["conservative_tokens_for_intents_without_reservation"], 256)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            self.gate("generation")

    def test_confirmed_url_retry_requires_specific_changed_version_and_releases(self):
        self.gate()
        first_deadline = self.controller.deadline()["deadline_epoch_ms"]
        self.assertTrue(self.load_failure()["retry_eligible"])
        with self.assertRaisesRegex(ValueError, "changed code"):
            self.gate(attempt="base-2")
        changed = self.version("base-1")
        self.gate(attempt="base-2", code=changed)
        self.assertEqual(self.controller.deadline()["deadline_epoch_ms"], first_deadline)

    def test_new_snapshot_path_with_identical_bytes_cannot_enable_retry(self):
        self.gate()
        self.load_failure()
        copied = self.path / "copied.js"
        copied.write_bytes(self.build.read_bytes())
        changed = self.controller.register_code_version({str(copied): file_sha256(copied)},
            preflight_path=self.proof, correction={"previous_attempt_id":"base-1", "failure_class":"fetch",
                "diagnosis":"Synthetic same failure", "correction":"Merely copied code"})
        with self.assertRaisesRegex(ValueError, "changed code"):
            self.gate(attempt="base-2", code=changed)

    def test_two_additional_loads_maximum_and_four_total(self):
        self.gate()
        self.load_failure()
        second = self.version("base-1")
        self.gate(attempt="base-2", code=second)
        self.load_failure("base-2")
        third = self.version("base-2")
        self.succeed(attempt="base-3", code=third)
        self.gate(arm="candidate", attempt="candidate-1", code=third)
        self.load_failure("candidate-1")
        fourth = self.version("candidate-1")
        with self.assertRaisesRegex(ValueError, "four engine-loading"):
            self.gate(arm="candidate", attempt="candidate-2", code=fourth)
        summary = self.controller.summary()
        self.assertEqual(summary["engine_load_attempts"], 4)
        self.assertEqual(summary["additional_load_attempts"], 2)

    def test_candidate_cannot_run_before_baseline_release(self):
        with self.assertRaisesRegex(ValueError, "released successful baseline"):
            self.gate(arm="candidate", attempt="candidate-1")
        self.gate()
        with self.assertRaisesRegex(ValueError, "not been released"):
            self.gate(arm="candidate", attempt="candidate-1")

    def test_out_of_memory_failure_cannot_be_relabelled_as_fetch_retry(self):
        self.gate()
        result = self.load_failure(evidence=PREGEN | {"oom_or_device_loss": True})
        self.assertFalse(result["retry_eligible"])
        self.assertTrue(self.controller.summary()["stopped"])

    def test_schema_failure_requires_offline_stop(self):
        self.gate()
        self.assertFalse(self.load_failure(failure_class="parameter_schema")["retry_eligible"])
        self.assertTrue(self.controller.summary()["stopped"])

    def test_unconfirmed_cleanup_forbids_loading_retry(self):
        self.gate()
        result = self.load_failure(cleanup=CLEAN | {"chrome_exit_confirmed": False})
        self.assertFalse(result["retry_eligible"])
        self.assertTrue(self.controller.summary()["stopped"])

    def test_dispatched_generation_forbids_retry_even_on_fetch_error(self):
        self.gate()
        self.gate("generation")
        self.assertFalse(self.load_failure()["retry_eligible"])
        self.assertTrue(self.controller.summary()["stopped"])

    def test_changed_served_build_or_preflight_rejected_before_load(self):
        self.build.write_text("unexpected mutation")
        with self.assertRaisesRegex(ValueError, "served code changed"):
            self.gate()
        self.assertEqual(self.controller.summary()["engine_load_attempts"], 0)

    def test_model_initialization_is_not_accepted_as_fetch_only(self):
        self.write(self.proof, {"status":"passed", "fetch_only": True,
                              "engine_loads": 1, "generation_dispatches": 0})
        with self.assertRaisesRegex(ValueError, "fetch-only"):
            self.version()

    def test_wrong_artifact_intent_cannot_consume_load(self):
        intent = self.intent() | {"artifact_sha256": "f"*64}
        with self.assertRaisesRegex(ValueError, "frozen arm"):
            self.controller.handle_intent(intent, arm="baseline", attempt_id="base-1", code_sha256=self.code)
        self.assertEqual(self.controller.summary()["engine_load_attempts"], 0)

    def test_global_deadline_survives_reopen_and_expires_permanently(self):
        self.gate()
        deadline = self.controller.deadline()
        self.controller.close()
        self.controller = v2.Controller(self.root)
        self.assertEqual(self.controller.deadline()["deadline_epoch_ms"], deadline["deadline_epoch_ms"])
        fake = SimpleNamespace(time=lambda: deadline["deadline_epoch_ms"]/1000+1,
                               monotonic=lambda: deadline["deadline_monotonic"]+1)
        with patch.object(v2, "time", fake):
            with self.assertRaisesRegex(TimeoutError, "deadline expired"):
                self.gate("generation")
        self.assertTrue(self.controller.summary()["stopped"])
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.gate("generation")

    def test_completion_after_global_deadline_is_failed_with_usage_retained(self):
        self.gate()
        self.gate("generation")
        deadline = self.controller.deadline()
        fake = SimpleNamespace(time=lambda: deadline["deadline_epoch_ms"]/1000+1,
                               monotonic=lambda: deadline["deadline_monotonic"]+1)
        with patch.object(v2, "time", fake):
            result = self.controller.finish_attempt("base-1", outcome="completed", cleanup=CLEAN,
                                                      generated_tokens=19, generation_seconds=2)
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(result["failure_class"], "deadline_expired")
        self.assertTrue(self.controller.summary()["stopped"])
        self.assertEqual(self.controller.summary()["generation_budget"]["observed"]["generated_tokens"], 19)

    def test_retained_owned_pid_overrides_successful_cleanup_flags(self):
        self.gate()
        self.gate("generation")
        result = self.controller.finish_attempt("base-1", outcome="completed",
            cleanup=CLEAN | {"retained_owned_pids_alive": [123]}, generated_tokens=19, generation_seconds=2)
        self.assertEqual(result["outcome"], "failed")
        self.assertTrue(self.controller.summary()["stopped"])

    def test_root_ack_writer_is_atomic_and_never_overwrites(self):
        from weight_space_lab.browser_v2_runner import durable_json
        path = self.path / "load-ack.json"
        first = {"approved":True,"kind":"load","attempt_id":"base-1","deadline_epoch_ms":12345}
        durable_json(path, first)
        self.assertEqual(json.loads(path.read_text()), first)
        with self.assertRaises(FileExistsError):
            durable_json(path, {"approved": False})
        self.assertEqual(json.loads(path.read_text()), first)
        self.assertEqual(list(self.path.glob(".load-ack.json.*")), [])

    def test_generation_gate_leaves_cleanup_headroom(self):
        self.gate()
        deadline = self.controller.deadline()
        fake = SimpleNamespace(time=lambda: deadline["deadline_epoch_ms"]/1000-74,
                               monotonic=lambda: deadline["deadline_monotonic"]-74)
        with patch.object(v2, "time", fake):
            with self.assertRaisesRegex(TimeoutError, "75 seconds"):
                self.gate("generation")
        self.assertEqual(self.controller.summary()["generation_dispatches_consumed"], 0)

    def test_clock_regression_durably_stops(self):
        self.gate()
        fake = SimpleNamespace(time=lambda: 1, monotonic=lambda: 1)
        with patch.object(v2, "time", fake):
            with self.assertRaisesRegex(ValueError, "clock regressed"):
                self.gate("generation")
        self.assertTrue(self.controller.summary()["stopped"])

    def test_token_overage_retained_and_stops(self):
        result = self.succeed(tokens=257)
        self.assertEqual(result["generated_tokens"], 257)
        summary = self.controller.summary()
        self.assertTrue(summary["stopped"])
        self.assertEqual(summary["generation_budget"]["observed"]["generated_tokens"], 257)

    def test_preparation_never_reopens_or_recreates_capacity(self):
        with self.assertRaisesRegex(ValueError, "unused manifest"):
            self.prepare()

    def test_authorization_mutation_rejected(self):
        (self.root / "authorization-source.txt").write_text("mutated")
        with self.assertRaisesRegex(ValueError, "authorization identity"):
            v2.Controller(self.root)


if __name__ == "__main__":
    unittest.main()
