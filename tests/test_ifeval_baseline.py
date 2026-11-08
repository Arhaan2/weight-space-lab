"""Budget and extraction regressions; no browser or target inference."""
import tempfile
from pathlib import Path
import unittest

from weight_space_lab import ifeval_baseline as baseline
from weight_space_lab.editor_budget import BudgetExhausted
from weight_space_lab.grading import _known_answer_item, separate_output
from weight_space_lab.browser_tasks import adapt_browser_output


class IFEvalBaselineTests(unittest.TestCase):
    def config(self):
        return {"configurationSha256": "a"*64, "convertedArtifactSha256": "b"*64,
                "runId": "fixture-baseline", "items": [{"item_id": f"item-{i}"} for i in range(32)]}

    def intent(self, index):
        config = self.config()
        return {"kind": "generation", "arm": "baseline", "request_id": f"generation-{index}",
                "attempt_id": "baseline-1", "code_version_sha256": "c"*64,
                "configuration_sha256": config["configurationSha256"],
                "artifact_sha256": config["convertedArtifactSha256"], "run_id": config["runId"],
                "warmup": index == 1, "sequence_index": index-1,
                "max_tokens": 16 if index == 1 else 1024,
                "item_id": "__warmup__" if index == 1 else f"item-{index-2}"}

    def test_intents_bind_order_and_warmup_limit(self):
        for index in range(1, 34):
            baseline.validate_intent(self.intent(index), self.config(), "c"*64, index)
        for key, value in (("max_tokens", 1024), ("warmup", False), ("sequence_index", 1)):
            intent = self.intent(1)
            intent[key] = value
            with self.assertRaises(ValueError):
                baseline.validate_intent(intent, self.config(), "c"*64, 1)
        intent = self.intent(2)
        intent["item_id"] = "item-1"
        with self.assertRaises(ValueError):
            baseline.validate_intent(intent, self.config(), "c"*64, 2)
        with self.assertRaises(ValueError):
            baseline.validate_intent(self.intent(34), self.config(), "c"*64, 34)

    def test_capacity_counts_warmup_and_zero_token_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            with baseline.ledger(directory, {"fixture": True}, create=True) as book:
                for index in range(1, 34):
                    request = f"generation-{index}"
                    book.reserve(request, kind="probe", sequences=1,
                        max_generated_tokens=16 if index == 1 else 1024,
                        timeout_seconds=120, max_billed_units=0)
                    book.mark_dispatched(request)
                    book.finish(request, status="completed", completed_sequences=1, failed_sequences=0,
                        generated_tokens=0, elapsed_seconds=0.1, billed_units=0)
                self.assertEqual(book.summary()["reserved"]["generated_tokens"], 32784)
                self.assertEqual(book.summary()["charged"]["requested_sequences"], 33)
                with self.assertRaises(BudgetExhausted):
                    book.reserve("generation-34", kind="probe", sequences=1, max_generated_tokens=1,
                                 timeout_seconds=1, max_billed_units=0)
            with self.assertRaises(FileExistsError):
                baseline.ledger(directory, {"fixture": True}, create=True)

    def test_ambiguous_output_keeps_conservative_budget(self):
        with tempfile.TemporaryDirectory() as directory, baseline.ledger(directory, {"fixture": True}, create=True) as book:
            book.reserve("generation-2", kind="probe", sequences=1, max_generated_tokens=1024,
                         timeout_seconds=120, max_billed_units=0)
            book.mark_dispatched("generation-2")
            with self.assertRaises(ValueError):
                baseline.reconcile(book, {"generated_tokens": 2, "response_latency_ms": 121000}, 2, self.config())
            self.assertEqual(book.summary()["charged"]["generated_tokens"], 1024)
            book.recover_interrupted(executor_stopped=True)
            self.assertEqual(book.summary()["charged"]["generated_tokens"], 1024)

    def test_nonthinking_empty_prefix_and_plain_content(self):
        self.assertEqual(separate_output("{\"ok\":true}", baseline.POLICY)[1:], ('{"ok":true}', "output_ready"))
        self.assertEqual(separate_output('<think>\n\n</think>\n\n{"ok":true}', baseline.POLICY)[1:],
                         ('\n\n{"ok":true}', "output_ready"))
        self.assertEqual(separate_output("<think>unfinished", baseline.POLICY)[2], "malformed_reasoning_boundary")
        self.assertEqual(separate_output("text<|im_end|>", baseline.POLICY)[1], "text<|im_end|>")

    def test_ifeval_missing_and_truncated_raw_records(self):
        item = _known_answer_item("ifeval", {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]})
        identity = {"model_id": "fixture", "revision": "a"*40, "sha256": "b"*64}
        args = {"run_id": "fixture", "checkpoint": identity, "configuration_sha256": "c"*64, "evidence_kind": "fixture"}
        missing, _ = adapt_browser_output(item, None, **args)
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["attempts"], [])
        raw = '<think>\n\n</think>\n\n{"ok":true}'
        record, _ = adapt_browser_output(item, {"item_id": item["item_id"], "raw_output": raw,
            "finish_reason": "length", "generated_tokens": 1024, "input_tokens": 28, "response_latency_ms": 20000}, **args)
        self.assertEqual(record["raw_output"], raw)
        self.assertEqual(record["status"], "truncated")
        self.assertEqual(record["token_counts"]["total"], 1052)


if __name__ == "__main__":
    unittest.main()
