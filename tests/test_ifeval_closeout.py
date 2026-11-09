"""Public closeout consistency and fail-closed checks; no inference or grading."""
from copy import deepcopy
import json
import unittest

from scripts.verify_ifeval_closeout import REPORTS, COMBINED, build_result, validate_completed


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((REPORTS / COMBINED / "result.json").read_text())

    def test_preserved_complete_record_and_total_browser_usage(self):
        validate_completed(self.source)
        usage = build_result()["full_v1_browser_usage"]
        self.assertEqual([usage[k] for k in ("engine_load_attempts", "warmup_dispatches",
                         "scored_dispatches", "generation_dispatches",
                         "scored_generated_tokens", "generated_tokens_including_warmup",
                         "reserved_generated_tokens")], [5, 5, 320, 325, 156276, 156336, 327760])

    def test_incomplete_or_changed_interval_is_not_a_closeout(self):
        cases = [deepcopy(self.source), deepcopy(self.source)]
        cases[0]["complete_paired_evaluation"] = False
        cases[1]["metrics"]["prompt_strict"]["ci95_percentage_points"] = [0, 1]
        for source in cases:
            with self.subTest(source=source["complete_paired_evaluation"]):
                with self.assertRaises(ValueError):
                    validate_completed(source)

    def test_denominator_and_session_token_changes_are_rejected(self):
        cases = [deepcopy(self.source), deepcopy(self.source)]
        cases[0]["metrics"]["instruction_strict"]["candidate"]["denominator"] = 128
        cases[1]["sessions"]["candidate_continuation"]["accounting"]["scored_generated_tokens"] += 1
        for source in cases:
            with self.assertRaises(ValueError):
                validate_completed(source)

    def test_paired_counts_must_be_present_not_inferred_from_net(self):
        del self.source["paired_prompt_outcomes"]["strict"]["improved"]
        with self.assertRaises(KeyError):
            validate_completed(self.source)


if __name__ == "__main__":
    unittest.main()
