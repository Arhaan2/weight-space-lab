import copy
import json
import unittest

from weight_space_lab.ifeval_data import (
    GROUND_FIELDS, MAPPINGS, QUARANTINED_FUNCTIONS, feasibility_reasons,
    map_ground_truth, normalize_prompt, rendered_segments, stratified_select,
)


def row(fn, **kwargs):
    gt = {key: None for key in GROUND_FIELDS}
    gt.update(func_name=fn, **kwargs)
    return {"ground_truth": json.dumps(gt), "constraint_type": MAPPINGS.get(fn, ("unknown",))[0], "dataset": "ifeval"}


class PracticeAdapterTests(unittest.TestCase):
    def test_explicit_argument_translation_and_integer_upper_bound(self):
        target = map_ground_truth(row("validate_word_constraint", N=7, quantifier="at most"))
        self.assertEqual(target, {"instruction_id_list": ["length_constraints:number_words"],
                                  "kwargs": [{"num_words": 8, "relation": "less than"}]})
        target = map_ground_truth(row("validate_paragraphs", N=4, i=3, first_word="Apple"))
        self.assertEqual(target["kwargs"], [{"num_paragraphs": 4, "nth_paragraph": 3, "first_word": "Apple"}])

    def test_unsupported_and_ambiguous_records_are_quarantined(self):
        for function in QUARANTINED_FUNCTIONS:
            with self.subTest(function=function), self.assertRaises(ValueError):
                map_ground_truth(row(function))
        with self.assertRaisesRegex(ValueError, "around"):
            map_ground_truth(row("validate_word_constraint", N=10, quantifier="around"))
        with self.assertRaisesRegex(ValueError, "unexpected_nonnull"):
            map_ground_truth(row("validate_no_commas", keyword_list=["hidden constraint"]))
        with self.assertRaisesRegex(ValueError, "contradictory_nth"):
            map_ground_truth(row("validate_paragraphs", N=2, i=3, first_word="hello"))

    def test_feasibility_rejects_whole_google_prompt(self):
        item = {"task_text": "A task", "target": {"instruction_id_list": ["punctuation:no_comma", "language:response_language"], "kwargs": [{}, {"language": "en"}]}}
        original = copy.deepcopy(item)
        self.assertIn("unsupported_instruction_type", feasibility_reasons(item, 50))
        self.assertEqual(item, original)
        self.assertIn("rendered_input_plus_output_exceeds_context", feasibility_reasons(item, 3073))

    def test_obvious_contradiction_is_preoutput_exclusion(self):
        item = {"task_text": "A task", "target": {"instruction_id_list": ["keywords:existence", "keywords:forbidden_words"], "kwargs": [{"keywords": ["Orange"]}, {"forbidden_words": ["orange"]}]}}
        self.assertIn("same_keyword_required_and_forbidden", feasibility_reasons(item, 40))

    def test_nonthinking_template_is_closed_before_output(self):
        segments = rendered_segments("Keep this prompt.\n")
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[1], "<|im_start|>user\nKeep this prompt.\n<|im_end|>\n")
        self.assertTrue(segments[-1].endswith("<think>\n\n</think>\n\n"))

    def test_selection_is_deterministic_and_covers_rare_types(self):
        items = [{"item_id": str(i), "task_text": f"Task {i}", "metadata": {},
                  "target": {"instruction_id_list": ["rare" if i == 0 else "common"]}} for i in range(20)]
        first = stratified_select(items, 5)
        second = stratified_select(list(reversed(items)), 5)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["item_id"], "0")

    def test_normalization_is_explicit(self):
        self.assertEqual(normalize_prompt("  Ａ  WORD\n"), "a word")


if __name__ == "__main__":
    unittest.main()
