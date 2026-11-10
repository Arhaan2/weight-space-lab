"""Independent Ai2 practice / Google IFEval subset, selected before inference.

Only explicit mappings to pinned official checkers are accepted. This module
does not generate responses, alter benchmark prompts, or reuse older split roles.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark_data import load_source, validate_item
from .protocol import canonical_hash

AI2_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
AI2_PATH = "data/train-00000-of-00001.parquet"
AI2_SHA256 = "75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971"
AI2_BYTES = 11653857
AI2_ROWS = 14973
GROUND_FIELDS = {"func_name", "N", "quantifier", "end_phrase", "keyword_list", "word",
                 "forbidden_words", "letter", "i", "first_word", "postscript_marker",
                 "options", "section_splitter", "original_prompt"}

# (source constraint label, official ID, explicit source -> official arguments)
MAPPINGS = {
    "validate_lowercase": ("All Lowercase", "change_case:english_lowercase", {}),
    "validate_uppercase": ("All Uppercase", "change_case:english_capital", {}),
    "validate_no_commas": ("No Commas", "punctuation:no_comma", {}),
    "validate_json_format": ("JSON Format", "detectable_format:json_format", {}),
    "validate_quotation": ("Quotation", "startend:quotation", {}),
    "validate_title": ("Title", "detectable_format:title", {}),
    "validate_two_responses": ("Two Responses", "combination:two_responses", {}),
    "verify_keywords": ("Include Keywords", "keywords:existence", {"keyword_list": "keywords"}),
    "validate_forbidden_words": ("Forbidden Words", "keywords:forbidden_words", {"forbidden_words": "forbidden_words"}),
    "validate_end": ("End Checker", "startend:end_checker", {"end_phrase": "end_phrase"}),
    "verify_postscript": ("Postscript", "detectable_content:postscript", {"postscript_marker": "postscript_marker"}),
    "validate_placeholders": ("Number Placeholder", "detectable_content:number_placeholders", {"N": "num_placeholders"}),
    "verify_bullet_points": ("Number Bullets", "detectable_format:number_bullet_lists", {"N": "num_bullets"}),
    "validate_highlighted_sections": ("Minimum Number Highlighted Section", "detectable_format:number_highlighted_sections", {"N": "num_highlights"}),
    "validate_paragraphs": ("Number Paragraphs + First Word in i-th Paragraph", "length_constraints:nth_paragraph_first_word", {"N": "num_paragraphs", "i": "nth_paragraph", "first_word": "first_word"}),
    "validate_word_constraint": ("Number Words", "length_constraints:number_words", {"N": "num_words"}),
    "verify_sentence_constraint": ("Number Sentences", "length_constraints:number_sentences", {"N": "num_sentences"}),
    "validate_frequency_capital_words": ("Frequency of All-capital Words", "change_case:capital_word_frequency", {"N": "capital_frequency"}),
}
QUARANTINED_FUNCTIONS = {
    "validate_choice": "arbitrary_options_not_supported_by_official_fixed_choice_checker",
    "validate_repeat_prompt": "ambiguous_original_prompt_includes_the_directive_to_exclude",
    "verify_paragraph_count": "spaced_markdown_divider_differs_from_official_checker",
    "verify_keyword_frequency": "exact_keyword_frequency_not_a_single_official_relational_constraint",
    "verify_letter_frequency": "exact_letter_frequency_not_a_single_official_relational_constraint",
    "validate_sections": "exact_section_requirement_differs_from_official_minimum_checker",
}
RELATIONAL = {"validate_word_constraint": ("num_words", "relation"),
              "verify_sentence_constraint": ("num_sentences", "relation"),
              "validate_frequency_capital_words": ("capital_frequency", "capital_relation")}
SUPPORTED_IDS = frozenset(spec[1] for spec in MAPPINGS.values())
TEMPLATE_SEGMENTS = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
                     "<|im_start|>user\n{prompt}<|im_end|>\n",
                     "<|im_start|>assistant\n<think>\n\n</think>\n\n"]
ELIGIBILITY_POLICY = {
    "id": "ifeval-browser-v1-preoutput-feasibility-v1",
    "context_tokens": 4096, "output_allowance": 1024,
    "input_rule": "sum of exact WebLLM template segment token counts + 1024 <= 4096",
    "output_rules": {"minimum_words_max": 400, "minimum_sentences_max": 40,
                     "paragraphs_max": 40, "placeholders_max": 40, "bullets_max": 40,
                     "highlights_max": 40, "capital_words_max": 40,
                     "explicit_task_word_requirement_max": 400},
    "contradictions": ["incompatible upper/lowercase requirements", "required and forbidden same keyword",
                       "invalid nth paragraph", "no-comma with mandatory comma", "incompatible exact output format",
                       "explicit incompatible response counts or output format in original task",
                       "pick-one option format with additive structure", "nontext image generation request",
                       "closed verdict/evidence JSON schema conflicts with added text or structure",
                       "manual task-only audit recorded before final selection"],
    "estimates_are_not_completion_guarantees": True,
    "selection_observes_model_outputs": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def normalize_prompt(prompt):
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def map_ground_truth(row):
    """Map one entire source constraint or reject it; never drop a constraint."""
    gt = json.loads(row["ground_truth"])
    if not isinstance(gt, dict) or set(gt) != GROUND_FIELDS:
        raise ValueError("unexpected_ground_truth_schema")
    fn = gt["func_name"]
    if fn not in MAPPINGS:
        raise ValueError(QUARANTINED_FUNCTIONS.get(fn, "unsupported_function"))
    label, official, fields = MAPPINGS[fn]
    if row["constraint_type"] != label or row["dataset"] != "ifeval":
        raise ValueError("source_label_disagrees_with_function")
    allowed = {"func_name", *fields}
    if fn in RELATIONAL:
        allowed.add("quantifier")
    if any(value is not None and key not in allowed for key, value in gt.items()):
        raise ValueError("unexpected_nonnull_source_argument")
    kwargs = {target: gt[source] for source, target in fields.items()}
    if any(value is None for value in kwargs.values()):
        raise ValueError("missing_source_argument")
    for key in ("N", "i"):
        if key in fields and (type(gt[key]) is not int or gt[key] < 1):
            raise ValueError("invalid_positive_integer_argument")
    for key in ("keyword_list", "forbidden_words"):
        if key in fields and (not isinstance(gt[key], list) or not gt[key]
                             or any(not isinstance(v, str) or not v.strip() for v in gt[key])):
            raise ValueError("invalid_keyword_list")
    for key in ("end_phrase", "postscript_marker", "first_word"):
        if key in fields and (not isinstance(gt[key], str) or not gt[key].strip()):
            raise ValueError("invalid_string_argument")
    if fn in RELATIONAL:
        threshold, relation = RELATIONAL[fn]
        if gt["quantifier"] == "at least":
            kwargs[relation] = "at least"
        elif gt["quantifier"] == "at most":
            # Integer count <= N is exactly equivalent to count < N+1.
            kwargs[threshold] += 1
            kwargs[relation] = "less than"
        else:
            raise ValueError("ambiguous_around_quantifier")
    if fn == "validate_paragraphs" and gt["i"] > gt["N"]:
        raise ValueError("contradictory_nth_paragraph")
    return {"instruction_id_list": [official], "kwargs": [kwargs]}


def rendered_segments(prompt):
    return [segment.replace("{prompt}", prompt) for segment in TEMPLATE_SEGMENTS]


def input_tokens(prompt, tokenizer):
    return sum(len(tokenizer.encode(segment, add_special_tokens=False).ids)
               for segment in rendered_segments(prompt))


def source_task_text(row):
    """Best-effort exact removal of the appended generated directive for audit.

    This value never becomes the model prompt. Missing or ambiguous exact matches
    remain unavailable, rather than guessing the underlying task.
    """
    gt = json.loads(row["ground_truth"])
    text = row["constraint"]
    values = {"N": gt["N"], "word": gt["word"], "letter": gt["letter"], "i": gt["i"],
              "first word": gt["first_word"], "end phrase": gt["end_phrase"],
              "postscript marker": gt["postscript_marker"], "section splitter": gt["section_splitter"],
              "options": gt["options"], "forbidden words": ", ".join(gt["forbidden_words"] or []),
              "keyword1": (gt["keyword_list"] or [None, None])[0],
              "keyword2": (gt["keyword_list"] or [None, None])[-1]}
    for name, value in values.items():
        if value is not None:
            text = text.replace("{" + name + "}", str(value))
    if gt["quantifier"]:
        text = text.replace("at least / around / at most", gt["quantifier"])
    prompt = row["messages"][0]["content"]
    if "{" in text or prompt.count(text) != 1:
        return None
    return prompt.replace(text, "", 1).strip()


def feasibility_reasons(item, count):
    """Task-only conservative screening; no response or model state is accepted."""
    reasons = []
    if count + 1024 > 4096:
        reasons.append("rendered_input_plus_output_exceeds_context")
    pairs = list(zip(item["target"]["instruction_id_list"], item["target"]["kwargs"]))
    ids = set(x[0] for x in pairs)
    if not ids <= SUPPORTED_IDS:
        reasons.append("unsupported_instruction_type")
    if {"change_case:english_capital", "change_case:english_lowercase"} <= ids:
        reasons.append("contradictory_upper_and_lowercase")
    required, forbidden = set(), set()
    for instruction, kwargs in pairs:
        if instruction == "keywords:existence":
            required.update(v.casefold() for v in kwargs["keywords"])
        if instruction == "keywords:forbidden_words":
            forbidden.update(v.casefold() for v in kwargs["forbidden_words"])
        for key, limit in [("num_paragraphs", 40), ("num_placeholders", 40), ("num_bullets", 40), ("num_highlights", 40)]:
            if kwargs.get(key, 0) > limit:
                reasons.append("structural_output_requirement_exceeds_predeclared_limit")
        if kwargs.get("relation") == "at least" and kwargs.get("num_words", 0) > 400:
            reasons.append("minimum_word_requirement_exceeds_predeclared_limit")
        if kwargs.get("relation") == "at least" and kwargs.get("num_sentences", 0) > 40:
            reasons.append("minimum_sentence_requirement_exceeds_predeclared_limit")
        if kwargs.get("capital_relation") == "at least" and kwargs.get("capital_frequency", 0) > 40:
            reasons.append("capital_word_requirement_exceeds_predeclared_limit")
        if kwargs.get("nth_paragraph", 1) > kwargs.get("num_paragraphs", 10**9):
            reasons.append("contradictory_nth_paragraph")
        if "punctuation:no_comma" in ids:
            mandatory = kwargs.get("keywords", []) + [kwargs.get("end_phrase", ""), kwargs.get("postscript_marker", "")]
            if any("," in v for v in mandatory):
                reasons.append("no_comma_conflicts_with_required_literal")
    if required & forbidden:
        reasons.append("same_keyword_required_and_forbidden")
    text = item["task_text"]
    if any(int(n.replace(",", "")) > 400 for n in re.findall(r"(?<![\w.])(\d[\d,]*)[- ]word\b", text, re.I)):
        reasons.append("task_text_requests_over_400_words")
    if re.search(r"\b(?:at least|minimum of|exactly)\s+(\d[\d,]*)\s+words\b", text, re.I):
        for n in re.findall(r"\b(?:at least|minimum of|exactly)\s+(\d[\d,]*)\s+words\b", text, re.I):
            if int(n.replace(",", "")) > 400:
                reasons.append("task_text_requests_over_400_words")
    # These source-task formulations expressly disallow additive formatting.
    exclusive = re.search(r"\b(?:answer|output|respond|reply)(?:\s+should\s+be|\s+must\s+be|\s+with|\s+in|\s+is)?\s+(?:only\s+)?(?:a\s+)?(?:single word|one word|single letter|one letter|just (?:yes|no)|yes or no)\b", text, re.I)
    additive = ids - {"punctuation:no_comma", "change_case:english_lowercase", "change_case:english_capital", "keywords:forbidden_words"}
    if exclusive and additive:
        reasons.append("exclusive_short_answer_conflicts_with_added_constraint")
    if re.search(r"\b(?:pick from the following|choose (?:one|a single) (?:answer|option))\b", text, re.I) and additive:
        reasons.append("pick_one_option_conflicts_with_added_constraint")
    if re.search(r"\b(?:automatically )?(?:generate|draw|create)\s+(?:a|an)\s+(?:logo|image|picture|photograph)\b", text, re.I) and not re.search(r"\b(?:svg|code|description|describe)\b", text, re.I):
        reasons.append("nontext_output_request")
    if "detectable_format:json_format" in ids and ("startend:quotation" in ids or "detectable_format:title" in ids or "detectable_format:number_bullet_lists" in ids):
        reasons.append("incompatible_json_and_other_output_format")
    if "return a JSON object with two fields" in text and '"verdict"' in text and '"evidence"' in text:
        incompatible = {"detectable_format:number_highlighted_sections", "detectable_format:title", "detectable_format:number_bullet_lists",
                        "detectable_content:number_placeholders", "detectable_content:postscript", "startend:end_checker",
                        "startend:quotation", "combination:two_responses", "length_constraints:nth_paragraph_first_word",
                        "change_case:english_lowercase", "change_case:english_capital"}
        if ids & incompatible:
            reasons.append("closed_json_schema_conflicts_with_added_constraint")
    return sorted(set(reasons))


def stratified_select(items, n, seed=42, *, key="instruction_id_list"):
    """Greedy rare-family coverage then balance counts; seeded stable tie order."""
    if len(items) < n:
        raise ValueError("insufficient eligible examples")
    shuffled = sorted(items, key=lambda item: canonical_hash({"seed": seed, "item_id": item["item_id"]}))
    priority = {item["item_id"]: rank for rank, item in enumerate(shuffled)}
    remaining = list(shuffled)
    counts = Counter()
    frequencies = Counter(f for item in items for f in set(item["target"][key]))
    selected, seen_tasks = [], set()
    while len(selected) < n:
        available = [i for i in remaining if i["metadata"].get("normalized_task_sha256", canonical_hash(normalize_prompt(i["task_text"]))) not in seen_tasks]
        if not available:
            raise ValueError("insufficient distinct eligible tasks")
        chosen = min(available, key=lambda item: (
            min(counts[f] for f in item["target"][key]),
            -sum(1 / frequencies[f] for f in item["target"][key] if counts[f] == 0),
            sum(counts[f] for f in item["target"][key]) / len(item["target"][key]),
            priority[item["item_id"]]))
        selected.append(chosen)
        remaining.remove(chosen)
        counts.update(chosen["target"][key])
        seen_tasks.add(chosen["metadata"].get("normalized_task_sha256", canonical_hash(normalize_prompt(chosen["task_text"]))))
    return selected


def family_coverage(items):
    return {"instruction_types": dict(sorted(Counter(f for item in items for f in item["target"]["instruction_id_list"]).items())),
            "instruction_families": dict(sorted(Counter(f.split(":")[0] for item in items for f in item["target"]["instruction_id_list"]).items())),
            "prompts": len(items), "instructions": sum(len(i["target"]["instruction_id_list"]) for i in items)}


def mapping_fixtures():
    """Independent semantic examples, including relational boundary cases."""
    english = "This is an English sentence about the weather and a pleasant day outside."
    return [
        ("validate_lowercase", {}, english.lower(), english),
        ("validate_uppercase", {}, english.upper(), english),
        ("validate_no_commas", {}, "One and two.", "One, two."),
        ("validate_json_format", {}, '{"count": 2}', "count equals two"),
        ("validate_quotation", {}, '"A complete response."', "A complete response."),
        ("validate_title", {}, "<<A useful title>>\nBody text.", "A useful title\nBody text."),
        ("validate_two_responses", {}, "First answer.******Second answer.", "First answer.******First answer."),
        ("verify_keywords", {"keyword_list": ["apple", "orange"]}, "apple and orange", "apple and pear"),
        ("validate_forbidden_words", {"forbidden_words": ["apple", "orange"]}, "pear and peach", "apple and peach"),
        ("validate_end", {"end_phrase": "The end."}, "A story. The end.", "The end. More words."),
        ("verify_postscript", {"postscript_marker": "P.S."}, "A letter.\nP.S. One more thought.", "A letter with no postscript."),
        ("validate_placeholders", {"N": 2}, "Dear [name], visit [address].", "Dear [name], visit here."),
        ("verify_bullet_points", {"N": 2}, "* First\n* Second", "* First\n* Second\n* Third"),
        ("validate_highlighted_sections", {"N": 2}, "*First* and *second*", "*First* and second"),
        ("validate_paragraphs", {"N": 2, "i": 2, "first_word": "Second"}, "First text.\n\nSecond text.", "First text.\n\nWrong text."),
        ("validate_word_constraint", {"N": 3, "quantifier": "at least"}, "one two three", "one two"),
        ("validate_word_constraint", {"N": 3, "quantifier": "at most"}, "one two three", "one two three four"),
        ("verify_sentence_constraint", {"N": 2, "quantifier": "at least"}, "The sun shines. The rain falls.", "The sun shines."),
        ("verify_sentence_constraint", {"N": 2, "quantifier": "at most"}, "The sun shines. The rain falls.", "The sun shines. The rain falls. The wind blows."),
        ("validate_frequency_capital_words", {"N": 2, "quantifier": "at least"}, "ONE TWO three", "ONE two three"),
        ("validate_frequency_capital_words", {"N": 2, "quantifier": "at most"}, "ONE TWO three", "ONE TWO THREE"),
    ]


def run_mapping_checks(ifeval_root):
    """Use pinned official evaluation, with installed-version/randomness checks."""
    from .grading_worker import evaluate
    from .grading import grader_identity, verify_ifeval_assets
    verify_ifeval_assets(ifeval_root)
    cases = []
    for index, (fn, arguments, passing, failing) in enumerate(mapping_fixtures()):
        gt = {key: None for key in GROUND_FIELDS}
        gt.update(func_name=fn, **arguments)
        row = {"ground_truth": json.dumps(gt), "constraint_type": MAPPINGS[fn][0], "dataset": "ifeval"}
        target = map_ground_truth(row)
        for expected, output in [(True, passing), (False, failing)]:
            result = evaluate({"benchmark": "ifeval", "ifeval_root": str(ifeval_root),
                               "item": {"item_id": str(index), "task_text": "Independent adapter known-answer case.", "target": target},
                               "output": output})
            passed = result["status"] == "graded" and result["correct"] is expected
            cases.append({"function": fn, "arguments": arguments, "target": target,
                          "expected_strict": expected, "observed": result, "passed": passed})
    return {"schema_version": 1, "kind": "ifeval_practice_mapping_known_answers", "adapter_sha256": sha(__file__),
            "grader": grader_identity("ifeval"), "supported_functions": sorted(MAPPINGS),
            "covered_functions": sorted(set(case["function"] for case in cases)),
            "cases": cases, "passed": all(case["passed"] for case in cases),
            "target_forward_passes": 0}


def prepare_data(root, tokenizer_path, google_cache, source_reference, *, manual_exclusions=None, publish=True, tokenizer_audit=None):
    """Freeze a new study; refusal on existing final manifest prevents reuse."""
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("study data already frozen")
    import pyarrow
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    if pyarrow.__version__ != "19.0.1" or importlib.metadata.version("tokenizers") != "0.22.2":
        raise ValueError("data preparation dependency mismatch")
    source = root / "source" / Path(AI2_PATH).name
    if source.stat().st_size != AI2_BYTES or sha(source) != AI2_SHA256:
        raise ValueError("Ai2 source identity mismatch")
    rows = pq.read_table(source).to_pylist()
    if len(rows) != AI2_ROWS:
        raise ValueError("Ai2 source row count mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    ref = json.loads(Path(source_reference).read_text())
    google_spec = next(s for s in ref["sources"] if s["key"] == "ifeval")
    google = load_source(google_spec, google_cache)
    for item in google:
        item["split"] = "test"
    google_exact = defaultdict(list)
    google_normal = defaultdict(list)
    for item in google:
        google_exact[item["task_text"]].append(item["item_id"])
        google_normal[normalize_prompt(item["task_text"])].append(item["item_id"])
    eligible, excluded, overlap = [], [], []
    task_available = 0
    manual_exclusions = manual_exclusions or {}
    for index, row in enumerate(rows):
        item_id = f"ai2-train:{index:06d}"
        reasons = []
        try:
            if len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
                raise ValueError("not_single_user_prompt")
            target = map_ground_truth(row)
        except (ValueError, KeyError, TypeError) as error:
            excluded.append({"item_id": item_id, "source_row_index": index, "reasons": [str(error)], "stage": "adapter_quarantine"})
            continue
        prompt = row["messages"][0]["content"]
        count = input_tokens(prompt, tokenizer)
        base = source_task_text(row)
        task_available += base is not None
        normalized = normalize_prompt(prompt)
        matches = {"exact_prompt": google_exact.get(prompt, []), "normalized_prompt": google_normal.get(normalized, []),
                   "derived_task_equals_google_prompt": google_normal.get(normalize_prompt(base), []) if base else []}
        if any(matches.values()):
            overlap.append({"practice_id": item_id, "matches": matches})
            reasons.append("practice_google_prompt_or_available_task_overlap")
        item = {"benchmark": "ifeval", "item_id": item_id, "dataset_revision": AI2_REVISION,
                "split": "development", "task_text": prompt, "target": target,
                "source": {"dataset_id": "allenai/RLVR-IFeval", "revision": AI2_REVISION, "split": "train", "row_id": str(index)},
                "metadata": {"source_file": AI2_PATH, "source_file_sha256": AI2_SHA256, "source_row_index": index,
                             "source_row_sha256": canonical_hash(row), "source_constraint_type": row["constraint_type"],
                             "source_ground_truth": json.loads(row["ground_truth"]), "source_constraint_template": row["constraint"],
                             "rendered_input_tokens": count, "derived_task_available": base is not None,
                             "normalized_task_sha256": canonical_hash(normalize_prompt(base or prompt))}}
        reasons += feasibility_reasons(item, count)
        reasons += manual_exclusions.get(item_id, [])
        if reasons:
            excluded.append({"item_id": item_id, "source_row_index": index, "reasons": sorted(set(reasons)), "stage": "feasibility_or_overlap"})
        else:
            eligible.append(validate_item(item))
    development = stratified_select(eligible, 32)
    evaluation_eligible, evaluation_excluded = [], []
    for item in google:
        count = input_tokens(item["task_text"], tokenizer)
        item["metadata"]["rendered_input_tokens"] = count
        reasons = feasibility_reasons(item, count) + manual_exclusions.get("google:" + item["item_id"], [])
        if reasons:
            evaluation_excluded.append({"item_id": item["item_id"], "reasons": sorted(set(reasons))})
        else:
            evaluation_eligible.append(item)
    evaluation = stratified_select(evaluation_eligible, 128)
    token_audit_identity = None
    if tokenizer_audit is not None:
        audit = json.loads(Path(tokenizer_audit).read_text())
        expected = {item["item_id"]: item["metadata"]["rendered_input_tokens"] for item in development + evaluation}
        observed = {item["item_id"]: item["rendered_input_tokens"] for item in audit["counts"]}
        if expected != observed or len(audit["counts"]) != 160 or audit["tokenizer_sha256"] != sha(tokenizer_path):
            raise ValueError("bundled tokenizer parity or selected item identity mismatch")
        if audit["engine_loads"] != 0 or audit["target_forward_passes"] != 0:
            raise ValueError("tokenizer preparation must not run model inference")
        token_audit_identity = {"sha256": sha(tokenizer_audit), "runtime_sha256": audit["runtime_sha256"],
                                "exact_selected_count_parity": True, "counts": len(expected)}
    elif publish:
        raise ValueError("exact bundled WebLLM tokenizer verification required before freeze")
    # Cross-source exact/normalized audit is made against all Google rows above,
    # not merely the selected evaluation subset.
    manifest = {"schema_version": 1, "kind": "independent_ifeval_study_data", "experiment_id": "ifeval-browser-v1",
                "study_role": "new independent practice development / IFEval evaluation subset",
                "old_partitions_modified": False, "selection_seed": 42,
                "selection_rule": "stable SHA256 canonical JSON {seed:42,item_id:ID} order; repeatedly select least-covered instruction type, prefer rare uncovered types, then mean coverage and seeded hash order; unique normalized available task; original order retained as selected",
                "selection_observed_model_outputs": False,
                "eligibility_policy": ELIGIBILITY_POLICY, "eligibility_policy_sha256": canonical_hash(ELIGIBILITY_POLICY),
                "adapter_sha256": sha(__file__), "tokenizer_sha256": sha(tokenizer_path),
                "template_segments": TEMPLATE_SEGMENTS, "template_sha256": canonical_hash(TEMPLATE_SEGMENTS),
                "bundled_tokenizer_audit": token_audit_identity,
                "python_version": list(sys.version_info[:3]), "preparation_packages": {"pyarrow": pyarrow.__version__, "tokenizers": importlib.metadata.version("tokenizers")},
                "sources": {"practice": {"dataset_id": "allenai/RLVR-IFeval", "revision": AI2_REVISION, "path": AI2_PATH, "sha256": AI2_SHA256, "bytes": AI2_BYTES, "rows": AI2_ROWS,
                                          "url": f"https://huggingface.co/datasets/allenai/RLVR-IFeval/resolve/{AI2_REVISION}/{AI2_PATH}"}, "evaluation": google_spec},
                "supported_mapping_count": len(MAPPINGS), "unsupported_functions": QUARANTINED_FUNCTIONS,
                "counts": {"practice_source": len(rows), "practice_eligible": len(eligible), "practice_excluded": len(excluded),
                           "evaluation_source": len(google), "evaluation_eligible": len(evaluation_eligible), "evaluation_excluded": len(evaluation_excluded)},
                "practice_exclusion_reasons": dict(sorted(Counter(r for entry in excluded for r in entry["reasons"]).items())),
                "evaluation_exclusion_reasons": dict(sorted(Counter(r for entry in evaluation_excluded for r in entry["reasons"]).items())),
                "development": {**family_coverage(development), "items_sha256": canonical_hash(development), "item_ids": [i["item_id"] for i in development]},
                "evaluation": {**family_coverage(evaluation), "items_sha256": canonical_hash(evaluation), "item_ids": [i["item_id"] for i in evaluation]},
                "overlap_audit": {"exact_and_normalized_prompt_audit": True, "derived_task_text_available_for_supported_practice": task_available,
                                  "matched_supported_practice_rows_excluded": len(overlap), "selected_prompt_overlap": 0,
                                  "source_task_ids_available": False, "pretraining_exposure_not_established": True},
                "evaluation_generations": 0, "manual_preoutput_exclusions": manual_exclusions,
                "instruction_types_shared_intentionally": True}
    manifest["sha256"] = canonical_hash(manifest)
    if not publish:
        return {"manifest": manifest, "development": development, "evaluation": evaluation}
    write_new(root / "development-items.json", development)
    write_new(root / "evaluation-items.json", evaluation)
    write_new(root / "exclusions.json", {"practice": excluded, "evaluation": evaluation_excluded})
    write_new(root / "overlap-private.json", {"matches": overlap, "limitations": ["No model pretraining exposure claim", "Ai2 does not expose original Tulu source task IDs", "Task comparison uses exact removal of recognized generated constraint only; other semantic paraphrases remain unverified"]})
    write_new(root / "source-manifest.json", {"schema_version": 1, "kind": "ifeval_independent_sources", "sources": manifest["sources"],
                                            "cached_ai2_licenses": {name: sha(root / "source" / name) for name in ("README.md", "ai2_impact_lr_license.md", "llama2_license.txt")},
                                            "source_attribution": ["Allen Institute for AI, Tulu 3 / RLVR-IFeval, ODC-BY dataset card and included upstream terms", "Google Research IFEval, Zhou et al., Apache-2.0"],
                                            "mapping_specification": MAPPINGS, "quarantined_functions": QUARANTINED_FUNCTIONS})
    write_new(root / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--google-cache", default="artifacts/phase04-benchmark-cache")
    parser.add_argument("--source-reference", default="configs/benchmark_sources.json")
    parser.add_argument("--manual-exclusions")
    parser.add_argument("--tokenizer-audit", required=True)
    args = parser.parse_args()
    exclusions = json.loads(Path(args.manual_exclusions).read_text()) if args.manual_exclusions else None
    result = prepare_data(args.root, args.tokenizer, args.google_cache, args.source_reference, manual_exclusions=exclusions, tokenizer_audit=args.tokenizer_audit)
    print(json.dumps({key: result[key] for key in ("sha256", "counts", "development", "evaluation")}, indent=2))


if __name__ == "__main__":
    main()
