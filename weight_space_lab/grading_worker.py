"""One bounded evaluator process. Dependencies remain optional to the project."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import random
import sys
import types

from .grading import IFEVAL_PACKAGES, MATH_PACKAGES, ifeval_metrics, verify_ifeval_assets


def _versions(packages):
    for name, version in packages.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("evaluator_dependency_version_mismatch")


def evaluate(payload):
    if payload["benchmark"] == "math":
        _versions(MATH_PACKAGES)
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig
        config = LatexExtractionConfig(boxed_match_priority=0, normalization_config=NormalizationConfig(
            basic_latex=True, units=True, malformed_operators=False, nits=False, boxed="all", equations=False))
        # Strict parsing, no string fallback and no search outside the selected box.
        options = dict(extraction_config=[config], fallback_mode="no_fallback", extraction_mode="first_match",
                       parsing_timeout=payload["timeout_seconds"], raise_on_error=True)
        gold = parse("$" + payload["target"]["answer"] + "$", **options)
        if not gold or any(isinstance(value, str) for value in gold):
            return {"status": "grader_error", "correct": None, "metrics": {}, "code": "unparseable_reference_answer"}
        prediction = parse("$" + payload["answer"] + "$", **options)
        if not prediction or any(isinstance(value, str) for value in prediction):
            return {"status": "graded", "correct": False, "metrics": {}, "code": "unparseable_prediction"}
        correct = bool(verify(gold, prediction, float_rounding=6, numeric_precision=15, strict=True,
                              allow_set_relation_comp=False, timeout_seconds=payload["timeout_seconds"], raise_on_error=True))
        return {"status": "graded", "correct": correct, "metrics": {}, "code": "verified"}
    if payload["benchmark"] == "ifeval":
        _versions(IFEVAL_PACKAGES)
        if payload["ifeval_root"] is None:
            raise ValueError("official_evaluator_assets_missing")
        root = Path(payload["ifeval_root"])
        verify_ifeval_assets(root)
        # Imports occur only after byte checks. NLTK is restricted to pinned data.
        root = root.resolve()
        sys.dont_write_bytecode = True
        package = types.ModuleType("instruction_following_eval")
        package.__path__ = [str(root / "instruction_following_eval")]
        sys.modules["instruction_following_eval"] = package
        import nltk
        nltk.data.path[:] = [str(root / "nltk_data")]
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        random.seed(0)
        from instruction_following_eval import evaluation_lib
        item = payload["item"]
        example = evaluation_lib.InputExample(key=item["item_id"], prompt=item["task_text"], **item["target"])
        responses = {example.prompt: payload["output"]}
        strict = evaluation_lib.test_instruction_following_strict(example, responses)
        loose = evaluation_lib.test_instruction_following_loose(example, responses)
        metrics = ifeval_metrics(strict.follow_instruction_list, loose.follow_instruction_list)
        return {"status": "graded", "correct": metrics["prompt_level_strict_accuracy"],
                "metrics": metrics, "code": "official_evaluator_completed"}
    raise ValueError("unsupported_benchmark")


def main():
    try:
        result = evaluate(json.load(sys.stdin))
    except importlib.metadata.PackageNotFoundError:
        result = {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_dependency_missing"}
    except (ImportError, ModuleNotFoundError):
        result = {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_import_failed"}
    except BaseException as error:
        # No source text, machine paths or arbitrary exception strings in evidence.
        code = "evaluator_timeout" if type(error).__name__ == "TimeoutException" else "evaluator_failed"
        result = {"status": "grader_timeout" if code == "evaluator_timeout" else "grader_error",
                  "correct": None, "metrics": {}, "code": code}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
