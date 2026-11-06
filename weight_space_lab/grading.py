"""Independent, pinned answer grading with bounded evaluator subprocesses.

Grading diagnostics are evidence, never compact paired-analysis inputs.
No model inference, network access, retry, or dependency installation occurs
while grading. Source acquisition is an explicit separate operation.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from decimal import Decimal, InvalidOperation

from .protocol import canonical_hash
from .records import run_record_sha256, validate_run_record


IFEVAL_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
IFEVAL_FILES = {
    "instruction_following_eval/evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "instruction_following_eval/instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instruction_following_eval/instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instruction_following_eval/instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
NLTK_DATA_REVISION = "4f15a3d89eefe9748ec1c05be495d91289197155"
NLTK_ARCHIVE_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
MATH_PACKAGES = {"math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0",
                 "sympy": "1.14.0", "antlr4-python3-runtime": "4.13.2", "mpmath": "1.3.0"}
IFEVAL_PACKAGES = {"absl-py": "2.5.1", "langdetect": "1.0.9", "nltk": "3.10.4",
                   "immutabledict": "4.3.1", "six": "1.17.0", "defusedxml": "0.7.1",
                   "click": "8.5.1", "joblib": "1.6.0", "regex": "2026.11.7",
                   "tqdm": "4.70.1", "cloudpickle": "3.1.2"}
_BENCHMARKS = {"math", "gsm8k", "ifeval"}
_POLICY_FIELDS = {"schema_version", "reasoning_policy", "output_eos_tokens", "truncation_policy",
                  "generation_retry", "grader_timeout_seconds", "grader_retry"}
_GRADE_FIELDS = {"schema_version", "record_type", "benchmark", "item_id", "seed", "sample_index",
                 "evidence_kind", "run_record_sha256", "item_sha256", "policy_sha256", "grader",
                 "status", "correct", "metrics", "diagnostics"}


def _hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_grading_policy(policy):
    """Require output interpretation, truncation and retry rules before grading."""
    if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
        raise ValueError("grading policy has missing or unknown fields")
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("grading policy schema_version must be 1")
    if policy["reasoning_policy"] not in ("plain", "think_tags", "qwen_thinking"):
        raise ValueError("invalid reasoning policy")
    tokens = policy["output_eos_tokens"]
    if not isinstance(tokens, list) or any(t not in ("<|im_end|>", "<|endoftext|>") for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("output_eos_tokens must explicitly list supported unique token strings")
    if policy["truncation_policy"] not in ("grade", "fail"):
        raise ValueError("invalid truncation policy")
    seconds = policy["grader_timeout_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 120:
        raise ValueError("grader timeout must be an integer from 1 through 120 seconds")
    retry = policy["generation_retry"]
    if not isinstance(retry, dict) or set(retry) != {"mode", "max_attempts"}:
        raise ValueError("generation retry policy has missing or unknown fields")
    if retry["mode"] not in ("none", "retry_errors") or type(retry["max_attempts"]) is not int or not 1 <= retry["max_attempts"] <= 10:
        raise ValueError("invalid bounded generation retry policy")
    if retry["mode"] == "none" and retry["max_attempts"] != 1:
        raise ValueError("no-retry generation policy requires one attempt")
    if policy["grader_retry"] != {"mode": "none", "max_attempts": 1} or type(policy["grader_retry"].get("max_attempts")) is not int:
        raise ValueError("grader retries are disabled; repair failures in a separately versioned grading run")
    return copy.deepcopy(policy)


def grader_identity(benchmark):
    """Identify adapter source, upstream code, numerical dependencies and policy version."""
    if benchmark not in _BENCHMARKS:
        raise ValueError("unsupported benchmark")
    names = {"math": "math_verify", "gsm8k": "gsm8k_decimal", "ifeval": "official_ifeval"}
    pins = MATH_PACKAGES if benchmark == "math" else IFEVAL_PACKAGES if benchmark == "ifeval" else {}
    identity = {"adapter_version": 1, "benchmark": benchmark, "packages": pins,
                "python_version": list(sys.version_info[:3]),
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "worker_sha256": hashlib.sha256(Path(__file__).with_name("grading_worker.py").read_bytes()).hexdigest()}
    if benchmark == "ifeval":
        identity.update(upstream_revision=IFEVAL_REVISION, files=IFEVAL_FILES,
                        nltk_data_revision=NLTK_DATA_REVISION, nltk_archive_sha256=NLTK_ARCHIVE_SHA256,
                        evaluator_random_seed=0)
    return {"name": names[benchmark], "revision": canonical_hash(identity)}


def separate_output(raw_output, policy):
    """Return (reasoning, final output, code), with no implicit token cleanup.

    qwen_thinking accepts a pre-opened prompt think block; think_tags accepts
    no block or one complete leading block. Malformed/unclosed blocks never
    expose reasoning to an answer grader. Whitespace in final output remains.
    """
    validate_grading_policy(policy)
    if not isinstance(raw_output, str):
        raise ValueError("raw output must be text")
    text = raw_output
    while True:
        token = next((t for t in policy["output_eos_tokens"] if text.rstrip().endswith(t)), None)
        if token is None:
            break
        text = text.rstrip()[:-len(token)]
    mode = policy["reasoning_policy"]
    if mode == "plain":
        return "", text, "output_ready"
    opening, closing = text.count("<think>"), text.count("</think>")
    if mode == "think_tags" and opening == closing == 0:
        return "", text, "output_ready"
    if closing != 1 or opening > 1 or (mode == "think_tags" and opening != 1):
        return text, "", "malformed_reasoning_boundary"
    before, after = text.split("</think>")
    if opening and not before.lstrip().startswith("<think>"):
        return text, "", "malformed_reasoning_boundary"
    if "<think>" in after:
        return text, "", "malformed_reasoning_boundary"
    reasoning = before.lstrip()[len("<think>"):] if opening else before
    return reasoning, after, "output_ready"


def extract_last_boxed(text):
    """Select the last balanced boxed expression, never a favorable earlier one."""
    pattern, position, answer = re.compile(r"\\boxed\s*\{"), 0, None
    while (marker := pattern.search(text, position)) is not None:
        start, depth = marker.end(), 1
        for index in range(start, len(text)):
            character = text[index]
            if character not in "{}":
                continue
            # A brace preceded by an odd number of backslashes is literal.
            backslashes, pos = 0, index - 1
            while pos >= 0 and text[pos] == "\\":
                backslashes += 1
                pos -= 1
            if backslashes % 2:
                continue
            depth += 1 if character == "{" else -1
            if depth == 0:
                answer = text[start:index].strip()
                if pattern.search(answer) is not None:
                    return None, "malformed_final_answer"
                position = index + 1
                break
        else:
            return None, "malformed_final_answer"
    if answer is None:
        return None, "missing_final_answer"
    return (answer, "answer_extracted") if answer else (None, "empty_final_answer")


_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")


def extract_gsm8k_answer(text):
    """Last #### marker, otherwise last box; no last-number or prose fallback."""
    if "####" in text:
        value = text.rsplit("####", 1)[1].strip()
    else:
        value, code = extract_last_boxed(text)
        if value is None:
            return None, code
    if _NUMBER.fullmatch(value) is None:
        return None, "malformed_final_answer"
    try:
        return str(Decimal(value.replace(",", ""))), "answer_extracted"
    except InvalidOperation:
        return None, "malformed_final_answer"


def _worker(payload, timeout_seconds):
    try:
        result = subprocess.run([sys.executable, "-m", "weight_space_lab.grading_worker"],
                                input=json.dumps(payload, allow_nan=False), text=True,
                                capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "grader_timeout", "correct": None, "metrics": {}, "code": "evaluator_timeout"}
    except OSError:
        return {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_process_start_failed"}
    except UnicodeError:
        return {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_invalid_response"}
    if result.returncode != 0:
        return {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_process_failed"}
    try:
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or set(response) != {"status", "correct", "metrics", "code"}:
            raise ValueError
        if response["status"] not in ("graded", "grader_error", "grader_timeout"):
            raise ValueError
        if (response["status"] == "graded" and type(response["correct"]) is not bool) or (response["status"] != "graded" and response["correct"] is not None):
            raise ValueError
        if not isinstance(response["metrics"], dict) or not isinstance(response["code"], str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", response["code"]) is None:
            raise ValueError
        return response
    except (ValueError, TypeError):
        return {"status": "grader_error", "correct": None, "metrics": {}, "code": "evaluator_invalid_response"}


def grade_record(record, item, policy, *, ifeval_root=None):
    """Grade one exact raw record against one canonical independent target."""
    record = validate_run_record(record)
    policy = validate_grading_policy(policy)
    from .benchmark_data import validate_item
    validate_item(item)
    if any(record[key] != item[key] for key in ("benchmark", "item_id", "split")):
        raise ValueError("run record and grading item identities disagree")
    if record["dataset_revision"] is not None and record["dataset_revision"] != item["dataset_revision"]:
        raise ValueError("run record and grading dataset revisions disagree")
    if record["retry_policy"] != policy["generation_retry"]:
        raise ValueError("record retries disagree with the frozen grading policy")
    grade = {"schema_version": 1, "record_type": "grade", **{k: record[k] for k in
             ("benchmark", "item_id", "seed", "sample_index", "evidence_kind")},
             "run_record_sha256": run_record_sha256(record), "item_sha256": canonical_hash(item),
             "policy_sha256": canonical_hash(policy), "grader": grader_identity(record["benchmark"]),
             "status": "not_applicable", "correct": None, "metrics": {},
             "diagnostics": {"code": "model_failure", "extracted_answer": None,
                             "output_sha256": None, "reasoning_sha256": None}}
    if record["status"] not in ("ok", "truncated"):
        return validate_grade_record(grade)
    used_worker = False
    reasoning, output, code = separate_output(record["raw_output"], policy)
    grade["diagnostics"].update(output_sha256=_hash_text(output), reasoning_sha256=_hash_text(reasoning))
    if record["status"] == "truncated" and policy["truncation_policy"] == "fail":
        result = {"status": "graded", "correct": False, "metrics": {}, "code": "truncation_policy_failure"}
    elif code != "output_ready":
        result = {"status": "graded", "correct": False, "metrics": {}, "code": code}
    elif item["benchmark"] == "gsm8k":
        answer, code = extract_gsm8k_answer(output)
        grade["diagnostics"]["extracted_answer"] = answer
        gold = item["target"]["answer"].strip().replace(",", "")
        try:
            if _NUMBER.fullmatch(item["target"]["answer"].strip()) is None:
                raise InvalidOperation
            result = {"status": "graded", "correct": answer is not None and Decimal(answer) == Decimal(gold),
                      "metrics": {}, "code": code}
        except InvalidOperation:
            result = {"status": "grader_error", "correct": None, "metrics": {}, "code": "invalid_reference_answer"}
    elif item["benchmark"] == "math":
        answer, code = extract_last_boxed(output)
        grade["diagnostics"]["extracted_answer"] = answer
        if answer is None:
            result = {"status": "graded", "correct": False, "metrics": {}, "code": code}
        else:
            used_worker = True
            result = _worker({"benchmark": "math", "answer": answer, "target": item["target"],
                              "timeout_seconds": policy["grader_timeout_seconds"]}, policy["grader_timeout_seconds"])
    else:
        used_worker = True
        result = _worker({"benchmark": "ifeval", "output": output, "item": item,
                          "ifeval_root": str(ifeval_root) if ifeval_root is not None else None}, policy["grader_timeout_seconds"])
    grade.update({key: result[key] for key in ("status", "correct", "metrics")})
    grade["diagnostics"]["code"] = result["code"]
    if grade["status"] == "graded" and item["benchmark"] == "ifeval" and not grade["metrics"] and not used_worker:
        count = len(item["target"]["instruction_id_list"])
        grade["metrics"] = ifeval_metrics([False] * count, [False] * count)
    try:
        return validate_grade_record(grade)
    except (ValueError, TypeError):
        if not used_worker:
            raise
        grade.update(status="grader_error", correct=None, metrics={})
        grade["diagnostics"]["code"] = "evaluator_invalid_response"
        return validate_grade_record(grade)


def ifeval_metrics(strict, loose):
    if not isinstance(strict, list) or not isinstance(loose, list) or not strict or len(strict) != len(loose) or any(type(v) is not bool for v in strict + loose):
        raise ValueError("invalid official instruction outcomes")
    return {"prompt_level_strict_accuracy": all(strict), "prompt_level_loose_accuracy": all(loose),
            "instruction_level_strict_accuracy": sum(strict) / len(strict),
            "instruction_level_loose_accuracy": sum(loose) / len(loose),
            "instruction_count": len(strict), "instruction_strict_correct": sum(strict),
            "instruction_loose_correct": sum(loose), "instruction_strict": strict, "instruction_loose": loose}


def validate_grade_record(grade):
    """Validate diagnostics independently of their raw evidence and target join."""
    if not isinstance(grade, dict) or set(grade) != _GRADE_FIELDS:
        raise ValueError("grade record has missing or unknown fields")
    if type(grade["schema_version"]) is not int or grade["schema_version"] != 1 or grade["record_type"] != "grade":
        raise ValueError("invalid grade record version or type")
    if grade["benchmark"] not in _BENCHMARKS or grade["evidence_kind"] not in ("fixture", "real_model"):
        raise ValueError("invalid grade benchmark or evidence kind")
    if not isinstance(grade["item_id"], str) or not grade["item_id"].strip():
        raise ValueError("grade item_id must be nonempty text")
    for key in ("seed", "sample_index"):
        if type(grade[key]) is not int or grade[key] < 0:
            raise ValueError("grade sample and seed must be nonnegative integers")
    for key in ("run_record_sha256", "item_sha256", "policy_sha256"):
        if not isinstance(grade[key], str) or re.fullmatch(r"[a-f0-9]{64}", grade[key]) is None:
            raise ValueError("invalid grading evidence digest")
    if grade["grader"] != grader_identity(grade["benchmark"]):
        raise ValueError("grade does not use the current pinned grader identity")
    if grade["status"] not in ("graded", "grader_error", "grader_timeout", "not_applicable"):
        raise ValueError("invalid grading status")
    if grade["status"] == "graded":
        if type(grade["correct"]) is not bool:
            raise ValueError("graded correctness must be a boolean")
    elif grade["correct"] is not None:
        raise ValueError("unresolved grading correctness must remain null")
    diagnostic = grade["diagnostics"]
    if not isinstance(diagnostic, dict) or set(diagnostic) != {"code", "extracted_answer", "output_sha256", "reasoning_sha256"}:
        raise ValueError("invalid grading diagnostics")
    if not isinstance(diagnostic["code"], str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", diagnostic["code"]) is None:
        raise ValueError("grading code must be a stable identifier")
    if diagnostic["extracted_answer"] is not None and not isinstance(diagnostic["extracted_answer"], str):
        raise ValueError("extracted answer must be text or null")
    for key in ("output_sha256", "reasoning_sha256"):
        value = diagnostic[key]
        if value is not None and (not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None):
            raise ValueError("invalid output separation digest")
    if not isinstance(grade["metrics"], dict):
        raise ValueError("grading metrics must be an object")
    if grade["status"] == "graded" and grade["benchmark"] == "ifeval":
        metrics = grade["metrics"]
        for key in ("prompt_level_strict_accuracy", "prompt_level_loose_accuracy"):
            if type(metrics.get(key)) is not bool:
                raise ValueError("prompt accuracy must be boolean per response")
        for key in ("instruction_count", "instruction_strict_correct", "instruction_loose_correct"):
            if type(metrics.get(key)) is not int or metrics[key] < 0:
                raise ValueError("instruction counts must be nonnegative integers")
        for key in ("instruction_level_strict_accuracy", "instruction_level_loose_accuracy"):
            if type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
                raise ValueError("instruction accuracy must be a finite ratio")
        try:
            expected = ifeval_metrics(metrics["instruction_strict"], metrics["instruction_loose"])
        except (KeyError, TypeError) as error:
            raise ValueError("missing official instruction outcomes") from error
        if metrics != expected or grade["correct"] != metrics["prompt_level_strict_accuracy"]:
            raise ValueError("IFEval metrics disagree with instruction outcomes")
    elif grade["metrics"]:
        raise ValueError("unexpected metrics for this benchmark or grading status")
    return copy.deepcopy(grade)


def verify_ifeval_assets(root):
    """Reject modified official evaluator source or tokenizer resources."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("evaluator assets must be a real directory")
    for relative, digest in IFEVAL_FILES.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("official evaluator source does not match its pin")
    archive = root / "punkt_tab.zip"
    if not archive.is_file() or archive.is_symlink() or hashlib.sha256(archive.read_bytes()).hexdigest() != NLTK_ARCHIVE_SHA256:
        raise ValueError("official sentence-tokenizer archive does not match its pin")
    with zipfile.ZipFile(archive) as source:
        allowed = set(IFEVAL_FILES) | {"punkt_tab.zip", "manifest.json"}
        for member in source.namelist():
            if member.startswith("punkt_tab/english/") and not member.endswith("/"):
                path = root / "nltk_data/tokenizers" / member
                allowed.add(path.relative_to(root).as_posix())
                if path.is_symlink() or not path.is_file() or path.read_bytes() != source.read(member):
                    raise ValueError("sentence-tokenizer files do not match the pinned archive")
        for path in root.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.relative_to(root).as_posix() not in allowed):
                raise ValueError("unexpected or linked evaluator asset")
    return {"schema_version": 1, "kind": "official_ifeval_assets", "revision": IFEVAL_REVISION,
            "files": dict(IFEVAL_FILES), "nltk_data_revision": NLTK_DATA_REVISION,
            "nltk_archive_sha256": NLTK_ARCHIVE_SHA256}


def acquire_ifeval(out):
    """Explicit bounded source acquisition; preserve upstream files and license."""
    out = Path(out)
    if out.exists():
        raise ValueError("evaluator output exists; choose a new directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".evaluator-", dir=out.parent))
    try:
        deadline = time.monotonic() + 300
        entries = [(path, f"https://raw.githubusercontent.com/google-research/google-research/{IFEVAL_REVISION}/{path}", digest)
                   for path, digest in IFEVAL_FILES.items()]
        entries.append(("punkt_tab.zip", f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/packages/tokenizers/punkt_tab.zip", NLTK_ARCHIVE_SHA256))
        total = 0
        for relative, url, digest in entries:
            if time.monotonic() >= deadline:
                raise ValueError("evaluator acquisition elapsed budget exceeded")
            with urllib.request.urlopen(url, timeout=60) as response:
                parts, size = [], 0
                while size <= 6_000_000:
                    if time.monotonic() >= deadline:
                        raise ValueError("evaluator acquisition elapsed budget exceeded")
                    part = response.read(min(64 * 1024, 6_000_001 - size))
                    if not part:
                        break
                    parts.append(part)
                    size += len(part)
                data = b"".join(parts)
            total += len(data)
            if len(data) > 6_000_000 or total > 15_000_000 or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("evaluator download exceeded bounds or failed its content pin")
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        with zipfile.ZipFile(io.BytesIO((stage / "punkt_tab.zip").read_bytes())) as archive:
            for member in archive.namelist():
                if member.startswith("punkt_tab/english/") and not member.endswith("/"):
                    path = stage / "nltk_data/tokenizers" / member
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(member))
        result = verify_ifeval_assets(stage)
        (stage / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        stage.rename(out)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _known_answer_item(benchmark="math", target=None):
    """Hand-authored test identity, never a downloaded benchmark item."""
    revision = hashlib.sha1(b"independent known-answer fixture").hexdigest()
    return {"benchmark": benchmark, "item_id": "known-answer-fixture", "dataset_revision": revision,
            "split": "development", "task_text": "Independent known-answer contract fixture.",
            "target": target or {"answer": r"\frac{1}{2}"}, "metadata": {"evidence_kind": "fixture"},
            "source": {"dataset_id": "independent-known-answers", "revision": revision,
                       "split": "fixture", "row_id": "known-answer-fixture"}}


def _known_answer_record(item, output, status="ok"):
    outcome = {"status": status, "raw_output": output,
               "finish_reason": "length" if status == "truncated" else "eos",
               "token_counts": {"input": None, "output": None, "total": None},
               "duration_seconds": 0.0, "error": None}
    return {"schema_version": 1, "record_type": "run", "run_id": "known-answer-fixtures",
            "evidence_kind": "fixture", "profile": "native_development_smoke", "stage": "smoke",
            **{k: item[k] for k in ("benchmark", "dataset_revision", "item_id", "split")},
            "sample_index": 0, "seed": 0,
            "checkpoint": {"model_id": "hand-authored-fixture", "revision": item["dataset_revision"],
                           "sha256": _hash_text("no model checkpoint: hand-authored outputs")},
            "configuration_sha256": _hash_text("independent known-answer fixtures"),
            "retry_policy": {"mode": "none", "max_attempts": 1},
            "attempts": [dict(outcome, attempt_index=0)], **outcome}


def run_known_answer_checks(*, ifeval_root=None):
    """Execute independent fixtures through actual pinned optional evaluators.

    Missing dependencies or assets are failed checks, never reported as passes.
    This function loads no language model and evaluates no benchmark outputs.
    """
    policy = {"schema_version": 1, "reasoning_policy": "plain", "output_eos_tokens": [],
              "truncation_policy": "grade", "generation_retry": {"mode": "none", "max_attempts": 1},
              "grader_timeout_seconds": 20, "grader_retry": {"mode": "none", "max_attempts": 1}}
    cases = [
        ("math_equivalent_fraction", "math", r"\boxed{2/4}", {"answer": r"\frac{1}{2}"}, True, "graded"),
        ("math_equivalent_symbolic", "math", r"\boxed{(x+1)^2}", {"answer": "x^2+2x+1"}, True, "graded"),
        ("math_missing", "math", "I cannot finish.", {"answer": "2"}, False, "graded"),
        ("math_last_box_wins", "math", r"\boxed{2} then \boxed{3}", {"answer": "2"}, False, "graded"),
        ("math_last_box_equivalent", "math", r"\boxed{3} then \boxed{4/2}", {"answer": "2"}, True, "graded"),
        ("math_malformed_box", "math", r"\boxed{\frac{2}{", {"answer": "2"}, False, "graded"),
        ("math_malformed_expression", "math", r"\boxed{\frac12}", {"answer": "2"}, False, "graded"),
        ("math_balanced_malformed_expression", "math", r"\boxed{\frac{2}{}}", {"answer": "2"}, False, "graded"),
        ("math_json_not_answer", "math", '{"answer":2}', {"answer": "2"}, False, "graded"),
        ("math_invalid_reference", "math", r"\boxed{2}", {"answer": r"\frac{"}, None, "grader_error"),
        ("gsm_decimal_equivalence", "gsm8k", "Work.\n#### 1,200.00", {"answer": "1200"}, True, "graded"),
        ("gsm_last_marker", "gsm8k", "#### 2\n#### 3", {"answer": "2"}, False, "graded"),
        ("gsm_no_last_number_fallback", "gsm8k", "I calculated 2", {"answer": "2"}, False, "graded"),
        ("gsm_malformed", "gsm8k", "#### 12,34", {"answer": "1234"}, False, "graded"),
        ("gsm_box", "gsm8k", r"\boxed{-2.50}", {"answer": "-2.5"}, True, "graded"),
        ("ifeval_json_and_keyword", "ifeval", '{"required":true}',
         {"instruction_id_list": ["detectable_format:json_format", "keywords:existence"], "kwargs": [{}, {"keywords": ["required"]}]}, True, "graded"),
        ("ifeval_valid_json_insufficient", "ifeval", '{"wrong":true}',
         {"instruction_id_list": ["detectable_format:json_format", "keywords:existence"], "kwargs": [{}, {"keywords": ["required"]}]}, False, "graded"),
        ("ifeval_strict_loose_difference", "ifeval", 'Here it is:\n{"answer":2}\nThanks!',
         {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]}, False, "graded"),
        ("ifeval_pinned_sentences", "ifeval", "One sentence. Another sentence.",
         {"instruction_id_list": ["length_constraints:number_sentences"], "kwargs": [{"num_sentences": 2, "relation": "at least"}]}, True, "graded"),
        ("ifeval_evaluator_failure", "ifeval", "An answer.",
         {"instruction_id_list": ["nonexistent:instruction"], "kwargs": [{}]}, None, "grader_error"),
    ]
    checks = []
    for name, benchmark, output, target, expected, status in cases:
        item = _known_answer_item(benchmark, target)
        result = grade_record(_known_answer_record(item, output), item, policy, ifeval_root=ifeval_root)
        passed = result["correct"] is expected and result["status"] == status
        if name == "ifeval_strict_loose_difference":
            passed = passed and result["metrics"].get("prompt_level_loose_accuracy") is True
        if name == "ifeval_valid_json_insufficient":
            passed = passed and result["metrics"].get("instruction_strict") == [True, False]
        checks.append({"case": name, "passed": passed, "expected_correct": expected,
                       "expected_status": status, "grade": result})
    item = _known_answer_item("gsm8k", {"answer": "2"})
    for name, output, status, changes, correct in [
        ("truncated_grade", "#### 2", "truncated", {}, True),
        ("truncated_fail", "#### 2", "truncated", {"truncation_policy": "fail"}, False),
        ("reasoning_not_answer", "<think>#### 2</think>Unfinished", "ok", {"reasoning_policy": "think_tags"}, False),
        ("qwen_preopened_think", "work</think>\n#### 2<|im_end|>", "ok", {"reasoning_policy": "qwen_thinking", "output_eos_tokens": ["<|im_end|>"]}, True),
        ("unclosed_reasoning", "<think>#### 2", "truncated", {"reasoning_policy": "think_tags"}, False),
    ]:
        result = grade_record(_known_answer_record(item, output, status), item, dict(policy, **changes))
        checks.append({"case": name, "passed": result["status"] == "graded" and result["correct"] is correct,
                       "expected_correct": correct, "expected_status": "graded", "grade": result})
    return {"schema_version": 1, "kind": "independent_grader_known_answers", "evidence_kind": "fixture",
            "scientific_result": False, "language_model_evaluated": False, "experiment_executed": False,
            "all_passed": all(c["passed"] for c in checks), "checks": checks}
