"""Versioned execution evidence, deliberately separate from compact grades.

Validation checks declarations and internal consistency, not external truth.
Raw provider exceptions and environment dumps are never schema fields.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping


RUN_PROFILES = {
    "native_development_smoke", "fixed_patch_evaluation", "fresh_editor_search",
    "frozen_final_evaluation", "browser_evaluation",
}
_OUTCOME_FIELDS = {"status", "raw_output", "finish_reason", "token_counts",
                   "duration_seconds", "error"}
_RUN_FIELDS = {
    "schema_version", "record_type", "run_id", "evidence_kind", "profile", "stage",
    "benchmark", "dataset_revision", "item_id", "split", "sample_index", "seed",
    "checkpoint", "configuration_sha256", "retry_policy", "attempts",
} | _OUTCOME_FIELDS
_STATUSES = {"ok", "truncated", "error", "timeout", "missing"}
_PLACEHOLDERS = {"unknown", "todo", "tbd", "latest", "main", "placeholder"}


def _object(value, keys, label):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{label} has missing or unknown fields")


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in _PLACEHOLDERS:
        raise ValueError(f"{label} must be a nonempty, non-placeholder string")


def _integer(value, label, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")


def _digest(value, label, *, revision=False, nullable=False):
    if value is None and nullable:
        return
    pattern = r"(?:[0-9a-f]{40}|[0-9a-f]{64})" if revision else r"[0-9a-f]{64}"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"{label} must be an immutable lowercase {'revision' if revision else 'SHA256 digest'}")


def _choice(value, choices, label):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} is invalid")


def _outcome(value):
    _choice(value["status"], _STATUSES, "status")
    status = value["status"]
    if value["raw_output"] is not None and not isinstance(value["raw_output"], str):
        raise ValueError("raw_output must be a string or null")
    _object(value["token_counts"], {"input", "output", "total"}, "token_counts")
    tokens = value["token_counts"]
    for count in tokens.values():
        if count is not None:
            _integer(count, "token count")
    if all(count is not None for count in tokens.values()) and tokens["total"] != tokens["input"] + tokens["output"]:
        raise ValueError("total tokens disagree with input and output")
    if tokens["total"] is not None and any(tokens[key] is not None and tokens[key] > tokens["total"] for key in ("input", "output")):
        raise ValueError("total tokens cannot be smaller than a known component")
    duration = value["duration_seconds"]
    if duration is not None:
        try:
            finite_duration = type(duration) in (int, float) and math.isfinite(duration) and duration >= 0
        except OverflowError:
            finite_duration = False
        if not finite_duration:
            raise ValueError("duration_seconds must be finite and nonnegative or null")
    reasons = {"ok": {"stop", "eos"}, "truncated": {"length"},
               "error": {"error"}, "timeout": {"timeout"}, "missing": {"not_started"}}
    _choice(value["finish_reason"], reasons[status], "finish_reason for status")
    if status in {"ok", "truncated"}:
        if value["raw_output"] is None or duration is None or value["error"] is not None:
            raise ValueError("completed attempts require raw output, measured duration and no error")
    elif status in {"error", "timeout"}:
        _object(value["error"], {"code", "retryable"}, "error")
        if not isinstance(value["error"]["code"], str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value["error"]["code"]) is None:
            raise ValueError("error.code must be a stable lowercase identifier, without provider text")
        if type(value["error"]["retryable"]) is not bool:
            raise ValueError("error.retryable must be boolean")
    else:
        if value["raw_output"] is not None or duration is not None or value["error"] is not None or any(v is not None for v in tokens.values()):
            raise ValueError("missing results must retain null output, duration, error and token measurements")


def validate_run_record(record):
    """Validate a final per-sample execution record, retaining every attempt.

    Unknown measurements use null, never fabricated zeros. A missing result
    may have unresolved checkpoint/dataset identity; completed results may not.
    Retries are permitted only after explicitly retryable errors or timeouts.
    """
    _object(record, _RUN_FIELDS, "run record")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("run record schema_version must be 1")
    if record["record_type"] != "run":
        raise ValueError("record_type must be run")
    for key in ("run_id", "benchmark", "item_id"):
        _text(record[key], key)
    _choice(record["evidence_kind"], {"fixture", "real_model"}, "evidence_kind")
    _choice(record["profile"], RUN_PROFILES, "profile")
    _choice(record["stage"], {"smoke", "pilot", "full"}, "stage")
    _choice(record["split"], {"development", "validation", "test"}, "split")
    if record["stage"] == "smoke" and record["split"] != "development":
        raise ValueError("smoke records must use development items")
    if record["profile"] == "native_development_smoke" and record["stage"] != "smoke":
        raise ValueError("native_development_smoke requires smoke stage")
    if record["profile"] != "native_development_smoke" and record["stage"] == "smoke":
        raise ValueError("smoke stage requires native_development_smoke profile")
    if record["profile"] == "fresh_editor_search" and record["split"] == "test":
        raise ValueError("fresh editor search may not contain final test items")
    if record["profile"] == "frozen_final_evaluation" and (record["stage"] != "full" or record["split"] != "test"):
        raise ValueError("frozen final records require full stage and test split")
    for key in ("sample_index", "seed"):
        _integer(record[key], key)
    _digest(record["configuration_sha256"], "configuration_sha256")
    completed = record["status"] in ("ok", "truncated")
    _digest(record["dataset_revision"], "dataset_revision", revision=True, nullable=not completed)
    _object(record["checkpoint"], {"model_id", "revision", "sha256"}, "checkpoint")
    checkpoint = record["checkpoint"]
    if checkpoint["model_id"] is not None or completed:
        _text(checkpoint["model_id"], "checkpoint.model_id")
    _digest(checkpoint["revision"], "checkpoint.revision", revision=True, nullable=not completed)
    _digest(checkpoint["sha256"], "checkpoint.sha256", nullable=not completed)
    _outcome(record)
    _object(record["retry_policy"], {"mode", "max_attempts"}, "retry_policy")
    retry = record["retry_policy"]
    _choice(retry["mode"], {"none", "retry_errors"}, "retry_policy.mode")
    _integer(retry["max_attempts"], "retry_policy.max_attempts", positive=True)
    if retry["mode"] == "none" and retry["max_attempts"] != 1:
        raise ValueError("no-retry policy must allow exactly one attempt")
    attempts = record["attempts"]
    if not isinstance(attempts, list) or len(attempts) > retry["max_attempts"]:
        raise ValueError("attempts must be a list within the declared maximum")
    if record["status"] == "missing":
        if attempts:
            raise ValueError("missing results may not have execution attempts")
    elif not attempts:
        raise ValueError("executed results require attempt history")
    for index, attempt in enumerate(attempts):
        _object(attempt, _OUTCOME_FIELDS | {"attempt_index"}, "attempt")
        _integer(attempt["attempt_index"], "attempt_index")
        if attempt["attempt_index"] != index:
            raise ValueError("attempt history must have contiguous zero-based indexes")
        _outcome(attempt)
        if attempt["status"] == "missing":
            raise ValueError("an execution attempt cannot be missing")
        if index < len(attempts) - 1 and (retry["mode"] != "retry_errors" or attempt["status"] not in {"error", "timeout"} or not attempt["error"]["retryable"]):
            raise ValueError("only declared retryable failures may precede another attempt")
    if attempts and any(record[key] != attempts[-1][key] for key in _OUTCOME_FIELDS):
        raise ValueError("record outcome must equal the final attempt; successful retries cannot be cherry-picked")
    return copy.deepcopy(dict(record))


def validate_run_records(records):
    """Validate an array and reject duplicate sample identities or seed aliases."""
    if not isinstance(records, list):
        raise ValueError("run records must be a JSON array")
    result, samples, seeds = [], set(), set()
    for record in records:
        checked = validate_run_record(record)
        identity = tuple(checked[key] for key in ("run_id", "benchmark", "item_id"))
        sample_key, seed_key = (*identity, checked["sample_index"]), (*identity, checked["seed"])
        if sample_key in samples or seed_key in seeds:
            raise ValueError("duplicate run/benchmark/item sample or seed")
        samples.add(sample_key)
        seeds.add(seed_key)
        result.append(checked)
    return result


def run_record_sha256(record):
    """Content identity used to bind independent grades to exact raw evidence."""
    checked = validate_run_record(record)
    return hashlib.sha256(json.dumps(checked, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def to_analysis_records(records, grades, *, benchmark, run_id):
    """Project one explicitly selected run and benchmark into compact grades.

    Grades are independent inputs with item_id, seed, correct, grader_revision,
    run_record_sha256 and optional score. No grade or successful retry is
    inferred. Retain these grade inputs and raw records with the analysis.
    """
    checked = validate_run_records(records)
    selected = [row for row in checked if row["benchmark"] == benchmark and row["run_id"] == run_id]
    if not selected:
        raise ValueError("selected run and benchmark contain no records")
    contexts = {(row["configuration_sha256"], row["evidence_kind"], row["stage"],
                 row["profile"], row["split"]) for row in selected}
    if len(contexts) != 1:
        raise ValueError("one analysis arm must have a single execution context")
    known_identities = [{row["dataset_revision"] for row in selected}]
    known_identities.extend({row["checkpoint"][key] for row in selected}
                            for key in ("model_id", "revision", "sha256"))
    if any(len(values - {None}) > 1 for values in known_identities):
        raise ValueError("one analysis arm cannot contain conflicting known dataset or checkpoint identities")
    sample_seeds, seed_samples = {}, {}
    for row in selected:
        sample, seed = row["sample_index"], row["seed"]
        if sample_seeds.setdefault(sample, seed) != seed or seed_samples.setdefault(seed, sample) != sample:
            raise ValueError("observed sample indexes and seeds must align within one benchmark")
    if not isinstance(grades, list):
        raise ValueError("independent grades must be a JSON array")
    indexed = {}
    grader_revisions = set()
    required = {"item_id", "seed", "correct", "grader_revision", "run_record_sha256"}
    for grade in grades:
        if not isinstance(grade, Mapping) or not required <= grade.keys() or set(grade) - required - {"score"}:
            raise ValueError("independent grade has missing or unknown fields")
        _text(grade["item_id"], "grade.item_id")
        _integer(grade["seed"], "grade.seed")
        _digest(grade["grader_revision"], "grader_revision", revision=True)
        grader_revisions.add(grade["grader_revision"])
        _digest(grade["run_record_sha256"], "run_record_sha256")
        key = grade["item_id"], grade["seed"]
        if key in indexed:
            raise ValueError("duplicate independent grade")
        indexed[key] = grade
    if len(grader_revisions) > 1:
        raise ValueError("one analysis arm must use a single pinned grader revision")
    result = []
    for row in selected:
        key = row["item_id"], row["seed"]
        compact = {"item_id": row["item_id"], "seed": row["seed"], "status": row["status"], "correct": False}
        if row["status"] in {"ok", "truncated"}:
            grade = indexed.pop(key, None)
            if grade is None or grade["run_record_sha256"] != run_record_sha256(row):
                raise ValueError("completed result requires an independent grade bound to its raw record")
            compact["correct"] = grade["correct"]
            if "score" in grade:
                compact["score"] = grade["score"]
        result.append(compact)
    if indexed:
        raise ValueError("grade references a failed or unselected result")
    from .evaluation import validate_records
    return validate_records(result)


_STATUS_KINDS = {"build": "build_log", "fixture_validation": "fixture_report",
                 "real_model_validation": "real_model_report", "experiment": "experiment_report"}


def empty_status_record():
    """Create independent, explicitly unexecuted status declarations."""
    return {"schema_version": 1, "record_type": "status", **{
        key: {"status": "not_started", "evidence": [], "reason": None, "outcome": None}
        for key in _STATUS_KINDS}}


def validate_status_record(record):
    """Validate separate progress tracks; fixture success never advances another."""
    _object(record, {"schema_version", "record_type", *_STATUS_KINDS}, "status record")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1 or record["record_type"] != "status":
        raise ValueError("status record requires schema_version 1 and record_type status")
    for key, kind in _STATUS_KINDS.items():
        track = record[key]
        _object(track, {"status", "evidence", "reason", "outcome"}, key)
        done = "completed" if key == "experiment" else "passed"
        _choice(track["status"], {"not_started", "in_progress", "failed", "blocked", done}, f"{key}.status")
        if track["reason"] is not None:
            _text(track["reason"], f"{key}.reason")
        if track["status"] in {"failed", "blocked"} and track["reason"] is None:
            raise ValueError("failed or blocked status requires an explicit reason")
        evidence = track["evidence"]
        if not isinstance(evidence, list):
            raise ValueError("status evidence must be a list")
        seen = set()
        for reference in evidence:
            _object(reference, {"kind", "sha256"}, "evidence reference")
            if reference["kind"] != kind:
                raise ValueError("status evidence kind must match its progress track")
            _digest(reference["sha256"], "evidence.sha256")
            if reference["sha256"] in seen:
                raise ValueError("duplicate status evidence")
            seen.add(reference["sha256"])
        if track["status"] in {done, "failed"} and not evidence:
            raise ValueError("completed or failed checks require typed evidence references")
        if track["status"] == "not_started" and evidence:
            raise ValueError("unexecuted status cannot contain evidence")
        if key == "experiment" and track["status"] == "completed":
            _choice(track["outcome"], {"positive", "negative", "inconclusive"}, "experiment.outcome")
        elif track["outcome"] is not None:
            raise ValueError("only a completed experiment declares a scientific outcome")
    return copy.deepcopy(dict(record))
