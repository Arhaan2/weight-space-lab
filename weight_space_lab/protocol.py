"""Validate study configuration before expensive or final evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_PINS = (
    "model_revision", "tokenizer_revision", "runtime_version",
    "grader_revision", "development_dataset_revision",
    "validation_dataset_revision", "test_dataset_revision",
    "editor_model_version", "chat_template_sha256",
)
REVISION_PINS = (
    "model_revision", "tokenizer_revision", "grader_revision",
    "development_dataset_revision", "validation_dataset_revision",
    "test_dataset_revision",
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def validate_protocol(config: Mapping[str, Any], *, require_ready: bool = False) -> dict:
    """Return explicit blockers; a design template cannot pass run readiness."""
    issues = []
    if not isinstance(config, Mapping):
        raise ValueError("protocol must be an object")
    if type(config.get("schema_version")) is not int or config.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if config.get("claim_type") not in {"informed_replication", "extension"}:
        issues.append("claim_type must distinguish informed_replication and extension")
    if config.get("editor_knows_published_mechanism") is not True:
        issues.append("this design must disclose the published mechanism was already read")
    if config.get("no_training") is not True:
        issues.append("no_training must be true for this study")
    sampling = config.get("sampling", {})
    if not isinstance(sampling, Mapping):
        sampling = {}
    for key in ("temperature", "top_p", "presence_penalty"):
        value = sampling.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            issues.append(f"sampling.{key} must be finite")
    if type(sampling.get("temperature")) in (int, float) and sampling["temperature"] <= 0:
        issues.append("sampling.temperature must be positive")
    if type(sampling.get("top_p")) in (int, float) and not 0 < sampling["top_p"] <= 1:
        issues.append("sampling.top_p must be in (0, 1]")
    for key in ("top_k", "max_output_tokens", "samples_per_item"):
        if not _positive_int(sampling.get(key)):
            issues.append(f"sampling.{key} must be a positive integer")
    if type(sampling.get("thinking")) is not bool:
        issues.append("sampling.thinking must be explicit")
    limits = config.get("probe_limits", {})
    if not isinstance(limits, Mapping):
        limits = {}
    for key in ("max_calls", "max_sequences_per_call", "max_total_generated_tokens"):
        if not _positive_int(limits.get(key)):
            issues.append(f"probe_limits.{key} must be a positive integer")
    fraction = config.get("max_changed_fraction")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        issues.append("max_changed_fraction must be in (0, 1]")
    pins = config.get("pins", {})
    if not isinstance(pins, Mapping):
        pins = {}
    unpinned = [key for key in REQUIRED_PINS if not isinstance(pins.get(key), str) or not pins[key].strip()]
    bad_pins = []
    for key in REVISION_PINS:
        value = pins.get(key)
        if value and not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", str(value)):
            bad_pins.append(key)
    template_hash = pins.get("chat_template_sha256")
    if template_hash and not re.fullmatch(r"[0-9a-f]{64}", str(template_hash)):
        bad_pins.append("chat_template_sha256")
    for key in ("runtime_version", "editor_model_version"):
        if str(pins.get(key, "")).strip().lower() in {"main", "latest", "unknown", "todo", "tbd"}:
            bad_pins.append(key)
    ready_issues = list(issues)
    ready_issues.extend(f"missing pin: {key}" for key in unpinned)
    ready_issues.extend(f"pin is not immutable: {key}" for key in bad_pins)
    if config.get("frozen") is not True:
        ready_issues.append("protocol is not frozen")
    # This checks configuration completeness, not external truth or isolation.
    result = {
        "valid_design": not issues,
        "run_configuration_complete": not ready_issues,
        "issues": ready_issues if require_ready else issues,
        "pending_run_requirements": ready_issues,
        "protocol_sha256": canonical_hash(config),
        "verification_scope": "configuration only; revisions and runtime must be verified separately",
    }
    return result


def validate_split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Reject cross-split repeated identifiers or normalized task text.

    Exact matching does not prove semantic deduplication. Keep separate audits
    for paraphrases, template families, and contamination inherited from training.
    """
    if not rows:
        raise ValueError("split manifest must not be empty")
    seen_ids: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    counts = {"development": 0, "validation": 0, "test": 0}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each split row must be an object")
        split = row.get("split")
        item_id, task_text = row.get("item_id"), row.get("task_text")
        if split not in counts:
            raise ValueError("unrecognized split")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be a nonempty string")
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("task_text must be a nonempty string")
        if item_id in seen_ids:
            raise ValueError("duplicate item_id in split manifest")
        task_hash = hashlib.sha256(" ".join(task_text.split()).encode("utf-8")).hexdigest()
        if task_hash in seen_text and seen_text[task_hash] != split:
            raise ValueError("task text overlaps across splits")
        seen_ids[item_id] = split
        seen_text[task_hash] = split
        counts[split] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("development, validation, and test must all be nonempty")
    return {"counts": counts, "manifest_sha256": canonical_hash(list(rows))}
