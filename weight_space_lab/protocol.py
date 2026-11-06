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


PROFILES = (
    "artifact_acquisition", "published_patch_reconstruction",
    "native_development_smoke", "fixed_patch_evaluation",
    "fresh_editor_search", "frozen_final_evaluation", "browser_evaluation",
)
_NATIVE_PINS = ("model_revision", "tokenizer_revision", "runtime_version",
                "chat_template_sha256", "checkpoint_sha256")
_PROFILE_PINS = {
    "artifact_acquisition": ("source_revision", "model_revision"),
    "published_patch_reconstruction": ("source_revision", "model_revision",
        "checkpoint_sha256", "manifest_sha256", "archive_sha256"),
    "native_development_smoke": _NATIVE_PINS,
    "fixed_patch_evaluation": _NATIVE_PINS + ("source_revision", "patch_sha256",
        "reconstruction_audit_sha256", "evaluation_code_revision"),
    "fresh_editor_search": _NATIVE_PINS + ("editor_model_version", "split_audit_sha256"),
    "frozen_final_evaluation": _NATIVE_PINS + ("evaluation_code_revision",),
    "browser_evaluation": _NATIVE_PINS + ("evaluation_code_revision",
        "benchmark_revision", "converter_revision", "native_checkpoint_sha256",
        "conversion_audit_sha256", "identity_validation_sha256"),
}
_PLACEHOLDERS = {"main", "latest", "unknown", "todo", "tbd", "placeholder",
                 "unresolved", "pending", "none", "null", "n/a"}


def _known_text(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip())
            and value.strip().lower() not in _PLACEHOLDERS)


def _digest(value: Any, *, revision: bool = False) -> bool:
    pattern = r"(?:[0-9a-f]{40}|[0-9a-f]{64})" if revision else r"[0-9a-f]{64}"
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def frozen_configuration_hash(config: Mapping[str, Any]) -> str:
    """Hash all configuration except the freeze record to avoid self-reference.

    Set ``frozen`` to true before calculating this digest. The full protocol
    digest returned by validate_protocol also includes the freeze record.
    """
    return canonical_hash({key: value for key, value in config.items() if key != "freeze"})


def build_benchmark_schedule(benchmark: str, item_ids: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    """Expand one benchmark's explicit seed list, without choosing defaults."""
    if not _known_text(benchmark):
        raise ValueError("benchmark must be a nonempty, resolved string")
    if not isinstance(item_ids, list) or not item_ids or any(not _known_text(item) for item in item_ids):
        raise ValueError("item_ids must be a nonempty string list")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must be unique within a benchmark")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty list of unique nonnegative integers")
    return [{"benchmark": benchmark, "item_id": item, "sample_index": sample, "seed": seed}
            for item in item_ids for sample, seed in enumerate(seeds)]


def validate_benchmark_schedule(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Check equal explicit sample/seed schedules within each benchmark.

    Different benchmarks may have different repeat counts. Run paired analysis
    separately per benchmark after projecting item_id and seed; never merge
    same-named items from different benchmarks into the compact analysis format.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark schedule must be a nonempty list")
    groups: dict[str, dict[str, dict[int, int]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"benchmark", "item_id", "sample_index", "seed"}:
            raise ValueError("schedule row must contain benchmark, item_id, sample_index and seed")
        if not _known_text(row["benchmark"]) or not _known_text(row["item_id"]):
            raise ValueError("benchmark and item_id must be nonempty, resolved strings")
        for name in ("sample_index", "seed"):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError("sample_index and seed must be nonnegative integers")
        samples = groups.setdefault(row["benchmark"], {}).setdefault(row["item_id"], {})
        if row["sample_index"] in samples or row["seed"] in samples.values():
            raise ValueError("duplicate sample or seed for a benchmark item")
        samples[row["sample_index"]] = row["seed"]
    summary = {}
    for benchmark, items in groups.items():
        reference = next(iter(items.values()))
        if set(reference) != set(range(len(reference))):
            raise ValueError("sample_index must be contiguous and start at zero")
        if any(samples != reference for samples in items.values()):
            raise ValueError("all items in a benchmark must share the same sample/seed schedule")
        summary[benchmark] = {"items": len(items), "samples_per_item": len(reference),
                              "scheduled_records": len(items) * len(reference)}
    ordered = sorted((dict(row) for row in rows),
                     key=lambda row: (row["benchmark"], row["item_id"], row["sample_index"]))
    return {"schema_version": 1, "benchmarks": summary, "schedule_sha256": canonical_hash(ordered)}


def _validate_sampling(sampling, label, issues, pending):
    if sampling is None:
        pending.append(f"missing requirement: {label}")
        return
    if not isinstance(sampling, Mapping):
        issues.append(f"{label} must be an object or null")
        return
    allowed = {"mode", "temperature", "top_p", "top_k", "presence_penalty",
               "thinking", "max_output_tokens"}
    if set(sampling) - allowed:
        issues.append(f"{label} has unsupported decoding fields")
    mode = sampling.get("mode")
    if mode not in ("stochastic", "greedy"):
        issues.append(f"{label}.mode must be stochastic or greedy")
    for name in ("temperature", "top_p", "presence_penalty"):
        value = sampling.get(name)
        if not _finite(value):
            issues.append(f"{label}.{name} must be finite")
    temperature, top_p = sampling.get("temperature"), sampling.get("top_p")
    if _finite(temperature) and (temperature != 0 if mode == "greedy" else temperature <= 0):
        issues.append(f"{label}.temperature must be zero for greedy and positive for stochastic")
    if _finite(top_p) and not 0 < top_p <= 1:
        issues.append(f"{label}.top_p must be in (0, 1]")
    top_k = sampling.get("top_k")
    if type(top_k) is not int or top_k < 0:
        issues.append(f"{label}.top_k must be a nonnegative integer; zero disables filtering")
    if mode == "greedy" and (top_p != 1 or top_k != 0):
        issues.append(f"{label}: greedy requires top_p=1 and top_k=0 to disable sampling filters")
    if not _positive_int(sampling.get("max_output_tokens")):
        issues.append(f"{label}.max_output_tokens must be a positive integer")
    if type(sampling.get("thinking")) is not bool:
        issues.append(f"{label}.thinking must be explicit")
    if "samples_per_item" in sampling:
        issues.append(f"{label}: use benchmark seeds to declare samples, not samples_per_item")


def validate_protocol(config: Mapping[str, Any], *, require_ready: bool = False) -> dict:
    """Validate declared profile requirements, never infer execution success.

    Version 1 retains its original behavior. Version 2 distinguishes missing
    execution information (null/missing values) from malformed declarations.
    Syntactically complete pins still require external verification.
    """
    if not isinstance(config, Mapping):
        raise ValueError("protocol must be an object")
    if type(config.get("schema_version")) is int and config["schema_version"] == 1:
        return _validate_legacy_protocol(config, require_ready=require_ready)
    issues, pending = [], []
    if type(config.get("schema_version")) is not int or config.get("schema_version") != 2:
        issues.append("schema_version must be 1 or 2")
    profile = config.get("profile")
    if not isinstance(profile, str) or profile not in PROFILES:
        issues.append("profile must name a supported execution profile")
        profile = None
    stage = config.get("stage")
    if stage not in ("preparation", "smoke", "pilot", "full"):
        issues.append("stage must be preparation, smoke, pilot or full")
    preparation = profile in ("artifact_acquisition", "published_patch_reconstruction")
    if preparation and stage != "preparation":
        issues.append("artifact profiles require stage preparation")
    if profile == "native_development_smoke" and stage != "smoke":
        issues.append("native_development_smoke requires stage smoke")
    if profile in ("fixed_patch_evaluation", "fresh_editor_search", "browser_evaluation") and stage not in ("pilot", "full"):
        issues.append("this profile requires stage pilot or full")
    if profile == "frozen_final_evaluation" and stage != "full":
        issues.append("frozen_final_evaluation requires stage full")
    claims = {
        "artifact_acquisition": ("artifact_preparation",),
        "published_patch_reconstruction": ("published_patch_replay",),
        "native_development_smoke": ("development_smoke",),
        "fixed_patch_evaluation": ("published_patch_replay",),
        "fresh_editor_search": ("informed_replication", "extension"),
        "frozen_final_evaluation": ("informed_replication", "extension", "published_patch_replay", "baseline_evaluation"),
        "browser_evaluation": ("extension",),
    }
    if profile and config.get("claim_type") not in claims[profile]:
        issues.append("claim_type does not match the selected profile")

    def need(value, label, check=_known_text):
        if value is None:
            pending.append(f"missing requirement: {label}")
        elif not check(value):
            issues.append(f"invalid requirement: {label}")

    pins = config.get("pins", {})
    if not isinstance(pins, Mapping):
        issues.append("pins must be an object")
        pins = {}
    required_pins = list(_PROFILE_PINS.get(profile, ()))
    origin = config.get("intervention_origin")
    if profile in ("frozen_final_evaluation", "browser_evaluation"):
        if origin not in ("fresh_editor", "published_patch", "unchanged"):
            issues.append("intervention_origin must be fresh_editor, published_patch or unchanged")
        if origin == "published_patch":
            required_pins += ["source_revision", "patch_sha256", "reconstruction_audit_sha256"]
        if profile == "frozen_final_evaluation":
            origin_claims = {"fresh_editor": ("informed_replication", "extension"),
                             "published_patch": ("published_patch_replay",),
                             "unchanged": ("baseline_evaluation",)}
            if isinstance(origin, str) and origin in origin_claims and config.get("claim_type") not in origin_claims[origin]:
                issues.append("claim_type does not match intervention_origin")
    editor_required = profile == "fresh_editor_search" or (
        profile in ("frozen_final_evaluation", "browser_evaluation") and origin == "fresh_editor")
    if editor_required:
        required_pins.append("editor_model_version")
        if config.get("editor_knows_published_mechanism") is not True:
            issues.append("this design must disclose the published mechanism was already read")
        if config.get("no_training") is not True:
            issues.append("no_training must be true for this study")
        editor = config.get("editor_provenance")
        if editor is None:
            editor = {}
        if not isinstance(editor, Mapping):
            issues.append("editor_provenance must be an object or null")
            editor = {}
        for name in ("model_id", "interface"):
            need(editor.get(name), f"editor_provenance.{name}")
        if profile != "fresh_editor_search":
            need(editor.get("record_sha256"), "editor_provenance.record_sha256", _digest)
    for name in dict.fromkeys(required_pins):
        value = pins.get(name)
        if value is None or value == "":
            pending.append(f"missing pin: {name}")
        elif ((name.endswith("_revision") and not _digest(value, revision=True))
              or (name.endswith("_sha256") and not _digest(value))
              or not _known_text(value)):
            issues.append(f"pin is not immutable: {name}")
    need(config.get("model"), "model")
    if preparation:
        source = config.get("source")
        if source is None:
            source = {}
        if not isinstance(source, Mapping):
            issues.append("source must be an object or null")
            source = {}
        for name in ("repository", "manifest_path", "archive_path"):
            need(source.get(name), f"source.{name}")
    elif profile:
        for name in ("backend", "precision"):
            need(config.get(name), name)
        limits = config.get("execution_limits")
        if limits is None:
            limits = {}
        if not isinstance(limits, Mapping):
            issues.append("execution_limits must be an object or null")
            limits = {}
        for name in ("max_sequences", "max_duration_seconds"):
            need(limits.get(name), f"execution_limits.{name}", _positive_int)
        plans = config.get("benchmarks")
        if plans is None:
            pending.append("missing requirement: benchmarks")
            plans = []
        elif not isinstance(plans, list) or not plans:
            issues.append("benchmarks must be a nonempty list or null")
            plans = []
        names, splits, all_rows, seen_items = set(), set(), [], set()
        complete_schedule = bool(plans)
        for index, plan in enumerate(plans):
            label = f"benchmarks[{index}]"
            if not isinstance(plan, Mapping):
                issues.append(f"{label} must be an object")
                complete_schedule = False
                continue
            name, split = plan.get("benchmark"), plan.get("split")
            if not _known_text(name):
                issues.append(f"{label}.benchmark must be a nonempty resolved string")
            if split not in ("development", "validation", "test"):
                issues.append(f"{label}.split must be development, validation or test")
            else:
                splits.add(split)
                if _known_text(name):
                    if (name, split) in names:
                        issues.append("benchmark/split plan pairs must be unique")
                    names.add((name, split))
            if profile == "native_development_smoke" and split != "development":
                issues.append("native smoke tasks must use the development split")
            if profile == "fresh_editor_search" and split == "test":
                issues.append("fresh-editor search cannot schedule final-test tasks")
            if profile == "frozen_final_evaluation" and split != "test":
                issues.append("frozen final evaluation requires test tasks")
            need(plan.get("dataset_revision"), f"{label}.dataset_revision", lambda v: _digest(v, revision=True))
            need(plan.get("task_manifest_sha256"), f"{label}.task_manifest_sha256", _digest)
            if profile != "native_development_smoke":
                need(plan.get("grader_revision"), f"{label}.grader_revision", lambda v: _digest(v, revision=True))
                need(plan.get("truncation_policy"), f"{label}.truncation_policy", lambda v: v in ("grade", "fail"))
            _validate_sampling(plan.get("sampling"), f"{label}.sampling", issues, pending)
            if plan.get("item_ids") is None or plan.get("seeds") is None:
                for field in ("item_ids", "seeds"):
                    if plan.get(field) is None:
                        pending.append(f"missing requirement: {label}.{field}")
                complete_schedule = False
            else:
                try:
                    rows = build_benchmark_schedule(name, plan["item_ids"], plan["seeds"])
                    plan_items = {(name, item) for item in plan["item_ids"]}
                    if plan_items & seen_items:
                        issues.append("benchmark item identities overlap across split plans")
                    seen_items.update(plan_items)
                    all_rows.extend(rows)
                except ValueError as error:
                    issues.append(f"{label}: {error}")
                    complete_schedule = False
        if profile == "fresh_editor_search" and splits != {"development", "validation"}:
            issues.append("fresh-editor search requires development and validation benchmark plans")
        if complete_schedule and _positive_int(limits.get("max_sequences")) and len(all_rows) > limits["max_sequences"]:
            issues.append("scheduled sequences exceed execution_limits.max_sequences")
        schedule_hash = None
        if complete_schedule:
            try:
                schedule_hash = validate_benchmark_schedule(all_rows)["schedule_sha256"]
            except ValueError as error:
                issues.append(str(error))
        if profile == "fresh_editor_search":
            probe = config.get("probe_limits", {})
            if not isinstance(probe, Mapping):
                probe = {}
                issues.append("probe_limits must be an object")
            for name in ("max_calls", "max_sequences_per_call", "max_total_generated_tokens"):
                need(probe.get(name), f"probe_limits.{name}", _positive_int)
            need(config.get("max_changed_fraction"), "max_changed_fraction",
                 lambda v: _finite(v) and 0 < v <= 1)
        if profile == "browser_evaluation":
            browser = config.get("browser", {})
            if not isinstance(browser, Mapping):
                browser = {}
                issues.append("browser must be an object")
            for name in ("version", "operating_system", "hardware", "quantization", "session_plan_sha256"):
                need(browser.get(name), f"browser.{name}", _digest if name.endswith("sha256") else _known_text)
        freeze_required = (profile == "frozen_final_evaluation"
                           or (profile == "browser_evaluation" and "test" in splits)
                           or (profile == "fixed_patch_evaluation" and stage == "full" and "test" in splits))
        if freeze_required:
            if config.get("frozen") is not True:
                pending.append("protocol is not frozen")
            freeze = config.get("freeze", {})
            if not isinstance(freeze, Mapping):
                freeze = {}
                issues.append("freeze must be an object")
            expected = {"configuration_sha256": frozen_configuration_hash(config),
                        "checkpoint_sha256": pins.get("checkpoint_sha256"),
                        "evaluation_code_revision": pins.get("evaluation_code_revision")}
            if schedule_hash is not None:
                expected["schedule_sha256"] = schedule_hash
            for name in ("configuration_sha256", "checkpoint_sha256", "schedule_sha256", "evaluation_code_revision"):
                value = freeze.get(name)
                need(value, f"freeze.{name}", lambda v, n=name: _digest(v, revision=n.endswith("revision")))
                if value is not None and name in expected and value != expected[name]:
                    issues.append(f"freeze.{name} does not match the current configuration")
    requirements = issues + pending
    return {"schema_version": 2, "profile": profile, "stage": stage,
            "valid_design": not issues, "run_configuration_complete": not requirements,
            "issues": requirements if require_ready else issues,
            "pending_run_requirements": requirements, "required_pins": list(dict.fromkeys(required_pins)),
            "protocol_sha256": canonical_hash(config),
            "verification_scope": "configuration only; artifacts, access, isolation and runtime must be verified separately"}


def _validate_legacy_protocol(config: Mapping[str, Any], *, require_ready: bool = False) -> dict:
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
