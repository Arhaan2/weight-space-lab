"""Reconcile durable completions immediately before a new browser dispatch gate.

Arhaan2. This is a separately versioned repair for the event-snapshot/intent-file
race. It preserves the original generation reconciliation and never reserves a
request, writes an acknowledgement, starts an engine, sleeps, or retries work.
The caller retains its frozen schedule, authorization, deadline, and cleanup.
"""
from __future__ import annotations

from collections.abc import MutableMapping
import math
import time

from .ifeval_baseline import read_events, reconcile
from .protocol import canonical_hash


def _schedule(config):
    items = config.get("items")
    if (not isinstance(items, list) or not items
            or any(not isinstance(item, dict) or not isinstance(item.get("item_id"), str)
                   for item in items)
            or len({item["item_id"] for item in items}) != len(items)):
        raise ValueError("a nonempty unique frozen browser schedule is required")
    return ["__warmup__", *[item["item_id"] for item in items]]


def _state(handled, seen_samples, size):
    if (not isinstance(handled, set) or any(type(i) is not int for i in handled)
            or (handled and handled != set(range(max(handled) + 1)))
            or any(i < 0 or i > size for i in handled)):
        raise ValueError("acknowledged dispatches must form one complete prefix")
    if (not isinstance(seen_samples, MutableMapping)
            or any(type(i) is not int or not 1 <= i <= size
                   or not isinstance(pin, str) or len(pin) != 64
                   for i, pin in seen_samples.items())
            or (seen_samples and set(seen_samples) != set(range(1, max(seen_samples) + 1)))):
        raise ValueError("reconciled samples require a complete prefix of content hashes")


def _runtime_failure(event):
    kind, data = event["type"], event.get("data")
    if kind in {"driver-error", "driver-gate-error", "failure-classification"}:
        raise ValueError("durable browser runtime failure; no further dispatch")
    if kind == "event" and isinstance(data, dict) and data.get("type") == "device-lost":
        raise ValueError("gpu_device_lost")
    if kind == "done" and (not isinstance(data, dict) or data.get("status") != "complete"):
        raise ValueError("browser session stopped or failed; no further dispatch")


def _validate_sample(event, expected_index, schedule):
    sample = event.get("data")
    warmup = expected_index == 1
    cap = 16 if warmup else 1024
    if not isinstance(sample, dict):
        raise ValueError("malformed durable completion event")
    tokens, duration = sample.get("generated_tokens"), sample.get("response_latency_ms")
    if (event["type"] != ("warmup-sample" if warmup else "task-sample")
            or expected_index > len(schedule)
            or sample.get("request_id") != f"generation-{expected_index}"
            or sample.get("item_id") != schedule[expected_index - 1]
            or sample.get("warmup") is not warmup
            or not isinstance(sample.get("raw_output"), str)
            or type(tokens) is not int or not 0 <= tokens <= cap
            or type(duration) not in (int, float) or not math.isfinite(duration)
            or not 0 <= duration <= 120000
            or sample.get("finish_reason") not in {"stop", "eos", "length"}
            or sample.get("execution_error") is not None
            or sample.get("cancelled", False) is not False
            or sample.get("local_webgpu_inference_observed") is not True
            or type(sample.get("webgpu_submitted_batches")) is not int
            or sample["webgpu_submitted_batches"] < 1):
        raise ValueError("duplicate, out-of-order, malformed, or failed durable completion")
    return sample


def _validate_ledger_sample(row, sample, index):
    cap = 16 if index == 1 else 1024
    if (row.get("kind") != "probe" or row.get("retry_of") is not None
            or row.get("attempt_index") != 0 or row.get("requested_sequences") != 1
            or row.get("max_generated_tokens") != cap or row.get("timeout_seconds") != 120
            or row.get("status") not in {"dispatched", "completed"}):
        raise ValueError("completion does not bind one original dispatched reservation")
    if row["status"] == "completed" and (
            row.get("completed_sequences") != 1 or row.get("failed_sequences") != 0
            or row.get("observed_generated_tokens") != sample["generated_tokens"]
            or row.get("observed_elapsed_seconds") != sample["response_latency_ms"] / 1000
            or row.get("observed_billed_units") != 0):
        raise ValueError("completion conflicts with previously reconciled ledger evidence")


def reconcile_events(target, book, config, handled, seen_samples, *, on_observed=None):
    """Validate one current durable event prefix and reconcile each sample once.

    ``seen_samples`` is a mutable ``{local_sequence_index: canonical_sample_hash}``
    map, initially empty. A warm-up is local sequence 1; scored requests follow
    the supplied config's item order. Previously read completions are validated
    again, but are not finished or reported twice. A partial trailing JSONL line
    remains withheld by the preserved ``read_events`` implementation.

    The whole snapshot is checked before any completion is reconciled. Unknown,
    duplicate, conflicting, missing-prefix, failed, or unacknowledged samples
    stop execution. This function performs no reservation or gate approval.
    """
    schedule = _schedule(config)
    _state(handled, seen_samples, len(schedule))
    book.check_active()
    events = read_events(target)
    attempts = {row["request_id"]: row for row in book.attempts()}
    if set(attempts) != {f"generation-{i}" for i in handled if i > 0}:
        raise ValueError("acknowledgement prefix and durable reservation set differ")
    samples = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("malformed durable browser event")
        _runtime_failure(event)
        if event["type"] not in {"warmup-sample", "task-sample"}:
            continue
        index = len(samples) + 1
        sample = _validate_sample(event, index, schedule)
        if index not in handled:
            raise ValueError("completion without an acknowledged dispatch")
        _validate_ledger_sample(attempts[f"generation-{index}"], sample, index)
        digest = canonical_hash(sample)
        if index in seen_samples and seen_samples[index] != digest:
            raise ValueError("previously reconciled completion changed")
        samples.append((index, sample, digest))
    if set(seen_samples) - {index for index, _, _ in samples}:
        raise ValueError("durable completion prefix was removed or truncated")
    for index, sample, digest in samples:
        if index in seen_samples:
            continue
        # The existing reconciliation is idempotent for a terminal ledger row.
        # Its values were independently bound above before that early return.
        reconcile(book, sample, index, config)
        seen_samples[index] = digest
        if on_observed is not None:
            on_observed(index, sample)
    for row in book.attempts():
        if row["status"] in {"reserved", "dispatched"}:
            book.check_active(row["request_id"])
    return events


def assert_predecessor(index, target, book, config, handled, seen_samples, grant,
                       *, on_observed=None):
    """Reread completions at the gate boundary before testing its predecessor.

    One immediate reread closes the original stale-snapshot race: a completion
    durably written before its following intent is included. A genuinely absent
    completion still fails immediately. The caller must separately validate the
    intent identity and retain its existing time reserve before reserving/ACKing.
    No already acknowledged or reserved request may pass this check again.
    """
    schedule = _schedule(config)
    if type(index) is not int or not 1 <= index <= len(schedule):
        raise ValueError("request falls outside the frozen local schedule")
    if not isinstance(grant, dict):
        raise ValueError("engine load grant required before generation")
    now, mono = time.time() * 1000, time.monotonic()
    wall_end, mono_end = grant.get("deadline_epoch_ms"), grant.get("deadline_monotonic")
    if (type(wall_end) not in (int, float) or not math.isfinite(wall_end)
            or type(mono_end) not in (int, float) or not math.isfinite(mono_end)
            or now >= wall_end or mono >= mono_end):
        raise TimeoutError("generation gate deadline expired or invalid")
    if index in handled or any(row["request_id"] == f"generation-{index}" for row in book.attempts()):
        raise ValueError("request already acknowledged or reserved; replay forbidden")
    events = reconcile_events(target, book, config, handled, seen_samples, on_observed=on_observed)
    if index - 1 not in handled or (index > 1 and index - 1 not in seen_samples):
        raise ValueError("requests must execute sequentially after a durable predecessor completion")
    return events
