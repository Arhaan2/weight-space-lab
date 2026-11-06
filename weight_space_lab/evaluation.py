"""Strict paired evaluation against an explicit case-and-seed schedule.

Public API::

    validate_records(records) -> list[dict]
    compare_paired(baseline, candidate, *, schedule=None, metric="accuracy",
                   truncation_policy="grade", bootstrap_samples=10000,
                   confidence=0.95, seed=0) -> dict

Each record has item_id (nonempty string), seed (nonnegative integer), status
("ok", "truncated", "error", "timeout", or "missing"), and correct (boolean).
Correct is required for ok/truncated, and must be absent or false otherwise.
An optional score in [0, 1] allows graded tasks. Other keys are rejected.
Schedule entries have exactly item_id and seed. Supplying a schedule makes
absent records explicit failures. Without a schedule, observed key sets must
match exactly; mutually absent cases cannot be inferred.

Accuracy uses correct, score uses score (required for successful/truncated
records), and completion_within_budget uses correct only for status ok.
For accuracy and score, truncation_policy="grade" honors the returned grade;
"fail" forces truncated outputs to zero. Error, timeout and missing always
score zero. Item means give each item equal weight across repeated seeds.
Confidence intervals resample complete items, retaining their seed groups.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


_STATUSES = {"ok", "truncated", "error", "timeout", "missing"}


def _key(row: Mapping[str, Any]) -> tuple[str, int]:
    item = row.get("item_id")
    seed = row.get("seed")
    if not isinstance(item, str) or not item.strip():
        raise ValueError("item_id must be a nonempty string")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer, not a boolean")
    return item, int(seed)


def validate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and copy records without coercing truthy values into grades."""
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of record objects")
    try:
        source = list(records)
    except TypeError as error:
        raise ValueError("records must be an iterable of record objects") from error
    seen = set()
    validated = []
    for row in source:
        if not isinstance(row, Mapping) or not {"item_id", "seed", "status"} <= row.keys():
            raise ValueError("record must contain item_id, seed and status")
        if set(row) - {"item_id", "seed", "status", "correct", "score"}:
            raise ValueError("record has unknown keys")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate item/seed record: {key!r}")
        seen.add(key)
        status = row["status"]
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("record status is invalid")
        if status in {"ok", "truncated"} and "correct" not in row:
            raise ValueError("ok and truncated records require a boolean correct field")
        correct = row.get("correct", False)
        if type(correct) is not bool:
            raise ValueError("correct must be a JSON boolean")
        if status in {"error", "timeout", "missing"} and correct:
            raise ValueError("failed records cannot be marked correct")
        result = {"item_id": key[0], "seed": key[1], "status": status, "correct": correct}
        if "score" in row:
            score = row["score"]
            if isinstance(score, (bool, np.bool_)) or not isinstance(score, numbers.Real):
                raise ValueError("score must be a finite number between 0 and 1")
            score = float(score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("score must be a finite number between 0 and 1")
            if status in {"error", "timeout", "missing"} and score != 0.0:
                raise ValueError("failed records cannot have a nonzero score")
            result["score"] = score
        validated.append(result)
    return validated


def _schedule_keys(schedule: Iterable[Mapping[str, Any]]) -> set[tuple[str, int]]:
    if isinstance(schedule, (str, bytes, Mapping)):
        raise ValueError("schedule must be an iterable of case objects")
    try:
        entries = list(schedule)
    except TypeError as error:
        raise ValueError("schedule must be an iterable of case objects") from error
    keys = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"item_id", "seed"}:
            raise ValueError("schedule entries must have exactly item_id and seed")
        key = _key(entry)
        if key in keys:
            raise ValueError(f"duplicate item/seed schedule entry: {key!r}")
        keys.add(key)
    if not keys:
        raise ValueError("schedule must not be empty")
    return keys


def compare_paired(
    baseline: Iterable[Mapping[str, Any]],
    candidate: Iterable[Mapping[str, Any]],
    *,
    schedule: Iterable[Mapping[str, Any]] | None = None,
    metric: str = "accuracy",
    truncation_policy: str = "grade",
    bootstrap_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare item means and bootstrap paired deltas by resampling items.

    ``paired_flips`` counts improved, regressed and tied item means, rather
    than treating stochastic repeats as independent benchmark items. Counts
    of individual binary correctness flips are reported separately. The
    confidence interval is descriptive for one preselected comparison; it
    does not account for searching over candidate edits.
    """
    if metric not in {"accuracy", "score", "completion_within_budget"}:
        raise ValueError("metric must be accuracy, score or completion_within_budget")
    if truncation_policy not in {"grade", "fail"}:
        raise ValueError("truncation_policy must be grade or fail")
    if isinstance(bootstrap_samples, (bool, np.bool_)) or not isinstance(bootstrap_samples, numbers.Integral) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, numbers.Real):
        raise ValueError("confidence must be a finite number between 0 and 1")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a finite number between 0 and 1")
    arms = [validate_records(baseline), validate_records(candidate)]
    indexed = [{(row["item_id"], row["seed"]): row for row in rows} for rows in arms]
    observed_sets = [set(index) for index in indexed]
    if schedule is None:
        if observed_sets[0] != observed_sets[1]:
            raise ValueError("arms have different item/seed sets; provide the predeclared schedule")
        keys = observed_sets[0]
        if not keys:
            raise ValueError("comparison requires at least one scheduled case")
        schedule_source = "matching observed cases; mutually absent cases are unknown"
    else:
        keys = _schedule_keys(schedule)
        if any(observed - keys for observed in observed_sets):
            raise ValueError("arm contains item/seed records outside the schedule")
        schedule_source = "explicit schedule"
    ordered_keys = sorted(keys)
    completed = []
    for index in indexed:
        rows = []
        for item, case_seed in ordered_keys:
            rows.append(index.get((item, case_seed), {
                "item_id": item, "seed": case_seed, "status": "missing", "correct": False
            }))
        completed.append(rows)

    def grade(row: Mapping[str, Any]) -> float:
        if row["status"] in {"error", "timeout", "missing"}:
            return 0.0
        if row["status"] == "truncated" and (truncation_policy == "fail" or metric == "completion_within_budget"):
            return 0.0
        if metric == "score":
            if "score" not in row:
                raise ValueError("score metric requires score for every graded record")
            return row["score"]
        return float(row["correct"])

    case_scores = [np.array([grade(row) for row in rows], dtype=np.float64) for rows in completed]
    positions = defaultdict(list)
    for position, (item, _) in enumerate(ordered_keys):
        positions[item].append(position)
    per_item = []
    for item in sorted(positions):
        offsets = positions[item]
        base_mean = float(case_scores[0][offsets].mean())
        candidate_mean = float(case_scores[1][offsets].mean())
        per_item.append({
            "item_id": item, "scheduled_repeats": len(offsets),
            "baseline_mean": base_mean, "candidate_mean": candidate_mean,
            "delta": candidate_mean - base_mean,
        })
    deltas = np.array([row["delta"] for row in per_item], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bootstrapped = np.empty(int(bootstrap_samples), dtype=np.float64)
    # Bound temporary index matrices, including on large benchmarks.
    batch_size = max(1, min(1024, 1_000_000 // len(deltas)))
    for start in range(0, int(bootstrap_samples), batch_size):
        count = min(batch_size, int(bootstrap_samples) - start)
        indexes = rng.integers(0, len(deltas), size=(count, len(deltas)))
        bootstrapped[start:start + count] = deltas[indexes].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrapped, [tail, 1.0 - tail])
    summaries = []
    for arm_index, rows in enumerate(completed):
        counts = Counter(row["status"] for row in rows)
        summaries.append({
            "mean": float(np.mean([row["baseline_mean" if arm_index == 0 else "candidate_mean"] for row in per_item])),
            "case_mean": float(case_scores[arm_index].mean()),
            "observed_records": len(indexed[arm_index]),
            "scheduled_records": len(keys),
            "imputed_missing_records": len(keys - observed_sets[arm_index]),
            "status_counts": {status: counts[status] for status in sorted(_STATUSES)},
            "completion_rate": counts["ok"] / len(keys),
        })
    case_delta = case_scores[1] - case_scores[0]
    report = {
        "schema_version": 1, "metric": metric,
        "truncation_policy": "fail" if metric == "completion_within_budget" else truncation_policy,
        "schedule_source": schedule_source, "scheduled_cases": len(keys),
        "items": len(per_item), "weighting": "equal item weight; mean of scheduled repeats",
        "baseline": summaries[0], "candidate": summaries[1],
        "delta": float(deltas.mean()),
        "confidence_interval": {
            "low": float(low), "high": float(high), "level": confidence,
            "method": "paired percentile bootstrap clustered by item",
            "resamples": int(bootstrap_samples), "seed": int(seed),
            "informative": len(per_item) > 1,
        },
        "paired_flips": {
            "basis": "item means", "improved": int(np.count_nonzero(deltas > 0)),
            "regressed": int(np.count_nonzero(deltas < 0)), "tied": int(np.count_nonzero(deltas == 0)),
        },
        "case_changes": {
            "basis": "scheduled item/seed scores",
            "improved": int(np.count_nonzero(case_delta > 0)),
            "regressed": int(np.count_nonzero(case_delta < 0)),
            "tied": int(np.count_nonzero(case_delta == 0)),
        },
        "per_item": per_item,
        "uncertainty_scope": "one preselected comparison; candidate-selection uncertainty is not included",
    }
    return report
