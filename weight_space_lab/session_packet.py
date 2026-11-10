"""Private paste-ready packet for the explicitly manual exploratory handoff.

This renders approved inputs and a response grammar, never a proposed edit.
Callers keep its result outside public artifacts. It launches no session and
cannot authenticate the model, establish isolation or cancel remote generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re

from .editor_contract import _exact, _json_copy
from .protocol import canonical_hash
from .session_protocol import validate_session_request


def _inspection_projection(inspections, request):
    """Accept only bounded statistics from approved tensor slices."""
    inspections = _json_copy(inspections)
    info = request["approved_context"]["model_info"]["allowed_tensors"]
    policy = request["approved_context"]["policy"]["inspection"]
    if type(inspections) is not list or not 1 <= len(inspections) <= min(16, policy["max_requests"]):
        raise ValueError("packet requires a bounded list of approved inspections")
    scanned_total = 0
    for entry in inspections:
        _exact(entry, {"request", "result"}, "packet inspection")
        query, result = entry["request"], entry["result"]
        _exact(query, {"op", "tensor", "slices"}, "packet inspection request")
        if query["op"] != "statistics" or query["tensor"] not in info:
            raise ValueError("packet accepts statistics of approved tensors only")
        if type(query["slices"]) is not list or len(query["slices"]) != 2:
            raise ValueError("packet inspection must declare both matrix slice ranges")
        shape = []
        for span, size in zip(query["slices"], info[query["tensor"]]["shape"]):
            if (type(span) is not list or len(span) != 2
                    or any(type(value) is not int for value in span)
                    or not 0 <= span[0] < span[1] <= size):
                raise ValueError("packet inspection slice is invalid")
            shape.append(span[1] - span[0])
        count = math.prod(shape)
        scanned_total += count
        if count > policy["max_scanned_coordinates"] or scanned_total > policy["max_total_scanned_coordinates"]:
            raise ValueError("packet inspection exceeds the existing exposure policy")
        _exact(result, {"tensor", "dtype", "shape", "statistics", "statistics_precision"}, "packet inspection result")
        if (result["tensor"] != query["tensor"] or result["shape"] != shape
                or result["dtype"] != info[query["tensor"]]["dtype"]):
            raise ValueError("packet inspection result differs from approved metadata")
        if result["statistics_precision"] != "exact extrema; binary64 approximate mean and L2":
            raise ValueError("unexpected inspection precision declaration")
        stats = result["statistics"]
        _exact(stats, {"count", "min", "max", "mean", "l2_norm", "finite"}, "packet statistics")
        if type(stats["count"]) is not int or stats["count"] != count or type(stats["finite"]) is not bool:
            raise ValueError("packet statistics count or finite flag is invalid")
        for key in ("min", "max", "mean", "l2_norm"):
            if type(stats[key]) not in (int, float) or not math.isfinite(stats[key]):
                raise ValueError("packet statistics must be finite numbers")
        if stats["min"] > stats["max"] or stats["l2_norm"] < 0:
            raise ValueError("packet statistics extrema or norm are invalid")
    return inspections


def build_manual_packet(*, request, protocol, inspections, operator_model="Astra",
                        operator_mode="Ultra", submission_expires_at_utc):
    """Render the actual private handoff, not an example candidate or launcher."""
    request = validate_session_request(request, protocol)
    if (operator_model != "Astra" or operator_mode != "Ultra"
            or protocol["editor"] != {"requested_selector": "gpt-6-astra", "requested_effort": "ultra"}):
        raise ValueError("this manual attempt must preserve the operator's Astra Ultra selection")
    if type(submission_expires_at_utc) is not str:
        raise ValueError("submission expiry must be an explicit UTC timestamp")
    try:
        expires = datetime.fromisoformat(submission_expires_at_utc.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("submission expiry must be an explicit UTC timestamp") from None
    if expires.tzinfo is None or expires.utcoffset() != timezone.utc.utcoffset(expires):
        raise ValueError("submission expiry must be an explicit UTC timestamp")
    expiry = expires.isoformat().replace("+00:00", "Z")
    inspections = _inspection_projection(inspections, request)
    context = request["approved_context"]
    model, policy, limits = context["model_info"], context["policy"], protocol["limits"]
    if (protocol["scope"]["baseline_sequences"] != 1 or protocol["scope"]["candidate_sequences"] != 1
            or limits["target_max_new_tokens"] != 128):
        raise ValueError("manual packet must preserve the existing one-pair 128-token capacity")
    dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    lines = [
        "You are the weight-proposal editor for one manual exploratory workflow validation.",
        "This is authenticated_session_v1 with the explicit manual exploratory protocol recorded by the operator.",
        "Editor isolation is UNVERIFIED. This is not a controlled research run or evidence of benchmark improvement.",
        f"The operator selected {operator_model}, {operator_mode} mode (requested selector gpt-6-astra, effort ultra).",
        "Runtime model identity, immutable version, effort metadata and frontier usage are unavailable; do not self-certify them.",
        "Use only this packet. Return one original weight proposal as the JSON response specified below, with no prose or code fences.",
        "No candidate operation, column choice, factor or direction is supplied for you to copy. Choose your own permitted proposal.",
        "Do not use external files, web search, connectors, other sessions, training, gradients or optimizer updates.",
        "Do not modify the architecture, tokenizer, harness, grader, evaluation settings or any unapproved tensor.",
        "Do not request another candidate, search iteration, inspection or forward pass. The local runner alone applies and evaluates your data proposal.",
        "",
        "PRISTINE CHECKPOINT AND APPROVED TENSORS",
        f"Checkpoint identity (SHA256 of canonical original file inventory): {model['checkpoint_sha256']}",
        f"Architecture: {dump(model['architectures'])}; stored coordinates: {model['stored_parameter_count']}.",
    ]
    for name, tensor in model["allowed_tensors"].items():
        lines.append(f"{name}: shape {dump(tensor['shape'])}, dtype {tensor['dtype']}, {tensor['parameter_count']} stored entries.")
    lines += ["", "ACTUAL BOUNDED PRISTINE-WEIGHT INSPECTIONS",
              "Ranges are zero-based half-open [start,end). These summaries cover only the stated slices, not whole tensors."]
    for item in inspections:
        query, result = item["request"], item["result"]
        lines.append(f"{query['tensor']}; slices={dump(query['slices'])}; statistics={dump(result['statistics'])}.")
    lines += [
        "Extrema are exact for stored values; mean and L2 norm use approximate binary64 arithmetic. No development generation or held-out score is included.",
        "",
        "DEVELOPMENT OBJECTIVE AND FIXED CAPACITY",
        "Propose a bounded intervention intended to preserve or improve the target's response on this single development item:",
        dump(context["development"]),
        "There is no supplied answer, score, grader internal, published patch or previous candidate feedback.",
        "One proposal and one comparison only: one baseline sequence and one candidate sequence, at most 128 generated target tokens each (256 total); no automatic retry.",
        "Both arms use the same development input, seed 17, greedy CPU bfloat16/eager execution, four threads and thinking disabled.",
        f"Local execution deadline: {limits['local_elapsed_seconds']} seconds from import; comparison deadline: {limits['target_timeout_seconds']} seconds shared by both arms.",
        f"The manual submission window is 48 hours from export; this submission expires at {expiry}.",
        "Submission expiry only rejects a late file. The runner neither launches nor cancels this session and supplies no frontier generation/reasoning-token cap.",
        "",
        "PERMITTED EDITS AND SAVED-CHECKPOINT LIMITS",
        "Operations affect all rows of the chosen matrix columns. Column indices must be unique integers in the tensor's width range; booleans are not integers.",
        f"Approved operations: {dump(policy['allowed_operations'])}; allowed columns by tensor: {dump(policy['allowed_columns'])}.",
        f"Scale factor range: {dump(policy['scale_factor_range'])}; direction strength range: {dump(policy['direction_strength_range'])} (inclusive).",
        f"At most {policy['max_operations']} operations; at most {policy['max_targeted_coordinates']} distinct targeted coordinates across their union.",
        f"Each operation and each tensor's combined selected columns must cover at most {policy['max_working_elements']} coordinates.",
        f"After saving/reloading: at most {policy['max_changed_coordinates']} changed stored entries, changed fraction <= {policy['max_changed_fraction']}, maximum absolute change <= {policy['max_abs_change']}.",
        "All numbers and saved weights must remain finite. Arithmetic is binary64 and each operation rounds to the original storage dtype; zero effective changes will be reported honestly.",
        "For scale_columns, W[:,columns] becomes factor * W[:,columns].",
        "For remove_direction_from_columns, direction has one finite number per matrix row and is not all zero; it is normalized to u, then W[:,columns] -= strength * u * (u.T @ W[:,columns]).",
        "",
        "EXACT RESPONSE GRAMMAR",
        "Uppercase PROPOSAL, OPERATION, TENSOR, COLUMNS, FACTOR, DIRECTION, STRENGTH below are schema variables, not literal output or a supplied proposal.",
        "All objects have exactly the shown keys, without duplicates. Return valid finite JSON, no comments, and at most 262144 UTF-8 bytes.",
        '{"schema_version":1,"kind":"editor_response","request_id":' + dump(request["request_id"]) + ',"request_sha256":' + dump(canonical_hash(request)) + ',"evidence_kind":"session_handoff","model_identity":{"provider":null,"model":null,"version":null,"interface":null},"action":{"tool":"propose","arguments":{"proposal":PROPOSAL}},"usage":{"generated_tokens":null,"billed_units":null},"scientific_result":false}',
        "PROPOSAL is exactly:",
        '{"schema_version":1,"base_sha256":' + dump(model["checkpoint_sha256"]) + ',"provenance":"editor_proposal","composition":"independent" or "sequential","operations":[OPERATION,...]}',
        "operations must be nonempty. independent permits exactly one operation; sequential permits one or more, subject to the above operation count. Every candidate starts from the pristine checkpoint.",
        "Each OPERATION must be exactly one of:",
        '{"op":"scale_columns","tensor":TENSOR,"columns":COLUMNS,"factor":FACTOR}',
        '{"op":"remove_direction_from_columns","tensor":TENSOR,"columns":COLUMNS,"direction":DIRECTION,"strength":STRENGTH}',
        "TENSOR is one exact approved tensor name. COLUMNS is a nonempty array of unique permitted integer columns. FACTOR and STRENGTH are finite numbers within their respective ranges. DIRECTION is the full nonzero finite row-length array, not a code expression, abbreviation or path.",
        "Do not add a rationale field, claimed model name, usage estimate, filesystem path, executable code, answer key or new evaluation request.",
    ]
    packet = "\n".join(lines) + "\n"
    if re.search(r"/Users/|/home/|/private/|[A-Za-z]:\\", packet):
        raise ValueError("manual packet must not expose a machine-specific path")
    return packet
