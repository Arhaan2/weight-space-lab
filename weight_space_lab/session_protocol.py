"""Versioned, development-only contract for an authenticated editor session.

This variant deliberately does not require a provider-enforced frontier token
cap: the audited subscription route exposes none. Local clocks, response byte
limits, and budget reservations are not substitutes for that capability.

All imported data remains untrusted. Validation here establishes a data shape,
not model identity, launch isolation, proposal validity, or scientific efficacy.
The controller must verify the actual launch boundary and validate proposals
against the immutable checkpoint policy before making any weight change.
"""

from __future__ import annotations

import re

from .checkpoint import validate_policy
from .editor_contract import (
    ALLOWED_TOOLS, MAX_HANDOFF_BYTES, _exact, _json_copy, _number,
    validate_request, validate_response,
)
from .protocol import canonical_hash


SESSION_VARIANT = "authenticated_session_v1"
ASTRA_SELECTOR = "gpt-6-astra"
SUPPORTED_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultra")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
# Metadata identifiers must not become a channel for prompts, URLs, or local
# paths in the public projection. Actual raw evidence belongs in private storage.
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_SCOPE = {
    "candidate_proposals": 1,
    "editor_requests": 1,
    "development_comparisons": 1,
    "baseline_sequences": 1,
    "candidate_sequences": 1,
    "automatic_retries": 0,
}
_RESTRICTIONS = {
    "development_only": True,
    "training": False,
    "gradients": False,
    "optimizer_updates": False,
    "architecture_changes": False,
    "evaluation_changes": False,
    "harness_changes": False,
    "grader_changes": False,
}


def _schema_one(value, label):
    if type(value) is not int or value != 1:
        raise ValueError(f"unsupported {label} schema_version")


def _identifier(value, label, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a bounded metadata identifier")


def _digest(value, label, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValueError(f"{label} must be a SHA256 digest")


def default_session_protocol(*, max_new_tokens, target_timeout_seconds,
                             local_elapsed_seconds, effort="high"):
    """Build the single-attempt variant; numerical limits have no defaults.

    Target generation bounds are per sequence. There is one baseline sequence
    and one candidate sequence, so the target-token reservation is twice the
    declared per-sequence maximum. ``target_timeout_seconds`` is the shared
    baseline/candidate comparison deadline: both arm reservations precede the
    baseline dispatch. Frontier generation has no advertised cap.
    """
    return validate_session_protocol({
        "schema_version": 1,
        "variant": SESSION_VARIANT,
        "editor": {"requested_selector": ASTRA_SELECTOR, "requested_effort": effort},
        "scope": dict(_SCOPE),
        "limits": {
            "target_max_new_tokens": max_new_tokens,
            "target_timeout_seconds": target_timeout_seconds,
            "local_elapsed_seconds": local_elapsed_seconds,
            "frontier_provider_token_cap": None,
            "frontier_provider_token_cap_status": "unavailable_subscription_route",
        },
        "restrictions": dict(_RESTRICTIONS),
    })


def validate_session_protocol(value):
    """Validate frozen controls for precisely one development attempt."""
    value = _json_copy(value)
    _exact(value, {"schema_version", "variant", "editor", "scope", "limits",
                   "restrictions"}, "session protocol")
    _schema_one(value["schema_version"], "session protocol")
    if value["variant"] != SESSION_VARIANT:
        raise ValueError("unsupported session protocol variant")
    editor = value["editor"]
    _exact(editor, {"requested_selector", "requested_effort"}, "editor")
    if editor["requested_selector"] != ASTRA_SELECTOR:
        raise ValueError("the session variant requires the exposed Astra selector")
    if editor["requested_effort"] not in SUPPORTED_EFFORTS:
        raise ValueError("requested_effort must be an explicitly supported effort")
    _exact(value["scope"], _SCOPE, "scope")
    for key, fixed in _SCOPE.items():
        if type(value["scope"][key]) is not int or value["scope"][key] != fixed:
            raise ValueError(f"scope.{key} is fixed at {fixed} for this variant")
    limits = value["limits"]
    _exact(limits, {"target_max_new_tokens", "target_timeout_seconds",
                    "local_elapsed_seconds", "frontier_provider_token_cap",
                    "frontier_provider_token_cap_status"}, "session limits")
    _number(limits["target_max_new_tokens"], "target_max_new_tokens", positive=True, integer=True)
    for key in ("target_timeout_seconds", "local_elapsed_seconds"):
        _number(limits[key], key, positive=True)
    if limits["frontier_provider_token_cap"] is not None:
        raise ValueError("a frontier provider token cap is unavailable through this session route")
    if limits["frontier_provider_token_cap_status"] != "unavailable_subscription_route":
        raise ValueError("the unavailable frontier provider cap must be explicit")
    _exact(value["restrictions"], _RESTRICTIONS, "restrictions")
    for key, fixed in _RESTRICTIONS.items():
        if value["restrictions"][key] is not fixed:
            raise ValueError(f"the session variant forbids changing restriction {key}")
    return value


def make_session_request(*, request_id, session_sha256, instruction,
                         approved_context, protocol):
    protocol = validate_session_protocol(protocol)
    return validate_session_request({
        "schema_version": 1,
        "kind": "editor_request",
        "request_id": request_id,
        "session_sha256": session_sha256,
        "protocol_sha256": canonical_hash(protocol),
        "mode": SESSION_VARIANT,
        "instruction": instruction,
        "approved_context": approved_context,
        "limits": {
            "max_response_bytes": MAX_HANDOFF_BYTES,
            "local_elapsed_seconds": protocol["limits"]["local_elapsed_seconds"],
            "frontier_provider_token_cap": None,
        },
        "allowed_tools": ["propose"],
    }, protocol)


def _normalized_request(value):
    """Reuse the established data-shape validator without publishing its mode.

    The internal output-token sentinel satisfies the older schema only; it is
    never exposed or executed as a frontier generation limit.
    """
    result = _json_copy(value)
    del result["protocol_sha256"]
    result["mode"] = "offline_integration"
    result["allowed_tools"] = list(ALLOWED_TOOLS)
    result["limits"] = {"max_output_tokens": 1,
                        "timeout_seconds": value["limits"]["local_elapsed_seconds"]}
    return result


def _validate_approved_context(context):
    """Close model metadata/policy channels using approved metadata only.

    This function does not read inventory or any checkpoint files. The existing
    policy validator can operate on this small tensor projection; the controller
    separately binds these values to its trusted immutable original inventory.
    """
    info = context["model_info"]
    _exact(info, {"checkpoint_sha256", "architectures", "stored_parameter_count",
                  "allowed_tensors"}, "approved model_info")
    _digest(info["checkpoint_sha256"], "checkpoint_sha256")
    _number(info["stored_parameter_count"], "stored_parameter_count", integer=True)
    architectures = info["architectures"]
    if type(architectures) is not list or len(architectures) > 32:
        raise ValueError("architectures must be a bounded list of metadata identifiers")
    for name in architectures:
        _identifier(name, "architecture")
    if len(set(architectures)) != len(architectures):
        raise ValueError("architecture identifiers must be unique")
    tensors = info["allowed_tensors"]
    if type(tensors) is not dict or not 1 <= len(tensors) <= 128:
        raise ValueError("allowed_tensors must be a bounded nonempty metadata mapping")
    parameters = 0
    for name, tensor in tensors.items():
        _identifier(name, "allowed tensor name")
        _exact(tensor, {"shape", "dtype", "parameter_count"}, "approved tensor metadata")
        _identifier(tensor["dtype"], "tensor dtype")
        shape = tensor["shape"]
        if type(shape) is not list or len(shape) != 2:
            raise ValueError("approved editable tensor shape must be a matrix")
        for dimension in shape:
            _number(dimension, "tensor dimension", integer=True)
        _number(tensor["parameter_count"], "tensor parameter_count", integer=True)
        if tensor["parameter_count"] != shape[0] * shape[1]:
            raise ValueError("tensor parameter_count must match its approved shape")
        parameters += tensor["parameter_count"]
    if parameters > info["stored_parameter_count"]:
        raise ValueError("approved tensors exceed the stored model parameter count")
    policy = validate_policy(context["policy"], {"tensors": tensors})
    if set(policy["allowed_tensors"]) != set(tensors):
        raise ValueError("policy and model metadata must expose the same approved tensors")
    if context["feedback"] is not None:
        raise ValueError("the one-request session starts without prior development feedback")


def validate_session_request(value, protocol):
    protocol = validate_session_protocol(protocol)
    value = _json_copy(value)
    _exact(value, {"schema_version", "kind", "request_id", "session_sha256",
                   "protocol_sha256", "mode", "instruction", "approved_context",
                   "limits", "allowed_tools"}, "session request")
    if value["mode"] != SESSION_VARIANT:
        raise ValueError("session request must use its explicit versioned mode")
    if value["protocol_sha256"] != canonical_hash(protocol):
        raise ValueError("session request differs from the frozen protocol")
    if value["allowed_tools"] != ["propose"]:
        raise ValueError("the session handoff permits only one proposal action")
    _exact(value["limits"], {"max_response_bytes", "local_elapsed_seconds",
                             "frontier_provider_token_cap"}, "session request limits")
    limits = value["limits"]
    if type(limits["max_response_bytes"]) is not int or limits["max_response_bytes"] != MAX_HANDOFF_BYTES:
        raise ValueError("max_response_bytes must match the bounded handoff reader")
    _number(limits["local_elapsed_seconds"], "local_elapsed_seconds", positive=True)
    if limits["local_elapsed_seconds"] != protocol["limits"]["local_elapsed_seconds"]:
        raise ValueError("request local elapsed limit differs from the frozen protocol")
    if limits["frontier_provider_token_cap"] is not None:
        raise ValueError("the session request cannot advertise a provider token cap")
    validate_request(_normalized_request(value))
    _validate_approved_context(value["approved_context"])
    if len(value["approved_context"]["development"]) != 1:
        raise ValueError("the single-comparison variant requires exactly one development example")
    return value


def validate_session_response(value, request, protocol):
    """Validate a proposal handoff, without authenticating its claimed origin.

    Usage in this response is model-authored and unverified. The controller must
    account from separately observed transport usage, or record it unavailable.
    Candidate policy validation remains mandatory after this envelope check.
    """
    request = validate_session_request(request, protocol)
    value = _json_copy(value)
    _exact(value, {"schema_version", "kind", "request_id", "request_sha256",
                   "evidence_kind", "model_identity", "action", "usage",
                   "scientific_result"}, "session response")
    if value["request_id"] != request["request_id"] or value["request_sha256"] != canonical_hash(request):
        raise ValueError("session response is not bound to this exact request")
    if value["evidence_kind"] != "session_handoff":
        raise ValueError("session response evidence_kind must be session_handoff")
    _exact(value["model_identity"], {"provider", "model", "version", "interface"}, "model_identity")
    if any(item is not None for item in value["model_identity"].values()):
        raise ValueError("model-authored identity is not session provenance; identity fields must be null")
    _exact(value["action"], {"tool", "arguments"}, "action")
    if value["action"]["tool"] != "propose":
        raise ValueError("the session response permits only a proposal action")
    normalized_request = _normalized_request(request)
    normalized_response = _json_copy(value)
    normalized_response["request_sha256"] = canonical_hash(normalized_request)
    normalized_response["evidence_kind"] = "unverified_file_handoff"
    validate_response(normalized_response, normalized_request)
    proposal = value["action"]["arguments"]["proposal"]
    _exact(proposal, {"schema_version", "base_sha256", "operations", "provenance",
                      "composition"}, "session proposal")
    _schema_one(proposal["schema_version"], "session proposal")
    _digest(proposal["base_sha256"], "proposal.base_sha256")
    if proposal["provenance"] != "editor_proposal":
        raise ValueError("a session candidate must declare editor_proposal provenance")
    if proposal["composition"] not in ("independent", "sequential"):
        raise ValueError("unsupported session proposal composition")
    if type(proposal["operations"]) is not list or not proposal["operations"]:
        raise ValueError("the bounded attempt requires a nonempty weight proposal")
    return value


def validate_session_provenance(value, protocol):
    """Validate evidence labels, not evidence truth or source authenticity.

    In particular, an operator attestation is never upgraded to transport
    metadata, and a digest merely binds separately retained evidence. The
    controller must verify the receipt against the actual isolated launch.
    """
    protocol = validate_session_protocol(protocol)
    value = _json_copy(value)
    _exact(value, {"schema_version", "source", "requested_selector", "requested_effort",
                   "accepted_selector", "observed_model", "observed_effort",
                   "immutable_model_version", "interface", "evidence_sha256"}, "session provenance")
    _schema_one(value["schema_version"], "session provenance")
    if value["source"] not in ("unavailable", "operator_attestation", "transport_metadata"):
        raise ValueError("provenance source must describe external evidence, never model self-description")
    for key in ("requested_selector", "requested_effort"):
        if value[key] != protocol["editor"][key]:
            raise ValueError(f"provenance {key} differs from the frozen configuration")
    if value["accepted_selector"] not in (None, ASTRA_SELECTOR):
        raise ValueError("accepted selector must not silently substitute another model")
    if value["observed_effort"] not in (None,) + SUPPORTED_EFFORTS:
        raise ValueError("unsupported observed effort")
    for key in ("observed_model", "immutable_model_version"):
        _identifier(value[key], key, nullable=True)
    _exact(value["interface"], {"name", "version"}, "interface")
    _identifier(value["interface"]["name"], "interface.name")
    _identifier(value["interface"]["version"], "interface.version", nullable=True)
    _digest(value["evidence_sha256"], "evidence_sha256", nullable=True)
    if value["source"] == "transport_metadata" and value["evidence_sha256"] is None:
        raise ValueError("transport metadata must bind separately retained evidence")
    if value["source"] == "unavailable" and any(value[key] is not None for key in (
            "accepted_selector", "observed_model", "observed_effort", "immutable_model_version",
            "evidence_sha256")):
        raise ValueError("unavailable provenance cannot assert observed identity")
    return value


def public_session_summary(protocol, provenance=None, usage=None):
    """Project only bounded public metadata; never accept a request/transcript.

    ``usage`` must come from a separately checked receipt, not the model-authored
    handoff. Unavailable counters remain null rather than being recorded as zero.
    This projection itself supplies no launch or identity verification.
    """
    protocol = validate_session_protocol(protocol)
    if provenance is not None:
        provenance = validate_session_provenance(provenance, protocol)
    if usage is None:
        usage = {"generated_tokens": None, "billed_units": None}
    usage = _json_copy(usage)
    _exact(usage, {"generated_tokens", "billed_units"}, "observed frontier usage")
    _number(usage["generated_tokens"], "generated_tokens", integer=True, nullable=True)
    _number(usage["billed_units"], "billed_units", nullable=True)
    return {
        "schema_version": 1,
        "variant": SESSION_VARIANT,
        "protocol_sha256": canonical_hash(protocol),
        "experimental_role": "Astra weight-proposal editor",
        "editor": protocol["editor"],
        "scope": protocol["scope"],
        "limits": protocol["limits"],
        "restrictions": protocol["restrictions"],
        "provenance": provenance,
        "identity_verification": "external_receipt_required",
        "observed_frontier_usage": usage,
        "scientific_result": False,
    }
