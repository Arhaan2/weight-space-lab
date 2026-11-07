"""Provider-independent, non-scientific Phase 7 file handoff contracts.

An imported response is untrusted data. It cannot authenticate model provenance,
grant tools, change the frozen study, or refund a reserved execution budget.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat

from .protocol import canonical_hash


ALLOWED_TOOLS = ("model_info", "development", "inspect", "propose", "probe", "stop")
MAX_HANDOFF_BYTES = 262144
_ID = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} has missing or unknown fields")


def _text(value, label, maximum, *, nullable=False):
    if value is None and nullable:
        return
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} must be bounded nonempty text")


def _number(value, label, *, positive=False, integer=False, nullable=False):
    if value is None and nullable:
        return
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid or (integer and value > 2**63 - 1):
        raise ValueError(f"{label} is outside the supported range")


def _json_copy(value, *, max_bytes=MAX_HANDOFF_BYTES):
    def check(item, depth):
        if depth > 32:
            raise ValueError("handoff nesting exceeds the contract limit")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("handoff contains a non-JSON value")
    try:
        check(value, 0)
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > max_bytes:
            raise ValueError("handoff exceeds maximum byte size")
        return json.loads(encoded)
    except (RecursionError, TypeError, UnicodeError) as error:
        raise ValueError("handoff is not bounded finite JSON") from error


def validate_development(value):
    """Accept only bounded, explicitly development-only user messages."""
    value = _json_copy(value)
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise ValueError("development must contain between 1 and 32 examples")
    seen = set()
    for item in value:
        _exact(item, {"item_id", "split", "messages"}, "development example")
        _text(item["item_id"], "item_id", 128)
        if item["item_id"] in seen:
            raise ValueError("development item IDs must be unique")
        seen.add(item["item_id"])
        if item["split"] != "development":
            raise ValueError("only the development split is allowed")
        messages = item["messages"]
        if type(messages) is not list or not 1 <= len(messages) <= 32:
            raise ValueError("development messages must be a bounded nonempty list")
        for message in messages:
            _exact(message, {"role", "content"}, "development message")
            if message["role"] != "user":
                raise ValueError("development messages must have the user role")
            _text(message["content"], "development content", 8192)
    return value


def validate_request(value):
    value = _json_copy(value)
    _exact(value, {"schema_version", "kind", "request_id", "session_sha256", "mode",
                   "instruction", "approved_context", "limits", "allowed_tools"}, "request")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported request schema_version")
    if value["kind"] != "editor_request" or value["mode"] != "offline_integration":
        raise ValueError("only offline integration editor requests are supported")
    if type(value["request_id"]) is not str or not _ID.fullmatch(value["request_id"]):
        raise ValueError("request_id must be a lowercase UUID hex string")
    if type(value["session_sha256"]) is not str or not _DIGEST.fullmatch(value["session_sha256"]):
        raise ValueError("session_sha256 must be a SHA256 digest")
    _text(value["instruction"], "instruction", 16000)
    if value["allowed_tools"] != list(ALLOWED_TOOLS):
        raise ValueError("allowed_tools must match the fixed development tool contract")
    context = value["approved_context"]
    _exact(context, {"model_info", "development", "policy", "feedback"}, "approved_context")
    for name in ("model_info", "policy"):
        if type(context[name]) is not dict:
            raise ValueError(f"approved_context.{name} must be an object")
    if context["feedback"] is not None and type(context["feedback"]) is not dict:
        raise ValueError("approved_context.feedback must be an object or null")
    context["development"] = validate_development(context["development"])
    _exact(value["limits"], {"max_output_tokens", "timeout_seconds"}, "limits")
    _number(value["limits"]["max_output_tokens"], "max_output_tokens", positive=True, integer=True)
    _number(value["limits"]["timeout_seconds"], "timeout_seconds", positive=True)
    return value


def validate_response(value, request):
    request = validate_request(request)
    value = _json_copy(value)
    _exact(value, {"schema_version", "kind", "request_id", "request_sha256", "evidence_kind",
                   "model_identity", "action", "usage", "scientific_result"}, "response")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported response schema_version")
    if value["kind"] != "editor_response":
        raise ValueError("response kind must be editor_response")
    if value["request_id"] != request["request_id"] or value["request_sha256"] != canonical_hash(request):
        raise ValueError("response is not bound to this exact request")
    if value["evidence_kind"] not in ("simulated", "unverified_file_handoff"):
        raise ValueError("live model evidence is not supported by the file handoff")
    if value["scientific_result"] is not False:
        raise ValueError("file handoff responses must remain explicitly non-scientific")
    _exact(value["model_identity"], {"provider", "model", "version", "interface"}, "model_identity")
    for key, identity in value["model_identity"].items():
        _text(identity, f"model_identity.{key}", 512, nullable=True)
    _exact(value["usage"], {"generated_tokens", "billed_units"}, "usage")
    _number(value["usage"]["generated_tokens"], "generated_tokens", integer=True, nullable=True)
    _number(value["usage"]["billed_units"], "billed_units", nullable=True)
    _exact(value["action"], {"tool", "arguments"}, "action")
    tool, arguments = value["action"]["tool"], value["action"]["arguments"]
    if type(tool) is not str or tool not in ALLOWED_TOOLS or type(arguments) is not dict:
        raise ValueError("action must use an approved tool with object arguments")
    if tool in ("model_info", "development", "stop"):
        _exact(arguments, set(), f"{tool} arguments")
    elif tool == "propose":
        _exact(arguments, {"proposal"}, "propose arguments")
        if type(arguments["proposal"]) is not dict:
            raise ValueError("proposal must be an object")
    elif tool == "probe":
        _exact(arguments, {"task_ids", "candidate_id"}, "probe arguments")
        ids = arguments["task_ids"]
        approved_ids = {item["item_id"] for item in request["approved_context"]["development"]}
        if (type(ids) is not list or not 1 <= len(ids) <= len(approved_ids)
                or any(type(item) is not str or item not in approved_ids for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("probe task IDs must be unique approved development IDs")
        candidate = arguments["candidate_id"]
        if type(candidate) is not str or (candidate != "base" and not _ID.fullmatch(candidate)):
            raise ValueError("candidate_id must be base or a controller-issued UUID hex string")
    # inspect and proposal bodies are validated by the checkpoint policy before
    # execution. They confer no paths, executable code, or additional tools.
    return value


def _read_bytes_bounded(path, max_bytes=MAX_HANDOFF_BYTES):
    """Read regular JSON files without following symlinks or blocking on FIFOs.

    Every directory is opened relative to its already-open parent with
    O_NOFOLLOW, avoiding the usual check-then-open symlink race. A file is data;
    nothing imported here is evaluated or executed.
    """
    _number(max_bytes, "max_bytes", positive=True, integer=True)
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed in handoff paths")
    path = path.absolute()
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(path.anchor, flags | os.O_DIRECTORY)
    file_descriptor = None
    try:
        for component in path.parts[1:-1]:
            next_directory = os.open(component, flags | os.O_DIRECTORY, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        file_descriptor = os.open(path.name, flags | os.O_NONBLOCK, dir_fd=directory)
        info = os.fstat(file_descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("handoff must be a bounded regular file")
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = None
            payload = stream.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise ValueError("handoff exceeds maximum byte size")
    except OSError as error:
        raise ValueError("handoff file is inaccessible or uses a symlink") from error
    finally:
        os.close(directory)
        if file_descriptor is not None:
            os.close(file_descriptor)
    return payload


def read_text_bounded(path, max_bytes=MAX_HANDOFF_BYTES):
    """Read a bounded regular UTF-8 handoff, with the same no-symlink boundary."""
    try:
        return _read_bytes_bounded(path, max_bytes).decode("utf-8")
    except UnicodeError as error:
        raise ValueError("handoff is not bounded UTF-8 text") from error


def read_json_bounded(path, max_bytes=MAX_HANDOFF_BYTES):
    """Read strict JSON through the bounded, no-symlink handoff reader."""
    payload = _read_bytes_bounded(path, max_bytes)

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    def reject_constant(_):
        raise ValueError("JSON must not contain nonfinite constants")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=reject_constant)
        return _json_copy(value, max_bytes=max_bytes)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("handoff is not bounded UTF-8 JSON") from error
