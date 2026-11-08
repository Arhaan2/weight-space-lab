"""Private, evidence-bound manual Astra handoff for the browser IFEval study.

Exports never call a frontier service or a target model. The later import path
accepts one data-only proposal and uses the existing saved-checkpoint verifier.
Arhaan2; exploratory on-device experiment, editor isolation unverified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

from .editor_budget import BudgetLedger
from .editor_contract import read_json_bounded
from .editor_tools import run_checkpoint_action
from .grading import separate_output, validate_grade_record
from .inventory import file_sha256, inventory_model, read_json
from .protocol import canonical_hash
from .records import run_record_sha256, validate_run_record


IMPORT_LIMITS = {"editor_requests": 1, "model_probes": 0,
                 "requested_sequences": 1, "generated_tokens": 0,
                 "elapsed_seconds": 900, "billed_units": None}
MAX_RESPONSE_BYTES = 262144
NEXT_COMMANDS = '''# Subsequent stage only: use after an original Astra proposal and candidate authorization.
# Save the returned JSON unchanged as artifacts/ifeval-browser-v1/response.json.
# Save the subsequent candidate-stage authorization as artifacts/ifeval-browser-v1/candidate-authorization.txt.
export PYTHONPYCACHEPREFIX=artifacts/ifeval-browser-v1/pycache
ifeval_python=artifacts/ifeval-browser-v1/evaluator-env/bin/python
ifeval_root=artifacts/ifeval-browser-v1

"$ifeval_python" -m weight_space_lab.ifeval_handoff import --root "$ifeval_root" --response "$ifeval_root/response.json"
"$ifeval_python" -m weight_space_lab.browser_candidate convert --source-original artifacts/on-device-recovery-v1/model --source-candidate "$ifeval_root/candidate/source" --baseline-manifest artifacts/browser-integration-v1/conversion/artifacts.json --converter-python artifacts/browser-integration-v1/conversion/venv/bin/python --policy "$ifeval_root/weight-information/policy.json" --output "$ifeval_root/candidate/converted"
"$ifeval_python" -m weight_space_lab.browser_candidate package-candidate --source-original artifacts/on-device-recovery-v1/model --source-candidate "$ifeval_root/candidate/source" --baseline-manifest artifacts/browser-integration-v1/conversion/artifacts.json --policy "$ifeval_root/weight-information/policy.json" --converted "$ifeval_root/candidate/converted" --output "$ifeval_root/candidate/deployed"
"$ifeval_python" -m weight_space_lab.ifeval_comparison bind --root "$ifeval_root" --candidate-artifact "$ifeval_root/candidate/deployed/candidate-artifact.json" --authorization "$ifeval_root/candidate-authorization.txt"
"$ifeval_python" -m weight_space_lab.ifeval_comparison run --root "$ifeval_root"
"$ifeval_python" -m weight_space_lab.ifeval_comparison compare --root "$ifeval_root"
'''


def _save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inspection_plan(policy, inventory):
    """Fixed spatial sampling, chosen without model responses or activations.

    For each allowed matrix, inspect sixteen spread-out complete columns and
    eight 60-column blocks. This is a sample, never a whole-tensor analysis.
    """
    names = policy["inspection"]["allowed_tensors"]
    requests = [{"op": "metadata", "tensors": names}]
    for name in names:
        rows, cols = inventory["tensors"][name]["shape"]
        if cols < 60:
            raise ValueError("fixed inspection plan requires at least 60 columns")
        for index in range(16):
            col = round(index * (cols - 1) / 15)
            requests.append({"op": "statistics", "tensor": name,
                             "slices": [[0, rows], [col, col + 1]]})
        for index in range(8):
            start = round(index * (cols - 60) / 7)
            requests.append({"op": "statistics", "tensor": name,
                             "slices": [[0, rows], [start, start + 60]]})
    return requests


def prepare_inspections(root, model, policy_path="configs/candidate_policy.json"):
    root, model = Path(root), Path(model)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "weight-information"
    target.mkdir(mode=0o700)
    policy = read_json(policy_path)
    inventory = inventory_model(model)
    base_sha256 = canonical_hash(inventory["files"])
    plan = inspection_plan(policy, inventory)
    _save(target / "policy.json", policy)
    _save(target / "plan.json", {"selection": "fixed spatial sampling; independent of development outputs",
        "requests": plan, "base_sha256": base_sha256, "target_forward_passes": 0})
    receipts = []
    started = time.monotonic()
    deadline = started + 900
    for index, request in enumerate(plan, 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("inspection preparation deadline reached")
        result = run_checkpoint_action(model=model, policy=policy,
            exposure_log=target / "exposures.jsonl", output=None, tool="inspect",
            arguments=request, timeout_seconds=min(remaining, 120), check_active=lambda: None,
            expected_original_sha256=base_sha256)
        receipt = {"sequence": index, "request": request, "result": result}
        _save(target / f"inspection-{index:02d}.json", receipt)
        receipts.append(receipt)
    journal = [json.loads(line) for line in (target / "exposures.jsonl").read_text().splitlines()]
    # Calculate unique geometric coverage separately from charged scans; blocks
    # and selected singleton columns can overlap and must not be double-counted.
    coverage = {}
    for name in policy["inspection"]["allowed_tensors"]:
        rows, cols = inventory["tensors"][name]["shape"]
        sampled = set()
        for request in plan[1:]:
            if request["tensor"] == name:
                sampled.update(range(*request["slices"][1]))
        coverage[name] = {"shape": [rows, cols], "dtype": inventory["tensors"][name]["dtype"],
            "parameter_count": rows * cols, "sampled_complete_columns": sorted(sampled),
            "unique_sampled_coordinates": rows * len(sampled),
            "unique_coordinate_fraction": len(sampled) / cols,
            "full_tensor_statistics": False}
    result = {"schema_version": 1, "kind": "ifeval_approved_weight_inspections",
        "base_sha256": base_sha256, "policy_sha256": canonical_hash(policy),
        "exposure_log_sha256": file_sha256(target / "exposures.jsonl"),
        "selection": "fixed spatial sampling, no activations or task-conditioned forward passes",
        "scope": "bounded sampled column/block statistics; not whole-tensor or functional analysis",
        "request_count": len(receipts), "values_exposed": sum(r["values_exposed"] for r in journal[1:]),
        "coordinates_scanned": sum(r["coordinates_scanned"] for r in journal[1:]),
        "coverage": coverage, "inspections": receipts, "target_forward_passes": 0,
        "elapsed_seconds": time.monotonic() - started}
    _save(target / "inspections.json", result)
    # Local paths belong only in host-private evidence, never the editor packet.
    _save(target / "host.json", {"model_path": str(model.resolve()),
        "base_sha256": base_sha256, "source_files": inventory["files"]})
    return {key: result[key] for key in ("base_sha256", "request_count", "values_exposed",
                                        "coordinates_scanned", "target_forward_passes", "elapsed_seconds")}


def baseline_feedback(items, records, grades, policy):
    ids = [item["item_id"] for item in items]
    if (len(ids) != 32 or len(set(ids)) != 32 or
            [r["item_id"] for r in records] != ids or [g["item_id"] for g in grades] != ids):
        raise ValueError("handoff requires the exact 32 frozen development item/record/grade pairs")
    feedback = []
    grader = None
    config_sha = None
    checkpoint = None
    for item, record, grade in zip(items, records, grades):
        validate_run_record(record)
        validate_grade_record(grade)
        if (item["benchmark"] != "ifeval" or item["split"] != "development" or
                record["benchmark"] != "ifeval" or record["split"] != "development" or
                record["evidence_kind"] != "real_model" or record["seed"] != 42 or
                record["sample_index"] != 0 or grade["seed"] != 42 or grade["sample_index"] != 0 or
                record["dataset_revision"] != item["dataset_revision"] or
                grade["item_sha256"] != canonical_hash(item) or
                grade["run_record_sha256"] != run_record_sha256(record) or
                grade["policy_sha256"] != canonical_hash(policy) or
                record["retry_policy"] != policy["generation_retry"]):
            raise ValueError("baseline feedback evidence binding changed")
        if record["status"] not in ("ok", "truncated") or grade["status"] != "graded":
            raise ValueError("technical execution and independent grading must resolve before handoff")
        if grader is None:
            grader, config_sha, checkpoint = grade["grader"], record["configuration_sha256"], record["checkpoint"]
        elif (grade["grader"] != grader or record["configuration_sha256"] != config_sha or
              record["checkpoint"] != checkpoint):
            raise ValueError("mixed baseline evaluator, runtime, or checkpoint identity")
        names, kwargs = item["target"]["instruction_id_list"], item["target"]["kwargs"]
        if grade["metrics"]["instruction_count"] != len(names):
            raise ValueError("instruction count differs from frozen item")
        reasoning, output, code = separate_output(record["raw_output"], policy)
        if code != "output_ready" or reasoning.strip():
            raise ValueError("non-thinking browser output does not match the frozen extraction policy")
        feedback.append({"item_id": item["item_id"], "prompt": item["task_text"],
            "raw_response": record["raw_output"], "graded_response": output,
            "status": record["status"], "finish_reason": record["finish_reason"],
            "token_counts": record["token_counts"], "duration_seconds": record["duration_seconds"],
            "completion_within_budget": record["status"] == "ok",
            "strict_prompt_compliant": grade["correct"], "grade_diagnostic": grade["diagnostics"]["code"],
            "constraints": [{"instruction_id": name, "kwargs": args, "strict": strict, "loose": loose}
                for name, args, strict, loose in zip(names, kwargs,
                    grade["metrics"]["instruction_strict"], grade["metrics"]["instruction_loose"])]})
    return feedback


def aggregate_feedback(feedback):
    count = sum(len(row["constraints"]) for row in feedback)
    strict = sum(c["strict"] for row in feedback for c in row["constraints"])
    loose = sum(c["loose"] for row in feedback for c in row["constraints"])
    return {"scheduled_prompts": len(feedback), "scheduled_instructions": count,
        "prompt_level_strict_accuracy": sum(all(c["strict"] for c in r["constraints"]) for r in feedback) / len(feedback),
        "prompt_level_loose_accuracy": sum(all(c["loose"] for c in r["constraints"]) for r in feedback) / len(feedback),
        "instruction_level_strict_accuracy": strict / count,
        "instruction_level_loose_accuracy": loose / count,
        "strict_compliant_instructions": strict, "loose_compliant_instructions": loose,
        "completed": sum(r["status"] == "ok" for r in feedback),
        "truncated": sum(r["status"] == "truncated" for r in feedback)}


def _identity_only(value):
    """Recursively omit local paths/URLs and evaluation content from identities."""
    if isinstance(value, dict):
        return {key: _identity_only(item) for key, item in value.items()
                if not any(word in key.lower() for word in ("path", "url", "command", "heldout", "overlap"))}
    if isinstance(value, list):
        return [_identity_only(item) for item in value]
    if isinstance(value, str) and (value.startswith(("/", "file:", "http://127.", "http://localhost"))):
        return "host-local location omitted"
    return value


def resource_summary(root, envelope):
    """Keep worker, owned-process, and host-wide observations separate."""
    root = Path(root)
    target = root / "execution/baseline-1"
    path = target / "resources.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    host = {}
    roles = {}
    for row in rows:
        for key, value in row.get("host", {}).items():
            if type(value) in (int, float):
                prior = host.setdefault(key, {"minimum": value, "maximum": value, "samples": 0})
                prior["minimum"], prior["maximum"] = min(prior["minimum"], value), max(prior["maximum"], value)
                prior["samples"] += 1
        for process in row.get("owned_processes", []):
            role = roles.setdefault(process["role"], {})
            for key in ("footprintBytes", "rssBytes", "wiredBytes"):
                value = process.get(key)
                if type(value) in (int, float):
                    role[key] = max(role.get(key, 0), value)
    return {"host_scope": "host-wide samples include unrelated workloads; not model-attributed physical GPU memory",
        "host_sampled_numeric_ranges": host, "host_sample_count": len(rows),
        "owned_process_scope": "maximum single observed process per role; RSS/footprint can include shared memory",
        "owned_role_sampled_maxima": roles,
        "worker_scope": "GPUBuffer descriptor accounting and WASM address space are not physical VRAM",
        "per_item_browser_observations": [{"item_id": m["item_id"],
            "resources": _identity_only(m.get("resources", {"status": "unavailable"})),
            "loading_time_ms": m.get("loading_time_ms"),
            "response_latency_ms": m.get("response_latency_ms")}
            for m in envelope.get("item_metrics", envelope.get("metrics", []))],
        "sampling_limit": "Observed sample maxima do not establish absolute peaks; unavailable metrics remain unavailable."}


def _ledger(root, request, *, create=False):
    return BudgetLedger(Path(root) / "handoff/import-ledger.sqlite3", run_id="ifeval-browser-v1-manual-import",
        limits=IMPORT_LIMITS, binding={"request_sha256": canonical_hash(request),
            "accounting_scope": "one external proposal ingestion; no frontier dispatch"}, create=create)


def export_packet(root, *, items_path=None):
    root = Path(root)
    items_path = Path(items_path) if items_path else root / "data/development-items.json"
    manifest = read_json(root / "manifest.json")
    config = read_json(root / "baseline-config.json")
    policy = read_json(root / "grading-policy.json")
    envelope = read_json(root / "baseline/graded.json")
    if envelope.get("summary", {}).get("technically_valid") is not True:
        raise ValueError("technically valid baseline and resolved grades are required for the manual handoff")
    items = read_json(items_path)
    feedback = baseline_feedback(items, envelope["records"], envelope["grades"], policy)
    inspections = read_json(root / "weight-information/inspections.json")
    edit_policy = read_json(root / "weight-information/policy.json")
    if inspections["exposure_log_sha256"] != file_sha256(root / "weight-information/exposures.jsonl"):
        raise ValueError("inspection exposure log changed")
    base = inspections["base_sha256"]
    if any(record["checkpoint"]["sha256"] != base for record in envelope["records"]):
        raise ValueError("baseline source checkpoint differs from inspected pristine weights")
    packet_dir = root / "handoff"
    packet_dir.mkdir(mode=0o700)
    now = time.time()
    artifacts = {"manifest.json": file_sha256(root / "manifest.json"),
        "baseline-config.json": file_sha256(root / "baseline-config.json"),
        "grading-policy.json": file_sha256(root / "grading-policy.json"),
        "baseline/graded.json": file_sha256(root / "baseline/graded.json"),
        "weight-information/inspections.json": file_sha256(root / "weight-information/inspections.json"),
        "weight-information/policy.json": file_sha256(root / "weight-information/policy.json"),
        "weight-information/exposures.jsonl": file_sha256(root / "weight-information/exposures.jsonl")}
    accounting = read_json(root / "accounting-after.json") if (root / "accounting-after.json").exists() else None
    if accounting is not None:
        artifacts["accounting-after.json"] = file_sha256(root / "accounting-after.json")
    if (root / "execution/baseline-1/resources.jsonl").exists():
        artifacts["execution/baseline-1/resources.jsonl"] = file_sha256(root / "execution/baseline-1/resources.jsonl")
    request = {"schema_version": 1, "kind": "ifeval_browser_manual_request", "study_id": "ifeval-browser-v1",
        "request_id": str(uuid.uuid4()), "created_at_epoch_seconds": now,
        "submission_expires_at_epoch_seconds": now + 7 * 24 * 3600,
        "scope": "exploratory on-device weight editing", "editor_isolation": "unverified",
        "editor": {"requested_display_name": "Astra", "requested_mode": "Ultra",
            "requested_selector": "gpt-6-astra", "requested_effort": "ultra",
            "selection_enforcement": "operator only", "observed_model": None,
            "observed_effort": None, "immutable_model_version": None, "frontier_usage": None, "billing": None},
        "objective": "Improve strict prompt-level IFEval instruction compliance at the identical fixed browser execution budget.",
        "checkpoint": {"model_id": "Qwen/Qwen3.5-0.8B", "base_sha256": base,
                       "identity": envelope["records"][0]["checkpoint"]},
        "deployment": _identity_only({key: value for key, value in config.items() if key not in {"tasks", "items"}}),
        "frozen_study_sha256": canonical_hash(manifest), "evidence_files_sha256": artifacts,
        "policy": edit_policy, "approved_weight_information": inspections,
        "scoring_policy": policy, "primary_metric": "prompt_level_strict_accuracy",
        "secondary_metrics": ["instruction_level_strict_accuracy", "prompt_level_loose_accuracy", "instruction_level_loose_accuracy"],
        "aggregate_metrics": aggregate_feedback(feedback), "development": feedback,
        "resource_observations": resource_summary(root, envelope),
        "execution_accounting": None if accounting is None else _identity_only({
            key: value for key, value in accounting.items() if key not in {"outputs", "warmup_outputs"}}),
        "warmup_observations": None if accounting is None else [{key: row.get(key) for key in
            ("generated_tokens", "input_tokens", "response_latency_ms", "finish_reason", "resources")}
            for row in accounting.get("warmup_outputs", [])],
        "proposal_contract": {"schema_version": 1, "base_sha256": base,
            "provenance": "editor_proposal", "composition": "independent for <=1 operation, sequential for multiple ordered operations",
            "operations": "array of existing bounded scale_columns/remove_direction_from_columns operations; no executable code"},
        "response_contract": {"required_top_level_keys": ["schema_version", "request_id", "request_sha256", "proposal"],
            "schema_version": 1, "proposal_required_keys": ["schema_version", "base_sha256", "provenance", "composition", "operations"],
            "operation_forms": {"scale_columns": {"required_keys": ["op", "tensor", "columns", "factor"]},
                "remove_direction_from_columns": {"required_keys": ["op", "tensor", "columns", "direction", "strength"],
                    "direction_length": 1024, "direction_rule": "finite nonzero vector; normalized by trusted editor"}}},
        "candidate_start": "Every candidate starts from the immutable pristine source; old integration scaling edit is not an original task-informed proposal.",
        "validation": "Host applies binary64 arithmetic to bounded coordinates, rounds to original dtype, reopens saved files, verifies all unchanged tensors/assets and actual stored changes, then converts and audits deployed quantized differences using the frozen MLC toolchain.",
        "inspection_helpers": {"CheckpointEditor.inspect": "data-only metadata/statistics/slice with persistent exposure journal",
            "CheckpointEditor.propose": "data-only bounded operations; always pristine source",
            "verify_candidate": "independent saved-checkpoint validation",
            "browser_candidate": "same converter, source binding, q4 deployment-difference audit"},
        "restrictions": ["One original proposal; no placeholder or copied integration edit.",
            "No training, gradients, optimizer, forward passes, tools, network, or extra inspections in the manual editor session.",
            "No architecture, tokenizer, template, harness, grader, prompt, decoding, or output cleanup changes.",
            "No held-out task content, outputs, or detailed overlap findings are supplied.",
            "Familiar instruction families are intentional; prompt disjointness is not proof of absent pretraining exposure."],
        "next_stage": "After manual return, import validates one data-only proposal. Candidate conversion and matched browser comparison are subsequent execution stages; no candidate or held-out generation is authorized by this packet.",
        "subsequent_commands": {"text": NEXT_COMMANDS, "private_copy": "handoff/next-commands.sh",
            "documentation": "docs/ifeval-manual-handoff.md", "executed_during_baseline_stage": False},
        "budgets": {"local_import_seconds": 900, "max_response_bytes": MAX_RESPONSE_BYTES,
            "manual_submission_slots": 1, "target_generations_in_export_or_import": 0,
            "automatic_frontier_requests": 0, "provider_frontier_token_cap": None}}
    _save(packet_dir / "request.json", request)
    text = ("Manual exploratory IFEval development handoff. Editor isolation is unverified.\n"
        "Operator: select Astra / Ultra in a fresh authenticated session. Requested selector is not observed runtime metadata.\n"
        "Supply one original task-informed weight proposal using only this packet. Return only a JSON object with exactly "
        "schema_version, request_id, request_sha256, and proposal. Do not run tools or a target forward pass.\n"
        f"request_id={request['request_id']}\nrequest_sha256={canonical_hash(request)}\n\n"
        + json.dumps(request, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (packet_dir / "editor-packet.txt").open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    (packet_dir / "editor-packet.txt").chmod(0o600)
    with (packet_dir / "next-commands.sh").open("x", encoding="utf-8") as stream:
        stream.write(NEXT_COMMANDS)
        stream.flush()
        os.fsync(stream.fileno())
    (packet_dir / "next-commands.sh").chmod(0o600)
    _save(packet_dir / "ready.json", {"request_sha256": canonical_hash(request),
        "packet_sha256": file_sha256(packet_dir / "editor-packet.txt"),
        "next_commands_sha256": file_sha256(packet_dir / "next-commands.sh"),
        "development_items_sha256": canonical_hash(items)})
    with _ledger(root, request, create=True) as ledger:
        ledger_status = ledger.summary()
    return {"exported": True, "packet": str(packet_dir / "editor-packet.txt"),
        "request_sha256": canonical_hash(request), "frontier_requests": 0,
        "editor_isolation": "unverified", "import_ledger": ledger_status}


def _load_request(root):
    root = Path(root)
    request = read_json(root / "handoff/request.json")
    ready = read_json(root / "handoff/ready.json")
    if (ready["request_sha256"] != canonical_hash(request) or
            ready["packet_sha256"] != file_sha256(root / "handoff/editor-packet.txt") or
            ready["next_commands_sha256"] != file_sha256(root / "handoff/next-commands.sh")):
        raise ValueError("manual packet or request changed")
    for relative, digest in request["evidence_files_sha256"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError("frozen handoff evidence changed")
    return request


def validate_response(response, request):
    if (not isinstance(response, dict) or set(response) != {"schema_version", "request_id", "request_sha256", "proposal"} or
            type(response["schema_version"]) is not int or response["schema_version"] != 1 or
            response["request_id"] != request["request_id"] or
            response["request_sha256"] != canonical_hash(request) or not isinstance(response["proposal"], dict) or
            response["proposal"].get("provenance") != "editor_proposal"):
        raise ValueError("invalid manual response envelope")
    return response["proposal"]


def import_response(root, response_path):
    """Future action only: consume one external submission before parsing it."""
    root, response_path = Path(root), Path(response_path)
    request = _load_request(root)
    if not response_path.is_file():
        raise ValueError("response file missing; no submission consumed")
    started = time.monotonic()
    request_id = request["request_id"]
    with _ledger(root, request) as ledger:
        row = ledger.reserve(request_id, kind="editor", sequences=1,
            max_generated_tokens=0, timeout_seconds=IMPORT_LIMITS["elapsed_seconds"])
        ledger.mark_dispatched(request_id)
        _save(root / "handoff/import-intent.json", {"request_sha256": canonical_hash(request),
            "local_ingestion_started": True, "frontier_dispatch": False, "target_generations": 0})
        try:
            if not request["created_at_epoch_seconds"] <= time.time() < request["submission_expires_at_epoch_seconds"]:
                raise ValueError("manual submission expired; slot remains consumed")
            with response_path.open("rb") as stream:
                raw = stream.read(MAX_RESPONSE_BYTES + 1)
            with (root / "handoff/supplied-response.json").open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("response exceeds frozen byte limit; bounded prefix retained")
            value = read_json_bounded(root / "handoff/supplied-response.json", max_bytes=MAX_RESPONSE_BYTES)
            proposal = validate_response(value, request)
            host = read_json(root / "weight-information/host.json")
            audit = run_checkpoint_action(model=host["model_path"], policy=request["policy"],
                exposure_log=root / "weight-information/exposures.jsonl", output=root / "candidate/source",
                tool="propose", arguments={"proposal": proposal},
                timeout_seconds=max(0.001, row["deadline_at"] - time.time() - 2),
                check_active=lambda: ledger.check_active(request_id),
                expected_original_sha256=request["checkpoint"]["base_sha256"])
            ledger.check_active(request_id)
            _save(root / "handoff/accepted-response.json", value)
            _save(root / "handoff/candidate-audit.json", audit)
            ledger.finish(request_id, status="completed", completed_sequences=1, failed_sequences=0,
                generated_tokens=0, elapsed_seconds=time.monotonic() - started)
            result = {"accepted": True, "candidate": "candidate/source", "audit": audit,
                "frontier_requests": 0, "target_generations": 0, "editor_observation": request["editor"],
                "budget": ledger.summary()}
            _save(root / "handoff/import-result.json", result)
            return result
        except Exception as error:
            ledger.finish(request_id, status="failed", completed_sequences=0, failed_sequences=1,
                generated_tokens=0, elapsed_seconds=time.monotonic() - started)
            _save(root / "handoff/import-result.json", {"accepted": False, "error": type(error).__name__,
                "frontier_requests": 0, "target_generations": 0, "budget": ledger.summary()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "export", "import"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--model")
    parser.add_argument("--policy", default="configs/candidate_policy.json")
    parser.add_argument("--items")
    parser.add_argument("--response")
    args = parser.parse_args()
    if args.command == "inspect":
        if not args.model:
            parser.error("inspect requires --model")
        result = prepare_inspections(args.root, args.model, args.policy)
    elif args.command == "export":
        result = export_packet(args.root, items_path=args.items)
    else:
        if not args.response:
            parser.error("import requires --response")
        result = import_response(args.root, args.response)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
