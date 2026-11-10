"""Frozen offline preparation and explicit, one-shot manual development jobs.

Nothing in prepare/check starts inference or a frontier request. This is an
exploratory controller, not a replacement for the strict editor readiness gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

from .checkpoint import validate_policy
from .editor_budget import BudgetLedger
from .editor_contract import read_json_bounded
from .editor_tools import run_checkpoint_action
from .grading import MATH_PACKAGES, grade_record, grader_identity, separate_output
from .inventory import file_sha256, inventory_model, read_json
from .native_assets import backend_spec, implementation_identity, runtime_environment
from .native_backend import _load_tokenizer, _prompt, validate_backend_spec
from .native_runner import _Process, _WorkerFailure, _failure, _lock, _missing, _write_json
from .protocol import canonical_hash
from .published_study import replay_tasks, _split
from .records import validate_run_record


def budgets():
    return {
        "calibration": {"proposals": 0, "inspections": 0, "baseline_sequences": 2,
            "candidate_sequences": 0, "target_sequences": 2, "target_tokens": 2048,
            "max_new_tokens": 1024, "retries": 0, "startup_seconds": 300,
            "sequence_seconds": 1800, "stage_seconds": 4200,
            "wall_seconds": 4500, "verification_seconds": 300, "grading_seconds": 40},
        "pilot": {"proposals": 1, "inspections": 2, "inspection_values": 12,
            "inspection_scanned_coordinates": 2048, "dynamic_inspections": 0,
            "baseline_sequences": 14, "candidate_sequences": 14,
            "target_sequences": 28, "target_tokens": 28672, "max_new_tokens": 1024,
            "retries": 0, "startup_seconds": 300, "sequence_seconds": 1800,
            "stage_seconds": 25800, "import_seconds": 900,
            "local_execution_seconds": 53400, "verification_seconds": 900,
            "inspection_seconds": 120, "wall_seconds": 259200,
            "submission_seconds": 172800, "grading_seconds_per_arm": 280,
            "frontier_requests": 1, "frontier_provider_token_cap": None,
            "frontier_usage": None, "frontier_deadline_enforcement": "operator_only",
            "operator_frontier_session_seconds": 1800,
            "max_response_bytes": 262144},
    }


def _save(path, value):
    _write_json(path, value, immutable=True)


def _dependencies():
    for name, pin in MATH_PACKAGES.items():
        if importlib.metadata.version(name) != pin:
            raise ValueError(f"pinned grader dependency required: {name}=={pin}")


def _new_root(root):
    root = Path(root)
    if root.exists() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("preparation requires a new regular private directory")
    # A private artifact location is a publication convention, not isolation.
    if "artifacts" not in root.parts:
        raise ValueError("prepare beneath ignored artifacts")
    root.mkdir(parents=True, mode=0o700)
    return root


def prepare(root, *, model, data, config="configs/phase8_development.json"):
    """Validate saved sources, tokenize only, compute two approved statistics."""
    _dependencies()
    design = read_json(config)
    items = read_json(Path(data) / "editor-development/items.json")
    indexed = {i["item_id"]: i for i in items if i["benchmark"] == "math"}
    selected = []
    for ref in design["item_refs"]:
        item = indexed[ref["item_id"]]
        if item["split"] != "development" or canonical_hash(item) != ref["item_sha256"]:
            raise ValueError("development item differs from frozen selection")
        selected.append(item)
    if len(selected) != 14 or len({i["item_id"] for i in selected}) != 14:
        raise ValueError("exactly fourteen unique frozen development items required")
    # Verify only source/split identities here, never copy held-out content.
    source_files = {"editor-development/items.json": "development_items_sha256",
        "editor-development/manifest.json": "development_manifest_sha256",
        "overlap-audit.json": "overlap_audit_sha256", "source-reference.json": "source_manifest_sha256"}
    for relative, key in source_files.items():
        if canonical_hash(read_json(Path(data) / relative)) != design["source"][key]:
            raise ValueError("pinned source or split audit changed")
    inventory = inventory_model(model)
    base = canonical_hash(inventory["files"])
    if base != "f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d":
        raise ValueError("Phase 8 requires the pristine checkpoint, never a previous candidate")
    policy = read_json("configs/candidate_policy.json")
    policy["inspection"].update(max_requests=2, max_values=6,
        max_scanned_coordinates=1024, max_total_values=12, max_total_scanned_coordinates=2048)
    validate_policy(policy, inventory)
    spec = backend_spec("transformers", inventory["chat_templates"]["chat_template.jinja"]["sha256"])
    spec["max_model_len"] = 4096
    spec["sampling"]["max_new_tokens"] = 1024
    validate_backend_spec(spec)
    tasks = replay_tasks(selected)
    if (canonical_hash(selected) != design["selected_items_sha256"] or
            canonical_hash(tasks) != design["tasks_sha256"] or
            grader_identity("math") != design["scoring"]["grader"] or
            read_json("configs/grading-policy.json") != design["scoring"]["policy"]):
        raise ValueError("frozen prompt adapter, selected items or grader differs")
    tokenizer, _, template = _load_tokenizer(Path(model), spec)
    prompts = []
    for task in tasks:
        ids, rendered = _prompt(tokenizer, template, spec, task["messages"], 0)
        prompts.append({"item_id": task["item_id"], "input_tokens": len(ids),
            "prompt_token_ids": ids, "rendered_prompt": rendered,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()})
    root = _new_root(root)
    for name in ("calibration", "pilot", "source"):
        (root / name).mkdir(mode=0o700)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, root / "source" / source.name)
    inspections = []
    for name in policy["allowed_tensors"]:
        request = {"op": "statistics", "tensor": name, "slices": [[0, 32], [0, 32]]}
        result = run_checkpoint_action(model=model, policy=policy,
            exposure_log=root / "exposures.jsonl", output=root / "inspection",
            tool="inspect", arguments=request, timeout_seconds=60, check_active=lambda: None,
            expected_original_sha256=base)
        inspections.append({"request": request, "result": result})
    payloads = {"items.json": selected, "tasks.json": tasks, "prompts.json": prompts,
        "design.json": design, "policy.json": policy, "backend.json": spec,
        "inspections.json": inspections, "inventory.json": inventory,
        "grading-policy.json": read_json("configs/grading-policy.json")}
    for name, value in payloads.items():
        _save(root / name, value)
    manifest = {"schema_version": 1, "kind": "phase8a_manual_exploratory",
        "author": "Arhaan2", "editor_isolation": "unverified", "strict_ready": False,
        "inference_authorized": False, "frontier_authorized": False,
        "model_path": str(Path(model).resolve()), "base_sha256": base,
        "implementation_sha256": implementation_identity(), "runtime": runtime_environment(),
        "grader": grader_identity("math"), "budgets": budgets(),
        "files": {name: file_sha256(root / name) for name in payloads},
        "source_files": {p.name: file_sha256(p) for p in sorted((root / "source").glob("*.py"))},
        "inspection_log_sha256": file_sha256(root / "exposures.jsonl"),
        "selection": "candidate only if all scheduled runs and grades resolve and correct count strictly exceeds baseline; otherwise pristine baseline; ties baseline",
        "lineage": "every candidate starts from pristine original; prior smoke candidate and feedback excluded",
        "calibration_reuse": "no; pilot baseline is separately budgeted and rerun",
        "scientific_result": False}
    _save(root / "manifest.json", manifest)
    _save(root / "ready.json", {"manifest_sha256": canonical_hash(manifest)})
    for phase in ("calibration", "pilot"):
        with _ledger(root, manifest, phase, create=True):
            pass
        public = {"schema_version": 1, "kind": f"phase8a_{phase}_proposal",
            "prepared_manifest_sha256": canonical_hash(manifest),
            "authorization": "not_authorized; requires a new explicit execution instruction",
            "limits": budgets()[phase], "baseline_included": True,
            "editor_isolation": "unverified", "strict_ready": False,
            "runtime": spec, "candidate_lineage": manifest["lineage"]}
        _save(root / f"{phase}-manifest.json", public)
    return {"prepared": True, "manifest_sha256": canonical_hash(manifest),
        "target_sequences_executed": 0, "frontier_requests": 0,
        "approved_inspections_completed": 2, "strict_ready": False,
        "editor_isolation": "unverified"}


def load(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if read_json(root / "ready.json") != {"manifest_sha256": canonical_hash(manifest)}:
        raise ValueError("preparation manifest changed")
    for name, digest in manifest["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("prepared input changed")
    for name, digest in manifest["source_files"].items():
        if file_sha256(root / "source" / name) != digest:
            raise ValueError("preserved implementation changed")
    if file_sha256(root / "exposures.jsonl") != manifest["inspection_log_sha256"]:
        raise ValueError("inspection exposure changed")
    if implementation_identity() != manifest["implementation_sha256"] or runtime_environment() != manifest["runtime"]:
        raise ValueError("source or environment changed; preserve this freeze and prepare a new version")
    if manifest["budgets"] != budgets() or grader_identity("math") != manifest["grader"]:
        raise ValueError("budget or grader changed")
    _dependencies()
    return root, manifest


def _ledger(root, manifest, phase, *, create=False):
    limit = manifest["budgets"][phase]
    limits = {"editor_requests": limit["proposals"], "model_probes": 1 if phase == "calibration" else 2,
        "requested_sequences": limit["target_sequences"] + limit["proposals"],
        "generated_tokens": limit["target_tokens"],
        "elapsed_seconds": limit.get("local_execution_seconds", limit["wall_seconds"]), "billed_units": None}
    return BudgetLedger(Path(root) / phase / "budget.sqlite3", run_id="phase8a-" + phase,
        limits=limits, binding={"manifest_sha256": canonical_hash(manifest)}, create=create)


def _active(root, manifest, phase):
    stamp = read_json(root / phase / "activation.json")
    if time.time() < stamp["started_at"] or time.time() >= stamp["deadline_at"]:
        raise ValueError("whole execution deadline expired; no retry")


def _activate(root, manifest, phase):
    now = time.time()
    _save(root / phase / "activation.json", {"started_at": now,
        "deadline_at": now + manifest["budgets"][phase]["wall_seconds"]})


def _identity(manifest, checkpoint_hash, arm):
    return {"run_id": "phase8a-" + arm, "profile": "fresh_editor_search",
        "configuration_sha256": canonical_hash({"manifest": canonical_hash(manifest),
            "checkpoint": checkpoint_hash, "arm": arm}),
        "checkpoint": {"model_id": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17", "sha256": checkpoint_hash}}


def _run_arm(root, manifest, phase, arm, model, checkpoint_hash, items):
    """One durable arm reservation; failures/interruptions never dispatch again."""
    spec = read_json(root / "backend.json")
    limits = manifest["budgets"][phase]
    tasks = replay_tasks(items)
    identity = _identity(manifest, checkpoint_hash, arm)
    records = [dict(schema_version=1, record_type="run", **identity, evidence_kind="real_model",
        stage="pilot", benchmark="math", dataset_revision=i["dataset_revision"],
        item_id=i["item_id"], split="development", sample_index=0, seed=0,
        retry_policy={"mode": "none", "max_attempts": 1}, attempts=[], **_missing()) for i in items]
    out = root / phase / arm
    out.mkdir(mode=0o700)  # Existing or partial run is intentionally non-resumable.
    _save(out / "schedule.json", [{"item_id": i["item_id"], "seed": 0} for i in items])
    _save(out / "identity.json", identity)
    _write_json(out / "records.json", records)
    with _ledger(root, manifest, phase) as ledger:
        _active(root, manifest, phase)
        reservation = ledger.reserve(arm, kind="probe", sequences=len(items),
            max_generated_tokens=len(items) * 1024, timeout_seconds=limits["stage_seconds"])
        def active():
            _active(root, manifest, phase)
            ledger.check_active(arm)
        worker = None
        started = time.monotonic()
        completed, token_count = 0, 0
        known_usage = True
        grades = []
        policy = read_json(root / "grading-policy.json")
        try:
            active()
            ledger.mark_dispatched(arm)
            worker = _Process(spec, model)
            whole_deadline = read_json(root / phase / "activation.json")["deadline_at"]
            worker.ready(min(300, reservation["deadline_at"] - time.time(), whole_deadline - time.time()))
            _save(out / "backend-metadata.json", worker.metadata)
            for n, task in enumerate(tasks):
                active()
                # Durable per-sequence intention survives interrupted result publication.
                _save(out / f"intent-{n:02d}.json", {"item_id": task["item_id"], "seed": 0})
                try:
                    result = worker.generate(task["messages"], 0,
                        min(limits["sequence_seconds"], reservation["deadline_at"] - time.time(), whole_deadline - time.time()))
                    active()
                    outcome = result["outcome"]
                    frozen_prompt = next(p for p in read_json(root / "prompts.json") if p["item_id"] == task["item_id"])
                    if (result["metadata"]["prompt_token_ids"] != frozen_prompt["prompt_token_ids"] or
                            result["metadata"]["rendered_prompt_sha256"] != frozen_prompt["rendered_prompt_sha256"] or
                            outcome["token_counts"]["output"] > 1024):
                        raise ValueError("runtime prompt or token bound differs from freeze")
                    _save(out / f"generation-{n:02d}.json", result)
                    completed += 1
                    token_count += outcome["token_counts"]["output"]
                except _WorkerFailure as error:
                    known_usage = False
                    outcome = _failure(error.code, timeout=error.timeout, retryable=False)
                records[n].update(outcome, attempts=[{"attempt_index": 0, **outcome}])
                validate_run_record(records[n])
                _write_json(out / "records.json", records)
                if not known_usage:
                    break
            worker.close()
            worker = None
            # Each existing grader is supervised for at most 20 seconds. Do
            # not launch one without that much remaining stage/wall budget.
            for record, item in zip(records, items):
                active()
                if min(reservation["deadline_at"], whole_deadline) - time.time() < 20:
                    raise ValueError("insufficient remaining deadline for grading; no retry")
                grades.append(grade_record(record, item, policy))
                active()
            _save(out / "grades.json", grades)
        except BaseException as error:
            # Unknown partial work retains the complete original reservation.
            known_usage = False
            _save(out / "failure.json", {"stage": "generation_or_grading",
                "error_type": type(error).__name__, "completed_sequences": completed,
                "usage_accounting": "full_reservation_retained"})
            raise
        finally:
            if worker is not None:
                worker.close()
            if known_usage:
                ledger.finish(arm, status="completed" if completed == len(items) else "failed",
                    completed_sequences=completed, failed_sequences=len(items)-completed,
                    generated_tokens=token_count, elapsed_seconds=time.monotonic()-started, billed_units=0)
            _save(out / "budget-after.json", ledger.summary())
    summary = arm_summary(records, grades, policy)
    _save(out / "summary.json", summary)
    return summary


def arm_summary(records, grades, policy):
    resolved = bool(records) and len(records) == len(grades) and all(r["status"] in {"ok", "truncated"} for r in records)
    resolved = resolved and all(g["status"] == "graded" for g in grades)
    credit = sum(g["correct"] is True for g in grades)
    return {"scheduled": len(records), "correct": sum(g["correct"] is True for g in grades),
        "accuracy": credit / len(records) if resolved else None,
        "zero_credit_bound": credit / len(records) if records else None,
        "mechanics_complete": resolved, "truncated": sum(r["status"] == "truncated" for r in records),
        "final_output_observed": any(r["raw_output"] is not None and
            separate_output(r["raw_output"], policy)[2] == "output_ready" for r in records),
        "scientific_result": False}


def calibrate(root):
    root, manifest = load(root)
    with _lock(root / "calibration"):
        _activate(root, manifest, "calibration")
        _verify_base(root, manifest, "calibration")
        ids = read_json(root / "design.json")["calibration_ids"]
        indexed = {i["item_id"]: i for i in read_json(root / "items.json")}
        items = [indexed[key] for key in ids]
        return _run_arm(root, manifest, "calibration", "calibration-baseline",
            manifest["model_path"], manifest["base_sha256"], items)


def _verify_base(root, manifest, phase):
    deadline = read_json(root / phase / "activation.json")["deadline_at"]
    return run_checkpoint_action(model=manifest["model_path"], policy=read_json(root / "policy.json"),
        exposure_log=root / phase / "verification-exposures.jsonl", output=root / phase,
        tool="verify", arguments={"candidate": False}, timeout_seconds=min(300, deadline-time.time()),
        check_active=lambda: _active(root, manifest, phase), expected_original_sha256=manifest["base_sha256"])


def baseline(root):
    root, manifest = load(root)
    calibration = read_json(root / "calibration/calibration-baseline/summary.json")
    if not calibration["mechanics_complete"] or not calibration["final_output_observed"]:
        raise ValueError("calibration needs protocol/format review, not an accuracy threshold; no pilot dispatch")
    with _lock(root / "pilot"):
        _activate(root, manifest, "pilot")
        _verify_base(root, manifest, "pilot")
        return _run_arm(root, manifest, "pilot", "baseline", manifest["model_path"],
            manifest["base_sha256"], read_json(root / "items.json"))


def export_packet(root):
    root, manifest = load(root)
    with _lock(root / "pilot"):
        _active(root, manifest, "pilot")
        summary = read_json(root / "pilot/baseline/summary.json")
        if not summary["mechanics_complete"]:
            raise ValueError("resolve baseline execution/grader failures in a new version before any frontier request")
        packet_dir = root / "pilot/handoff"
        packet_dir.mkdir(mode=0o700)
        now = time.time()
        # Exclusive export consumes the sole manual frontier handoff slot.
        # The local import ledger reservation occurs later, before parsing;
        # human waiting cannot expire/restart a local execution reservation.
        inv = read_json(root / "inventory.json")
        policy = read_json(root / "policy.json")
        request = {"kind": "phase8a_manual_request", "schema_version": 1,
            "manifest_sha256": canonical_hash(manifest),
            "editor": {"requested_selector": "gpt-6-astra", "requested_effort": "ultra",
                "accepted_selector": None, "immutable_model_version": None, "usage": None},
            "editor_isolation": "unverified", "controlled_research_run": False,
            "created_at": now, "submission_expires_at": min(now + 172800,
                read_json(root / "pilot/activation.json")["deadline_at"]),
            "operator_session_seconds": 1800, "frontier_provider_token_cap": None,
            "base_sha256": manifest["base_sha256"], "policy": policy,
            "proposal_schema": {"schema_version": 1, "base_sha256": manifest["base_sha256"],
                "provenance": "editor_proposal", "composition": "independent for zero/one operation; sequential for multiple ordered operations",
                "operations": [{"op": "scale_columns", "tensor": "allowlisted exact name", "columns": [0], "factor": 1.0},
                    {"op": "remove_direction_from_columns", "tensor": "allowlisted exact name", "columns": [0],
                     "direction": "finite nonzero array of exactly tensor.shape[0] numbers", "strength": 0.0}],
                "note": "The two operations above describe alternatives, not requested edits; return an actual bounded operations array."},
            "model_info": {"model_id": "Qwen/Qwen3.5-0.8B", "tensors":
                {name: inv["tensors"][name] for name in policy["allowed_tensors"]}},
            "inspections": read_json(root / "inspections.json"),
            "development": read_json(root / "tasks.json"),
            "baseline_records": read_json(root / "pilot/baseline/records.json"),
            "baseline_grades": read_json(root / "pilot/baseline/grades.json"),
            "scoring": read_json(root / "design.json")["scoring"],
            "selection": manifest["selection"], "budget": manifest["budgets"]["pilot"]}
        _save(packet_dir / "request.json", request)
        text = ("Manual exploratory development; editor isolation unverified. Use only the attached approved development information. "
            "Propose one data-only checkpoint edit from the pristine base. No tools, additional inspections, inference, network, "
            "training, gradients, optimizer updates, architecture changes, harness/grader changes or held-out inputs. "
            "Prior smoke proposals/results are excluded. Return only JSON with exactly schema_version (1), "
            "request_sha256 (the digest below), and proposal (existing Phase6 proposal schema, provenance editor_proposal). "
            "No retries or second proposal. Operator stops this session after 1800 seconds; this is not provider enforcement.\n"
            + "request_sha256=" + canonical_hash(request) + "\n" + json.dumps(request, indent=2) + "\n")
        (packet_dir / "editor-packet.txt").write_text(text)
        _save(packet_dir / "ready.json", {"request_sha256": canonical_hash(request),
            "packet_sha256": file_sha256(packet_dir / "editor-packet.txt")})
        return {"exported": True, "frontier_launched": False, "editor_isolation": "unverified"}


def import_proposal(root, response):
    root, manifest = load(root)
    with _lock(root / "pilot"):
        _active(root, manifest, "pilot")
        handoff = root / "pilot/handoff"
        request = read_json(handoff / "request.json")
        ready = read_json(handoff / "ready.json")
        if ready != {"request_sha256": canonical_hash(request),
                "packet_sha256": file_sha256(handoff / "editor-packet.txt")}:
            raise ValueError("manual request changed")
        if not Path(response).is_file():
            raise ValueError("response file missing")
        start = time.monotonic()
        with _ledger(root, manifest, "pilot") as ledger:
            ledger.reserve("manual-proposal", kind="editor", sequences=1,
                max_generated_tokens=0, timeout_seconds=900)
            ledger.mark_dispatched("manual-proposal")
        with Path(response).open("rb") as stream:
            raw = stream.read(262145)
        with (root / "pilot/supplied-response.json").open("xb") as stream:
            stream.write(raw)
        _save(root / "pilot/import-intent.json", {"response_file_sha256": hashlib.sha256(raw).hexdigest(),
            "started_at": time.time(), "remote_dispatch": False})
        if not request["created_at"] <= time.time() < request["submission_expires_at"]:
            raise ValueError("manual response expired; slot remains spent")
        if len(raw) > 262144:
            raise ValueError("response exceeds byte budget; bounded prefix preserved, no retry")
        value = read_json_bounded(root / "pilot/supplied-response.json")
        if (not isinstance(value, dict) or set(value) != {"schema_version", "request_sha256", "proposal"} or
                type(value["schema_version"]) is not int or value["schema_version"] != 1 or
                value["request_sha256"] != canonical_hash(request) or
                not isinstance(value["proposal"], dict) or
                value["proposal"].get("provenance") != "editor_proposal"):
            raise ValueError("response envelope invalid; no retry")
        _save(root / "pilot/response.json", value)
        # Frontier waiting never shares the local import deadline.
        def active():
            _active(root, manifest, "pilot")
            with _ledger(root, manifest, "pilot") as ledger:
                ledger.check_active("manual-proposal")
        audit = run_checkpoint_action(model=manifest["model_path"], policy=read_json(root / "policy.json"),
            exposure_log=root / "pilot/candidate-exposures.jsonl", output=root / "pilot/candidate",
            tool="propose", arguments={"proposal": value["proposal"]},
            timeout_seconds=min(900, read_json(root / "pilot/activation.json")["deadline_at"]-time.time()),
            check_active=active, expected_original_sha256=manifest["base_sha256"])
        _save(root / "pilot/candidate-audit.json", audit)
        with _ledger(root, manifest, "pilot") as ledger:
            ledger.finish("manual-proposal", status="completed", completed_sequences=1,
                failed_sequences=0, generated_tokens=0,
                elapsed_seconds=time.monotonic()-start, billed_units=None)
        _save(root / "pilot/import-result.json", {"accepted": True,
            "response_sha256": canonical_hash(value), "candidate_audit_sha256": canonical_hash(audit)})
        # This completed ledger unit is local ingestion, never frontier inference.
        return {"accepted": True, "candidate_sha256": audit["result_sha256"],
            "changed_stored_entries": audit["changed_stored_entries"], "frontier_usage": None}


def compare(root):
    from .analysis_pipeline import build_analysis_manifest, compare_graded_runs
    root, manifest = load(root)
    with _lock(root / "pilot"):
        _active(root, manifest, "pilot")
        _verify_base(root, manifest, "pilot")
        receipt = read_json(root / "pilot/import-result.json")
        response = read_json(root / "pilot/response.json")
        if receipt["response_sha256"] != canonical_hash(response):
            raise ValueError("imported response changed")
        with _ledger(root, manifest, "pilot") as ledger:
            imported = [r for r in ledger.attempts() if r["request_id"] == "manual-proposal"]
            if len(imported) != 1 or imported[0]["status"] != "completed":
                raise ValueError("candidate requires a completed local import")
        deadline = read_json(root / "pilot/activation.json")["deadline_at"]
        verification = run_checkpoint_action(model=manifest["model_path"], policy=read_json(root / "policy.json"),
            exposure_log=root / "pilot/verification-exposures.jsonl", output=root / "pilot/candidate",
            tool="verify", arguments={"candidate": True}, timeout_seconds=min(300, deadline-time.time()),
            check_active=lambda: _active(root, manifest, "pilot"), expected_original_sha256=manifest["base_sha256"])
        expected = read_json(root / "pilot/candidate-audit.json")
        full_audit = read_json(root / "pilot/candidate/audit.json")
        audit = {"original_sha256": full_audit["base_sha256"],
            **{k: full_audit[k] for k in ("result_sha256", "changed_stored_entries",
                "changed_numerical_entries", "targeted_coordinates")}}
        if (verification != {"verified": True, "original_sha256": manifest["base_sha256"]} or
                audit != expected or canonical_hash(audit) != receipt["candidate_audit_sha256"] or
                read_json(root / "pilot/candidate/proposal.json") != response["proposal"]):
            raise ValueError("candidate audit changed")
        items = read_json(root / "items.json")
        design = read_json(root / "design.json")
        policy = read_json(root / "grading-policy.json")
        arms = {"baseline": _identity(manifest, manifest["base_sha256"], "baseline"),
                "candidate": _identity(manifest, audit["result_sha256"], "candidate-evaluation")}
        analysis = build_analysis_manifest(_split(items, design["source"]["source_manifest_sha256"]), items,
            benchmark="math", seeds=[0], policy=policy, arms=arms, stage="pilot")
        _save(root / "pilot/analysis-manifest.json", analysis)
        candidate = _run_arm(root, manifest, "pilot", "candidate-evaluation", root / "pilot/candidate/model",
            audit["result_sha256"], items)
        baseline_summary = read_json(root / "pilot/baseline/summary.json")
        result = compare_graded_runs(read_json(root / "pilot/baseline/records.json"),
            read_json(root / "pilot/candidate-evaluation/records.json"),
            read_json(root / "pilot/baseline/grades.json"), read_json(root / "pilot/candidate-evaluation/grades.json"), analysis)
        _save(root / "pilot/paired-analysis.json", result)
        choice = select_candidate(baseline_summary, candidate)
        final = {"selected": choice, "selected_sha256": audit["result_sha256"] if choice == "candidate" else manifest["base_sha256"],
            "scientific_result": False, "editor_isolation": "unverified", "strict_ready": False,
            "held_out_executed": False}
        _save(root / "pilot/selection.json", final)
        return final


def select_candidate(baseline_summary, candidate_summary):
    return "candidate" if (baseline_summary["mechanics_complete"] and candidate_summary["mechanics_complete"]
        and baseline_summary["scheduled"] == candidate_summary["scheduled"] == 14
        and candidate_summary["correct"] > baseline_summary["correct"]) else "baseline"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "calibrate", "baseline", "export", "import", "compare"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--execute", action="store_true", help="operator acknowledgement of separately authorized new execution")
    args = parser.parse_args()
    if args.command in {"calibrate", "baseline", "export", "import", "compare"} and not args.execute:
        parser.error("execution is not authorized by preparation; use --execute only after a new explicit instruction")
    if args.command == "prepare":
        if args.model is None or args.data is None:
            parser.error("prepare requires --model and --data")
        result = prepare(args.root, model=args.model, data=args.data)
    elif args.command == "check":
        root, manifest = load(args.root)
        result = {"valid": True, "budgets": manifest["budgets"], "strict_ready": False, "editor_isolation": "unverified"}
    else:
        if args.command == "import" and args.response is None:
            parser.error("import requires --response")
        function = {"calibrate": calibrate, "baseline": baseline, "export": export_packet,
            "import": import_proposal, "compare": compare}[args.command]
        result = function(args.root, args.response) if args.command == "import" else function(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
