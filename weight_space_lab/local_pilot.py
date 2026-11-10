"""Prepare/operate the v2 native development profile using the existing MPS pilot.

Preparation consumes no inference or proposal budget. Execution requires a new
explicit operator authorization receipt; old capacity is never imported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from .inventory import file_sha256, inventory_model, read_json
from .native_assets import implementation_identity, runtime_environment
from .native_backend import validate_backend_spec
from .phase8 import _dependencies, _new_root, _save
from .protocol import canonical_hash
from .published_study import replay_tasks
from .grading import grader_identity

KIND = "on_device_native_mps_pilot_v2"
PROFILE = Path(__file__).resolve().parents[1] / "configs/on_device_native_pilot_v2.json"
INPUTS = ("items.json", "tasks.json", "prompts.json", "design.json", "policy.json",
          "inventory.json", "inspections.json", "grading-policy.json", "exposures.jsonl",
          "backend.json", "schedule.json")


def _observation(path, expected):
    """Do not accept empty cloud-placeholder reads as fingerprints."""
    path = Path(path)
    try:
        stat = path.stat()
        if getattr(stat, "st_flags", 0) & 0x40000000:  # macOS SF_DATALESS; no implicit hydration.
            return {"expected_sha256": expected, "status": "unavailable_dataless"}
        if stat.st_size and stat.st_blocks == 0:
            return {"expected_sha256": expected, "status": "unavailable_zero_blocks"}
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError("protected file hash differs: " + str(path))
        return {"expected_sha256": expected, "status": "matched"}
    except FileNotFoundError:
        return {"expected_sha256": expected, "status": "unavailable_missing"}


def check_history(observations):
    results = {}
    for path, before in observations.items():
        after = _observation(path, before["expected_sha256"])
        if before["status"] == "matched" and after["status"] != "matched":
            raise ValueError("previously readable protected file is unavailable: " + path)
        results[path] = after
    return results


def prepare(root, *, model, previous="artifacts/phase8-mps-pilot-v1"):
    from . import phase8_mps as runner
    previous, model = Path(previous), Path(model)
    old = read_json(previous / "manifest.json")
    if old["kind"] != "phase8_manual_mps_pilot_v1" or read_json(previous / "ready.json") != {"manifest_sha256": canonical_hash(old)}:
        raise ValueError("historical MPS manifest identity differs")
    for name in INPUTS:
        if file_sha256(previous / name) != old["files"][name]:
            raise ValueError("required local input differs: " + name)
    _dependencies()
    if runtime_environment() != old["runtime"] or grader_identity("math") != old["grader"]:
        raise ValueError("pinned runtime/grader changed")
    inventory = inventory_model(model)
    if canonical_hash(inventory["files"]) != old["base_sha256"]:
        raise ValueError("recovered model is not the pinned pristine checkpoint")
    if inventory != read_json(previous / "inventory.json"):
        raise ValueError("recovered model inventory differs")
    items, tasks = read_json(previous / "items.json"), read_json(previous / "tasks.json")
    design = read_json(previous / "design.json")
    if (len(items) != 14 or any(i["split"] != "development" for i in items)
            or canonical_hash(items) != design["selected_items_sha256"]
            or canonical_hash(tasks) != design["tasks_sha256"] or replay_tasks(items) != tasks):
        raise ValueError("fixed development tasks differ")
    spec = read_json(previous / "backend.json")
    validate_backend_spec(spec)
    if spec["device"] != "mps" or spec["max_model_len"] != 8192 or spec["sampling"]["max_new_tokens"] != 4096:
        raise ValueError("historical runtime does not match v2")
    # Check tokenizer rendering offline, with no model load or forward pass.
    from .native_backend import _load_tokenizer, _prompt
    tokenizer, _, template = _load_tokenizer(model, spec)
    prompts = read_json(previous / "prompts.json")
    for task, prompt in zip(tasks, prompts, strict=True):
        token_ids, rendered = _prompt(tokenizer, template, spec, task["messages"], 0)
        import hashlib
        if (prompt["item_id"] != task["item_id"] or prompt["prompt_token_ids"] != token_ids
                or prompt["rendered_prompt_sha256"] != hashlib.sha256(rendered.encode()).hexdigest()
                or len(token_ids) + 4096 > 8192):
            raise ValueError("recovered tokenizer prompt differs")
    protected = dict(read_json(previous / "preserved-before.json"))
    protected.update({str(previous / name): digest for name, digest in old["files"].items()})
    protected.update({str(previous / "source" / name): digest for name, digest in old["source_files"].items()})
    # Include terminal v1 grades, records and spent ledgers without opening SQLite.
    for path in previous.rglob("*"):
        if path.is_file() and str(path) not in protected:
            stat = path.stat()
            if not (getattr(stat, "st_flags", 0) & 0x40000000) and (stat.st_size == 0 or stat.st_blocks > 0):
                protected[str(path)] = file_sha256(path)
    observations = {name: _observation(name, digest) for name, digest in sorted(protected.items())}
    root = _new_root(root)
    (root / "pilot").mkdir(mode=0o700)
    (root / "source").mkdir(mode=0o700)
    for name in INPUTS:
        shutil.copyfile(previous / name, root / name)
    _save(root / "deployment-profile.json", read_json(PROFILE))
    _save(root / "historical-preservation.json", observations)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, root / "source" / path.name)
    names = [*INPUTS, "deployment-profile.json", "historical-preservation.json"]
    manifest = {**{key: old[key] for key in ("base_sha256", "runtime", "grader", "lineage", "selection")},
        "schema_version": 2, "kind": KIND, "author": "Arhaan2", "strict_ready": False,
        "editor_isolation": "unverified", "model_path": str(model),
        "budgets": runner.budgets(), "environment": runner.ENVIRONMENT,
        "authorization": "proposed; fresh explicit operator authorization required; no historical capacity transferred",
        "deployment_profile_sha256": canonical_hash(read_json(PROFILE)),
        "historical_manifest_sha256": canonical_hash(old),
        "implementation_sha256": implementation_identity(),
        "history_policy": "Each readable protected file must match. Existing unavailable historical files remain explicitly unverified; they do not replace any new runtime input verification.",
        "completion_rule": "valid final boundary, status ok, and return within declared response deadline; late outputs retain independent correctness grades",
        "protocol_revision": "native development only; baseline correctness/completion not prerequisites; first-two operational resource gate retained; raw memory evidence preserved; no browser claim",
        "files": {name: file_sha256(root / name) for name in names},
        "source_files": {p.name: file_sha256(p) for p in sorted((root / "source").glob("*.py"))}}
    _save(root / "manifest.json", manifest)
    _save(root / "ready.json", {"manifest_sha256": canonical_hash(manifest)})
    for kind in ("target", "proposal"):
        with runner.ledger(root, manifest, kind, create=True) as book:
            _save(root / "pilot" / (kind + "-budget-before.json"), book.summary())
    return {"prepared": True, "execution_authorized": False, "target_generations": 0,
        "manifest_sha256": canonical_hash(manifest), "inputs_verified": True,
        "historical_matched": sum(v["status"] == "matched" for v in observations.values()),
        "historical_unavailable": sum(v["status"] != "matched" for v in observations.values())}


def load(root):
    from . import phase8_mps as runner
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("kind") != KIND or read_json(root / "ready.json") != {"manifest_sha256": canonical_hash(manifest)}:
        raise ValueError("v2 manifest differs")
    for name, digest in manifest["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("v2 frozen input differs")
    for name, digest in manifest["source_files"].items():
        if file_sha256(root / "source" / name) != digest:
            raise ValueError("v2 frozen source differs")
    if (manifest["implementation_sha256"] != implementation_identity()
            or manifest["runtime"] != runtime_environment() or manifest["grader"] != grader_identity("math")
            or manifest["budgets"] != runner.budgets() or manifest["environment"] != runner.ENVIRONMENT
            or read_json(root / "deployment-profile.json") != read_json(PROFILE)
            or manifest["deployment_profile_sha256"] != canonical_hash(read_json(PROFILE))):
        raise ValueError("v2 runtime/profile/source changed; no dispatch")
    check_history(read_json(root / "historical-preservation.json"))
    return root, manifest


def require_authorization(root, manifest):
    path = Path(root) / "pilot/fresh-authorization.json"
    if not path.is_file():
        raise ValueError("fresh explicit v2 execution authorization is required; old reservations cannot be reused")
    receipt = read_json(path)
    if (receipt.get("manifest_sha256") != canonical_hash(manifest)
            or receipt.get("deployment_profile_sha256") != manifest["deployment_profile_sha256"]
            or not isinstance(receipt.get("authorization_id"), str) or not receipt["authorization_id"].strip()):
        raise ValueError("fresh authorization receipt does not match this v2 freeze")


def authorize(root, authorization_id):
    root, manifest = load(root)
    if not isinstance(authorization_id, str) or not authorization_id.strip():
        raise ValueError("the explicit new authorization reference is required")
    _save(root / "pilot/fresh-authorization.json", {"authorization_id": authorization_id,
        "manifest_sha256": canonical_hash(manifest),
        "deployment_profile_sha256": manifest["deployment_profile_sha256"],
        "recorded_at": time.time(), "scope": "this fresh proposed v2 budget only",
        "meaning": "operator record of an actual explicit authorization; an identifier alone does not grant permission"})
    return {"recorded": True, "inference_executed": False}


def response_metrics(record, request_seconds, limit_seconds):
    within = request_seconds <= limit_seconds
    return {"response_latency_seconds": request_seconds, "response_deadline_seconds": limit_seconds,
        "response_deadline_met": within, "generation_seconds": record["duration_seconds"],
        "worker_execution_status": record["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "check", "authorize", "baseline", "export", "import", "compare"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--previous", default="artifacts/phase8-mps-pilot-v1")
    parser.add_argument("--authorization-id")
    parser.add_argument("--response", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.model is None:
            parser.error("prepare requires --model containing the verified recovered pristine bytes")
        result = prepare(args.root, model=args.model, previous=args.previous)
    elif args.command == "check":
        _, manifest = load(args.root)
        result = {"valid": True, "manifest_sha256": canonical_hash(manifest), "inference_executed": False}
    else:
        if not args.execute:
            parser.error("--execute and a separately recorded fresh explicit authorization are required")
        if args.command == "authorize":
            result = authorize(args.root, args.authorization_id)
        else:
            from . import phase8_mps as runner
            from .phase8_mps_handoff import export_packet, import_proposal, compare
            root, manifest = load(args.root)
            require_authorization(root, manifest)
            if args.command == "import" and args.response is None:
                parser.error("import requires --response")
            function = {"baseline": runner.baseline, "export": export_packet,
                        "import": import_proposal, "compare": compare}[args.command]
            result = function(root, args.response) if args.command == "import" else function(root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
