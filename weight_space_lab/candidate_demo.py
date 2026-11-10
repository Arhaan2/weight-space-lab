"""A deliberately hand-authored candidate edit and development-only smoke.

This module does not call Astra, search for interventions, grade a benchmark,
or select a candidate using model feedback. Its proposal is fixed in code.
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile

from .acquisition import _remove_stage
from .inventory import file_sha256, inventory_model
from .native_assets import _sync_directory, _write
from .protocol import canonical_hash


DEMO_TENSOR = "model.language_model.layers.15.mlp.down_proj.weight"
DEVELOPMENT_INPUT = {
    "benchmark": "engineering-development-arithmetic",
    "item_id": "authored-addition-7-5",
    "split": "development",
    "origin": "hand_authored_engineering_input",
    "messages": [{"role": "user", "content": "Compute 7 + 5."}],
}


def engineering_policy(name, rows, total):
    """Restrict the demonstration to one declared column and one operation."""
    return {
        "schema_version": 1,
        "allowed_tensors": [name],
        "allowed_operations": ["scale_columns"],
        "allowed_columns": {name: [0]},
        "max_changed_fraction": rows / total,
        "max_changed_coordinates": rows,
        "max_targeted_coordinates": rows,
        "max_operations": 1,
        "max_working_elements": rows,
        "scale_factor_range": [0.99, 0.99],
        "direction_strength_range": [0.0, 0.0],
        "max_abs_change": 1.0,
        "inspection": {
            "allowed_tensors": [name], "max_requests": 3,
            "max_values": 16, "max_scanned_coordinates": 16,
            "max_total_values": 32, "max_total_scanned_coordinates": 32,
        },
    }


def _development_generation(model, inventory, task, backend):
    """One bounded native request; retain failures without fabricating outputs."""
    from .native_assets import backend_spec, implementation_identity, runtime_environment
    from .native_runner import _Process, _WorkerFailure

    template = inventory["chat_templates"]["chat_template.jinja"]["sha256"]
    spec = backend_spec(backend, template)
    spec["sampling"].update(mode="greedy", temperature=0, top_p=1, top_k=0,
                            max_new_tokens=4)
    result = {
        "schema_version": 1, "kind": "engineering_development_generation",
        "intervention_origin": "hand_authored_engineering_edit",
        "astra_discovered": False, "scientific_score": None,
        "task_sha256": canonical_hash(task), "split": "development",
        "checkpoint_sha256": canonical_hash(inventory["files"]),
        "backend_spec": spec, "seed": 17,
        "limits": {"max_sequences": 1, "max_new_tokens": 4,
                   "startup_timeout_seconds": 120, "request_timeout_seconds": 60},
        "implementation_sha256": implementation_identity(),
        "runtime_environment": runtime_environment(),
    }
    worker = _Process(spec, model)
    try:
        worker.ready(120)
        result["backend_metadata"] = worker.metadata
        result["generation"] = worker.generate(task["messages"], 17, 60)
        result["status"] = result["generation"]["outcome"]["status"]
    except _WorkerFailure as error:
        result.update(status="timeout" if error.timeout else "error", error=error.code)
    finally:
        worker.close()
    return result


def demo_candidate_edit(model, output, *, backend=None, tensor=DEMO_TENSOR):
    """Save independent identity/edit candidates and optionally generate 4 tokens.

    The demonstration uses an authored development input. It never consumes
    validation/final tasks or interprets a generation as improved capability.
    Existing outputs and outputs nested in the original checkpoint are refused.
    """
    from .checkpoint import CheckpointEditor, verify_candidate

    model, output = Path(model), Path(output)
    if backend not in (None, "transformers"):
        raise ValueError("engineering demo supports only explicit transformers inference")
    if output.exists() or output.is_symlink():
        raise FileExistsError("engineering demo output must be new")
    if output.resolve() == model.resolve() or model.resolve() in output.resolve().parents:
        raise ValueError("engineering demo must be outside the immutable original")
    original = inventory_model(model)
    target = original["tensors"].get(tensor)
    if (target is None or target["dtype"] not in {"BF16", "F16", "F32", "F64"}
            or len(target["shape"]) != 2 or not all(target["shape"])):
        raise ValueError("engineering demo requires the declared nonempty floating matrix")
    rows = target["shape"][0]
    policy = engineering_policy(tensor, rows, original["parameter_count"])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        editor = CheckpointEditor(model, policy, exposure_log=stage / "exposures.jsonl")
        editor.inspect({"op": "metadata", "tensors": [tensor]})
        slices = [[0, min(4, rows)], [0, 1]]
        editor.inspect({"op": "statistics", "tensor": tensor, "slices": slices})
        editor.inspect({"op": "slice", "tensor": tensor, "slices": slices})
        identity = {
            "schema_version": 1, "base_sha256": editor.original_sha256,
            "provenance": "engineering_edit", "composition": "independent", "operations": [],
        }
        proposal = copy.deepcopy(identity)
        proposal["operations"] = [{"op": "scale_columns", "tensor": tensor,
                                   "columns": [0], "factor": 0.99}]
        _write(stage / "policy.json", policy)
        _write(stage / "identity-proposal.json", identity)
        _write(stage / "engineering-proposal.json", proposal)
        _write(stage / "development-input.json", DEVELOPMENT_INPUT)
        audits = {}
        for arm, patch in (("identity", identity), ("engineering", proposal)):
            editor.propose(patch, stage / arm)
            audits[arm] = verify_candidate(model, stage / arm, policy)
        # This deliberate rejected proposal proves the fixed policy is active.
        rejected = copy.deepcopy(proposal)
        rejected["operations"][0]["factor"] = 0.5
        try:
            editor.propose(rejected, stage / "rejected")
        except ValueError:
            rejection_checked = not (stage / "rejected").exists()
        else:
            raise ValueError("engineering policy unexpectedly accepted the rejected proposal")
        if not rejection_checked:
            raise ValueError("rejected proposal left a candidate output")
        _write(stage / "rejected-proposal.json", rejected)
        generations = {}
        if backend is not None:
            for arm in ("identity", "engineering"):
                # Reopen/verify immediately before loading the candidate.
                verify_candidate(model, stage / arm, policy)
                saved_inventory = inventory_model(stage / arm / "model")
                generations[arm] = _development_generation(
                    stage / arm / "model", saved_inventory, DEVELOPMENT_INPUT, backend)
                _write(stage / f"{arm}-generation.json", generations[arm])
        report = {
            "schema_version": 1, "kind": "engineering_candidate_edit_demonstration",
            "intervention_origin": "hand_authored_engineering_edit",
            "astra_discovered": False, "editor_called": False,
            "experiment_executed": False, "scientific_result": False,
            "explanation": "A fixed 0.99 column scale checks implementation; it was not discovered by Astra and makes no improvement claim.",
            "development_input_sha256": canonical_hash(DEVELOPMENT_INPUT),
            "original_sha256": editor.original_sha256,
            "policy_sha256": canonical_hash(policy),
            "proposal_sha256": canonical_hash(proposal),
            "exposure_log_sha256": file_sha256(stage / "exposures.jsonl"),
            "target_tensor": tensor, "target_dtype": target["dtype"],
            "identity_verified": True, "valid_edit_verified": True,
            "rejected_edit_left_no_candidate": rejection_checked,
            "candidate_audits": {arm: file_sha256(stage / arm / "audit.json") for arm in audits},
            "saved_changes": {
                arm: {key: audit[key] for key in (
                    "changed_stored_entries", "changed_numerical_entries",
                    "bit_only_changed_entries", "targeted_coordinates",
                    "total_stored_coordinates", "unchanged_tensor_count")}
                for arm, audit in audits.items()
            },
            "stored_tensor_count": len(original["tensors"]),
            "stored_dtypes": sorted({info["dtype"] for info in original["tensors"].values()}),
            "inference_requested": backend is not None,
            "inference_executed": any(value["status"] in {"ok", "truncated"} for value in generations.values()),
            "generation_statuses": {arm: value["status"] for arm, value in generations.items()},
            "generation_scope": "At most one authored development input, seed 17 and four output tokens per arm; no grades or candidate selection.",
            "checks_scope": "Identity, valid edit, rejected edit, and saved-byte verification; injected failed-write checks run in the fixture test suite.",
        }
        if inventory_model(model) != original:
            raise ValueError("immutable original changed during the engineering demonstration")
        _write(stage / "report.json", report)
        _sync_directory(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError("engineering demo output appeared during preparation")
        stage.rename(output)
    except BaseException:
        _remove_stage(stage)
        raise
    return report
