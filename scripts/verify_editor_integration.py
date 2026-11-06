"""Persist a non-scientific Phase 7 integration exercise; no model calls.

Run from the checkout: .venv/bin/python scripts/verify_editor_integration.py --out artifacts/editor-verification
The host must permit installation of the nested restrictive macOS sandbox.
"""

import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weight_space_lab.candidate_demo import engineering_policy
from weight_space_lab.editor_budget import BudgetExhausted
from weight_space_lab.editor_controller import EditorController
from weight_space_lab.editor_contract import read_json_bounded
from weight_space_lab.native_runner import _write_json


def verify(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("verification output must be new")
    output.mkdir(parents=True)
    output = output.resolve()
    model = output / "synthetic-original"
    model.mkdir()
    tensor = "synthetic.mlp.weight"
    for name, value in (("config.json", {"architectures": ["SyntheticFixture"], "model_type": "fixture"}),
                        ("tokenizer_config.json", {"chat_template": "{{ messages }}"}),
                        ("tokenizer.json", {"fixture": True})):
        _write_json(model / name, value, immutable=True)
    header = json.dumps({tensor: {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    header += b" " * (-len(header) % 8)
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<4f", 1, 2, 3, 4))
    root = output / "session"
    controller = EditorController.create(root, model=model, policy=engineering_policy(tensor, 2, 4),
        development=[{"item_id": "synthetic-development", "split": "development",
                      "messages": [{"role": "user", "content": "Synthetic integration input; do not score."}]}],
        limits={"editor_requests": 6, "model_probes": 1, "requested_sequences": 7,
                "generated_tokens": 4096, "elapsed_seconds": 120, "billed_units": None},
        probe={"backend": {"name": "fixture", "behavior": "success", "delay_seconds": 0},
               "seeds": [17], "max_new_tokens": 4, "timeout_seconds": 5},
        interface_observation=read_json_bounded(Path(__file__).resolve().parents[1] / "configs/editor_interface_observation.json"))
    try:
        results = [controller.simulate()]
        results.append(controller.simulate(tool="inspect", arguments={"op": "metadata", "tensors": [tensor]}))
        proposal = {"schema_version": 1, "base_sha256": controller.manifest["model_info"]["checkpoint_sha256"],
                    "provenance": "editor_proposal", "composition": "independent",
                    "operations": [{"op": "scale_columns", "tensor": tensor, "columns": [0], "factor": 0.99}]}
        accepted = controller.simulate(tool="propose", arguments={"proposal": proposal})
        results.append(accepted)
        if accepted["status"] != "ok":
            raise ValueError("synthetic candidate was not accepted")
        results.append(controller.simulate(tool="probe", arguments={"task_ids": ["synthetic-development"],
                                                                    "candidate_id": accepted["result"]["candidate_id"]}))
        proposal["operations"][0]["factor"] = 0.5
        results.append(controller.simulate(tool="propose", arguments={"proposal": proposal}))
        abandoned = controller.export_request("Abandoned offline request for conservative recovery check.",
                                              max_output_tokens=32, timeout_seconds=10)
    finally:
        controller.ledger.close()
    reopened = EditorController(root)
    try:
        recovered = reopened.recover()
        exhausted = False
        try:
            reopened.export_request("Must fail before dispatch", max_output_tokens=1, timeout_seconds=1)
        except BudgetExhausted:
            exhausted = True
        checks = {"isolated_simulated_interactions_completed": all(row["status"] == "ok" for row in results[:4]),
                  "invalid_proposal_rejected": results[4]["status"] == "rejected",
                  "budget_exhaustion_denied": exhausted,
                  "restart_preserved_six_editor_requests": recovered["budget"]["charged"]["editor_requests"] == 6,
                  "interrupted_request_conservatively_failed": any(
                      row["request_id"] == abandoned["request_id"] and row["status"] == "interrupted"
                      and row["charged_generated_tokens"] == 32 for row in recovered["attempts"])}
        report = {"schema_version": 1, "kind": "phase7_offline_integration_verification", "checks": checks,
                  "all_passed": all(checks.values()), "responses": results, "budget": recovered["budget"],
                  "evidence_kind": "synthetic_fixture", "editor_model_called": False,
                  "live_astra_status": "pending_enforced_adapter", "actual_editor_identity": None,
                  "scientific_result": False, "experiment_executed": False,
                  "implementation": reopened.manifest["implementation"]}
        _write_json(output / "report.json", report, immutable=True)
        return report
    finally:
        reopened.ledger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report = verify(arguments.out)
    print(json.dumps({key: report[key] for key in ("all_passed", "checks", "scientific_result", "experiment_executed")}, indent=2))
    raise SystemExit(0 if report["all_passed"] else 2)
