#!/usr/bin/env python3
"""Export one private Astra R4 development packet; never dispatch any model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weight_space_lab.ifeval_baseline import POLICY, SETTINGS
from weight_space_lab.ifeval_handoff import aggregate_feedback, baseline_feedback, _identity_only
from weight_space_lab.ifeval_v2_controls import TENSORS
from weight_space_lab.protocol import canonical_hash

BASE = "f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d"
DEV = "9aa732f4c0dbe06248a93c329299f3fb732a5f148447d3be5a057c93e5fae627"
ARMS = ("pristine", "v1", "v2-r1", "v2-r2", "v2-r3")
CLEANUP = ("unload_confirmed", "worker_terminated", "chrome_exit_confirmed",
           "local_harness_exit_confirmed", "controller_exit_verified", "chrome_os_exit_verified")


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def collect_quality(root, items, study_sha):
    feedback, summaries, bindings = {}, {}, {}
    for position, key in enumerate(ARMS, 1):
        session = root / "sessions" / f"quality-{position:02d}-{key}"
        arm = "baseline" if key == "pristine" else "candidate"
        manifest = read(session / "manifest.json")
        config = read(session / "baseline-config.json")
        graded_path = session / arm / "graded.json"
        graded = read(graded_path)
        execution_path = session / "execution" / f"{arm}-1" / "supervisor-result.json"
        execution = read(execution_path)
        if (manifest["settings"] != SETTINGS or config["settings"] != SETTINGS
                or manifest["policy"] != POLICY or manifest["items_sha256"] != DEV
                or config["configurationSha256"] != canonical_hash({
                    k: v for k, v in config.items() if k != "configurationSha256"})
                or manifest["artifact_sha256"] != config["convertedArtifactSha256"]
                or manifest["scheduled_prompts"] != 32
                or graded["manifest_sha256"] != canonical_hash(manifest)
                or graded["study_manifest_sha256"] != study_sha
                or manifest["study_manifest_sha256"] != study_sha
                or graded["summary"].get("technically_valid") is not True
                or execution.get("status") != "completed"
                or execution.get("engine_load_attempts") != 1
                or execution.get("warmup_dispatches") != 1
                or execution.get("scored_dispatches") != 32
                or execution.get("generation_dispatches") != 33
                or any(execution.get("cleanup", {}).get(k) is not True for k in CLEANUP)
                or any(r["checkpoint"]["sha256"] != config["sourceCheckpointSha256"]
                       for r in graded["records"])):
            raise ValueError(f"Fresh completed quality evidence is not bound: {key}")
        if any(r["configuration_sha256"] != config["configurationSha256"]
               for r in graded["records"]):
            raise ValueError(f"Fresh record configuration differs: {key}")
        rows = baseline_feedback(items, graded["records"], graded["grades"], POLICY)
        feedback[key] = rows
        summaries[key] = {
            "strict_passes": sum(r["strict_prompt_compliant"] for r in rows),
            "loose_passes": sum(all(c["loose"] for c in r["constraints"]) for r in rows),
            "scored_output_tokens": sum(r["token_counts"]["output"] for r in rows),
            "natural_completions": sum(r["status"] == "ok" for r in rows),
            "truncations": sum(r["status"] == "truncated" for r in rows),
            "response_seconds": sum(r["duration_seconds"] for r in rows),
            "source_checkpoint_sha256": config["sourceCheckpointSha256"],
            "deployed_artifact_sha256": config["convertedArtifactSha256"],
            "aggregate_metrics": aggregate_feedback(rows),
        }
        official = graded["summary"]["metrics"]
        if (official["prompt_level_strict_accuracy"]["passed"] != summaries[key]["strict_passes"]
                or official["prompt_level_loose_accuracy"]["passed"] != summaries[key]["loose_passes"]
                or graded["summary"]["known_generated_tokens"] != summaries[key]["scored_output_tokens"]):
            raise ValueError(f"Reported and independently recomputed scores differ: {key}")
        bindings[key] = {"graded_file_sha256": file_hash(graded_path),
                         "manifest_sha256": canonical_hash(manifest),
                         "execution_file_sha256": file_hash(execution_path)}
    return feedback, summaries, bindings


def case_feedback(feedback):
    """Retain every development observation; deduplicate only identical text."""
    cases = []
    for index, base in enumerate(feedback["pristine"]):
        constraints = [{"instruction_id": c["instruction_id"], "kwargs": c["kwargs"]}
                       for c in base["constraints"]]
        case = {"item_id": base["item_id"], "prompt": base["prompt"],
                "constraints": constraints, "observations": {}}
        seen = {}
        for key in ARMS:
            row = feedback[key][index]
            if (row["item_id"] != base["item_id"] or row["prompt"] != base["prompt"]
                    or [{"instruction_id": c["instruction_id"], "kwargs": c["kwargs"]}
                        for c in row["constraints"]] != constraints):
                raise ValueError("Cross-artifact development alignment differs")
            observation = {name: row[name] for name in (
                "status", "finish_reason", "token_counts", "duration_seconds",
                "strict_prompt_compliant", "grade_diagnostic")}
            observation["constraint_outcomes"] = [{"strict": c["strict"], "loose": c["loose"]}
                                                   for c in row["constraints"]]
            text = row["raw_response"]
            if text in seen:
                observation["raw_response_same_as"] = seen[text]
            else:
                observation["raw_response"] = text
                seen[text] = key
            case["observations"][key] = observation
        cases.append(case)
    return cases


def export(root, output, expires_at, historical_resource_report):
    root, output = Path(root).resolve(), Path(output).resolve()
    now = time.time()
    if not now < expires_at <= now + 7 * 86400:
        raise ValueError("Supply the actual future study deadline, no more than seven days away")
    if output.exists():
        raise FileExistsError("One immutable editor packet; choose no replacement for an existing packet")
    study = read(root / "study.json")
    if expires_at != study["deadline_epoch_seconds"]:
        raise ValueError("Editor expiry must equal the unchanged frozen study deadline")
    items = read(root / "baseline/data/development-items.json")
    if canonical_hash(items) != DEV or len(items) != 32:
        raise ValueError("Only the frozen 32 exposed development items are admissible")
    study_sha = canonical_hash(study)
    feedback, summaries, bindings = collect_quality(root, items, study_sha)
    inspections_path = root / "baseline/weight-information/inspections.json"
    inspections = read(inspections_path)
    if inspections["base_sha256"] != BASE or inspections.get("target_forward_passes") != 0:
        raise ValueError("Approved source statistics differ")
    policy = read(root / "baseline/weight-information/policy.json")
    if canonical_hash(policy) != inspections["policy_sha256"]:
        raise ValueError("Approved inspection policy differs")
    prior_resource = read(historical_resource_report)
    best_prior = max(row["strict_passes"] for row in summaries.values())
    request = {
        "schema_version": 1, "kind": "fresh_baseline_astra_round4_request",
        "study_id": "fresh-baseline-r4-v1", "study_sha256": study_sha,
        "request_id": str(uuid.uuid4()), "created_at_epoch_seconds": now,
        "submission_expires_at_epoch_seconds": expires_at,
        "editor": {"requested_selector": "gpt-6-astra", "observed_provider_version": None,
                   "observed_effort": None, "provider_usage": None, "isolation": "unverified"},
        "objective": "Maximize actual strict IFEval prompt compliance. Aim to exceed BOTH the fresh pristine score and the best freshly re-evaluated existing edit, while aiming for no more output tokens and same or lower measured browser resource use.",
        "objective_thresholds": {
            "fresh_pristine_strict_passes": summaries["pristine"]["strict_passes"],
            "fresh_best_existing_strict_passes": best_prior, "denominator": 32,
            "secondary_output_token_target": summaries["pristine"]["scored_output_tokens"],
            "quality_gain_with_more_tokens": "Retain, measure and report as quality/cost tradeoff; it is not discarded.",
            "resource_qualification_gate": False,
        },
        "settings": SETTINGS, "scoring_policy": POLICY,
        "checkpoint": {"model_id": "Qwen/Qwen3.5-0.8B", "base_sha256": BASE},
        "environment": _identity_only(study["environment"]),
        "edit_family": {"tensors": list(TENSORS), "operation": "scale_columns",
                        "operations": 2, "shape": [1024, 3584], "columns_per_tensor": 60,
                        "block_start_range": [0, 3524], "factors": [0.5, 0.75, 1.25, 1.5],
                        "targeted_coordinates": 122880,
                        "independent_choices": "Choose start and factor independently for each tensor; prior column positions and equal multipliers are not required."},
        "policy": policy, "approved_weight_information": _identity_only(inspections),
        "fresh_quality": summaries, "quality_evidence_bindings": bindings,
        "development_cases": case_feedback(feedback),
        "historical_resource_feedback": {
            "report_sha256": file_hash(historical_resource_report),
            "scope": "Earlier separate two-session pilot, not the forthcoming fresh resource cohort. Rates are not cumulative work; output-token counts are not compute proof. Power unavailable. The old run included a documented gap and retained both slow and fast R3 sessions.",
            "artifacts": _identity_only(prior_resource["artifacts"]),
            "aggregates": _identity_only(prior_resource["aggregates"]),
        },
        "prior_edit_history": {
            "v1": "Both tensors columns0–59, factor0.75; original editor attribution unverified.",
            "astra_1_to_3": "All used layer15 columns2014–2073 and layer19 columns3524–3583, with paired factors0.75,1.25,1.5 respectively. Their original JSON responses did not record rationales. Fresh scores above supersede historical scores for this comparison.",
            "instruction": "Use fresh paired failures/regressions and the V1 result rather than assuming the previously repeated supports are optimal. Sampled norms do not identify neuron semantics or causal task features.",
        },
        "restrictions": [
            "One original proposal from pristine; no duplicate of V1 or any prior Astra edit.",
            "No target inference, gradients, training, optimizer, network, extra weight inspection or extra data. Only read the approved packet parts; write only the final response if the parent explicitly supplies an output path.",
            "No architecture, tokenizer, prompt, template, decoding, harness, grader, answer repair or output cleanup changes.",
            "Aim at real instruction following, not memorized answers, hidden grader exceptions or output tricks.",
            "These 32 prompts are already exposed development data. No held-out prompt/output is supplied; no generalization or untouched final-quality claim follows.",
            "A fixed parameter count and allocation size do not prove lower measured compute or energy.",
        ],
        "response_contract": {"required_top_level_keys": ["schema_version", "request_id", "request_sha256", "proposal"],
            "proposal_required_keys": ["schema_version", "base_sha256", "provenance", "composition", "operations"],
            "proposal_constants": {"schema_version": 1, "base_sha256": BASE, "provenance": "editor_proposal", "composition": "sequential"},
            "operation_required_keys": ["op", "tensor", "columns", "factor"],
            "columns": "An explicit ordered list of all60 contiguous integer columns for each operation.",
            "max_response_bytes": 262144, "submission_slots": 1,
            "format": "The response artifact is JSON only, with no markdown fence or additional rationale keys."},
        "decision_summary_contract": {
            "separate_file": "rationale.txt", "maximum_words": 150,
            "content": "A concise decision summary naming the intervention and its predicted quality/resource effects, with uncertainty. Keep this separate from the response JSON.",
            "timing": "Supply once with the proposal, before candidate build or evaluation. The host hashes both files before execution. Do not revise the prediction after observing the new candidate's score.",
            "output_locations": "The parent supplies the exact approved response.json and rationale.txt paths.",
            "interpretation": "Publish as Astra's pre-evaluation prediction, separately from measured outcomes.",
        },
    }
    output.mkdir(parents=True, mode=0o700)
    save(output / "request.json", request)
    digest = canonical_hash(request)
    header = ("Astra R4: one independent proposal after five fresh quality evaluations.\n"
              "Read ALL indexed parts before responding. No new scientific feedback is supplied during delivery.\n"
              f"request_id={request['request_id']}\nrequest_sha256={digest}\n\n")
    packet = header + json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output / "editor-packet.txt").write_text(packet)
    (output / "editor-packet.txt").chmod(0o600)
    # Character chunks stay below 12,000 UTF-8 bytes, safe for a bounded tool read.
    # They are consecutive slices of the exact packet, not independently valid JSON.
    chunks, buffer, size = [], [], 0
    for character in packet:
        length = len(character.encode())
        if size + length > 12000:
            chunks.append("".join(buffer)); buffer, size = [], 0
        buffer.append(character); size += length
    if buffer:
        chunks.append("".join(buffer))
    parts = []
    for number, chunk in enumerate(chunks, 1):
        name = f"part-{number:03d}.txt"
        (output / name).write_text(chunk); (output / name).chmod(0o600)
        parts.append({"file": name, "bytes": len(chunk.encode()), "sha256": file_hash(output / name)})
    receipt = {"schema_version": 1, "request_sha256": digest,
               "request_file_sha256": file_hash(output / "request.json"),
               "packet_sha256": file_hash(output / "editor-packet.txt"),
               "packet_bytes": len(packet.encode()), "parts": parts,
               "fresh_artifacts": list(ARMS), "development_cases": 32,
               "fresh_observations": 160, "model_calls": 0, "target_generations": 0,
               "approved_inspection_file_sha256": file_hash(inspections_path),
               "charged_export_seconds": time.time() - now}
    save(output / "ready.json", receipt)
    return {k: v for k, v in receipt.items() if k != "parts"} | {"part_count": len(parts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expires-at", type=float, required=True)
    parser.add_argument("--historical-resource-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.output or args.root / "editor-round4",
                            args.expires_at, args.historical_resource_report), indent=2))


if __name__ == "__main__":
    main()
