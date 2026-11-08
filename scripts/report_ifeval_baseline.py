"""Export sanitized measured results and an immutable resource packet appendix.

The appendix supplements browser resources omitted from event-only raw records;
it cannot alter raw text, grades, runtime settings, or the original request hash.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from weight_space_lab.ifeval_baseline import load, save
from weight_space_lab.ifeval_handoff import resource_summary
from weight_space_lab.inventory import file_sha256, read_json
from weight_space_lab.protocol import canonical_hash


def report(root, public):
    root, manifest = load(root)
    public = Path(public)
    graded = read_json(root/"baseline/graded.json")
    execution = read_json(root/"execution/baseline-1/supervisor-result.json")
    terminal_path = root/"execution/baseline-1/result.json"
    terminal = read_json(terminal_path)
    expected = {o["item_id"]: o for o in execution["outputs"]}
    actual = {o["item_id"]: o for o in terminal["outputs"]}
    if set(actual) != set(expected) or len(actual) != 32:
        raise ValueError("resource appendix requires the same complete development observations")
    for item_id, output in actual.items():
        for key in ("raw_output", "generated_tokens", "input_tokens", "finish_reason", "response_latency_ms"):
            if output.get(key) != expected[item_id].get(key):
                raise ValueError("terminal browser observation differs from preserved execution: "+key)
    resources = resource_summary(root, graded)
    resources.pop("per_item_browser_observations")
    events = [json.loads(line) for line in (root/"execution/baseline-1/events.jsonl").read_text().splitlines()]
    points = [event["data"] for event in events if event["type"] == "point"]
    signals = [event["data"] for event in events if event["type"] == "signal"]
    peak = {}
    for key in ("gpuBufferBytes", "wasmBytes", "heapBytes", "eventLoopDelayMs"):
        values = [point[key] for point in points if type(point.get(key)) in (int, float)]
        peak[key] = max(values) if values else None
    resources.update(worker_signals=signals, worker_sampled_maxima=peak,
        worker_sample_count=len(points),
        worker_series_scope="one complete worker run, including load, warmup, requests and cleanup; the client repeats this shared series on every terminal output",
        initialization=[event["data"] for event in events if event["type"] == "initialization-computation"])
    appendix = {"kind": "ifeval_browser_resource_appendix", "version": 1,
        "purpose": "Supplement worker resource measurements unavailable in the event-only grading envelope. No text, grades, configuration, or original request binding is changed.",
        "joined_observations": 32, "resources": resources, "cleanup": execution["cleanup"],
        "terminal_result_sha256": file_sha256(terminal_path),
        "events_sha256": file_sha256(root/"execution/baseline-1/events.jsonl")}
    save(root/"handoff/resource-appendix.json", appendix)
    core = root/"handoff/editor-packet.txt"
    bundle = root/"handoff/editor-packet-bundle.txt"
    with bundle.open("x") as stream:
        stream.write(core.read_text())
        stream.write("\n\nRESOURCE EVIDENCE APPENDIX\nThe response still uses the original request_id and request_sha256 above. This appendix supplements available worker resources; it changes neither the experiment nor the response schema.\n")
        stream.write(json.dumps(appendix, indent=2)+"\n")
    bundle.chmod(0o600)
    request = read_json(root/"handoff/request.json")
    binding = {"kind": "ifeval_private_handoff_bundle", "original_request_sha256": canonical_hash(request),
        "files": {name: file_sha256(root/name) for name in ["handoff/editor-packet.txt", "handoff/resource-appendix.json",
            "handoff/editor-packet-bundle.txt", "execution/baseline-1/result.json", "execution/baseline-1/events.jsonl",
            "baseline/graded.json", "manifest.json"]},
        "original_request_binding_unchanged": True, "score_changes": 0}
    save(root/"handoff/bundle-manifest.json", binding)
    data = read_json(root/"data/manifest.json")
    config = read_json(root/"baseline-config.json")
    runtime = read_json(root/"runtime-limit-audit.json")
    coverage = {role: {key: data[role][key] for key in ["prompts", "instructions", "instruction_families", "instruction_types"]}
                for role in ["development", "evaluation"]}
    summary = graded["summary"]
    budget = execution["generation_budget"]
    result = {"schema_version": 1, "kind": "ifeval_development_baseline_report", "author": "Arhaan2",
        "scope": "exploratory on-device experiment; editor isolation unverified",
        "status": execution["status"], "checkpoint": manifest["checkpoint"],
        "artifact_sha256": manifest["artifact_sha256"], "configuration_sha256": manifest["configuration_sha256"],
        "manifest_sha256": canonical_hash(manifest), "data_manifest_sha256": canonical_hash(data),
        "sources": data["sources"], "coverage": coverage, "source_counts": data["counts"],
        "selection_rule": data["selection_rule"], "selection_seed": 42,
        "practice_exclusions": data["practice_exclusion_reasons"], "evaluation_exclusions": data["evaluation_exclusion_reasons"],
        "overlap": {"exact_and_normalized_selected_prompt_overlap": 0,
                    "familiar_types_intentional": True, "semantic_paraphrases_and_pretraining_exposure_unverified": True},
        "runtime": runtime, "settings": config["settings"], "grading_policy": manifest["policy"], "grader": manifest["grader"],
        "summary": summary, "resources": resources, "cleanup": execution["cleanup"],
        "budget": {"limits": manifest["limits"], "engine_load_attempts": execution["engine_load_attempts"],
            "warmup_dispatches": execution["warmup_dispatches"], "scored_dispatches": execution["scored_dispatches"],
            "generation_dispatches": execution["generation_dispatches"], "reserved": budget["reserved"],
            "observed": budget["observed"], "charged": budget["charged"], "terminal": budget["cancelled"],
            "candidate_generations": 0, "heldout_generations": 0, "experimental_frontier_requests": 0,
            "first_load_through_cleanup_seconds": execution["first_load_through_cleanup_seconds"]},
        "warmup": [{key: row.get(key) for key in ["generated_tokens", "input_tokens", "response_latency_ms", "finish_reason", "assistant_prefix_tokens", "request_max_tokens"]}
                   for row in execution["warmup_outputs"]],
        "load_ms": next(event["data"] for event in events if event["type"] == "loaded"),
        "attribution": ["Arhaan Aggarwal (Arhaan2)", "Existing BrowserLLM work and Gufran's contribution preserved", "Google Research IFEval, Zhou et al., Apache-2.0", "Allen Institute for AI RLVR-IFeval; upstream Ai2 ImpACT and Llama 2 notices preserved"]}
    save(public/"result.json", result)
    metrics = summary["metrics"]
    lines = ["# IFEval browser development baseline", "", "Arhaan2. The pristine Qwen3.5-0.8B baseline completed locally using WebLLM 0.2.86, WebGPU, and the existing q4f16_1 artifact. This is an exploratory development measurement; editor isolation remains unverified.", "",
        "| Metric | Passed / scheduled | Compliance |", "| --- | ---: | ---: |"]
    for name, label in [("prompt_level_strict_accuracy", "Strict prompt (primary)"), ("instruction_level_strict_accuracy", "Strict instruction"), ("prompt_level_loose_accuracy", "Loose prompt"), ("instruction_level_loose_accuracy", "Loose instruction")]:
        m=metrics[name]
        lines.append(f"| {label} | {m['passed']} / {m['scheduled']} | {m['value']:.2%} |")
    states=summary["execution_status_counts"]
    lines += ["", f"All 32 scheduled development prompts remain in the denominator: {states.get('ok',0)} natural completions, {states.get('truncated',0)} length truncations, {states.get('missing',0)} unexecuted, and {summary['unresolved_grader_errors']} unresolved grader errors. Truncated text was graded under the frozen policy. No response retries or cleanup rewrites occurred.", "",
        "Strict failures by instruction type: "+", ".join(f"`{name}` {count}" for name,count in summary["strict_failures_by_instruction"].items())+".", "",
        f"Scored generation used {summary['known_generated_tokens']:,} reported completion tokens and {summary['known_input_tokens']:,} input tokens over {summary['generation_seconds']:.3f} seconds. Engine loading took {result['load_ms']/1000:.3f} seconds. First durable load grant through final cleanup took {execution['first_load_through_cleanup_seconds']:.3f} seconds of 5,400.", "",
        f"Consumption: one engine-load attempt, one warmup, 32 scored dispatches, {budget['observed']['generated_tokens']:,} of 32,784 allowed reported completion tokens, and no retries. The warmup was capped at 16 tokens and accounted separately. WebLLM's four injected empty-thinking prefix IDs count toward `max_tokens` but not reported sampled tokens; the 1,024-token setting therefore permits at most 1,020 sampled tokens. Candidate, held-out, and experimental frontier consumption remain zero.", "",
        f"The newly frozen data contains 32 Ai2 practice prompts (32 instructions) and a separate 128-prompt Google IFEval subset (173 instructions). Both cover 18 instruction types across eight families. Pre-output screening excluded {data['counts']['practice_excluded']:,} practice records and {data['counts']['evaluation_excluded']} Google records. Selection uses seed 42 and balances supported instruction types with stable hashed tie-breaking. Exact/normalized selected prompts are disjoint; semantic paraphrases and model pretraining exposure remain unverified. Prior Google study partitions were preserved.", "",
        "The actual compiled WASM supports the effective 4,096-token context, verified from embedded metadata and runtime source. Exact deployed-tokenizer counts agree for all 160 selected prompts. Development input spans 44–1,644 tokens and evaluation input 42–238 tokens; every input plus the 1,024-token output allowance fits. This feasibility check was frozen before any new output.", "",
        f"Observed worker descriptor accounting peaked at {peak['gpuBufferBytes']:,} GPUBuffer bytes and {peak['wasmBytes']:,} WASM address-space bytes. Neither is physical VRAM. JS heap was unavailable. Host-wide readings and owned-process footprints are reported separately in the result, with shared-memory and sampling limits. Engines unloaded, the worker and dedicated browser exited, and no owned process remained. The collector's Xcode loader warning did not prevent scoped resource sampling.", "",
        "The private Astra/Ultra packet contains all development responses, independent per-constraint grades, recorded bounded tensor statistics, and exact subsequent import/conversion/comparison commands. Its resource appendix is joined to immutable execution evidence. Requested editor identity is separate from unavailable observed model/version/effort/usage/billing. The original proposal is still required; no placeholder or integration edit was supplied.", "",
        "The existing official Google checkers, independent grading and paired-analysis pipeline, checkpoint editor, conversion audits, browser v2 asset serving, and resource watchdog are reused. The Ai2 adapter has explicit supported mappings and 42 passing official known-answer checks. Detailed eligibility, coverage, hashes, resources and accounting appear in [result.json](result.json); future commands are in the [manual handoff procedure](../../docs/ifeval-manual-handoff.md).", "",
        "Sources: [Ai2 RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), [pinned Google IFEval](https://github.com/google-research/google-research/tree/e6890f85757dd84e27ca6df2dd30651dafad28e0/instruction_following_eval). Upstream licenses, scientific authorship, and Gufran's contribution remain preserved. Private prompts, packets, machine paths and session records are excluded from public artifacts.", ""]
    public.mkdir(parents=True,exist_ok=True)
    with (public/"README.md").open("x") as stream:
        stream.write("\n".join(lines))
    return {"public_result": str(public/"result.json"), "private_bundle": str(bundle), "metrics": metrics}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="artifacts/ifeval-browser-v1")
    parser.add_argument("--public",default="reports/ifeval-browser-v1")
    args=parser.parse_args()
    print(json.dumps(report(args.root,args.public),indent=2))
