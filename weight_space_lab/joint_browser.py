"""One IFEval inference workload, independently graded and resource-measured.

The existing bounded browser runner remains responsible for artifact validation,
dispatch accounting, timeouts and cleanup. This adapter enables its full native
collector for task requests, then joins grades and telemetry by frozen item IDs.
It never performs another generation to obtain resource measurements.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from .ifeval_baseline import POLICY, SETTINGS, save
from .inventory import file_sha256, read_json
from .protocol import canonical_hash

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = "ifeval-joint-browser-v1"


def _replace_once(text, old, new, name):
    if text.count(old) != 1:
        raise ValueError("Pinned browser adapter source differs: " + name)
    return text.replace(old, new, 1)


def install_adapter(benchmark):
    """Install in a new copied resource-enabled harness; does not build or infer.

    Refuses unexpected source and verifies subsequent calls against its receipt.
    Run the upstream build-worker.mjs afterward (it builds worker and task client).
    """
    benchmark = Path(benchmark).resolve()
    receipt = benchmark / "joint-adapter-install.json"
    if receipt.exists():
        value = read_json(receipt)
        if any(file_sha256(benchmark / p) != digest for p, digest in value["after"].items()):
            raise ValueError("Installed joint adapter changed")
        if value["adapter_sha256"] != file_sha256(__file__) or value["helper_sha256"] != file_sha256(SOURCE_ROOT / "integrations/joint-resource-report.ts"):
            raise ValueError("Installed joint adapter revision differs; use a fresh harness copy")
        return value
    names = ["src/benchmark/task-evaluation/protocol.ts", "src/benchmark/task-evaluation/client.ts", "scripts/run-task-evaluation.mjs"]
    originals = {name: (benchmark / name).read_text() for name in names}
    changes = dict(originals)
    name = names[0]
    changes[name] = _replace_once(changes[name], "    resourceComparison: z.literal(true).optional(),",
        "    resourceComparison: z.literal(true).optional(),\n    jointMeasurement: z.literal(true).optional(),\n    jointPromptCount: z.number().int().min(1).max(128).optional(),", name)
    changes[name] = _replace_once(changes[name], '    const continuation = c.evaluationStage === "heldout-continuation";',
        '    if (c.jointMeasurement && (!c.study || c.resourceComparison || c.evaluationStage || c.jointPromptCount !== c.items.length))\n'
        '      ctx.addIssue({ code: "custom", message: "Joint measurement requires its own IFEval schedule" });\n'
        '    if (c.jointPromptCount !== undefined && !c.jointMeasurement)\n'
        '      ctx.addIssue({ code: "custom", message: "Joint count requires joint measurement" });\n'
        '    const continuation = c.evaluationStage === "heldout-continuation";', name)
    changes[name] = _replace_once(changes[name], 'c.items.length !== (continuation ? 34 : c.evaluationStage === "heldout" ? (c.narrowedFinalPromptCount ?? 128) : 32)',
        'c.items.length !== (c.jointMeasurement ? c.jointPromptCount : continuation ? 34 : c.evaluationStage === "heldout" ? (c.narrowedFinalPromptCount ?? 128) : 32)', name)
    name = names[1]
    changes[name] = _replace_once(changes[name], 'import { resourceReport } from "../resource-comparison/report";',
        'import { resourceReport } from "../resource-comparison/report";\nimport { jointReport } from "./joint-resource-report";', name)
    changes[name] = _replace_once(changes[name], '  const standard = task.resourceComparison ? resourceReport(task) : undefined;',
        '  const joint = task.jointMeasurement ? jointReport(task) : undefined;\n  const standard = task.resourceComparison ? resourceReport(task) : undefined;', name)
    changes[name] = _replace_once(changes[name], '  const startedEpochMs = standard?.report.profile!.startedEpochMs ?? clockMs();',
        '  const startedEpochMs = joint?.report.profile.startedEpochMs ?? standard?.report.profile!.startedEpochMs ?? clockMs();', name)
    changes[name] = _replace_once(changes[name], '    standard?.observe(type, data);', '    standard?.observe(type, data);\n    joint?.observe(type, data);', name)
    changes[name] = _replace_once(changes[name], '        standard_report: standard?.finish(done, forced),',
        '        standard_report: standard?.finish(done, forced),\n        joint_resource_report: joint?.finish(done, forced),', name)
    name = names[2]
    changes[name] = _replace_once(changes[name], '  if (config.resourceComparison) {\n    const capture = nativeCapture',
        '  if (config.resourceComparison || config.jointMeasurement) {\n    const capture = nativeCapture', name)
    changes[name] = _replace_once(changes[name], '"--max-seconds", "1800"], "native-capture");',
        '"--max-seconds", String(config.jointMeasurement ? 5400 : 1800)], "native-capture");', name)
    changes[name] = _replace_once(changes[name], ': !config.resourceComparison,', ': !(config.resourceComparison || config.jointMeasurement),', name)
    changes[name] = _replace_once(changes[name], '(!config.resourceComparison || result.cleanup.native_capture_exit_confirmed)',
        '(!(config.resourceComparison || config.jointMeasurement) || result.cleanup.native_capture_exit_confirmed)', name)
    helper = "src/benchmark/task-evaluation/joint-resource-report.ts"
    destination = benchmark / helper
    if destination.exists():
        raise FileExistsError("Unexpected joint resource helper already present")
    # All anchors and destinations are validated before source files are changed.
    for name, content in changes.items():
        (benchmark / name).write_text(content)
    destination.write_bytes((SOURCE_ROOT / "integrations/joint-resource-report.ts").read_bytes())
    value = {"schema_version": 1, "kind": "joint_browser_adapter_install", "target_calls": 0,
        "adapter_sha256": file_sha256(__file__), "helper_sha256": file_sha256(destination),
        "before": {name: __import__("hashlib").sha256(content.encode()).hexdigest() for name, content in originals.items()},
        "after": {name: file_sha256(benchmark / name) for name in [*names, helper]}}
    save(receipt, value)
    return value


def prepare_session(root, *, benchmark, config, items, checkpoint, official_ifeval_root,
                    study_sha256, environment, browser_seconds=5400):
    """Freeze exactly one arm's scored requests; no engine or model is loaded."""
    from . import ifeval_v2_runtime as runtime
    from .browser_resource import served_build
    from .grading import grader_identity, verify_ifeval_assets
    from .benchmark_data import validate_item
    root, benchmark = Path(root).resolve(), Path(benchmark).resolve()
    if root.exists():
        raise FileExistsError("No replacement or repeat of a prepared joint session")
    if type(browser_seconds) not in (int, float) or not 140 <= browser_seconds <= 5400:
        raise ValueError("Browser allowance must leave request/cleanup time and not exceed 5400 seconds")
    install = read_json(benchmark / "joint-adapter-install.json")
    if any(file_sha256(benchmark / p) != d for p, d in install["after"].items()):
        raise ValueError("Joint browser adapter changed since installation")
    tasks = copy.deepcopy(read_json(items) if isinstance(items, (str, Path)) else items)
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 128:
        raise ValueError("Exactly one IFEval schedule of 1–128 items is required")
    for item in tasks:
        validate_item(item)
        if item["benchmark"] != "ifeval" or item["split"] not in ("development", "test"):
            raise ValueError("This harness has one accuracy benchmark: IFEval")
    if len({i["item_id"] for i in tasks}) != len(tasks) or len({i["split"] for i in tasks}) != 1:
        raise ValueError("Duplicate item IDs or mixed evaluation splits")
    value = copy.deepcopy(read_json(config) if isinstance(config, (str, Path)) else config)
    if value["settings"] != SETTINGS or value["sourceCheckpointSha256"] != checkpoint["sha256"]:
        raise ValueError("Joint configuration settings/checkpoint differ")
    if [{"item_id": i["item_id"], "prompt": i["prompt"]} for i in value["items"]] != [
            {"item_id": i["item_id"], "prompt": i["task_text"]} for i in tasks]:
        raise ValueError("Browser prompts differ from scored schedule")
    if any(type(i.get("rendered_input_tokens")) is not int or not 0 < i["rendered_input_tokens"] <= 3072 for i in value["items"]):
        raise ValueError("Frozen rendered input-token counts are required")
    for name in ("resourceComparison", "resourceArtifact", "evaluationStage", "narrowedFinalPromptCount", "originalScheduleOffset", "originalScheduleLength", "configurationSha256", "integrationGate"):
        value.pop(name, None)
    value.update(study="ifeval-browser-v1", jointMeasurement=True, jointPromptCount=len(tasks),
                 warmupPrompt="Reply with OK.", totalLimitMs=5400000)
    origin = urlparse(value["model"]["modelUrl"])
    value["servedBuild"] = served_build(benchmark, origin.scheme + "://" + origin.netloc)
    # The built closure must include the new profile hook, not just the old task client.
    if "joint_resource_report" not in (benchmark / "public/task-evaluation.js").read_text():
        raise ValueError("Joint task client must be rebuilt before freezing")
    value["configurationSha256"] = canonical_hash(value)
    assets = verify_ifeval_assets(official_ifeval_root)
    root.mkdir(parents=True, mode=0o700)
    (root / "benchmark").symlink_to(benchmark, target_is_directory=True)
    save(root / "items.json", tasks)
    save(root / "baseline-config.json", value)
    files = [root / "baseline-config.json", root / "items.json", benchmark / "joint-adapter-install.json"]
    files += [benchmark / p for p in install["after"]]
    files += list((benchmark / "harness").rglob("*.py"))
    files += list((SOURCE_ROOT / "weight_space_lab").glob("*.py"))
    harness_files = set(benchmark / p for p in install["after"])
    for directory, patterns in (("src", ("*.ts",)), ("scripts", ("*.mjs", "*.js", "*.py")),
                                ("harness", ("*.py",))):
        for pattern in patterns:
            harness_files.update((benchmark / directory).rglob(pattern))
    harness_files.update(p for p in (benchmark / "package.json", benchmark / "package-lock.json") if p.is_file())
    for asset in value["servedBuild"]:
        harness_files.add(benchmark / "public" / urlparse(asset["url"]).path.lstrip("/"))
    files += sorted(harness_files)
    harness_sha256 = canonical_hash({str(p.relative_to(benchmark)): file_sha256(p) for p in sorted(harness_files)})
    manifest = {"schema_version": 1, "kind": "joint_browser_ifeval_session", "arm": value["arm"],
        "run_id": value["runId"], "study_manifest_sha256": study_sha256,
        "configuration_sha256": value["configurationSha256"], "artifact_sha256": value["convertedArtifactSha256"],
        "checkpoint": checkpoint, "scheduled_prompts": len(tasks), "items_path": str(root / "items.json"),
        "items_sha256": canonical_hash(tasks), "settings": SETTINGS, "policy": POLICY,
        "workload_identity": WORKLOAD, "browser_seconds": browser_seconds,
        "grader": grader_identity("ifeval"), "official_ifeval_root": str(Path(official_ifeval_root).resolve()),
        "official_assets": assets, "environment": environment,
        "harness_sha256": harness_sha256,
        "files": {str(p.resolve()): file_sha256(p) for p in files},
        "model_lib_sha256": next(a["sha256"] for a in value["assets"] if a["url"] == value["model"]["modelLib"]),
        "resource_measurement": "same scored requests; one warmup excluded; no separate repeat workload",
        "replication": "one session per arm; descriptive paired comparison, no equivalence claim"}
    save(root / "manifest.json", manifest)
    save(root / "ready.json", {"manifest_sha256": canonical_hash(manifest)})
    with runtime.ledger(root, manifest, create=True) as book:
        save(root / "accounting-before.json", book.summary())
    return manifest


def _number(value, name, *, nullable=True, integer=False):
    if value is None and nullable:
        return None
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative measurement")
    return value


def interval_metric(intervals, windows):
    """Count only complete observed intervals inside request windows, once.

    Does not clip cross-boundary intervals or impute uniform rates to them.
    The integral is observed covered work/energy, not the unobserved full cost.
    """
    duration = sum(end - start for start, end in windows)
    weighted = covered = 0.0
    previous = -math.inf
    for start, end, value in sorted(intervals, key=lambda row: (row[0], row[1])):
        if value is None or not all(type(x) in (int, float) and math.isfinite(x) for x in (start, end, value)) or end <= start or value < 0:
            continue
        if not any(start >= a and end <= b for a, b in windows):
            continue
        start = max(start, previous)
        if end <= start:
            continue
        weighted += value * (end - start)
        covered += end - start
        previous = end
    return {"mean": weighted / covered if covered else None,
        "coverage": min(1.0, covered / duration) if duration else 0.0,
        "covered_ms": covered, "observed_integral_seconds": weighted / 1000 if covered else None}


def resource_windows(report, windows):
    """Summarize native and browser signals within the actual request bounds."""
    trace = report.get("nativeTrace") or {}
    points = sorted(trace.get("points", []), key=lambda p: p["tMs"])
    logical = trace.get("logicalCpus", 0)
    cpu = interval_metric([(points[i - 1]["tMs"] if i else p["tMs"], p["tMs"],
        p.get("cpuOneCorePercent") / logical if logical and p.get("incompleteCpuIntervals") == 0 and p.get("cpuOneCorePercent") is not None else None)
        for i, p in enumerate(points)], windows)
    gpu = interval_metric([(p.get("startMs") if p.get("startMs") is not None else p["tMs"], p["tMs"],
        p["gpuTimeMs"] * 1000 / p["intervalMs"] if p.get("gpuTimeMs") is not None and p.get("intervalMs", 0) > 0 and not p.get("issues") else None)
        for p in trace.get("gpu", [])], windows)
    power = interval_metric([(p["tMs"] - p["intervalMs"], p["tMs"], p.get("combinedWatts"))
        for p in trace.get("power", [])], windows)
    def peak(rows, key):
        values = [p[key] for p in rows if p.get(key) is not None and any(a <= p["tMs"] < b for a, b in windows)]
        for v in values:
            _number(v, key)
        return max(values) if values else None
    footprint = peak(points, "footprintBytes")
    browser = report.get("profile", {}).get("points", [])
    cpu_ms = cpu["observed_integral_seconds"] * logical * 10 if cpu["observed_integral_seconds"] is not None else None
    return {"native_peak_footprint_gb": footprint / 1e9 if footprint is not None else None,
        "gpu_buffer_descriptor_bytes_sampled_max": peak(browser, "gpuBufferBytes"),
        "wasm_address_space_bytes_sampled_max": peak(browser, "wasmBytes"),
        "cpu_capacity_percent": cpu["mean"], "cpu_coverage": cpu["coverage"],
        "observed_cpu_time_ms": cpu_ms, "cpu_time_ms": cpu_ms if cpu["coverage"] >= 1 - 1e-9 else None,
        "gpu_time_ms_per_second": gpu["mean"], "gpu_coverage": gpu["coverage"],
        "observed_gpu_time_ms": gpu["observed_integral_seconds"],
        "gpu_time_ms": gpu["observed_integral_seconds"] if gpu["coverage"] >= 1 - 1e-9 else None,
        "host_power_watts": power["mean"], "power_coverage": power["coverage"],
        "observed_host_energy_joules": power["observed_integral_seconds"],
        "host_energy_joules": power["observed_integral_seconds"] if power["coverage"] >= 1 - 1e-9 else None,
        "native_samples": sum(any(a <= p["tMs"] < b for a, b in windows) for p in points)}


def summarize(items, graded, report, *, execution, manifest):
    """Join one generation's accuracy and cost; missing work remains explicit."""
    ids = [item["item_id"] for item in items]
    if not ids or len(ids) != len(set(ids)) or any(i["benchmark"] != "ifeval" for i in items):
        raise ValueError("A unique IFEval schedule is required")
    if report.get("configuration_sha256") != manifest["configuration_sha256"] or report.get("artifact_sha256") != manifest["artifact_sha256"]:
        raise ValueError("Resource report belongs to a different artifact/session")
    if graded.get("manifest_sha256") != canonical_hash(manifest):
        raise ValueError("Grades belong to another session")
    records = {r["item_id"]: r for r in graded["records"]}
    grades = {g["item_id"]: g for g in graded["grades"]}
    if len(records) != len(graded["records"]) or len(grades) != len(graded["grades"]) or set(records) != set(ids) or set(grades) != set(ids):
        raise ValueError("Graded schedule is missing, duplicated or substituted")
    samples = {}
    warmups = []
    windows = []
    for sample in report.get("samples", []):
        if type(sample.get("warmup")) is not bool:
            raise ValueError("Explicit warmup identity required")
        if sample["warmup"]:
            if sample.get("item_id") != "__warmup__" or sample.get("request_id") != "generation-1":
                raise ValueError("Warmup identity differs")
            warmups.append(sample)
            if len(warmups) > 1:
                raise ValueError("Only one warmup is scheduled")
            continue
        item_id = sample.get("item_id")
        if item_id not in ids or item_id in samples or sample.get("request_id") != f"generation-{ids.index(item_id) + 2}":
            raise ValueError("Duplicate, substituted or unscheduled resource sample")
        start = _number(sample.get("startedMs"), "startedMs", nullable=False)
        end = _number(sample.get("endedMs"), "endedMs", nullable=False)
        response = _number(sample.get("responseMs"), "responseMs", nullable=False)
        tokens = _number(sample.get("outputTokens"), "outputTokens", integer=True)
        if end < start or abs(response - (end - start)) > 1e-5 or response > SETTINGS["requestLimitMs"] or (tokens is not None and tokens > SETTINGS["maxTokens"]):
            raise ValueError("Request timing/token bounds differ")
        record = records[item_id]
        if record.get("token_counts", {}).get("output") != tokens or (record.get("duration_seconds") is not None and abs(record["duration_seconds"] * 1000 - response) > 1e-4):
            raise ValueError("Quality and resource records are not the same generation")
        if record.get("raw_output") != sample.get("output"):
            raise ValueError("Quality and resource output bytes differ")
        windows.append((start, end))
        samples[item_id] = sample
    if any(b > c for (a, b), (c, d) in zip(sorted(windows), sorted(windows)[1:])):
        raise ValueError("Sequential request windows overlap")
    rows = []
    for item_id in ids:
        grade, record, sample = grades[item_id], records[item_id], samples.get(item_id)
        metrics = grade.get("metrics") or {}
        strict = metrics.get("prompt_level_strict_accuracy") if grade.get("status") == "graded" else None
        loose = metrics.get("prompt_level_loose_accuracy") if grade.get("status") == "graded" else None
        rows.append({"item_id": item_id, "request_id": f"generation-{ids.index(item_id) + 2}",
            "status": record["status"], "grading_status": grade["status"],
            "strict_pass": strict, "loose_pass": loose,
            "output_tokens": sample.get("outputTokens") if sample else None,
            "response_ms": sample["responseMs"] if sample else None,
            "started_ms": sample["startedMs"] if sample else None, "ended_ms": sample["endedMs"] if sample else None,
            "resources": resource_windows(report, [(sample["startedMs"], sample["endedMs"])]) if sample else resource_windows(report, [])})
    passed = sum(r["strict_pass"] is True or r["strict_pass"] == 1 for r in rows)
    loose_passed = sum(r["loose_pass"] is True or r["loose_pass"] == 1 for r in rows)
    unresolved = sum(r["grading_status"] in ("grader_error", "grader_timeout") for r in rows)
    complete = len(samples) == len(ids) and execution.get("status") == "completed"
    totals = {key: sum(r[key] for r in rows) if complete and all(r[key] is not None for r in rows) else None for key in ("output_tokens", "response_ms")}
    totals.update(known_output_tokens=sum(r["output_tokens"] or 0 for r in rows),
        known_response_ms=sum(r["response_ms"] or 0 for r in rows), measured_requests=len(samples),
        mean_response_ms=totals["response_ms"] / len(ids) if totals["response_ms"] is not None else None,
        full_response_tokens_per_second=totals["output_tokens"] * 1000 / totals["response_ms"] if totals["output_tokens"] is not None and totals["response_ms"] else None)
    resources = resource_windows(report, windows)
    if not complete:
        for name in ("cpu_time_ms", "gpu_time_ms", "host_energy_joules"):
            resources[name] = None
    quality = {"strict_passed": passed, "scheduled": len(ids), "strict_accuracy": passed / len(ids),
        "loose_passed": loose_passed, "loose_accuracy": loose_passed / len(ids),
        "unresolved_grades": unresolved, "interpretation": "lower_bound_unresolved_grades" if unresolved else "scheduled_prompt_compliance"}
    cost = {"tokens_per_strict_pass": totals["output_tokens"] / passed if passed and not unresolved and totals["output_tokens"] is not None else None,
        "response_ms_per_strict_pass": totals["response_ms"] / passed if passed and not unresolved and totals["response_ms"] is not None else None,
        "host_joules_per_strict_pass": resources["host_energy_joules"] / passed if complete and passed and not unresolved and resources["host_energy_joules"] is not None else None,
        "scope": "All scheduled scored requests, including incorrect/truncated responses; warmup and model load reported separately."}
    return {"schema_version": 1, "kind": "joint_browser_accuracy_resource_result", "workload": WORKLOAD,
        "status": "completed" if complete and not unresolved else "incomplete", "benchmark": "ifeval",
        "study_sha256": manifest["study_manifest_sha256"], "manifest_sha256": canonical_hash(manifest),
        "artifact_sha256": manifest["artifact_sha256"], "source_checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "items_sha256": manifest["items_sha256"], "settings": manifest["settings"],
        "environment": manifest.get("environment"), "observed_environment": report.get("environment"),
        "harness_sha256": manifest.get("harness_sha256"),
        "model_lib_sha256": manifest.get("model_lib_sha256"), "runtime": report.get("runtime"), "quantization": report.get("quantization"),
        "quality": quality, "totals": totals, "resources": resources, "cost_per_compliant_answer": cost,
        "load_ms": report.get("loadMs"), "warmup": {"count": len(warmups), "output_tokens": sum(s["outputTokens"] for s in warmups) if warmups and all(s.get("outputTokens") is not None for s in warmups) else None},
        "items": rows, "replicates_per_arm": 1,
        "limitations": ["One session per arm; descriptive comparison, no significance or equivalence claim.",
            "Native footprint sums process accounting; GPUBuffer descriptors and WASM address space are not physical VRAM.",
            "CPU/GPU rates use only complete intervals inside each request; short requests may have no telemetry coverage.",
            "Energy is host subsystem energy, not model attribution; incomplete power coverage leaves full cost unavailable."]}


def pair(baseline, candidate):
    """Require the same task/settings/environment before reporting paired deltas."""
    fields = ("study_sha256", "items_sha256", "settings", "environment", "observed_environment", "harness_sha256", "model_lib_sha256", "runtime", "quantization")
    matched = all(baseline.get(k) is not None and baseline.get(k) == candidate.get(k) for k in fields)
    complete = all(x.get("status") == "completed" for x in (baseline, candidate))
    if baseline.get("artifact_sha256") == candidate.get("artifact_sha256"):
        raise ValueError("Candidate must have a distinct deployed artifact identity")
    counts = None
    if matched and complete:
        if [r["item_id"] for r in baseline["items"]] != [r["item_id"] for r in candidate["items"]]:
            raise ValueError("Pair item order differs")
        counts = {"improved": 0, "regressed": 0, "tied": 0}
        for a, b in zip(baseline["items"], candidate["items"], strict=True):
            aa, bb = bool(a["strict_pass"]), bool(b["strict_pass"])
            counts["tied" if aa == bb else "improved" if bb else "regressed"] += 1
    def delta(group, key):
        a, b = baseline.get(group, {}).get(key), candidate.get(group, {}).get(key)
        return b - a if matched and complete and a is not None and b is not None else None
    return {"schema_version": 1, "kind": "joint_browser_accuracy_resource_comparison", "benchmark": "ifeval",
        "status": "completed" if matched and complete else "unmatched_or_incomplete",
        "matched_environment_and_workload": matched, "replicates_per_arm": 1, "paired_outcomes": counts,
        "strict_accuracy_delta": delta("quality", "strict_accuracy"),
        "output_tokens_delta": delta("totals", "output_tokens"), "response_ms_delta": delta("totals", "response_ms"),
        "native_peak_footprint_gb_delta": delta("resources", "native_peak_footprint_gb"),
        "host_energy_joules_delta": delta("resources", "host_energy_joules"),
        "baseline": baseline, "candidate": candidate,
        "interpretation": "Observed paired trade-offs on one shared workload; no statistical improvement or equivalence claim."}


def execute(root, *, on_load, preparation_seconds=300):
    """Generate once, grade saved outputs once, and attach same-request resources."""
    from . import ifeval_v2_runtime as runtime
    root, manifest = runtime.load(root)
    if manifest.get("workload_identity") != WORKLOAD:
        raise ValueError("Not a joint browser session")
    from .browser_resource import environment_identity
    if environment_identity() != manifest.get("environment"):
        raise ValueError("Observed device/browser/runtime differs from frozen joint session")
    execution = runtime.execute(root, on_load=on_load, preparation_seconds=preparation_seconds)
    runtime.grade(root)
    arm = manifest["arm"]
    target = root / "execution" / (arm + "-1")
    terminal = read_json(target / "result.json") if (target / "result.json").exists() else {}
    report = terminal.get("joint_resource_report")
    if report is None:
        report = {"configuration_sha256": manifest["configuration_sha256"], "artifact_sha256": manifest["artifact_sha256"], "samples": [], "profile": {"points": []}}
    save(target / "joint-browser-profile.json", report)
    pointer = target / "current-capture.json"
    if pointer.exists() and report.get("profile", {}).get("startedEpochMs") is not None:
        capture = Path(read_json(pointer)["capturePath"]).resolve()
        if not capture.is_relative_to(target):
            raise ValueError("Native capture path escapes this session")
        script = ('import json,sys,pathlib\nfrom harness.processing.merge_native_report import combine,compact_report\n'
            'p=pathlib.Path(sys.argv[1]); c=pathlib.Path(sys.argv[2]); r=combine(json.loads(p.read_text()),c)\n'
            '(p.parent/"joint-combined-full.json").write_text(json.dumps(r,allow_nan=False)+"\\n")\n'
            '(p.parent/"joint-combined.json").write_text(json.dumps(compact_report(r,c),allow_nan=False)+"\\n")\n')
        subprocess.run([os.environ.get("WSL_NATIVE_PYTHON", "python3"), "-c", script,
            str(target / "joint-browser-profile.json"), str(capture)], cwd=root / "benchmark",
            check=True, timeout=60, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        report = read_json(target / "joint-combined.json")
    result = summarize(read_json(manifest["items_path"]), read_json(root / arm / "graded.json"), report,
                       execution=execution, manifest=manifest)
    if execution.get("status") == "completed" and terminal.get("cleanup", {}).get("native_capture_exit_confirmed") is not True:
        raise ValueError("Native capture cleanup unconfirmed")
    evidence = [root / arm / "graded.json", target / "supervisor-result.json", target / "joint-browser-profile.json"]
    evidence += [p for p in (target / "result.json", target / "events.jsonl", target / "joint-combined.json", target / "joint-combined-full.json", pointer) if p.exists()]
    evidence_files = {str(p.relative_to(root)): file_sha256(p) for p in evidence}
    result["evidence_sha256"] = canonical_hash(evidence_files)
    save(root / "joint-report.json", result)
    save(root / "joint-report-binding.json", {"schema_version": 1, "manifest_sha256": canonical_hash(manifest),
        "result_sha256": canonical_hash(result),
        "files": evidence_files})
    return result


def verify_report(root):
    """Validate the frozen session and the exact saved generations behind its report.

    This is read-only: it neither regrades nor reruns inference. Call it before
    exposing private baseline outputs to the editor or publishing a comparison.
    """
    from . import ifeval_v2_runtime as runtime
    root, manifest = runtime.load(root)
    if manifest.get("workload_identity") != WORKLOAD:
        raise ValueError("Not a joint browser session")
    binding = read_json(root / "joint-report-binding.json")
    result = read_json(root / "joint-report.json")
    if binding.get("manifest_sha256") != canonical_hash(manifest) or binding.get("result_sha256") != canonical_hash(result):
        raise ValueError("Joint report identity changed")
    arm = manifest["arm"]
    target = Path("execution") / (arm + "-1")
    required = {str(Path(arm) / "graded.json"), str(target / "supervisor-result.json"), str(target / "joint-browser-profile.json")}
    evidence = binding.get("files", {})
    if not required.issubset(evidence) or result.get("evidence_sha256") != canonical_hash(evidence):
        raise ValueError("Joint report evidence binding is incomplete")
    for name, digest in evidence.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError("Joint report evidence changed")
    profile_path = target / "joint-combined.json" if str(target / "joint-combined.json") in evidence else target / "joint-browser-profile.json"
    rebuilt = summarize(read_json(manifest["items_path"]), read_json(root / arm / "graded.json"),
        read_json(root / profile_path), execution=read_json(root / target / "supervisor-result.json"), manifest=manifest)
    rebuilt["evidence_sha256"] = canonical_hash(evidence)
    if canonical_hash(rebuilt) != canonical_hash(result):
        raise ValueError("Joint summary no longer matches its saved generations")
    return result


compare_reports = pair
