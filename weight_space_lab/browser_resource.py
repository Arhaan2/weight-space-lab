"""Matched, bounded Browser LLM Bench sessions; no task-data substitution.

The worker continues through the upstream repeat/session-count workload and
report protocol. This adapter only binds local artifacts and durable study gates.
"""
from __future__ import annotations
import copy
import json
import os
import platform
import plistlib
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse

from .inventory import file_sha256, read_json
from .protocol import canonical_hash
from .ifeval_baseline import save

RESOURCE_SETTINGS = {"temperature": 0, "topP": 1, "seed": 42, "maxTokens": 64,
    "contextWindow": 4096, "warmups": 1, "warmupTokens": 16,
    "requestLimitMs": 60000, "thinking": False}
RESOURCE_PROMPT = "In three short sentences, explain why leaves change color in autumn."
RESOURCE_ORDER = ("pristine_baseline", "selected_astra", "selected_astra", "pristine_baseline")
SOURCE_FILES = ("harness/presets/comparison.json", "src/benchmark/protocol/settings.ts",
    "src/benchmark/protocol/schedule.ts", "src/benchmark/protocol/workloads.ts")
PATCH = Path(__file__).resolve().parents[1] / "integrations/browser-resource-comparison.patch"


def inspect_preset(benchmark):
    """Resolve the actual pinned TypeScript plan without loading any model."""
    benchmark = Path(benchmark).resolve()
    script = '''import {readFileSync} from 'node:fs';
import {planFor} from './src/benchmark/protocol/schedule.ts';
import {defaultRunSettings} from './src/benchmark/protocol/settings.ts';
import {workload} from './src/benchmark/protocol/workloads.ts';
const p=JSON.parse(readFileSync('harness/presets/comparison.json','utf8'));
console.log(JSON.stringify({preset:p,plan:planFor(p.workload,p.mode,p.settings),
 defaultSettings:defaultRunSettings(p.workload),prompts:workload(p.workload)}));'''
    p = subprocess.run(["node", "--import", "tsx", "--input-type=module", "-e", script],
        cwd=benchmark, capture_output=True, text=True, timeout=30, check=True)
    actual = json.loads(p.stdout)
    validate_derivation(actual)
    return {"upstream_repository": "https://github.com/gufranmirza/browser-llm-bench",
        "upstream_revision": "b39187b1aff1f218493622d7fcf3cd5d24773b41",
        "source_files": {p: file_sha256(benchmark/p) for p in SOURCE_FILES},
        "resolved_standard": actual, "workload_identity": "browser-llm-bench-standard",
        "changes": ["queue only the two selected local artifact identities in separate A-B-B-A sessions",
                    "requirePower=false: supported unprivileged native capture",
                    "fresh dedicated profile, hash-verified local assets, durable per-dispatch study gates"],
        "adapter_patch_sha256": file_sha256(PATCH)}


def validate_derivation(actual):
    p, plan = actual["preset"], actual["plan"]
    expected = {"mode": "count", "measuredRequests": 60, "activeLimitMs": None,
        "totalLimitMs": 1800000, "requestLimitMs": 60000, "baselineMs": 15000,
        "idleMs": 30000, "recoveryMs": 15000,
        "countSchedule": {"bursts": 3, "requestsPerBurst": 20, "betweenMs": 90000, "requestGapMs": 0},
        "resetPolicy": "resetChat before every request; fixed prompt for repeats; seed 42", "finishInFlight": True}
    if (p["workload"] != "repeat" or p["mode"] != "session-count" or p["settings"] != RESOURCE_SETTINGS
            or actual["defaultSettings"] != RESOURCE_SETTINGS or plan != expected
            or actual["prompts"] != [RESOURCE_PROMPT]*3 or p["profiling"] is not True):
        raise ValueError("Pinned standard comparison protocol changed")


def install_adapter(benchmark):
    """Preparation only: apply to a new copied harness, never to historical sources."""
    benchmark = Path(benchmark).resolve()
    marker = benchmark/"src/benchmark/resource-comparison/report.ts"
    if marker.exists():
        # Never silently accept an independently modified installation.
        subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=benchmark,
            check=True, capture_output=True)
        return
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=benchmark, check=True, capture_output=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=benchmark, check=True, capture_output=True)
    if not marker.is_file() or "resourceComparison:" not in (benchmark/"src/benchmark/task-evaluation/protocol.ts").read_text():
        raise ValueError("Resource patch did not install in the intended benchmark checkout")


def served_build(benchmark, origin):
    """Freeze the actual build closure, including newly hashed runtime chunk names."""
    benchmark = Path(benchmark)
    public = benchmark/"public"
    paths = [public/"task-evaluation.html", public/"task-evaluation.js",
        *sorted((public/"benchmark-worker").glob("*.js"))]
    if not (public/"benchmark-worker/benchmark.worker.js").is_file():
        raise ValueError("Built worker entrypoint missing")
    return [{"url":origin+"/"+str(p.relative_to(public)), "sha256":file_sha256(p),
        "kind":"html" if p.suffix == ".html" else "javascript"} for p in paths]


def environment_identity():
    chrome = Path(os.environ.get("WSL_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    info = chrome.parents[1]/"Info.plist"
    version = plistlib.loads(info.read_bytes()).get("CFBundleShortVersionString") if info.exists() else None
    hardware = {}
    if platform.system() == "Darwin":
        for name in ("hw.model", "hw.memsize", "hw.logicalcpu", "machdep.cpu.brand_string"):
            result = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5)
            hardware[name] = result.stdout.strip() if result.returncode == 0 else None
    return {"system":platform.system(), "release":platform.release(), "machine":platform.machine(),
        "macos":platform.mac_ver()[0] or None, "hardware":hardware,
        "browser_version":version, "browser_executable_sha256":file_sha256(chrome) if chrome.exists() else None,
        "runtime":"WebLLM 0.2.86", "quantization":"q4f16_1"}


def validate_session(root, manifest, config):
    items = [{"item_id":f"resource-{i+1}", "prompt":RESOURCE_PROMPT} for i in range(60)]
    if (manifest.get("workload_identity") != "browser-llm-bench-standard" or
            manifest.get("settings") != RESOURCE_SETTINGS or manifest.get("scheduled_prompts") != 60 or
            config.get("settings") != RESOURCE_SETTINGS or config.get("resourceComparison") is not True or
            any(k in config for k in ("study", "evaluationStage", "warmupPrompt")) or
            config.get("totalLimitMs") != 1800000 or config.get("items") != items or
            read_json(manifest["items_path"]) != items or manifest.get("items_sha256") != canonical_hash(items) or
            config.get("configurationSha256") != canonical_hash({k:v for k,v in config.items() if k != "configurationSha256"})):
        raise ValueError("Resource workload or session identity changed")


def freeze_comparison(root, study, selection):
    """Resolve only the frozen pristine/edited pair; no catalog-wide dispatch."""
    root = Path(root)
    order = selection["resource_order"]
    if not order:
        return None
    if order != list(RESOURCE_ORDER):
        raise ValueError("Unexpected resource session order")
    baseline = read_json(Path(study["baseline"]["baseline_root"])/"baseline-config.json")
    selected_row = selection["resource_artifacts"]["selected_astra"]
    artifact = read_json(selected_row["artifact_path"])
    if (artifact["artifact_sha256"] != selected_row["artifact_sha256"] or
            artifact["artifact_sha256"] == baseline["convertedArtifactSha256"]):
        raise ValueError("Resource comparison requires two distinct selected identities")
    derived = {"name":"Matched Qwen weight-edit comparison", "models":[baseline["model"]["id"],artifact["model_id"]],
        "workload":"repeat", "mode":"session-count", "deviceLabel":"Matched local Qwen comparison",
        "profiling":True, "requirePower":False, "settings":RESOURCE_SETTINGS}
    document = {"preset":derived, "session_order":list(order), "artifact_identities":{
        "pristine_baseline":baseline["convertedArtifactSha256"], "selected_astra":artifact["artifact_sha256"]},
        "derivation_sha256":canonical_hash(study["resource_preset"]), "generation_dispatches_max":244,
        "target_tokens_max":15424, "browser_seconds_max":1800,
        "cache_policy":"fresh dedicated profile each session; empty IndexedDB weights; same local no-store fetch"}
    path = root/"resource-comparison-preset.json"
    if path.exists():
        if read_json(path) != document: raise ValueError("Resource comparison preset already frozen differently")
    else: save(path,document)
    return canonical_hash(document)


def prepare_session(root, study, key, artifact=None):
    """Bind one prescheduled resource arm; compilation must already be frozen."""
    from . import ifeval_v2_runtime as runtime
    from .browser_candidate import _files
    root = Path(root).resolve()
    benchmark, baseline = root/"benchmark", Path(study["baseline"]["baseline_root"])
    selection = read_json(root/"final-selection.json")
    if canonical_hash(read_json(root/"resource-comparison-preset.json")) != selection.get("resource_preset_sha256"):
        raise ValueError("Frozen resource comparison preset changed")
    if not (benchmark/"src/benchmark/resource-comparison/report.ts").exists():
        raise ValueError("Resource adapter must be installed and built before production freeze")
    session = root/"sessions"/key
    if session.exists():
        raise ValueError("Resource session already prepared; no replacement or retry")
    config = copy.deepcopy(read_json(baseline/"baseline-config.json"))
    for name in ("configurationSha256", "study", "evaluationStage", "warmupPrompt"):
        config.pop(name, None)
    config.update(runId=study["config"]["study_id"]+"-"+key if "config" in study else "weight-space-resource-"+key,
        resourceComparison=True, arm="baseline" if artifact is None else "candidate",
        settings=RESOURCE_SETTINGS, totalLimitMs=1800000,
        sourceRevision=read_json(baseline/"manifest.json")["checkpoint"]["revision"],
        items=[{"item_id":f"resource-{i+1}", "prompt":RESOURCE_PROMPT} for i in range(60)])
    if artifact is not None:
        if _files(artifact["path"]) != artifact["files"] or canonical_hash(artifact["files"]) != artifact["artifact_sha256"]:
            raise ValueError("Selected resource artifact changed")
        relative = "models/v2-"+artifact["artifact_sha256"]+"/resolve/"+artifact["artifact_sha256"]+"/"
        target = benchmark/"public"/relative
        if not target.exists():
            shutil.copytree(artifact["path"], target, copy_function=os.link)
        origin = urlparse(config["model"]["modelUrl"])
        base_url = origin.scheme+"://"+origin.netloc+"/"+relative
        config["model"].update(id=artifact["model_id"], modelUrl=base_url)
        config.update(sourceCheckpointSha256=artifact["source_checkpoint_sha256"], convertedArtifactSha256=artifact["artifact_sha256"])
        names = ["mlc-chat-config.json", "tensor-cache.json", "tokenizer.json", *[r["dataPath"] for r in read_json(target/"tensor-cache.json")["records"]]]
        config["assets"] = [{"url":base_url+n, "sha256":artifact["files"][n]["sha256"]} for n in names]+[
            {"url":config["model"]["modelLib"], "sha256":artifact["files"]["model.wasm"]["sha256"]}]
    shards = [a for a in config["assets"] if a["url"].endswith(".bin")]
    config["resourceArtifact"] = {"weightShardBytes":sum((benchmark/"public"/urlparse(a["url"]).path.lstrip("/")).stat().st_size for a in shards),
        "shards":len(shards), "nominalParameters":800000000}
    origin = urlparse(config["model"]["modelUrl"])
    origin = origin.scheme+"://"+origin.netloc
    config["servedBuild"] = served_build(benchmark, origin)
    config["configurationSha256"] = canonical_hash(config)
    environment = environment_identity()
    for previous in sorted((root/"sessions").glob("*/manifest.json")):
        prior = read_json(previous)
        if prior.get("workload_identity") == "browser-llm-bench-standard" and prior.get("environment") != environment:
            raise ValueError("Matched resource comparison device/browser identity changed")
    session.mkdir(parents=True, mode=0o700)
    (session/"benchmark").symlink_to(benchmark, target_is_directory=True)
    save(session/"baseline-config.json", config)
    save(session/"resource-items.json", config["items"])
    files = {str(p):file_sha256(p) for p in benchmark.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix not in (".pyc", ".pyo")
        and not any(x in p.relative_to(benchmark).parts for x in ("models", "node_modules", "artifacts", ".next", "out", "__pycache__"))}
    files.update({str(p):file_sha256(p) for p in (session/"baseline-config.json", session/"resource-items.json")})
    manifest = {"schema_version":1, "kind":"browser_resource_session", "workload_identity":"browser-llm-bench-standard",
        "arm":config["arm"], "run_id":config["runId"], "slot":key,
        "study_manifest_sha256":canonical_hash(study), "configuration_sha256":config["configurationSha256"],
        "artifact_sha256":config["convertedArtifactSha256"],
        "checkpoint":{**read_json(baseline/"manifest.json")["checkpoint"], "sha256":config["sourceCheckpointSha256"]},
        "scheduled_prompts":60, "scheduled_instructions":0, "browser_seconds":1800,
        "items_path":str(session/"resource-items.json"), "items_sha256":canonical_hash(config["items"]),
        "settings":RESOURCE_SETTINGS, "files":files, "environment":environment,
        "preset_derivation":study.get("resource_preset"),
        "cache_policy":"fresh dedicated profile each session; empty IndexedDB weights; same hash-verified local no-store fetch",
        "power":"unavailable; unprivileged native collector only"}
    save(session/"manifest.json",manifest)
    save(session/"ready.json", {"manifest_sha256":canonical_hash(manifest)})
    with runtime.ledger(session, manifest, create=True) as book:
        save(session/"accounting-before.json",book.summary())
    return session


def execute(root, *, on_load, preparation_seconds):
    from . import ifeval_v2_runtime as runtime
    manifest = read_json(Path(root)/"manifest.json")
    if manifest.get("environment") != environment_identity():
        raise ValueError("Resource session environment changed before execution")
    result = runtime.execute(root, on_load=on_load, preparation_seconds=preparation_seconds)
    finalize(root)
    return result


def finalize(root):
    """Use upstream native merge/interval definitions; retain missing power as null."""
    root = Path(root).resolve()
    config = read_json(root/"baseline-config.json")
    target = root/"execution"/(config["arm"]+"-1")
    terminal_path = target/"result.json"
    if not terminal_path.exists():
        return {"status":"unavailable", "reason":"terminal browser report missing"}
    terminal = read_json(terminal_path)
    report = terminal.get("standard_report")
    if not report:
        return {"status":"unavailable", "reason":"standard browser report missing"}
    save(target/"browser-report.json",report)
    pointer = target/"current-capture.json"
    if pointer.exists():
        capture = read_json(pointer)["capturePath"]
        script = '''import json,sys,pathlib
from harness.processing.merge_native_report import combine,compact_report
p=pathlib.Path(sys.argv[1]); capture=pathlib.Path(sys.argv[2]); data=json.loads(p.read_text())
merged=combine(data,capture); (p.parent/'combined-report-full.json').write_text(json.dumps(merged,indent=2)+'\\n')
(p.parent/'combined-report.json').write_text(json.dumps(compact_report(merged,capture),indent=2)+'\\n')'''
        subprocess.run([os.environ.get("WSL_NATIVE_PYTHON", "python3"), "-c", script,
            str(target/"browser-report.json"),capture], cwd=root/"benchmark",check=True,timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE":"1"})
    combined_path = target/"combined-report.json"
    combined = read_json(combined_path) if combined_path.exists() else report
    native_summary = None
    if combined.get("nativeTrace"):
        # Reuse the original timing-window/interval weighting and footprint summaries.
        script = "import {readFileSync} from 'node:fs'; import {summarizeResources} from './src/benchmark/metrics/resources.ts'; console.log(JSON.stringify(summarizeResources(JSON.parse(readFileSync(process.argv[1],'utf8')))));"
        process = subprocess.run(["node", "--import", "tsx", "--input-type=module", "-e", script, str(combined_path)],
            cwd=root/"benchmark", capture_output=True, text=True, check=True, timeout=30)
        native_summary = json.loads(process.stdout)
    summary = summarize_report(combined, native_summary=native_summary)
    save(target/"resource-summary.json",summary)
    inputs = [root/"manifest.json", target/"browser-report.json"]
    if combined_path.exists(): inputs.append(combined_path)
    if pointer.exists():
        capture_root = Path(read_json(pointer)["capturePath"])
        inputs += [p for p in capture_root.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")]
    save(target/"resource-summary-binding.json", {"summary_sha256":canonical_hash(summary),
        "input_files":{str(p):file_sha256(p) for p in inputs}})
    return summary


def summarize_report(report, *, native_summary=None):
    measured = [s for s in report["samples"] if not s["warmup"]]
    warmups = [s for s in report["samples"] if s["warmup"]]
    profile = report["profile"]
    def total(key, rows):
        return sum(s[key] for s in rows) if rows and all(s.get(key) is not None for s in rows) else None
    points = profile.get("points", [])
    def peak(key):
        values = [p[key] for p in points if p.get(key) is not None]
        return max(values) if values else None
    native = report.get("nativeTrace")
    metrics = native_summary or {}
    return {"workload_identity":"browser-llm-bench-standard", "model":report["model"], "status":report["status"],
        "environment":report.get("environment"), "model_lib":report.get("modelLib"),
        "runtime":report.get("runtime"), "quantization":report.get("quantization"),
        "scheduled_requests":60, "measured_requests":len(measured), "warmups":len(warmups),
        "scored_output_tokens":total("outputTokens", measured), "warmup_output_tokens":total("outputTokens",warmups),
        "output_tokens_each_request":[s.get("outputTokens") for s in measured],
        "response_ms_each_request":[s["responseMs"] for s in measured], "response_ms_sum":total("responseMs",measured),
        "load_ms":report["loadMs"], "session_ms":profile["totalMs"],
        "natural_completions":sum(s["finishReason"] == "stop" for s in measured),
        "truncations":sum(s["finishReason"] == "length" for s in measured),
        "idle_gaps_ms":[measured[i+1]["startedMs"]-measured[i]["endedMs"] for i in (19,39) if len(measured)>i+1],
        "gpu_buffer_descriptor_bytes_sampled_max":peak("gpuBufferBytes"), "wasm_address_space_bytes_sampled_max":peak("wasmBytes"),
        "upstream_resource_summary":native_summary,
        "native_peak_footprint_gb":metrics.get("peakMemory"), "native_idle_footprint_gb":metrics.get("idleMemory"),
        "native_recovery_footprint_gb":metrics.get("releasedMemory"), "cpu_capacity_percent":metrics.get("cpu"),
        "gpu_time_ms_per_second":metrics.get("gpuTime"), "host_power_watts":metrics.get("power"),
        "cpu_coverage":metrics.get("cpuCoverage"), "gpu_coverage":metrics.get("gpuCoverage"), "power_coverage":metrics.get("powerCoverage"),
        "native_trace":native, "native_coverage":native.get("coverage") if native else None,
        "energy":native.get("energy") if native else None,
        "power_availability":"unavailable unless the saved native trace explicitly contains observed power",
        "memory_note":"Allocation descriptors and WASM address space are not physical VRAM; do not add these to native process footprint.",
        "replication_note":"Two prescheduled sessions per artifact; requests within a session are not independent experiments."}


def aggregate(summaries):
    """Only session-level summaries; no request-level pseudoreplication."""
    groups = {}
    for s in summaries:
        groups.setdefault(s["model"],[]).append(s)
    metrics = ("scored_output_tokens", "response_ms_sum", "load_ms", "session_ms",
        "gpu_buffer_descriptor_bytes_sampled_max", "wasm_address_space_bytes_sampled_max",
        "native_peak_footprint_gb", "native_idle_footprint_gb", "native_recovery_footprint_gb",
        "cpu_capacity_percent", "gpu_time_ms_per_second", "host_power_watts", "cpu_coverage", "gpu_coverage", "power_coverage")
    return {model:{"sessions":len(rows), "complete_sessions":sum(r["status"]=="complete" for r in rows),
        "metrics":{k:{"values":[r.get(k) for r in rows],
            "mean":sum(r[k] for r in rows)/len(rows) if rows and all(r.get(k) is not None for r in rows) else None}
            for k in metrics}} for model,rows in groups.items()}


def analyze(root):
    """Retain every frozen session position, including missing/incomplete evidence."""
    root = Path(root).resolve()
    state = read_json(root/"state.json")
    selection = state.get("selection")
    if selection is None or read_json(root/"final-selection.json") != selection:
        raise ValueError("Resource analysis requires frozen selection")
    order = selection["resource_order"]
    if order not in ([], list(RESOURCE_ORDER)):
        raise ValueError("Resource order differs from the frozen A-B-B-A comparison")
    sessions = []
    for index, arm in enumerate(order, 1):
        key = f"resource-{index}-{arm}"
        manifest_path = root/"sessions"/key/"manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["slot"] != key or manifest["workload_identity"] != "browser-llm-bench-standard":
                raise ValueError("Resource session position mismatch")
            from . import ifeval_v2_runtime as runtime
            runtime.load(manifest_path.parent)
            if manifest["study_manifest_sha256"] != selection["study_sha256"]:
                raise ValueError("Resource evidence belongs to another study")
            expected = selection["resource_artifacts"][arm]["artifact_sha256"]
            if manifest["artifact_sha256"] != expected:
                raise ValueError("Resource artifact differs from frozen selection")
            path = manifest_path.parent/"execution"/(manifest["arm"]+"-1")/"resource-summary.json"
            if path.exists():
                binding = read_json(path.with_name("resource-summary-binding.json"))
                if (binding["summary_sha256"] != canonical_hash(read_json(path)) or
                        any(file_sha256(p) != digest for p,digest in binding["input_files"].items())):
                    raise ValueError("Resource summary or its frozen evidence changed")
            sessions.append({"position":index, "scheduled_arm":arm, "artifact_sha256":expected,
                "environment":manifest.get("environment"), "summary":read_json(path) if path.exists() else None})
        else:
            sessions.append({"position":index, "scheduled_arm":arm, "summary":None})
    summaries = [r["summary"] for r in sessions if r["summary"]]
    environments = [r.get("environment") for r in sessions if r["summary"]]
    browser_environments = [s.get("environment") for s in summaries]
    matched = bool(len(summaries)==4 and
        all(e and e.get("browser_version") and e.get("browser_executable_sha256") and e.get("machine") for e in environments) and
        all(e and e.get("userAgent") and e.get("platform") and e.get("hardwareConcurrency",0)>0 for e in browser_environments) and
        all(e==environments[0] for e in environments) and
        all(e==browser_environments[0] for e in browser_environments) and
        len({(s.get("model_lib"),s.get("runtime"),s.get("quantization")) for s in summaries})==1)
    result = {"workload_identity":"browser-llm-bench-standard", "sessions":sessions,
        "scheduled_sessions":len(order), "aggregates":aggregate(summaries), "matched_environment":matched,
        "complete_sessions":sum(s["status"]=="complete" for s in summaries),
        "status":"pristine_retained_no_comparison" if not order else "pilot_complete" if matched and all(s["status"]=="complete" for s in summaries) else "incomplete",
        "interpretation":"Small measurement pilot; two sessions per artifact. No precise resource equality or quality-preservation claim. The joint goal requires separate untouched final task confirmation; resource results alone cannot establish it.",
        "power":"Missing power remains unavailable, not zero. Existing native merge energy intervals are unchanged."}
    path = root/"resource-analysis.json"
    if path.exists():
        if read_json(path) != result: raise ValueError("Resource analysis already saved; no silent replacement")
    else: save(path,result)
    return result
