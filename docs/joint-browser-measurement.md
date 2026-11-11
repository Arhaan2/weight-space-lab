# Accuracy and resources from the same browser requests

The mechanistic study uses one IFEval schedule and one browser session per arm: the pristine model, then one final edited model. Browser LLM Bench profiles those exact IFEval generations. The official IFEval evaluator grades their saved text afterward; grading does not generate another answer. There is no separate 60-request resource workload or repeated sequence of candidate versions in this protocol.

This differs from the historical resource comparisons, which retain their original artifacts and workload. The new protocol is `ifeval-joint-browser-v1`, implemented in `weight_space_lab/joint_browser.py` and the copied browser harness adapter in `integrations/joint-resource-report.ts`. The orchestrator is documented in [the study design](study-design.md) and [mechanistic inspection](mechanistic-inspection.md).

## Frozen execution

The study prepares a separate benchmark checkout, installs the adapter, and rebuilds both worker and client before freezing the code. Preparation does not load a target model. Each arm freezes the same ordered IFEval items, exact rendered input-token counts, generation settings, local deployment asset hashes, official evaluator assets, and device/browser identity. The comparison additionally matches the compiled model-library hash and a path-independent fingerprint of browser source, build assets, controller, and native measurement code. Different artifact URLs do not make identical compiled libraries appear different.

The existing bounded task runner retains its ledger, request grants, dispatch reconciliation, memory/thermal stops, deadlines, and cleanup checks. There is one 16-token warmup, followed by 1–128 scored prompts. Scored requests retain the frozen 1,024-output-token cap, 4,096-token context, deterministic settings, and 120-second request limit. A consumed or interrupted arm cannot automatically run again. The native collector runs alongside the task worker and must exit cleanly before a completed result is accepted.

Each scored item has a stable `item_id` and one request identity (`generation-2` onward; `generation-1` is warmup). Joining rejects duplicated or substituted items, overlapping sequential request windows, changed artifact/configuration identities, and disagreement between the graded record and the profiled output text, token count, or duration.

## What the numbers mean

IFEval strict prompt compliance divides passed prompts by **all scheduled prompts**. Missing or failed requests remain in this denominator. Evaluator failures are separately counted; when unresolved, the reported score is a lower bound and cost per compliant answer is unavailable.

Latency and output tokens come from the same worker generation as the saved answer. Total response time includes all measured scored requests, including incorrect and truncated answers. Full-response tokens per second is total output tokens divided by total response time; it is not a decode-only throughput measurement. Load time and warmup token count are reported separately and excluded from scored totals.

Native measurements are aligned to the browser profile clock, then attributed to the recorded start and end of each scored request:

- Physical footprint is the maximum sampled process-accounting footprint inside the request windows. It is not physical GPU memory.
- Browser GPUBuffer descriptor bytes and WASM address-space bytes are separate sampled indicators. They are not labeled VRAM or resident memory.
- CPU capacity and GPU activity use only complete measured intervals lying within one request. Intervals crossing a request boundary are excluded, and overlapping intervals are not counted twice. Coverage accompanies each rate. `observed_cpu_time_ms` and `observed_gpu_time_ms` integrate covered intervals; full `cpu_time_ms` and `gpu_time_ms` require complete duration coverage and an executed schedule. CPU time sums work across the measured process tree's cores. A short request may legitimately have no usable interval.
- Power remains unavailable when the collector has no valid power data. `observed_host_energy_joules` integrates covered host-power intervals only. Full `host_energy_joules` is populated only when the complete scored response duration is covered. This is host subsystem energy, not isolated model energy.

The aggregate resource metrics are calculated over the union of scored request windows; they do not copy the whole-session mean into every item. Model loading, warmup, idle gaps, and cleanup are excluded from these metrics. Missing measurements remain `null`, with zero coverage where appropriate.

Cost per compliant answer charges **all scored work**, including wrong answers: total output tokens, total response milliseconds, or fully observed host energy divided by the count of strict passes. A zero-pass arm, unresolved grading, or incomplete cost measurement produces `null` instead of a fabricated zero, infinity, or estimate. The full observed totals are retained alongside quality so a favorable ratio cannot hide an accuracy loss.

## Saved evidence and comparison

Each session writes private raw outputs, independent grades, browser/native profiles, and a sanitized `joint-report.json`. `joint-report-binding.json` hashes the evidence files, and the report contains the hash of that binding. The parent study records the report hash. `verify_report(session_root)` revalidates the frozen session, evidence hashes, and reconstructed summary without inference or regrading before saved outputs are exposed to the editor or a report is published.

The sanitized report includes per-item IDs, pass flags, execution/grading statuses, token counts, durations, resource measurements, coverage, and artifact/environment identities. It omits prompts, generated text, and local file paths. The website imports this report through the study wrapper and continues to show “not run” until actual evidence is supplied.

The paired comparison reports quality changes, token/time/footprint/energy changes when measured, and which prompts improved or regressed. It requires matched task order, settings, runtime, environment, harness build, and compiled library. Unmatched or incomplete arms do not produce comparison deltas. One session per arm supports a descriptive observed trade-off, not a significance, equivalence, or repeatability claim. Because Astra sees the baseline workload during diagnosis, this is adaptive development evidence, not an untouched accuracy test.

## Verification without model execution

Python fixtures cover generation identity, failures in denominators, wrong-answer costs, missing power, request-boundary attribution, interval overlap, source guards, paired comparability, and saved-output tampering:

```sh
python3 -m unittest tests.test_joint_browser -v
```

After installing the adapter into a disposable resource-enabled benchmark copy, copy `integrations/joint-browser.test.ts` into its `tests/unit/` directory. Its three tests validate the actual browser schema, schedule, request identities, and profiler helper without models or GPU execution:

```sh
node --import tsx --test tests/unit/joint-browser.test.ts
node scripts/build-worker.mjs
node --check scripts/run-task-evaluation.mjs
```

These checks verify implementation and build compatibility. They are not accuracy or performance results for the target model.
