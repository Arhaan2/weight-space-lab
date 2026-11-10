# Browser quality and resource study

Maintainer: [Arhaan Aggarwal (Arhaan2)](https://github.com/Arhaan2).

This is the current V2 workflow. Improve Qwen instruction compliance through up to four manual Astra / Ultra proposals while preserving or reducing measured local browser resource use. The [V1 result](../reports/ifeval-browser-v1-final/README.md) remains closed and inconclusive. The [earlier V2 control-and-authored-data proposal](ifeval-browser-v2-protocol.md), its configuration and private preparation are historical; they do not allocate capacity to this study. Random controls, a newly authored benchmark, native CPU/MPS or hosted GPU experiments, and automated frontier API integration are optional follow-ups.

The executable contract is [ifeval_browser_quality_resource.json](../configs/ifeval_browser_quality_resource.json). Use the [manual runbook](ifeval-browser-v2-runbook.md). Preparation and synthetic tests are implementation evidence; only completed model runs support experimental claims.

## Model, development baseline and editing

Reuse the verified Qwen3.5-0.8B pristine source identity `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d` and local MLC deployment identity `5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0`. Keep architecture, tokenizer, q4f16_1 conversion, WebLLM 0.2.86, compiled model library and generation settings fixed. Every candidate starts from pristine. Apply only the approved tensors/operations, independently reopen and audit the saved checkpoint, convert through the pinned virtual environment, verify deployed tensor changes and shared quantization-group effects, then evaluate that deployed artifact.

Reuse the compatible saved 32-prompt Ai2 RLVR-IFeval development baseline: strict compliance 16/32, loose 17/32, and 13,941 scored sampled output tokens. The ordered development-item hash is `9aa732f4c0dbe06248a93c329299f3fb732a5f148447d3be5a057c93e5fae627`. No new baseline run or diagnostic generation is allocated. Existing [V1 reproduction details](ifeval-v1-reproduction.md) supply model, converter, grader and source pins.

Every task request starts a fresh conversation, disables thinking, sets temperature 0, top-p 1, seed 42, context 4,096, output ceiling 1,024 and request limit 120 seconds. Each load gets exactly one warm-up capped at 16 tokens. The pinned implementation's injected empty-thinking IDs consume four positions in those caps; record sampled output lengths separately.

## Manual submissions and feedback

The operator selects Astra / Ultra in the authenticated manual interface and supplies its original response. The controller does not make an API request, launch another frontier model, supply a fabricated proposal or fill missing proposal slots with fixtures. Each begun submission consumes one of four slots. Invalid, expired, duplicate or abandoned submissions remain recorded and are not replaced. Re-exporting the same packet does not renew its request identity or deadline. No automatic retries occur.

The first private packet binds a real request ID and digest, pristine checkpoint, approved weight information, exact response schema, all 32 development tasks, saved outputs/grades, and available resource observations. It explicitly requests higher compliance with no greater resource use. Later packets add actual preceding development feedback: strict/loose metrics, paired improvements and regressions, generated outputs, natural completions/truncations, scored output tokens, timing/resource limitations, and source/conversion/audit identities. No final prompts or final outcomes belong in any editor packet.

Evaluator/editor separation is procedural; technical isolation is unverified. Provider-enforced computation/cancellation caps, frontier and reasoning-token usage, immutable model version, accepted effort and marginal billing remain unavailable where the interface does not expose them. The two-editor-hour allowance is planning only. Self-description and a matching response envelope do not authenticate the model.

## Selection before final outcomes

A candidate qualifies only after a complete, valid development evaluation, strict prompt compliance greater than pristine, scored output-token consumption no greater than pristine on the identical workload, and compliance with frozen deployment and memory limits. Rank qualifying candidates by higher strict compliance, fewer scored output tokens, then earlier round. Pristine wins quality ties. Keep nonqualifying results as trade-offs.

Show latency and memory observations and flag regressions without inventing an allowable percentage. A single timing measurement cannot establish equal or faster performance. Freeze selected source and deployed identities before opening any final results. Never revise selection based on final quality or resource outcomes. If none qualifies, retain pristine and skip pristine-versus-pristine final and resource comparisons. Unused capacity expires.

## Final task data

Audit the pinned official Google IFEval source and project exposure inventory before the first packet. Deterministically select up to 128 runtime-eligible official prompts, excluding V1 development/final material, known exact/near duplicates and unresolved exposure. Shared instruction type alone is not duplication. Use the existing justified feasibility rules, without new generations or performance-based filtering.

Freeze actual ordered IDs, prompt/instruction counts, source revision and hashes, exclusion rules, grader configuration and exposure limitations. Use fewer than 128 when that is all that qualifies; use zero when none can be established as untouched. Zero means no fresh final-quality confirmation, and does not prevent authorized development or resource work. Never reuse V1's exposed test as fresh confirmation or automatically author replacements. The [final-data audit](ifeval-v2-final-data.md) reports the actual outcome.

If available and an edit qualifies, compare pristine and selected candidate once on the frozen final set. Report all four official compliance metrics, actual denominators, paired outcomes, failures, completions, truncations, and the existing paired prompt-cluster percentile bootstrap (10,000 resamples, seed 0, lexicographic item order, linear quantiles). Do not infer quality preservation from nonsignificance.

## Original browser resource workload

The resource workload is distinct from IFEval. Derive a dedicated two-artifact preset from Gufran Mirza and Arhaan Aggarwal's [Browser LLM Bench](https://github.com/gufranmirza/browser-llm-bench) standard profile at revision `b39187b1aff1f218493622d7fcf3cd5d24773b41`. Keep integration changes in this repository and do not publish to the upstream benchmark.

Freeze the actual pinned workload source/preset identities. Queue only pristine A and selected B in four planned sessions A–B–B–A. Each session uses 60 measured requests in three windows of 20, two 90-second loaded-idle gaps, original conversation resets, temperature 0, top-p 1, seed 42, thinking disabled, context 4,096, output cap 64, one 16-token warm-up, original 60-second request timeout and session safety controls. These repeated sessions are prescheduled measurements, never retry capacity.

Both artifacts use the same current device/browser/runtime/library/quantization and declared cache policy. Record actual environment identity and fetched model bytes. Historical leaderboard values are not the matched control. Use supported native CPU/process-memory/GPU collectors and unprivileged capture where available; power is optional and missing values remain unavailable.

Publish each session and aggregates with actual output lengths, latency and per-token observations, resource-counter scope, coverage and missing signals. GPUBuffer descriptors and WASM address space are allocation accounting, not physical VRAM. Native process footprint has its documented scope; do not add overlapping memory counters. Host-wide power is not model-only power. Preserve original energy intervals. Two sessions per artifact are a small pilot, not 60 independent replications per session or precise proof of resource equality.

## Hard limits and clocks

| Stage | Loads / warm-ups | Non-warm-up requests | Dispatches | Reserved target tokens | Browser minutes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four candidate development evaluations | 4 / 4 | 128 | 132 | 131,136 | 60 |
| Final task comparison, at most 128 per arm | 2 / 2 | 256 | 258 | 262,176 | 90 |
| Four standard resource sessions | 4 / 4 | 240 | 244 | 15,424 | 30 |
| Maximum | 10 / 10 | 624 | 634 | 408,736 | 180 |

At most four editor submissions and four candidate imports, conversions and packages. At most four active local experimental hours, including three browser hours. Browser-stage clocks include loads, warm-ups, required idle gaps and cleanup. Implementation/refactoring and manual waiting are outside active processing; waiting still consumes the seven-day calendar window, anchored once at first production packet export. No restart or export renews that window. Retain at most 32 GiB of new study artifacts. Use a new durable ledger, preserving the earlier preparation and all closed V1 ledgers.

These are ceilings, not guaranteed completion times. A smaller final set, failed candidate or pristine selection reduces consumption; capacity is not transferable to extra proposals, retries, smoke tests, probes, loads or replacement generations. Unknown dispatched usage is charged at its reservation. Preserve incomplete outcomes on exhaustion. Source/configuration checks remain strict, and frozen runtime files cannot be refactored while execution is active.

The supplied study authorization covers the bounded execution after offline verification and production freeze. The only planned inputs are genuine manual Astra responses. After a response, import/audit/convert/evaluate and export the next packet; after the final eligible round, freeze selection and perform the available final comparisons. Concrete integrity, runtime, cleanup or budget failures stop the affected work with preserved evidence. Recovery reconciles state and never silently redispatches an uncertain request.

## Reporting and publication

Separate V1 results, V2 implementation, actual V2 development, actual final quality and actual standard-resource results. Report joint-goal success only when quality and resource evidence support it; retain trade-offs, negative findings and inconclusive outcomes. With no untouched final data, development improvements cannot establish fresh final quality.

Publish reviewed code, provenance, sanitized aggregate evidence and reproducible commands to the existing Arhaan2 repository through a normal topic-branch pull request. Preserve verified author/committer identity and upstream attribution. Exclude private instructions, authorization, packets/transcripts, credentials, machine paths and tool-generated coauthor trailers. Historical evidence and source hashes remain unchanged; new implementation receives a new identity. A private immutable source snapshot preserves the complete actual local Python tree, including pre-existing working modules outside the publication changes. Public commit identity and that full local source identity are reported separately; exact local-state replay requires the retained private evidence.
