# V2 completed: no qualifying edit

Maintainer: Arhaan Aggarwal (Arhaan2). Completed November 10, 2026.

**Three genuine Astra weight-edit proposals were built and evaluated on all 32 development prompts. None improved strict compliance, so the frozen rule retained pristine. No original Browser LLM Bench A–B–B–A resource sessions were run.** The joint goal of better benchmark quality with the same or lower browser resource use remains unestablished.

All three candidates scored strict 16/32 and loose 17/32, matching the saved baseline totals. Rounds 2 and 3 used fewer sampled output tokens, but token counts on IFEval are not the standard resource workload. There are zero untouched final prompts; no fresh final-quality comparison was possible. [Machine-readable results](result.json) contain every measured round, exact operations, identities, usage accounting and limitations.

## Measured development results

| Artifact | Strict | Loose | Scored tokens | Generation seconds | Natural / truncated | Qualifies |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pristine (saved) | 16/32 | 17/32 | 13,941 | 263.545 | 20 / 12 | Baseline |
| Astra 1 | 16/32 | 17/32 | 15,019 | 279.592 | 18 / 14 | No |
| Astra 2 | 16/32 | 17/32 | 12,644 | 303.146 | 22 / 10 | No |
| Astra 3 | 16/32 | 17/32 | 12,591 | 234.570 | 22 / 10 | No |

The workload is the fixed 32-prompt Ai2 RLVR-IFeval development set, Qwen3.5-0.8B, WebLLM 0.2.86 and MLC `q4f16_1`, thinking disabled, temperature 0, top-p 1, seed 42, context 4,096 and a 1,024-position output ceiling. The runtime's four injected thinking-boundary positions leave at most 1,020 sampled output tokens per scored request. Every new load had one warm-up; scored-token totals exclude warm-ups.

Qualification required **strict passes above 16/32**, no more than **13,941 scored tokens**, complete valid grading, and the frozen deployment/memory conditions. All three completed and passed those technical conditions, with zero unresolved grader errors. Round 1 had one strict improvement and one regression; rounds 2 and 3 had zero improvements or regressions against the saved labels. No candidate met the strict-gain requirement.

Round 2 used 9.30% fewer tokens but took 303.146 generation seconds versus the saved baseline's 263.545. Round 3 used 9.68% fewer tokens and took 234.570 seconds. These are unpaired development observations, not an established speedup or resource saving. The reused baseline ran **Chrome 156.0.8078.80** and the candidates ran **Chrome 157.0.8103.39**; saved profile versions and user agents agree. Both report MacIntel and hardware concurrency 14. This browser difference prevents an isolated weight-only causal claim. The frozen reuse policy did not allocate a replacement baseline.

## Reproducible edits and selection

Every proposal starts independently from pristine checkpoint identity `f3272ddbea59cc854268e91e387416cd84a633a97ae7d9cfaa1325890eac602d`. Each applies `scale_columns` to columns **2014–2073 inclusive** of `model.language_model.layers.15.mlp.down_proj.weight` and columns **3524–3583 inclusive** of `model.language_model.layers.19.mlp.down_proj.weight`. Both factors are **0.75 in round 1**, **1.25 in round 2** and **1.5 in round 3**. These are three independent proposals, not successive edits to an evolving checkpoint.

Each saved source audit verified **122,880 stored and numerical changes**. Conversion preserved the edits; only the two approved projections' quantized weights/scales changed, with quantization-group effects audited. Full operation arrays and each source/deployed identity are in the JSON report. Architecture, tokenizer and deployment settings were fixed.

Selection is `pristine_fallback`, identity `5ead3e9c60107ab7168bee5373f6cd636c839b82557fed6550aaaa30170aa5b0`. The selected and baseline identities are therefore the same. The prescribed resource order is empty, with **zero sessions scheduled and zero completed**; the runner skipped a redundant pristine-versus-pristine comparison. No matched-resource equality, native-memory reduction, GPU saving or power claim follows. Missing power remains unavailable, not zero, and allocation descriptors are not physical VRAM.

## Actual execution and preserved failures

Actual new browser work was **3 loads, 3 warm-ups, 96 scored generations and 99 total generations**, producing **40,290 sampled tokens**: 40,254 scored plus 36 warm-up tokens. There were **zero failed target sequences, zero generation retries and zero unresolved generation sessions**. All three browser unloads, workers, native captures, Chrome processes and local harnesses exited with verified cleanup; no retained owned PID remained. The saved baseline is excluded from these new-work totals.

The ledger charged **1124.457 local experimental seconds** and **926.825 browser-stage seconds**, including inherited failed-stage costs and the separately charged offline grading recovery. These are experiment-ledger times, not total assistant/editor waiting or offline preparation time. The original calendar and global limits were retained.


The first browser attempt stopped before launching Node/Chrome or loading the model because the sandbox denied the native memory-pressure query. It recorded zero loads, warm-ups and generations. Its state/ledger were preserved. A read-only check of the same collector outside the sandbox returned the required signals; an accompanying Xcode SDK-loader warning was nonfatal. A distinct continuation ran the unchanged controller with native host telemetry and carried one submission/import/conversion/package, 52.1134 local seconds and 1.07935 browser seconds. The failed pre-dispatch attempt still counted against four development-browser attempts, leaving three real candidate evaluations. Round 4 was recorded and closed through the existing API without another Astra call or candidate execution.

After round 3's completed browser run and verified cleanup, offline grading stopped before any checker invocation because files had become dataless. Recovery restored 19 copied harness files to their exact frozen bytes and 171 unavailable pinned evaluator payloads from verified cache/RECORD evidence; 347 unavailable generated bytecode caches were quarantined. Four additional frozen inputs were recovered only after their original hashes matched, including ordinary filesystem materialization of the original conversion manifest. No package versions or readable package payloads changed. The full post-restoration preflight matched **758/758 frozen inputs**, with no unavailable files or mismatches.

One bounded offline pass then graded the already-saved round 3 outputs and generated feedback, charging **27.657 seconds**, with **zero target loads/generations or editor calls**. The failed grading stage and exact pre-recovery halt snapshot remain retained; no budget was reset. The final exporter independently reverified all **606 original pre-continuation evidence files** against their retained hashes. No frozen integrity guard was waived.

## Editor route and evidence limits

The study used **three actual fresh-context `gpt-6-astra` collaboration agents and four editor delivery turns**, not the historical manual interface. Round 3's first delivery could not read the entire growing packet because tool output truncated it. A second delivery turn to the same agent used lossless minification and bounded chunks of the unchanged original packet; the agent reported reading all original content before saving its one proposal. No new scientific feedback, proposal slot or target attempt was added by that delivery repair.

Four submission slots were recorded, including the unused fourth turn closed for browser-attempt capacity. Only three proposals were produced. No reasoning-effort override or Ultra claim is made. Provider immutable version, request/token usage and billing remain unavailable; technical editor isolation remains unverified. Legacy manual labels in the unchanged importer are not evidence of manual-interface use or zero frontier interaction.

The result is exploratory development evidence. Browser-version differences, no untouched final set and absent standard-resource sessions prevent a claim that the project's joint objective was achieved. The [route and recovery history](../../docs/2026-11-10-direction.md), [frozen protocol](../../docs/ifeval-browser-quality-resource-protocol.md) and [runbook](../../docs/ifeval-browser-v2-runbook.md) preserve context. The closed V1 held-out result remains separately inconclusive. Public files exclude raw prompts, responses, private packets, authorization, credentials and session transcripts.
