# Return to the paired browser experiment

**Completed outcome (November 10, 2026):** three actual Astra candidates completed 96 scored browser generations. All three scored strict 16/32 and loose 17/32, matching the saved baseline totals. Rounds 2 and 3 used 12,644 and 12,591 scored tokens versus 13,941, but none met the required strict-compliance gain. Pristine was retained, and the prescribed standard-resource comparison was skipped. The joint goal remains unestablished. See the [terminal report](../reports/ifeval-browser-v2-collaboration-run/README.md) and [aggregate evidence](../reports/ifeval-browser-v2-collaboration-run/result.json).

The terminal record includes three Astra agents/four delivery turns, the preserved zero-dispatch telemetry failure, exact-byte recovery followed by offline grading of round 3's saved outputs, the Chrome 156-to-157 development comparison confound, and zero untouched final prompts. No generation was retried and no budget was reset. The paragraphs below preserve the plan and execution history; they do not describe unfinished candidate runs.

## Recorded plan and execution history

The project goal is to edit Qwen3.5-0.8B weights with Astra so instruction-following quality improves while measured browser resource use stays the same or falls. The target runtime is WebLLM/MLC on the local device. Native CPU/MPS diagnostics and published-patch replay remain supporting history, not prerequisites for this comparison.

The existing 32-prompt development baseline is complete: strict compliance 16/32, loose compliance 17/32 and 13,941 scored output tokens. The closed V1 held-out comparison was inconclusive: 70/128 candidate versus 69/128 pristine. It did not establish improvement or equivalence. The original Browser LLM Bench matched resource sessions have not yet produced results.

The current bounded study is already prepared and authorized. Its next work is up to four independent Astra proposals, each applied to pristine weights, audited, converted with the pinned toolchain and evaluated on the same 32 prompts. Qualification requires more strict passes than pristine with no more scored output tokens, plus the frozen deployment/memory conditions. A qualifying selected edit receives the original standard Browser LLM Bench workload in A–B–B–A order. If none qualifies, pristine remains selected and redundant comparisons are skipped.

## Editor-route amendment

On November 10, the renewed user request to use Astra is being executed through actual `gpt-6-astra` collaboration agents with fresh context. This differs from the original manual-interface route. Each agent receives only its original development packet, may read that packet for delivery and write its original response, and is instructed not to inspect other files, access final data, use network tools or run the target model. No reasoning-effort override is requested and no Ultra claim is made.

The frozen packets, runtime, configuration, deadlines, limits and ledgers are retained unchanged. A separate private provenance sidecar records the actual dispatch route, requested model, agent identifier, original packet and response hashes, and timing for each round. Legacy `manual` labels and `automatic_frontier_requests: 0` outputs describe the unchanged local importer; they are not evidence that these new proposals used the manual interface or that no frontier-model dispatch occurred. Provider immutable version, token use and billing remain unavailable. Editor isolation remains procedural and technically unverified.

## Limits on the eventual conclusion

No official final prompts could be established as untouched under the existing exposure audit. The current final set therefore contains zero prompts. Development improvement can be reported, but it cannot become a fresh held-out-quality claim. Two resource sessions per artifact form a small pilot; equal-looking or nonsignificant measurements do not prove precise resource equality. Power remains unavailable unless actually measured, and allocation descriptors are not physical VRAM.

The current execution record is [the collaboration-run report](../reports/ifeval-browser-v2-collaboration-run/README.md). The [canonical protocol](ifeval-browser-quality-resource-protocol.md) and [runbook](ifeval-browser-v2-runbook.md) retain the full measurement contract. The results website should display actual completed observations, all qualification failures, measurement scopes and unresolved questions, with links to the original evidence.

## Pre-dispatch telemetry correction

The first actual Astra proposal was accepted, independently audited, converted and packaged. It changed 122,880 stored and numerical source entries; the edit survived quantization. Its browser attempt stopped before any model load, warm-up or generation because the execution sandbox denied the native memory-pressure query. The original failed ledger remains preserved.

A read-only check of the same native collector outside the sandbox returned memory pressure 1, thermal state 0 and the expected process/GPU signals. The accompanying Xcode loader warning was nonfatal: SDK discovery returned successfully. No system modification or weakening of resource guards was required.

A separate continuation adopts the exact already-built candidate and carries all spent costs, the original calendar and original limits. The failed pre-dispatch stage still consumes one of four browser-development attempts, leaving room for at most three real candidate evaluations. Only two further Astra proposals may be dispatched; the fourth proposal turn will be closed without an Astra call when development-attempt capacity is exhausted. The new continuation manifest records the original state and evidence hashes, the explicit mapping of the inherited failed stage, and zero load/generation evidence. Original state and ledgers are not reset.

The third Astra agent initially returned no proposal because tool output truncated the growing feedback packet. An additional delivery turn to that same agent is explicitly recorded. It permits lossless whitespace minification and bounded reads of the unchanged original packet, with all keys/values retained; it adds no scientific feedback or target-model attempt. Reports distinguish distinct editor agents/submission slots from editor delivery turns and leave provider request/token/billing totals unavailable.
