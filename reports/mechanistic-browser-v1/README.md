# First mechanistic IFEval / browser comparison

**This edit did not produce a joint accuracy/resource improvement.** On November 11, 2026, strict IFEval compliance fell from 16/32 to 15/32 and output tokens rose 1.25%. Observed response time was 2.97% lower, but both tokens and time per compliant answer increased. A single development pair does not establish a causal speedup.

| Measurement on the same 32 prompts | Pristine | Astra directional edit | Change |
| --- | ---: | ---: | ---: |
| Strict compliant prompts | 16/32 (50.00%) | 15/32 (46.875%) | −3.125 percentage points |
| Loose compliant prompts | 17/32 | 16/32 | −1 prompt |
| Output tokens | 13,941 | 14,115 | +174 (+1.25%) |
| Summed response duration | 264.341 s | 256.484 s | −7.857 s (−2.97%) |
| Sampled native process-tree peak | 3.1211 GB | 3.1232 GB | +0.07% |
| Output tokens per strict pass | 871.31 | 941.00 | +8.00% |
| Response seconds per strict pass | 16.52 s | 17.10 s | +3.50% |

There were no improved strict outcomes, one regression and 31 ties. Nineteen answers were byte-identical and 13 changed. Truncated outputs increased from 12 to 13. The sole strict regression (`ai2-train:001570`) failed the minimum-20-placeholder constraint: the baseline contained 121 matched bracket spans; the candidate contained none and reached its token limit. All scheduled outputs, including failures and truncated answers, contribute to the cost numerators. Model loading, warm-up, diagnosis and startup interruption are outside the scored response totals and are reported separately. No weighted composite score was introduced.

## What Astra changed

The requested `gpt-6-astra` editor used a fresh context with the actual baseline outputs, grades and same-request resource evidence. It could request bounded weight inspections and source-model activation captures. It selected 63 noncontiguous columns at layer 19 and 64 at layer 22, each with strength 2:

```text
W'[:, C] = W[:, C] - 2 * u * (u.T @ W[:, C])
```

The unit direction in each layer is a failure-minus-reference residual contrast. Four matched pairs contrasted actual looping baseline prefixes with editor-written task-completing references. References were hypotheses, not verified successful model generations. Columns had positive directional writer attribution in every pair. The rationale predicted fewer loops and lower tokens/latency without a fixed-memory reduction. That joint prediction was not supported by this run. The [pre-evaluation rationale](rationale.txt) is retained unchanged.

Three weight inspections and one direction derivation completed. Three activation requests were recorded: one failed input validation, one exceeded the configured element allowance before forwarding, and one successful CPU capture executed eight teacher-forced sequences across all 24 MLP layers, retaining 16 tail positions. No training, gradients or extra browser-quality probes were used. The exact [proposal](proposal.json), [evidence sidecar](proposal.evidence.json), and [intervention audit](intervention.json) are public; prompts, generated answers and activation arrays remain private.

The source audit targeted 130,048 coordinates and changed 127,400 stored values. The effective q4 audit found 164,516 changed decoded values. Of these, 121,142 lie outside the selected source columns because of shared quantization scales; none lie outside the affected quantization groups. Only the two selected layers' quantized weight/scale tensors changed; 431 deployed tensors remained unchanged. The deployed effect therefore is not confined to exactly 127 columns.

## One measured workload and its limits

Both arms used the same 32 exposed IFEval development prompts, order, official checker, tokenizer, model library and settings: temperature 0, top-p 1, seed 42, thinking disabled, context 4,096, maximum 1,024 output tokens and a 120-second request limit. Each measured model session had one excluded warm-up and 32 scored generations. The same saved response supplied its grade, output-token count, response boundaries and available native telemetry; no separate resource-prompt loop ran.

The host was an Apple M4 Pro, 14 logical CPUs, 24 GiB memory, macOS 26.6.2, Chrome 157.0.8103.39, WebLLM 0.2.86 and q4f16_1. Source diagnostics used CPU Torch 2.10.0 / Transformers 5.3.0; they are not quantized WebGPU activations. Dataset revision: `47c03c73621c4aab2b824b7818681117d662770e`. The original source is Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`.

CPU interval coverage was 88.91% / 88.50% and GPU interval coverage 90.79% / 90.04% for pristine / candidate. Short requests can contain no usable interval. Covered observations are retained, but whole-workload CPU/GPU totals remain unavailable. Power and energy were not observed. Sampled native footprint is process-tree accounting rather than globally deduplicated RAM; GPU buffer descriptors and WASM address space are allocation bookkeeping, not physical VRAM. Those two allocation maxima were identical across the pair.

All prompts were exposed for adaptive development. This is not an untouched accuracy test, an equivalence result, a repetition-specific causal mechanism, or evidence that a roughly 3% timing difference will reproduce. One fewer strict pass and higher output tokens do not become an efficiency win because the second response-time observation was shorter.

## Recorded startup interruption

The first candidate Chrome launch closed before any model-load intent, acknowledgment, load grant, ledger attempt or generation. The cause was not established. The stopped session, its zero-call ledger and original study state are preserved. An explicit pre-load continuation amended the planned no-retry launch protocol, with one bounded replacement launch of the same frozen candidate. It repeated no baseline, proposal, conversion or target sequence.

Actual accounting: one baseline browser launch and two candidate browser launches, but only one model load, one warm-up and 32 scored generations per arm: 66 generated sequences total. The failed pre-load launch consumed 6.120 seconds of supervisor time, disclosed separately from scored response time. No model call was automatically retried. Both measured sessions and the failed launch verified release of their owned browser/collector processes.

The [canonical joint report](joint-comparison.json) binds the paired observations, original failure proof and continuation plan. The source runtime is pinned at `3630ead13aeb05f5a17a7a335b466e69d14539f6`; the explicit continuation controller is pinned at `0f47634` and records its full source hash in its plan. Historical V1/V2/R4 results are unchanged.
