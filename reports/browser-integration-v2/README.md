# Browser integration v2 — completed

Arhaan2. Both the verified pristine artifact and the existing edited MLC artifact executed the same frozen smoke input locally on Apple WebGPU. Their distinct artifact identities and all 15 runtime asset hashes were observed during each load. WebLLM 0.2.86, q4f16_1, the pinned WASM library, tokenizer, and effective generation settings remained fixed.

| Arm | Load | Generation | Reported tokens | Stop | Actual grader |
| --- | ---: | ---: | ---: | --- | --- |
| Pristine | 3.490 s | 6.837 s | 252 | Length, truncated | Incorrect: missing final answer |
| Existing edited artifact | 2.722 s | 4.153 s | 252 | Length, truncated | Incorrect: missing final answer |

The outputs match byte for byte. This does not invalidate the integration milestone: the correct distinct artifacts were demonstrably loaded and both made 1,513 WebGPU queue submissions during their sole application generation. Each load separately recorded one initialization submission and one WebGPU device. No application warm-up generation occurred. Loading is therefore recorded as computation even before generation dispatch.

Consumption was **2 of 4 engine loads, 2 of 2 generation dispatches, and 504 of 512 reported completion tokens**, with a 256-token cap per arm. There were no extra load attempts, generation retries, proposals, frontier requests, pilot runs, or held-out executions. The single deadline was unchanged across all four durable acknowledgements. First load grant through final owned-resource cleanup took **41.654 seconds of 1,200 seconds**. The new ledger is terminal.

Before engine creation, real browser fetch-only checks verified 21 served files per arm: the application HTML, client, complete worker import graph, configuration, tokenizer, tensor manifest, 11 shards, and WASM. They checked status, MIME/content, HTML fallback, resolved URLs, and SHA-256 identities without creating a worker, adapter, device, or engine. Every attempt used its own fresh profile; unrelated browser data was untouched. Generation acknowledgements followed durable accounting and validation of the loaded artifact, effective settings, and device evidence.

Both engines unloaded; workers, dedicated Chrome processes, and local harnesses exited. OS checks found no retained owned processes. Baseline cleanup completed before the candidate load. No runtime error, device loss, resource stop, cancellation, or forced worker termination was observed. The host collector emitted an Xcode loader warning while its SDK lookup and scoped measurements completed.

Observed worker GPUBuffer descriptor peaks were 1,747,697,720 bytes for each arm, and WASM address space peaked at 283,181,056 bytes. These are not physical VRAM measurements. Owned process footprints and host AGX measurements are retained separately; host readings include unrelated workloads and process memory can overlap. Sampling does not establish an absolute peak. JS heap measurements were unavailable.

The immutable v1 record remains its baseline loading failure after 8.41 seconds, zero generation, candidate unrun, and verified cleanup. All 555 preservation entries match the before snapshot; no old reservation was reopened or ledger reset. The 86 historical provenance limitations remain, including 84 currently unavailable files and two matching current files whose historical provenance is still unverified.

This short development check establishes browser integration only. It does not establish 4,096-token memory fit, sustained performance, or benchmark improvement. Both outputs were truncated and received an actual incorrect grade. Editor isolation and strict readiness remain unverified. The prepared 14+14 pilot, new Astra proposal, and 500 held-out items remain unexecuted.

Evidence: [sanitized result](result.json), [independent verification](verification.json), [preparation audit](preparation-audit.json), and [runtime implementation](../../docs/browser-integration-v2-runtime.md). Private authorization, prompts, reference answers, raw outputs, full grader records, event streams, resources, and dispatch ledgers remain under the private experiment directory and are excluded from public reports.
